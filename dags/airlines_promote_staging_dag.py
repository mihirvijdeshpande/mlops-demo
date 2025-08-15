from airflow.decorators import dag, task
import pendulum
from mlflow.tracking import MlflowClient
import subprocess
import os
import time
import json
import pandas as pd
from sklearn.metrics import r2_score
import mlflow
import mlflow.sklearn
import joblib

MLFLOW_TRACKING_URI = "http://mlflow:5000"
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
DATA_DIR = "/opt/airflow/dags/files/"
MODEL_DIR = os.path.join(DATA_DIR, "models/")
VAL_PATH = os.path.join(DATA_DIR, "val.csv")
REPORT_PATH = os.path.join(DATA_DIR, "staging_val_report.json")
MODEL_NAME = "FlightPricePredictionModel"
STAGING_PORT = 8000


@dag(
    schedule=None,
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    tags=["flights", "ml", "regression", "promote-model"]
)
def airlines_promote_staging_dag():
    
    @task
    def promote_to_staging():
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        latest_versions = client.get_latest_versions(MODEL_NAME)

        if not latest_versions:
            raise ValueError(f"No versions found for model '{MODEL_NAME}'")

        latest_model = max(latest_versions, key=lambda v: int(v.version))

        if latest_model.current_stage != "Staging":
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=latest_model.version,
                stage="Staging",
                archive_existing_versions=True
            )
            print(f"✅ Model {MODEL_NAME} version {latest_model.version} moved to Staging.")
        else:
            print(f"ℹ️ Model {MODEL_NAME} version {latest_model.version} is already in Staging.")

        source_uri = latest_model.source
        print(f" Promoted model {source_uri} (Staging)")

        staging_info = {
            "model_name": MODEL_NAME,
            "model_version": latest_model.version,
            "stage": "Staging",
            "model_uri": source_uri
        }
        return staging_info
    
    @task
    def load_validation_data():
        return pd.read_csv(VAL_PATH).to_json(orient="split")
    
    @task
    def test_in_staging(staging_info, val_json):
        val_df = pd.read_json(val_json, orient="split")
        x_val = val_df.drop(columns=["price"])
        y_val = val_df["price"]
        report = {}
        runs_meta = {}
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            model_name = staging_info["model_name"]
            model_version = staging_info["model_version"]
            model_uri = staging_info["model_uri"]

            with mlflow.start_run(run_name=model_name) as run:
                model = mlflow.sklearn.load_model(model_uri)

                predictions = model.predict(x_val)
                
                r2 = r2_score(y_val, predictions)
                r2_threshold = 0.95

                performance_check = r2 >= r2_threshold

                joblib_path = os.path.join(MODEL_DIR, f"prod-{model_name}.pkl")
                joblib.dump(model, joblib_path)

                mlflow.log_param("model_type", model_name)
                mlflow.log_metric("R2", r2)
                mlflow.log_artifact(joblib_path)

                report[model_name] = {"R2": r2}
                runs_meta[model_name] = run.info.run_id

            with open(REPORT_PATH, "w") as f:
                json.dump(report, f, indent=2)

            # Save run IDs mapping
            runs_meta_path = os.path.join(DATA_DIR, "runs_meta.json")
            with open(runs_meta_path, "w") as f:
                json.dump(runs_meta, f, indent=2)

            staging_val_results = {
                "model_name": model_name,
                "model_version": model_version,
                "model_uri": model_uri,
                "ready_for_production": performance_check
            }
        except ValueError:
            staging_val_results = {
                "model_name": "NA",
                "model_version": "NA",
                "model_uri": "NA",
                "ready_for_production": False
            }
        
        return staging_val_results
    
    @task
    def promote_to_prod(staging_val_results):
        try:
            if not staging_val_results["ready_for_production"]:
                raise ValueError("Model failed validation tests. Cannot promote to production")
            
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = MlflowClient()

            model_name = staging_val_results["model_name"]
            model_version = staging_val_results["model_version"]

            client.transition_model_version_stage(
                name=model_name,
                version=model_version,
                stage="Production",
                archive_existing_versions=True
            )
            print(f"✅ Model {model_name} version {model_version} moved to Production.")
        except ValueError:
            print(f" Model Unknown OR version Unknown : Failed to move to Production.")


    staging_info = promote_to_staging()
    val_json = load_validation_data()
    staging_val_results=test_in_staging(staging_info, val_json)
    promote_to_prod(staging_val_results)

airlines_promote_staging_dag()
