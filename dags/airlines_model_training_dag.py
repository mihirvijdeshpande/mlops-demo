from airflow.decorators import dag, task
import pendulum
import os
import pandas as pd
import hashlib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import joblib
import json
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

DATA_DIR = "/opt/airflow/dags/files/"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
MODEL_DIR = os.path.join(DATA_DIR, "models/")
REPORT_PATH = os.path.join(DATA_DIR, "evaluation_report.json")


def generate_dataset_hash(df: pd.DataFrame) -> str:
    """Generate SHA256 hash of dataset contents for governance tracking."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()


def generate_schema_info(df: pd.DataFrame) -> dict:
    """Return a simple schema dict with column names and dtypes."""
    return {col: str(dtype) for col, dtype in df.dtypes.items()}


@dag(
    schedule=None,
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    tags=["flights", "ml", "regression", "auto-model-select"]
)
def airlines_model_training():

    @task
    def load_data():
        return pd.read_csv(TRAIN_PATH).to_json(orient="split")

    @task
    def load_test_data():
        return pd.read_csv(TEST_PATH).to_json(orient="split")

    @task
    def train_and_evaluate_models(train_json, test_json):
        os.makedirs(MODEL_DIR, exist_ok=True)
        train_df = pd.read_json(train_json, orient="split")
        test_df = pd.read_json(test_json, orient="split")

        x_train = train_df.drop(columns=["price"])
        y_train = train_df["price"]
        x_test = test_df.drop(columns=["price"])
        y_test = test_df["price"]

        # Governance info
        dataset_hash = generate_dataset_hash(train_df)
        dataset_schema = generate_schema_info(train_df)

        models = {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(random_state=42),
        }

        report = {}
        runs_meta = {}

        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("FlightPricePrediction")

        for name, model in models.items():
            with mlflow.start_run(run_name=name) as run:
                model.fit(x_train, y_train)
                preds = model.predict(x_test)

                mae = mean_absolute_error(y_test, preds)
                rmse = root_mean_squared_error(y_test, preds)
                r2 = r2_score(y_test, preds)

                joblib_path = os.path.join(MODEL_DIR, f"{name}.pkl")
                joblib.dump(model, joblib_path)

                # Log governance params
                mlflow.log_param("model_type", name)
                mlflow.log_param("dataset_hash", dataset_hash)
                mlflow.log_param("dataset_schema", json.dumps(dataset_schema))

                # Log metrics
                mlflow.log_metric("MAE", mae)
                mlflow.log_metric("RMSE", rmse)
                mlflow.log_metric("R2", r2)

                # Log model file
                mlflow.log_artifact(joblib_path)

                # Log dataset profile (stats + histograms) for drift checks
                profile_dir = os.path.join(DATA_DIR, "profile")
                os.makedirs(profile_dir, exist_ok=True)
                profile_path = os.path.join(profile_dir, f"{name}_data_profile.json")

                profile_data = {
                    "describe": train_df.describe().to_dict(),
                    "histograms": {col: train_df[col].value_counts().to_dict() for col in train_df.columns}
                }
                with open(profile_path, "w") as pf:
                    json.dump(profile_data, pf, indent=2)

                mlflow.log_artifact(profile_path, artifact_path="reference_data_profile")

                # Log model with schema
                signature = infer_signature(x_train, model.predict(x_train))
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    signature=signature
                )

                report[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
                runs_meta[name] = run.info.run_id

        # Save evaluation metrics locally
        with open(REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)

        # Save run IDs mapping locally
        runs_meta_path = os.path.join(DATA_DIR, "runs_meta.json")
        with open(runs_meta_path, "w") as f:
            json.dump(runs_meta, f, indent=2)

        return report

    @task
    def log_report(report_dict):
        log_path = "/opt/airflow/dags/files/evaluation_log.json"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(report_dict, f, indent=2)
        print(f"✅ Evaluation report saved to: {log_path}")

    @task
    def register_best_model(report_dict):
        runs_meta_path = "/opt/airflow/dags/files/runs_meta.json"
        with open(runs_meta_path, "r") as f:
            runs_meta = json.load(f)

        best_model = min(report_dict, key=lambda m: report_dict[m]["RMSE"])
        best_run_id = runs_meta[best_model]

        print(f"🏆 Best model: {best_model} (Run ID: {best_run_id})")

        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.register_model(
            model_uri=f"runs:/{best_run_id}/model",
            name="FlightPricePredictionModel",
            tags={
                "dataset": "airlines",
                "model_name": best_model
            }
        )

    # DAG structure
    train_json = load_data()
    test_json = load_test_data()
    train_and_evaluate_models_output = train_and_evaluate_models(train_json, test_json)
    log_report_output = log_report(train_and_evaluate_models_output)
    register_best_model(train_and_evaluate_models_output)


airlines_model_training()
