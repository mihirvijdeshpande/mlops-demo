# from airflow import DAG
from airflow.decorators import dag, task
import pendulum
from airflow.models.baseoperator import chain
import mlflow
from mlflow.tracking import MlflowClient
import subprocess
import os

MLFLOW_TRACKING_URI = "http://mlflow:5000"  # Change to your actual MLflow tracking URI
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI

MODEL_NAME = "FlightPricePredictionModel"
STAGING_PORT = 8000

default_args = {
    "owner": "airflow",
    "depends_on_past": True,  # ensures sequential execution
    "retries": 0
}

@dag(
    default_args=default_args,
    schedule=None,
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
)
def airlines_promote_staging_dag():
    @task
    def promote_to_staging():
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        latest_versions = client.get_latest_versions(MODEL_NAME, stages=["None"])

        if not latest_versions:
            raise ValueError(f"No un-staged versions found for model '{MODEL_NAME}'")

        latest_model = max(latest_versions, key=lambda v: int(v.version))
        response = client.transition_model_version_stage(
            name=MODEL_NAME,
            version=latest_model.version,
            stage="Staging"
        )
        print(f"✅ Model {MODEL_NAME} version {latest_model.version} moved to Staging.")

        return str(response)

    @task
    def serve_model(resp):
        print(f"response from server: {resp}")
        try:
            subprocess.run(
                [
                    "mlflow", "models", "serve",
                    "-m", f"models:/{MODEL_NAME}/Staging",
                    "-h", "0.0.0.0",
                    "-p", str(STAGING_PORT)
                ],
                env={
                    **os.environ,
                    "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI
                },
                check=True  # raises CalledProcessError if command fails
            )
            print(f"🚀 Serving model {MODEL_NAME} in Staging on port {STAGING_PORT}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Model serving failed: {e}")

    resp=promote_to_staging(),
    serve_model(resp)

airlines_promote_staging_dag()
