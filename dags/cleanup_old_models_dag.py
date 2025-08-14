from airflow.decorators import dag, task
from datetime import datetime, timedelta
import os
import glob
import mlflow

MODEL_DIR = "/opt/airflow/dags/files/models/"  # adjust if different
MLFLOW_TRACKING_URI = "http://mlflow:5000"
EXPERIMENT_NAME = "FlightPricePrediction"  # adjust if different

@dag(
    schedule=None,  # Run manually
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["maintenance", "mlflow", "cleanup"]
)
def cleanup_old_models():
    
    @task
    def cleanup_models():
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if not experiment:
            print(f"❌ Experiment '{EXPERIMENT_NAME}' not found.")
            return
        
        # Get all registered model names from MLflow registry
        registered_models = set()
        try:
            for model in mlflow.search_registered_models():
                for version in model.latest_versions:
                    registered_models.add(version.name)
        except Exception as e:
            print(f"⚠ Could not fetch registered models: {e}")

        # List all .pkl files in MODEL_DIR
        model_files = glob.glob(os.path.join(MODEL_DIR, "*.pkl"))
        if not model_files:
            print("📂 No model files found.")
            return

        # Identify unregistered models
        unregistered_files = []
        for file_path in model_files:
            model_name = os.path.splitext(os.path.basename(file_path))[0]
            if model_name not in registered_models:
                unregistered_files.append((file_path, os.path.getmtime(file_path)))

        if not unregistered_files:
            print("✅ No unregistered models found.")
            return

        # Sort by modification time (newest first)
        unregistered_files.sort(key=lambda x: x[1], reverse=True)

        # Keep only latest 3 unregistered models
        to_delete = unregistered_files[1:] if len(unregistered_files) > 1 else []

        for file_path, _ in to_delete:
            try:
                os.remove(file_path)
                print(f"🗑 Deleted old unregistered model: {file_path}")
            except Exception as e:
                print(f"⚠ Could not delete {file_path}: {e}")

        print(f"✅ Cleanup complete. Kept {min(3, len(unregistered_files))} latest unregistered models.")

    cleanup_models()

cleanup_old_models()
