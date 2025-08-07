
from airflow.decorators import dag, task
from datetime import timedelta
import pendulum
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import joblib
import json

import mlflow
import mlflow.sklearn



DATA_DIR = "/opt/airflow/dags/files/"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
VAL_PATH = os.path.join(DATA_DIR, "val.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
MODEL_DIR = os.path.join(DATA_DIR, "models/")
REPORT_PATH = os.path.join(DATA_DIR, "evaluation_report.json")


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
    def load_validation_data():
        return pd.read_csv(VAL_PATH).to_json(orient="split")

    @task
    def load_test_data():
        return pd.read_csv(TEST_PATH).to_json(orient="split")

    @task
    def train_and_evaluate_models(train_json, val_json, test_json):
        os.makedirs(MODEL_DIR, exist_ok=True)
        train_df = pd.read_json(train_json, orient="split")
        val_df = pd.read_json(val_json, orient="split")
        test_df = pd.read_json(test_json, orient="split")

        X_train = train_df.drop(columns=["price"])
        y_train = train_df["price"]
        X_val = val_df.drop(columns=["price"])
        y_val = val_df["price"]
        X_test = test_df.drop(columns=["price"])
        y_test = test_df["price"]

        models = {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(random_state=42),
            # "XGBRegressor": XGBRegressor(random_state=42, verbosity=0)
        }

        report = {}

        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("FlightPricePrediction")

        for name, model in models.items():
            with mlflow.start_run(run_name=name):
                model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
                preds = model.predict(X_test)

                mae = mean_absolute_error(y_test, preds)
                rmse = root_mean_squared_error(y_test, preds)
                r2 = r2_score(y_test, preds)

                joblib_path = os.path.join(MODEL_DIR, f"{name}.pkl")
                joblib.dump(model, joblib_path)
                # joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))

                mlflow.log_param("model_type", name)
                mlflow.log_metric("MAE", mae)
                mlflow.log_metric("RMSE", rmse)
                mlflow.log_metric("R2", r2)
                mlflow.log_artifact(joblib_path)  # Store .pkl file

                # (optional) Log the model with MLflow Model Registry format
                mlflow.sklearn.log_model(model, artifact_path="model")

                report[name] = {
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2
                }

        with open(REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)

        return report
    
    # @task
    # def log_report(report_dict):
    #     print("\nMODEL EVALUATION REPORT:")
    #     for model, metrics in report_dict.items():
    #         if model == "best_model":
    #             print(f"\n🏆 Best Model: {metrics}")
    #         else:
    #             print(f"\n▶ {model}")
    #             for metric, value in metrics.items():
    #                 print(f"   {metric}: {value:.2f}")

    @task
    def log_report(report_dict):
        log_path = "/opt/airflow/dags/files/evaluation_log.json"

        # Make sure the directory exists
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # Write the full report dictionary to a JSON file
        with open(log_path, "w") as f:
            json.dump(report_dict, f, indent=2)

        print(f"✅ Evaluation report saved to: {log_path}")


    # DAG structure
    train_json = load_data()
    val_json = load_validation_data()
    test_json = load_test_data()
    train_and_evaluate_models_output =  train_and_evaluate_models(train_json, val_json, test_json)
    log_report(train_and_evaluate_models_output)


airlines_model_training()

