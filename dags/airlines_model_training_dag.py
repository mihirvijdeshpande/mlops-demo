from airflow.decorators import dag, task
import pendulum
import os
import json
import hashlib
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Optional: If Optuna is not installed in your Airflow image, add it.
import optuna


# -----------------------------
# Constants and Paths
# -----------------------------
DATA_DIR = "/opt/airflow/dags/files/"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
MODEL_DIR = os.path.join(DATA_DIR, "models/")
REPORT_PATH = os.path.join(DATA_DIR, "evaluation_report.json")
RUNS_META_PATH = os.path.join(DATA_DIR, "runs_meta.json")
PROFILE_DIR = os.path.join(DATA_DIR, "profile")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PROFILE_DIR, exist_ok=True)

MLFLOW_URI = "http://mlflow:5000"
EXPERIMENT_NAME = "FlightPricePrediction_RF_Combined"


# -----------------------------
# Utility Functions
# -----------------------------
def generate_dataset_hash(df: pd.DataFrame) -> str:
    """Generate SHA256 hash of dataset contents for governance tracking."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()


def generate_schema_info(df: pd.DataFrame) -> dict:
    """Return a simple schema dict with column names and dtypes."""
    return {col: str(dtype) for col, dtype in df.dtypes.items()}


def log_reference_profile(train_df: pd.DataFrame, name: str) -> str:
    """Create and save a simple data profile for drift checks and return its path."""
    profile_path = os.path.join(PROFILE_DIR, f"{name}_data_profile.json")
    profile_data = {
        "describe": train_df.describe(include="all").to_dict(),
        "histograms": {col: train_df[col].value_counts(dropna=False).to_dict()
                       for col in train_df.columns}
    }
    with open(profile_path, "w") as pf:
        json.dump(profile_data, pf, indent=2)
    return profile_path


def evaluate_and_log(model, name: str, x_train, y_train, x_test, y_test,
                     dataset_hash: str, dataset_schema: dict, extra_params: dict = None):
    """
    Fit is assumed done outside if needed. This logs metrics, params, artifacts, and the model to MLflow.
    Returns dict metrics and the mlflow run_id.
    """
    preds = model.predict(x_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # Persist model to disk
    joblib_path = os.path.join(MODEL_DIR, f"{name}.pkl")
    joblib.dump(model, joblib_path)

    # MLflow logging
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=name) as run:
        # Governance params
        mlflow.log_param("model_name", name)
        mlflow.log_param("dataset_hash", dataset_hash)
        mlflow.log_param("dataset_schema", json.dumps(dataset_schema))

        # Extra params (e.g., hyperparameters)
        if extra_params:
            for k, v in extra_params.items():
                mlflow.log_param(k, v)

        # Metrics
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        # Artifacts
        mlflow.log_artifact(joblib_path)

        # Log profile
        profile_path = log_reference_profile(pd.concat([x_train, y_train], axis=1), name)
        mlflow.log_artifact(profile_path, artifact_path="reference_data_profile")

        # Log model with signature
        signature = infer_signature(x_train, model.predict(x_train))
        mlflow.sklearn.log_model(model, artifact_path="model", signature=signature)

        run_id = run.info.run_id

    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}
    return metrics, run_id


# -----------------------------
# Training Helpers
# -----------------------------
def train_rfr_simple(x_train, y_train) -> RandomForestRegressor:
    """
    Train a RandomForestRegressor with simple, fixed hyperparameters.
    Model name: rfr-simple
    """
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=1
    )
    model.fit(x_train, y_train)
    return model


def train_rfr_hypergrid(x_train, y_train) -> tuple[RandomForestRegressor, dict]:
    """
    Train RFR via GridSearchCV with a defined parameter grid.
    Model name: rfr-hypergrid
    Returns the fitted best estimator and its best_params_.
    """
    base = RandomForestRegressor(random_state=42, n_jobs=1)

    param_grid = {
        "n_estimators": [100, 300],          
        "max_depth": [10, 20],               
        "min_samples_split": [5],         
        "min_samples_leaf": [2] 
    }

    grid_search = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=3,
        n_jobs=1,
        verbose=1
    )
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model, grid_search.best_params_


def train_rfr_optuna(x_train, y_train, n_trials: int = 50) -> tuple[RandomForestRegressor, dict]:
    """
    Train RFR using Optuna to optimize hyperparameters.
    Model name: rfr-optuna
    Returns the fitted best estimator and its params.
    """
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 6, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "random_state": 42,
            "n_jobs": 1
        }
        model = RandomForestRegressor(**params)
        scores = cross_val_score(
            model, x_train, y_train,
            cv=5, scoring="neg_root_mean_squared_error", n_jobs=1
        )
        return -scores.mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=1)
    best_model.fit(x_train, y_train)
    return best_model, best_params


# -----------------------------
# Airflow DAG
# -----------------------------
@dag(
    schedule=None,
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    tags=["flights", "ml", "regression", "rf-combined"]
)
def airlines_rf_all_in_one():

    @task
    def load_data():
        return pd.read_csv(TRAIN_PATH).to_json(orient="split")

    @task
    def load_test_data():
        return pd.read_csv(TEST_PATH).to_json(orient="split")

    @task
    def train_compare_and_log(train_json, test_json):
        # Prepare data
        train_df = pd.read_json(train_json, orient="split")
        test_df = pd.read_json(test_json, orient="split")

        x_train = train_df.drop(columns=["price"])
        y_train = train_df["price"]
        x_test = test_df.drop(columns=["price"])
        y_test = test_df["price"]

        # Governance
        dataset_hash = generate_dataset_hash(train_df)
        dataset_schema = generate_schema_info(train_df)

        # Ensure MLflow config
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        report = {}
        runs_meta = {}

        # 1) rfr-simple
        simple_model = train_rfr_simple(x_train, y_train)
        simple_metrics, simple_run = evaluate_and_log(
            model=simple_model,
            name="rfr-simple",
            x_train=x_train, y_train=y_train,
            x_test=x_test, y_test=y_test,
            dataset_hash=dataset_hash,
            dataset_schema=dataset_schema,
            extra_params={"tuning": "none"}
        )
        report["rfr-simple"] = simple_metrics
        runs_meta["rfr-simple"] = simple_run

        # 2) rfr-hypergrid (GridSearchCV)
        grid_model, grid_params = train_rfr_hypergrid(x_train, y_train)
        grid_metrics, grid_run = evaluate_and_log(
            model=grid_model,
            name="rfr-hypergrid",
            x_train=x_train, y_train=y_train,
            x_test=x_test, y_test=y_test,
            dataset_hash=dataset_hash,
            dataset_schema=dataset_schema,
            extra_params={"tuning": "grid_search", **grid_params}
        )
        report["rfr-hypergrid"] = grid_metrics
        runs_meta["rfr-hypergrid"] = grid_run

        # 3) rfr-optuna
        opt_model, opt_params = train_rfr_optuna(x_train, y_train, n_trials=50)
        opt_metrics, opt_run = evaluate_and_log(
            model=opt_model,
            name="rfr-optuna",
            x_train=x_train, y_train=y_train,
            x_test=x_test, y_test=y_test,
            dataset_hash=dataset_hash,
            dataset_schema=dataset_schema,
            extra_params={"tuning": "optuna", **opt_params}
        )
        report["rfr-optuna"] = opt_metrics
        runs_meta["rfr-optuna"] = opt_run

        # Save evaluation metrics locally
        with open(REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)

        # Save run IDs mapping locally
        with open(RUNS_META_PATH, "w") as f:
            json.dump(runs_meta, f, indent=2)

        return {"report": report, "runs_meta": runs_meta}

    @task
    def register_best_model(payload):
        report = payload["report"]
        runs_meta = payload["runs_meta"]

        # Select best by RMSE
        best_model_key = min(report.keys(), key=lambda k: report[k]["RMSE"])
        best_run_id = runs_meta[best_model_key]

        print(f"🏆 Best model: {best_model_key} (Run ID: {best_run_id})")

        # Register best model
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.register_model(
            model_uri=f"runs:/{best_run_id}/model",
            name="FlightPricePredictionModel_RF_Combined",
            tags={
                "dataset": "airlines",
                "model_name": best_model_key
            }
        )

    train_json = load_data()
    test_json = load_test_data()
    payload = train_compare_and_log(train_json, test_json)
    register_best_model(payload)


airlines_rf_all_in_one()
