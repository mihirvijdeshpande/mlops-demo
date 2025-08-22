# ... (imports and constants unchanged)

import re

# ... (rest of your code unchanged)

def get_next_model_id(model_name: str, model_dir: str) -> int:
    """
    Scan the model_dir for files matching model_name-<id>.pkl and return the next available id (starting from 1).
    """
    existing = [
        fname for fname in os.listdir(model_dir)
        if re.match(rf"^{re.escape(model_name)}-(\d+)\.pkl$", fname)
    ]
    if not existing:
        return 1
    ids = [
        int(re.match(rf"^{re.escape(model_name)}-(\d+)\.pkl$", fname).group(1))
        for fname in existing
    ]
    return max(ids) + 1

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

    # Persist model to disk with incrementing ID
    model_id = get_next_model_id(name, MODEL_DIR)
    joblib_path = os.path.join(MODEL_DIR, f"{name}-{model_id}.pkl")
    joblib.dump(model, joblib_path)

    # MLflow logging (unchanged)
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=name) as run:
        # Governance params
        mlflow.log_param("model_name", name)
        mlflow.log_param("model_file", os.path.basename(joblib_path))
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

# ... (rest of your code unchanged)
