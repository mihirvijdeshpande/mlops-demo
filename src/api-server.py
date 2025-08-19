from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Literal
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import pandas as pd
import os
import logging
import sys
import time
import uuid
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric
import numpy as np
import tempfile

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram, Counter, Gauge

# -----------------
# Logging Setup (Loki-friendly: JSON structured)
# -----------------
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO
)
logger = logging.getLogger("flight-price-api")

def log_json(message: str, **kwargs):
    log_entry = {"message": message, **kwargs}
    logger.info(json.dumps(log_entry))

# Prometheus Gauge for drift score (0 = no drift, 1 = drift detected)
DRIFT_SCORE = Gauge(
    'feature_drift_score',
    'Average drift score across monitored features'
)

predictions_log = []  # list of (predicted, actual)

# Prometheus gauges
MAE_GAUGE = Gauge('model_mae', 'Mean Absolute Error of predictions')
RMSE_GAUGE = Gauge('model_rmse', 'Root Mean Squared Error of predictions')
R2_GAUGE = Gauge('model_r2', 'R² score of predictions')

# Store prediction data
DATA_DIR = "/app/data"
RECENT_PREDICTIONS_CSV = os.path.join(DATA_DIR, "recent_predictions.csv")
PREDICTIONS_LOG_FILE = os.path.join(DATA_DIR, "perf.log")
RETRAINING_DATASET_CSV = os.path.join(DATA_DIR, "retraining_dataset.csv")
# Tolerance (in seconds) for matching feature and perf logs by timestamp
RETRAIN_MATCH_TOLERANCE_SEC = 300


# -----------------
# MLflow Config
# -----------------
MLFLOW_TRACKING_URI = "http://mlflow:5000"
MODEL_NAME = "FlightPricePredictionModel"
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI

# -----------------
# FastAPI App
# -----------------
app = FastAPI(
    title="Flight Price Prediction API",
    description=(
        "Predict flight prices from human-friendly input. "
        "Model loaded from MLflow.\n\n"
        "Metrics available at `/metrics` for Prometheus scraping."
    ),
    version="1.1.0"
)

Instrumentator().instrument(app).expose(app)

# -----------------
# Metrics
# -----------------
PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Time spent making predictions'
)

PREDICTIONS_TOTAL = Counter(
    'predictions_total',
    'Total number of prediction requests'
)

PREDICTIONS_FAILED = Counter(
    'predictions_failed_total',
    'Total number of failed prediction requests'
)

# -----------------
# Model Globals
# -----------------
model = None
model_version = None
model_source = None

# -----------------
# Input Schema
# -----------------
class FlightInput(BaseModel):
    duration: float
    days_left: int
    airline: Literal["AirAsia", "Air India", "GO FIRST", "Indigo", "SpiceJet", "Vistara"]
    departure_time: Literal["Afternoon", "Early Morning", "Evening", "Late Night", "Morning", "Night"]
    stops: Literal["one", "two_or_more", "zero"]
    source_city: Literal["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"]
    destination_city: Literal["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"]
    travel_class: Literal["Business", "Economy"]

# -----------------
# Encoding Function
# -----------------
def encode_input(data: FlightInput):
    feature_dict = {
        "duration": data.duration,
        "days_left": data.days_left
    }
    for airline in ["AirAsia", "Air India", "GO FIRST", "Indigo", "SpiceJet", "Vistara"]:
        feature_dict[f"airline_{airline.replace(' ', '_')}"] = (data.airline == airline)
    for dep in ["Afternoon", "Early Morning", "Evening", "Late Night", "Morning", "Night"]:
        feature_dict[f"departure_time_{dep.replace(' ', '_')}"] = (data.departure_time == dep)
    for stop in ["one", "two_or_more", "zero"]:
        feature_dict[f"stops_{stop}"] = (data.stops == stop)
    for city in ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"]:
        feature_dict[f"source_city_{city}"] = (data.source_city == city)
    for city in ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"]:
        feature_dict[f"destination_city_{city}"] = (data.destination_city == city)
    for cls in ["Business", "Economy"]:
        feature_dict[f"class_{cls}"] = (data.travel_class == cls)
    return pd.DataFrame([feature_dict])

# Write data to device
def append_recent_input(df: pd.DataFrame):
    """
    Append encoded input rows to RECENT_PREDICTIONS_CSV with a stable schema.
    If the file does not exist, write with header. If it exists, append without header.
    """
    try:
        # Add optional metadata columns if useful for drift filtering later
        df_to_write = df.copy()
        df_to_write["logged_at"] = pd.Timestamp.utcnow().isoformat()

        # If file exists, align columns to existing header to avoid drifting schema
        if os.path.exists(RECENT_PREDICTIONS_CSV):
            try:
                # Read only header to get column order
                existing_cols = pd.read_csv(RECENT_PREDICTIONS_CSV, nrows=0).columns.tolist()
                # Ensure all existing columns are present in df (add missing as NaN/False)
                for c in existing_cols:
                    if c not in df_to_write.columns:
                        df_to_write[c] = np.nan
                # Add any new columns at the end (helps forward-compatibility)
                for c in df_to_write.columns:
                    if c not in existing_cols:
                        existing_cols.append(c)
                df_to_write = df_to_write[existing_cols]
                df_to_write.to_csv(RECENT_PREDICTIONS_CSV, mode="a", header=False, index=False)
            except Exception as e:
                # If header read fails (corrupt file), rewrite cleanly
                logger.warning(f"Recent predictions file corrupt or unreadable, rewriting: {e}")
                df_to_write.to_csv(RECENT_PREDICTIONS_CSV, mode="w", header=True, index=False)
        else:
            # First write
            df_to_write.to_csv(RECENT_PREDICTIONS_CSV, mode="w", header=True, index=False)
    except Exception as e:
        logger.warning(f"Failed to append recent input to CSV: {e}")

# -----------------
# MLflow Utility Functions (inline)
# -----------------
def fetch_reference_dataset_from_mlflow(model_version: str) -> pd.DataFrame:
    """Fetch the reference dataset artifact from MLflow for a given model version."""
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    run_id = client.get_model_version(MODEL_NAME, model_version).run_id

    artifacts = client.list_artifacts(run_id, path="")
    ref_file_path = None
    for artifact in artifacts:
        if artifact.path.endswith("reference_data.csv"):
            ref_file_path = artifact.path
            break
    if not ref_file_path:
        raise HTTPException(status_code=500, detail="Reference dataset artifact not found in MLflow.")

    with tempfile.TemporaryDirectory() as tmpdir:
        client.download_artifacts(run_id, ref_file_path, tmpdir)
        csv_path = os.path.join(tmpdir, ref_file_path)
        return pd.read_csv(csv_path)

def _load_recent_features_tail(n_rows: int = 500) -> pd.DataFrame:
    """
    Load up to the last n_rows of the features CSV for quick nearest-timestamp match.
    Requires 'logged_at' column to exist in recent_predictions.csv.
    """
    if not os.path.exists(RECENT_PREDICTIONS_CSV):
        return pd.DataFrame()
    try:
        # Fast path: read tail by reading entire CSV when small; for large files, consider an indexed store
        df = pd.read_csv(RECENT_PREDICTIONS_CSV)
        if df.empty or "logged_at" not in df.columns:
            return pd.DataFrame()
        if len(df) > n_rows:
            df = df.tail(n_rows).reset_index(drop=True)
        df["logged_at_ts"] = pd.to_datetime(df["logged_at"], errors="coerce", utc=True)
        df = df.dropna(subset=["logged_at_ts"]).reset_index(drop=True)
        return df
    except Exception as e:
        logging.warning(f"Failed to read recent features tail: {e}")
        return pd.DataFrame()


def _select_nearest_feature_row(features_df: pd.DataFrame, target_ts: pd.Timestamp, tolerance_sec: int) -> pd.Series | None:
    """
    Given a features_df with 'logged_at_ts', find the row closest in time to target_ts within tolerance.
    Returns a pandas Series (row) or None if not found.
    """
    if features_df.empty:
        return None
    # Compute absolute time delta
    deltas = (features_df["logged_at_ts"] - target_ts).abs()
    idx = deltas.idxmin()
    if pd.isna(idx):
        return None
    if deltas.loc[idx] <= pd.Timedelta(seconds=tolerance_sec):
        return features_df.loc[idx]
    return None


def _append_retraining_row(feature_row: pd.Series, actual_price: float):
    """
    Append a single row to retraining_dataset.csv combining feature columns plus 'price'.
    Preserves header and column order based on existing file when present.
    Drops helper column 'logged_at_ts' if it exists.
    """
    try:
        # Build DataFrame from the matched feature row
        row_dict = feature_row.to_dict()
        # Remove helper column if present
        row_dict.pop("logged_at_ts", None)
        row_dict.pop("logged_at", None)
        # Add target
        row_dict["price"] = float(actual_price)

        out_df = pd.DataFrame([row_dict])

        # Align with existing file columns if file exists
        if os.path.exists(RETRAINING_DATASET_CSV):
            try:
                existing_cols = pd.read_csv(RETRAINING_DATASET_CSV, nrows=0).columns.tolist()
                for c in existing_cols:
                    if c not in out_df.columns:
                        out_df[c] = np.nan
                for c in out_df.columns:
                    if c not in existing_cols:
                        existing_cols.append(c)
                out_df = out_df[existing_cols]
                out_df.to_csv(RETRAINING_DATASET_CSV, mode="a", header=False, index=False)
            except Exception as e:
                logging.warning(f"Retraining dataset header read failed; rewriting: {e}")
                out_df.to_csv(RETRAINING_DATASET_CSV, mode="w", header=True, index=False)
        else:
            os.makedirs(DATA_DIR, exist_ok=True)
            out_df.to_csv(RETRAINING_DATASET_CSV, mode="w", header=True, index=False)
    except Exception as e:
        logging.warning(f"Failed to append retraining row: {e}")


# -----------------
# Model Loader
# -----------------
def load_latest_model():
    global model, model_version, model_source
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    latest_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if not latest_versions:
        log_json("No model found", model_name=MODEL_NAME, level="error")
        # raise RuntimeError(f"No model found in Staging for {MODEL_NAME}")
        raise HTTPException(status_code=503, detail="Model not available yet")
    latest_model = max(latest_versions, key=lambda v: int(v.version))
    model_uri = f"models:/{MODEL_NAME}/Production"
    model = mlflow.pyfunc.load_model(model_uri)
    model_version = latest_model.version
    model_source = latest_model.source
    log_json("Model loaded", version=model_version, source=model_source)

# -----------------
# Startup Event
# -----------------
@app.on_event("startup")
def startup_event():
    log_json("Starting API and loading model...")

    try:
        # Ensure data dir exists
        os.makedirs(DATA_DIR, exist_ok=True)
    except Exception as e:
        logger.warning(f"⚠️ Could not create data directory {DATA_DIR}: {e}")

    try:
        load_latest_model()
    except Exception as e:
        logger.warning(f"⚠️ Could not load ML model at startup: {e}")
        global model
        model = None

    log_json("Prometheus metrics enabled", endpoint="/metrics")

# -----------------
# Middleware for Request Logging
# -----------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    log_json(
        "Request completed",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=process_time
    )
    return response

# -----------------
# Endpoints
# -----------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }

@app.get("/info")
def info():
    return {
        "model_name": MODEL_NAME,
        "model_version": model_version,
        "model_source": model_source
    }

@app.get("/reload_model")
def reload_model():
    global model
    try:
        load_latest_model()
        message = {"status": "reloaded", "model_version": model_version}
    except Exception as e:
        log_json("Failed to reload model", error=str(e))
        logger.warning(f"⚠️ Could not load ML model at startup: {e}")
        model = None
        message = "model not found"
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "message": message
    }


@app.post("/predict")
@PREDICTION_LATENCY.time()
def predict(input_data: FlightInput):
    request_id = str(uuid.uuid4())
    PREDICTIONS_TOTAL.inc()
    if model is None:
        log_json("Prediction failed - model not loaded", request_id=request_id)
        PREDICTIONS_FAILED.inc()
        raise HTTPException(status_code=503, detail="Model not available yet")
    encoded_df = encode_input(input_data)
    
    # Persist the encoded features for drift analysis (best-effort)
    try:
        append_recent_input(encoded_df)
    except Exception as e:
        logger.warning(f"Failed to persist recent input for drift: {e}")
    
    # Prediction code
    try:
        prediction = model.predict(encoded_df)
        prediction_value = float(prediction[0])
        log_json(
            "Prediction successful",
            request_id=request_id,
            model_version=model_version,
            input=input_data.dict(),
            prediction=prediction_value
        )
        return {"predicted_price": prediction_value, "request_id": request_id}
    except Exception as e:
        log_json("Prediction failed", request_id=request_id, error=str(e))
        PREDICTIONS_FAILED.inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "You are at root! POST /predict to get predictions. GET /info for model info.",
        "metrics_endpoint": "/metrics"
    }

@app.post("/drift")
def check_drift():
    """
    Runs a feature drift check between reference data (from MLflow artifact) and recent prediction inputs.
    Updates Prometheus metric for Grafana.
    """
    # Guard: if model/model_version not available yet
    if model is None or model_version is None:
        DRIFT_SCORE.set(-1)
        logger.error("Drift check requested but model/model_version is not available.")
        raise HTTPException(status_code=503, detail="Model not loaded yet; drift cannot be computed.")

    # 1. Fetch reference dataset from MLflow for current model version
    try:
        reference_df = fetch_reference_dataset_from_mlflow(model_version)
    except Exception as e:
        DRIFT_SCORE.set(-1)
        logger.error(f"Failed to fetch reference data from MLflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # 2. Load recent input data for drift detection
    recent_data_path = RECENT_PREDICTIONS_CSV
    if not os.path.exists(recent_data_path):
        DRIFT_SCORE.set(-1)
        logger.error("No recent prediction data available.")
        raise HTTPException(status_code=500, detail="No recent prediction data available.")
    try:
        current_df = pd.read_csv(recent_data_path)
    except Exception as e:
        DRIFT_SCORE.set(-1)
        logger.error(f"Failed to read recent prediction data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read recent prediction data: {e}")

    # Optional: empty-data guard
    if reference_df.empty or current_df.empty:
        DRIFT_SCORE.set(-1)
        logger.error("Reference or current dataset is empty.")
        raise HTTPException(status_code=500, detail="Reference or current dataset is empty.")

    # 3. Run Evidently drift report
    try:
        drift_report = Report(metrics=[ColumnDriftMetric(column_name=col) for col in reference_df.columns])
        drift_report.run(reference_data=reference_df, current_data=current_df)
        drift_results = drift_report.as_dict()
    except Exception as e:
        DRIFT_SCORE.set(-1)
        logger.error(f"Evidently drift computation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evidently failed: {e}")

    # 4. Calculate average drift score for Prometheus
    try:
        drift_flags = [1 if metric["result"]["drift_detected"] else 0 for metric in drift_results["metrics"]]
        avg_drift_score = float(np.mean(drift_flags)) if drift_flags else 0.0
        DRIFT_SCORE.set(avg_drift_score)
    except Exception as e:
        DRIFT_SCORE.set(-1)
        logger.error(f"Failed to compute/set drift score: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to compute drift score: {e}")

    log_json(
    "Drift check completed",
    avg_drift_score=avg_drift_score,
    stage="drift_check",
    status="success"
)

    return {
        "average_drift_score": avg_drift_score,
        "details": drift_results
    }


@app.post("/perf/log")
def log_performance(predicted_price: float, actual_price: float):
    # Update in-memory stats
    predictions_log.append((predicted_price, actual_price))
    logger.info(f"Logged actual price: predicted={predicted_price}, actual={actual_price}")

    # Build a timestamp for this perf event
    logged_at_iso = pd.Timestamp.utcnow().isoformat()

    # Best-effort durable append (JSON Lines) for Loki-friendly logs
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        record = {
            "logged_at": logged_at_iso,
            "event": "perf_log",
            "predicted_price": predicted_price,
            "actual_price": actual_price,
            "model_name": MODEL_NAME,
            "model_version": model_version,
        }
        with open(PREDICTIONS_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Failed to persist performance log record: {e}")

    # Synchronously materialize one joined retraining row (best-effort)
    try:
        # Load a tail window of recent features for timestamp matching
        features_tail = _load_recent_features_tail(n_rows=500)
        if not features_tail.empty:
            target_ts = pd.to_datetime(logged_at_iso, utc=True)
            feature_row = _select_nearest_feature_row(features_tail, target_ts, RETRAIN_MATCH_TOLERANCE_SEC)
            if feature_row is not None:
                _append_retraining_row(feature_row, actual_price)
            else:
                logger.info("No feature row matched within tolerance; skipping retraining append.")
        else:
            logger.info("No recent features available to join; skipping retraining append.")
    except Exception as e:
        logger.warning(f"Failed to update retraining dataset: {e}")

    return {"status": "logged", "total_records": len(predictions_log)}

@app.get("/perf")
def get_performance_metrics():
    if not predictions_log:
        return {"error": "No performance data logged yet."}

    y_pred = [p for p, _ in predictions_log]
    y_true = [a for _, a in predictions_log]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Update Prometheus metrics
    MAE_GAUGE.set(mae)
    RMSE_GAUGE.set(rmse)
    R2_GAUGE.set(r2)
    logger.info(f"Performance metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "total_records": len(predictions_log)
    }

# -----------------
# Run
# -----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
