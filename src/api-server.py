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

# Path to reference dataset (used for drift comparison)
REFERENCE_DATA_PATH = "/app/reference_data.csv"

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
        "Model loaded from MLflow Staging.\n\n"
        "Metrics available at `/metrics` for Prometheus scraping."
    ),
    version="1.1.0"
)

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

# -----------------
# Model Loader
# -----------------
def load_latest_staging_model():
    global model, model_version, model_source
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    latest_versions = client.get_latest_versions(MODEL_NAME, stages=["Staging"])
    if not latest_versions:
        log_json("No model found in Staging", model_name=MODEL_NAME, level="error")
        raise RuntimeError(f"No model found in Staging for {MODEL_NAME}")
    latest_model = max(latest_versions, key=lambda v: int(v.version))
    model_uri = f"models:/{MODEL_NAME}/Staging"
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
    load_latest_staging_model()
    Instrumentator().instrument(app).expose(app)
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
@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/info")
def info():
    return {
        "model_name": MODEL_NAME,
        "model_version": model_version,
        "model_source": model_source
    }

@app.post("/reload_model")
def reload_model():
    try:
        load_latest_staging_model()
        return {"status": "reloaded", "model_version": model_version}
    except Exception as e:
        log_json("Failed to reload model", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
@PREDICTION_LATENCY.time()
def predict(input_data: FlightInput):
    request_id = str(uuid.uuid4())
    PREDICTIONS_TOTAL.inc()
    if model is None:
        log_json("Prediction failed - model not loaded", request_id=request_id)
        PREDICTIONS_FAILED.inc()
        raise HTTPException(status_code=500, detail="Model not loaded")
    encoded_df = encode_input(input_data)
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
    Runs a feature drift check between reference data and recent prediction inputs.
    Updates Prometheus metric for Grafana.
    """
    import json

    # 1. Load reference dataset (training data snapshot)
    if not os.path.exists(REFERENCE_DATA_PATH):
        logger.error("Reference dataset not found.")
        raise HTTPException(status_code=500, detail="Reference dataset not found.")

    reference_df = pd.read_csv(REFERENCE_DATA_PATH)

    # 2. Load recent input data for drift detection (could be a rolling CSV or DB table)
    recent_data_path = "/app/recent_predictions.csv"
    if not os.path.exists(recent_data_path):
        logger.error("No recent prediction data available.")
        raise HTTPException(status_code=500, detail="No recent prediction data available.")

    current_df = pd.read_csv(recent_data_path)

    # 3. Run Evidently drift report (per feature drift)
    drift_report = Report(metrics=[ColumnDriftMetric(column_name=col) for col in reference_df.columns])
    drift_report.run(reference_data=reference_df, current_data=current_df)

    drift_results = drift_report.as_dict()

    # 4. Calculate average drift score for Prometheus
    drift_flags = []
    for metric in drift_results["metrics"]:
        drift_flags.append(1 if metric["result"]["drift_detected"] else 0)

    avg_drift_score = np.mean(drift_flags) if drift_flags else 0
    DRIFT_SCORE.set(avg_drift_score)

    logger.info(f"Drift check completed. Avg drift score: {avg_drift_score:.2f}")

    return {
        "average_drift_score": avg_drift_score,
        "details": drift_results
    }

@app.post("/perf/log")
def log_performance(predicted_price: float, actual_price: float):
    """
    Manually log actual flight price after checking real booking sites.
    """
    predictions_log.append((predicted_price, actual_price))
    logger.info(f"Logged actual price: predicted={predicted_price}, actual={actual_price}")
    return {"status": "logged", "total_records": len(predictions_log)}

@app.get("/perf")
def get_performance_metrics():
    """
    Calculate MAE, RMSE, and R² based on logged predictions vs actuals.
    Updates Prometheus metrics for Grafana visualization.
    """
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
