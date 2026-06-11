"""
AgriTech FastAPI — endpoints de prédiction de rendement agricole.

Endpoints :
  GET  /health                        — statut du service et du modèle
  GET  /parcels                       — liste des parcelles
  GET  /parcels/{parcel_id}           — détail d'une parcelle
  POST /predict/yield                 — prédiction XGBoost + recommandation irrigation
  GET  /predictions                   — historique des prédictions
  GET  /pesticide/countries           — liste des pays disponibles
  GET  /pesticide/history/{country}   — données historiques pesticides
  GET  /pesticide/forecast/{country}  — forecast Prophet 2017-2021 (via MLflow)
"""
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import mlflow
import mlflow.xgboost
import pandas as pd
import sqlalchemy
from fastapi import FastAPI, HTTPException, Query
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, ConfigDict
from sklearn.preprocessing import LabelEncoder

# ── Métriques Prometheus personnalisées ─────────────────────────────────────

PREDICTION_REQUESTS = Counter(
    "agritech_predictions_total",
    "Nombre total de prédictions de rendement",
    ["crop_type"],
)

PREDICTION_LATENCY = Histogram(
    "agritech_prediction_duration_seconds",
    "Durée de l'inférence XGBoost (hors réseau et DB)",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

MODEL_ERRORS = Counter(
    "agritech_model_errors_total",
    "Erreurs de chargement ou d'inférence du modèle ML",
)

DATABASE_URL       = os.getenv("DATABASE_URL", "postgresql://agritech:agritech_secret@postgres:5432/agritech")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# Sorted alphabetically — matches sklearn LabelEncoder fit order from simulate_data.py
CROP_CLASSES = ["corn", "rice", "soybean", "sunflower", "wheat"]

# Optimal seasonal water requirements (mm) per crop type
CROP_WATER_NEEDS = {
    "corn":      650,
    "rice":      1200,
    "soybean":   700,
    "sunflower": 450,
    "wheat":     500,
}

_label_encoder = LabelEncoder()
_label_encoder.fit(CROP_CLASSES)

engine = sqlalchemy.create_engine(DATABASE_URL, pool_pre_ping=True)

_model = None
_model_version: Optional[str] = None


def _load_model():
    global _model, _model_version
    if _model is not None:
        return _model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        _model = mlflow.xgboost.load_model("models:/yield-xgboost/latest")
        client = mlflow.MlflowClient()
        versions = client.search_model_versions("name='yield-xgboost'")
        if versions:
            _model_version = versions[0].version
        return _model
    except Exception as exc:
        MODEL_ERRORS.inc()
        raise HTTPException(status_code=503, detail=f"Model not available: {exc}") from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        _load_model()
        print(f"[api] Model yield-xgboost v{_model_version} loaded")
    except Exception as e:
        print(f"[api] Model not loaded at startup: {e}")
    yield


app = FastAPI(
    title="AgriTech API",
    version="1.0.0",
    description="API de prédiction de rendement agricole et forecast pesticides",
    lifespan=lifespan,
)

# Instrumentation HTTP automatique : durée requête, nb requêtes, codes statut
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


# ── Pydantic schemas ─────────────────────────────────────────────────────────

class YieldPredictionRequest(BaseModel):
    parcel_id:    int
    crop_type:    str
    year:         int
    soil_moisture: float
    soil_ph:      float
    nitrogen_ppm: float
    air_temp_c:   float
    humidity_pct: float
    rainfall_mm:  float
    solar_rad_wm2: float
    country:      str


class YieldPredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    parcel_id:         int
    predicted_yield:   float
    irrigation_rec_mm: float
    model_name:        str = "yield-xgboost"
    model_version:     Optional[str] = None


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["monitoring"])
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_version": _model_version,
    }


@app.get("/parcels", tags=["parcels"])
def list_parcels():
    with engine.connect() as conn:
        rows = conn.execute(sqlalchemy.text(
            "SELECT id, name, crop_type, area_ha, soil_type, country, latitude, longitude "
            "FROM parcels ORDER BY id"
        )).fetchall()
    return [dict(r._mapping) for r in rows]


@app.get("/parcels/{parcel_id}", tags=["parcels"])
def get_parcel(parcel_id: int):
    with engine.connect() as conn:
        row = conn.execute(
            sqlalchemy.text("SELECT * FROM parcels WHERE id = :id"),
            {"id": parcel_id},
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Parcel not found")
    return dict(row._mapping)


@app.post("/predict/yield", response_model=YieldPredictionResponse, tags=["predictions"])
def predict_yield(req: YieldPredictionRequest):
    model = _load_model()

    if req.crop_type not in CROP_CLASSES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown crop_type '{req.crop_type}'. Valid values: {CROP_CLASSES}",
        )

    # Fetch pesticide features for the given country + year
    with engine.connect() as conn:
        pest = conn.execute(
            sqlalchemy.text("""
                SELECT value_tonnes, yoy_growth_pct, ma5_tonnes, value_normalized
                FROM pesticide_use
                WHERE area = :country AND year = :year
                LIMIT 1
            """),
            {"country": req.country, "year": req.year},
        ).fetchone()

    if pest:
        value_tonnes     = float(pest.value_tonnes)
        yoy_growth_pct   = float(pest.yoy_growth_pct or 0.0)
        ma5_tonnes       = float(pest.ma5_tonnes)
        value_normalized = float(pest.value_normalized)
    else:
        value_tonnes = yoy_growth_pct = ma5_tonnes = value_normalized = 0.0

    crop_encoded = int(_label_encoder.transform([req.crop_type])[0])

    features = pd.DataFrame([{
        "crop_encoded":     crop_encoded,
        "year":             req.year,
        "soil_moisture":    req.soil_moisture,
        "soil_ph":          req.soil_ph,
        "nitrogen_ppm":     req.nitrogen_ppm,
        "air_temp_c":       req.air_temp_c,
        "humidity_pct":     req.humidity_pct,
        "rainfall_mm":      req.rainfall_mm,
        "solar_rad_wm2":    req.solar_rad_wm2,
        "value_tonnes":     value_tonnes,
        "yoy_growth_pct":   yoy_growth_pct,
        "ma5_tonnes":       ma5_tonnes,
        "value_normalized": value_normalized,
    }])

    t0 = time.perf_counter()
    predicted_yield = round(max(0.0, float(model.predict(features)[0])), 3)
    PREDICTION_LATENCY.observe(time.perf_counter() - t0)
    PREDICTION_REQUESTS.labels(crop_type=req.crop_type).inc()

    water_target      = CROP_WATER_NEEDS.get(req.crop_type, 600)
    irrigation_rec_mm = round(max(0.0, water_target - req.rainfall_mm), 1)

    with engine.begin() as conn:
        conn.execute(
            sqlalchemy.text("""
                INSERT INTO ml_predictions
                    (parcel_id, model_name, model_version, predicted_yield, irrigation_rec_mm, confidence)
                VALUES
                    (:parcel_id, 'yield-xgboost', :version, :yield_val, :irr, :conf)
            """),
            {
                "parcel_id": req.parcel_id,
                "version":   _model_version or "unknown",
                "yield_val": predicted_yield,
                "irr":       irrigation_rec_mm,
                "conf":      0.8643,
            },
        )

    return YieldPredictionResponse(
        parcel_id=req.parcel_id,
        predicted_yield=predicted_yield,
        irrigation_rec_mm=irrigation_rec_mm,
        model_version=_model_version,
    )


@app.get("/predictions", tags=["predictions"])
def list_predictions(limit: int = Query(default=50, le=500)):
    with engine.connect() as conn:
        rows = conn.execute(
            sqlalchemy.text("""
                SELECT mp.id, mp.parcel_id, p.name AS parcel_name,
                       p.crop_type, p.country,
                       mp.model_name, mp.model_version,
                       mp.predicted_yield, mp.irrigation_rec_mm,
                       mp.confidence, mp.predicted_at
                FROM ml_predictions mp
                JOIN parcels p ON p.id = mp.parcel_id
                ORDER BY mp.predicted_at DESC
                LIMIT :limit
            """),
            {"limit": limit},
        ).fetchall()
    return [dict(r._mapping) for r in rows]


@app.get("/pesticide/countries", tags=["pesticides"])
def pesticide_countries():
    with engine.connect() as conn:
        rows = conn.execute(sqlalchemy.text(
            "SELECT DISTINCT area FROM pesticide_use ORDER BY area"
        )).fetchall()
    return [r.area for r in rows]


@app.get("/pesticide/history/{country}", tags=["pesticides"])
def pesticide_history(country: str):
    with engine.connect() as conn:
        rows = conn.execute(
            sqlalchemy.text("""
                SELECT year, value_tonnes, yoy_growth_pct, ma5_tonnes,
                       cagr_5y_pct, value_normalized, pct_vs_global_avg
                FROM pesticide_use
                WHERE area = :country
                ORDER BY year
            """),
            {"country": country},
        ).fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail=f"No data found for country '{country}'")
    return [dict(r._mapping) for r in rows]


@app.get("/pesticide/forecast/{country}", tags=["pesticides"])
def pesticide_forecast(country: str):
    """Forecast Prophet 2017-2021 récupéré depuis les artifacts MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()

    exp = client.get_experiment_by_name("pesticide-forecast-prophet")
    if exp is None:
        raise HTTPException(status_code=404, detail="No Prophet experiment found in MLflow")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise HTTPException(
            status_code=404,
            detail="No Prophet run found. Run train_prophet.py first (inside Docker or via Airflow DAG).",
        )

    run_id = runs[0].info.run_id
    country_key = country.replace(" ", "_").replace(",", "").replace("/", "_")

    try:
        local_path = client.download_artifacts(run_id, f"forecasts/{country_key}.json")
        with open(local_path) as f:
            data = json.load(f)
        return {"country": country, "forecast": data}
    except Exception:
        raise HTTPException(
            status_code=404,
            detail=f"No forecast artifact for country '{country}'. "
                   "Run train_prophet.py first and ensure the country is in the top 10.",
        )
