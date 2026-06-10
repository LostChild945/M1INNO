"""
Entraînement XGBoost — prédiction du rendement agricole (t/ha).

Features :
  - crop_type (encodé)
  - sol : soil_ph, soil_moisture, nitrogen_ppm
  - météo : air_temp_c, humidity_pct, rainfall_mm, solar_rad_wm2
  - pesticides : value_tonnes, yoy_growth_pct, ma5_tonnes, value_normalized
  - temporel : year

Target : yield_t_per_ha

Tracking MLflow : paramètres, RMSE / MAE / R², feature importances, modèle.

Lancement :
  python3 src/ml/train_xgboost.py
"""
import os
import numpy as np
import pandas as pd
import sqlalchemy
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

DATABASE_URL   = os.getenv("DATABASE_URL", "postgresql://agritech:agritech_secret@localhost:5432/agritech")
MLFLOW_URI     = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
ARTIFACT_ROOT  = os.getenv("MLFLOW_ARTIFACT_ROOT", "/tmp/mlflow-artifacts")

PARAMS = {
    "n_estimators":     300,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "random_state":     42,
}


def load_dataset(engine) -> pd.DataFrame:
    query = """
        SELECT
            p.crop_type,
            p.country,
            EXTRACT(YEAR FROM s.recorded_at)::INT  AS year,
            s.soil_moisture,
            s.soil_ph,
            s.nitrogen_ppm,
            s.air_temp_c,
            s.humidity_pct,
            s.rainfall_mm,
            s.solar_rad_wm2,
            pu.value_tonnes,
            COALESCE(pu.yoy_growth_pct, 0)         AS yoy_growth_pct,
            pu.ma5_tonnes,
            pu.value_normalized,
            y.yield_t_per_ha
        FROM sensor_readings s
        JOIN parcels    p  ON p.id = s.parcel_id
        JOIN yield_records y
            ON y.parcel_id = s.parcel_id
            AND y.harvest_year = EXTRACT(YEAR FROM s.recorded_at)::INT
        LEFT JOIN pesticide_use pu
            ON pu.area = p.country
            AND pu.year = EXTRACT(YEAR FROM s.recorded_at)::INT
        WHERE y.yield_t_per_ha IS NOT NULL
    """
    with engine.connect() as conn:
        return pd.read_sql(sqlalchemy.text(query), conn)


def build_features(df: pd.DataFrame):
    le = LabelEncoder()
    df = df.copy()
    df["crop_encoded"] = le.fit_transform(df["crop_type"])

    feature_cols = [
        "crop_encoded", "year",
        "soil_moisture", "soil_ph", "nitrogen_ppm",
        "air_temp_c", "humidity_pct", "rainfall_mm", "solar_rad_wm2",
        "value_tonnes", "yoy_growth_pct", "ma5_tonnes", "value_normalized",
    ]
    X = df[feature_cols].fillna(0)
    y = df["yield_t_per_ha"]
    return X, y, le, feature_cols


def main():
    os.makedirs(ARTIFACT_ROOT, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_URI)

    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name("yield-prediction-xgboost")
    if exp is None:
        client.create_experiment(
            "yield-prediction-xgboost",
            artifact_location=f"file://{ARTIFACT_ROOT}/yield-prediction-xgboost",
        )
    mlflow.set_experiment("yield-prediction-xgboost")

    engine = sqlalchemy.create_engine(DATABASE_URL)
    df = load_dataset(engine)
    engine.dispose()

    if df.empty:
        print("[xgboost] Aucune donnée — lancez simulate_data.py d'abord.")
        return

    print(f"[xgboost] {len(df)} observations chargées")

    X, y, le, feature_cols = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run(run_name="xgboost-yield"):
        mlflow.log_params(PARAMS)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        mlflow.log_param("crop_classes", list(le.classes_))

        model = xgb.XGBRegressor(**PARAMS)

        # Cross-validation sur le jeu d'entraînement
        cv_rmse = -cross_val_score(
            model, X_train, y_train,
            cv=5, scoring="neg_root_mean_squared_error"
        )
        mlflow.log_metric("cv_rmse_mean", round(cv_rmse.mean(), 4))
        mlflow.log_metric("cv_rmse_std",  round(cv_rmse.std(), 4))

        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)

        mlflow.log_metric("test_rmse", round(rmse, 4))
        mlflow.log_metric("test_mae",  round(mae, 4))
        mlflow.log_metric("test_r2",   round(r2, 4))

        # Feature importances
        importances = dict(zip(feature_cols, model.feature_importances_.round(4).tolist()))
        mlflow.log_dict(importances, "feature_importances.json")

        mlflow.xgboost.log_model(model, artifact_path="model", registered_model_name="yield-xgboost")

        print(f"[xgboost] RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")
        top3 = sorted(importances.items(), key=lambda x: -x[1])[:3]
        print(f"[xgboost] Top features: {top3}")


if __name__ == "__main__":
    main()
