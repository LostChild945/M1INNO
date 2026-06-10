"""
DAG Airflow : pipeline ML AgriTech.

Étapes :
  1. simulate_data      — génère parcelles, capteurs et rendements (idempotent)
  2. train_xgboost      — prédiction rendement, tracking MLflow
  3. train_prophet      — forecast pesticides, tracking MLflow

Planification : hebdomadaire (re-entraînement régulier).
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

ML_SRC  = "/opt/airflow/src/ml"
DB_URL  = "postgresql://agritech:agritech_secret@postgres:5432/agritech"
MF_URI  = "http://mlflow:5000"

_env = {
    "DATABASE_URL":          DB_URL,
    "MLFLOW_TRACKING_URI":   MF_URI,
}

default_args = {
    "owner":          "agritech",
    "retries":        1,
    "retry_delay":    timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="ml_pipeline",
    description="Simulation données + entraînement XGBoost & Prophet",
    schedule_interval="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["agritech", "ml", "xgboost", "prophet"],
) as dag:

    simulate = BashOperator(
        task_id="simulate_agricultural_data",
        bash_command=f"python3 {ML_SRC}/simulate_data.py",
        env=_env,
    )

    train_xgb = BashOperator(
        task_id="train_xgboost_yield",
        bash_command=f"python3 {ML_SRC}/train_xgboost.py",
        env=_env,
    )

    train_proph = BashOperator(
        task_id="train_prophet_forecast",
        bash_command=f"python3 {ML_SRC}/train_prophet.py",
        env=_env,
    )

    simulate >> [train_xgb, train_proph]
