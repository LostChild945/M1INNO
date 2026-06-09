"""
DAG Airflow : pipeline données FAO Pesticides Use.

Étapes :
  1. ingest_csv_to_parquet  — spark-submit ingest_pesticides.py
  2. transform_and_load     — spark-submit transform_pesticides.py (features + JDBC -> PG)

Prérequis : le conteneur Airflow doit avoir accès au socket Docker
(/var/run/docker.sock monté en volume).
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

SPARK_MASTER = "spark://spark-master:7077"
SPARK_CONTAINER = "agritech-spark-master"
WORK_DIR = "/opt/spark/work-dir"
PG_JDBC = "jdbc:postgresql://postgres:5432/agritech"

_submit = (
    f"docker exec {SPARK_CONTAINER} "
    f"/opt/spark/bin/spark-submit --master {SPARK_MASTER}"
)

default_args = {
    "owner": "agritech",
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
    "email_on_failure": False,
}

with DAG(
    dag_id="pesticides_pipeline",
    description="FAO Pesticides : CSV -> Parquet -> features -> PostgreSQL",
    schedule_interval="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["agritech", "etl", "pesticides", "spark"],
) as dag:

    ingest = BashOperator(
        task_id="ingest_csv_to_parquet",
        bash_command=(
            f"{_submit} "
            f"{WORK_DIR}/src/ingestion/ingest_pesticides.py"
        ),
    )

    transform_and_load = BashOperator(
        task_id="transform_and_load_to_postgres",
        bash_command=(
            f"{_submit} "
            f"--packages org.postgresql:postgresql:42.7.3 "
            f"--conf spark.executor.extraJavaOptions=-Djava.security.egd=file:/dev/./urandom "
            f"--conf spark.driver.extraJavaOptions=-Djava.security.egd=file:/dev/./urandom "
            f"{WORK_DIR}/src/processing/transform_pesticides.py"
        ),
        env={
            "PG_JDBC_URL": PG_JDBC,
            "PG_USER": "agritech",
            "PG_PASSWORD": "agritech_secret",
        },
    )

    ingest >> transform_and_load
