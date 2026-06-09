"""
Ingestion FAO Pesticides Use : CSV brut -> Parquet partitionné par année.

Lancement via spark-submit :
  spark-submit --master spark://spark-master:7077 \
    /opt/spark/work-dir/src/ingestion/ingest_pesticides.py
"""
import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, FloatType

DATA_RAW = os.getenv("DATA_RAW", "/opt/spark/work-dir/data/raw")
DATA_PROCESSED = os.getenv("DATA_PROCESSED", "/opt/spark/work-dir/data/processed")


def main():
    spark = (
        SparkSession.builder
        .appName("agritech-ingest-pesticides")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "false")
        .csv(f"{DATA_RAW}/pesticides.csv")
    )

    df = (
        df
        .select(
            F.col("Area").alias("area"),
            F.col("Year").cast(IntegerType()).alias("year"),
            F.col("Value").cast(FloatType()).alias("value_tonnes"),
        )
        .dropna(subset=["area", "year", "value_tonnes"])
        .filter(F.col("value_tonnes") >= 0)
    )

    (
        df.write
        .mode("overwrite")
        .partitionBy("year")
        .parquet(f"{DATA_PROCESSED}/pesticides_raw")
    )

    count = df.count()
    print(f"[ingest] {count} lignes écrites -> {DATA_PROCESSED}/pesticides_raw")
    spark.stop()


if __name__ == "__main__":
    main()
