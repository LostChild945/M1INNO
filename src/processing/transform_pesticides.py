"""
Feature engineering sur les données FAO Pesticides Use.
Parquet brut -> Parquet features + chargement PostgreSQL via JDBC.

Features calculées :
  - yoy_growth_pct   : croissance année/année en %
  - ma5_tonnes       : moyenne mobile 5 ans (tendance lissée)
  - cagr_5y_pct      : taux de croissance annuel composé sur 5 ans
  - value_normalized : normalisation min-max par pays (0-1)
  - pct_vs_global_avg: écart en % par rapport à la moyenne mondiale de l'année

Lancement via spark-submit :
  spark-submit \
    --master spark://spark-master:7077 \
    --packages org.postgresql:postgresql:42.7.3 \
    /opt/spark/work-dir/src/processing/transform_pesticides.py
"""
import os
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

DATA_PROCESSED = os.getenv("DATA_PROCESSED", "/opt/spark/work-dir/data/processed")
PG_URL = os.getenv("PG_JDBC_URL", "jdbc:postgresql://postgres:5432/agritech")
PG_USER = os.getenv("PG_USER", "agritech")
PG_PASSWORD = os.getenv("PG_PASSWORD", "agritech_secret")


def build_features(df):
    w_area = Window.partitionBy("area").orderBy("year")
    w_area_5y = Window.partitionBy("area").orderBy("year").rowsBetween(-4, 0)
    w_area_all = Window.partitionBy("area")
    w_year = Window.partitionBy("year")

    return (
        df
        # Croissance YoY
        .withColumn(
            "yoy_growth_pct",
            F.round(
                (F.col("value_tonnes") - F.lag("value_tonnes", 1).over(w_area))
                / F.when(
                    F.lag("value_tonnes", 1).over(w_area) == 0, F.lit(None)
                ).otherwise(F.lag("value_tonnes", 1).over(w_area)) * 100,
                2,
            ),
        )
        # Moyenne mobile 5 ans
        .withColumn(
            "ma5_tonnes",
            F.round(F.avg("value_tonnes").over(w_area_5y), 2),
        )
        # CAGR 5 ans : (V_t / V_{t-5})^(1/5) - 1
        .withColumn(
            "cagr_5y_pct",
            F.round(
                F.when(
                    F.lag("value_tonnes", 5).over(w_area) > 0,
                    (
                        F.pow(
                            F.col("value_tonnes") / F.lag("value_tonnes", 5).over(w_area),
                            F.lit(1.0 / 5),
                        )
                        - 1
                    )
                    * 100,
                ).otherwise(F.lit(None)),
                2,
            ),
        )
        # Normalisation min-max par pays
        .withColumn("_min", F.min("value_tonnes").over(w_area_all))
        .withColumn("_max", F.max("value_tonnes").over(w_area_all))
        .withColumn(
            "value_normalized",
            F.round(
                (F.col("value_tonnes") - F.col("_min"))
                / F.when(
                    F.col("_max") - F.col("_min") == 0, F.lit(1.0)
                ).otherwise(F.col("_max") - F.col("_min")),
                4,
            ),
        )
        .drop("_min", "_max")
        # Écart % par rapport à la moyenne mondiale de l'année
        .withColumn("_global_avg", F.avg("value_tonnes").over(w_year))
        .withColumn(
            "pct_vs_global_avg",
            F.round(
                F.when(
                    F.col("_global_avg") > 0,
                    (F.col("value_tonnes") - F.col("_global_avg"))
                    / F.col("_global_avg") * 100,
                ).otherwise(F.lit(None)),
                2,
            ),
        )
        .drop("_global_avg")
    )


def write_to_postgres(df):
    (
        df.write
        .format("jdbc")
        .option("url", PG_URL)
        .option("dbtable", "pesticide_use")
        .option("user", PG_USER)
        .option("password", PG_PASSWORD)
        .option("driver", "org.postgresql.Driver")
        .option("truncate", "true")
        .mode("overwrite")
        .save()
    )


def main():
    spark = (
        SparkSession.builder
        .appName("agritech-transform-pesticides")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.parquet(f"{DATA_PROCESSED}/pesticides_raw")
    df_features = build_features(df)

    # Parquet features (partitionné par année pour requêtes temporelles rapides)
    (
        df_features.write
        .mode("overwrite")
        .partitionBy("year")
        .parquet(f"{DATA_PROCESSED}/pesticides_features")
    )

    write_to_postgres(df_features)

    count = df_features.count()
    print(f"[transform] {count} lignes -> Parquet + PostgreSQL")
    spark.stop()


if __name__ == "__main__":
    main()
