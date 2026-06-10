"""
Entraînement Prophet — forecast de l'utilisation des pesticides.

Pour chaque pays du top 10 (par volume total), entraîne un modèle Prophet
sur la série 1990-2013, évalue sur 2014-2016, puis génère une prévision
sur 5 ans (2017-2021).

Tracking MLflow : MAPE par pays, modèles sérialisés, dataframe de forecast.

Lancement :
  python3 src/ml/train_prophet.py
"""
import os
import warnings
import numpy as np
import pandas as pd
import sqlalchemy
import mlflow
from prophet import Prophet

warnings.filterwarnings("ignore")

DATABASE_URL  = os.getenv("DATABASE_URL", "postgresql://agritech:agritech_secret@localhost:5432/agritech")
MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
ARTIFACT_ROOT = os.getenv("MLFLOW_ARTIFACT_ROOT", "/tmp/mlflow-artifacts")
TRAIN_CUTOFF  = 2013
TOP_N         = 10
FORECAST_YEARS = 5


def load_series(engine) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(
            sqlalchemy.text("SELECT area, year, value_tonnes FROM pesticide_use ORDER BY area, year"),
            conn,
        )


def mape(y_true, y_pred) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def train_country(df_country: pd.DataFrame):
    """Retourne (model, forecast_df, mape_test)."""
    df_prophet = df_country.rename(columns={"year_dt": "ds", "value_tonnes": "y"})

    train = df_prophet[df_prophet["ds"].dt.year <= TRAIN_CUTOFF]
    test  = df_prophet[df_prophet["ds"].dt.year >  TRAIN_CUTOFF]

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.3,
        interval_width=0.95,
    )
    model.fit(train)

    # Évaluation sur 2014-2016
    mape_score = None
    if not test.empty:
        pred_test = model.predict(test[["ds"]])
        mape_score = round(mape(test["y"].values, pred_test["yhat"].values), 2)

    # Forecast 2017-2021
    last_year  = df_prophet["ds"].max().year
    future_dates = pd.DataFrame({
        "ds": pd.date_range(
            start=f"{last_year + 1}-01-01",
            periods=FORECAST_YEARS,
            freq="YS",
        )
    })
    forecast = model.predict(future_dates)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    forecast["yhat"] = forecast["yhat"].clip(lower=0).round(1)

    return model, forecast, mape_score


def main():
    mlflow.set_tracking_uri(MLFLOW_URI)

    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name("pesticide-forecast-prophet")
    if exp is None:
        kwargs = {}
        if ARTIFACT_ROOT:
            os.makedirs(ARTIFACT_ROOT, exist_ok=True)
            kwargs["artifact_location"] = f"file://{ARTIFACT_ROOT}/pesticide-forecast-prophet"
        client.create_experiment("pesticide-forecast-prophet", **kwargs)
    mlflow.set_experiment("pesticide-forecast-prophet")

    engine = sqlalchemy.create_engine(DATABASE_URL)
    df = load_series(engine)
    engine.dispose()

    # Top N pays par volume total
    top_countries = (
        df.groupby("area")["value_tonnes"].sum()
        .nlargest(TOP_N).index.tolist()
    )

    df["year_dt"] = pd.to_datetime(df["year"].astype(str) + "-01-01")
    all_forecasts = []
    mape_scores   = {}

    with mlflow.start_run(run_name="prophet-pesticide-forecast"):
        mlflow.log_param("top_n_countries", TOP_N)
        mlflow.log_param("train_cutoff",    TRAIN_CUTOFF)
        mlflow.log_param("forecast_years",  FORECAST_YEARS)

        for country in top_countries:
            df_c = df[df["area"] == country].copy()
            model, forecast, mape_score = train_country(df_c)

            forecast["area"] = country
            all_forecasts.append(forecast)

            if mape_score is not None:
                mape_scores[country] = mape_score
                mlflow.log_metric(
                    f"mape_{country.replace(' ', '_').replace(',', '')}",
                    mape_score,
                )

            country_key = country.replace(' ', '_').replace(',', '').replace('/', '_')
            mlflow.log_dict(
                forecast.to_dict(orient="records"),
                f"forecasts/{country_key}.json",
            )
            print(f"[prophet] {country:<35} MAPE={mape_score}%")

        # Log forecast consolidé
        df_all = pd.concat(all_forecasts, ignore_index=True)
        df_all["ds"] = df_all["ds"].dt.strftime("%Y-%m-%d")
        mlflow.log_dict(
            df_all.to_dict(orient="records"),
            "forecast_2017_2021.json",
        )

        avg_mape = round(np.mean(list(mape_scores.values())), 2)
        mlflow.log_metric("avg_mape", avg_mape)
        print(f"[prophet] MAPE moyen = {avg_mape}%")


if __name__ == "__main__":
    main()
