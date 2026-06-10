"""
AgriTech Dashboard — Streamlit + Folium + Plotly

Pages :
  - Carte des parcelles  : carte Folium avec markers par culture
  - Prédiction           : formulaire → POST /predict/yield via API
  - Historique           : tableau + graphique des prédictions stockées
  - Pesticides           : séries temporelles FAO + forecast Prophet
"""
import os

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_folium import st_folium

API_URL = os.getenv("API_URL", "http://localhost:8000")

CROP_COLORS = {
    "corn":      "#f4a261",
    "rice":      "#2a9d8f",
    "soybean":   "#e9c46a",
    "sunflower": "#e76f51",
    "wheat":     "#264653",
}

st.set_page_config(
    page_title="AgriTech Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("AgriTech")
st.sidebar.caption("Analytics Agricole & Prédiction de Rendement")
page = st.sidebar.radio(
    "Navigation",
    ["Carte des parcelles", "Prédiction de rendement", "Historique", "Pesticides"],
)


# ── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def api_get(path: str):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=8)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return None
    except Exception as e:
        st.error(f"Erreur API ({path}) : {e}")
        return None


def api_post(path: str, payload: dict):
    try:
        r = requests.post(f"{API_URL}{path}", json=payload, timeout=8)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        detail = e.response.json().get("detail", str(e)) if e.response else str(e)
        st.error(f"Erreur API : {detail}")
        return None
    except Exception as e:
        st.error(f"Erreur API : {e}")
        return None


def check_api():
    health = api_get("/health")
    if health is None:
        st.warning("L'API n'est pas joignable. Vérifiez que le service `api` est démarré.")
        st.stop()
    return health


# ── Page : Carte des parcelles ────────────────────────────────────────────────

if page == "Carte des parcelles":
    st.title("Carte des parcelles")

    health = check_api()
    col_status, col_model = st.columns(2)
    col_status.metric("API", "En ligne")
    col_model.metric(
        "Modèle XGBoost",
        f"v{health.get('model_version', '?')}" if health.get("model_loaded") else "Non chargé",
    )

    parcels = api_get("/parcels")
    if not parcels:
        st.info("Aucune parcelle en base. Lancez `simulate_data.py` pour générer les données.")
        st.stop()

    df = pd.DataFrame(parcels)

    col_map, col_table = st.columns([3, 2])

    with col_map:
        m = folium.Map(location=[20, 10], zoom_start=2, tiles="CartoDB positron")
        for _, row in df.iterrows():
            color = CROP_COLORS.get(row.get("crop_type", ""), "#888888")
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=7,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                tooltip=(
                    f"<b>{row.get('name', '')}</b><br>"
                    f"Culture : {row.get('crop_type', '')}<br>"
                    f"Pays : {row.get('country', '')}<br>"
                    f"Surface : {row.get('area_ha', '')} ha"
                ),
            ).add_to(m)
        st_folium(m, width=700, height=450)

    with col_table:
        st.subheader("Parcelles")
        cols_show = ["id", "name", "crop_type", "country", "area_ha", "soil_type"]
        cols_show = [c for c in cols_show if c in df.columns]
        st.dataframe(df[cols_show], use_container_width=True, hide_index=True)

        st.subheader("Répartition par culture")
        counts = df["crop_type"].value_counts().reset_index()
        counts.columns = ["culture", "nb_parcelles"]
        fig = px.pie(counts, names="culture", values="nb_parcelles",
                     color="culture", color_discrete_map=CROP_COLORS,
                     hole=0.4)
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)


# ── Page : Prédiction de rendement ───────────────────────────────────────────

elif page == "Prédiction de rendement":
    st.title("Prédiction de rendement")
    check_api()

    parcels = api_get("/parcels")
    if not parcels:
        st.info("Aucune parcelle en base. Lancez `simulate_data.py` pour générer les données.")
        st.stop()

    df_parcels = pd.DataFrame(parcels)
    parcel_options = {
        f"{p['id']} — {p.get('name', '')} ({p.get('crop_type', '')}, {p.get('country', '')})"
        : p
        for p in parcels
    }

    st.subheader("Sélectionner une parcelle")
    selected_label = st.selectbox("Parcelle", list(parcel_options.keys()))
    parcel = parcel_options[selected_label]

    st.subheader("Mesures capteurs")
    col1, col2, col3 = st.columns(3)
    with col1:
        year         = st.number_input("Année", min_value=1990, max_value=2030, value=2015)
        soil_moisture = st.slider("Humidité sol (%)", 0.0, 100.0, 42.0, 0.5)
        soil_ph       = st.slider("pH sol", 4.0, 9.0, 6.8, 0.1)
        nitrogen_ppm  = st.number_input("Azote (ppm)", min_value=0.0, max_value=200.0, value=55.0)
    with col2:
        air_temp_c    = st.number_input("Température air (°C)", min_value=-20.0, max_value=50.0, value=14.0)
        humidity_pct  = st.slider("Humidité air (%)", 0.0, 100.0, 65.0, 0.5)
        rainfall_mm   = st.number_input("Pluviométrie (mm)", min_value=0.0, max_value=5000.0, value=320.0)
        solar_rad_wm2 = st.number_input("Radiation solaire (W/m²)", min_value=0.0, max_value=500.0, value=180.0)
    with col3:
        country = st.text_input("Pays (pour lookup pesticides)", value=parcel.get("country", ""))
        st.caption(
            "Le pays est utilisé pour récupérer les données pesticides FAO "
            "et enrichir la prédiction."
        )

    if st.button("Prédire le rendement", type="primary"):
        payload = {
            "parcel_id":    parcel["id"],
            "crop_type":    parcel["crop_type"],
            "year":         int(year),
            "soil_moisture": soil_moisture,
            "soil_ph":      soil_ph,
            "nitrogen_ppm": nitrogen_ppm,
            "air_temp_c":   air_temp_c,
            "humidity_pct": humidity_pct,
            "rainfall_mm":  rainfall_mm,
            "solar_rad_wm2": solar_rad_wm2,
            "country":      country,
        }
        with st.spinner("Calcul en cours..."):
            result = api_post("/predict/yield", payload)
        if result:
            st.success("Prédiction réalisée et enregistrée")
            r1, r2, r3 = st.columns(3)
            r1.metric("Rendement prédit", f"{result['predicted_yield']} t/ha")
            r2.metric("Irrigation recommandée", f"{result['irrigation_rec_mm']} mm")
            r3.metric("Modèle", f"{result['model_name']} v{result.get('model_version', '?')}")


# ── Page : Historique ─────────────────────────────────────────────────────────

elif page == "Historique":
    st.title("Historique des prédictions")
    check_api()

    limit = st.slider("Nombre de prédictions", 10, 200, 50)
    data = api_get(f"/predictions?limit={limit}")

    if not data:
        st.info("Aucune prédiction en base.")
        st.stop()

    df = pd.DataFrame(data)
    df["predicted_at"] = pd.to_datetime(df["predicted_at"])

    st.dataframe(
        df[["predicted_at", "parcel_name", "crop_type", "country",
            "predicted_yield", "irrigation_rec_mm", "model_version"]],
        use_container_width=True,
        hide_index=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.box(df, x="crop_type", y="predicted_yield",
                     color="crop_type", color_discrete_map=CROP_COLORS,
                     title="Distribution des rendements par culture",
                     labels={"crop_type": "Culture", "predicted_yield": "Rendement (t/ha)"})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        if "country" in df.columns:
            avg_by_country = (
                df.groupby("country")["predicted_yield"].mean()
                .sort_values(ascending=True)
                .reset_index()
            )
            fig2 = px.bar(avg_by_country, x="predicted_yield", y="country",
                          orientation="h",
                          title="Rendement moyen par pays",
                          labels={"predicted_yield": "Rendement moyen (t/ha)", "country": ""})
            st.plotly_chart(fig2, use_container_width=True)


# ── Page : Pesticides ─────────────────────────────────────────────────────────

elif page == "Pesticides":
    st.title("Utilisation des pesticides")
    check_api()

    countries = api_get("/pesticide/countries")
    if not countries:
        st.info("Aucune donnée pesticides en base. Lancez le pipeline Spark.")
        st.stop()

    country = st.selectbox("Pays", countries,
                           index=countries.index("France") if "France" in countries else 0)

    history = api_get(f"/pesticide/history/{requests.utils.quote(country)}")
    if not history:
        st.stop()

    df_hist = pd.DataFrame(history)

    col_main, col_stats = st.columns([3, 1])

    with col_main:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_hist["year"], y=df_hist["value_tonnes"],
            mode="lines+markers", name="Utilisation réelle",
            line=dict(color="#2a9d8f", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=df_hist["year"], y=df_hist["ma5_tonnes"],
            mode="lines", name="Moyenne mobile 5 ans",
            line=dict(color="#e76f51", width=2, dash="dot"),
        ))

        forecast = api_get(f"/pesticide/forecast/{requests.utils.quote(country)}")
        if forecast and "forecast" in forecast:
            df_fc = pd.DataFrame(forecast["forecast"])
            df_fc["ds"] = pd.to_datetime(df_fc["ds"])
            fig.add_trace(go.Scatter(
                x=df_fc["ds"].dt.year, y=df_fc["yhat"],
                mode="lines+markers", name="Forecast Prophet (2017-2021)",
                line=dict(color="#f4a261", width=2, dash="dash"),
            ))
            fig.add_trace(go.Scatter(
                x=list(df_fc["ds"].dt.year) + list(df_fc["ds"].dt.year[::-1]),
                y=list(df_fc["yhat_upper"]) + list(df_fc["yhat_lower"][::-1]),
                fill="toself", fillcolor="rgba(244,162,97,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Intervalle de confiance 95%",
            ))

        fig.update_layout(
            title=f"Utilisation des pesticides — {country}",
            xaxis_title="Année",
            yaxis_title="Tonnes d'ingrédients actifs",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_stats:
        st.subheader("Statistiques")
        st.metric("Total (t)", f"{df_hist['value_tonnes'].sum():,.0f}")
        st.metric("Pic", f"{df_hist['value_tonnes'].max():,.0f} t ({int(df_hist.loc[df_hist['value_tonnes'].idxmax(), 'year'])})")
        st.metric("Dernière valeur", f"{df_hist['value_tonnes'].iloc[-1]:,.0f} t ({int(df_hist['year'].iloc[-1])})")
        latest_yoy = df_hist["yoy_growth_pct"].dropna().iloc[-1] if not df_hist["yoy_growth_pct"].dropna().empty else None
        if latest_yoy is not None:
            st.metric("Croissance YoY (dernière)", f"{latest_yoy:.1f}%")

    st.subheader("Croissance annuelle (%)")
    df_yoy = df_hist[df_hist["yoy_growth_pct"].notna()].copy()
    fig_yoy = px.bar(df_yoy, x="year", y="yoy_growth_pct",
                     color="yoy_growth_pct",
                     color_continuous_scale=["#e76f51", "#f4f4f4", "#2a9d8f"],
                     color_continuous_midpoint=0,
                     labels={"year": "Année", "yoy_growth_pct": "Croissance YoY (%)"},
                     title="Variation annuelle de l'utilisation des pesticides")
    fig_yoy.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_yoy, use_container_width=True)
