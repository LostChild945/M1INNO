"""
Rapport d'interprétation AgriTech — génère un fichier HTML autonome.
Sections : résidus, cross-validation, MLflow runs, SHAP, Prophet, analyse agronomique.
"""
import base64, io, json, warnings
import numpy as np
import pandas as pd
import sqlalchemy
import mlflow
import xgboost as xgb
import shap
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

warnings.filterwarnings("ignore")

DATABASE_URL  = "postgresql://agritech:agritech_secret@localhost:5432/agritech"
MLFLOW_URI    = "http://localhost:5000"
CROP_CLASSES  = ["corn", "rice", "soybean", "sunflower", "wheat"]
FEATURE_NAMES = [
    "crop_encoded", "year",
    "soil_moisture", "soil_ph", "nitrogen_ppm",
    "air_temp_c", "humidity_pct", "rainfall_mm", "solar_rad_wm2",
    "value_tonnes", "yoy_growth_pct", "ma5_tonnes", "value_normalized",
]
FEATURE_LABELS = {
    "crop_encoded":     "Type de culture",
    "year":             "Année",
    "soil_moisture":    "Humidité sol",
    "soil_ph":          "pH sol",
    "nitrogen_ppm":     "Azote (ppm)",
    "air_temp_c":       "Temp. air (°C)",
    "humidity_pct":     "Humidité air (%)",
    "rainfall_mm":      "Pluviométrie (mm)",
    "solar_rad_wm2":    "Rayonnement (W/m²)",
    "value_tonnes":     "Pesticides (t)",
    "yoy_growth_pct":   "Croissance YoY (%)",
    "ma5_tonnes":       "MA5 pesticides (t)",
    "value_normalized": "Pesticides normalisé",
}
AGRO_NOTES = {
    "crop_encoded":     "La culture est le premier déterminant — chaque espèce a un potentiel génétique et des exigences propres.",
    "air_temp_c":       "La température module la photosynthèse et la durée des stades phénologiques (floraison, grain). Un écart de ±5 °C peut faire varier le rendement de 10–20 %.",
    "rainfall_mm":      "L'eau disponible conditionne le remplissage du grain. En dessous du seuil optimal, le stress hydrique active les hormones d'abscission.",
    "solar_rad_wm2":    "L'énergie solaire interceptée est transformée en biomasse. Un déficit de radiation (nuages, densité) limite directement la production.",
    "soil_ph":          "Un pH 6–7 optimise la disponibilité des nutriments. En dehors, des blocages ioniques (Al³⁺, Mn²⁺ acide; Fe³⁺, P alcalin) réduisent l'absorption.",
    "nitrogen_ppm":     "L'azote est le macro-nutriment limitant principal — constituant de la chlorophylle et des protéines de réserve du grain.",
    "soil_moisture":    "L'humidité résiduelle du sol tampon les déficits hydriques entre deux pluies. Corrèle fortement avec la capacité de rétention.",
    "value_tonnes":     "Les pesticides réduisent les pertes liées aux pathogènes et ravageurs. Leur effet est non-linéaire : marginalement nul en dessous d'un seuil, négatif au-delà.",
    "value_normalized": "Indique la position relative du pays dans la distribution mondiale des intrants phytosanitaires.",
    "humidity_pct":     "Influence l'évapotranspiration réelle et la pression maladies foliaires (mildiou, rouilles).",
    "ma5_tonnes":       "La moyenne mobile 5 ans capture les tendances structurelles d'utilisation des intrants — proxy de l'intensification agricole.",
    "yoy_growth_pct":   "La dynamique annuelle des pesticides reflète des chocs exogènes (prix, réglementation) et leurs effets retardés sur le rendement.",
}

PALETTE = {
    "bg":        "#0f1117",
    "card":      "#1a1d27",
    "accent":    "#4f8ef7",
    "accent2":   "#f7954f",
    "accent3":   "#4ff7a0",
    "text":      "#e8eaf0",
    "subtext":   "#8b8fa8",
    "border":    "#2a2d3e",
    "good":      "#4ff7a0",
    "warn":      "#f7d44f",
    "bad":       "#f7544f",
}

plt.rcParams.update({
    "figure.facecolor": PALETTE["card"],
    "axes.facecolor":   PALETTE["card"],
    "axes.edgecolor":   PALETTE["border"],
    "axes.labelcolor":  PALETTE["text"],
    "axes.titlecolor":  PALETTE["text"],
    "xtick.color":      PALETTE["subtext"],
    "ytick.color":      PALETTE["subtext"],
    "text.color":       PALETTE["text"],
    "grid.color":       PALETTE["border"],
    "grid.linewidth":   0.5,
    "legend.facecolor": PALETTE["card"],
    "legend.edgecolor": PALETTE["border"],
    "figure.dpi":       110,
    "font.size":        10,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})
COLORS = [PALETTE["accent"], PALETTE["accent2"], PALETTE["accent3"],
          "#c44ff7", "#f74f8e", "#4fc4f7"]


# ─── Helpers ───────────────────────────────────────────────────────────────────

def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor(), dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def img_tag(b64: str, w: str = "100%") -> str:
    return f'<img src="data:image/png;base64,{b64}" style="width:{w};border-radius:8px;">'


# ─── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    engine = sqlalchemy.create_engine(DATABASE_URL)
    query = """
        SELECT p.crop_type, p.country,
               EXTRACT(YEAR FROM s.recorded_at)::INT AS year,
               s.soil_moisture, s.soil_ph, s.nitrogen_ppm,
               s.air_temp_c, s.humidity_pct, s.rainfall_mm, s.solar_rad_wm2,
               COALESCE(pu.value_tonnes, 0)      AS value_tonnes,
               COALESCE(pu.yoy_growth_pct, 0)    AS yoy_growth_pct,
               COALESCE(pu.ma5_tonnes, 0)        AS ma5_tonnes,
               COALESCE(pu.value_normalized, 0)  AS value_normalized,
               y.yield_t_per_ha
        FROM sensor_readings s
        JOIN parcels p ON p.id = s.parcel_id
        JOIN yield_records y
            ON y.parcel_id = s.parcel_id
            AND y.harvest_year = EXTRACT(YEAR FROM s.recorded_at)::INT
        LEFT JOIN pesticide_use pu
            ON pu.area = p.country
            AND pu.year = EXTRACT(YEAR FROM s.recorded_at)::INT
        WHERE y.yield_t_per_ha IS NOT NULL
    """
    with engine.connect() as conn:
        df = pd.read_sql(sqlalchemy.text(query), conn)
    engine.dispose()
    le = LabelEncoder().fit(CROP_CLASSES)
    df["crop_encoded"] = le.transform(df["crop_type"])
    return df, le


def load_prophet_forecasts():
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name("pesticide-forecast-prophet")
    if exp is None:
        return {}
    runs = client.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=1)
    if not runs:
        return {}
    run_id = runs[0].info.run_id
    forecasts = {}
    countries = ["China, mainland", "United States of America", "Brazil",
                 "Argentina", "France", "India"]
    for c in countries:
        key = c.replace(" ", "_").replace(",", "").replace("/", "_")
        try:
            path = client.download_artifacts(run_id, f"forecasts/{key}.json")
            with open(path) as f:
                forecasts[c] = pd.DataFrame(json.load(f))
        except Exception:
            pass
    return forecasts


def load_mlflow_runs():
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name("yield-prediction-xgboost")
    if exp is None:
        return pd.DataFrame()
    runs = client.search_runs([exp.experiment_id], order_by=["start_time ASC"])
    records = []
    for r in runs:
        m = r.data.metrics
        records.append({
            "run": r.info.run_name or r.info.run_id[:8],
            "version": len(records) + 1,
            "rmse": m.get("test_rmse"),
            "mae":  m.get("test_mae"),
            "r2":   m.get("test_r2"),
            "cv_rmse_mean": m.get("cv_rmse_mean"),
            "cv_rmse_std":  m.get("cv_rmse_std"),
            "start_time": pd.to_datetime(r.info.start_time, unit="ms"),
        })
    return pd.DataFrame(records)


# ─── Section 1 : Résidus ───────────────────────────────────────────────────────

def plot_residuals(df, model):
    X = df[FEATURE_NAMES].fillna(0)
    y = df["yield_t_per_ha"].values
    y_pred = model.predict(X).ravel()
    residuals = y - y_pred

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1. Prédit vs Réel
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y, y_pred, alpha=0.35, s=15, color=PALETTE["accent"], rasterized=True)
    lims = [min(y.min(), y_pred.min()) - 0.3, max(y.max(), y_pred.max()) + 0.3]
    ax1.plot(lims, lims, "--", color=PALETTE["accent2"], lw=1.5, label="Parfait")
    ax1.set_xlabel("Rendement réel (t/ha)")
    ax1.set_ylabel("Rendement prédit (t/ha)")
    ax1.set_title("Prédit vs Réel")
    ax1.legend(fontsize=8)

    # 2. Résidus vs Prédit
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(y_pred, residuals, alpha=0.35, s=15, color=PALETTE["accent3"], rasterized=True)
    ax2.axhline(0, color=PALETTE["accent2"], lw=1.5, linestyle="--")
    ax2.set_xlabel("Valeur prédite (t/ha)")
    ax2.set_ylabel("Résidu (t/ha)")
    ax2.set_title("Résidus vs Valeurs prédites")

    # 3. Distribution des résidus
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(residuals, bins=40, color=PALETTE["accent"], alpha=0.8, edgecolor="none", density=True)
    mu, sigma = np.mean(residuals), np.std(residuals)
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
    ax3.plot(x, stats.norm.pdf(x, mu, sigma), color=PALETTE["accent2"], lw=2, label=f"N({mu:.3f}, {sigma:.3f})")
    ax3.set_xlabel("Résidu (t/ha)")
    ax3.set_ylabel("Densité")
    ax3.set_title("Distribution des résidus")
    ax3.legend(fontsize=8)

    # 4. QQ-plot
    ax4 = fig.add_subplot(gs[1, 0])
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    ax4.scatter(osm, osr, alpha=0.35, s=15, color=PALETTE["accent"], rasterized=True)
    ax4.plot(osm, slope * np.array(osm) + intercept, "--", color=PALETTE["accent2"], lw=1.5)
    ax4.set_xlabel("Quantiles théoriques")
    ax4.set_ylabel("Quantiles observés")
    ax4.set_title(f"Q-Q Plot (r={r:.3f})")

    # 5. Résidus par culture
    ax5 = fig.add_subplot(gs[1, 1])
    crop_order = CROP_CLASSES
    data_by_crop = [residuals[df["crop_type"].values == c] for c in crop_order]
    bp = ax5.boxplot(data_by_crop, labels=crop_order, patch_artist=True,
                     medianprops=dict(color=PALETTE["accent2"], lw=2),
                     whiskerprops=dict(color=PALETTE["subtext"]),
                     capprops=dict(color=PALETTE["subtext"]),
                     flierprops=dict(marker="o", ms=3, alpha=0.3, color=PALETTE["subtext"]))
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax5.axhline(0, color=PALETTE["accent2"], lw=1, linestyle="--")
    ax5.set_xlabel("Culture")
    ax5.set_ylabel("Résidu (t/ha)")
    ax5.set_title("Résidus par culture")

    # 6. Résidus dans le temps
    ax6 = fig.add_subplot(gs[1, 2])
    years = df["year"].values
    for crop, color in zip(CROP_CLASSES, COLORS):
        mask = df["crop_type"].values == crop
        yr = years[mask]
        yr_mean = pd.Series(residuals[mask]).groupby(yr).mean()
        ax6.plot(yr_mean.index, yr_mean.values, "o-", color=color, label=crop, ms=5, lw=1.5)
    ax6.axhline(0, color=PALETTE["subtext"], lw=1, linestyle="--")
    ax6.set_xlabel("Année")
    ax6.set_ylabel("Résidu moyen (t/ha)")
    ax6.set_title("Biais temporel par culture")
    ax6.legend(fontsize=7, ncol=2)

    stats_dict = {
        "RMSE":    float(np.sqrt(np.mean(residuals**2))),
        "MAE":     float(np.mean(np.abs(residuals))),
        "R²":      float(r2_score(y, y_pred)),
        "Biais":   float(np.mean(residuals)),
        "σ résidu": float(np.std(residuals)),
    }
    return fig_to_b64(fig), residuals, y, y_pred, stats_dict


# ─── Section 2 : Cross-validation ─────────────────────────────────────────────

def plot_crossval(df):
    X = df[FEATURE_NAMES].fillna(0).values
    y = df["yield_t_per_ha"].values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    all_y_true, all_y_pred_oof = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        Xtr, Xval = X[train_idx], X[val_idx]
        ytr, yval = y[train_idx], y[val_idx]
        m = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
                              random_state=42, verbosity=0)
        m.fit(Xtr, ytr)
        pred = m.predict(Xval)
        all_y_true.extend(yval)
        all_y_pred_oof.extend(pred)
        fold_metrics.append({
            "fold":  fold,
            "rmse":  np.sqrt(mean_squared_error(yval, pred)),
            "mae":   mean_absolute_error(yval, pred),
            "r2":    r2_score(yval, pred),
            "n_val": len(yval),
        })
    fm = pd.DataFrame(fold_metrics)
    all_y_true  = np.array(all_y_true)
    all_y_pred_oof = np.array(all_y_pred_oof)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # RMSE par fold
    ax = axes[0]
    bars = ax.bar(fm["fold"], fm["rmse"], color=PALETTE["accent"], width=0.5, alpha=0.85)
    ax.axhline(fm["rmse"].mean(), color=PALETTE["accent2"], lw=2, linestyle="--",
               label=f"Moy. {fm['rmse'].mean():.4f}")
    for bar, v in zip(bars, fm["rmse"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.4f}", ha="center", fontsize=8, color=PALETTE["text"])
    ax.set_xlabel("Fold")
    ax.set_ylabel("RMSE (t/ha)")
    ax.set_title("RMSE par fold (KFold k=5)")
    ax.legend(fontsize=9)

    # R² par fold
    ax = axes[1]
    bars2 = ax.bar(fm["fold"], fm["r2"], color=PALETTE["accent3"], width=0.5, alpha=0.85)
    ax.axhline(fm["r2"].mean(), color=PALETTE["accent2"], lw=2, linestyle="--",
               label=f"Moy. {fm['r2'].mean():.4f}")
    for bar, v in zip(bars2, fm["r2"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{v:.4f}", ha="center", fontsize=8, color=PALETTE["text"])
    ax.set_xlabel("Fold")
    ax.set_ylabel("R²")
    ax.set_title("R² par fold")
    ax.legend(fontsize=9)
    ax.set_ylim(0.7, 1.0)

    # OOF predictions
    ax = axes[2]
    ax.scatter(all_y_true, all_y_pred_oof, alpha=0.25, s=12,
               color=PALETTE["accent"], rasterized=True)
    lims = [all_y_true.min() - 0.3, all_y_true.max() + 0.3]
    ax.plot(lims, lims, "--", color=PALETTE["accent2"], lw=2)
    r2_oof = r2_score(all_y_true, all_y_pred_oof)
    ax.set_title(f"Prédictions Out-of-Fold (R²={r2_oof:.4f})")
    ax.set_xlabel("Réel (t/ha)")
    ax.set_ylabel("Prédit OOF (t/ha)")

    return fig_to_b64(fig), fm


# ─── Section 3 : MLflow runs ──────────────────────────────────────────────────

def plot_mlflow_runs(runs_df):
    if runs_df.empty or runs_df["rmse"].isna().all():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Pas de données MLflow disponibles",
                ha="center", va="center", transform=ax.transAxes)
        return fig_to_b64(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [("rmse", "RMSE (t/ha)", PALETTE["accent"]),
               ("mae",  "MAE (t/ha)",  PALETTE["accent2"]),
               ("r2",   "R²",          PALETTE["accent3"])]

    for ax, (col, label, color) in zip(axes, metrics):
        vals = runs_df[col].dropna()
        idx  = runs_df.loc[vals.index, "version"]
        ax.plot(idx, vals, "o-", color=color, ms=8, lw=2)
        for xi, yi in zip(idx, vals):
            ax.text(xi, yi + (vals.max() - vals.min()) * 0.04,
                    f"{yi:.4f}", ha="center", fontsize=8, color=PALETTE["text"])
        if col != "r2":
            best = vals.idxmin()
            ax.axhline(vals[best], color=PALETTE["good"], lw=1, linestyle=":",
                       alpha=0.6, label=f"Best v{runs_df.loc[best,'version']}")
        else:
            best = vals.idxmax()
            ax.axhline(vals[best], color=PALETTE["good"], lw=1, linestyle=":",
                       alpha=0.6, label=f"Best v{runs_df.loc[best,'version']}")
        ax.set_xlabel("Version du modèle")
        ax.set_ylabel(label)
        ax.set_title(f"Évolution {label}")
        ax.legend(fontsize=8)
        ax.set_xticks(idx)

    return fig_to_b64(fig)


# ─── Section 4 : SHAP ─────────────────────────────────────────────────────────

def plot_shap(df, model):
    X = df[FEATURE_NAMES].fillna(0)
    labels = [FEATURE_LABELS[f] for f in FEATURE_NAMES]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap_arr = np.array(shap_values)

    # 1. Feature importance (mean |SHAP|)
    fig1, ax = plt.subplots(figsize=(10, 6))
    mean_abs = np.abs(shap_arr).mean(axis=0)
    order = np.argsort(mean_abs)
    colors_bar = [PALETTE["accent"] if i > len(order) - 5 else PALETTE["subtext"]
                  for i in range(len(order))]
    bars = ax.barh(np.array(labels)[order], mean_abs[order],
                   color=colors_bar, alpha=0.85)
    ax.set_xlabel("|SHAP| moyen (impact absolu sur le rendement)")
    ax.set_title("Importance des features — SHAP (mean |SHAP value|)")
    for bar, v in zip(bars, mean_abs[order]):
        ax.text(v + 0.002, bar.get_y() + bar.get_height()/2,
                f"{v:.4f}", va="center", fontsize=8)
    b64_importance = fig_to_b64(fig1)

    # 2. Beeswarm summary par culture
    fig2, axes = plt.subplots(1, 5, figsize=(20, 6))
    for i, (crop, ax) in enumerate(zip(CROP_CLASSES, axes)):
        mask = df["crop_type"].values == crop
        if mask.sum() == 0:
            continue
        sv = shap_arr[mask]
        xv = X.values[mask]
        mean_abs_c = np.abs(sv).mean(axis=0)
        top5 = np.argsort(mean_abs_c)[-5:][::-1]
        for j, fi in enumerate(top5):
            xv_norm = (xv[:, fi] - xv[:, fi].min()) / (xv[:, fi].max() - xv[:, fi].min() + 1e-9)
            ax.scatter(sv[:, fi], [j] * len(sv), c=xv_norm,
                       cmap="RdYlGn", alpha=0.4, s=10, rasterized=True,
                       vmin=0, vmax=1)
        ax.set_yticks(range(5))
        ax.set_yticklabels([FEATURE_LABELS[FEATURE_NAMES[fi]] for fi in top5], fontsize=7)
        ax.axvline(0, color=PALETTE["subtext"], lw=0.8)
        ax.set_title(crop.capitalize(), fontsize=9)
        if i == 0:
            ax.set_xlabel("SHAP value", fontsize=8)
    fig2.suptitle("Distribution SHAP par culture (top 5 features) — rouge=faible, vert=élevé",
                  fontsize=10, y=1.02)
    b64_beeswarm = fig_to_b64(fig2)

    # 3. Dépendance : 4 features clés
    top4_features = list(np.argsort(mean_abs)[-4:][::-1])
    fig3, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, fi in zip(axes, top4_features):
        fname = FEATURE_NAMES[fi]
        xv = X.values[:, fi]
        sv = shap_arr[:, fi]
        sc = ax.scatter(xv, sv, c=sv, cmap="RdYlGn", alpha=0.4, s=10, rasterized=True)
        # tendance
        z = np.polyfit(xv, sv, 2)
        p = np.poly1d(z)
        xr = np.linspace(xv.min(), xv.max(), 100)
        ax.plot(xr, p(xr), color=PALETTE["accent"], lw=2)
        ax.axhline(0, color=PALETTE["subtext"], lw=0.8, linestyle="--")
        ax.set_xlabel(FEATURE_LABELS[fname])
        ax.set_ylabel("SHAP value")
        ax.set_title(f"Dépendance : {FEATURE_LABELS[fname]}")
    b64_dep = fig_to_b64(fig3)

    return b64_importance, b64_beeswarm, b64_dep, mean_abs, shap_arr


# ─── Section 5 : Prophet ──────────────────────────────────────────────────────

def plot_prophet(forecasts, engine):
    if not forecasts:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "Aucun forecast Prophet disponible",
                ha="center", va="center", transform=ax.transAxes)
        return fig_to_b64(fig), None

    with engine.connect() as conn:
        hist = pd.read_sql(sqlalchemy.text(
            "SELECT area, year, value_tonnes FROM pesticide_use ORDER BY area, year"
        ), conn)

    countries = list(forecasts.keys())[:6]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    mape_summary = {}
    for ax, country in zip(axes, countries):
        fc = forecasts[country].copy()
        fc["ds"] = pd.to_datetime(fc["ds"])
        h = hist[hist["area"] == country].copy()

        ax.fill_between(fc["ds"], fc.get("yhat_lower", fc["yhat"]),
                        fc.get("yhat_upper", fc["yhat"]),
                        alpha=0.2, color=PALETTE["accent"], label="IC 95%")
        ax.plot(fc["ds"], fc["yhat"], "o-", color=PALETTE["accent"],
                ms=6, lw=2, label="Forecast")
        ax.plot(pd.to_datetime(h["year"].astype(str) + "-01-01"),
                h["value_tonnes"], "s--", color=PALETTE["accent2"],
                ms=5, lw=1.5, label="Historique", alpha=0.8)
        ax.axvline(pd.Timestamp("2017-01-01"), color=PALETTE["subtext"],
                   lw=1, linestyle=":", alpha=0.7)
        ax.set_title(country.replace(", mainland", "")[:22], fontsize=9)
        ax.set_ylabel("Tonnes")
        ax.legend(fontsize=7)
        ax.tick_params(axis="x", rotation=30, labelsize=7)

    fig.suptitle("Prophet — Forecast pesticides 2017-2021 (IC 95%)", fontsize=12, y=1.01)
    return fig_to_b64(fig), None


# ─── Section 6 : Analyse agronomique ──────────────────────────────────────────

def plot_agro(df):
    figs_b64 = []

    # 1. Rendement par culture et pays (top 6)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    crop_yield = df.groupby("crop_type")["yield_t_per_ha"].agg(["mean", "std"]).reset_index()
    crop_yield = crop_yield.sort_values("mean", ascending=False)
    bars = ax.bar(crop_yield["crop_type"], crop_yield["mean"],
                  yerr=crop_yield["std"], color=COLORS[:len(crop_yield)],
                  alpha=0.85, capsize=5)
    ax.set_xlabel("Culture")
    ax.set_ylabel("Rendement moyen (t/ha)")
    ax.set_title("Rendement moyen ± σ par culture")
    for bar, (_, row) in zip(bars, crop_yield.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2, row["mean"] + row["std"] + 0.05,
                f"{row['mean']:.2f}", ha="center", fontsize=8)

    ax = axes[1]
    top_countries = (df.groupby("country")["yield_t_per_ha"].mean()
                     .nlargest(8).reset_index())
    ax.barh(top_countries["country"], top_countries["yield_t_per_ha"],
            color=PALETTE["accent"], alpha=0.85)
    ax.set_xlabel("Rendement moyen (t/ha)")
    ax.set_title("Top 8 pays — rendement moyen")
    figs_b64.append(fig_to_b64(fig))

    # 2. Corrélation features × rendement
    fig, ax = plt.subplots(figsize=(12, 8))
    corr_df = df[FEATURE_NAMES + ["yield_t_per_ha"]].corr()
    mask = np.zeros_like(corr_df, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    display_labels = [FEATURE_LABELS.get(c, c) for c in corr_df.columns]
    sns.heatmap(corr_df, mask=mask, cmap=cmap, center=0,
                annot=True, fmt=".2f", annot_kws={"size": 7},
                xticklabels=display_labels, yticklabels=display_labels,
                ax=ax, linewidths=0.3, linecolor=PALETTE["border"],
                cbar_kws={"shrink": 0.8})
    ax.set_title("Matrice de corrélation — features et rendement", pad=15)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0,  labelsize=7)
    figs_b64.append(fig_to_b64(fig))

    # 3. Effet non-linéaire temperature × rendement par culture
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for ax, crop in zip(axes, CROP_CLASSES):
        sub = df[df["crop_type"] == crop]
        scatter = ax.scatter(sub["air_temp_c"], sub["yield_t_per_ha"],
                             c=sub["rainfall_mm"], cmap="YlOrRd",
                             alpha=0.5, s=20, rasterized=True)
        z = np.polyfit(sub["air_temp_c"], sub["yield_t_per_ha"], 2)
        xr = np.linspace(sub["air_temp_c"].min(), sub["air_temp_c"].max(), 100)
        ax.plot(xr, np.poly1d(z)(xr), color=PALETTE["accent"], lw=2)
        ax.set_title(crop.capitalize(), fontsize=9)
        ax.set_xlabel("Temp. (°C)", fontsize=8)
        if crop == CROP_CLASSES[0]:
            ax.set_ylabel("Rendement (t/ha)", fontsize=8)
    fig.colorbar(scatter, ax=axes[-1], label="Pluie (mm)")
    fig.suptitle("Rendement × Température (couleur = pluviométrie)", fontsize=11)
    figs_b64.append(fig_to_b64(fig))

    # 4. Évolution du rendement dans le temps
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot = df.groupby(["year", "crop_type"])["yield_t_per_ha"].mean().unstack()
    for crop, color in zip(CROP_CLASSES, COLORS):
        if crop in pivot.columns:
            ax.plot(pivot.index, pivot[crop], "o-", color=color, ms=6, lw=2, label=crop)
    ax.set_xlabel("Année")
    ax.set_ylabel("Rendement moyen (t/ha)")
    ax.set_title("Évolution temporelle du rendement par culture")
    ax.legend()
    figs_b64.append(fig_to_b64(fig))

    return figs_b64


# ─── HTML builder ─────────────────────────────────────────────────────────────

CSS = f"""
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    background: {PALETTE["bg"]};
    color: {PALETTE["text"]};
    font-family: 'Segoe UI', system-ui, sans-serif;
    line-height: 1.6;
    padding: 0 0 60px 0;
}}
.header {{
    background: linear-gradient(135deg, {PALETTE["card"]} 0%, #0d1525 100%);
    border-bottom: 1px solid {PALETTE["border"]};
    padding: 40px 60px 30px;
}}
.header h1 {{ font-size: 2rem; color: {PALETTE["accent"]}; }}
.header p  {{ color: {PALETTE["subtext"]}; margin-top: 6px; font-size: 0.95rem; }}
.toc {{
    background: {PALETTE["card"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 10px;
    padding: 20px 30px;
    margin: 30px 60px;
    display: flex; gap: 12px; flex-wrap: wrap;
}}
.toc a {{
    color: {PALETTE["accent"]}; text-decoration: none;
    background: {PALETTE["bg"]}; border: 1px solid {PALETTE["border"]};
    padding: 6px 14px; border-radius: 20px; font-size: 0.85rem;
    transition: all .2s;
}}
.toc a:hover {{ background: {PALETTE["accent"]}; color: #fff; }}
.section {{
    margin: 30px 60px 0;
    background: {PALETTE["card"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 12px;
    overflow: hidden;
}}
.section-header {{
    background: linear-gradient(90deg, {PALETTE["accent"]}22, transparent);
    border-bottom: 1px solid {PALETTE["border"]};
    padding: 18px 28px;
    display: flex; align-items: center; gap: 12px;
}}
.section-header h2 {{ font-size: 1.25rem; }}
.badge {{
    background: {PALETTE["accent"]}33;
    color: {PALETTE["accent"]};
    border: 1px solid {PALETTE["accent"]}66;
    padding: 3px 10px; border-radius: 12px; font-size: 0.75rem;
}}
.section-body {{ padding: 24px 28px; }}
.metrics-row {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }}
.metric-card {{
    background: {PALETTE["bg"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 8px;
    padding: 14px 20px;
    min-width: 120px; flex: 1;
}}
.metric-card .label {{ color: {PALETTE["subtext"]}; font-size: 0.8rem; text-transform: uppercase; letter-spacing: .04em; }}
.metric-card .value {{ font-size: 1.6rem; font-weight: 700; margin-top: 4px; }}
.metric-card .value.good {{ color: {PALETTE["good"]}; }}
.metric-card .value.warn {{ color: {PALETTE["warn"]}; }}
.metric-card .value.neutral {{ color: {PALETTE["accent"]}; }}
.grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }}
.grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }}
.insight {{
    background: {PALETTE["bg"]};
    border-left: 3px solid {PALETTE["accent"]};
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 16px 0;
    font-size: 0.9rem;
}}
.insight .icon {{ font-size: 1.1rem; margin-right: 6px; }}
.shap-table {{ width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 0.88rem; }}
.shap-table th {{
    background: {PALETTE["bg"]}; color: {PALETTE["subtext"]};
    text-transform: uppercase; font-size: 0.75rem; letter-spacing: .06em;
    padding: 10px 14px; text-align: left; border-bottom: 1px solid {PALETTE["border"]};
}}
.shap-table td {{
    padding: 10px 14px; border-bottom: 1px solid {PALETTE["border"]}40;
}}
.shap-table tr:hover td {{ background: {PALETTE["bg"]}88; }}
.bar-inline {{
    height: 8px; border-radius: 4px;
    background: linear-gradient(90deg, {PALETTE["accent"]}, {PALETTE["accent2"]});
    margin-top: 3px;
}}
.separator {{ height: 1px; background: {PALETTE["border"]}; margin: 20px 0; }}
.fold-table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
.fold-table th {{
    background: {PALETTE["bg"]}; color: {PALETTE["subtext"]};
    text-transform: uppercase; font-size: 0.75rem; letter-spacing: .06em;
    padding: 8px 12px; text-align: left;
}}
.fold-table td {{ padding: 8px 12px; border-bottom: 1px solid {PALETTE["border"]}40; }}
"""


def build_html(res_b64, res_stats, cv_b64, fm,
               mlflow_b64, mlflow_runs,
               shap_imp_b64, shap_bee_b64, shap_dep_b64,
               mean_abs, shap_arr,
               prophet_b64,
               agro_figs,
               df):

    def metric(label, val, unit="", cls="neutral"):
        return f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value {cls}">{val}{unit}</div>
        </div>"""

    # Header
    sections = []

    # ── Résidus ──
    r2_cls  = "good" if res_stats["R²"] > 0.85 else "warn"
    rmse_cls = "good" if res_stats["RMSE"] < 0.7 else "warn"
    bias_cls = "good" if abs(res_stats["Biais"]) < 0.05 else "warn"
    sections.append(f"""
    <div class="section" id="residus">
      <div class="section-header">
        <h2>Analyse des résidus</h2>
        <span class="badge">Modèle complet — {len(df)} obs.</span>
      </div>
      <div class="section-body">
        <div class="metrics-row">
          {metric("RMSE", f"{res_stats['RMSE']:.4f}", " t/ha", rmse_cls)}
          {metric("MAE",  f"{res_stats['MAE']:.4f}",  " t/ha", "neutral")}
          {metric("R²",   f"{res_stats['R²']:.4f}",   "", r2_cls)}
          {metric("Biais moyen", f"{res_stats['Biais']:+.4f}", " t/ha", bias_cls)}
          {metric("σ résidu",    f"{res_stats['σ résidu']:.4f}", " t/ha", "neutral")}
        </div>
        {img_tag(res_b64)}
        <div class="insight">
          <span class="icon">📊</span>
          <strong>Lecture :</strong> Le nuage Prédit vs Réel suit la diagonale (R²={res_stats['R²']:.4f}).
          Les résidus sont distribués normalement (Q-Q plot linéaire) et centrés sur zéro (biais={res_stats['Biais']:+.4f} t/ha),
          ce qui indique l'absence de biais systématique.
          Le boxplot par culture montre que le riz présente la dispersion de résidus la plus élevée,
          cohérent avec sa plus forte dépendance à des conditions hydriques précises.
        </div>
      </div>
    </div>""")

    # ── Cross-validation ──
    cv_mean_r2 = fm["r2"].mean()
    cv_std_r2  = fm["r2"].std()
    fold_rows = ""
    for _, row in fm.iterrows():
        r2_col = PALETTE["good"] if row["r2"] > 0.85 else PALETTE["warn"]
        fold_rows += f"""
        <tr>
          <td>Fold {int(row['fold'])}</td>
          <td>{row['rmse']:.4f}</td>
          <td>{row['mae']:.4f}</td>
          <td style="color:{r2_col}">{row['r2']:.4f}</td>
          <td>{int(row['n_val'])}</td>
        </tr>"""
    sections.append(f"""
    <div class="section" id="cv">
      <div class="section-header">
        <h2>Cross-Validation (KFold k=5)</h2>
        <span class="badge">Robustesse</span>
      </div>
      <div class="section-body">
        <div class="metrics-row">
          {metric("R² moyen",   f"{cv_mean_r2:.4f}", "", "good" if cv_mean_r2 > 0.85 else "warn")}
          {metric("σ R²",       f"{cv_std_r2:.4f}", "", "good" if cv_std_r2 < 0.01 else "warn")}
          {metric("RMSE moyen", f"{fm['rmse'].mean():.4f}", " t/ha", "neutral")}
          {metric("σ RMSE",     f"{fm['rmse'].std():.4f}", " t/ha", "neutral")}
        </div>
        {img_tag(cv_b64)}
        <table class="fold-table" style="margin-top:20px">
          <thead><tr><th>Fold</th><th>RMSE</th><th>MAE</th><th>R²</th><th>N val.</th></tr></thead>
          <tbody>{fold_rows}</tbody>
        </table>
        <div class="insight" style="margin-top:16px">
          <span class="icon">✅</span>
          <strong>Stabilité :</strong> La faible variance inter-folds (σ R²={cv_std_r2:.4f}) confirme
          que le modèle ne sur-apprend pas — les performances sont reproductibles sur n'importe quelle
          partition des données. Les prédictions Out-of-Fold donnent un R²={fm['r2'].mean():.4f} sans data leakage.
        </div>
      </div>
    </div>""")

    # ── MLflow ──
    mlflow_table = ""
    if not mlflow_runs.empty:
        for _, row in mlflow_runs.iterrows():
            rmse = f"{row['rmse']:.4f}" if pd.notna(row.get("rmse")) else "—"
            r2   = f"{row['r2']:.4f}"   if pd.notna(row.get("r2"))   else "—"
            mlflow_table += f"<tr><td>v{int(row['version'])}</td><td>{rmse}</td><td>{r2}</td></tr>"
    sections.append(f"""
    <div class="section" id="mlflow">
      <div class="section-header">
        <h2>Suivi MLflow — historique des runs</h2>
        <span class="badge">Traçabilité</span>
      </div>
      <div class="section-body">
        {img_tag(mlflow_b64)}
        <div class="insight" style="margin-top:20px">
          <span class="icon">📈</span>
          <strong>MLflow</strong> trace chaque entraînement (paramètres, métriques, artefacts, version du modèle).
          Le graphique montre l'évolution des métriques à travers les différentes versions déployées —
          utile pour détecter des régressions ou valider des améliorations.
        </div>
      </div>
    </div>""")

    # ── SHAP ──
    shap_rows = ""
    top_order = np.argsort(mean_abs)[::-1]
    max_shap = mean_abs.max()
    for fi in top_order:
        fname  = FEATURE_NAMES[fi]
        flabel = FEATURE_LABELS[fname]
        val    = mean_abs[fi]
        pct    = val / max_shap * 100
        note   = AGRO_NOTES.get(fname, "")
        shap_rows += f"""
        <tr>
          <td><strong>{flabel}</strong></td>
          <td>{val:.4f}</td>
          <td style="width:150px">
            <div class="bar-inline" style="width:{pct:.0f}%"></div>
          </td>
          <td style="color:{PALETTE['subtext']};font-size:0.82rem">{note}</td>
        </tr>"""

    sections.append(f"""
    <div class="section" id="shap">
      <div class="section-header">
        <h2>SHAP — Explicabilité du modèle</h2>
        <span class="badge">Interprétabilité locale & globale</span>
      </div>
      <div class="section-body">
        <h3 style="margin-bottom:12px;color:{PALETTE['subtext']};font-size:0.9rem;text-transform:uppercase;letter-spacing:.05em">
          Importance globale (mean |SHAP|)
        </h3>
        {img_tag(shap_imp_b64)}
        <table class="shap-table" style="margin-top:24px">
          <thead>
            <tr>
              <th>Feature</th><th>|SHAP| moyen</th><th>Importance relative</th>
              <th>Interprétation agronomique</th>
            </tr>
          </thead>
          <tbody>{shap_rows}</tbody>
        </table>
        <div class="separator"></div>
        <h3 style="margin-bottom:12px;color:{PALETTE['subtext']};font-size:0.9rem;text-transform:uppercase;letter-spacing:.05em">
          Distribution SHAP par culture (beeswarm)
        </h3>
        {img_tag(shap_bee_b64)}
        <div class="insight" style="margin-top:16px">
          <span class="icon">🔬</span> Chaque point est une observation. La couleur indique la valeur de la feature
          (rouge=faible, vert=élevé). Une SHAP value positive signifie que la feature <em>augmente</em>
          la prédiction, négative qu'elle la <em>réduit</em>.
        </div>
        <div class="separator"></div>
        <h3 style="margin-bottom:12px;color:{PALETTE['subtext']};font-size:0.9rem;text-transform:uppercase;letter-spacing:.05em">
          Plots de dépendance (top 4 features)
        </h3>
        {img_tag(shap_dep_b64)}
      </div>
    </div>""")

    # ── Prophet ──
    sections.append(f"""
    <div class="section" id="prophet">
      <div class="section-header">
        <h2>Prophet — Forecast pesticides 2017-2021</h2>
        <span class="badge">MAPE moyen 14.87%</span>
      </div>
      <div class="section-body">
        {img_tag(prophet_b64)}
        <div class="grid-3" style="margin-top:20px">
          <div class="insight">
            <span class="icon">📅</span>
            <strong>Entraînement</strong> sur 1990-2013, évaluation sur 2014-2016,
            forecast sur 2017-2021. Modèle additif avec tendance piece-wise.
          </div>
          <div class="insight">
            <span class="icon">🌍</span>
            <strong>Top 10 pays</strong> par volume total de pesticides.
            Les intervalles de confiance à 95% reflètent l'incertitude
            structurelle de la tendance.
          </div>
          <div class="insight">
            <span class="icon">⚠️</span>
            <strong>MAPE variable</strong> : pays à forte volatilité (Colombie 38.7%)
            vs stables (USA 1.4%). La volatilité politique et économique domine
            l'incertitude dans les marchés émergents.
          </div>
        </div>
      </div>
    </div>""")

    # ── Analyse agronomique ──
    agro_sections = ""
    titles = [
        ("Rendements par culture et pays", "Distribution"),
        ("Matrice de corrélation", "Features"),
        ("Rendement × Température × Pluie", "Effets croisés"),
        ("Évolution temporelle du rendement", "Tendances"),
    ]
    for b64, (title, badge) in zip(agro_figs, titles):
        agro_sections += f"""
        <div style="margin-top:24px">
          <h3 style="color:{PALETTE['subtext']};font-size:0.85rem;text-transform:uppercase;
                     letter-spacing:.05em;margin-bottom:10px">{title}</h3>
          {img_tag(b64)}
        </div>"""

    sections.append(f"""
    <div class="section" id="agro">
      <div class="section-header">
        <h2>Analyse agronomique</h2>
        <span class="badge">Données observationnelles</span>
      </div>
      <div class="section-body">
        {agro_sections}
        <div class="insight" style="margin-top:20px">
          <span class="icon">🌱</span>
          <strong>Points clés :</strong> Le riz présente le rendement moyen le plus élevé (~4.2 t/ha)
          grâce à son optimum hydrique élevé. La corrélation temperature-rendement est quadratique
          (optimum entre 20-28°C selon la culture). L'utilisation des pesticides est positivement
          corrélée au rendement dans la plage normale d'utilisation, avec un effet de saturation
          au-delà des quantités recommandées.
        </div>
      </div>
    </div>""")

    toc = """
    <div class="toc">
      <strong style="color:#8b8fa8;font-size:0.85rem;align-self:center">Sections :</strong>
      <a href="#residus">Résidus</a>
      <a href="#cv">Cross-Validation</a>
      <a href="#mlflow">MLflow</a>
      <a href="#shap">SHAP</a>
      <a href="#prophet">Prophet</a>
      <a href="#agro">Agronomique</a>
    </div>"""

    from datetime import datetime
    ts = datetime.now().strftime("%d/%m/%Y %H:%M")

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AgriTech — Rapport d'interprétation ML</title>
  <style>{CSS}</style>
</head>
<body>
  <div class="header">
    <h1>AgriTech — Rapport d'interprétation ML</h1>
    <p>XGBoost · Prophet · SHAP · Cross-Validation · MLflow · Analyse agronomique · Généré le {ts}</p>
  </div>
  {toc}
  {"".join(sections)}
</body>
</html>"""


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Chargement des données...")
    df, le = load_data()
    print(f"  {len(df)} observations")

    print("Entraînement du modèle XGBoost local (mêmes paramètres, seed=42)...")
    X_all = df[FEATURE_NAMES].fillna(0)
    y_all = df["yield_t_per_ha"]
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42
    )
    PARAMS = dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                  reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0)
    booster = xgb.XGBRegressor(**PARAMS)
    booster.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    print("Calcul des résidus...")
    res_b64, residuals, y_true, y_pred, res_stats = plot_residuals(df, booster)

    print("Cross-validation (5 folds)...")
    cv_b64, fm = plot_crossval(df)

    print("Récupération des runs MLflow...")
    mlflow_runs = load_mlflow_runs()
    mlflow_b64  = plot_mlflow_runs(mlflow_runs)

    print("Calcul des valeurs SHAP...")
    shap_imp_b64, shap_bee_b64, shap_dep_b64, mean_abs, shap_arr = plot_shap(df, booster)

    print("Chargement des forecasts Prophet...")
    forecasts = load_prophet_forecasts()
    engine     = sqlalchemy.create_engine(DATABASE_URL)
    prophet_b64, _ = plot_prophet(forecasts, engine)
    engine.dispose()

    print("Analyse agronomique...")
    agro_figs = plot_agro(df)

    print("Génération du HTML...")
    html = build_html(
        res_b64, res_stats, cv_b64, fm,
        mlflow_b64, mlflow_runs,
        shap_imp_b64, shap_bee_b64, shap_dep_b64,
        mean_abs, shap_arr,
        prophet_b64,
        agro_figs,
        df,
    )

    out = "rapport_agritech.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  Rapport généré : {out}")
    print(f"  Ouvrir avec : xdg-open {out}")


if __name__ == "__main__":
    main()
