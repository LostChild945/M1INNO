# AgriTech — Analytics Agricole & Prédiction de Rendement

Projet de precision agriculture combinant traitement Big Data Spark, IoT simulé et Machine Learning pour prédire les rendements et optimiser l'irrigation.

---

## Stack technique

| Couche | Technologies |
|---|---|
| Traitement Big Data | PySpark 3.5, Parquet |
| Orchestration | Apache Airflow 2.9 |
| Stockage | PostgreSQL 16 |
| Machine Learning | XGBoost 2.0, Prophet 1.1, MLflow 3.1 |
| API | FastAPI |
| Dashboard | Streamlit, Folium |
| Infrastructure | Docker Compose |

**Ressources cibles :** 8 cores CPU · 15-16 GB RAM · 60 GB stockage

---

## Architecture

```
données brutes (CSV / IoT)
        │
        ▼
  PySpark — Ingestion
  (CSV → Parquet brut)
        │
        ▼
  PySpark — Feature Engineering
  (Parquet brut → features → PostgreSQL via JDBC)
        │
        ▼
  ML — XGBoost (rendement) + Prophet (séries temporelles)
  MLflow (tracking expériences)
        │
        ▼
  FastAPI (endpoints prédiction)
        │
        ▼
  Streamlit + Folium (dashboard cartographique)

  Airflow orchestre l'ensemble (DAGs hebdomadaires)
```

---

## Services Docker

| Service | URL | Identifiants |
|---|---|---|
| Spark UI | http://localhost:8080 | — |
| Airflow | http://localhost:8081 | admin / admin |
| MLflow | http://localhost:5000 | — |
| FastAPI (Swagger) | http://localhost:8000/docs | — |
| Streamlit | http://localhost:8501 | — |
| PostgreSQL | localhost:5432 | agritech / agritech_secret |

---

## Lancement rapide

### Prérequis

- Docker 24+ et Docker Compose v2
- 16 GB RAM minimum alloués à Docker

### Démarrage

```bash
# 1. Copier les variables d'environnement
cp .env.example .env

# 2. Lancer la stack complète
docker compose up -d --build

# 3. Vérifier que tous les services sont up
docker compose ps
```

L'initialisation d'Airflow (migration DB + création user admin) se fait automatiquement au premier démarrage via le service `airflow-init`.

---

## Structure du projet

```
├── data/
│   ├── raw/                    # Données sources (CSV FAO, IoT simulé)
│   └── processed/              # Parquet générés par Spark
├── src/
│   ├── ingestion/              # Jobs Spark — lecture sources → Parquet brut
│   ├── processing/             # Jobs Spark — feature engineering → PostgreSQL
│   ├── ml/                     # Modèles XGBoost + Prophet + MLflow
│   ├── api/                    # FastAPI — endpoints prédiction
│   └── dashboard/              # Streamlit — visualisation
├── dags/                       # DAGs Airflow
├── infra/
│   ├── docker/                 # Dockerfiles (API, Dashboard, Airflow, MLflow)
│   └── postgres/               # Schéma SQL + migrations
├── docker-compose.yml
└── .env.example
```

---

## Pipeline de données

### Données disponibles

| Dataset | Source | Période | Description |
|---|---|---|---|
| `pesticides.csv` | FAO | 1990–2016 | Utilisation pesticides par pays (168 pays, tonnes d'ingrédients actifs) |

### Lancer le pipeline pesticides manuellement

```bash
# Étape 1 — Ingestion CSV → Parquet
docker exec agritech-spark-master \
  /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  /opt/spark/work-dir/src/ingestion/ingest_pesticides.py

# Étape 2 — Feature engineering + chargement PostgreSQL
docker exec agritech-spark-master \
  /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  --packages org.postgresql:postgresql:42.7.3 \
  /opt/spark/work-dir/src/processing/transform_pesticides.py
```

### Features calculées

| Feature | Description |
|---|---|
| `yoy_growth_pct` | Croissance annuelle en % (year-over-year) |
| `ma5_tonnes` | Moyenne mobile sur 5 ans |
| `cagr_5y_pct` | Taux de croissance annuel composé sur 5 ans |
| `value_normalized` | Normalisation min-max par pays (0–1) |
| `pct_vs_global_avg` | Écart en % par rapport à la moyenne mondiale de l'année |

### Via Airflow

Le DAG `pesticides_pipeline` orchestre les deux étapes ci-dessus avec une planification hebdomadaire. Il est accessible depuis l'interface Airflow sur http://localhost:8081.

---

## Schéma PostgreSQL

| Table | Description |
|---|---|
| `parcels` | Parcelles agricoles (localisation, culture, surface) |
| `sensor_readings` | Relevés capteurs IoT (humidité, température, pH…) |
| `ndvi_observations` | Indice de végétation NDVI par parcelle et par date |
| `yield_records` | Rendements réels historiques (t/ha) |
| `pesticide_use` | Utilisation pesticides FAO avec features calculées |
| `ml_predictions` | Prédictions ML (rendement, irrigation recommandée) |

---

## Pipeline ML

### Variables d'environnement

| Variable | Par défaut | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql://agritech:agritech_secret@localhost:5432/agritech` | Connexion PostgreSQL |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | URL du serveur MLflow |
| `MLFLOW_ARTIFACT_ROOT` | `/tmp/mlflow-artifacts` | Dossier local pour les artifacts MLflow |

### Lancer le pipeline ML manuellement

> Prérequis : la stack Docker doit être démarrée (`docker compose up -d`).

```bash
# 1. Générer les données simulées (60 parcelles, 2010–2016)
python3 src/ml/simulate_data.py

# 2a. Entraîner le modèle XGBoost (prédiction de rendement)
python3 src/ml/train_xgboost.py

# 2b. Entraîner les modèles Prophet (forecast pesticides)
python3 src/ml/train_prophet.py
```

Les étapes 2a et 2b sont indépendantes et peuvent être lancées en parallèle.

### Résultats

**XGBoost — prédiction de rendement (t/ha)**

| Métrique | Valeur |
|---|---|
| RMSE (test) | 0.6606 |
| MAE (test) | 0.4775 |
| R² (test) | 0.8643 |
| CV RMSE (5-fold) | ~0.67 |

Features les plus importantes : `value_tonnes`, `year`, `soil_ph` (voir `feature_importances.json` dans MLflow).

**Prophet — forecast utilisation pesticides (2017–2021)**

| Métrique | Valeur |
|---|---|
| MAPE moyen (top 10 pays) | 14.87 % |
| Horizon de prévision | 5 ans (2017–2021) |
| Données d'entraînement | 1990–2013 |
| Évaluation | 2014–2016 |

Les prédictions par pays sont accessibles dans MLflow sous forme de fichiers JSON (`forecasts/<pays>.json`).

### Via Airflow

Le DAG `ml_pipeline` orchestre les trois scripts avec une planification hebdomadaire (simulate → train_xgboost + train_prophet en parallèle). Il est accessible depuis l'interface Airflow sur http://localhost:8081.

---

## État d'avancement

- [x] Infrastructure Docker (Spark, Airflow, PostgreSQL, MLflow, FastAPI, Streamlit)
- [x] Pipeline ingestion données FAO Pesticides (CSV → Parquet → PostgreSQL)
- [x] Feature engineering Spark (YoY, MA5, CAGR, normalisation)
- [x] Simulation IoT (60 parcelles, 20 pays, capteurs sol/météo, 2010–2016)
- [x] Modèles ML — XGBoost (R²=0.86) + Prophet (MAPE=14.87 %)
- [ ] API FastAPI — endpoints prédiction
- [ ] Dashboard Streamlit + Folium

---

## Migrations base de données

Les migrations sont dans `infra/postgres/migrations/`. Pour appliquer une migration sur un conteneur en cours :

```bash
# Migration 001 — table pesticide_use
docker cp infra/postgres/migrations/001_add_pesticide_use.sql agritech-postgres:/tmp/
docker exec agritech-postgres psql -U agritech -d agritech -f /tmp/001_add_pesticide_use.sql

# Migration 002 — colonne country dans parcels
docker cp infra/postgres/migrations/002_add_country_to_parcels.sql agritech-postgres:/tmp/
docker exec agritech-postgres psql -U agritech -d agritech -f /tmp/002_add_country_to_parcels.sql
```
