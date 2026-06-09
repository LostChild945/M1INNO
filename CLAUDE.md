# AgriTech M1INNO — CLAUDE.md

## Présentation du projet

Plateforme MLOps AgriTech complète déployée via Docker Compose. Elle ingère des données de capteurs agricoles (IoT), les traite avec Spark, orchestre les pipelines avec Airflow, suit les expériences ML avec MLflow, expose des prédictions via une API FastAPI et les visualise dans un dashboard Streamlit.

**Stack technique :**
- **API** : FastAPI + SQLAlchemy + psycopg2 (`src/api/`)
- **Dashboard** : Streamlit (`src/dashboard/`)
- **Pipelines** : Apache Airflow 2.9.3 DAGs dans `dags/`
- **Traitement** : Apache Spark 3.5.3
- **ML tracking** : MLflow 2.14.3 (backend PostgreSQL + artifacts volume)
- **Base de données** : PostgreSQL 16 (3 instances : projet, Airflow, MLflow)
- **Infra** : Docker Compose (`docker-compose.yml`), Dockerfiles dans `infra/docker/`
- **CI/CD** : GitHub Actions + semantic-release (conventional commits)

## Architecture Docker

| Service | Port | Usage |
|---|---|---|
| `postgres` | 5432 | Données projet (parcelles, capteurs, ML) |
| `postgres-airflow` | — | Métadonnées Airflow (interne) |
| `postgres-mlflow` | — | Backend MLflow (interne) |
| `spark-master` | 8080, 7077 | Spark UI + master |
| `spark-worker` | — | Worker Spark |
| `mlflow` | 5000 | Tracking server |
| `airflow-webserver` | 8081 | UI Airflow |
| `airflow-scheduler` | — | Scheduler Airflow |
| `api` | 8000 | FastAPI prédiction |
| `dashboard` | 8501 | Streamlit |

Tous les services partagent le réseau `agritech-net`.

## Schéma de base de données

Tables dans `infra/postgres/init.sql` :
- `parcels` — parcelles agricoles (crop_type, area_ha, lat/lon, soil_type)
- `sensor_readings` — relevés IoT (moisture, temp, pH, N, humidité, pluie, radiation solaire)
- `ndvi_observations` — index de végétation par date
- `yield_records` — rendements historiques (t/ha)
- `ml_predictions` — prédictions stockées (model_name, version, yield, irrigation, confidence)

## Commandes essentielles

```bash
# Démarrer toute la stack
docker compose up -d

# Rebuild un service après modification
docker compose build <service> && docker compose up -d <service>

# Logs en temps réel
docker compose logs -f <service>

# Arrêt propre
docker compose down

# Reset complet (données incluses)
docker compose down -v
```

## Variables d'environnement

Copier `.env.example` vers `.env` avant de démarrer. Ne jamais committer `.env`.

## Conventions de commit (obligatoires)

Le projet utilise **semantic-release** — les messages de commit pilotent le versioning automatique sur `main`.

| Préfixe | Effet | Exemple |
|---|---|---|
| `feat:` | bump mineur (0.x.0) | `feat(api): add yield prediction endpoint` |
| `fix:` | bump patch (0.0.x) | `fix(infra): correct mlflow dependency` |
| `feat!:` ou `BREAKING CHANGE:` | bump majeur (x.0.0) | `feat!: redesign sensor schema` |
| `chore:`, `docs:`, `style:`, `refactor:`, `test:`, `ci:` | pas de release | `docs: update README` |

Format : `<type>(<scope optionnel>): <description en minuscules>`

Scopes utiles : `api`, `dashboard`, `dags`, `infra`, `spark`, `mlflow`, `db`

## Structure des fichiers

```
M1INNO/
├── dags/               # DAGs Airflow
├── data/
│   ├── raw/            # Données brutes (non versionnées)
│   └── processed/      # Données traitées (non versionnées)
├── infra/
│   ├── docker/         # Dockerfiles (api, dashboard, mlflow)
│   └── postgres/       # init.sql — schéma initial
├── src/
│   ├── api/            # FastAPI app + requirements.txt
│   └── dashboard/      # Streamlit app + requirements.txt
├── docker-compose.yml
├── .env.example
└── .releaserc.json
```

## Règles de développement

- Ne jamais modifier `.env` — utiliser `.env.example` comme référence
- Les dépendances Python vont dans `src/api/requirements.txt` ou `src/dashboard/requirements.txt` selon le service
- Les nouveaux Dockerfiles vont dans `infra/docker/`
- Les scripts SQL de migration vont dans `infra/postgres/`
- `data/raw/` et `data/processed/` ne sont pas versionnés (gitignored)
- Les DAGs Airflow utilisent `PYTHONPATH=/opt/airflow/src` — importer depuis `src/` directement

## Modèles ML

- Framework principal : XGBoost (déjà dans requirements)
- Tracking obligatoire via MLflow (`http://mlflow:5000` depuis les conteneurs)
- Les prédictions doivent être loggées dans la table `ml_predictions`
- Utiliser les run_id MLflow pour tracer les versions de modèle

## Tests

Pas encore de framework de test configuré. Lors de l'ajout de tests :
- API : pytest + httpx (TestClient FastAPI)
- Pas de mock de base de données — utiliser une instance PostgreSQL de test dédiée
