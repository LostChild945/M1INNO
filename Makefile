.PHONY: setup start wait pipeline ml stop reset logs help

# Lance toute la stack depuis zéro : services → données → ML
setup: start wait pipeline ml
	@echo ""
	@echo "Stack AgriTech opérationnelle :"
	@echo "  API       → http://localhost:8000/docs"
	@echo "  Dashboard → http://localhost:8501"
	@echo "  MLflow    → http://localhost:5000"
	@echo "  Airflow   → http://localhost:8081  (admin / admin)"
	@echo "  Spark UI  → http://localhost:8080"

# Démarre tous les services Docker
start:
	docker compose up -d --build

# Attend que les services critiques soient prêts
wait:
	@echo "Attente de PostgreSQL..."
	@until docker compose exec -T postgres pg_isready -U $${POSTGRES_USER:-agritech} > /dev/null 2>&1; do sleep 2; done
	@echo "Attente de MLflow..."
	@until curl -sf http://localhost:5000/health > /dev/null 2>&1; do sleep 3; done
	@echo "Attente de l'API Spark (30s)..."
	@sleep 30
	@echo "Services prêts."

# Pipeline Spark : ingestion FAO + feature engineering → PostgreSQL
pipeline:
	@echo "Spark — Ingestion CSV → Parquet..."
	docker exec agritech-spark-master \
	  /opt/spark/bin/spark-submit \
	  --master spark://spark-master:7077 \
	  /opt/spark/work-dir/src/ingestion/ingest_pesticides.py
	@echo "Spark — Feature engineering → PostgreSQL..."
	docker exec agritech-spark-master \
	  /opt/spark/bin/spark-submit \
	  --master spark://spark-master:7077 \
	  --packages org.postgresql:postgresql:42.7.3 \
	  /opt/spark/work-dir/src/processing/transform_pesticides.py

# Pipeline ML : simulation IoT + entraînement XGBoost + Prophet
ml:
	@echo "ML — Simulation + Entraînement (peut prendre plusieurs minutes)..."
	docker compose --profile ml run --rm --build ml-runner

# Arrête tous les services (données conservées)
stop:
	docker compose down

# Supprime tous les services ET les volumes (reset complet)
reset:
	docker compose --profile ml down -v

# Affiche les logs en temps réel
logs:
	docker compose logs -f

help:
	@echo "Targets disponibles :"
	@echo "  make setup    — Lance tout depuis zéro (start + pipeline + ml)"
	@echo "  make start    — Démarre les services Docker"
	@echo "  make pipeline — Lance le pipeline Spark (pesticides)"
	@echo "  make ml       — Lance l'entraînement ML"
	@echo "  make stop     — Arrête les services"
	@echo "  make reset    — Reset complet (services + volumes)"
	@echo "  make logs     — Logs en temps réel"
