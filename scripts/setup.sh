#!/usr/bin/env bash
# setup.sh — Lance la stack AgriTech complète depuis zéro
set -euo pipefail

YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

step() { echo -e "\n${YELLOW}▶ $1${NC}"; }
ok()   { echo -e "${GREEN}✓ $1${NC}"; }

# ── 1. Services Docker ────────────────────────────────────────────────────────
step "Démarrage des services Docker..."
docker compose up -d --build
ok "Services lancés"

# ── 2. Attente PostgreSQL ─────────────────────────────────────────────────────
step "Attente de PostgreSQL..."
until docker compose exec -T postgres pg_isready -U "${POSTGRES_USER:-agritech}" > /dev/null 2>&1; do
  sleep 2
done
ok "PostgreSQL prêt"

# ── 3. Attente MLflow ─────────────────────────────────────────────────────────
step "Attente de MLflow..."
until curl -sf http://localhost:5000/health > /dev/null 2>&1; do
  sleep 3
done
ok "MLflow prêt"

# ── 4. Attente Spark ──────────────────────────────────────────────────────────
step "Attente de Spark (30s)..."
sleep 30
ok "Spark prêt"

# ── 5. Pipeline Spark — Ingestion ─────────────────────────────────────────────
step "Spark — Ingestion CSV → Parquet..."
docker exec agritech-spark-master \
  /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  /opt/spark/work-dir/src/ingestion/ingest_pesticides.py
ok "Ingestion terminée"

# ── 6. Pipeline Spark — Feature engineering ───────────────────────────────────
step "Spark — Feature engineering → PostgreSQL..."
docker exec agritech-spark-master \
  /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  --packages org.postgresql:postgresql:42.7.3 \
  /opt/spark/work-dir/src/processing/transform_pesticides.py
ok "Feature engineering terminé"

# ── 7. Pipeline ML ────────────────────────────────────────────────────────────
step "ML — Simulation IoT + Entraînement XGBoost + Prophet..."
echo "(Cette étape peut prendre 5-10 minutes au premier lancement — compilation CmdStan)"
docker compose --profile ml run --rm ml-runner
ok "Pipeline ML terminé"

# ── Récapitulatif ─────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}Stack AgriTech opérationnelle :${NC}"
echo "  API       → http://localhost:8000/docs"
echo "  Dashboard → http://localhost:8501"
echo "  MLflow    → http://localhost:5000"
echo "  Airflow   → http://localhost:8081  (admin / admin)"
echo "  Spark UI  → http://localhost:8080"
