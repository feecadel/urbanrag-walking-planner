#!/usr/bin/env bash
set -euo pipefail

# Configuration defaults
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_USER="${DB_USER:-postgres}"
DB_NAME="${DB_NAME:-geovector}"
APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-8000}"

echo "[INFO] Checking PostgreSQL connectivity at ${DB_HOST}:${DB_PORT}..."
until pg_isready -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" > /dev/null 2>&1; do
  sleep 1
done
echo "[INFO] PostgreSQL is available and accepting connections."

export PGPASSWORD="${DB_PASSWORD:-postgrespassword}"

# Verify schema and table presence
TABLE_EXISTS=$(psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -tAc \
  "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'places');")

ROW_COUNT=0
if [ "${TABLE_EXISTS}" = "t" ]; then
  ROW_COUNT=$(psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -tAc "SELECT count(*) FROM places;")
fi

if [ "${ROW_COUNT}" -eq 0 ]; then
  echo "[INFO] Table 'places' is uninitialized or empty. Running ingest_osm.py bootstrap..."
  python -m src.ingest_osm
  echo "[INFO] POI ingestion and vector index creation completed."
else
  echo "[INFO] POI database verified: ${ROW_COUNT} records indexed."
fi

# Ensure data directory exists for topological graph cache
mkdir -p data

echo "[INFO] Starting UrbanRAG engine service on ${APP_HOST}:${APP_PORT}..."
exec uvicorn src.main:app --host "${APP_HOST}" --port "${APP_PORT}"