#!/usr/bin/env bash
# Start everything for local development: Postgres, the API and the frontend.
#
#   ./start.sh
#   API_PORT=8001 WEB_PORT=5174 ./start.sh      # if the default ports are taken
#
# API on http://localhost:$API_PORT, app on http://localhost:$WEB_PORT (it
# proxies /api to the API). Ctrl+C stops both; Postgres keeps running
# (docker compose down).
set -euo pipefail
cd "$(dirname "$0")"

API_PORT="${API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-5173}"
export API_PORT  # read by frontend/vite.config.ts for its /api proxy

for port in "$API_PORT" "$WEB_PORT"; do
  if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "port $port is already in use; pick others, e.g. API_PORT=8001 WEB_PORT=5174 ./start.sh"
    exit 1
  fi
done

[ -f .env ] || { echo "missing .env (Postgres, agent, ingest and API keys)"; exit 1; }

echo "Starting Postgres…"
docker compose up -d
until docker exec postgres_ithome pg_isready -q; do sleep 1; done

echo "Installing dependencies…"
(cd backend && uv sync --quiet)
(cd frontend && { [ -d node_modules ] || npm install; })

# Stop both servers together, on Ctrl+C or if either one exits.
# (A polling loop, not `wait -n`: macOS ships bash 3.2, which lacks it.)
trap 'kill 0' EXIT INT TERM
(cd backend && uv run --env-file ../.env uvicorn api.main:app --reload --port "$API_PORT") &
api=$!
(cd frontend && npm run dev -- --port "$WEB_PORT" --strictPort) &
web=$!
while kill -0 "$api" 2>/dev/null && kill -0 "$web" 2>/dev/null; do sleep 1; done
