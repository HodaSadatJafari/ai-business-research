#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ ! -f .env ]]; then
  echo "Create .env from .env.example and configure the answer model and Opik." >&2
  exit 1
fi

docker --context default compose -f compose.weaviate.yaml up -d
docker --context default build -t ai-business-research:weaviate .
docker --context default run --rm --network host \
  --user "$(id -u):$(id -g)" \
  -e WEAVIATE_URL=http://127.0.0.1:18080 \
  -e WEAVIATE_GRPC_PORT=15051 \
  -e WEAVIATE_API_KEY= \
  -e UV_CACHE_DIR=/tmp/uv-cache \
  -e OPIK_USAGE_REPORT_ENABLED=false \
  -v "$repo_root/.env:/app/.env:ro" \
  -v "$repo_root/evals:/app/evals:ro" \
  -v "$repo_root/experiments:/app/experiments" \
  ai-business-research:weaviate \
  uv run --frozen --no-sync --no-dev python -m evals.run_six_way "$@"
