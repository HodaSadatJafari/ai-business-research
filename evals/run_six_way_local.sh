#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
eval_module="${EVAL_MODULE:-evals.run_six_way}"

if [[ ! -f .env ]]; then
  echo "Create .env from .env.example and configure the answer model and Opik." >&2
  exit 1
fi

compose_files=(-f compose.weaviate.yaml)
if [[ "${RERANKER_ENABLED:-false}" == "true" ]]; then
  compose_files+=(-f compose.reranker.yaml)
fi
docker --context default compose "${compose_files[@]}" up -d
docker --context default build -t ai-business-research:weaviate .
docker --context default run --rm --network host \
  --user "$(id -u):$(id -g)" \
  -e WEAVIATE_URL=http://127.0.0.1:18080 \
  -e WEAVIATE_GRPC_PORT=15051 \
  -e WEAVIATE_API_KEY= \
  -e "RERANKER_ENABLED=${RERANKER_ENABLED:-false}" \
  -e UV_CACHE_DIR=/tmp/uv-cache \
  -e OPIK_USAGE_REPORT_ENABLED=false \
  -v "$repo_root/.env:/app/.env:ro" \
  -v "$repo_root/evals:/app/evals:ro" \
  -v "$repo_root/experiments:/app/experiments" \
  ai-business-research:weaviate \
  uv run --frozen --no-sync --no-dev python -m "$eval_module" "$@"
