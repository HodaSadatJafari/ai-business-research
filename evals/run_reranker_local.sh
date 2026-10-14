#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_MODULE=evals.run_reranker RERANKER_ENABLED=true \
  exec "$repo_root/evals/run_six_way_local.sh" "$@"
