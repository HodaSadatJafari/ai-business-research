# Weaviate Reranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare the current top-two hybrid retrieval with a local Weaviate reranker that selects two answer chunks from eight hybrid candidates.

**Architecture:** Add Weaviate's local `reranker-transformers` service and configure it on the project collection. Keep the existing `hybrid` search unchanged; add `hybrid_rerank` to the retriever. Run paired baseline and candidate Opik experiments on the same pinned 17-case dataset, prompt, answer model, and collection, then read both runs back case by case.

**Tech Stack:** Python 3.13, Weaviate 1.34 and Python client, Docker Compose, Opik, pytest, Ruff.

**Spec:** Approved in-chat design from the 2026-10-13 user request and the existing six-way evaluation in `experiments/007_weaviate_vectorizer_six_way.md`.

## Global Constraints

- Use `uv` for Python commands; run `uv run pytest -q` and `uv run ruff check .` after changes.
- Keep the answer context at two document chunks and make no paid calls for retrieval-only checks.
- Use Weaviate for all retrieval and reranking; do not add an application-side search fallback.
- Keep the current six-way results immutable and write new experiment outputs under 008 filenames.
- Never display or commit `.env` or credentials.

## Review Focus

- A missing reranker module should fail explicitly, without silently returning unreranked results.
- The candidate must rerank more than two initial hits and send only two chunks to the answer model.
- Both Opik runs must use identical cases, prompt, model, scoring, and final context size.
- A higher document-path recall must not be reported as fact-level evidence improvement.
- A failed or partial Opik run must not be reported as a complete comparison.

---

### Task 1: Local reranking retrieval

**Files:** `compose.weaviate.yaml`, `.env.example`, `app/config.py`, `app/retrieval.py`, `tests/test_weaviate_retrieval.py`.

**Interfaces:** `Retriever.search(query, *, k=2, method="hybrid_rerank", candidate_k=8)` returns at most two reranked `Hit`s. Plain `hybrid` remains the baseline path.

- [x] Write a failing test that checks the rerank query uses eight candidates, `Rerank(prop="text")`, and returns only the best two; check invalid candidate limits.
- [x] Run the focused test and confirm it fails for the missing behavior.
- [x] Configure the local reranker image in Compose and the collection; add the new search method without changing baseline search.
- [x] Run focused tests and a retrieval-only live smoke check for the two previously failed cases.

### Task 2: Paired Opik comparison

**Files:** `evals/run_reranker.py`, `evals/run_reranker_local.sh`, `evals/six_way_runner.py`, `tests/test_six_way_runner.py`.

**Interfaces:** The runner executes `hybrid_web_sql` and `hybrid_rerank_web_sql` on one pinned dataset, using the existing answer prompt and metric classes. It saves an 008 report with complete Opik readback, changed cases, retrieval diagnostics, config, and a two-experiment compare link.

- [x] Write failing tests for variant evidence routing and complete baseline/candidate comparisons.
- [x] Run the focused tests and confirm expected failures.
- [x] Implement the runner and a local Docker wrapper without overwriting 007 results.
- [x] Run the focused tests, full pytest suite, Ruff, and shell syntax check.

### Task 3: Measurement and report

**Files:** `experiments/008_weaviate_reranking.md`, `experiments/results/008_weaviate_reranking.json`, `README.md`, `AGENTS.md`.

- [x] Start local Weaviate and verify both retrieval paths with no answer-model calls.
- [x] Run the paired 17-case Opik evaluation once, read back both runs, and verify case IDs and metric completeness.
- [x] Report fixes, regressions, task success, fact evidence, latency, token use, and measurement limits. Link the Opik comparison.
- [x] Re-run full pytest, Ruff, and `git diff --check`; review the final tree for credentials and unrelated files.
