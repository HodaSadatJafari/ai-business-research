# Weaviate reranking: retrieval gains did not improve strict answer success

Run date: 2026-10-13. [Compare top-two hybrid with eight-candidate reranking in Opik](https://www.comet.com/opik/hoda/experiments/01a10627-6356-7032-830c-901dbc6453b9/compare?experiments=%5B%2201a133df-5ae5-7f89-a08d-8dc0b64b1790%22%2C%20%2201a133e0-1d4b-7dff-be65-e94f745d7125%22%5D). [Compare the exploratory three-candidate run with the same baseline](https://www.comet.com/opik/hoda/experiments/01a10627-6356-7032-830c-901dbc6453b9/compare?experiments=%5B%2201a133df-5ae5-7f89-a08d-8dc0b64b1790%22%2C%20%2201a133e2-aa9a-7e15-98a5-f9f8ae9ca58c%22%5D).

The eight-candidate [Opik readback](results/008_weaviate_reranker_comparison.json) and [retrieval audit](results/008_weaviate_reranker_retrieval.json) contain complete case-level results. The corresponding three-candidate files are [Opik readback](results/009_weaviate_reranker_k3_comparison.json) and [retrieval audit](results/009_weaviate_reranker_k3_retrieval.json).

## Setup

The baseline and both candidates used the same 17 cases, pinned dataset version `v1`, `gpt-5.6-luna` answer model, prompt hash `2ba9bf59d89e`, fixed web fixture, SQL data, Weaviate 1.34.0 collection `ResearchChunksefa50d2bf63d`, hybrid alpha `0.5`, and **two final document chunks**. The baseline took Weaviate's first two hybrid hits. The candidates used Weaviate's `reranker-transformers` integration with `cross-encoder/ms-marco-MiniLM-L-6-v2` to reorder either eight or three hybrid hits and then kept two. The local inference service used CPU PyTorch. Every Opik experiment was read back and checked for all 17 distinct case IDs and all four scores.

The eight-candidate setting was chosen before the live result. After seeing that it displaced a needed chunk, I explored candidate sizes without LLM calls. Three candidates gave full fact evidence on this dataset, so I ran a second Opik candidate experiment and reused the already measured baseline. **The three-candidate result is exploratory and selected on the evaluation cases; it is not a held-out estimate.**

To reproduce the primary paired run, use
`./evals/run_reranker_local.sh`. The exploratory run reused the saved baseline:

```bash
./evals/run_reranker_local.sh --candidate-k 3 \
  --report-stem 009_weaviate_reranker_k3 \
  --baseline-experiment-id 01a133df-5ae5-7f89-a08d-8dc0b64b1790 \
  --baseline-report experiments/results/008_weaviate_reranker_comparison.json
```

## Results

Strict success requires every expected fact to appear in a finding with a valid source ID and an exact quote containing the fact. A fact merely appearing in the summary is insufficient.

| Metric | Hybrid, top 2 | Rerank 8 → 2 | Rerank 3 → 2, exploratory |
| --- | ---: | ---: | ---: |
| Strict task success | 15/17 | 15/17 | 13/17 |
| Fact coverage | 94.1% | 97.1% | 100% |
| Grounded fact coverage | 94.1% | 91.2% | 85.3% |
| Citation integrity | 100% | 94.1% | 82.4% |
| Full document-fact evidence in retrieval audit | 9/11 | 10/11 | 11/11 |
| Mean retrieval time in Opik item output | 0.008 s | 0.222 s | 0.130 s |
| Mean end-to-end item time | 2.600 s | 2.553 s | 2.561 s |
| Prompt / completion tokens, total | 10,587 / 4,661 | 10,498 / 4,697 | 10,490 / 4,556 |

The eight-candidate run fixed `doc_sql_echomap`: it included the company pricing chunk, allowing the answer to cite the $1,200 starting price alongside the SQL ticket count. It still missed `hybrid_price_queue`: although it promoted the $480 pricing chunk, an unrelated data-policy chunk displaced the queue-dashboard chunk. It also regressed `sql_signaldesk_july_tickets`: the response gave the correct 128 tickets but used an invalid source ID for its finding. Thus the extra fact access did not raise strict success.

At three candidates, both previously missed questions passed. Four other cases regressed: `doc_sql_signaldesk`, `doc_web_trends`, `sql_echomap_august_response`, and `sql_signaldesk_july_tickets`. Three had source-ID or uncited-derived-claim errors. In `doc_web_trends`, the answer stated the expected phrase in its summary and quote, but the finding's claim used different wording, so the exact-phrase grounded-fact scorer did not count it. These cases show that having the correct chunks is necessary but does not guarantee a fully grounded structured answer.

## Interpretation and limits

Reranking improved access to the missing pricing facts, but **neither measured configuration improved end-to-end quality**. The eight-candidate option also made retrieval materially slower. The apparent end-to-end latency decrease is within provider and single-run variation; it is not evidence that reranking speeds up requests. Token totals are recorded, but this run has no stable monetary cost estimate. Each answer variant was run once per case, so differences in wording and citations may include answer-model variation. The three-candidate selection used these same cases, making it optimistic as a retrieval tuning result. The small fictional corpus, fixed web fixture, and preselected SQL input do not measure live web search or autonomous agent tool selection.

The cross-encoder is selected by model name without a pinned Hugging Face
revision, so a later rebuild may fetch different weights. Rerun the retrieval
audit before comparing a rebuilt image with these results.

Keep ordinary hybrid retrieval as the default. The `hybrid_rerank` method remains available for further experiments. The next useful test is to validate source IDs and exact quotes before accepting a structured answer, and then rerun a held-out paired evaluation. Candidate size and reranker model should be selected on development cases, with the held-out set reserved for the final comparison.

The reranker integration is opt-in through `RERANKER_ENABLED=true` and the
`compose.reranker.yaml` override; the local reranker evaluation script sets
both for the experiment. BM25, vector, and hybrid search can still use a
Weaviate deployment without the local reranker module.

Verification: `uv run pytest -q` passed 55 tests; `uv run ruff check .`
passed. The live retrieval audit and all three Opik experiment readbacks
completed. The opt-in configuration was also checked against the measured
local collection without a new answer-model call.
