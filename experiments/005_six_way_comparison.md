# Six-way Opik evidence comparison

Run date: 2026-10-04. This is a controlled, one-run pilot over 14 fictional
business-research questions. All six experiments use Opik dataset
[`business-research-six-way-4aac2548b0`](https://www.comet.com/opik/hoda/projects/01a0ae12-7606-70ae-bcdd-b318b870b675/datasets/01a10584-3422-7781-a26c-7efeadc8f78b/items),
version `v1`. [Compare all six experiments in Opik](https://www.comet.com/opik/hoda/experiments/01a10584-3422-7781-a26c-7efeadc8f78b/compare?experiments=%5B%2201a10584-3f27-7336-899a-30dbba50636f%22%2C%20%2201a10584-a9c2-7325-8b14-6fb553046672%22%2C%20%2201a10585-5696-7467-b62c-703384be5590%22%2C%20%2201a10586-171e-796a-b9c5-caacb44f55a8%22%2C%20%2201a10586-bbee-7b4e-bf91-e88a23017ec7%22%2C%20%2201a10587-605c-7359-9602-da491ebbbeeb%22%5D).
The item-level Opik readback, experiment IDs, category summaries, model tokens,
and case summaries are saved in [`results/005_six_way_comparison.json`](results/005_six_way_comparison.json).

## What was held constant

Each configuration received the same 14 questions, `gpt-5.6-luna` answer
model, answer prompt, Pydantic response schema, and top-four document limit.
Document and SQL queries were declared in the dataset, and the web result was
a fixed fictional fixture. Only the evidence available to the answer model
changed:

| Configuration | Document evidence | Web evidence | SQL evidence |
| --- | --- | --- | --- |
| Pure LLM | None | None | None |
| BM25 | BM25 | None | None |
| Semantic vector | `text-embedding-3-small` cosine ranking | None | None |
| BM25 + vector | Reciprocal-rank fusion | None | None |
| BM25 + vector + web | Reciprocal-rank fusion | Fixed fixture | None |
| BM25 + vector + web + SQL | Reciprocal-rank fusion | Fixed fixture | Read-only support metrics |

The semantic experiment uses actual embeddings through the configured
OpenAI-compatible API. It does not use the repository's existing hashed-token
`vector` method. Embedding setup took 8.999 seconds and 510 tokens for 12
document chunks and the unique queries. These shared setup costs are excluded
from the per-case latency and model-token totals below. In particular, the
pure LLM and BM25 variants do not require embeddings when run alone.

## Results

`Task success` requires every expected fact to be present in a finding with
the correct source type and locator, a valid source ID, and an exact quote.
The insufficient-evidence case additionally requires no findings and a stated
limitation. `Grounded facts` is the mean fraction of expected facts satisfying
that citation check. `Citation integrity` requires every emitted finding to
have a source ID and exact quote present in supplied evidence; an answer with
no findings passes this metric vacuously.

| Configuration | Task success | Grounded facts | Citation integrity | Mean case latency | Answer tokens, prompt + completion |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pure LLM | 1/14 (7.1%) | 7.1% | 100.0% | 1.575 s | 6,388 + 1,657 |
| BM25 | 4/14 (28.6%) | 38.1% | 57.1% | 2.655 s | 9,284 + 4,455 |
| Semantic vector | 4/14 (28.6%) | 38.1% | 71.4% | 2.525 s | 9,272 + 4,379 |
| BM25 + vector | 4/14 (28.6%) | 41.7% | 57.1% | 2.552 s | 9,363 + 4,317 |
| + web fixture | 6/14 (42.9%) | 61.9% | 64.3% | 2.604 s | 9,598 + 4,512 |
| + SQL | 12/14 (85.7%) | 85.7% | 92.9% | 2.408 s | 9,916 + 4,107 |

All three document retrieval methods recalled every annotated relevant
document within the top four (`document_recall_at_4 = 1.0`). Hybrid retrieval
improved partial grounded fact coverage by 3.6 percentage points against
either standalone retrieval method, but it did not improve strict task
success in this small corpus. Adding the web fixture resolved the two
web-only cases. Adding SQL resolved both SQL-only cases and the web+SQL and
all-three cases, as well as one document+SQL case.

Strict task success by category (numerator is successful cases):

| Configuration | Docs 4 | Web 2 | SQL 2 | Docs+SQL 2 | Docs+web 1 | Web+SQL 1 | All three 1 | Insufficient 1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pure LLM | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| BM25 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Semantic vector | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| BM25 + vector | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| + web fixture | 4 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| + SQL | 4 | 2 | 2 | 1 | 1 | 1 | 1 | 0 |

The full configuration missed `doc_sql_signaldesk`: it stated both required
numbers, but put text from the document and SQL record together in one
`evidence_quote`. That combined string occurs in neither source, so the
findings fail the exact quote check. It also missed
`insufficient_orion_price`: its summary and limitations correctly declined
to invent a price, but it emitted a finding about the absence of evidence
and cited an unrelated SignalDesk pricing passage. The strict abstention
rule requires no findings. These are answer-format and citation failures;
they do not reflect missing retrieval evidence.

## Limits and interpretation

- This measures the value of **available evidence**, not autonomous agent
  tool selection. A case receives its declared web or SQL evidence when that
  source is enabled. It does not test whether the Week 3 agent chooses the
  right tool.
- The web page is a fixed fictional fixture; no Tavily call was made. These
  results make no claim about live search coverage or source reliability.
- One model call per case, small fictional corpus, and exact phrase matching
  make these pilot numbers sensitive to wording and model variation. A
  paraphrase can count as missing, and quote containment does not establish
  that a source semantically supports a claim.
- Mean latency is measured after embedding setup and query prewarming. Token
  counts reflect answer calls, not a monetary cost comparison. The embedding
  setup tokens and time are reported separately above.
- The 100% citation integrity for pure LLM means it emitted no uncited
  findings, not that it answered the questions.

## Reproduce

This is a historical run from before document search moved to Weaviate. The
original runner revision was not committed, so this result cannot be reproduced
exactly from the current checkout. Its case-level results and configuration
remain archived here. The current `uv run python -m evals.run_six_way` command
runs the Weaviate-backed comparison described in
`007_weaviate_vectorizer_six_way.md` and makes paid answer-model calls.

The dataset source files are `evals/six_way_cases.json`,
`evals/six_way_web_fixture.json`, `data/support_metrics.csv`, and the local
knowledge-base Markdown files. A source-content hash is used in the dataset
name, and the dataset is pinned to a version for all six experiments. The
runner reads each experiment back from Opik and writes the local JSON report.
