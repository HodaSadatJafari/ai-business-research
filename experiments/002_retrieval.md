# Experiment 002 — Local retrieval and RAG

**Status:** Week 2 complete for the local fictional corpus; Week 3 can begin.

## Hypothesis

A small domain knowledge base should let the research service answer product-specific questions with traceable evidence. Retrieval must be measured before judging generated answers.

## Corpus and method

Six Markdown documents describe fictional Aster Analytics, Brio Insights, their products, and illustrative customer interviews. These facts were invented for the experiment and must not be presented as real market research.

At startup the service splits documents by heading and word window, attaching source path, title, section, document ID, and chunk ID. The local vector method uses hashed token and adjacent-token features with cosine similarity. It is **lexical**, not a learned semantic embedding. BM25 and reciprocal-rank hybrid search are available alongside it. No vector database or external embedding service is required.

## Retrieval evaluation

Run `uv run python -m evals.run_week2_retrieval`. The eight cases in `evals/week2_retrieval_cases.json` label relevant *documents*. Recall@2 is the fraction of relevant documents found in the first two chunks; MRR@2 uses the rank of the first relevant document.

| Method | Recall@2 | MRR@2 |
| --- | ---: | ---: |
| Token vector | 0.500 | 0.562 |
| BM25 | 0.938 | 0.812 |
| Hybrid | 0.625 | 0.625 |

BM25 is the API default because it performed best on this corpus. Hybrid fusion did not help here. The dataset is too small and too specific to generalize this result.

## Answer generation

`POST /api/v1/research/rag` retrieves up to four chunks, passes them as evidence to the LLM, and returns findings with source IDs and source metadata. After the first live run exposed citation errors, the service was changed to require an exact evidence quote from each cited chunk. It removes findings with unknown IDs or quotes absent from their cited chunks, and abstains if no valid findings remain. An exact quote improves traceability but still does not prove that the quote entails the claim. The stricter version was run on all eight live cases. After that run, the service was also changed to build summaries from retained findings, preventing an extra summary claim without a finding. That final summary rule is covered by a local test; it does not change retrieval or model generation.

The new endpoint keeps the v0.1 response fields and adds `source_ids` and `evidence_quote` on findings plus `sources` on the response.

## Initial live answer comparison — September 2026

The configured `gpt-5.6-luna` model at `api.avalai.ir` answered all eight questions through both services (16 calls total). The questions and fictional facts are in `evals/week2_answer_cases.json`. The run used the earlier ID-only citation check, before exact quote validation was added.

| Diagnostic | v0.1 baseline | RAG |
| --- | ---: | ---: |
| Required fact phrase present | 0/8 | 8/8 |
| Mean observed latency | 2.542 s | 2.401 s |

Manual review found the RAG summaries factually consistent with this synthetic corpus in all eight cases. Six of eight answers had findings that directly addressed the question and were cited to supporting sections. In the integrations case, the summary answered correctly but the sole finding discussed unsupported inputs rather than integrations. In the phone-call case, one finding cited the Features section for a restriction stated in the Limits section. This is why the service now requires exact evidence quotes.

The required-phrase score is only a diagnostic. The corpus is tiny and fictional, each service ran once per case, and the latency difference is too small to interpret. This run did not capture comparable token or cost data. Case-level output is preserved in `experiments/results/002_initial_answers.json`.

## Revised RAG run — September 2026

The same eight questions were run once more through the RAG service after exact-quote validation (eight additional calls). The required phrase was present in 8/8 answers; mean observed latency was 3.531 seconds. Manual review found that the primary finding in every answer directly addressed its question and quoted a supporting passage from the correct retrieved section (8/8). The integrations and phone-call citation failures from the first run did not recur. The answer for SignalDesk retention added an Enterprise detail in its summary without a separate finding; the later deterministic summary rule removes that extra detail.

The full revised output is preserved in `experiments/results/002_quote_answers.json`. The sample is too small to infer a latency change from 2.401 to 3.531 seconds. Exact quotes verify that cited text was supplied, but a human still must judge whether a quote supports its claim.

## Decision

The Week 2 objective is met on this small fictional corpus: retrieval supplies facts the baseline could not answer, and the revised RAG findings are traceable to supporting passages. BM25 remains the default because it beat local token-vector and hybrid retrieval on the labeled cases. Week 3 can proceed using this RAG endpoint as the document-search tool.

The local vector is lexical rather than a learned semantic embedding. Reranking was not added because the measured hybrid method already underperformed BM25, and there is no demonstrated ranking problem for these cases. Token and cost measurements remain for later production engineering. These results do not establish performance on real documents or open-ended market research.
