# Weaviate six-way evidence comparison

Run date: 2026-10-04. [Open the six experiments in Opik](https://www.comet.com/opik/hoda/experiments/01a10627-6356-7032-830c-901dbc6453b9/compare?experiments=%5B%2201a10627-6f1f-7c90-94f8-e8b3653a2e20%22%2C%20%2201a10627-e9a8-7700-a65f-05fb86b48656%22%2C%20%2201a10628-9453-7dff-8051-cbc1e58b8759%22%2C%20%2201a10629-3dd6-79ca-b6ff-9ef45587878d%22%2C%20%2201a10629-dccb-794d-9320-184fefa7619a%22%2C%20%2201a1062b-4249-71f8-81d8-3a175ea043f8%22%5D).
Machine-readable [experiment results](results/006_weaviate_six_way.json) and [retrieval preflight](results/006_weaviate_retrieval.json) contain the case-level results.

## Setup and controls

- The same 17-case Opik dataset (`business-research-weaviate-six-way-ef9a02f4a5`, version `v1`) and answer prompt/model (`gpt-5.6-luna`) were used for all six variants. Results were read back from Opik for every item.
- The 12 knowledge-base chunks were indexed once in local Weaviate 1.34.0, collection `ResearchChunks6de1aa6019e5`. BM25 used Weaviate `bm25`, vector search used Weaviate `near_vector`, and their combination used Weaviate `hybrid` with relative-score fusion and `alpha=0.5`. All document variants requested top 2 chunks.
- Semantic vectors came from the local `sentence-transformers/all-MiniLM-L6-v2` ONNX model through FastEmbed. No corpus content was sent to an embedding provider.
- Web evidence was a fixed fixture; SQL evidence came from the checked-in support-metrics CSV through the read-only query tool. The answer model was the only paid inference service in this run.
- The three new document diagnostics were selected using retrieval-only checks before answer scoring. They test a semantic query and two questions requiring complementary lexical and semantic hits. They are targeted cases, not a random sample or a held-out test set.
- A read-only check of all six Opik experiment outputs found no findings with multiple source IDs or quotes absent from their cited source. Tightening the local citation checker to require exactly one source ID therefore does not change the recorded scores.

## Results

Strict task success requires every expected fact to be present and grounded. Document recall is averaged over the 11 cases with annotated document sources.

| Evidence available | Task success | Grounded fact coverage | Citation integrity | Document recall@2 | Mean latency | Answer tokens, prompt + completion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pure LLM | 1/17 (5.9%) | 5.9% | 100% | 0% | 1.50 s | 8,256 + 1,736 |
| Weaviate BM25 | 5/17 (29.4%) | 40.2% | 100% | 81.8% | 2.17 s | 9,972 + 3,672 |
| Weaviate vector | 6/17 (35.3%) | 52.0% | 100% | 90.9% | 2.13 s | 9,945 + 3,920 |
| Weaviate hybrid | 8/17 (47.1%) | 54.9% | 100% | 100% | 2.08 s | 9,968 + 3,936 |
| Hybrid + web | 11/17 (64.7%) | 74.5% | 100% | 100% | 5.08 s | 10,203 + 4,244 |
| Hybrid + web + SQL | 16/17 (94.1%) | 97.1% | 100% | 100% | 2.20 s | 10,521 + 4,452 |

Each step increased strict task success on this dataset without losing a previously successful case:

- BM25 answered four internal-document cases that the pure LLM could not.
- Vector search added `vector_maildock_import`, where BM25 missed the relevant document at top 2.
- Hybrid added `hybrid_price_coaching` and `hybrid_price_queue`; each single search method found only one of two required documents at top 2.
- Fixed web evidence added `doc_web_trends`, `web_aurora_date`, and `web_aurora_product`.
- SQL evidence added `all_three_signaldesk_aurora`, `doc_sql_signaldesk`, `sql_echomap_august_response`, `sql_signaldesk_july_tickets`, and `web_sql_aurora_signaldesk`.

The final variant still failed `doc_sql_echomap`: it covered one of two required facts. A later review of the stored Opik evidence found that retrieval returned the annotated company document's **Positioning** chunk, not its pricing chunk. Document-level recall counted the source path but overstated whether the required fact was available. The full JSON records the per-case scores and Opik trace IDs.

A read-only audit of the stored Opik evidence found full **fact-level** document evidence in 7/11 BM25 cases, 9/11 vector cases, and 10/11 hybrid cases. This is more precise than the source-path recall shown in the original table.

## Limits

The cases use a small fictional corpus and deliberately probe each added source's potential value. The rising scores demonstrate that these configurations *can* answer more of these questions; they do not estimate production accuracy. The web stage does not test live Tavily reliability, and this runner gives each variant its evidence directly rather than evaluating the agent's tool selection. The heuristic fact and citation metrics inspect model output against known facts and exact evidence quotes; they are not a human quality review. Citation integrity measures citations that were emitted, so an empty answer passes that metric while failing fact coverage. A single run does not measure output variance. Token totals are model usage, not monetary cost. The higher hybrid-plus-web mean latency includes service variability and should not be read as a controlled latency effect of adding web evidence.

The configured Weaviate Cloud endpoint returned HTTP 503 during setup, so this run used local Weaviate. The first evaluation container had no route to Opik; Docker host networking reached both Opik and the local Weaviate ports. The earlier [14-case comparison](005_six_way_comparison.md) remains historical and is not directly comparable to these 17-case scores because the retrieval engine, embeddings, prompt, and cases changed.

Validation: `uv run pytest -q` passed 46 tests, `uv run ruff check .` passed, and a local RAG endpoint smoke check returned HTTP 200 with a Weaviate-retrieved, cited `company/aster.md` chunk while using a fake answer model.
