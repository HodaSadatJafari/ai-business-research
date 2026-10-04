# Six-way comparison with Weaviate-managed embeddings

Run date: 2026-10-04. [Compare all six experiments in Opik](https://www.comet.com/opik/hoda/experiments/01a10627-6356-7032-830c-901dbc6453b9/compare?experiments=%5B%2201a10682-295e-7fc2-951c-04b378c32ed1%22%2C%20%2201a10683-2c2e-70a4-8425-9267721b4568%22%2C%20%2201a10683-f45d-733d-9dd8-c31395077cde%22%2C%20%2201a10684-ba29-7a85-945e-c395dc3cacc6%22%2C%20%2201a10685-f27a-7556-a968-fab21b38d474%22%2C%20%2201a10686-bb4e-7cf6-855a-fce527dba5fa%22%5D). The [Opik readback JSON](results/007_weaviate_vectorizer_six_way.json) contains case-level scores and trace IDs; the [retrieval audit](results/007_weaviate_retrieval.json) contains the retrieved chunk IDs and fact-level evidence checks.

[Compare the two final variants directly in Opik](https://www.comet.com/opik/hoda/experiments/01a10627-6356-7032-830c-901dbc6453b9/compare?experiments=%5B%2201a1062b-4249-71f8-81d8-3a175ea043f8%22%2C%20%2201a10686-bb4e-7cf6-855a-fce527dba5fa%22%5D).

## Setup

The same 17-case dataset (`business-research-weaviate-six-way-ef9a02f4a5`, version `v1`), `gpt-5.6-luna` answer model, answer prompt (hash `2ba9bf59d89e`), scoring, fixed web fixture, SQL data, top-2 document limit, and hybrid alpha (`0.5`) were used in the [earlier run](006_weaviate_six_way.md) and this run. All six new experiment rows were read back from Opik.

The change is the embedding generator **and model**: local Weaviate 1.34.0 now uses its [`text2vec-model2vec` module](https://docs.weaviate.io/weaviate/model-providers/model2vec/embeddings) with the local `minishlab/potion-base-8M` inference container. Weaviate generated embeddings during import and vectorized text queries for `near_text` and `hybrid`. BM25, vector, and hybrid search all ran in Weaviate collection `ResearchChunks1d2b30f77e28`. The app no longer depends on FastEmbed or supplies vectors.

## Did it reproduce the earlier result?

**The progression remained, but the scores were not identical.** Strict task success requires every required fact to have a valid quote from its designated evidence source.

| Evidence available | Earlier FastEmbed vectors | Weaviate Model2Vec vectors | Grounded fact coverage, new |
| --- | ---: | ---: | ---: |
| Pure LLM | 1/17 | 1/17 | 5.9% |
| Weaviate BM25 | 5/17 | 5/17 | 43.1% |
| Weaviate vector | 6/17 | 6/17 | 52.0% |
| Weaviate hybrid | 8/17 | 7/17 | 52.0% |
| Hybrid + fixed web evidence | 11/17 | 10/17 | 71.6% |
| Hybrid + fixed web evidence + SQL | 16/17 | 15/17 | 94.1% |

Each added evidence source improved strict success within the **new** run without losing a previously successful case: BM25 added four internal-document answers; vector added `vector_maildock_import`; hybrid added `hybrid_price_coaching`; fixed web evidence added three web cases; SQL added five structured-data cases. The final configuration missed `hybrid_price_queue` and `doc_sql_echomap`.

The only case whose strict success changed between the two runs was `hybrid_price_queue`. The earlier hybrid result retrieved Aster Analytics' **Pricing** chunk and EchoMap's **Features** chunk. The Weaviate-managed Model2Vec run retrieved EchoMap's **Features** chunk and Aster's **Positioning** chunk. Both documents were counted in source-path recall, but the latter chunk did not contain the required `$480` price. This ranking difference explains the lost fact in hybrid, hybrid + web, and hybrid + web + SQL. `doc_sql_echomap` missed EchoMap's pricing chunk in both runs.

## Retrieval audit

The table separates finding the right *document* from finding a chunk containing the required *fact*. Counts use the 11 cases with document facts.

| Weaviate mode | Full document evidence, earlier → new | Full fact evidence, earlier → new | New mean fact recall |
| --- | ---: | ---: | ---: |
| BM25 | 8/11 → 8/11 | 7/11 → 7/11 | 72.7% |
| Vector | 9/11 → 10/11 | 9/11 → 9/11 | 90.9% |
| Hybrid | 11/11 → 11/11 | 10/11 → 9/11 | 86.4% |

The earlier fact-evidence counts were calculated by reading the stored Opik item outputs without making new model calls. The new counts come from the retrieval-only audit. Hybrid found both annotated document paths for every case but missed the needed chunk in two of them. Source-path recall alone is therefore too generous for this corpus.

## Limits and verification

These are targeted questions over a small fictional corpus. They show that the stages *can* add useful evidence, not expected production accuracy. The fixed web fixture does not test live Tavily results or agent tool selection. The answer model was run once per case and variant; there is no variance estimate or seed-controlled repeat, and latency differences mix retrieval and provider variability. Citation integrity was 100% in each run, but an empty answer passes that metric; strict task success and grounded fact coverage carry the answer-quality signal. Tokens are recorded in the JSON, without a monetary cost estimate.

After review, the grounded-fact scorer was tightened to require the expected
phrase inside the exact cited quote. All 204 saved Opik item outputs in the 006
and 007 experiments were read back and rescored without new model calls; no
item score changed. Future local runs take the inference image tag from
`MODEL2VEC_IMAGE_TAG`, which is shared with the local Compose service.

`uv run pytest -q` passed 49 tests and `uv run ruff check .` passed after moving embedding generation into Weaviate and addressing review findings. `./evals/run_six_way_local.sh --retrieval-only` completed against the live collection. The rebuilt application image was checked and contains no FastEmbed package. The previous 006 report remains a historical measurement with the FastEmbed path.
