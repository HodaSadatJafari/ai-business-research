# Project handoff

Last updated: 2026-10-13.

## Purpose and current state

Build an evaluated business research system in small, measured phases. Week 1
baseline, Week 2 RAG, and Week 3 tool-using agent are
implemented. Week 3 was committed as `a817e9b`. Its fixed-fixture Opik
comparison is in `experiments/results/003_comparison.json`. On 2026-09-28,
the user reported a successful local `uv run python -m evals.smoke_tavily`
run. The assistant environment could not independently reproduce it because
requests returned HTTP 403 and later failed DNS resolution. One live search
does not establish broader web-answer quality.

The first Week 4 slice fixes the baseline API's synchronous route and maps
model-provider failures to stable, safe 502/503 responses across the three
research endpoints. Before/after local measurements are in
`experiments/004_api_reliability.md`.

All active document BM25, vector, and hybrid search now runs in Weaviate.
The local Compose setup uses Weaviate's Model2Vec vectorizer for embeddings.
The 17-case six-way Opik comparison and its limits are in
`experiments/007_weaviate_vectorizer_six_way.md`. Its earlier FastEmbed run
remains historical in `experiments/006_weaviate_six_way.md`. Use
`./evals/run_six_way_local.sh --retrieval-only` for a no-LLM retrieval check;
omit the flag for the full paid Opik evaluation.

An experimental local Weaviate reranker now reorders hybrid candidates before
selecting two answer chunks. The paired Opik results and retrieval limits are
in `experiments/008_weaviate_reranking.md`. Eight candidates improved
fact-level retrieval but held strict answer success at 15/17; an exploratory
three-candidate run reached 13/17 because citation errors offset the retrieved
facts. Keep ordinary hybrid as the default. Use
`./evals/run_reranker_local.sh --retrieval-only` for a no-LLM audit; omit the
flag for the 34-call paired Opik evaluation.

## Next task

Continue Week 4 with a small, measured citation-validation and error-analysis
slice. The reranker supplied facts that the answer model sometimes cited with
wrong source IDs or incomplete claim wording. Compare any fix against the
15/17 hybrid baseline on held-out cases. The roadmap in
`docs/project_context.md` also prioritizes latency and token/cost tracking,
CI/CD, security, and deployment. Record limits in `experiments/`.

## Working conventions

- Use `uv` for Python commands. The project requires Python 3.13 or newer.
- Run `uv run pytest -q` and `uv run ruff check .` after code changes.
- Keep live-service checks separate from deterministic tests and avoid
  unnecessary calls that consume API quota.
- Keep FastAPI responses structured and support agent findings with source
  IDs and exact evidence quotes.
- Keep model tool use bounded: two selection rounds, four tool calls total,
  and two web searches.
- Never display or commit `.env` or credentials.
- Avoid adding frameworks without a measured need.
