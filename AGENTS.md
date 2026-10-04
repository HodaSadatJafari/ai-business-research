# Project handoff

Last updated: 2026-10-04.

## Purpose and current state

Build an evaluated business research system in small, measured phases. Week 1
baseline, Week 2 local retrieval and RAG, and Week 3 tool-using agent are
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

## Next task

Continue Week 4 with another small, measured production-engineering slice.
The roadmap in `docs/project_context.md` prioritizes latency and token/cost
tracking, error analysis, CI/CD, security, and deployment. Compare each change
against the prior behavior and record limits in `experiments/`.

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
