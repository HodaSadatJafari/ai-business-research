# Experiment 003 — Tool-using research agent

**Status:** Implemented and evaluated on fixed fictional cases. User reported a successful local live Tavily smoke check on September 28, 2026.

## Hypothesis

An LLM-directed agent can answer questions that require internal documents, current web evidence, or structured support metrics more completely than the Week 2 document-only RAG service, while staying within a bounded tool budget.

## Implementation

`POST /api/v1/research/agent` exposes `document_search`, `web_search`, and `query_support_metrics` to the configured OpenAI-compatible model. The agent allows two selection rounds, four total tool calls, two web searches, and one final structured answer call. SQL uses fixed parameterized queries over a fictional CSV fixture loaded into SQLite. Web search uses Tavily's authenticated Search API; deterministic evaluations use a fixed web fixture. All final findings need a source ID and an exact quote from that evidence. The summary is built from retained findings. Opik records a `general` agent span, `tool` spans, and OpenAI integration `llm` spans.

The configured `gpt-5.6-luna` model successfully called the SQL tool and returned a grounded answer. [Verified Opik trace](https://www.comet.com/opik/api/v1/session/redirect/projects/?trace_id=01a0e203-1fba-7e8f-aa0f-e767b924fb71&path=aHR0cHM6Ly93d3cuY29tZXQuY29tL29waWsvYXBpLw==).

## Evaluation

Six cases cover document, SQL, web, combined, insufficient-evidence, and tool-failure behavior. The web responses are fixed so both services see repeatable questions; live web search is a separate smoke check. Case content is versioned by a hash in the Opik dataset name. Mechanical scores check tool choice, required fact or abstention, citations, and the four-call limit. These checks do not measure general factual accuracy or quote entailment.

| Metric | Week 2 RAG | Week 3 agent |
| --- | ---: | ---: |
| Tool selection | 1/6 | 6/6 |
| Fact or expected limitation | 1/6 | 6/6 |
| Citation and budget check | 6/6 | 6/6 |
| Mean observed latency | 3.355 s | 5.259 s |

The tool-selection score is descriptive for RAG, which has no tool chooser. The agent answered document, SQL, web, and combined questions, and returned limitations on insufficient-evidence and simulated web-outage cases. Case-level scores, tool choices, and latency are in `experiments/results/003_comparison.json`. [Compare the two Opik experiments](https://www.comet.com/opik/hoda/experiments/01a0e208-e834-7611-b94b-ad7360ec0e92/compare?experiments=%5B%2201a0e208-f6d3-75c3-b94f-c54db40bd9ed%22%2C%2201a0e209-52a4-7112-8d47-45074784d8c3%22%5D).

## Limits and decision

The dataset is small and fictional. Scores are from one run per case with a fixed web fixture; they are not claims about live market research. The agent made more model calls and took longer on average, but this sample is too small for a stable latency estimate. Opik captured tokens, but no cost estimate for the configured model. On September 28, 2026, the user reported a successful local run of `uv run python -m evals.smoke_tavily`, which makes one live Tavily search and requires usable results. The result output was not shared, and this single request does not measure live answer quality. Attempts from the assistant environment returned HTTP 403 and later a DNS resolution error, so success is based on the user's local report rather than assistant-observed output.
