# AI Business Research & Intelligence Agent

An AI-powered research system built with Python, FastAPI, structured LLM outputs, and Opik.

The goal of this project is to build a production-oriented AI system step by step — starting with a simple LLM baseline and gradually adding retrieval, tools, agents, evaluation, and observability.

> **Build → Measure → Learn → Improve**

This repository is being developed as a practical AI engineering project, with an emphasis on understanding the engineering decisions behind the system rather than simply connecting an LLM to an API.

---

## Project Goal

The final system will be able to answer business research questions by combining:

* Internal knowledge
* Web research
* Structured data
* AI reasoning
* Evidence and citations
* Evaluation
* Observability

For example:

```text
"What are the main AI customer support trends in 2026?"
```

The final system should be able to research the question, gather relevant evidence, analyze it, and produce a structured research brief.

---

## Current Status

### Week 1 — Foundation ✅

The first version establishes the basic AI application architecture.

Current flow:

```text
Research Question
       ↓
    FastAPI
       ↓
 Research Service
       ↓
      LLM
       ↓
Structured Pydantic Response
       ↓
    API Response
```

Implemented:

* Python 3.13+
* FastAPI
* OpenAI-compatible LLM client
* Pydantic structured outputs
* Environment-based configuration
* Basic API endpoint
* Health check
* pytest tests
* Ruff linting/formatting
* uv dependency management
* Docker support
* Initial Opik integration

---

## Tech Stack

| Technology            | Purpose                                    |
| --------------------- | ------------------------------------------ |
| Python                | Main programming language                  |
| FastAPI               | API framework                              |
| Pydantic              | Data validation and structured LLM outputs |
| OpenAI-compatible API | LLM interface                              |
| Opik                  | LLM observability and evaluation           |
| Weaviate              | BM25, vector, and hybrid document search   |
| Model2Vec             | Local embeddings through Weaviate          |
| pytest                | Testing                                    |
| Ruff                  | Linting and formatting                     |
| uv                    | Python package and environment management  |
| Docker                | Containerization                           |

---

## Project Structure

```text
ai-business-research/
├── app/                    # API, research services, evidence tools, retrieval
├── knowledge_base/         # Fictional business documents
├── data/                   # Fictional support metrics
├── evals/                  # Evaluation cases and Opik runners
├── experiments/            # Measurement reports and result JSON
├── tests/
├── docs/
├── .env.example
├── .gitignore
├── Dockerfile
├── compose.weaviate.yaml
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Week 2 — Knowledge retrieval

The project now includes a small fictional knowledge base in `knowledge_base/`.
`POST /api/v1/research/rag` retrieves evidence and returns structured findings
with source IDs. The retrieval methods are `bm25` (default), `vector`, and
`hybrid`. All three search modes run in Weaviate. The local Weaviate
`text2vec-model2vec` module generates Potion 8M embeddings for vector and
hybrid search when documents are indexed or queried.

Start local Weaviate before running retrieval or RAG:

```bash
docker compose -f compose.weaviate.yaml up -d
```

Set `WEAVIATE_URL` and `WEAVIATE_GRPC_PORT` for that instance. The default
values in `.env.example` use ports `18080` and `15051`. The Compose file
starts the local inference container with Weaviate. A remote Weaviate instance
must also have the `text2vec-model2vec` module configured to index this project.
`MODEL2VEC_IMAGE_TAG` in `.env` selects the local inference image tag and is
also used in the collection identity and experiment configuration. Changing it
creates a new collection so embeddings from different inference images do not
mix. For a remote instance, set it to the tag of that instance's inference
image.

```bash
uv run python -m evals.run_week2_retrieval
uv run pytest -q tests/test_week2_retrieval.py tests/test_week2_rag.py
```

Set `RETRIEVAL_METHOD`, `RETRIEVAL_K`, and `KNOWLEDGE_BASE_DIR` if needed.
For a live baseline versus RAG answer comparison, run:

```bash
uv run python -m evals.run_week2_answers --output week2_answer_results.json
```

The live comparison calls the configured LLM for each service and question.
See `experiments/002_retrieval.md` for the historical Week 2 measurement and
`experiments/007_weaviate_vectorizer_six_way.md` for current Weaviate results.

---

## Week 3 — Research agent

`POST /api/v1/research/agent` uses model tool calling to select internal document
search, Tavily web search, or read-only fictional support metrics in SQLite.
Each request is limited to two selection rounds, four tool calls, and two web
searches. Final findings include exact evidence quotes and source types.

Set `TAVILY_API_KEY` in `.env` for live web search. `AGENT_MODEL` optionally
selects a separate tool-capable model; otherwise the agent uses `OPENAI_MODEL`.
Without a Tavily key, web questions return a limitation. Use these commands to
check the implementation and run the Opik comparison:

```bash
uv run pytest -q
uv run python -m evals.run_week3
uv run python -m evals.smoke_tavily
```

The Opik comparison uses a fixed web fixture and live model calls. The Tavily
smoke script makes one live search and requires the API key. See
`experiments/003_agent_tools.md` for the results and limitations.

---

## Six-way Opik evidence comparison

Compare a prompt-only LLM, Weaviate BM25, Weaviate semantic vector search,
Weaviate hybrid search, then hybrid search with fixed web evidence and
read-only SQL evidence. All six configurations use one versioned, 17-case
Opik dataset and the same answer model and prompt. The extra document cases
target questions where semantic search or hybrid fusion can supply evidence
that the preceding search mode misses at top two.

```bash
./evals/run_six_way_local.sh --retrieval-only
./evals/run_six_way_local.sh
```

The script starts local Weaviate, rebuilds the app image, and uses Docker host
networking so the evaluation can reach Weaviate, the answer provider, and Opik.
It uses the configured credentials from `.env` without adding them to the image.
For an already reachable Weaviate instance with `text2vec-model2vec` enabled,
you can run `uv run python -m evals.run_six_way` directly after setting its
HTTP and gRPC connection values.

The full command makes paid answer-model calls. Weaviate generates embeddings
locally through its configured vectorizer. It writes experiment IDs, scores,
and the Opik compare link to
`experiments/results/007_weaviate_vectorizer_six_way.json`; the retrieval-only
preflight goes to `experiments/results/007_weaviate_retrieval.json`. See
`experiments/007_weaviate_vectorizer_six_way.md` for results and limits. The
earlier comparisons in `experiments/005_six_way_comparison.md` and
`experiments/006_weaviate_six_way.md` remain historical. The web stage uses a
fixed fixture, so it measures evidence access rather than live web search or
agent tool selection.

---

## Running Locally

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd ai-business-research
```

### 2. Install dependencies

This project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

### 3. Configure environment variables

Copy the example environment file:

```bash
cp .env.example .env
```

Then configure your LLM and Opik credentials.

Example:

```env
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=
OPENAI_MODEL=your-model

OPIK_API_KEY=
OPIK_WORKSPACE=
OPIK_URL_OVERRIDE=

ENVIRONMENT=development
```

Never commit the `.env` file.

---

## Run the API

```bash
uv run uvicorn app.main:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

---

## API

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "ok",
  "environment": "development"
}
```

---

### Research

```http
POST /api/v1/research
```

Request:

```json
{
  "question": "What are the main AI customer support trends in 2026?"
}
```

The system returns a structured research response containing:

* The original question
* A summary
* Research findings
* Limitations

Example structure:

```json
{
  "question": "What are the main AI customer support trends in 2026?",
  "summary": "...",
  "findings": [
    {
      "claim": "...",
      "explanation": "..."
    }
  ],
  "limitations": [
    "..."
  ]
}
```

---

## Testing

Run the test suite:

```bash
uv run pytest
```

Run with verbose output:

```bash
uv run pytest -v
```

---

## Code Quality

Run Ruff:

```bash
uv run ruff check .
```

Format the project:

```bash
uv run ruff format .
```

---

# Development Roadmap

This project will be developed incrementally.

## Week 1 — Foundation

**Goal:** Establish a reliable baseline.

```text
Question
   ↓
LLM
   ↓
Structured Response
```

Focus:

* FastAPI
* Pydantic
* LLM integration
* Configuration
* Testing
* Docker
* Opik

---

## Week 2 — Retrieval & RAG

The system will gain access to its own knowledge.

```text
Question
   ↓
Retrieval
   ↓
Relevant Documents
   ↓
LLM
   ↓
Research Brief
```

Planned work:

* Document ingestion
* Chunking
* Embeddings
* Vector search
* BM25
* Hybrid retrieval
* Metadata filtering
* Reranking
* Retrieval evaluation

---

## Week 3 — Agent & Tools

The system will become capable of deciding which tools it needs.

Potential tools:

```text
Research Agent
│
├── Search Web
├── Search Documents
├── Query Structured Data
└── Summarize / Analyze
```

The focus will be on:

* Tool calling
* Agent state
* Context management
* Tool failures
* Agent evaluation
* Failure analysis

---

## Week 4 — Production Engineering

The first Week 4 slice (September 29, 2026) improves API reliability: all
three research endpoints return stable 502/503 responses for model-provider
errors, and the baseline endpoint correctly handles its synchronous service.
See `experiments/004_api_reliability.md` for the measured before/after behavior.

Remaining work:

* API latency and token/cost tracking
* Error analysis and broader integration tests
* Bounded retries and timeouts where measurements justify them
* CI/CD, security considerations, and deployment

---

# Engineering Principles

A major goal of this project is to practice AI engineering rather than simply AI prototyping.

### 1. Build a baseline first

Before adding sophisticated components, establish something that works.

### 2. Measure before optimizing

When we introduce RAG, agents, or other components, we'll compare them against previous versions.

### 3. Treat LLMs as probabilistic components

The system should validate and evaluate model outputs rather than blindly trusting them.

### 4. Make experiments reproducible

Important experiments and decisions will be documented in the repository.

### 5. Design for failure

We will intentionally test situations where:

* Retrieval returns irrelevant documents
* The LLM produces incorrect answers
* Tools fail
* Context becomes too large
* Latency becomes unacceptable
* Costs increase

### 6. Keep the architecture understandable

New technology should solve a real problem in the system.

The goal is not to use every AI framework available.

---

# Experiments

Development decisions will be documented as experiments.

Example:

```text
experiments/
├── 001_baseline.md
├── 002_retrieval.md
├── 003_hybrid_search.md
└── ...
```

Each experiment will record:

* Goal
* Setup
* Hypothesis
* Results
* Problems
* Metrics
* Lessons learned
* Next step

---

# Why I'm Building This

There are many tutorials showing how to build an AI agent.

This project has a different goal:

> **Understand what it takes to turn an AI prototype into a measurable, reliable software system.**

The interesting part isn't just getting an LLM to answer a question.

The interesting questions are:

* How do we know the answer is good?
* How do we evaluate retrieval?
* When should an agent use a tool?
* What happens when a tool fails?
* How much does each request cost?
* Where does latency come from?
* How do we debug an incorrect answer?
* How do we know whether a new version is actually better?

Those questions will drive the development of this project.

---

# Project Status

Current implementation: v0.1 baseline, Week 2 RAG, Week 3 agent endpoints,
and the first Week 4 API reliability slice.
The agent passed a six-case fixed-fixture Opik comparison. On September 28,
2026, the user reported a successful local run of
`uv run python -m evals.smoke_tavily`. The command makes one live Tavily
search. This is evidence that the live-search path worked on that machine; it
does not measure broader web-answer quality.

---

## License

This project is currently for educational and portfolio purposes.
