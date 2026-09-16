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

* Python 3.12
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
| pytest                | Testing                                    |
| Ruff                  | Linting and formatting                     |
| uv                    | Python package and environment management  |
| Docker                | Containerization                           |

---

## Project Structure

```text
ai-business-research/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── research.py
│   │
│   └── services/
│       ├── __init__.py
│       └── research.py
│
├── tests/
│   ├── test_health.py
│   └── test_research.py
│
├── experiments/
│   └── 001_baseline.md
│
├── docs/
│   └── architecture.md
│
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── uv.lock
└── README.md
```

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

The final stage will focus on making the system measurable and production-oriented.

Planned work:

* Opik tracing
* Evaluation datasets
* Automated evaluations
* Latency measurement
* Token/cost tracking
* Error analysis
* Integration tests
* CI/CD
* Security considerations
* Production deployment

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

Current version:

**v0.1 — LLM Baseline**

Next milestone:

**v0.2 — Retrieval & RAG**

---

## License

This project is currently for educational and portfolio purposes.
