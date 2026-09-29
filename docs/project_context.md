# AI Business Research — Project Context

> Working context for continuing development of the project across sessions.

**Last updated:** September 28, 2026
**Current phase:** Week 3 agent implemented and evaluated; user reports successful live Tavily smoke check
**Current version:** v0.1 baseline, Week 2 RAG, and Week 3 agent endpoints

---

# 1. Project Overview

## Goal

Build a production-oriented **AI Business Research & Intelligence system**.

The system should eventually accept a business research question and combine:

* internal knowledge
* document retrieval
* web research
* structured data
* AI reasoning
* evidence/citations
* evaluation
* observability

to produce a structured research brief.

The goal is **not** to build another simple chatbot demo.

The project is intended to demonstrate practical AI software engineering:

> Build → Measure → Learn → Improve

---

# 2. Core Product Idea

A user should eventually be able to ask questions such as:

```text
What are the main competitors in this market?

How does Company X position its product?

What problems do customers commonly report?

What are the differences between these products?

What opportunities exist for a company entering this market?
```

The system should gather relevant information, reason over it, and return a structured answer with supporting evidence.

---

# 3. Current Architecture

## Week 1 — Baseline

```text
User
  │
  ▼
FastAPI
  │
  ▼
Research Service
  │
  ▼
OpenAI-compatible LLM
  │
  ▼
Pydantic Structured Response
  │
  ▼
Opik Evaluation
```

The v0.1 baseline has no retrieval. An experimental Week 2 RAG endpoint now adds local document retrieval; web search and agents are still planned. See `experiments/002_retrieval.md` for the measured retrieval results and remaining live answer comparison.

---

# 4. Current Repository Structure

```text
ai-business-research/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── observability.py
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
├── evals/
│   ├── __init__.py
│   ├── baseline_dataset.json
│   ├── create_dataset.py
│   └── run_baseline.py
│
├── experiments/
│   └── 001_baseline.md
│
├── docs/
│   ├── architecture.md
│   └── project_context.md
│
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── uv.lock
└── README.md
```

Week 2 adds `knowledge_base/`, `app/retrieval.py`, `app/services/rag.py`, `evals/run_week2_retrieval.py`, `evals/run_week2_answers.py`, and `experiments/002_retrieval.md`.

The structure can evolve as the system becomes more complex.

Do not prematurely introduce a large framework or complicated architecture.

---

# 5. Technology Stack

Current stack:

* Python 3.13
* FastAPI
* Pydantic v2
* Pydantic Settings
* OpenAI Python SDK
* OpenAI-compatible APIs
* Opik 2.2.66
* pytest
* httpx
* Ruff
* uv
* Docker

Planned technologies:

* PostgreSQL
* vector database / vector search
* BM25
* embeddings
* reranking
* web search
* agent/tool orchestration
* CI/CD
* production deployment

Potential future technologies should be introduced only when they solve a concrete problem.

---

# 6. Important Architecture Decisions

## Decision 1 — Start with a baseline

Do not begin with RAG or agents.

The first version should answer:

> What can a simple LLM system do?

Only after measuring that should additional complexity be introduced.

---

## Decision 2 — Structured output

Research responses use Pydantic models.

Current conceptual schema:

```text
ResearchResponse
├── question
├── summary
├── findings[]
│   ├── claim
│   └── explanation
└── limitations[]
```

The system should maintain a stable structured output contract as capabilities are added.

---

## Decision 3 — Evaluation is part of the product

Evaluation should not be added at the end.

Every major capability should have measurable experiments.

Examples:

```text
Baseline
    ↓
Baseline + RAG
    ↓
RAG + Hybrid Search
    ↓
RAG + Reranking
    ↓
Agent + Tools
    ↓
Production optimization
```

Each stage should be compared against the previous version.

---

## Decision 4 — Avoid unnecessary frameworks

LangChain/LangGraph are intentionally not required for Week 1.

The goal is to understand the underlying mechanics before introducing orchestration frameworks.

They may be introduced later if the agent architecture actually benefits from them.

---

# 7. Week 1 Status

## Completed

* [x] Project structure
* [x] FastAPI application
* [x] Configuration with environment variables
* [x] Pydantic schemas
* [x] OpenAI-compatible LLM integration
* [x] Structured LLM output
* [x] Docker setup
* [x] Tests
* [x] Opik integration
* [x] Opik evaluation dataset
* [x] 20 baseline evaluation cases
* [x] Answer Relevance evaluation
* [x] Baseline experiment
* [x] Experiment documentation

---

# 8. Week 1 Baseline Result

Dataset:

```text
business-research-baseline
Version: v1
Cases: 20
```

Metric:

```text
AnswerRelevance
```

Observed example/result:

```text
Answer relevance: 0.95
```

Important interpretation:

This is **not** a claim that the system is 95% accurate.

The dataset is small and consists of conceptual questions created for the experiment.

The result is primarily useful as a reference point for future versions.

See:

```text
experiments/001_baseline.md
```

for details.

---

# 9. Problems Encountered in Week 1

## Opik dataset IDs

Opik expected UUIDs for item IDs.

Human-readable IDs such as:

```text
business_001
```

were moved into a separate `case_id` field.

---

## Opik workspace mismatch

The dataset and experiment were initially being resolved in different workspaces.

This produced:

```text
409 Dataset version not found
```

even though the dataset version existed.

The issue was resolved by aligning the Opik workspace configuration.

---

## Async evaluation

The initial evaluation used:

```python
asyncio.run(service.research(question))
```

inside the evaluation task.

This caused:

```text
RuntimeError: Event loop is closed
```

The evaluation service was changed to a synchronous OpenAI client/service.

The FastAPI application can remain asynchronous independently.

---

## OpenAI/Opik serialization warning

The following warning appeared:

```text
PydanticSerializationUnexpectedValue
Expected `Message`
Expected `StreamingChoices`
```

The warning comes from serialization of OpenAI SDK response objects.

It does not currently prevent the evaluation from running.

It is considered a non-blocking issue for Week 1.

Do not spend significant time fixing this unless it affects evaluation, observability, or production behavior.

---

# 10. Current Research Service

The baseline flow is approximately:

```text
research(question)
    ↓
_generate_research(question)
    ↓
OpenAI-compatible API
    ↓
structured ResearchResponse
```

The service currently uses a synchronous OpenAI client for evaluation stability.

---

# 11. Evaluation Strategy

The evaluation should evolve with the system.

## Current

```text
Answer Relevance
```

## Planned

For RAG:

```text
Answer Relevance
Context Precision
Context Recall
Retrieval quality
```

For production engineering:

```text
Latency
Token usage
Cost
Failure rate
```

For agentic behavior:

```text
Tool selection
Tool correctness
Task completion
Evidence quality
```

The exact metrics should be chosen based on the capability being tested.

---

# 12. Week 2 — RAG

## Objective

Introduce external/domain knowledge retrieval and determine whether it improves research quality.

The key question:

> Does retrieval actually improve the answers?

Do not treat "adding RAG" as success by itself.

---

# 13. Planned Week 2 Architecture

```text
Documents
    │
    ▼
Ingestion
    │
    ▼
Chunking
    │
    ▼
Embeddings
    │
    ├──────────────┐
    ▼              ▼
Vector Search    BM25
    │              │
    └──────┬───────┘
           ▼
      Hybrid Search
           │
           ▼
        Reranking
           │
           ▼
      Retrieved Context
           │
           ▼
           LLM
           │
           ▼
Structured Research Response
           │
           ▼
        Evaluation
```

---

# 14. Week 2 Implementation Plan

### Step 1 — Define the knowledge base

Create a small realistic document collection.

Possible categories:

```text
docs/
├── company/
├── product/
├── market/
└── technical/
```

The documents should contain information that the baseline model does not reliably know.

---

### Step 2 — Build ingestion

Implement:

```text
document
   ↓
text extraction
   ↓
cleaning
   ↓
metadata
   ↓
chunks
```

Metadata should eventually include information such as:

```text
document_id
source
title
section
chunk_id
created_at
```

---

### Step 3 — Implement vector retrieval

Add embeddings and vector search.

Measure retrieval independently before connecting it to the LLM.

---

### Step 4 — Implement BM25

Vector search is not always good at exact terminology.

BM25 should help with:

* exact product names
* company names
* technical terms
* identifiers
* rare terminology

---

### Step 5 — Hybrid retrieval

Combine:

```text
Vector similarity
+
BM25
```

Then compare hybrid retrieval with each individual method.

---

### Step 6 — Reranking

Introduce reranking only after basic retrieval works.

The goal is to improve the ordering of retrieved chunks.

---

### Step 7 — Connect retrieval to the LLM

The new pipeline becomes:

```text
Question
   ↓
Retrieve relevant context
   ↓
Build prompt
   ↓
LLM
   ↓
Structured response
```

---

### Step 8 — Evaluate against v0.1

Compare:

```text
v0.1 = LLM only

v0.2 = LLM + RAG
```

Potential comparison:

```text
                    v0.1        v0.2
Answer relevance      ✓           ✓
Context precision     —           ✓
Context recall        —           ✓
Latency               ✓           ✓
Tokens                ✓           ✓
Cost                  ✓           ✓
```

---

# 15. Week 3 — Agent + Tools

Implementation: `POST /api/v1/research/agent` provides LLM-selected document,
Tavily web, and read-only SQLite metrics tools. The agent has bounded calls,
source quotes, and Opik traces. See `experiments/003_agent_tools.md` for the
six-case comparison. On September 28, 2026, the user reported that
`uv run python -m evals.smoke_tavily` succeeded locally. This is one live
search, separate from the fixed-fixture evaluation.

Planned capabilities:

```text
Research Agent
     │
     ├── Document Search
     ├── Web Search
     ├── Structured Data / SQL
     └── Summarization
```

The agent should decide which tools are useful for a research question.

Example:

```text
Question
   ↓
Agent
   ├── Internal documents?
   ├── Web research?
   ├── Structured data?
   └── Multiple sources?
```

The goal is not to make the system "agentic" for its own sake.

Agent behavior should solve a real orchestration problem.

---

# 16. Week 4 — Production Engineering

Planned work:

* observability
* evaluation improvements
* latency measurement
* token/cost tracking
* caching where useful
* error handling
* retries
* security basics
* CI/CD
* deployment
* API reliability
* documentation

Potential production architecture:

```text
Client
   ↓
API
   ↓
Research Orchestrator
   ├── Retrieval
   ├── Web Search
   ├── Structured Data
   └── LLM
          ↓
      Observability
          ↓
       Evaluation
```

---

# 17. Experiment Philosophy

Every major feature should follow:

```text
Hypothesis
    ↓
Implementation
    ↓
Evaluation
    ↓
Comparison
    ↓
Conclusion
```

Examples:

### RAG

> Does retrieval improve answer quality for knowledge-specific questions?

### Hybrid search

> Does combining BM25 and vector search retrieve better evidence than either method alone?

### Reranking

> Does reranking improve the quality of the context passed to the LLM?

### Agent

> Does tool-based orchestration improve research completeness without unacceptable latency/cost?

---

# 18. What Not to Do

Avoid adding technology just because it is popular.

Do not automatically add:

* LangChain
* LangGraph
* multiple vector databases
* multiple LLM providers
* complex memory
* Kubernetes
* microservices
* elaborate frontend
* complicated agent loops

unless the project has a concrete reason for them.

The project should demonstrate **engineering judgment**, not the number of technologies used.

---

# 19. Portfolio Goal

This project should demonstrate that I can:

* build AI-backed APIs
* integrate LLMs into software systems
* design structured outputs
* implement RAG
* build hybrid retrieval
* work with vector search
* build tool-using agents
* evaluate AI systems
* monitor AI applications
* reason about latency/cost
* debug real production-style problems
* document engineering decisions

The final project should be understandable to a technical hiring manager or client without requiring them to read the entire codebase.

---

# 20. LinkedIn Content Strategy

The project should also produce a small number of high-quality engineering posts.

The content should focus on **what was learned while building**, rather than generic AI explanations.

Potential posts:

### Post 1 — Baseline

Story:

> I wanted to build an AI research agent, but started by deliberately not building an agent.

Explain:

* baseline
* evaluation
* 20 test cases
* Opik
* measuring before adding complexity

---

### Post 2 — RAG

Story:

> The hard part of RAG isn't putting documents into a vector database.

Discuss:

* retrieval quality
* chunking
* hybrid search
* exact terminology
* evaluation

---

### Post 3 — Hybrid Search

Explain why:

```text
Vector search ≠ exact terminology
BM25 ≠ semantic similarity
```

and why combining them can be useful.

---

### Post 4 — Agent

Explain what changed when the system gained tools.

Focus on engineering tradeoffs rather than "AI agents are the future."

---

### Post 5 — Production

Discuss:

* latency
* cost
* observability
* failures
* evaluation

The posts should reflect the actual project rather than presenting hypothetical knowledge.

---

# 21. Current Priority

The immediate priority is:

```text
1. Week 2 knowledge base         ✓
2. Local ingestion and retrieval ✓
3. Retrieval comparison          ✓
4. Initial live comparison       ✓
5. Recheck exact quote citations ✓
6. Week 3 agent and Opik eval    ✓
7. Live Tavily smoke check       ✓ User-reported local run (2026-09-28)
```

Do not return to minor Week 1 issues unless they block progress.

---

# 22. Current Definition of Done

Week 1 is considered complete because:

* the baseline works
* structured responses work
* the evaluation dataset works
* Opik evaluation works
* Answer Relevance is measured
* the result is documented
* known issues are recorded

The baseline does **not** need to be perfect.

The purpose of Week 1 is to establish the reference point for the rest of the project.

---

# 23. Guiding Principle

The project should tell one coherent engineering story:

> **Start simple. Measure it. Add complexity only when there is a reason. Then measure again.**

The final system should demonstrate not only that I can build AI features, but that I can determine **whether those features actually improve a system.**
