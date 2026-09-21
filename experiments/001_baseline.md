# Experiment 001 — Baseline

**Date:** September 2026
**Status:** Completed
**Version:** v0.1
**Focus:** Establish a measurable baseline before adding RAG, tools, or agents.

---

## 1. Objective

The goal of this experiment was to build the simplest useful version of the AI Business Research system and establish a baseline for future improvements.

The baseline intentionally does **not** use:

* RAG
* Web search
* External tools
* Agents
* Long-term memory
* LangChain / LangGraph

The system receives a research question, sends it to an LLM, and returns a structured research response.

The purpose is to answer:

> How well can a simple LLM-based research system perform before adding retrieval and additional capabilities?

---

## 2. Baseline Architecture

```text
User Question
      │
      ▼
   FastAPI
      │
      ▼
Research Service
      │
      ▼
OpenAI-compatible LLM API
      │
      ▼
Structured Pydantic Response
      │
      ▼
Evaluation with Opik
```

The response is structured using Pydantic rather than returning unrestricted text.

---

## 3. Technology

* Python 3.13
* FastAPI
* Pydantic v2
* OpenAI Python SDK
* OpenAI-compatible API
* Opik 2.2.66
* pytest
* Ruff
* uv
* Docker

---

## 4. Response Schema

The baseline returns:

```text
ResearchResponse
├── question
├── summary
├── findings[]
│   ├── claim
│   └── explanation
└── limitations[]
```

This gives the later versions of the system a stable output contract.

---

## 5. Evaluation Dataset

The baseline was evaluated against a dataset containing:

* **20 test cases**
* Conceptual AI/software/business research questions
* Expected outputs for comparison

The questions were intentionally selected so that the baseline could be evaluated without requiring current web information.

Dataset:

```text
business-research-baseline
Version: v1
Items: 20
```

---

## 6. Evaluation Metric

### Answer Relevance

The first evaluation metric was Opik's `AnswerRelevance`.

The metric measures how relevant the generated answer is to the input question.

This metric should **not** be interpreted as factual accuracy.

For example, a highly relevant answer can still contain an incorrect claim.

Future experiments will add retrieval-specific and other quality metrics.

---

## 7. Result

The baseline evaluation successfully ran against the dataset.

Example observed score:

```text
Answer Relevance: 0.95
```

The evaluation completed successfully across the 20-item dataset.

> Note: The dataset is relatively small and consists of conceptual questions created for this experiment. Therefore, this score is useful primarily as a baseline for comparison with future versions of this project, not as a general measure of LLM quality.

---

## 8. Engineering Issues Encountered

Several issues appeared while building the baseline evaluation.

### 8.1 Opik dataset item IDs

Initially, human-readable IDs such as:

```text
business_001
```

were passed as dataset item IDs.

Opik expected UUIDs.

The dataset was changed to use a separate human-readable field:

```json
{
  "case_id": "business_001",
  "input": "...",
  "expected_output": "..."
}
```

while allowing Opik to generate the actual item ID.

---

### 8.2 Opik workspace mismatch

The dataset and experiment were initially associated with different Opik workspaces.

This resulted in a misleading:

```text
409 Dataset version not found
```

The dataset version itself existed and could be accessed through the lower-level API.

The actual problem was that the resources were being resolved in different workspaces.

After aligning the workspace configuration, the evaluation worked correctly.

---

### 8.3 Async evaluation and event loops

The initial implementation used:

```python
asyncio.run(service.research(question))
```

inside the Opik evaluation task.

This caused an event-loop error:

```text
RuntimeError: Event loop is closed
```

The evaluation runner was changed to use a synchronous service/client for this baseline experiment.

The FastAPI application can remain asynchronous independently.

---

### 8.4 OpenAI / Opik serialization warnings

Pydantic warnings appeared while Opik was serializing OpenAI SDK response objects:

```text
PydanticSerializationUnexpectedValue
Expected `Message`
Expected `StreamingChoices`
```

These warnings did not prevent the evaluation from completing or the Answer Relevance scores from being generated.

They are currently considered a non-blocking issue and will be investigated separately if they become relevant to later observability work.

---

## 9. What This Baseline Gives Us

The main outcome of Week 1 is not the absolute score.

It is the ability to make a controlled comparison.

The baseline provides:

```text
                    v0.1 Baseline
                    ──────────────
LLM response              ✓
Structured output         ✓
Evaluation dataset        ✓
Answer relevance          ✓
Experiment tracking       ✓
RAG                       ✗
Web research              ✗
Tools                     ✗
Agent                     ✗
```

This means future versions can be evaluated against a known starting point.

---

## 10. Limitations

The baseline has several important limitations:

1. It has no external knowledge retrieval.
2. It cannot reliably answer questions requiring current information.
3. It does not provide evidence or citations.
4. It does not use a domain-specific knowledge base.
5. Answer Relevance does not measure factual correctness.
6. The evaluation dataset contains only 20 cases.
7. No retrieval metrics are applicable yet.
8. Latency and token/cost measurements have not yet been systematically compared.

---

## 11. Hypothesis for Week 2

The next experiment will introduce Retrieval-Augmented Generation.

### Hypothesis

> Providing the model with relevant retrieved information should improve its ability to answer questions that depend on specific knowledge, while introducing additional latency and token usage.

The goal is not simply to "add RAG."

The goal is to **measure whether retrieval actually improves the system.**

---

## 12. Next Experiment

### Experiment 002 — RAG Baseline

Planned components:

```text
Documents
    ↓
Ingestion
    ↓
Chunking
    ↓
Embeddings
    ↓
Vector Retrieval
    +
BM25 Retrieval
    ↓
Hybrid Retrieval
    ↓
Reranking
    ↓
LLM
    ↓
Structured Research Response
    ↓
Evaluation
```

Metrics will be expanded to include retrieval quality where appropriate:

```text
                    v0.1        v0.2
                    LLM         + RAG
Answer relevance      ✓           ✓
Retrieval quality     —           ✓
Latency               ✓           ✓
Tokens                —           ✓
Cost                  —           ✓
```

The primary objective is to compare **v0.1 vs. v0.2**, rather than optimizing either system in isolation.

---

## 13. Engineering Principle

The main principle established during Week 1:

> **Build → Measure → Learn → Improve**

Instead of adding complexity first, establish a simple baseline, measure it, and use the results to justify the next engineering decision.
