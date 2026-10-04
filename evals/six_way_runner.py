"""Controlled evidence assembly shared by all six evaluation variants."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace

from app.agent_tools import SupportMetrics
from app.retrieval import Retriever
from app.schemas.research import ResearchResponse
from app.services.evidence import Evidence

VARIANTS = {
    "pure_llm": {"retrieval": None, "web": False, "sql": False},
    "bm25": {"retrieval": "bm25", "web": False, "sql": False},
    "semantic_vector": {"retrieval": "vector", "web": False, "sql": False},
    "hybrid": {"retrieval": "hybrid", "web": False, "sql": False},
    "hybrid_web": {"retrieval": "hybrid", "web": True, "sql": False},
    "hybrid_web_sql": {"retrieval": "hybrid", "web": True, "sql": True},
}

Answer = Callable[[str, list[Evidence]], tuple[ResearchResponse, dict[str, int | None]]]


def retrieval_diagnostics(cases: list[dict], retriever: Retriever, *, k: int) -> dict:
    """Measure document-path recall and support for required document facts."""
    document_cases = [case for case in cases if case["relevant_sources"]]
    results = {}
    for method in ("bm25", "vector", "hybrid"):
        rows = []
        for case in document_cases:
            hits = retriever.search(case["document_query"], k=k, method=method)
            sources = [hit.chunk.source for hit in hits]
            relevant = set(case["relevant_sources"])
            recall = len(relevant.intersection(sources)) / len(relevant)
            document_facts = [
                fact
                for fact in case["expected_facts"]
                if fact["source_type"] == "document"
            ]
            fact_recall = (
                sum(
                    any(
                        hit.chunk.source == fact["locator"]
                        and fact["phrase"].casefold() in hit.chunk.text.casefold()
                        for hit in hits
                    )
                    for fact in document_facts
                )
                / len(document_facts)
                if document_facts
                else None
            )
            rows.append(
                {
                    "case_id": case["case_id"],
                    "retrieved_sources": sources,
                    "retrieved_chunks": [
                        {
                            "chunk_id": hit.chunk.chunk_id,
                            "source": hit.chunk.source,
                            "section": hit.chunk.section,
                        }
                        for hit in hits
                    ],
                    "document_recall": recall,
                    "full_evidence": recall == 1.0,
                    "document_fact_recall": fact_recall,
                    "full_fact_evidence": fact_recall == 1.0,
                }
            )
        fact_recalls = [
            row["document_fact_recall"]
            for row in rows
            if row["document_fact_recall"] is not None
        ]
        results[method] = {
            "count": len(rows),
            "mean_document_recall": round(
                sum(row["document_recall"] for row in rows) / len(rows), 3
            ),
            "full_evidence_cases": sum(row["full_evidence"] for row in rows),
            "mean_document_fact_recall": (
                round(sum(fact_recalls) / len(fact_recalls), 3)
                if fact_recalls
                else None
            ),
            "full_fact_evidence_cases": sum(
                row["full_fact_evidence"] for row in rows
            ),
            "cases": rows,
        }
    return results


def validate_dataset(existing: list[dict], expected: list[dict]) -> bool:
    """Return whether a dataset exists, rejecting contents changed under its name."""
    if not existing:
        return False
    fields = set(expected[0])
    actual = {
        item["case_id"]: {key: item.get(key) for key in fields} for item in existing
    }
    wanted = {item["case_id"]: item for item in expected}
    if actual != wanted or len(existing) != len(expected):
        raise ValueError("Opik dataset has different contents than the local cases")
    return True


def aggregate_rows(rows: list[dict]) -> dict:
    """Average scores by category while keeping token totals explicit."""
    if not rows:
        raise ValueError("Cannot summarize an experiment with no cases")

    def summarize(group: list[dict]) -> dict:
        scores = {
            name: round(sum(row["scores"][name] for row in group) / len(group), 3)
            for name in group[0]["scores"]
        }
        scores["count"] = len(group)
        scores["mean_latency_seconds"] = round(
            sum(row["latency_seconds"] for row in group) / len(group), 3
        )
        for token_type in ("prompt_tokens", "completion_tokens"):
            known = [row[token_type] for row in group if row[token_type] is not None]
            scores[token_type] = sum(known) if len(known) == len(group) else None
        recalls = [
            row["document_recall"]
            for row in group
            if row.get("document_recall") is not None
        ]
        scores["mean_document_recall"] = (
            round(sum(recalls) / len(recalls), 3) if recalls else None
        )
        return scores

    categories = sorted({row["category"] for row in rows})
    return {
        "overall": summarize(rows),
        "by_category": {
            category: summarize([row for row in rows if row["category"] == category])
            for category in categories
        },
    }


def experiment_rows(
    items: list, expected_case_ids: set[str], expected_scores: set[str]
) -> list[dict]:
    """Read complete item-level outputs and feedback from an Opik experiment."""
    rows = []
    for item in items:
        case = item.dataset_item_data
        output = item.evaluation_task_output["output"]
        scores = {score["name"]: score["value"] for score in item.feedback_scores}
        if set(scores) != expected_scores:
            raise ValueError("Opik experiment has incomplete feedback scores")
        rows.append(
            {
                "case_id": case["case_id"],
                "category": case["category"],
                "scores": scores,
                "latency_seconds": output["latency_seconds"],
                "prompt_tokens": output["prompt_tokens"],
                "completion_tokens": output["completion_tokens"],
                "document_recall": output.get("document_recall"),
                "summary": output["response"]["summary"],
                "limitations": output["response"]["limitations"],
                "trace_id": item.trace_id,
            }
        )
    case_ids = {row["case_id"] for row in rows}
    if case_ids != expected_case_ids or len(rows) != len(expected_case_ids):
        raise ValueError("Opik experiment is incomplete or has duplicate cases")
    return sorted(rows, key=lambda row: row["case_id"])


def run_case(
    case: dict,
    variant: str,
    *,
    retriever: Retriever,
    web: dict[str, Evidence],
    sql: SupportMetrics,
    answer: Answer,
    k: int = 4,
) -> dict:
    """Run one question with only the evidence available to its variant."""
    configuration = VARIANTS[variant]
    start = time.perf_counter()
    evidence: list[Evidence] = []
    retrieved_paths: list[str] = []
    query = case["document_query"]
    method = configuration["retrieval"]
    hits = retriever.search(query, k=k, method=method) if method else []
    for hit in hits:
        chunk = hit.chunk
        retrieved_paths.append(chunk.source)
        evidence.append(
            Evidence(
                source_id=chunk.chunk_id,
                title=chunk.title,
                section=chunk.section,
                path=chunk.source,
                text=chunk.text,
            )
        )
    relevant = set(case["relevant_sources"])
    recall = (
        len(relevant.intersection(retrieved_paths)) / len(relevant)
        if relevant
        else None
    )
    if configuration["web"] and case["web_fixture"]:
        evidence.append(web[case["web_fixture"]])
    if configuration["sql"] and case["sql_args"]:
        evidence.extend(sql.query(**case["sql_args"]))
    evidence = [
        replace(item, source_id=f"E{index}")
        for index, item in enumerate(evidence, start=1)
    ]
    response, usage = answer(case["input"], evidence)
    return {
        "response": response.model_dump(mode="json"),
        "evidence": [item.__dict__ for item in evidence],
        "retrieved_paths": retrieved_paths,
        "latency_seconds": round(time.perf_counter() - start, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "document_recall": recall,
    }
