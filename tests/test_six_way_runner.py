from types import SimpleNamespace

import pytest

from app.agent_tools import SupportMetrics
from app.retrieval import Chunk, Hit
from app.schemas.research import ResearchResponse
from app.services.evidence import Evidence
from evals.six_way_runner import (
    VARIANTS,
    aggregate_rows,
    experiment_rows,
    retrieval_diagnostics,
    run_case,
    validate_dataset,
)


def test_each_variant_receives_only_its_enabled_evidence():
    chunk = Chunk(
        "doc-1",
        "doc",
        "company/aster.md",
        "Aster Analytics",
        "Pricing",
        "SignalDesk Team costs $480 per month.",
    )
    calls = []

    def search(query, *, k, method):
        calls.append((query, k, method))
        return [Hit(chunk, 1.0)]

    retriever = SimpleNamespace(search=search)
    web = {
        "aurora_pulseboard": Evidence(
            "web-1",
            "Aurora",
            "Announcement",
            "https://example.com/aurora",
            "Aurora announced PulseBoard.",
            "web",
        )
    }
    case = {
        "input": "Compare SignalDesk, Aurora, and July ticket counts.",
        "document_query": "SignalDesk Team price",
        "web_fixture": "aurora_pulseboard",
        "sql_args": {"product": "SignalDesk", "month": "2026-07"},
        "relevant_sources": ["company/aster.md"],
    }

    def answer(question, evidence):
        return (
            ResearchResponse(
                question=question,
                summary="No answer.",
                findings=[],
                limitations=["No answer."],
            ),
            {"prompt_tokens": 10, "completion_tokens": 2},
        )

    expected = {
        "pure_llm": set(),
        "bm25": {"document"},
        "semantic_vector": {"document"},
        "hybrid": {"document"},
        "hybrid_web": {"document", "web"},
        "hybrid_web_sql": {"document", "web", "sql"},
    }
    for name in VARIANTS:
        result = run_case(
            case,
            name,
            retriever=retriever,
            web=web,
            sql=SupportMetrics(),
            answer=answer,
            k=2,
        )
        assert {item["source_type"] for item in result["evidence"]} == expected[name]
        assert result["response"]["question"] == case["input"]
        assert result["document_recall"] == (0.0 if name == "pure_llm" else 1.0)
    assert [method for _, _, method in calls] == [
        "bm25",
        "vector",
        "hybrid",
        "hybrid",
        "hybrid",
    ]


def test_retrieval_diagnostics_exposes_complementary_document_evidence():
    chunks = [
        Chunk("a", "a", "a.md", "A", "A", "First fact"),
        Chunk("b", "b", "b.md", "B", "B", "Second fact"),
    ]

    def search(query, *, k, method):
        selected = {
            "bm25": [chunks[0]],
            "vector": [chunks[1]],
            "hybrid": chunks,
        }[method]
        return [Hit(chunk, 1.0) for chunk in selected]

    cases = [
        {
            "case_id": "both",
            "document_query": "two facts",
            "relevant_sources": ["a.md", "b.md"],
            "expected_facts": [
                {"phrase": "First fact", "source_type": "document", "locator": "a.md"},
                {"phrase": "Second fact", "source_type": "document", "locator": "b.md"},
            ],
        }
    ]
    summary = retrieval_diagnostics(
        cases, SimpleNamespace(search=search), k=2
    )
    assert summary["bm25"]["full_evidence_cases"] == 0
    assert summary["bm25"]["mean_document_fact_recall"] == 0.5
    assert summary["vector"]["mean_document_recall"] == 0.5
    assert summary["hybrid"]["full_evidence_cases"] == 1
    assert summary["hybrid"]["full_fact_evidence_cases"] == 1


def test_document_recall_does_not_count_an_unrelated_chunk_as_fact_evidence():
    wrong_chunk = Chunk(
        "a-positioning", "a", "a.md", "A", "Positioning", "We sell support tools."
    )
    case = {
        "case_id": "price",
        "document_query": "price",
        "relevant_sources": ["a.md"],
        "expected_facts": [
            {"phrase": "$480", "source_type": "document", "locator": "a.md"}
        ],
    }
    retriever = SimpleNamespace(
        search=lambda query, *, k, method: [Hit(wrong_chunk, 1.0)]
    )
    summary = retrieval_diagnostics([case], retriever, k=2)
    assert summary["hybrid"]["mean_document_recall"] == 1.0
    assert summary["hybrid"]["mean_document_fact_recall"] == 0.0


def test_existing_dataset_must_match_cases_before_reuse():
    expected = [{"case_id": "one", "input": "Question one"}]
    assert validate_dataset([], expected) is False
    assert validate_dataset([{"id": "remote-id", **expected[0]}], expected) is True
    with pytest.raises(ValueError, match="different contents"):
        validate_dataset(
            [{"id": "remote-id", "case_id": "one", "input": "Changed question"}],
            expected,
        )


def test_aggregate_rows_keeps_categories_and_token_totals_separate():
    rows = [
        {
            "case_id": "doc",
            "category": "document",
            "scores": {"task_success": 1.0},
            "latency_seconds": 2.0,
            "prompt_tokens": 100,
            "completion_tokens": 20,
        },
        {
            "case_id": "web",
            "category": "web",
            "scores": {"task_success": 0.0},
            "latency_seconds": 4.0,
            "prompt_tokens": 200,
            "completion_tokens": 30,
        },
    ]

    summary = aggregate_rows(rows)

    assert summary["overall"]["task_success"] == 0.5
    assert summary["by_category"]["document"]["task_success"] == 1.0
    assert summary["by_category"]["web"]["task_success"] == 0.0
    assert summary["overall"]["mean_latency_seconds"] == 3.0
    assert summary["overall"]["prompt_tokens"] == 300


def test_opik_readback_rejects_a_partial_experiment():
    item = SimpleNamespace(
        dataset_item_data={"case_id": "doc", "category": "document"},
        evaluation_task_output={
            "output": {
                "latency_seconds": 2.0,
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "document_recall": 1.0,
                "response": {"summary": "Answer", "limitations": []},
            }
        },
        feedback_scores=[{"name": "task_success", "value": 1.0, "reason": "grounded"}],
        trace_id="trace-1",
    )
    rows = experiment_rows([item], {"doc"}, {"task_success"})
    assert rows[0]["case_id"] == "doc"
    assert rows[0]["scores"]["task_success"] == 1.0

    with pytest.raises(ValueError, match="incomplete"):
        experiment_rows([item], {"doc", "web"}, {"task_success"})
