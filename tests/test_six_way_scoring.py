import json
from pathlib import Path

from app.agent_tools import SupportMetrics
from evals.six_way_scoring import score_case


def test_scores_each_required_fact_against_matching_exact_quote():
    case = {
        "expected_facts": [
            {
                "phrase": "$480",
                "source_type": "document",
                "locator": "company/aster.md",
            },
            {
                "phrase": "128",
                "source_type": "sql",
                "locator": "sql:support_metrics:sd-2026-07",
            },
        ]
    }
    result = {
        "response": {
            "summary": "SignalDesk Team costs $480 and had 128 tickets.",
            "findings": [
                {
                    "claim": "SignalDesk Team costs $480 per month.",
                    "source_ids": ["E1"],
                    "evidence_quote": "SignalDesk Team costs $480 per month",
                },
                {
                    "claim": "SignalDesk had 128 tickets in July.",
                    "source_ids": ["E2"],
                    "evidence_quote": "SignalDesk had 128 support tickets",
                },
            ],
            "limitations": [],
        },
        "evidence": [
            {
                "source_id": "E1",
                "source_type": "document",
                "path": "company/aster.md",
                "text": "SignalDesk Team costs $480 per month for 20 agents.",
            },
            {
                "source_id": "E2",
                "source_type": "sql",
                "path": "sql:support_metrics:sd-2026-07",
                "text": "In July, SignalDesk had 128 support tickets.",
            },
        ],
    }

    assert score_case(case, result) == {
        "fact_coverage": 1.0,
        "grounded_fact_coverage": 1.0,
        "citation_integrity": 1.0,
        "task_success": 1.0,
    }

    result["response"]["findings"][1]["evidence_quote"] = "fabricated quote"
    assert score_case(case, result) == {
        "fact_coverage": 1.0,
        "grounded_fact_coverage": 0.5,
        "citation_integrity": 0.0,
        "task_success": 0.0,
    }

    result["response"]["findings"][1]["evidence_quote"] = (
        "In July, SignalDesk had 128 support tickets."
    )
    result["response"]["findings"][1]["source_ids"] = ["E1", "E2"]
    assert score_case(case, result)["citation_integrity"] == 0.0


def test_absence_case_requires_a_clear_limitation_and_no_findings():
    case = {"expected_facts": []}
    result = {
        "response": {
            "summary": "No evidence available.",
            "findings": [],
            "limitations": ["The exact price is not established."],
        },
        "evidence": [],
    }
    assert score_case(case, result)["task_success"] == 1.0

    result["response"]["findings"] = [
        {"claim": "The price is $900.", "source_ids": [], "evidence_quote": None}
    ]
    assert score_case(case, result)["task_success"] == 0.0


def test_unrelated_exact_quote_does_not_ground_a_claimed_fact():
    case = {
        "expected_facts": [
            {
                "phrase": "$480",
                "source_type": "document",
                "locator": "company/aster.md",
            }
        ]
    }
    result = {
        "response": {
            "summary": "SignalDesk costs $480.",
            "findings": [
                {
                    "claim": "SignalDesk costs $480.",
                    "source_ids": ["E1"],
                    "evidence_quote": "SignalDesk includes analytics dashboards.",
                }
            ],
            "limitations": [],
        },
        "evidence": [
            {
                "source_id": "E1",
                "source_type": "document",
                "path": "company/aster.md",
                "text": "SignalDesk includes analytics dashboards. The price is $480.",
            }
        ],
    }

    assert score_case(case, result) == {
        "fact_coverage": 1.0,
        "grounded_fact_coverage": 0.0,
        "citation_integrity": 1.0,
        "task_success": 0.0,
    }


def test_dataset_facts_exist_in_their_declared_sources():
    root = Path(__file__).resolve().parents[1]
    cases = json.loads((root / "evals/six_way_cases.json").read_text())
    web = json.loads((root / "evals/six_way_web_fixture.json").read_text())
    assert len(cases) == len({case["case_id"] for case in cases}) == 17

    for case in cases:
        for fact in case["expected_facts"]:
            if fact["source_type"] == "document":
                text = (root / "knowledge_base" / fact["locator"]).read_text()
            elif fact["source_type"] == "web":
                source = web[case["web_fixture"]]
                assert source["path"] == fact["locator"]
                text = source["text"]
            else:
                source = SupportMetrics().query(**case["sql_args"])
                text = next(
                    item.text for item in source if item.path == fact["locator"]
                )
            assert fact["phrase"].casefold() in text.casefold(), case["case_id"]
