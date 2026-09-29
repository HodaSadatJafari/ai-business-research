import json
from pathlib import Path

from evals.run_week3 import (
    CitationAndBudget,
    FactAndOutcome,
    FixedWebSearch,
    ToolSelection,
)


def test_eval_cases_cover_each_tool_and_failures():
    path = Path(__file__).resolve().parents[1] / "evals" / "week3_cases.json"
    cases = json.loads(path.read_text(encoding="utf-8"))
    assert {case["category"] for case in cases} == {
        "document",
        "sql",
        "web",
        "combined",
        "insufficient",
        "tool_failure",
    }


def test_fixed_web_fixture_and_code_metrics():
    fixture = FixedWebSearch().search("Aurora Support launch")
    assert fixture[0].source_type == "web"
    assert "PulseBoard" in fixture[0].text
    output = {
        "response": {
            "summary": "Aurora Support announced PulseBoard.",
            "findings": [
                {
                    "source_ids": [fixture[0].source_id],
                    "evidence_quote": fixture[0].text,
                }
            ],
            "sources": [{"source_id": fixture[0].source_id}],
            "tools_used": ["web_search"],
            "tool_call_count": 1,
            "limitations": [],
        }
    }
    assert ToolSelection().score(output, ["web_search"]).value == 1
    assert FactAndOutcome().score(output, ["PulseBoard"], "answer").value == 1
    assert CitationAndBudget().score(output).value == 1
