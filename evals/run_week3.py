"""Run a repeatable Opik comparison of the Week 3 agent and Week 2 RAG."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import opik
from opik.evaluation import evaluate
from opik.evaluation.metrics import BaseMetric
from opik.evaluation.metrics.score_result import ScoreResult

from app.config import get_settings
from app.observability import configure_opik
from app.services.agent import AgentResearchService
from app.services.evidence import Evidence
from app.services.rag import RAGResearchService

ROOT = Path(__file__).resolve().parents[1]
CASES = Path(__file__).with_name("week3_cases.json")
DATASET_NAME = (
    "business-research-week3-" + hashlib.sha256(CASES.read_bytes()).hexdigest()[:8]
)
REPORT = ROOT / "experiments" / "results" / "003_comparison.json"


class FixedWebSearch:
    def __init__(self, fail: bool = False):
        self.fail = fail

    def search(self, query: str) -> list[Evidence]:
        if self.fail:
            raise RuntimeError("Simulated search outage")
        return (
            [
                Evidence(
                    source_id="web:aurora-pulseboard",
                    title="Aurora Support announcement (fixture)",
                    section="Announcement",
                    path="https://example.com/aurora-pulseboard",
                    text=(
                        "On 2026-09-01, Aurora Support announced PulseBoard, "
                        "a real-time support analytics dashboard."
                    ),
                    source_type="web",
                )
            ]
            if "aurora" in query.lower()
            else []
        )


class ToolSelection(BaseMetric):
    def __init__(self):
        super().__init__(name="tool_selection")

    def score(self, output: dict, expected_tools: list[str], **kwargs) -> ScoreResult:
        if not expected_tools:
            return ScoreResult(
                name=self.name, value=1.0, reason="No exact tool set required"
            )
        actual = set(output["response"]["tools_used"])
        expected = set(expected_tools)
        return ScoreResult(
            name=self.name,
            value=float(actual == expected),
            reason=f"Expected {sorted(expected)}; got {sorted(actual)}",
        )


class FactAndOutcome(BaseMetric):
    def __init__(self):
        super().__init__(name="fact_and_outcome")

    def score(
        self,
        output: dict,
        expected_phrases: list[str],
        expected_outcome: str,
        **kwargs,
    ) -> ScoreResult:
        response = output["response"]
        if expected_outcome == "limitation":
            passed = not response["findings"] and bool(response["limitations"])
        else:
            summary = response["summary"].lower()
            passed = bool(response["findings"]) and all(
                phrase.lower() in summary for phrase in expected_phrases
            )
        return ScoreResult(
            name=self.name,
            value=float(passed),
            reason=f"Outcome={expected_outcome}; phrases={expected_phrases}",
        )


class CitationAndBudget(BaseMetric):
    def __init__(self):
        super().__init__(name="citation_and_budget")

    def score(self, output: dict, **kwargs) -> ScoreResult:
        response = output["response"]
        sources = {item["source_id"] for item in response["sources"]}
        valid = all(
            finding["source_ids"]
            and finding["evidence_quote"]
            and set(finding["source_ids"]).issubset(sources)
            for finding in response["findings"]
        )
        bounded = response["tool_call_count"] <= 4
        return ScoreResult(
            name=self.name,
            value=float(valid and bounded),
            reason=f"Citations valid={valid}; tool calls={response['tool_call_count']}",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service", choices=("both", "agent", "rag"), default="both")
    args = parser.parse_args()
    configure_opik()
    settings = get_settings()
    client = opik.Opik(project_name=settings.opik_project_name)
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    dataset = client.get_or_create_dataset(
        name=DATASET_NAME, project_name=settings.opik_project_name
    )
    if not dataset.get_items():
        dataset.insert(cases)
    baseline = RAGResearchService() if args.service in ("both", "rag") else None

    def rag_task(item: dict) -> dict:
        start = time.perf_counter()
        assert baseline is not None
        response = baseline.research(item["input"])
        return {
            "output": {
                "response": response.model_dump(mode="json"),
                "latency_seconds": round(time.perf_counter() - start, 3),
            }
        }

    def agent_task(item: dict) -> dict:
        service = AgentResearchService(
            web=FixedWebSearch(fail=item["category"] == "tool_failure")
        )
        start = time.perf_counter()
        response = service.research(item["input"])
        return {
            "output": {
                "response": response.model_dump(mode="json"),
                "latency_seconds": round(time.perf_counter() - start, 3),
            }
        }

    metrics = [ToolSelection(), FactAndOutcome(), CitationAndBudget()]
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    for name, task in (("rag-week2", rag_task), ("agent-week3", agent_task)):
        if args.service != "both" and not name.startswith(args.service):
            continue
        result = evaluate(
            dataset=dataset,
            task=task,
            scoring_metrics=metrics,
            experiment_name=name,
            nb_samples=len(cases),
            task_threads=1,
        )
        rows = []
        for item in result.test_results:
            output = item.test_case.task_output["output"]
            rows.append(
                {
                    "category": item.test_case.dataset_item_content["category"],
                    "scores": {score.name: score.value for score in item.score_results},
                    "tools_used": output["response"]["tools_used"],
                    "latency_seconds": output["latency_seconds"],
                    "summary": output["response"]["summary"],
                    "limitations": output["response"]["limitations"],
                }
            )
        scores = {
            metric.name: round(
                sum(row["scores"][metric.name] for row in rows) / len(rows), 3
            )
            for metric in metrics
        }
        report[name] = {
            "experiment_id": result.experiment_id,
            "experiment_url": result.experiment_url,
            "scores": scores,
            "cases": rows,
        }
        print(name, result.experiment_url, scores)
        REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    opik.flush_tracker()


if __name__ == "__main__":
    main()
