"""Run the same knowledge-specific questions through v0.1 and Week 2 RAG.

The required-text check is a small diagnostic, not a factuality metric. Inspect the
saved answers and citations before drawing a quality conclusion.
"""

import argparse
import json
import time
from pathlib import Path

from app.services.rag import RAGResearchService
from app.services.research import ResearchService

CASES = Path(__file__).with_name("week2_answer_cases.json")


def run(limit: int | None = None, selected: str = "both") -> dict:
    cases = json.loads(CASES.read_text(encoding="utf-8"))[:limit]
    services = {}
    if selected in ("baseline", "both"):
        services["baseline"] = ResearchService()
    if selected in ("rag", "both"):
        services["rag"] = RAGResearchService()
    rows = []
    for case in cases:
        row = {"question": case["question"], "required_text": case["required_text"]}
        for name, service in services.items():
            start = time.perf_counter()
            response = service.research(case["question"])
            elapsed = time.perf_counter() - start
            answer = response.model_dump(mode="json")
            visible_text = " ".join(
                [answer["summary"]]
                + [
                    part
                    for finding in answer["findings"]
                    for part in (finding["claim"], finding["explanation"])
                ]
            )
            row[name] = {
                "response": answer,
                "latency_seconds": round(elapsed, 3),
                "required_text_found": case["required_text"].lower()
                in visible_text.lower(),
            }
        rows.append(row)
    return {
        "cases": rows,
        "summary": {
            name: {
                "required_text_found": sum(
                    row[name]["required_text_found"] for row in rows
                ),
                "total": len(rows),
                "mean_latency_seconds": round(
                    sum(row[name]["latency_seconds"] for row in rows) / len(rows), 3
                ),
            }
            for name in services
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--service", choices=("baseline", "rag", "both"), default="both"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("week2_answer_results.json")
    )
    args = parser.parse_args()
    result = run(args.limit, args.service)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved case-level answers to {args.output}")
