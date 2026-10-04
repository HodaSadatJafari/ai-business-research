"""Measure retrieval independently of the LLM and compare search methods."""

import json
from pathlib import Path

from app.retrieval import Retriever, load_chunks

ROOT = Path(__file__).resolve().parents[1]
CASES = Path(__file__).with_name("week2_retrieval_cases.json")


def evaluate(k: int = 2) -> dict[str, dict[str, float]]:
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    retriever = Retriever(load_chunks(ROOT / "knowledge_base"))
    try:
        results = {}
        for method in ("vector", "bm25", "hybrid"):
            recalls = []
            reciprocal_ranks = []
            for case in cases:
                relevant = set(case["relevant_sources"])
                hits = retriever.search(case["question"], k=k, method=method)
                sources = [hit.chunk.source for hit in hits]
                recalls.append(len(relevant.intersection(sources)) / len(relevant))
                reciprocal_ranks.append(
                    next(
                        (
                            1 / rank
                            for rank, source in enumerate(sources, 1)
                            if source in relevant
                        ),
                        0,
                    )
                )
            results[method] = {
                f"recall@{k}": round(sum(recalls) / len(cases), 3),
                f"mrr@{k}": round(sum(reciprocal_ranks) / len(cases), 3),
            }
        return results
    finally:
        retriever.close()


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
