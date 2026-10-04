"""Deterministic checks for the six-way evidence comparison."""

from __future__ import annotations


def _valid_citation(finding: dict, by_id: dict[str, dict]) -> bool:
    source_ids = finding.get("source_ids") or []
    quote = (finding.get("evidence_quote") or "").strip()
    return bool(
        len(source_ids) == 1
        and quote
        and source_ids[0] in by_id
        and quote in by_id[source_ids[0]]["text"]
    )


def score_case(case: dict, result: dict) -> dict[str, float]:
    """Score required facts and their exact cited evidence for one response."""
    response = result["response"]
    findings = response.get("findings") or []
    by_id = {item["source_id"]: item for item in result.get("evidence", [])}
    facts = case["expected_facts"]
    integrity = float(all(_valid_citation(item, by_id) for item in findings))

    if not facts:
        abstained = not findings and bool(response.get("limitations"))
        value = float(abstained)
        return {
            "fact_coverage": value,
            "grounded_fact_coverage": value,
            "citation_integrity": integrity,
            "task_success": value,
        }

    answer_text = " ".join(
        [response.get("summary", "")]
        + [finding.get("claim", "") for finding in findings]
    ).casefold()
    matched = 0
    grounded = 0
    for fact in facts:
        phrase = fact["phrase"].casefold()
        matched += phrase in answer_text
        for finding in findings:
            if phrase not in finding.get("claim", "").casefold():
                continue
            if not _valid_citation(finding, by_id):
                continue
            if any(
                by_id[source_id]["source_type"] == fact["source_type"]
                and by_id[source_id]["path"] == fact["locator"]
                and phrase in finding["evidence_quote"].casefold()
                for source_id in finding["source_ids"]
            ):
                grounded += 1
                break
    return {
        "fact_coverage": matched / len(facts),
        "grounded_fact_coverage": grounded / len(facts),
        "citation_integrity": integrity,
        "task_success": float(grounded == len(facts) and bool(integrity)),
    }
