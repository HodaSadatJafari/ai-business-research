"""Evidence contract for grounded agent responses."""

import re
from dataclasses import dataclass
from typing import Literal

from app.schemas.research import ResearchResponse, ResearchSource


@dataclass(frozen=True)
class Evidence:
    source_id: str
    title: str
    section: str
    path: str
    text: str
    source_type: Literal["document", "web", "sql"] = "document"


def grounded_response(
    question: str,
    parsed: ResearchResponse,
    evidence: list[Evidence],
    *,
    limitations: list[str] | None = None,
    tools_used: list[str] | None = None,
    tool_call_count: int = 0,
) -> ResearchResponse:
    allowed = {item.source_id: item for item in evidence}
    findings = []
    for finding in parsed.findings:
        # Absence of an answer belongs in limitations, not a cited finding.
        if re.search(
            r"\b(cannot be determined|not provided|does not identify|"
            r"insufficient evidence|unknown|no evidence|not established)\b",
            finding.claim,
            flags=re.IGNORECASE,
        ):
            continue
        quote = (finding.evidence_quote or "").strip()
        ids = (
            list(
                dict.fromkeys(
                    source_id
                    for source_id in finding.source_ids
                    if source_id in allowed and quote in allowed[source_id].text
                )
            )
            if quote
            else []
        )
        if ids:
            findings.append(finding.model_copy(update={"source_ids": ids}))
    combined_limitations = list(parsed.limitations) + list(limitations or [])
    if not findings:
        combined_limitations.append(
            "No finding had a valid quote from retrieved evidence."
        )
        return ResearchResponse(
            question=question,
            summary="The available evidence did not yield a cited answer.",
            findings=[],
            limitations=combined_limitations,
            tools_used=list(tools_used or []),
            tool_call_count=tool_call_count,
        )
    cited = {source_id for finding in findings for source_id in finding.source_ids}
    sources = [
        ResearchSource(
            source_id=item.source_id,
            title=item.title,
            section=item.section,
            path=item.path,
            source_type=item.source_type,
        )
        for item in evidence
        if item.source_id in cited
    ]
    return ResearchResponse(
        question=question,
        summary=" ".join(finding.claim for finding in findings),
        findings=findings,
        limitations=combined_limitations,
        sources=sources,
        tools_used=list(tools_used or []),
        tool_call_count=tool_call_count,
    )
