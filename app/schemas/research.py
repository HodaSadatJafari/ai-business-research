from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    question: str = Field(min_length=5, max_length=2000)


class ResearchFinding(BaseModel):
    claim: str
    explanation: str
    source_ids: list[str] = Field(default_factory=list)
    evidence_quote: str | None = None


class ResearchSource(BaseModel):
    source_id: str
    title: str
    section: str
    path: str


class ResearchResponse(BaseModel):
    question: str
    summary: str
    findings: list[ResearchFinding]
    limitations: list[str]
    sources: list[ResearchSource] = Field(default_factory=list)
