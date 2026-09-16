from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    question: str = Field(min_length=5, max_length=2000)


class ResearchFinding(BaseModel):
    claim: str
    explanation: str


class ResearchResponse(BaseModel):
    question: str
    summary: str
    findings: list[ResearchFinding]
    limitations: list[str]
