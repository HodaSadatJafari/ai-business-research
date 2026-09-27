from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.api import routes
from app.main import app
from app.schemas.research import ResearchFinding, ResearchResponse, ResearchSource


def test_rag_endpoint_returns_cited_response(monkeypatch):
    response = ResearchResponse(
        question="How much does SignalDesk Team cost?",
        summary="SignalDesk Team costs $480 per month.",
        findings=[
            ResearchFinding(
                claim="SignalDesk Team costs $480 per month.",
                explanation="Listed in the pricing section.",
                source_ids=["source-1"],
                evidence_quote="SignalDesk Team costs $480 per month",
            )
        ],
        limitations=[],
        sources=[
            ResearchSource(
                source_id="source-1",
                title="Aster Analytics",
                section="Pricing",
                path="company/aster.md",
            )
        ],
    )
    monkeypatch.setattr(
        routes, "get_rag_service", lambda: SimpleNamespace(research=lambda _: response)
    )
    result = TestClient(app).post(
        "/api/v1/research/rag", json={"question": response.question}
    )
    assert result.status_code == 200
    assert result.json()["findings"][0]["source_ids"] == ["source-1"]
    assert result.json()["sources"][0]["path"] == "company/aster.md"
