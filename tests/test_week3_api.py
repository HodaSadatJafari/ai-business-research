from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.api import routes
from app.main import app
from app.schemas.research import ResearchResponse
from app.services.agent import AgentModelUnsupported


def test_agent_endpoint_returns_structured_response(monkeypatch):
    response = ResearchResponse(
        question="What are the support metrics?",
        summary="No usable evidence was retrieved.",
        findings=[],
        limitations=["The agent selected no evidence tools."],
        tools_used=[],
        tool_call_count=0,
    )
    monkeypatch.setattr(
        routes,
        "get_agent_service",
        lambda: SimpleNamespace(research=lambda _: response),
    )
    result = TestClient(app).post(
        "/api/v1/research/agent", json={"question": response.question}
    )
    assert result.status_code == 200
    assert result.json()["tool_call_count"] == 0


def test_agent_endpoint_reports_unsupported_tool_model(monkeypatch):
    def fail(_):
        raise AgentModelUnsupported("Configure AGENT_MODEL")

    monkeypatch.setattr(
        routes, "get_agent_service", lambda: SimpleNamespace(research=fail)
    )
    result = TestClient(app).post(
        "/api/v1/research/agent", json={"question": "What is current?"}
    )
    assert result.status_code == 502
    assert "AGENT_MODEL" in result.json()["detail"]
