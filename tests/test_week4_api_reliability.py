"""Production API behavior for synchronous services and upstream failures."""

from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import APIConnectionError, APIStatusError, RateLimitError

from app.api import routes
from app.main import app
from app.schemas.research import ResearchResponse

QUESTION = "What changed this month?"
ENDPOINTS = (
    "/api/v1/research",
    "/api/v1/research/rag",
    "/api/v1/research/agent",
)


def _set_service(monkeypatch, endpoint: str, service: SimpleNamespace) -> None:
    if endpoint == "/api/v1/research":
        monkeypatch.setattr(routes, "research_service", service)
    elif endpoint == "/api/v1/research/rag":
        monkeypatch.setattr(routes, "get_rag_service", lambda: service)
    else:
        monkeypatch.setattr(routes, "get_agent_service", lambda: service)


def test_baseline_endpoint_returns_synchronous_service_response(monkeypatch):
    response = ResearchResponse(
        question=QUESTION, summary="Done", findings=[], limitations=[]
    )
    _set_service(
        monkeypatch,
        "/api/v1/research",
        SimpleNamespace(research=lambda _: response),
    )

    result = TestClient(app, raise_server_exceptions=False).post(
        "/api/v1/research", json={"question": QUESTION}
    )

    assert result.status_code == 200
    assert result.json()["summary"] == "Done"


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize(
    ("error_kind", "expected_status", "expected_detail"),
    [
        ("connection", 503, "Model provider temporarily unavailable."),
        ("rate_limit", 503, "Model provider temporarily unavailable."),
        ("auth", 502, "Model provider request failed."),
    ],
)
def test_upstream_errors_have_stable_safe_responses(
    monkeypatch, endpoint, error_kind, expected_status, expected_detail
):
    request = httpx.Request("POST", "https://provider.example/v1/chat")
    secret_detail = "private upstream detail"
    if error_kind == "connection":
        error = APIConnectionError(message=secret_detail, request=request)
    else:
        response = httpx.Response(
            429 if error_kind == "rate_limit" else 401, request=request
        )
        error_type = RateLimitError if error_kind == "rate_limit" else APIStatusError
        error = error_type(secret_detail, response=response, body=None)

    def fail(_):
        raise error

    _set_service(monkeypatch, endpoint, SimpleNamespace(research=fail))
    result = TestClient(app, raise_server_exceptions=False).post(
        endpoint, json={"question": QUESTION}
    )

    assert result.status_code == expected_status
    assert result.json() == {"detail": expected_detail}
    assert secret_detail not in result.text
