import io
import json
from pathlib import Path

import pytest

from app.agent_tools import SupportMetrics, TavilySearch


def test_support_metrics_uses_fixed_rows_and_validates_filters():
    rows = SupportMetrics().query(product="SignalDesk", month="2026-07")
    assert len(rows) == 1
    assert rows[0].source_id == "sql:sd-2026-07"
    assert "128 support tickets" in rows[0].text
    assert rows[0].source_type == "sql"
    with pytest.raises(ValueError):
        SupportMetrics().query(product="SignalDesk; DROP TABLE metrics")


def test_tavily_result_is_bounded_and_preserves_url(monkeypatch):
    payload = {
        "results": [
            {
                "url": f"https://example.com/{i}",
                "title": f"Result {i}",
                "content": "Fact.",
            }
            for i in range(7)
        ]
    }

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.close()

    def fake_urlopen(request, timeout):
        assert timeout == 10
        assert request.full_url == "https://api.tavily.com/search"
        return Response(json.dumps(payload).encode())

    monkeypatch.setattr("app.agent_tools.urlopen", fake_urlopen)
    results = TavilySearch("test-key").search("test query")
    assert len(results) == 5
    assert results[0].path == "https://example.com/0"
    assert results[0].source_type == "web"


def test_metrics_fixture_is_in_repository():
    assert (
        Path(__file__)
        .resolve()
        .parents[1]
        .joinpath("data/support_metrics.csv")
        .exists()
    )
