import json
from types import SimpleNamespace

from app.schemas.research import ResearchFinding, ResearchResponse
from app.services import agent
from app.services.evidence import Evidence


def tool_call(name, args, identifier):
    return SimpleNamespace(
        id=identifier,
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


class FakeCompletions:
    def __init__(self, rounds, parsed=None):
        self.rounds = iter(rounds)
        self.parsed = parsed
        self.final_messages = None

    def create(self, **kwargs):
        calls = next(self.rounds)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=calls))
            ]
        )

    def parse(self, **kwargs):
        self.final_messages = kwargs["messages"]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=self.parsed))]
        )


def make_service(monkeypatch, rounds, parsed=None, *, web=None):
    monkeypatch.setattr(
        agent,
        "get_settings",
        lambda: SimpleNamespace(
            openai_api_key="test",
            openai_base_url=None,
            openai_model="test-model",
            agent_model=None,
            knowledge_base_dir="unused",
            tavily_api_key=None,
        ),
    )
    completions = FakeCompletions(rounds, parsed)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
        beta=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    document = Evidence("doc:1", "Doc", "Section", "doc.md", "The price is $480.")
    metric = Evidence(
        "sql:1",
        "Metrics",
        "July",
        "sql:metrics:1",
        "In July, there were 128 tickets.",
        "sql",
    )
    docs = SimpleNamespace(search=lambda query: [document])
    metrics = SimpleNamespace(query=lambda **kwargs: [metric])
    web_tool = web or SimpleNamespace(search=lambda query: [])
    service = agent.AgentResearchService(
        client=client, documents=docs, metrics=metrics, web=web_tool
    )
    return service, completions


def test_agent_uses_multiple_tools_and_retains_only_quoted_sources(monkeypatch):
    parsed = ResearchResponse(
        question="test",
        summary="Invented extra claim.",
        findings=[
            ResearchFinding(
                claim="The price is $480.",
                explanation="Document pricing.",
                source_ids=["E1"],
                evidence_quote="The price is $480.",
            ),
            ResearchFinding(
                claim="There were 128 tickets.",
                explanation="July metric.",
                source_ids=["E2"],
                evidence_quote="there were 128 tickets.",
            ),
            ResearchFinding(
                claim="Unsupported.",
                explanation="Wrong citation.",
                source_ids=["E1"],
                evidence_quote="Not in the document",
            ),
        ],
        limitations=[],
    )
    rounds = [
        [
            tool_call("document_search", {"query": "price"}, "a"),
            tool_call("query_support_metrics", {"product": "SignalDesk"}, "b"),
        ],
        [],
    ]
    service, completions = make_service(monkeypatch, rounds, parsed)
    answer = service.research("Compare price and tickets")
    assert answer.tools_used == ["document_search", "query_support_metrics"]
    assert answer.tool_call_count == 2
    assert len(answer.findings) == 2
    assert answer.summary == "The price is $480. There were 128 tickets."
    assert {source.source_type for source in answer.sources} == {"document", "sql"}
    assert "E1" in completions.final_messages[1]["content"]


def test_agent_reports_missing_web_key_without_fabricating_answer(monkeypatch):
    def no_web(query):
        raise RuntimeError("TAVILY_API_KEY is not configured")

    service, completions = make_service(
        monkeypatch,
        [[tool_call("web_search", {"query": "current market"}, "a")], []],
        web=SimpleNamespace(search=no_web),
    )
    answer = service.research("What is new in this market?")
    assert answer.findings == []
    assert answer.tool_call_count == 1
    assert any("web_search failed" in item for item in answer.limitations)
    assert completions.final_messages is None


def test_agent_reports_empty_tool_results_accurately(monkeypatch):
    service, _ = make_service(
        monkeypatch,
        [[tool_call("web_search", {"query": "unknown"}, "a")], []],
    )
    answer = service.research("What happened to an unknown company?")
    assert answer.findings == []
    assert answer.limitations == ["No evidence was returned by selected tools."]


def test_agent_caps_total_and_web_calls(monkeypatch):
    calls = [
        tool_call("web_search", {"query": f"market {i}"}, str(i)) for i in range(5)
    ]
    service, _ = make_service(
        monkeypatch,
        [calls, []],
        web=SimpleNamespace(search=lambda query: []),
    )
    answer = service.research("Current market research")
    assert answer.tool_call_count == 2
    assert any("web limit" in item for item in answer.limitations)


def test_agent_caps_all_tool_calls_at_four(monkeypatch):
    calls = [
        tool_call("document_search", {"query": f"price {i}"}, str(i)) for i in range(5)
    ]
    parsed = ResearchResponse(
        question="Price?",
        summary="The price is $480.",
        findings=[
            ResearchFinding(
                claim="The price is $480.",
                explanation="Pricing document.",
                source_ids=["E1"],
                evidence_quote="The price is $480.",
            )
        ],
        limitations=[],
    )
    service, _ = make_service(monkeypatch, [calls], parsed)
    answer = service.research("What is the price?")
    assert answer.tool_call_count == 4
    assert any("four-call" in item for item in answer.limitations)


def test_agent_does_not_cite_unrelated_source_for_no_answer(monkeypatch):
    parsed = ResearchResponse(
        question="What is Orion Support's price?",
        summary="The price is unknown.",
        findings=[
            ResearchFinding(
                claim="The price is not provided.",
                explanation="The document is about another product.",
                source_ids=["E1"],
                evidence_quote="The price is $480.",
            )
        ],
        limitations=["Orion Support pricing is unavailable."],
    )
    service, _ = make_service(
        monkeypatch,
        [[tool_call("document_search", {"query": "Orion"}, "a")], []],
        parsed,
    )
    answer = service.research(parsed.question)
    assert answer.findings == []
    assert answer.sources == []
    assert answer.limitations
