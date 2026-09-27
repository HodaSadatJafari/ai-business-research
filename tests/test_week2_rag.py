from pathlib import Path
from types import SimpleNamespace

from app.retrieval import Retriever, load_chunks
from app.schemas.research import ResearchFinding, ResearchResponse
from app.services import rag

ROOT = Path(__file__).resolve().parents[1] / "knowledge_base"


class FakeParse:
    def __init__(self, parsed):
        self.parsed = parsed
        self.messages = None

    def parse(self, **kwargs):
        self.messages = kwargs["messages"]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=self.parsed))]
        )


def service(monkeypatch, parsed):
    monkeypatch.setattr(
        rag,
        "get_settings",
        lambda: SimpleNamespace(
            knowledge_base_dir=ROOT,
            openai_api_key="test",
            openai_base_url=None,
            openai_model="test-model",
            retrieval_k=2,
            retrieval_method="bm25",
        ),
    )
    fake = FakeParse(parsed)
    client = SimpleNamespace(
        beta=SimpleNamespace(chat=SimpleNamespace(completions=fake))
    )
    return rag.RAGResearchService(
        retriever=Retriever(load_chunks(ROOT)), client=client
    ), fake


def test_rag_exposes_only_retrieved_citations(monkeypatch):
    question = "How much does SignalDesk Team cost?"
    retriever = Retriever(load_chunks(ROOT))
    expected_id = next(
        hit.chunk.chunk_id
        for hit in retriever.search(question, k=2, method="bm25")
        if hit.chunk.source == "company/aster.md"
    )
    parsed = ResearchResponse(
        question=question,
        summary="SignalDesk Team costs $480 per month. Unsupported extra claim.",
        findings=[
            ResearchFinding(
                claim="Team costs $480 per month.",
                explanation="The pricing section lists this price.",
                source_ids=[expected_id, "invented-id"],
                evidence_quote="SignalDesk Team costs $480 per month",
            )
        ],
        limitations=[],
    )
    rag_service, fake = service(monkeypatch, parsed)
    answer = rag_service.research(question)
    assert answer.findings[0].source_ids == [expected_id]
    assert answer.summary == "Team costs $480 per month."
    assert answer.sources[0].path == "company/aster.md"
    assert expected_id in fake.messages[1]["content"]


def test_rag_abstains_when_model_has_no_valid_citations(monkeypatch):
    parsed = ResearchResponse(
        question="How much does SignalDesk Team cost?",
        summary="An unsupported answer",
        findings=[
            ResearchFinding(
                claim="Unsupported", explanation="None", source_ids=["fake"]
            )
        ],
        limitations=[],
    )
    rag_service, _ = service(monkeypatch, parsed)
    answer = rag_service.research(parsed.question)
    assert answer.findings == []
    assert answer.sources == []
    assert answer.limitations


def test_rag_rejects_citation_to_wrong_section(monkeypatch):
    question = "Can SignalDesk analyze phone calls?"
    retriever = Retriever(load_chunks(ROOT))
    features_id = next(
        hit.chunk.chunk_id
        for hit in retriever.search(question, k=4, method="bm25")
        if hit.chunk.source == "product/signaldesk.md"
        and hit.chunk.section == "Features"
    )
    parsed = ResearchResponse(
        question=question,
        summary="SignalDesk does not ingest phone transcripts.",
        findings=[
            ResearchFinding(
                claim="SignalDesk does not ingest phone transcripts.",
                explanation="The product limit states this.",
                source_ids=[features_id],
                evidence_quote="It does not ingest phone transcripts",
            )
        ],
        limitations=[],
    )
    rag_service, _ = service(monkeypatch, parsed)
    rag_service.k = 4
    answer = rag_service.research(question)
    assert answer.findings == []
    assert answer.limitations
