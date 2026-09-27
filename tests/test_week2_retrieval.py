from pathlib import Path

from app.retrieval import Retriever, load_chunks

ROOT = Path(__file__).resolve().parents[1] / "knowledge_base"


def test_ingestion_keeps_source_metadata_and_stable_ids():
    first = load_chunks(ROOT)
    second = load_chunks(ROOT)
    assert first == second
    assert len(first) >= 6
    assert len({chunk.chunk_id for chunk in first}) == len(first)
    assert all(chunk.source and chunk.title and chunk.section for chunk in first)


def test_hybrid_finds_exact_product_fact():
    retriever = Retriever(load_chunks(ROOT))
    hits = retriever.search("How much does SignalDesk Team cost?", k=2)
    assert any(hit.chunk.source == "company/aster.md" for hit in hits)


def test_unknown_query_has_no_results():
    retriever = Retriever(load_chunks(ROOT))
    assert retriever.search("zzzxxyy qqwwvv") == []
