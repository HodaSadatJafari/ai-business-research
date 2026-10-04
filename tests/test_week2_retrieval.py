from pathlib import Path

from app.retrieval import load_chunks

ROOT = Path(__file__).resolve().parents[1] / "knowledge_base"


def test_ingestion_keeps_source_metadata_and_stable_ids():
    first = load_chunks(ROOT)
    second = load_chunks(ROOT)
    assert first == second
    assert len(first) >= 6
    assert len({chunk.chunk_id for chunk in first}) == len(first)
    assert all(chunk.source and chunk.title and chunk.section for chunk in first)
