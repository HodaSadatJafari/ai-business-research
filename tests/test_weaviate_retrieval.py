from pathlib import Path
from types import SimpleNamespace

import pytest

from app.retrieval import (
    Chunk,
    Retriever,
    collection_name,
    load_chunks,
)

ROOT = Path(__file__).resolve().parents[1] / "knowledge_base"


def one_chunk():
    return Chunk(
        "chunk-1", "document-1", "doc.md", "Doc", "Info", "Alpha price is $10."
    )


def fake_object(chunk, *, score=0.8, distance=0.2):
    return SimpleNamespace(
        properties={
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "source": chunk.source,
            "title": chunk.title,
            "section": chunk.section,
            "text": chunk.text,
        },
        metadata=SimpleNamespace(score=score, distance=distance),
    )


def test_collection_identity_changes_with_content_and_inference_image_tag():
    chunk = one_chunk()
    original = collection_name([chunk], "image-a")
    assert original == collection_name([chunk], "image-a")
    assert original != collection_name([chunk], "image-b")
    changed = Chunk(**{**chunk.__dict__, "text": "Different content"})
    assert original != collection_name([changed], "image-a")


def test_retriever_uses_inference_image_tag_for_collection_identity(monkeypatch):
    monkeypatch.setattr(
        "app.retrieval.get_settings",
        lambda: SimpleNamespace(
            embedding_model="old-label", model2vec_image_tag="image-b"
        ),
    )
    chunk = one_chunk()
    collection = SimpleNamespace(
        aggregate=SimpleNamespace(
            over_all=lambda total_count: SimpleNamespace(total_count=1)
        ),
        query=SimpleNamespace(
            fetch_objects=lambda **kwargs: SimpleNamespace(objects=[fake_object(chunk)])
        ),
    )
    client = SimpleNamespace(
        collections=SimpleNamespace(
            exists=lambda name: True, get=lambda name: collection
        )
    )

    retriever = Retriever([chunk], client=client)
    assert retriever.collection_name == collection_name([chunk], "image-b")


def test_retriever_uses_weaviate_bm25_vector_and_hybrid_queries():
    chunk = one_chunk()
    calls = []

    def query(name):
        def invoke(**kwargs):
            calls.append((name, kwargs))
            return SimpleNamespace(objects=[fake_object(chunk)])

        return invoke

    collection = SimpleNamespace(
        aggregate=SimpleNamespace(
            over_all=lambda total_count: SimpleNamespace(total_count=1)
        ),
        query=SimpleNamespace(
            fetch_objects=lambda **kwargs: SimpleNamespace(
                objects=[fake_object(chunk)]
            ),
            bm25=query("bm25"),
            near_text=query("near_text"),
            hybrid=query("hybrid"),
        ),
    )
    client = SimpleNamespace(
        collections=SimpleNamespace(
            exists=lambda name: True, get=lambda name: collection
        )
    )
    retriever = Retriever([chunk], client=client)

    for method in ("bm25", "vector", "hybrid"):
        hits = retriever.search("Alpha price", k=1, method=method)
        assert [hit.chunk for hit in hits] == [chunk]

    assert [name for name, _ in calls] == ["bm25", "near_text", "hybrid"]
    assert calls[0][1]["query"] == "Alpha price"
    assert calls[1][1]["query"] == "Alpha price"
    assert calls[2][1]["query"] == "Alpha price"
    assert "vector" not in calls[2][1]
    assert calls[2][1]["alpha"] == 0.5


def test_new_collection_lets_weaviate_vectorize_the_text():
    chunk = one_chunk()
    inserted = []
    collection = SimpleNamespace(
        data=SimpleNamespace(
            insert_many=lambda objects: (
                inserted.extend(objects) or SimpleNamespace(has_errors=False)
            )
        ),
        aggregate=SimpleNamespace(
            over_all=lambda total_count: SimpleNamespace(total_count=len(inserted))
        ),
    )
    created = []
    client = SimpleNamespace(
        collections=SimpleNamespace(
            exists=lambda name: False,
            create=lambda **kwargs: created.append(kwargs) or collection,
        )
    )
    retriever = Retriever([chunk], client=client)
    assert created[0]["name"] == retriever.collection_name
    vector_config = created[0]["vector_config"]
    assert vector_config.properties == ["text"]
    assert vector_config.vectorizer.vectorizer.value == "text2vec-model2vec"
    assert len(inserted) == 1
    assert inserted[0].properties["text"] == chunk.text
    assert inserted[0].vector is None


def test_partial_existing_collection_is_rejected():
    collection = SimpleNamespace(
        aggregate=SimpleNamespace(
            over_all=lambda total_count: SimpleNamespace(total_count=0)
        )
    )
    client = SimpleNamespace(
        collections=SimpleNamespace(
            exists=lambda name: True, get=lambda name: collection
        )
    )
    with pytest.raises(ValueError, match="incomplete"):
        Retriever([one_chunk()], client=client)


def test_failed_new_import_removes_the_incomplete_collection():
    deleted = []
    collection = SimpleNamespace(
        data=SimpleNamespace(
            insert_many=lambda objects: SimpleNamespace(
                has_errors=True, errors={0: "failed"}
            )
        )
    )
    client = SimpleNamespace(
        collections=SimpleNamespace(
            exists=lambda name: False,
            create=lambda **kwargs: collection,
            delete=lambda name: deleted.append(name),
        )
    )

    with pytest.raises(RuntimeError, match="import failed"):
        Retriever([one_chunk()], client=client, collection_name="FailedImport")
    assert deleted == ["FailedImport"]


def test_existing_collection_with_wrong_content_is_rejected():
    chunk = one_chunk()
    wrong = Chunk(**{**chunk.__dict__, "text": "Unrelated content"})
    collection = SimpleNamespace(
        aggregate=SimpleNamespace(
            over_all=lambda total_count: SimpleNamespace(total_count=1)
        ),
        query=SimpleNamespace(
            fetch_objects=lambda **kwargs: SimpleNamespace(objects=[fake_object(wrong)])
        ),
    )
    client = SimpleNamespace(
        collections=SimpleNamespace(
            exists=lambda name: True, get=lambda name: collection
        )
    )
    with pytest.raises(ValueError, match="different content"):
        Retriever([chunk], client=client)


def test_live_queries_do_not_fall_back_to_local_search():
    chunk = one_chunk()

    def unavailable(**kwargs):
        raise RuntimeError("Weaviate is unavailable")

    collection = SimpleNamespace(
        aggregate=SimpleNamespace(
            over_all=lambda total_count: SimpleNamespace(total_count=1)
        ),
        query=SimpleNamespace(
            fetch_objects=lambda **kwargs: SimpleNamespace(
                objects=[fake_object(chunk)]
            ),
            bm25=unavailable,
        ),
    )
    client = SimpleNamespace(
        collections=SimpleNamespace(
            exists=lambda name: True, get=lambda name: collection
        )
    )
    retriever = Retriever([chunk], client=client)
    with pytest.raises(RuntimeError, match="Weaviate is unavailable"):
        retriever.search("Alpha", method="bm25")


def test_chunk_loader_still_preserves_source_metadata():
    chunks = load_chunks(ROOT)
    assert len(chunks) >= 6
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert all(chunk.source and chunk.title and chunk.section for chunk in chunks)
