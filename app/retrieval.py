"""Deterministic document chunks vectorized and searched in Weaviate."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import weaviate
from weaviate.auth import Auth
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.data import DataObject
from weaviate.classes.query import HybridFusion, MetadataQuery

from app.config import get_settings

HYBRID_ALPHA = 0.5


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    source: str
    title: str
    section: str
    text: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


def load_chunks(
    root: Path, *, chunk_words: int = 120, overlap: int = 20
) -> list[Chunk]:
    if not 0 <= overlap < chunk_words:
        raise ValueError("overlap must be smaller than chunk_words")
    chunks = []
    for path in sorted(root.rglob("*.md")):
        source = path.relative_to(root).as_posix()
        document_id = hashlib.sha256(source.encode()).hexdigest()[:12]
        title = path.stem.replace("_", " ").title()
        section = title
        sections: list[tuple[str, str]] = []
        lines: list[str] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            if raw.startswith("# "):
                title = raw[2:].strip()
                section = title
            elif raw.startswith("## "):
                if lines:
                    sections.append((section, " ".join(lines)))
                    lines = []
                section = raw[3:].strip()
            elif raw.strip():
                lines.append(raw.strip())
        if lines:
            sections.append((section, " ".join(lines)))
        for section_name, body in sections:
            words = body.split()
            step = chunk_words - overlap
            for start in range(0, len(words), step):
                piece = words[start : start + chunk_words]
                if not piece:
                    continue
                ordinal = start // step
                chunk_id = f"{document_id}:{len(chunks)}:{ordinal}"
                chunks.append(
                    Chunk(
                        chunk_id,
                        document_id,
                        source,
                        title,
                        section_name,
                        " ".join(piece),
                    )
                )
                if start + chunk_words >= len(words):
                    break
    return chunks


def name_for_corpus(chunks: list[Chunk], model2vec_image_tag: str) -> str:
    """Name an immutable index from its corpus and inference image tag."""
    identity = f"weaviate-text2vec-model2vec:{model2vec_image_tag}"
    digest = hashlib.sha256(identity.encode())
    for chunk in chunks:
        for field in (
            chunk.chunk_id,
            chunk.document_id,
            chunk.source,
            chunk.title,
            chunk.section,
            chunk.text,
        ):
            digest.update(b"\0")
            digest.update(field.encode())
    return f"ResearchChunks{digest.hexdigest()[:12]}"


collection_name = name_for_corpus


def connect_weaviate():
    """Connect to local Weaviate over HTTP or authenticated Weaviate Cloud."""
    settings = get_settings()
    raw = settings.weaviate_url
    url = raw if "://" in raw else f"https://{raw}"
    parsed = urlsplit(url)
    if not parsed.hostname:
        raise ValueError("WEAVIATE_URL must contain a host")
    credentials = (
        Auth.api_key(settings.weaviate_api_key) if settings.weaviate_api_key else None
    )
    if parsed.scheme == "https":
        if credentials is None:
            raise ValueError("WEAVIATE_API_KEY is required for Weaviate Cloud")
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=url, auth_credentials=credentials
        )
    if parsed.scheme == "http":
        return weaviate.connect_to_local(
            host=parsed.hostname,
            port=parsed.port or 8080,
            grpc_port=settings.weaviate_grpc_port,
            auth_credentials=credentials,
        )
    raise ValueError("WEAVIATE_URL must use http or https")


class Retriever:
    """A Weaviate-backed document index with three native search modes."""

    def __init__(
        self,
        chunks: list[Chunk],
        *,
        client=None,
        collection_name: str | None = None,
    ):
        if not chunks:
            raise ValueError("No knowledge base document chunks to index")
        settings = get_settings()
        self.chunks = chunks
        self.collection_name = collection_name or name_for_corpus(
            chunks, settings.model2vec_image_tag
        )
        self._owns_client = client is None
        self.client = client or connect_weaviate()
        try:
            self.collection = self._ensure_collection()
        except BaseException:
            if self._owns_client:
                self.client.close()
            raise

    def _ensure_collection(self):
        created = False
        try:
            existing = self.client.collections.exists(self.collection_name)
            if existing:
                collection = self.client.collections.get(self.collection_name)
            else:
                collection = self.client.collections.create(
                    name=self.collection_name,
                    vector_config=Configure.Vectors.text2vec_model2vec(
                        source_properties=["text"], vectorize_collection_name=False
                    ),
                    properties=[
                        Property(
                            name=name, data_type=DataType.TEXT, index_searchable=False
                        )
                        for name in (
                            "chunk_id",
                            "document_id",
                            "source",
                            "title",
                            "section",
                        )
                    ]
                    + [Property(name="text", data_type=DataType.TEXT)],
                )
                created = True
                objects = [
                    DataObject(properties=chunk.__dict__) for chunk in self.chunks
                ]
                result = collection.data.insert_many(objects)
                if getattr(result, "has_errors", False) or getattr(
                    result, "errors", None
                ):
                    raise RuntimeError(
                        f"Weaviate import failed for collection {self.collection_name}"
                    )
            count = collection.aggregate.over_all(total_count=True).total_count
            if count != len(self.chunks):
                raise ValueError(
                    f"Weaviate collection {self.collection_name} is incomplete: "
                    f"expected {len(self.chunks)} chunks, found {count}"
                )
            if existing:
                stored = collection.query.fetch_objects(
                    limit=len(self.chunks),
                    return_properties=list(Chunk.__dataclass_fields__),
                ).objects
                actual = {
                    item.properties["chunk_id"]: item.properties for item in stored
                }
                expected = {chunk.chunk_id: chunk.__dict__ for chunk in self.chunks}
                if actual != expected:
                    raise ValueError(
                        f"Weaviate collection {self.collection_name} "
                        "has different content"
                    )
            return collection
        except BaseException as exc:
            if created:
                try:
                    self.client.collections.delete(self.collection_name)
                except Exception as cleanup_error:
                    exc.add_note(
                        f"Failed to remove incomplete collection: {cleanup_error}"
                    )
            raise

    def search(self, query: str, *, k: int = 4, method: str = "hybrid") -> list[Hit]:
        if method not in {"vector", "bm25", "hybrid"}:
            raise ValueError(f"Unknown retrieval method: {method}")
        if not query.strip() or k <= 0:
            return []
        if method == "bm25":
            response = self.collection.query.bm25(
                query=query,
                query_properties=["text"],
                limit=k,
                return_metadata=MetadataQuery(score=True),
            )
        elif method == "vector":
            response = self.collection.query.near_text(
                query=query,
                limit=k,
                return_metadata=MetadataQuery(distance=True),
            )
        else:
            response = self.collection.query.hybrid(
                query=query,
                alpha=HYBRID_ALPHA,
                fusion_type=HybridFusion.RELATIVE_SCORE,
                query_properties=["text"],
                limit=k,
                return_metadata=MetadataQuery(score=True),
            )
        hits = []
        for item in response.objects:
            chunk = Chunk(
                **{
                    field: item.properties[field]
                    for field in Chunk.__dataclass_fields__
                }
            )
            score = getattr(item.metadata, "score", None)
            if score is None:
                distance = getattr(item.metadata, "distance", None)
                score = -float(distance) if distance is not None else 0.0
            hits.append(Hit(chunk=chunk, score=float(score)))
        return hits

    def close(self) -> None:
        if self._owns_client:
            self.client.close()
