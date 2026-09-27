"""Small, reproducible local index for the Week 2 knowledge base."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

WORD = re.compile(r"[a-z0-9]+", re.IGNORECASE)
DIMENSIONS = 512


def tokens(text: str) -> list[str]:
    return WORD.findall(text.lower())


def vectorize(text: str) -> dict[int, float]:
    """Hash word and adjacent-word features into a local dense-search vector."""
    words = tokens(text)
    features = words + [f"{a}_{b}" for a, b in zip(words, words[1:], strict=False)]
    counts: Counter[int] = Counter()
    for feature in features:
        bucket = (
            int.from_bytes(
                hashlib.blake2b(feature.encode(), digest_size=4).digest(), "big"
            )
            % DIMENSIONS
        )
        counts[bucket] += 1
    norm = math.sqrt(sum(value * value for value in counts.values()))
    return {bucket: value / norm for bucket, value in counts.items()} if norm else {}


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


class Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vectors = [vectorize(chunk.text) for chunk in chunks]
        self.term_counts = [Counter(tokens(chunk.text)) for chunk in chunks]
        self.lengths = [sum(counts.values()) for counts in self.term_counts]
        self.avg_length = sum(self.lengths) / len(chunks) if chunks else 0
        self.document_frequency = Counter(
            term for counts in self.term_counts for term in counts
        )

    def search(self, query: str, *, k: int = 4, method: str = "hybrid") -> list[Hit]:
        if method not in {"vector", "bm25", "hybrid"}:
            raise ValueError(f"Unknown retrieval method: {method}")
        if not self.chunks or not tokens(query) or k <= 0:
            return []
        query_terms = set(tokens(query))
        if not query_terms.intersection(self.document_frequency):
            return []
        query_vector = vectorize(query)
        vector_scores = [
            sum(
                weight * vector.get(bucket, 0)
                for bucket, weight in query_vector.items()
            )
            for vector in self.vectors
        ]
        bm25_scores = []
        for counts, length in zip(self.term_counts, self.lengths, strict=True):
            score = 0.0
            for term in query_terms:
                frequency = counts[term]
                if not frequency:
                    continue
                df = self.document_frequency[term]
                idf = math.log(1 + (len(self.chunks) - df + 0.5) / (df + 0.5))
                score += (
                    idf
                    * frequency
                    * 2.2
                    / (frequency + 1.2 * (0.25 + 0.75 * length / self.avg_length))
                )
            bm25_scores.append(score)
        if method == "hybrid":
            # Reciprocal rank fusion avoids comparing incompatible raw score scales.
            scores = [0.0] * len(self.chunks)
            for ranking in (vector_scores, bm25_scores):
                ranked = sorted(range(len(ranking)), key=lambda i: (-ranking[i], i))
                for rank, index in enumerate(ranked, start=1):
                    if ranking[index] > 0:
                        scores[index] += 1 / (60 + rank)
        else:
            scores = vector_scores if method == "vector" else bm25_scores
        ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [Hit(self.chunks[i], scores[i]) for i in ranked if scores[i] > 0][:k]
