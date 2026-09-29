"""Read-only evidence tools available to the Week 3 agent."""

import csv
import hashlib
import json
import re
import sqlite3
from pathlib import Path
from urllib.request import Request, urlopen

from app.retrieval import Retriever
from app.services.evidence import Evidence

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
METRICS_CSV = Path(__file__).resolve().parents[1] / "data" / "support_metrics.csv"


class DocumentSearch:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def search(self, query: str) -> list[Evidence]:
        hits = self.retriever.search(query, k=4, method="bm25")
        return [
            Evidence(
                source_id=hit.chunk.chunk_id,
                title=hit.chunk.title,
                section=hit.chunk.section,
                path=hit.chunk.source,
                text=hit.chunk.text,
            )
            for hit in hits
        ]


class TavilySearch:
    def __init__(self, api_key: str | None):
        self.api_key = api_key

    def search(self, query: str) -> list[Evidence]:
        if not self.api_key:
            raise RuntimeError("TAVILY_API_KEY is not configured")
        if not query.strip() or len(query) > 400:
            raise ValueError("Web search query must be 1–400 characters")
        body = json.dumps(
            {"query": query, "search_depth": "basic", "max_results": 5}
        ).encode("utf-8")
        request = Request(
            TAVILY_SEARCH_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=10) as response:
            payload = json.load(response)
        items = []
        seen = set()
        for result in payload.get("results", [])[:5]:
            url = result.get("url", "")
            content = result.get("content", "")
            if (
                not url.startswith(("https://", "http://"))
                or not content
                or url in seen
            ):
                continue
            seen.add(url)
            excerpt = content[:1200]
            digest = hashlib.sha256(f"{url}\n{excerpt}".encode()).hexdigest()[:16]
            items.append(
                Evidence(
                    source_id=f"web:{digest}",
                    title=result.get("title") or url,
                    section="Search excerpt",
                    path=url,
                    text=excerpt,
                    source_type="web",
                )
            )
        return items


class SupportMetrics:
    def __init__(self, csv_path: Path = METRICS_CSV):
        self.csv_path = csv_path

    def query(self, product: str = "all", month: str | None = None) -> list[Evidence]:
        if product not in {"all", "SignalDesk", "EchoMap"}:
            raise ValueError("Unknown product")
        if month is not None and not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", month):
            raise ValueError("Month must be YYYY-MM")
        with sqlite3.connect(":memory:") as db:
            db.execute(
                "CREATE TABLE metrics (row_id TEXT, product TEXT, month TEXT, "
                "ticket_count INTEGER, avg_first_response_hours REAL)"
            )
            with self.csv_path.open(newline="", encoding="utf-8") as file:
                rows = csv.DictReader(file)
                db.executemany(
                    "INSERT INTO metrics VALUES (?, ?, ?, ?, ?)",
                    (
                        (
                            row["row_id"],
                            row["product"],
                            row["month"],
                            int(row["ticket_count"]),
                            float(row["avg_first_response_hours"]),
                        )
                        for row in rows
                    ),
                )
            result = db.execute(
                "SELECT row_id, product, month, ticket_count, avg_first_response_hours "
                "FROM metrics WHERE (? = 'all' OR product = ?) "
                "AND (? IS NULL OR month = ?) ORDER BY product, month",
                (product, product, month, month),
            ).fetchall()
        return [
            Evidence(
                source_id=f"sql:{row_id}",
                title="Fictional support metrics",
                section=month_value,
                path=f"sql:support_metrics:{row_id}",
                text=(
                    f"In {month_value}, {product_value} had {tickets} support tickets "
                    f"and average first response time {hours:g} hours."
                ),
                source_type="sql",
            )
            for row_id, product_value, month_value, tickets, hours in result
        ]
