"""Load stage: export results to JSON / CSV / SQLite."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .pipeline import ETLResult


def to_json(results: ETLResult | Iterable[ETLResult], include_text: bool = False) -> str:
    if isinstance(results, ETLResult):
        return json.dumps(results.to_dict(include_text), indent=2, ensure_ascii=False)
    return json.dumps([r.to_dict(include_text) for r in results], indent=2, ensure_ascii=False)


def flatten(result: ETLResult) -> dict[str, Any]:
    """One flat row per document — ready for Excel / BI tools."""
    rec = result.record
    row: dict[str, Any] = {
        "doc_id": result.doc_id,
        "filename": result.filename,
        "document_type": rec.get("document_type"),
        "quality_score": result.quality_score,
        "ocr_confidence": result.ocr_confidence,
        "extraction_method": result.extraction_method,
        "emails": "; ".join(rec.get("emails", [])),
        "phones": "; ".join(rec.get("phones", [])),
        "currency": rec.get("currency"),
    }
    for key, value in rec.get("fields", {}).items():
        row[f"field.{key}"] = value if not isinstance(value, (dict, list)) else json.dumps(value)
    for d in rec.get("dates", []):
        row.setdefault(f"date.{d['label']}", d["value"])
    for a in rec.get("amounts", []):
        row.setdefault(f"amount.{a['label']}", a["value"])
    return row


def to_dataframe(results: Iterable[ETLResult]) -> pd.DataFrame:
    return pd.DataFrame([flatten(r) for r in results])


def checks_dataframe(result: ETLResult) -> pd.DataFrame:
    return pd.DataFrame([c.to_dict() for c in result.checks])


class SQLiteLoader:
    """Idempotent loader (upsert on document hash)."""

    def __init__(self, path: str | Path = "data/docai_etl.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    filename TEXT,
                    document_type TEXT,
                    processed_at TEXT,
                    extraction_method TEXT,
                    quality_score REAL,
                    ocr_confidence REAL,
                    record_json TEXT,
                    raw_text TEXT
                );
                CREATE TABLE IF NOT EXISTS extracted_fields (
                    doc_id TEXT,
                    category TEXT,
                    name TEXT,
                    value TEXT,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
                );
                CREATE INDEX IF NOT EXISTS idx_fields_doc ON extracted_fields(doc_id);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def save(self, result: ETLResult) -> None:
        rec = result.record
        rows: list[tuple[str, str, str, str]] = []
        for k, v in rec.get("fields", {}).items():
            rows.append((result.doc_id, "field", k, json.dumps(v) if isinstance(v, (dict, list)) else str(v)))
        rows += [(result.doc_id, "date", d["label"], d["value"]) for d in rec.get("dates", [])]
        rows += [(result.doc_id, "email", "email", e) for e in rec.get("emails", [])]
        rows += [(result.doc_id, "phone", "phone", p) for p in rec.get("phones", [])]
        rows += [(result.doc_id, "amount", a["label"], f"{a['value']:.2f}") for a in rec.get("amounts", [])]

        with self._connect() as con:
            con.execute("DELETE FROM extracted_fields WHERE doc_id = ?", (result.doc_id,))
            con.execute(
                "INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    result.doc_id,
                    result.filename,
                    rec.get("document_type"),
                    result.processed_at,
                    result.extraction_method,
                    result.quality_score,
                    result.ocr_confidence,
                    json.dumps(rec, ensure_ascii=False),
                    result.raw_text,
                ),
            )
            con.executemany("INSERT INTO extracted_fields VALUES (?,?,?,?)", rows)

    def history(self, limit: int = 200) -> pd.DataFrame:
        with self._connect() as con:
            return pd.read_sql_query(
                "SELECT doc_id, filename, document_type, processed_at, extraction_method, "
                "quality_score, ocr_confidence FROM documents ORDER BY processed_at DESC LIMIT ?",
                con,
                params=(limit,),
            )

    def fields(self, doc_id: str) -> pd.DataFrame:
        with self._connect() as con:
            return pd.read_sql_query(
                "SELECT category, name, value FROM extracted_fields WHERE doc_id = ?", con, params=(doc_id,)
            )
