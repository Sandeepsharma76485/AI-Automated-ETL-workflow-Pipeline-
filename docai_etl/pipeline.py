"""End-to-end ETL orchestration: Extract -> AI Transform -> Rule Validation -> (Load)."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from .config import PipelineConfig
from .heuristics import heuristic_extract
from .llm import LLMError, LLMExtractor
from .ocr import OCRPage, extract_document
from .rules import RuleCheck, RuleEngine, quality_score

logger = logging.getLogger("docai_etl")

ProgressCallback = Callable[[str, float], None]


@dataclass
class ETLResult:
    doc_id: str
    filename: str
    processed_at: str
    pages: list[OCRPage]
    raw_text: str
    extraction_method: str
    llm_output: dict[str, Any]
    record: dict[str, Any]
    checks: list[RuleCheck]
    quality_score: float
    timings: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def ocr_confidence(self) -> float | None:
        confs = [p.confidence for p in self.pages if p.confidence is not None]
        return round(sum(confs) / len(confs), 1) if confs else None

    def to_dict(self, include_text: bool = True) -> dict[str, Any]:
        out = {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "processed_at": self.processed_at,
            "extraction_method": self.extraction_method,
            "quality_score": self.quality_score,
            "ocr_confidence": self.ocr_confidence,
            "pages": [
                {
                    "page": p.page_number,
                    "method": p.method,
                    "confidence": p.confidence,
                    "words": p.word_count,
                    "preprocessing": p.preprocessing_steps,
                }
                for p in self.pages
            ],
            "data": self.record,
            "rule_checks": [c.to_dict() for c in self.checks],
            "timings_sec": self.timings,
            "warnings": self.warnings,
        }
        if include_text:
            out["raw_text"] = self.raw_text
        return out


class DocumentETLPipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.rules = RuleEngine(self.config.rules)
        self.llm = LLMExtractor(self.config.llm)

    def run(self, file_bytes: bytes, filename: str, progress: ProgressCallback | None = None) -> ETLResult:
        notify = progress or (lambda _msg, _pct: None)
        timings: dict[str, float] = {}
        warnings: list[str] = []
        doc_id = hashlib.sha256(file_bytes).hexdigest()[:12]

        # ---------------- EXTRACT ----------------
        notify("Extracting text (OpenCV + Tesseract OCR)", 0.10)
        t0 = time.perf_counter()
        pages = extract_document(file_bytes, filename, self.config.ocr, self.config.preprocess)
        timings["extract"] = round(time.perf_counter() - t0, 2)
        raw_text = "\n\n".join(p.text for p in pages if p.text).strip()
        if not raw_text:
            warnings.append("No text could be extracted from the document.")
        low_conf = [p.page_number for p in pages if p.confidence is not None and p.confidence < 60]
        if low_conf:
            warnings.append(f"Low OCR confidence on page(s) {low_conf}; consider a higher-quality scan.")

        # ---------------- TRANSFORM (AI) ----------------
        t1 = time.perf_counter()
        extracted: dict[str, Any] = {}
        method = "heuristic"
        if self.config.llm.enabled and raw_text:
            notify(f"Structuring data with {self.config.llm.model}", 0.45)
            try:
                extracted = self.llm.extract(raw_text)
                method = self.llm.label
            except LLMError as exc:
                logger.warning("LLM extraction failed, falling back to heuristics: %s", exc)
                warnings.append(f"LLM unavailable ({exc}); used rule-based heuristic extraction.")
        if not extracted:
            notify("Structuring data with heuristic extractor", 0.45)
            extracted = heuristic_extract(raw_text)
        timings["transform_ai"] = round(time.perf_counter() - t1, 2)

        # ---------------- VALIDATE (Rules) ----------------
        notify("Applying validation rules (dates, emails, phones, amounts)", 0.80)
        t2 = time.perf_counter()
        record, checks = self.rules.apply(extracted, raw_text)
        timings["validate_rules"] = round(time.perf_counter() - t2, 3)
        timings["total"] = round(sum(timings.values()), 2)
        notify("Done", 1.0)

        return ETLResult(
            doc_id=doc_id,
            filename=filename,
            processed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            pages=pages,
            raw_text=raw_text,
            extraction_method=method,
            llm_output=extracted,
            record=record,
            checks=checks,
            quality_score=quality_score(checks),
            timings=timings,
            warnings=warnings,
        )
