"""Integration tests: real OpenCV + Tesseract on the generated sample documents (LLM disabled)."""

import json
from pathlib import Path

import pytest

from docai_etl import DocumentETLPipeline, PipelineConfig
from docai_etl.llm import LLMError, parse_json_response
from docai_etl.loaders import SQLiteLoader, to_dataframe, to_json
from docai_etl.ocr import UnsupportedFileError, tesseract_available

SAMPLES = Path(__file__).resolve().parents[1] / "samples"

needs_tesseract = pytest.mark.skipif(not tesseract_available(), reason="Tesseract not installed")


@pytest.fixture(scope="module")
def pipeline():
    cfg = PipelineConfig()
    cfg.llm.enabled = False
    return DocumentETLPipeline(cfg)


@pytest.fixture(scope="module")
def samples():
    if not (SAMPLES / "invoice_scanned.jpg").exists():
        from scripts.generate_samples import main

        main()
    return SAMPLES


@needs_tesseract
def test_scanned_invoice(pipeline, samples):
    f = samples / "invoice_scanned.jpg"
    res = pipeline.run(f.read_bytes(), f.name)
    rec = res.record
    assert rec["document_type"] == "invoice"
    assert res.pages[0].method == "tesseract"
    assert any(s.startswith("deskew") for s in res.pages[0].preprocessing_steps)
    dates = [d["value"] for d in rec["dates"]]
    assert {"15/03/2024", "14/04/2024", "05/04/2024"} <= set(dates)
    assert "rohan.mehta@pinkcityretail.com" in rec["emails"]
    assert "billing@acmecloud.co.in" in rec["emails"]
    assert "+919876543210" in rec["phones"]
    assert rec["fields"]["gstin"] == "08ABCDE1234F1Z5"
    assert res.quality_score >= 90


@needs_tesseract
def test_scanned_pdf_receipt(pipeline, samples):
    f = samples / "receipt_scanned.pdf"
    res = pipeline.run(f.read_bytes(), f.name)
    assert res.pages[0].method == "tesseract"
    assert "05/11/2024" in [d["value"] for d in res.record["dates"]]
    assert "hello@chaipointcafe.com" in res.record["emails"]


def test_native_pdf_offer_letter(pipeline, samples):
    f = samples / "offer_letter.pdf"
    res = pipeline.run(f.read_bytes(), f.name)
    assert res.pages[0].method == "native-pdf"
    assert [d["value"] for d in res.record["dates"]] == ["21/10/2024", "02/12/2024", "15/11/2024"]
    assert res.record["emails"] == ["hr@novatechanalytics.com", "ananya.rao@gmail.com"]


def test_llm_failure_falls_back_to_heuristics(samples):
    cfg = PipelineConfig()
    cfg.llm.enabled = True
    cfg.llm.base_url = "http://127.0.0.1:9"   # nothing listens here
    cfg.llm.timeout = 2
    f = samples / "offer_letter.pdf"
    res = DocumentETLPipeline(cfg).run(f.read_bytes(), f.name)
    assert res.extraction_method == "heuristic"
    assert any("LLM unavailable" in w for w in res.warnings)
    assert res.record["emails"]


def test_unsupported_file(pipeline):
    with pytest.raises(UnsupportedFileError):
        pipeline.run(b"hello", "notes.docx")


def test_loaders(pipeline, samples, tmp_path):
    f = samples / "offer_letter.pdf"
    res = pipeline.run(f.read_bytes(), f.name)
    assert json.loads(to_json(res))["data"]["emails"]
    df = to_dataframe([res])
    assert df.loc[0, "document_type"] == "letter"
    loader = SQLiteLoader(tmp_path / "t.db")
    loader.save(res)
    loader.save(res)                        # idempotent upsert
    assert len(loader.history()) == 1
    assert not loader.fields(res.doc_id).empty


def test_parse_json_response_strips_fences():
    assert parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_response('Sure! Here it is: {"a": 2} hope it helps') == {"a": 2}
    with pytest.raises(LLMError):
        parse_json_response("no json here")
