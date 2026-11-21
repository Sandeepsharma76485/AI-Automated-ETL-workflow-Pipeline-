"""LLM integration tested against a mock Ollama / OpenAI-compatible server (no GPU needed)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from docai_etl import DocumentETLPipeline, PipelineConfig
from docai_etl.config import LLMConfig
from docai_etl.llm import LLMExtractor

# What Llama 3 8B typically returns: raw values, NOT normalised
MODEL_JSON = {
    "document_type": "letter",
    "summary": "Offer letter for a Data Scientist role.",
    "fields": {"candidate_name": "Ananya Rao", "date_of_joining": "2024/12/02", "candidate_email": "Ananya.Rao@Gmail.com",
               "annual_ctc": "INR 18,50,000"},
    "dates": [{"label": "letter date", "value": "21 October 2024"}, {"label": "acceptance deadline", "value": "15-11-2024"}],
    "emails": ["hr@NovaTechAnalytics.com"],
    "phones": ["+91-99887-66554"],
    "amounts": [],
    "line_items": [],
}


class _Handler(BaseHTTPRequestHandler):
    requests_seen: list = []

    def log_message(self, *args):  # silence
        pass

    def _send(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send({"models": [{"name": "llama3.1:8b"}]})
        else:
            self._send({"data": [{"id": "llama-3.1-8b-instant"}]})

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        req = json.loads(self.rfile.read(length))
        _Handler.requests_seen.append((self.path, req))
        content = "```json\n" + json.dumps(MODEL_JSON) + "\n```"
        if self.path == "/api/chat":
            self._send({"message": {"role": "assistant", "content": content}})
        else:
            self._send({"choices": [{"message": {"role": "assistant", "content": content}}]})


@pytest.fixture(scope="module")
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def test_ollama_health_and_extract(server):
    llm = LLMExtractor(LLMConfig(provider="ollama", base_url=server, model="llama3.1:8b"))
    ok, msg = llm.health_check()
    assert ok, msg
    data = llm.extract("some text")
    assert data["document_type"] == "letter"
    path, req = _Handler.requests_seen[-1]
    assert path == "/api/chat" and req["format"] == "json" and req["options"]["temperature"] == 0


def test_ollama_reports_missing_model(server):
    ok, msg = LLMExtractor(LLMConfig(provider="ollama", base_url=server, model="mistral")).health_check()
    assert not ok and "ollama pull mistral" in msg


def test_openai_compatible_provider(server):
    llm = LLMExtractor(LLMConfig(provider="openai_compatible", base_url=server + "/v1",
                                 model="llama-3.1-8b-instant", api_key="test"))
    assert llm.health_check()[0]
    assert llm.extract("x")["fields"]["candidate_name"] == "Ananya Rao"
    path, req = _Handler.requests_seen[-1]
    assert path == "/v1/chat/completions" and req["response_format"] == {"type": "json_object"}


def test_pipeline_with_llm_applies_rules(server):
    from pathlib import Path

    cfg = PipelineConfig()
    cfg.llm.base_url, cfg.llm.provider = server, "ollama"
    f = Path(__file__).resolve().parents[1] / "samples" / "offer_letter.pdf"
    res = DocumentETLPipeline(cfg).run(f.read_bytes(), f.name)
    assert res.extraction_method == "llm:llama3.1:8b"
    rec = res.record
    assert rec["fields"]["date_of_joining"] == "02/12/2024"          # LLM value normalised by rules
    assert rec["fields"]["candidate_email"] == "ananya.rao@gmail.com"
    assert rec["fields"]["annual_ctc"] == "1850000.00"
    labels = {d["label"]: d["value"] for d in rec["dates"]}
    assert labels["letter_date"] == "21/10/2024" and labels["acceptance_deadline"] == "15/11/2024"
    assert rec["phones"] == ["+919988766554"]
