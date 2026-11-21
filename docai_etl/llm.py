"""Transform stage: Llama 3 8B turns raw OCR text into structured JSON.

Two providers, one interface:
* ``ollama``            -> local Llama 3.1 8B via Ollama's native /api/chat (JSON mode, 8k context)
* ``openai_compatible`` -> any OpenAI-style endpoint serving Llama 3 8B
                           (Groq ``llama-3.1-8b-instant``, Together, vLLM, LM Studio ...)

If no LLM is reachable the pipeline falls back to :mod:`docai_etl.heuristics`,
so the app always produces a result.
"""

from __future__ import annotations

import json
import re
from typing import Any

import requests

from .config import LLMConfig

SYSTEM_PROMPT = """You are an expert document data-extraction engine inside an ETL pipeline.
You receive raw OCR text (it may contain OCR noise) and must return ONLY one JSON object
with exactly this schema:

{
  "document_type": "invoice | receipt | purchase_order | bank_statement | resume | id_card | letter | form | other",
  "summary": "one short sentence describing the document",
  "fields": { "<snake_case_key>": "<value>" },
  "dates":  [ { "label": "<what the date is, e.g. invoice_date>", "value": "<date exactly as written>" } ],
  "emails": [ "<email exactly as written>" ],
  "phones": [ "<phone exactly as written>" ],
  "amounts": [ { "label": "<e.g. subtotal, tax, total>", "value": "<amount as written>", "currency": "<ISO code or null>" } ],
  "line_items": [ { "description": "...", "quantity": "...", "unit_price": "...", "amount": "..." } ]
}

Rules:
- "fields" holds the important key facts (e.g. invoice_number, vendor_name, customer_name,
  gstin, address, person_name, designation, account_number, po_number).
- Copy values as they appear in the text. Do NOT reformat dates, emails or numbers;
  a downstream rule engine normalises them.
- Never invent data. Use [] or omit keys when information is absent.
- Output valid JSON only: no markdown, no comments, no explanations."""

USER_TEMPLATE = "Extract structured data from this document text:\n\n<<<\n{text}\n>>>"


class LLMError(RuntimeError):
    pass


def parse_json_response(content: str) -> dict[str, Any]:
    """Robustly parse model output: strips ```json fences / chatter around the object."""
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I)
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end <= start:
            raise LLMError("Model did not return JSON")
        try:
            data = json.loads(content[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"Invalid JSON from model: {exc}") from exc
    if not isinstance(data, dict):
        raise LLMError("Model returned JSON that is not an object")
    return data


class LLMExtractor:
    def __init__(self, cfg: LLMConfig | None = None):
        self.cfg = cfg or LLMConfig()

    @property
    def label(self) -> str:
        return f"llm:{self.cfg.model}"

    # ------------------------------------------------------------------ #
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        return headers

    def _base(self) -> str:
        return self.cfg.base_url.rstrip("/")

    def health_check(self) -> tuple[bool, str]:
        """Returns (ok, message). For Ollama also verifies that the model is pulled."""
        try:
            if self.cfg.provider == "ollama":
                r = requests.get(f"{self._base()}/api/tags", timeout=5)
                r.raise_for_status()
                models = [m.get("name", "") for m in r.json().get("models", [])]
                wanted = self.cfg.model if ":" in self.cfg.model else f"{self.cfg.model}:latest"
                if wanted not in models and self.cfg.model not in models:
                    return False, (
                        f"Ollama is running but '{self.cfg.model}' is not pulled. "
                        f"Run: ollama pull {self.cfg.model}"
                    )
                return True, f"Ollama ready with {self.cfg.model}"
            r = requests.get(f"{self._base()}/models", headers=self._headers(), timeout=10)
            r.raise_for_status()
            return True, f"Endpoint reachable ({self.cfg.model})"
        except requests.RequestException as exc:
            return False, f"LLM endpoint not reachable: {exc.__class__.__name__}"

    # ------------------------------------------------------------------ #
    def extract(self, text: str) -> dict[str, Any]:
        text = text[: self.cfg.max_input_chars]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(text=text)},
        ]
        try:
            if self.cfg.provider == "ollama":
                payload = {
                    "model": self.cfg.model,
                    "messages": messages,
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": self.cfg.temperature, "num_ctx": self.cfg.num_ctx},
                }
                r = requests.post(f"{self._base()}/api/chat", json=payload, timeout=self.cfg.timeout)
                r.raise_for_status()
                content = r.json()["message"]["content"]
            else:
                payload = {
                    "model": self.cfg.model,
                    "messages": messages,
                    "temperature": self.cfg.temperature,
                    "response_format": {"type": "json_object"},
                }
                r = requests.post(
                    f"{self._base()}/chat/completions",
                    json=payload,
                    headers=self._headers(),
                    timeout=self.cfg.timeout,
                )
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"]
        except requests.RequestException as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"Unexpected LLM response format: {exc}") from exc
        return parse_json_response(content)
