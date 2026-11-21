"""Deterministic fallback extractor used when no LLM is available.

Produces the same JSON shape as the LLM so the rule engine and UI work unchanged.
"""

from __future__ import annotations

import re
from typing import Any

from .rules import find_dates, find_emails, find_phones

_DOC_KEYWORDS = {
    "invoice": ["tax invoice", "invoice no", "invoice #", "invoice number", "invoice"],
    "receipt": ["receipt", "cash memo", "thank you for shopping", "txn id"],
    "purchase_order": ["purchase order", "p.o. no", "po number"],
    "bank_statement": ["account statement", "opening balance", "closing balance"],
    "resume": ["curriculum vitae", "resume", "work experience", "education", "skills"],
    "letter": ["dear ", "sincerely", "regards", "offer letter", "subject:"],
    "id_card": ["date of birth", "dob", "aadhaar", "pan", "identity card"],
}

_KEY_CONNECTORS = {"of", "no", "no.", "to", "the", "#", "&", "id", "on"}


def split_key_values(line: str) -> list[tuple[str, str]]:
    """Split 'Email: a@b.com Phone: +91 98290 12345' into [('Email', ...), ('Phone', ...)].

    For each ':' we walk backwards over up to 4 words that look like a label
    (Capitalised / ALL-CAPS / connectors such as 'of', 'No'); a word with digits,
    '@' or a trailing '.' ends the label.
    """
    colons = [m.start() for m in re.finditer(r":(?=\s|$)", line)]
    keys: list[tuple[int, int, str]] = []      # (key_start, colon_pos, key)
    for pos in colons:
        words = line[:pos].split(" ")
        label: list[str] = []
        for word in reversed(words):
            if not word or len(label) >= 4:
                break
            if re.search(r"[\d@]", word) or (word.endswith(".") and word.lower() not in _KEY_CONNECTORS):
                break
            if word[0].isupper() or word.lower() in _KEY_CONNECTORS:
                label.insert(0, word)
            else:
                break
        while label and label[0].lower() in _KEY_CONNECTORS:
            label.pop(0)
        if label:
            key = " ".join(label)
            keys.append((pos - len(key), pos, key))
    pairs = []
    for i, (_start, colon, key) in enumerate(keys):
        end = keys[i + 1][0] if i + 1 < len(keys) else len(line)
        value = line[colon + 1 : end].strip()
        if value:
            pairs.append((key, value))
    return pairs

_AMOUNT_LABELS = re.compile(r"\b(sub\s?total|total|grand total|amount due|balance due|tax|gst|cgst|sgst|igst|vat|discount)\b", re.I)
_MONEY_RE = re.compile(r"(?:₹|Rs\.?|INR|\$|USD|€|EUR|£)?\s?-?\d[\d,]*(?:\.\d{1,2})?")
_LINE_ITEM_RE = re.compile(
    r"^\s*(?P<desc>[A-Za-z][A-Za-z0-9 &()/.,'\-]{2,}?)\s+(?P<qty>\d+(?:\.\d+)?)\s+"
    r"(?P<unit>[\d,]+\.\d{2})\s+(?P<amt>[\d,]+\.\d{2})\s*$"
)


def _snake(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def guess_document_type(text: str) -> str:
    lowered = text.lower()
    scores = {doc: sum(lowered.count(k) for k in kws) for doc, kws in _DOC_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "other"


def heuristic_extract(text: str) -> dict[str, Any]:
    fields: dict[str, str] = {}
    amounts: list[dict[str, Any]] = []
    line_items: list[dict[str, Any]] = []
    dates = [{"label": "date", "value": d} for d in find_dates(text)]

    for line in text.splitlines():
        item = _LINE_ITEM_RE.match(line)
        if item and not _AMOUNT_LABELS.search(item.group("desc")):
            line_items.append(
                {
                    "description": item.group("desc").strip(),
                    "quantity": item.group("qty"),
                    "unit_price": item.group("unit"),
                    "amount": item.group("amt"),
                }
            )
            continue

        label_match = _AMOUNT_LABELS.search(line)
        if label_match:
            money = [m.group(0).strip() for m in _MONEY_RE.finditer(line[label_match.end():]) if re.search(r"\d", m.group(0))]
            if money:
                amounts.append({"label": _snake(label_match.group(1)), "value": money[-1], "currency": None})
                continue

        for raw_key, value in split_key_values(line):
            key = _snake(raw_key)
            if key and key not in fields and len(value) < 200:
                fields[key] = value

    # label dates using the key/value pairs when possible
    for key, value in fields.items():
        if "date" in key or key in {"dob", "due", "dated"}:
            for d in dates:
                if d["value"] in value and d["label"] == "date":
                    d["label"] = key
                    break

    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return {
        "document_type": guess_document_type(text),
        "summary": f"Document beginning with '{first_line[:60]}'" if first_line else "",
        "fields": fields,
        "dates": dates,
        "emails": find_emails(text),
        "phones": find_phones(text),
        "amounts": amounts,
        "line_items": line_items,
    }
