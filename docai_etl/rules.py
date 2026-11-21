"""Rule-based validation & normalisation layer.

The LLM is good at *understanding* a document but is not trusted to *format* data.
Every value it returns (plus everything found directly in the OCR text by regex)
passes through deterministic rules:

* Dates    -> DD/MM/YYYY (ISO, numeric, textual, ordinal, 2-digit years, OCR noise)
* Emails   -> lower-case, de-obfuscated ("[at]", "(dot)"), OCR-repaired, RFC-style validated
* Phones   -> E.164 (+<country><number>)
* Amounts  -> float with 2 decimals, currency detected (Indian 1,23,456.78 & EU 1.234,56 aware)
* Line items -> quantity x unit price = amount;  sum(line items) vs total

Each decision is logged as a RuleCheck so the UI can show an audit trail.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any

from dateutil import parser as date_parser

from .config import RulesConfig

# --------------------------------------------------------------------------- #
# Audit trail
# --------------------------------------------------------------------------- #

PASS, FIXED, FAIL = "pass", "fixed", "fail"


@dataclass
class RuleCheck:
    rule: str
    field: str
    original: str
    result: str | None
    status: str          # pass | fixed | fail
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #

_MONTHS = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
    r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
_DATE_PATTERNS = [
    re.compile(r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b"),                                  # 2024-03-15
    re.compile(r"\b\d{1,2}[-/.]\d{1,2}[-/.](?:\d{4}|\d{2})\b"),                         # 15/03/2024, 15-03-24
    re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?[\s\-]+{_MONTHS}\.?,?[\s\-]+\d{{2,4}}\b", re.I),  # 15th March 2024
    re.compile(rf"\b{_MONTHS}\.?[\s\-]+\d{{1,2}}(?:st|nd|rd|th)?,?[\s\-]+\d{{2,4}}\b", re.I),  # March 15, 2024
]
_ISO_RE = re.compile(r"^\s*(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\s*$")
_ORDINAL_RE = re.compile(r"(\d{1,2})(st|nd|rd|th)\b", re.I)


def _repair_ocr_digits(value: str) -> str:
    """Fix common OCR confusions inside numeric tokens: O->0, l/I/|->1, S->5, B->8."""
    table = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "|": "1", "S": "5", "B": "8"})

    def fix(match: re.Match) -> str:
        return match.group(0).translate(table)

    # only touch tokens that are mostly digits and separators (avoid mangling month names)
    return re.sub(r"(?<![A-Za-z])[0-9OolIS|B]{1,4}(?=[-/.])|(?<=[-/.])[0-9OolIS|B]{1,4}(?![A-Za-z])", fix, value)


_OCR_DIGIT_TABLE = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "|": "1", "S": "5", "B": "8"})
_TEXTUAL_DAY_RE = re.compile(rf"\b([0-9OolISB|]{{1,2}})(st|nd|rd|th)?(?=[\s\-]+(?i:{_MONTHS})\b)")


def _repair_textual_day(value: str) -> str:
    """'Sth Nov 2024' -> '5th Nov 2024' (OCR reads 5 as S, 1 as l, 0 as O)."""

    def fix(m: re.Match) -> str:
        day = m.group(1).translate(_OCR_DIGIT_TABLE)
        return (day + (m.group(2) or "")) if day.isdigit() else m.group(0)

    return _TEXTUAL_DAY_RE.sub(fix, value)


def _strict_parse(text: str, dayfirst: bool) -> datetime | None:
    """Parse with dateutil but reject incomplete dates (dateutil silently fills a missing
    day/month/year from 'today' — unacceptable in an ETL pipeline)."""
    try:
        a = date_parser.parse(text, dayfirst=dayfirst, default=datetime(2000, 1, 1))
        b = date_parser.parse(text, dayfirst=dayfirst, default=datetime(2001, 2, 2))
    except (ValueError, OverflowError, TypeError):
        return None
    return a if a.date() == b.date() else None


def normalize_date(raw: Any, cfg: RulesConfig | None = None) -> str | None:
    """Return the date in cfg.date_output_format (default DD/MM/YYYY) or None if invalid."""
    cfg = cfg or RulesConfig()
    if raw is None:
        return None
    if isinstance(raw, (datetime, date)):
        parsed = raw if isinstance(raw, datetime) else datetime(raw.year, raw.month, raw.day)
        return parsed.strftime(cfg.date_output_format)

    text = str(raw).strip().strip(".,;:()[]")
    if not text or not re.search(r"\d|[OolISB]", text):
        return None
    text = _repair_textual_day(_repair_ocr_digits(text))

    def parse_one(candidate: str) -> datetime | None:
        iso = _ISO_RE.match(candidate)
        if iso:  # year-first is unambiguous: never let dayfirst flip month and day
            try:
                return datetime(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
            except ValueError:
                return None
        return _strict_parse(_ORDINAL_RE.sub(r"\1", candidate), cfg.dayfirst)

    parsed = parse_one(text)
    if parsed is None:  # value has extra words, e.g. "5 Nov 2024 Time: 18:42" -> use the date inside it
        for candidate in find_dates(text):
            parsed = parse_one(candidate)
            if parsed:
                break

    if parsed is None or not (cfg.min_year <= parsed.year <= cfg.max_year):
        return None
    return parsed.strftime(cfg.date_output_format)


def find_dates(text: str) -> list[str]:
    """All date-looking substrings in text, in order of appearance, de-duplicated."""
    text = _repair_textual_day(text)
    hits: list[tuple[int, str]] = []
    taken: list[tuple[int, int]] = []
    for pattern in _DATE_PATTERNS:
        for m in pattern.finditer(text):
            span = m.span()
            if any(s < span[1] and span[0] < e for s, e in taken):
                continue
            taken.append(span)
            hits.append((span[0], m.group(0)))
    seen, ordered = set(), []
    for _, value in sorted(hits):
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


# --------------------------------------------------------------------------- #
# Emails
# --------------------------------------------------------------------------- #

_EMAIL_VALID_RE = re.compile(
    r"^(?=.{6,254}$)(?=.{1,64}@)"
    r"[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*"
    r"@(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$"
)
_COMMON_TLDS = r"(?i:com|net|org|in|co|io|ai|edu|gov|info|biz|me|uk|us|de|app|dev|tech)"
_EMAIL_FIND_RE = re.compile(
    r"[A-Za-z0-9._%+\-]+\s?@\s?[A-Za-z0-9\-]+(?:[.,][A-Za-z0-9\-]+)*"
    rf"(?:\s?[.,]\s?{_COMMON_TLDS}\b|[.,][A-Za-z]{{2,}})"
)
_AT_OBFUSCATION = re.compile(r"\s*(?:\[\s*at\s*\]|\(\s*at\s*\)|\{\s*at\s*\}|\s+at\s+)\s*", re.I)
_DOT_OBFUSCATION = re.compile(r"\s*(?:\[\s*dot\s*\]|\(\s*dot\s*\)|\{\s*dot\s*\}|\s+dot\s+)\s*", re.I)


def deobfuscate_emails(text: str) -> str:
    """'john [at] acme (dot) com' -> 'john@acme.com' (only where the result looks like an email)."""

    def fix(match: re.Match) -> str:
        candidate = _DOT_OBFUSCATION.sub(".", _AT_OBFUSCATION.sub("@", match.group(0)))
        return candidate if "@" in candidate else match.group(0)

    return re.sub(
        r"[A-Za-z0-9._%+\-]+\s*(?:\[\s*at\s*\]|\(\s*at\s*\)|\{\s*at\s*\})\s*[A-Za-z0-9\-]+"
        r"(?:\s*(?:\[\s*dot\s*\]|\(\s*dot\s*\)|\{\s*dot\s*\}|\.)\s*[A-Za-z0-9\-]+)+",
        fix,
        text,
        flags=re.I,
    )


def normalize_email(raw: Any) -> str | None:
    """Return a standard, lower-case, validated email address or None."""
    if raw is None:
        return None
    s = str(raw).strip()
    s = re.sub(r"^mailto:", "", s, flags=re.I)
    s = s.strip(" \t\r\n.,;:()<>[]{}\"'")
    s = _DOT_OBFUSCATION.sub(".", _AT_OBFUSCATION.sub("@", s)) if "@" not in s else s
    s = re.sub(r"\s+", "", s)                   # OCR inserts spaces: "john @ acme .com"
    if s.count("@") != 1:
        return None
    local, domain = s.split("@")
    domain = domain.replace(",", ".")           # OCR reads '.' as ',' in domains
    domain = re.sub(r"\.{2,}", ".", domain).strip(".-")
    local = re.sub(r"\.{2,}", ".", local).strip(".")
    candidate = f"{local}@{domain}".lower()     # standard practice: case-insensitive storage
    if _EMAIL_VALID_RE.match(candidate):
        return candidate
    # value may carry extra text ("a@b.com Phone: ...") -> pull the email out of it
    embedded = find_emails(str(raw))
    if embedded and embedded[0].strip() != str(raw).strip():
        return normalize_email(embedded[0])
    return None


def find_emails(text: str) -> list[str]:
    text = deobfuscate_emails(text)
    seen, out = set(), []
    for m in _EMAIL_FIND_RE.finditer(text):
        value = m.group(0)
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


# --------------------------------------------------------------------------- #
# Phones
# --------------------------------------------------------------------------- #

_PHONE_CONTEXT = re.compile(r"\b(?:phone|ph|tel|telephone|mob|mobile|cell|contact|call|whatsapp|fax)\b", re.I)
_PHONE_FIND_RE = re.compile(r"(?<![\w/])(\+?\(?\d[\d\s\-().]{7,18}\d)(?![\w/])")


def normalize_phone(raw: Any, cfg: RulesConfig | None = None) -> str | None:
    """Return phone in E.164 format (e.g. +919876543210) or None."""
    cfg = cfg or RulesConfig()
    if raw is None:
        return None
    s = str(raw).strip()
    has_plus = s.startswith("+") or s.startswith("00")
    digits = re.sub(r"\D", "", s)
    if s.startswith("00"):
        digits = digits[2:]
    cc = re.sub(r"\D", "", cfg.default_country_code)

    if has_plus:
        number = digits
    elif len(digits) == 10:
        number = cc + digits
    elif len(digits) == 11 and digits.startswith("0"):
        number = cc + digits[1:]
    elif digits.startswith(cc) and len(digits) == len(cc) + 10:
        number = digits
    else:
        number = digits if 11 <= len(digits) <= 15 else ""

    if not (8 <= len(number) <= 15):
        return None
    return "+" + number


def find_phones(text: str) -> list[str]:
    """Phone numbers on lines that mention phone/tel/mobile..., or written with a leading '+'."""
    out, seen = [], set()
    for line in text.splitlines():
        context = bool(_PHONE_CONTEXT.search(line))
        for m in _PHONE_FIND_RE.finditer(line):
            value = m.group(1).strip()
            digits = re.sub(r"\D", "", value)
            if not (10 <= len(digits) <= 15):
                continue
            if (context or value.startswith("+")) and value not in seen:
                seen.add(value)
                out.append(value)
    return out


# --------------------------------------------------------------------------- #
# Amounts
# --------------------------------------------------------------------------- #

_CURRENCY_MAP = {
    "₹": "INR", "rs": "INR", "inr": "INR", "rs.": "INR",
    "$": "USD", "usd": "USD", "€": "EUR", "eur": "EUR", "£": "GBP", "gbp": "GBP",
}


def normalize_amount(raw: Any) -> tuple[float | None, str | None]:
    """'₹ 1,23,456.50' -> (123456.5, 'INR');  '1.234,56 €' -> (1234.56, 'EUR')."""
    if raw is None:
        return None, None
    if isinstance(raw, (int, float)):
        return round(float(raw), 2), None
    s = str(raw).strip()
    currency = None
    lowered = s.lower()
    for token, code in _CURRENCY_MAP.items():
        if token in lowered:
            currency = code
            break
    negative = s.startswith("-") or (s.startswith("(") and s.endswith(")"))
    token = re.search(r"\d[\d.,\s]*\d|\d", s)   # "Rs. 1,23,456.50 /-" -> "1,23,456.50"
    if not token:
        return None, currency
    num = re.sub(r"\s", "", token.group(0)).strip(".,")

    if "," in num and "." in num:
        if num.rfind(",") > num.rfind("."):     # EU style 1.234,56
            num = num.replace(".", "").replace(",", ".")
        else:                                   # 1,234.56 / 1,23,456.78
            num = num.replace(",", "")
    elif "," in num:
        parts = num.split(",")
        num = num.replace(",", ".") if len(parts) == 2 and len(parts[1]) == 2 else num.replace(",", "")
    elif num.count(".") > 1:                    # 1.234.567
        num = num.replace(".", "")
    try:
        value = round(float(num), 2)
    except ValueError:
        return None, currency
    return (-value if negative else value), currency


# --------------------------------------------------------------------------- #
# Rule engine
# --------------------------------------------------------------------------- #

def _key_re(words: str) -> re.Pattern:
    return re.compile(rf"(?:^|_)(?:{words})(?:$|_)", re.I)


_DATE_KEY = _key_re(r"date|dob|birth|issued|expiry|expires|valid|due|dated|period|joining")
_EMAIL_KEY = re.compile(r"e_?mail", re.I)
_PHONE_KEY = _key_re(r"phone|mobile|mob|tel|telephone|cell|fax|contact_number|contact_no|whatsapp")
_AMOUNT_KEY = _key_re(
    r"amount|total|grand_total|price|subtotal|sub_total|tax|gst|cgst|sgst|igst|vat|balance|fee|cost|salary|ctc|discount"
)
_GSTIN_KEY = _key_re(r"gstin|gst_no|gst_number|gstin_no")
_GSTIN_RE = re.compile(r"^\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")


def normalize_gstin(raw: Any) -> str | None:
    """Indian GSTIN: 2-digit state code + PAN + entity + 'Z' + checksum. Repairs O/0 in the state code."""
    s = re.sub(r"[\s\-]", "", str(raw or "")).upper()
    if len(s) != 15:
        return None
    s = s[:2].translate(_OCR_DIGIT_TABLE) + s[2:]
    return s if _GSTIN_RE.match(s) else None


class RuleEngine:
    def __init__(self, cfg: RulesConfig | None = None):
        self.cfg = cfg or RulesConfig()

    # -- individual rules ---------------------------------------------------
    def _date_check(self, field: str, raw: Any) -> RuleCheck:
        norm = normalize_date(raw, self.cfg)
        raw_s = str(raw)
        if norm is None:
            return RuleCheck("date_format", field, raw_s, None, FAIL, "Not a valid calendar date")
        status = PASS if raw_s.strip() == norm else FIXED
        msg = "Already DD/MM/YYYY" if status == PASS else "Converted to DD/MM/YYYY"
        return RuleCheck("date_format", field, raw_s, norm, status, msg)

    def _email_check(self, field: str, raw: Any) -> RuleCheck:
        norm = normalize_email(raw)
        raw_s = str(raw)
        if norm is None:
            return RuleCheck("email_format", field, raw_s, None, FAIL, "Not a valid email address")
        status = PASS if raw_s.strip() == norm else FIXED
        msg = "Valid standard email" if status == PASS else "Cleaned to standard email format"
        return RuleCheck("email_format", field, raw_s, norm, status, msg)

    def _phone_check(self, field: str, raw: Any) -> RuleCheck:
        norm = normalize_phone(raw, self.cfg)
        raw_s = str(raw)
        if norm is None:
            return RuleCheck("phone_format", field, raw_s, None, FAIL, "Not a valid phone number")
        status = PASS if raw_s.strip() == norm else FIXED
        return RuleCheck("phone_format", field, raw_s, norm, status, "E.164 format")

    def _amount_check(self, field: str, raw: Any) -> tuple[RuleCheck, str | None]:
        value, currency = normalize_amount(raw)
        raw_s = str(raw)
        if value is None:
            return RuleCheck("amount_format", field, raw_s, None, FAIL, "Not a numeric amount"), None
        result = f"{value:.2f}"
        status = PASS if raw_s.strip() == result else FIXED
        return RuleCheck("amount_format", field, raw_s, result, status, "Numeric, 2 decimals"), currency

    # -- orchestration ------------------------------------------------------
    def apply(self, extracted: dict[str, Any], raw_text: str) -> tuple[dict[str, Any], list[RuleCheck]]:
        checks: list[RuleCheck] = []
        record: dict[str, Any] = {
            "document_type": (extracted.get("document_type") or "other"),
            "summary": extracted.get("summary") or "",
            "fields": {},
            "dates": [],
            "emails": [],
            "phones": [],
            "amounts": [],
            "line_items": [],
        }
        detected_currency: str | None = extracted.get("currency")

        # 1. Generic key/value fields: apply the rule that matches the key
        for key, value in (extracted.get("fields") or {}).items():
            if value in (None, "", [], {}):
                continue
            if isinstance(value, (dict, list)):
                record["fields"][key] = value
                continue
            if _EMAIL_KEY.search(key):
                chk = self._email_check(key, value)
            elif _DATE_KEY.search(key) and normalize_date(value, self.cfg):
                chk = self._date_check(key, value)
            elif _PHONE_KEY.search(key):
                chk = self._phone_check(key, value)
            elif _GSTIN_KEY.search(key):
                norm = normalize_gstin(value)
                status = FAIL if norm is None else (PASS if norm == str(value).strip() else FIXED)
                chk = RuleCheck("gstin_format", key, str(value), norm, status,
                                "Valid GSTIN" if norm else "Does not match GSTIN pattern")
            elif _AMOUNT_KEY.search(key) and re.search(r"\d", str(value)):
                chk, cur = self._amount_check(key, value)
                detected_currency = detected_currency or cur
            elif "@" in str(value) and normalize_email(value):   # email under any key name
                chk = self._email_check(key, value)
            else:
                record["fields"][key] = str(value).strip()
                continue
            checks.append(chk)
            record["fields"][key] = chk.result if chk.result is not None else str(value).strip()

        # 2. Dates: LLM-labelled dates + any date in the raw text the LLM missed
        seen_dates: set[str] = set()
        for item in extracted.get("dates") or []:
            label, value = self._label_value(item, "date")
            chk = self._date_check(label, value)
            checks.append(chk)
            if chk.result and chk.result not in seen_dates:
                seen_dates.add(chk.result)
                record["dates"].append({"label": label, "value": chk.result, "original": str(value)})
        for raw in find_dates(raw_text):
            norm = normalize_date(raw, self.cfg)
            if norm and norm not in seen_dates:
                seen_dates.add(norm)
                chk = self._date_check("detected_in_text", raw)
                checks.append(chk)
                record["dates"].append({"label": "detected_in_text", "value": norm, "original": raw})

        # 3. Emails: union of LLM + regex, normalised & de-duplicated
        seen_emails: set[str] = set()
        for raw in list(extracted.get("emails") or []) + find_emails(raw_text):
            chk = self._email_check("email", raw)
            if chk.result in seen_emails:
                continue
            checks.append(chk)
            if chk.result:
                seen_emails.add(chk.result)
                record["emails"].append(chk.result)

        # 4. Phones
        seen_phones: set[str] = set()
        for raw in list(extracted.get("phones") or []) + find_phones(raw_text):
            chk = self._phone_check("phone", raw)
            if chk.result in seen_phones:
                continue
            checks.append(chk)
            if chk.result:
                seen_phones.add(chk.result)
                record["phones"].append(chk.result)

        # 5. Amounts
        for item in extracted.get("amounts") or []:
            label, value = self._label_value(item, "amount")
            chk, cur = self._amount_check(label, value)
            checks.append(chk)
            currency = (item.get("currency") if isinstance(item, dict) else None) or cur or detected_currency
            detected_currency = detected_currency or currency
            if chk.result is not None:
                record["amounts"].append({"label": label, "value": float(chk.result), "currency": currency})

        # 6. Line items + arithmetic checks
        line_total = 0.0
        for idx, item in enumerate(extracted.get("line_items") or [], start=1):
            if not isinstance(item, dict):
                continue
            qty, _ = normalize_amount(item.get("quantity"))
            unit, _ = normalize_amount(item.get("unit_price"))
            amt, _ = normalize_amount(item.get("amount"))
            clean = {
                "description": str(item.get("description") or "").strip(),
                "quantity": qty,
                "unit_price": unit,
                "amount": amt,
            }
            if qty is not None and unit is not None and amt is not None:
                expected = round(qty * unit, 2)
                ok = abs(expected - amt) <= max(self.cfg.amount_tolerance, 0.005 * amt)
                checks.append(
                    RuleCheck(
                        "line_item_math", f"line_item[{idx}]",
                        f"{qty:g} x {unit:.2f} = {amt:.2f}", f"{expected:.2f}",
                        PASS if ok else FAIL,
                        "Quantity x unit price matches amount" if ok else f"Expected {expected:.2f}",
                    )
                )
            if amt is not None:
                line_total += amt
            record["line_items"].append(clean)

        subtotal = self._find_amount(record, ("subtotal", "sub_total", "sub total"))
        if record["line_items"] and subtotal is not None:
            ok = abs(round(line_total, 2) - subtotal) <= self.cfg.amount_tolerance
            checks.append(
                RuleCheck(
                    "subtotal_reconciliation", "subtotal", f"{subtotal:.2f}", f"{line_total:.2f}",
                    PASS if ok else FAIL,
                    "Line items sum to subtotal" if ok else "Line items do not sum to subtotal",
                )
            )

        total = self._find_amount(record, ("total", "grand_total", "grand total", "total_amount", "amount_due"))
        if subtotal is not None and total is not None:
            taxes = sum(a["value"] for a in record["amounts"]
                        if re.search(r"tax|gst|vat|cess|shipping|service", a["label"]))
            discount = sum(a["value"] for a in record["amounts"] if "discount" in a["label"])
            expected = round(subtotal + taxes - abs(discount), 2)
            ok = abs(expected - total) <= self.cfg.amount_tolerance
            checks.append(
                RuleCheck(
                    "total_reconciliation", "total", f"{total:.2f}", f"{expected:.2f}",
                    PASS if ok else FAIL,
                    "Subtotal + taxes - discounts = total" if ok else "Total does not reconcile",
                )
            )

        record["currency"] = detected_currency
        # the same raw value is often seen twice (LLM field + regex scan): keep one audit row
        unique: dict[tuple[str, str], RuleCheck] = {}
        for chk in checks:
            unique.setdefault((chk.rule, chk.original.strip()), chk)
        return record, list(unique.values())

    @staticmethod
    def _label_value(item: Any, default_label: str) -> tuple[str, Any]:
        if isinstance(item, dict):
            label = str(item.get("label") or default_label).strip().lower().replace(" ", "_")
            return label, item.get("value")
        return default_label, item

    @staticmethod
    def _find_amount(record: dict[str, Any], names: tuple[str, ...]) -> float | None:
        for amount in record["amounts"]:
            if amount["label"].replace("_", " ") in {n.replace("_", " ") for n in names}:
                return amount["value"]
        for key, value in record["fields"].items():
            if key.lower() in names:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None
        return None


def quality_score(checks: list[RuleCheck]) -> float:
    """Share of rule checks that passed or were auto-fixed (0-100)."""
    if not checks:
        return 100.0
    good = sum(1 for c in checks if c.status in (PASS, FIXED))
    return round(100.0 * good / len(checks), 1)
