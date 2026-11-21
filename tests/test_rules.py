import pytest

from docai_etl.config import RulesConfig
from docai_etl.rules import (
    FAIL,
    FIXED,
    PASS,
    RuleEngine,
    find_dates,
    find_emails,
    find_phones,
    normalize_amount,
    normalize_date,
    normalize_email,
    normalize_gstin,
    normalize_phone,
    quality_score,
)


# ----------------------------- dates -> DD/MM/YYYY ----------------------------- #
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2024-03-15", "15/03/2024"),          # ISO must NOT be flipped by dayfirst
        ("2024/12/02", "02/12/2024"),
        ("15/03/2024", "15/03/2024"),
        ("05.11.2024", "05/11/2024"),
        ("5-11-24", "05/11/2024"),
        ("April 14, 2024", "14/04/2024"),
        ("Mar 3 2023", "03/03/2023"),
        ("21 October 2024", "21/10/2024"),
        ("5th Nov 2024", "05/11/2024"),
        ("1st January 2025", "01/01/2025"),
        ("Sth Nov 2024", "05/11/2024"),        # OCR read 5 as S
        ("12/O3/2024", "12/03/2024"),          # OCR read 0 as O
        ("Date: 5 Nov 2024 Time: 18:42", "05/11/2024"),
    ],
)
def test_normalize_date(raw, expected):
    assert normalize_date(raw) == expected


@pytest.mark.parametrize("raw", ["31/02/2024", "not a date", "", None, "Nov 2024", "18:42", "1850-01-01"])
def test_normalize_date_rejects_invalid_or_incomplete(raw):
    assert normalize_date(raw) is None


def test_month_first_mode():
    cfg = RulesConfig(dayfirst=False)
    assert normalize_date("03/12/2024", cfg) == "12/03/2024"


def test_find_dates():
    text = "Invoice Date: 2024-03-15  Due: April 14, 2024  Paid 05.04.2024"
    assert find_dates(text) == ["2024-03-15", "April 14, 2024", "05.04.2024"]


# ----------------------------- emails ----------------------------- #
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("john.doe@example.com", "john.doe@example.com"),
        ("John.Doe@Example.COM", "john.doe@example.com"),
        ("  <billing@acme.co.in>; ", "billing@acme.co.in"),
        ("mailto:HR@Company.com", "hr@company.com"),
        ("john @ example .com", "john@example.com"),
        ("john@example,com", "john@example.com"),
        ("john [at] example [dot] com", "john@example.com"),
        ("john(at)example(dot)co(dot)in", "john@example.co.in"),
        ("a@b.com Phone: +91 98290 12345", "a@b.com"),
    ],
)
def test_normalize_email(raw, expected):
    assert normalize_email(raw) == expected


@pytest.mark.parametrize("raw", ["john.example.com", "john@", "@example.com", "john@@x.com", "john@example", "j@x.c", None])
def test_normalize_email_invalid(raw):
    assert normalize_email(raw) is None


def test_find_emails_handles_obfuscation_and_ocr_spaces():
    text = "Mail rohan [at] pinkcity [dot] com or HELLO@Cafe .com. Thanks"
    found = [normalize_email(e) for e in find_emails(text)]
    assert found == ["rohan@pinkcity.com", "hello@cafe.com"]


# ----------------------------- phones / amounts / GSTIN ----------------------------- #
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("98765 43210", "+919876543210"),
        ("+91-99887-66554", "+919988766554"),
        ("09876543210", "+919876543210"),
        ("919876543210", "+919876543210"),
        ("+1 (415) 555-0132", "+14155550132"),
        ("0141-2365478", "+911412365478"),
    ],
)
def test_normalize_phone(raw, expected):
    assert normalize_phone(raw) == expected


def test_find_phones_needs_context():
    text = "Invoice No 1234567890\nMobile: 98765 43210\nWhatsApp +91 90000 11111"
    assert find_phones(text) == ["98765 43210", "+91 90000 11111"]


@pytest.mark.parametrize(
    "raw, value, currency",
    [
        ("Rs. 1,23,456.50", 123456.5, "INR"),
        ("₹514.50", 514.5, "INR"),
        ("$1,234.56", 1234.56, "USD"),
        ("1.234,56 €", 1234.56, "EUR"),
        ("INR 18,50,000", 1850000.0, "INR"),
        ("12", 12.0, None),
        (99.999, 100.0, None),
    ],
)
def test_normalize_amount(raw, value, currency):
    assert normalize_amount(raw) == (value, currency)


def test_gstin_repairs_ocr_state_code():
    assert normalize_gstin("O8ABCDE1234F1Z5") == "08ABCDE1234F1Z5"
    assert normalize_gstin("08ABCDE1234F1Z") is None


# ----------------------------- rule engine ----------------------------- #
def test_rule_engine_end_to_end():
    extracted = {
        "document_type": "invoice",
        "fields": {"invoice_number": "INV-1", "invoice_date": "2024-03-15", "vendor_email": "Sales@Acme.COM",
                   "gstin": "O8ABCDE1234F1Z5"},
        "dates": [{"label": "Due Date", "value": "April 14, 2024"}],
        "emails": ["sales@acme.com"],
        "amounts": [
            {"label": "subtotal", "value": "Rs. 1,000.00"},
            {"label": "gst", "value": "180.00"},
            {"label": "total", "value": "Rs. 1,180.00"},
        ],
        "line_items": [
            {"description": "A", "quantity": "2", "unit_price": "250.00", "amount": "500.00"},
            {"description": "B", "quantity": "1", "unit_price": "500.00", "amount": "500.00"},
        ],
    }
    record, checks = RuleEngine().apply(extracted, raw_text="Paid on 20.04.2024")
    assert record["fields"]["invoice_date"] == "15/03/2024"
    assert record["fields"]["vendor_email"] == "sales@acme.com"
    assert record["fields"]["gstin"] == "08ABCDE1234F1Z5"
    assert [d["value"] for d in record["dates"]] == ["14/04/2024", "20/04/2024"]
    assert record["dates"][0]["label"] == "due_date"
    assert record["emails"] == ["sales@acme.com"]          # de-duplicated
    assert record["currency"] == "INR"
    rules = {c.rule: c.status for c in checks}
    assert rules["line_item_math"] == PASS
    assert rules["subtotal_reconciliation"] == PASS
    assert rules["total_reconciliation"] == PASS
    assert quality_score(checks) == 100.0


def test_rule_engine_flags_bad_math_and_bad_values():
    extracted = {
        "fields": {"contact_email": "not-an-email", "date": "31/02/2024"},
        "line_items": [{"description": "X", "quantity": "3", "unit_price": "10", "amount": "40"}],
    }
    _, checks = RuleEngine().apply(extracted, raw_text="")
    statuses = {(c.rule, c.status) for c in checks}
    assert ("email_format", FAIL) in statuses
    assert ("line_item_math", FAIL) in statuses
    assert quality_score(checks) < 100


def test_fixed_status_when_value_changes():
    _, checks = RuleEngine().apply({"dates": ["2024-01-05"]}, raw_text="")
    assert checks[0].status == FIXED and checks[0].result == "05/01/2024"
