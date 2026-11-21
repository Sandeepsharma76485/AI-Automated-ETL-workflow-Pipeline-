"""Generate realistic sample documents for demos and tests.

    python scripts/generate_samples.py

Creates in samples/:
  invoice_scanned.jpg   - skewed, noisy "phone scan" of a tax invoice (mixed date formats, messy emails)
  receipt_scanned.pdf   - image-only PDF (forces the OCR path)
  offer_letter.pdf      - digital PDF with a text layer (native extraction path)
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "samples"

FONT_CANDIDATES = [
    "DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = (["DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "arialbd.ttf", "C:/Windows/Fonts/arialbd.ttf"] if bold else []) + FONT_CANDIDATES
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def scanify(img: Image.Image, angle: float, noise: int = 18, seed: int = 7) -> Image.Image:
    """Make a clean render look like a phone scan: rotation, blur, uneven light, sensor noise."""
    rng = np.random.default_rng(seed)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(235, 233, 228))
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    arr = np.asarray(img).astype(np.int16)
    h, w = arr.shape[:2]
    gradient = np.linspace(-18, 10, w)[None, :, None]       # uneven lighting
    arr = arr + gradient + rng.normal(0, noise, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def invoice() -> Image.Image:
    W, H = 1700, 2200
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f_title, f_b, f = font(64, True), font(34, True), font(32)

    d.text((100, 90), "ACME CLOUD SOLUTIONS PVT LTD", font=f_title, fill="black")
    d.text((100, 180), "42 MI Road, Jaipur, Rajasthan 302001", font=f, fill="black")
    d.text((100, 225), "GSTIN: 08ABCDE1234F1Z5", font=f, fill="black")
    d.text((100, 270), "Email: Billing@AcmeCloud.CO.IN    Phone: +91 98290 12345", font=f, fill="black")
    d.text((1220, 90), "TAX INVOICE", font=f_b, fill="black")

    y = 380
    for line in [
        "Invoice No: INV-2024-0457",
        "Invoice Date: 2024-03-15",
        "Due Date: April 14, 2024",
        "PO Number: PO-88231",
    ]:
        d.text((100, y), line, font=f, fill="black")
        y += 48

    y += 30
    d.text((100, y), "Bill To:", font=f_b, fill="black"); y += 50
    for line in [
        "Customer Name: Rohan Mehta",
        "Company: Pinkcity Retail LLP",
        "Customer Email: rohan.mehta [at] pinkcityretail [dot] com",
        "Mobile: 98765 43210",
    ]:
        d.text((100, y), line, font=f, fill="black")
        y += 48

    y += 40
    d.line((100, y, 1600, y), fill="black", width=3); y += 15
    d.text((100, y), "Description", font=f_b, fill="black")
    d.text((900, y), "Qty", font=f_b, fill="black")
    d.text((1080, y), "Rate", font=f_b, fill="black")
    d.text((1350, y), "Amount", font=f_b, fill="black")
    y += 55
    d.line((100, y, 1600, y), fill="black", width=2); y += 20
    items = [
        ("Cloud Hosting Plan", 12, 2500.00),
        ("SSL Certificate", 2, 1800.00),
        ("Data Migration Service", 1, 15000.00),
        ("Technical Support Hours", 10, 950.00),
    ]
    subtotal = 0.0
    for desc, qty, rate in items:
        amt = qty * rate
        subtotal += amt
        d.text((100, y), desc, font=f, fill="black")
        d.text((900, y), str(qty), font=f, fill="black")
        d.text((1080, y), f"{rate:,.2f}", font=f, fill="black")
        d.text((1350, y), f"{amt:,.2f}", font=f, fill="black")
        y += 52
    d.line((100, y, 1600, y), fill="black", width=2); y += 30
    gst = round(subtotal * 0.18, 2)
    for label, value in [("Subtotal", subtotal), ("GST 18%", gst), ("Total", subtotal + gst)]:
        d.text((1000, y), f"{label}:", font=f_b, fill="black")
        d.text((1350, y), f"Rs. {value:,.2f}", font=f, fill="black")
        y += 52

    y += 60
    d.text((100, y), "Payment received on 05.04.2024 via NEFT. Queries: accounts@acmecloud.co.in", font=f, fill="black")
    y += 48
    d.text((100, y), "Thank you for your business!", font=f, fill="black")
    return scanify(img, angle=2.2, seed=11)


def receipt() -> Image.Image:
    W, H = 1000, 1500
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f_b, f = font(40, True), font(30)
    d.text((240, 60), "CHAI POINT CAFE", font=f_b, fill="black")
    d.text((180, 120), "C-Scheme, Jaipur | Tel: 0141-2365478", font=f, fill="black")
    y = 220
    lines = [
        "Receipt No: R-77812",
        "Date: 5th Nov 2024   Time: 18:42",
        "Cashier: Priya",
        "",
        "Masala Chai          2    60.00   120.00",
        "Paneer Sandwich      1   180.00   180.00",
        "Brownie              2    95.00   190.00",
        "",
        "Subtotal:                     490.00",
        "CGST 2.5%:                     12.25",
        "SGST 2.5%:                     12.25",
        "Total:                    Rs. 514.50",
        "",
        "Feedback: HELLO@ChaiPointCafe .com",
        "Visit again before 31/12/2024!",
    ]
    for line in lines:
        d.text((80, y), line, font=f, fill="black")
        y += 52
    return scanify(img, angle=-1.6, noise=14, seed=5)


def offer_letter(path: Path) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=A4)
    w, h = A4
    y = h - 72
    lines = [
        ("Helvetica-Bold", 16, "NovaTech Analytics Pvt Ltd"),
        ("Helvetica", 11, "Tower B, Cyber City, Gurugram 122002  |  hr@NovaTechAnalytics.com"),
        ("Helvetica", 11, ""),
        ("Helvetica", 11, "Date: 21 October 2024"),
        ("Helvetica", 11, "Ref: NTA/HR/2024/1187"),
        ("Helvetica", 11, ""),
        ("Helvetica-Bold", 12, "Subject: Offer of Employment - Data Scientist"),
        ("Helvetica", 11, ""),
        ("Helvetica", 11, "Dear Ms. Ananya Rao,"),
        ("Helvetica", 11, "We are pleased to offer you the position of Data Scientist (GenAI) at NovaTech Analytics."),
        ("Helvetica", 11, "Your date of joining will be 2024/12/02 and your annual CTC will be INR 18,50,000."),
        ("Helvetica", 11, "Please confirm your acceptance by 15-11-2024 by replying to this letter."),
        ("Helvetica", 11, "Candidate Email: Ananya.Rao@Gmail.com     Phone: +91-99887-66554"),
        ("Helvetica", 11, ""),
        ("Helvetica", 11, "Sincerely,"),
        ("Helvetica", 11, "Vikram Singh, Head of Talent Acquisition"),
    ]
    for face, size, text in lines:
        c.setFont(face, size)
        c.drawString(72, y, text)
        y -= 22
    c.showPage()
    c.save()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    random.seed(0)
    invoice().save(OUT / "invoice_scanned.jpg", quality=82)
    receipt().convert("RGB").save(OUT / "receipt_scanned.pdf", "PDF", resolution=200)
    try:
        offer_letter(OUT / "offer_letter.pdf")
    except ImportError:
        print("reportlab not installed - skipping offer_letter.pdf (pip install reportlab)")
    print(f"Samples written to {OUT}")


if __name__ == "__main__":
    main()
