"""Extract stage: turn images and PDFs into text.

* Native (digital) PDF pages -> text layer via pdfplumber (fast, lossless).
* Scanned PDF pages and images -> OpenCV preprocessing + Tesseract OCR.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytesseract
from PIL import Image, ImageOps

from .config import OCRConfig, PreprocessConfig
from .preprocessing import preprocess

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | PDF_EXTENSIONS


class UnsupportedFileError(ValueError):
    pass


class OCRUnavailableError(RuntimeError):
    pass


@dataclass
class OCRPage:
    page_number: int
    text: str
    method: str                                  # "native-pdf" | "tesseract"
    confidence: float | None = None              # mean word confidence (0-100) for OCR pages
    word_count: int = 0
    preprocessing_steps: list[str] = field(default_factory=list)
    original_image: Image.Image | None = None
    processed_image: np.ndarray | None = None


def configure_tesseract(cfg: OCRConfig) -> None:
    if cfg.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = cfg.tesseract_cmd


def tesseract_available(cfg: OCRConfig | None = None) -> bool:
    cfg = cfg or OCRConfig()
    configure_tesseract(cfg)
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _data_to_text(data: dict) -> tuple[str, float | None, int]:
    """Rebuild layout-preserving text from image_to_data output (one Tesseract call
    gives both the text and per-word confidences)."""
    lines: dict[tuple[int, int, int], list[str]] = {}
    confidences: list[float] = []
    for i, word in enumerate(data["text"]):
        word = (word or "").strip()
        if not word:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1
        if conf >= 0:
            confidences.append(conf)

    text_lines: list[str] = []
    previous_block = None
    for (block, par, _line), words in sorted(lines.items()):
        if previous_block is not None and (block, par) != previous_block:
            text_lines.append("")  # paragraph break
        text_lines.append(" ".join(words))
        previous_block = (block, par)

    mean_conf = round(float(np.mean(confidences)), 1) if confidences else None
    return "\n".join(text_lines).strip(), mean_conf, len(confidences)


def ocr_image(image: Image.Image, ocr_cfg: OCRConfig, pre_cfg: PreprocessConfig, page_number: int = 1) -> OCRPage:
    configure_tesseract(ocr_cfg)
    image = ImageOps.exif_transpose(image)  # respect phone-camera orientation
    pre = preprocess(image, pre_cfg)
    try:
        data = pytesseract.image_to_data(
            pre.image,
            lang=ocr_cfg.lang,
            config=f"--oem {ocr_cfg.oem} --psm {ocr_cfg.psm}",
            output_type=pytesseract.Output.DICT,
        )
    except pytesseract.TesseractNotFoundError as exc:
        raise OCRUnavailableError(
            "Tesseract OCR is not installed or not on PATH. Install it and/or set TESSERACT_CMD."
        ) from exc
    text, conf, words = _data_to_text(data)
    return OCRPage(
        page_number=page_number,
        text=text,
        method="tesseract",
        confidence=conf,
        word_count=words,
        preprocessing_steps=pre.steps,
        original_image=image.convert("RGB"),
        processed_image=pre.image,
    )


def _extract_pdf(file_bytes: bytes, ocr_cfg: OCRConfig, pre_cfg: PreprocessConfig) -> list[OCRPage]:
    import pdfplumber  # local import keeps module import light

    pages: list[OCRPage] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for idx, page in enumerate(pdf.pages[: ocr_cfg.max_pages], start=1):
            native_text = (page.extract_text() or "").strip()
            if len(native_text) >= ocr_cfg.min_native_chars:
                pages.append(
                    OCRPage(
                        page_number=idx,
                        text=native_text,
                        method="native-pdf",
                        confidence=100.0,
                        word_count=len(native_text.split()),
                    )
                )
                continue
            # Scanned page: render with pdfium (bundled with pdfplumber, no Poppler needed)
            rendered = page.to_image(resolution=ocr_cfg.pdf_dpi).original
            pages.append(ocr_image(rendered, ocr_cfg, pre_cfg, page_number=idx))
    return pages


def extract_document(
    file_bytes: bytes,
    filename: str,
    ocr_cfg: OCRConfig | None = None,
    pre_cfg: PreprocessConfig | None = None,
) -> list[OCRPage]:
    ocr_cfg = ocr_cfg or OCRConfig()
    pre_cfg = pre_cfg or PreprocessConfig()
    ext = Path(filename).suffix.lower()

    if ext in PDF_EXTENSIONS:
        return _extract_pdf(file_bytes, ocr_cfg, pre_cfg)

    if ext in IMAGE_EXTENSIONS:
        image = Image.open(io.BytesIO(file_bytes))
        pages: list[OCRPage] = []
        n_frames = getattr(image, "n_frames", 1)  # multi-page TIFF support
        for frame in range(min(n_frames, ocr_cfg.max_pages)):
            image.seek(frame)
            pages.append(ocr_image(image.copy(), ocr_cfg, pre_cfg, page_number=frame + 1))
        return pages

    raise UnsupportedFileError(
        f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )
