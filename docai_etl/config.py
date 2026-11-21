"""Central configuration. Every value can be overridden with environment variables
(see .env.example) or at runtime from the Streamlit sidebar."""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def detect_tesseract_cmd() -> str | None:
    """Find the Tesseract binary: env var -> PATH -> common Windows install locations."""
    env_cmd = os.getenv("TESSERACT_CMD")
    if env_cmd:
        return env_cmd
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path
    if platform.system() == "Windows":
        candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
    return None


@dataclass
class PreprocessConfig:
    grayscale: bool = True
    upscale: bool = True          # upscale small images so text height suits Tesseract
    min_height: int = 1600
    denoise: bool = True
    deskew: bool = True
    threshold: bool = True
    threshold_method: str = "otsu"  # "otsu" (clean scans) | "adaptive" (phone photos, uneven light)


@dataclass
class OCRConfig:
    lang: str = os.getenv("OCR_LANG", "eng")
    psm: int = int(os.getenv("OCR_PSM", "3"))   # 3 = fully automatic page segmentation
    oem: int = 3                                  # 3 = default (LSTM)
    pdf_dpi: int = 300
    min_native_chars: int = 30   # below this a PDF page is treated as scanned and OCR'd
    max_pages: int = 20
    tesseract_cmd: str | None = field(default_factory=detect_tesseract_cmd)


@dataclass
class LLMConfig:
    enabled: bool = _env_bool("LLM_ENABLED", True)
    provider: str = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" | "openai_compatible"
    base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:11434")
    model: str = os.getenv("LLM_MODEL", "llama3.1:8b")
    api_key: str = os.getenv("LLM_API_KEY", "")
    temperature: float = 0.0
    num_ctx: int = 8192
    max_input_chars: int = 12000
    timeout: int = int(os.getenv("LLM_TIMEOUT", "180"))


@dataclass
class RulesConfig:
    date_output_format: str = "%d/%m/%Y"   # DD/MM/YYYY
    dayfirst: bool = True                  # 05/11/2024 -> 5 Nov (Indian / UK convention)
    min_year: int = 1900
    max_year: int = 2100
    default_country_code: str = os.getenv("DEFAULT_COUNTRY_CODE", "+91")
    amount_tolerance: float = 0.05         # tolerance for line-item arithmetic checks


@dataclass
class PipelineConfig:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    rules: RulesConfig = field(default_factory=RulesConfig)
    sqlite_path: str = os.getenv("SQLITE_PATH", "data/docai_etl.db")
