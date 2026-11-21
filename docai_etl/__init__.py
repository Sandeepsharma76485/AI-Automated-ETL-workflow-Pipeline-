"""DocAI ETL — AI-powered document extraction, transformation and loading.

Pipeline:  Extract (OpenCV + Tesseract OCR / native PDF text)
        -> Transform (Llama 3 8B structured extraction)
        -> Validate (rule engine: dates DD/MM/YYYY, RFC-style emails, E.164 phones, amounts)
        -> Load (JSON / CSV / SQLite)
"""

from .config import PipelineConfig, LLMConfig, OCRConfig, PreprocessConfig, RulesConfig
from .pipeline import DocumentETLPipeline, ETLResult

__version__ = "1.0.0"

__all__ = [
    "DocumentETLPipeline",
    "ETLResult",
    "PipelineConfig",
    "LLMConfig",
    "OCRConfig",
    "PreprocessConfig",
    "RulesConfig",
    "__version__",
]
