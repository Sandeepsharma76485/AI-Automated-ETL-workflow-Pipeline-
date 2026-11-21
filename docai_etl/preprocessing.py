"""OpenCV image preprocessing to maximise OCR accuracy.

Steps (each optional): grayscale -> upscale -> denoise -> deskew -> binarise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from .config import PreprocessConfig


@dataclass
class PreprocessResult:
    image: np.ndarray                     # processed image (grayscale / binary)
    steps: list[str] = field(default_factory=list)
    skew_angle: float = 0.0


def pil_to_cv(image: Image.Image) -> np.ndarray:
    """PIL (RGB) -> OpenCV (BGR)."""
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def upscale_if_small(gray: np.ndarray, min_height: int) -> tuple[np.ndarray, bool]:
    h = gray.shape[0]
    if h >= min_height:
        return gray, False
    scale = min(min_height / h, 4.0)
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC), True


def denoise(gray: np.ndarray) -> np.ndarray:
    # Non-local means removes scanner speckle while keeping character edges sharp.
    return cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)


def estimate_skew(gray: np.ndarray, max_angle: float = 10.0, step: float = 0.25) -> float:
    """Projection-profile skew estimation.

    Rotates a downscaled binary copy across candidate angles and picks the one whose
    horizontal projection has the highest variance (text lines become sharp peaks).
    Robust and independent of OpenCV's minAreaRect angle conventions.
    """
    h, w = gray.shape[:2]
    scale = 1000.0 / max(w, 1) if w > 1000 else 1.0
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1.0 else gray
    binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    if cv2.countNonZero(binary) < 50:
        return 0.0

    sh, sw = binary.shape
    center = (sw / 2, sh / 2)

    def score(angle: float) -> float:
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(binary, m, (sw, sh), flags=cv2.INTER_NEAREST, borderValue=0)
        return float(np.var(rotated.sum(axis=1)))

    # coarse search, then refine around the best coarse angle
    coarse = np.arange(-max_angle, max_angle + 1e-6, 1.0)
    best = max(coarse, key=score)
    fine = np.arange(best - 1.0, best + 1.0 + 1e-6, step)
    best = max(fine, key=score)
    return float(round(best, 2))


def rotate(gray: np.ndarray, angle: float) -> np.ndarray:
    h, w = gray.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def binarize(gray: np.ndarray, method: str = "otsu") -> np.ndarray:
    if method == "adaptive":
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
        )
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


def preprocess(image: Image.Image | np.ndarray, cfg: PreprocessConfig | None = None) -> PreprocessResult:
    cfg = cfg or PreprocessConfig()
    img = pil_to_cv(image) if isinstance(image, Image.Image) else image.copy()
    steps: list[str] = []

    gray = to_grayscale(img)
    if cfg.grayscale:
        steps.append("grayscale")

    if cfg.upscale:
        gray, upscaled = upscale_if_small(gray, cfg.min_height)
        if upscaled:
            steps.append("upscale")

    if cfg.denoise:
        gray = denoise(gray)
        steps.append("denoise")

    angle = 0.0
    if cfg.deskew:
        angle = estimate_skew(gray)
        if abs(angle) >= 0.3:
            gray = rotate(gray, angle)
            steps.append(f"deskew({angle:+.2f}°)")

    if cfg.threshold:
        gray = binarize(gray, cfg.threshold_method)
        steps.append(f"threshold({cfg.threshold_method})")

    return PreprocessResult(image=gray, steps=steps, skew_angle=angle)
