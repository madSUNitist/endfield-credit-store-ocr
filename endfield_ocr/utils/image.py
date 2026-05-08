# endfield_ocr/utils/image.py
"""
Image I/O, safe resizing, and cropping helpers.
Ensures all images are loaded in BGR format and clamped to memory-safe dimensions.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from .geometry import clip_rect


def load_image_safe(path: str | Path) -> np.ndarray:
    """
    Safely loads an image in BGR format. Raises ValueError if the file is missing or corrupted.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode image: {path}")
    return img


def maybe_resize(img: np.ndarray, max_side: Optional[int] = None) -> np.ndarray:
    """
    Downscales image if max side exceeds limit, preserving aspect ratio.
    Uses INTER_AREA for high-quality downscaling suitable for OCR.
    """
    if max_side is None:
        return img
    h, w = img.shape[:2]
    current_max = max(h, w)
    if current_max <= max_side:
        return img
    scale = max_side / current_max
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def crop_rect_img(img: np.ndarray, rect: tuple[float, float, float, float], pad: int = 0) -> np.ndarray:
    """
    Crops an image using a (left, top, right, bottom) rectangle.
    Automatically clamps coordinates to image boundaries using clip_rect.
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = clip_rect(rect, w, h, pad=pad)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return img[y1:y2, x1:x2]