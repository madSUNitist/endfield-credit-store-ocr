# endfield_ocr/backend/ocr_base.py
"""Abstract base class for all OCR engines.
Enables dependency injection and runtime backend swapping without touching the pipeline.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import List

class OCRBackend(ABC):
    """Base interface for text recognition engines.
    All concrete backends must implement recognition and explicit resource cleanup.
    """

    @abstractmethod
    def recognize(self, image: np.ndarray, source: str = "ocr") -> List:
        """
        Run OCR on a single BGR image.
        Returns a list of structured text blocks (typically Token objects).
        Coordinates must be in absolute pixels relative to the input image.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release model weights, GPU/CPU memory, and internal inference caches."""
        ...