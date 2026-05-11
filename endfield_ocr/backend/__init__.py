# endfield_ocr/backend/__init__.py
"""OCR backend abstraction layer.
Provides a stable interface for text recognition engines.
"""
from .ocr_base import OCRBackend
from .paddle import PaddleOCRBackend
from .remote import RemotePaddleOCRBackend

__all__ = [
    "OCRBackend",
    "PaddleOCRBackend", 
    "RemotePaddleOCRBackend"
]