# endfield_ocr/__init__.py
"""
Public API for the Endfield Shop OCR module.
Exposes only stable entry points for external mod scripts.
Internal pipeline, backend, and utility modules remain hidden.
"""
from .processor import ShopOCRProcessor
from .config import PipelineConfig, OCRConfig, DetectionConfig, SlotConfig
from .models import Token, Slot, ShopResult
from .pipeline.matcher import load_ref_items

__all__ = [
    # Core Orchestrator
    "ShopOCRProcessor",
    # Configuration Dataclasses
    "PipelineConfig", "OCRConfig", "DetectionConfig", "SlotConfig",
    # Data Models
    "Token", "Slot", "ShopResult",
    # Reference Utilities
    "load_ref_items",
]