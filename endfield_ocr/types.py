# endfield_ocr/types.py
"""
Centralized type definitions, aliases, and protocols.
Replaces implicit dictionary contracts and magic strings with explicit
static types for IDE autocomplete, mypy validation, and cross-module consistency.
"""
from typing import Tuple, Dict, Optional, Any, Literal, Protocol


# ==========================================================
# Coordinate & Geometry Aliases
# ==========================================================

# Axis-aligned bounding box format: (left, top, right, bottom) in absolute pixels.
BBox = Tuple[float, float, float, float]

# Perspective quadrilateral: 4 vertices in clockwise order starting from top-left.
QuadPoints = Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]

# Normalized ROI relative to a slot's BODY rectangle. Values range [0.0, 1.0].
# Tokens in the namebar extension will yield ny > 1.0, which is intentional.
NormalizedROI = Tuple[float, float, float, float]


# ==========================================================
# Configuration & Source Literals
# ==========================================================

# OCR execution strategy.
# 'fast' = single full-image pass + tiny footer rescans.
# 'smart' = full pass + targeted fallback for missing price/namebar regions.
# 'full' = per-card crop (slow, debug/validation only).
OCRMode = Literal["fast", "smart", "full"]

# Supported text recognition backends.
BackendName = Literal["paddle", "tesseract"]

# Source tracking for how a slot's geometry was initially established.
SlotSource = Literal[
    "card",
    "projected_body",
    "white_body_only_no_namebar",
    "white_body_plus_local_namebar",
    "namebar_supplement",
    "rectified_edge",
    "grid_gap_completion"
]

# Tracking tag for OCR token origin. Used for confidence weighting and debug tracing.
TokenSource = Literal[
    "paddle_full", "paddle_namebar", "paddle_price_roi",
    "paddle_uid_tiny_roi", "paddle_soldout_card", "paddle_footer_refresh",
    "tesseract", "json", "joined_numeric"
]

# Icon matching algorithm used to resolve item names on cards without a detected namebar.
MatchMethod = Literal["fast_LAB_NCC", "partial_Canny_fallback"]

# Origin method for UID parsing. Indicates which fallback branch succeeded.
UIDSource = Literal[
    "uid_one_token_anchor", "uid_anchor_plus_digits", "uid_tiny_roi_numeric_fallback"
]


# ==========================================================
# Result & Metadata Dictionary Aliases
# ==========================================================

# Structured output for UID detection. Keys: uid, text, bbox, roi_bbox, confidence, source
UIDResult = Dict[str, Any]

# Structured output for shop refresh counter. Keys: remaining, total, text, anchor_text, confidence
RefreshResult = Dict[str, Any]

# Structured output for template-based icon matching. Keys: name, method, score, lab_best, lab_score, lab_margin, lab_top5
MatchResult = Dict[str, Any]

# Intermediate quadrilateral metadata passed between detection and homography stages.
QuadDict = Dict[str, Any]

# Generic rectangle dict used during grid construction and NMS.
RectDict = Dict[str, Any]

# Metadata dictionary returned by slot building pipeline.
SlotMeta = Dict[str, Any]

# Metadata dictionary tracking OCR passes and fallback sources.
OCRMeta = Dict[str, Any]


# ==========================================================
# Callback Protocols
# ==========================================================

# Callback signature for streaming batch progress.
# Receives current index, total count (or None if unknown), and the processed result.
class ProgressCallback(Protocol):
    def __call__(self, index: int, total: Optional[int], result: Any) -> None: ...

# Callback signature for error handling during batch processing.
# Returning True instructs the pipeline to skip the failed file and continue iteration.
class ErrorCallback(Protocol):
    def __call__(self, path: Any, error: Exception) -> bool: ...