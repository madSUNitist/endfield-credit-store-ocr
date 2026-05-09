"""
Centralized configuration for the Endfield Shop OCR pipeline.
Replaces scattered magic numbers and global constants with structured,
type-safe dataclasses. All thresholds and parameters are documented in English.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Literal, Optional

IMG_EXTS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})

@dataclass(frozen=True)
class ROIConfig(object):
    """
    Normalized coordinates (left, top, right, bottom) relative to the slot's BODY rect.
    Values are in the range [0.0, 1.0] and represent fractions of the card body width/height.
    These ROIs are used to isolate specific UI elements for parsing after token assignment.
    """
    # Item icon area: central region of the card body, excluding borders and bottom UI.
    item_image: Tuple[float, float, float, float] = (0.06, 0.06, 0.94, 0.68)
    # Name bar area: strictly below the card body. ny > 1.0 indicates it's outside the body.
    name_bar:   Tuple[float, float, float, float] = (0.00, 0.82, 1.00, 1.00)
    # Discount badge: top-right corner of the card body.
    discount:   Tuple[float, float, float, float] = (0.55, -0.03, 1.02, 0.18)
    # Price panel: bottom-right area. Shifted downward to avoid overlapping with quantity badges (x10/x2000).
    price:      Tuple[float, float, float, float] = (0.54, 0.70, 1.02, 0.97)
    # Quantity badge: center-left/middle area, typically where "x1", "x10" appear.
    quantity:   Tuple[float, float, float, float] = (0.28, 0.45, 0.78, 0.72)
    # Sold-out overlay: covers most of the card body. Used to detect "售罄" or dark overlay.
    sold_out:   Tuple[float, float, float, float] = (0.12, 0.20, 0.88, 0.78)

@dataclass(frozen=True)
class UIDConfig(object):
    """
    Configuration for UID detection and validation.
    UID is typically a small, faint numeric string in the bottom-left footer.
    """
    # Minimum and maximum expected UID length. Filters out latency (e.g., "61ms") or price fragments.
    min_length: int = 5
    max_length: int = 20
    # Normalized ratios (left, top, right, bottom) for the initial UID search in the rectified image.
    # Covers the bottom-left 34% width and top 12.5% of the footer area. Converted to absolute pixels at runtime.
    footer_search_ratios: Tuple[float, float, float, float] = (0.0, 0.875, 0.34, 0.995)

@dataclass(frozen=True)
class DetectionConfig(object):
    """
    Parameters for the initial card quadrilateral detection on the raw screenshot/photo.
    Uses contour area, aspect ratio, and spatial constraints to isolate UI cards.
    """
    # Valid card area range relative to the total image area. Filters out tiny artifacts and full-screen backgrounds.
    card_area_ratio: Tuple[float, float] = (0.004, 0.09)
    # Valid bounding box aspect ratio (width / height). Matches the game's card proportions.
    card_aspect_ratio: Tuple[float, float] = (0.45, 1.35)
    # Minimum horizontal/vertical distance for Non-Maximum Suppression (NMS), relative to the smaller image dimension.
    nms_distance_ratio: float = 0.055
    # Maximum allowed side length for the rectified output image. Prevents OOM during homography warping.
    max_output_side: int = 3000
    # Vertical position constraints to exclude top tabs and bottom navigation buttons.
    exclude_top_ratio: float = 0.10
    exclude_bottom_ratio: float = 0.86

@dataclass(frozen=True)
class SlotConfig(object):
    """
    Parameters for slot generation and namebar detection after perspective rectification.
    """
    # Vertical search range below the card body to find the namebar, relative to the body height.
    namebar_search_margin_ratio: float = 0.20
    # Minimum pixel density required for a detected region to be considered a valid namebar strip.
    min_namebar_density: float = 0.38
    # Maximum vertical gap allowed between the card body bottom and the namebar top. Rejects distant footer elements.
    max_namebar_body_gap_ratio: float = 0.045

@dataclass
class OCRConfig(object):
    """
    PaddleOCR initialization and runtime parameters.
    Uses v3.x parameter names to suppress deprecation warnings and improve compatibility.
    """
    # OCR processing mode: 'fast' (single pass), 'smart' (targeted fallback), or 'full' (per-card exhaustive).
    mode: Literal["fast", "smart", "full"] = "fast"
    language: str = "ch"
    # Enable text line orientation classification. Disabled by default as rectification already aligns text.
    use_textline_orientation: bool = False
    # Maximum side length for the internal detection network input. Balances speed and accuracy.
    text_det_limit_side_len: int = 1600
    # Confidence threshold for text detection. Lower values catch faint UI text but may increase noise.
    text_det_box_thresh: float = 0.30
    # Unclip ratio for DBPost processing. Controls how much the detection polygon expands to cover text boundaries.
    text_det_unclip_ratio: float = 1.7
    # Number of images processed in a single batch during recognition. Tuned for VRAM/CPU balance.
    text_recognition_batch_size: int = 16

@dataclass
class PipelineConfig(object):
    """
    Master configuration aggregating all subsystem parameters.
    Provides sensible defaults for processing ~10GB of shop screenshots.
    """
    roi: ROIConfig = field(default_factory=ROIConfig)
    uid: UIDConfig = field(default_factory=UIDConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    slot: SlotConfig = field(default_factory=SlotConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)

    # If True, skips the OCR step entirely. Useful for debugging geometry/detection pipelines.
    skip_ocr: bool = False
    # Maximum allowed width/height for input images before processing. Prevents memory spikes on high-res screenshots.
    max_input_side: Optional[int] = 3000
    # If True, recursively scans reference directories for icon matching.
    recursive_refs: bool = False
    # If set, saves debug visualizations to this directory (created if not exists)
    debug_save_dir: Optional[Path] = None
