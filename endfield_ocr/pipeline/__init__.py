# endfield_ocr/pipeline/__init__.py
"""
Pipeline module exposing core stage functions for the ShopOCR processor.
All imports are explicitly declared. Internal helpers are documented but not exported.
This file serves as the single entry point for pipeline dependencies.
"""

# ==========================================================
# Detection & Rectification Stage
# ==========================================================
from .detector import (
    detect_card_quads, 
    group_quads_rows, 
    rectify_by_card_plane
)

# ==========================================================
# Slot Generation & Grid Modeling Stage
# ==========================================================
from .slot_builder import (
    build_slots_after_rectification, 
    GridModel, 
    projected_rects_from_quads, 
    detect_rectified_card_edge_rects, 
    merge_anchor_rects, 
    complete_interior_grid_gaps, 
    group_rects_rows, 
    estimate_initial_model, 
    detect_local_namebar_band, 
    refine_body_height_with_namebars, 
    detect_global_namebar_components
)

# ==========================================================
# OCR Parsing & Field Extraction Stage
# ==========================================================
from .parser import (
    assign_tokens_to_slots, 
    tokens_in_region, 
    tokens_in_namebar, 
    match_item_name, 
    parse_name, 
    parse_sold_out, 
    parse_discount, 
    parse_quantity, 
    parse_prices, 
    parse_refresh, 
    default_uid_footer_roi, 
    parse_uid, 
    deduplicate_tokens, 
    group_tokens_lines, 
    collect_smart_fallback_rects
)

# ==========================================================
# Icon Matching & Template Stage (No-Namebar Fallback)
# ==========================================================
from .matcher import (
    load_ref_items, 
    match_card_bgr_to_refs, 
    RefItem, 
    MatchCardItem, 
    read_ref, 
    build_ignore_mask, 
    read_card_from_bgr, 
    fast_lab_ncc, 
    canny_fallback
)

# ==========================================================
# Public API Boundary
# ==========================================================
__all__ = [
    # Detector
    "detect_card_quads", "group_quads_rows", "rectify_by_card_plane",
    # Slot Builder
    "build_slots_after_rectification", "GridModel", "projected_rects_from_quads",
    "detect_rectified_card_edge_rects", "merge_anchor_rects", "complete_interior_grid_gaps",
    "group_rects_rows", "estimate_initial_model", "detect_local_namebar_band",
    "refine_body_height_with_namebars", "detect_global_namebar_components",
    # Parser
    "assign_tokens_to_slots", "tokens_in_region", "tokens_in_namebar", "match_item_name",
    "parse_name", "parse_sold_out", "parse_discount", "parse_quantity", "parse_prices",
    "parse_refresh", "default_uid_footer_roi", "parse_uid", "deduplicate_tokens", "group_tokens_lines",
    "collect_smart_fallback_rects", 
    # Matcher
    "load_ref_items", "match_card_bgr_to_refs", "RefItem", "MatchCardItem",
    "read_ref", "build_ignore_mask", "read_card_from_bgr", "fast_lab_ncc", "canny_fallback",
]