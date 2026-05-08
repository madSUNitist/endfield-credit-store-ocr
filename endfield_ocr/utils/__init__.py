"""Pure helper modules for geometry, image processing, and text normalization."""
from .geometry import order_quad_points, box_center_size, transform_points, clip_rect
from .geometry import iou_rect, rect_area, token_rect, rect_inter_area, union_token_box, rect_to_list
from .image import load_image_safe, maybe_resize, crop_rect_img
from .text import normalize_text, normalize_num_text, clean_name, has_chinese, group_tokens_lines

__all__ = [
    "order_quad_points", "box_center_size", "transform_points", "clip_rect",
    "iou_rect", "rect_area", "token_rect", "rect_inter_area", "union_token_box", "rect_to_list",
    "load_image_safe", "maybe_resize", "crop_rect_img",
    "normalize_text", "normalize_num_text", "clean_name", "has_chinese", "group_tokens_lines",
]