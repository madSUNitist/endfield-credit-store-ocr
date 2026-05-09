# endfield_ocr/utils/geometry.py
"""
Low-level geometric operations for bounding boxes, polygons, and homography transforms.
All functions are pure (stateless) and thread-safe.
"""
import math
import cv2
import numpy as np
from typing import Tuple, List

from ..models import Token
from ..types import BBox


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """
    Reorders 4 polygon vertices to [top-left, top-right, bottom-right, bottom-left].
    Uses sum of coordinates for TL/BR and difference for TR/BL.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def box_center_size(box: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Computes center (cx, cy), average width, and average height from a 4-point polygon.
    Used to normalize OCR token coordinates and assist in spatial filtering.
    """
    box = np.asarray(box, dtype=np.float32).reshape(4, 2)
    cx = float(box[:, 0].mean())
    cy = float(box[:, 1].mean())
    top_w = np.linalg.norm(box[1] - box[0])
    bot_w = np.linalg.norm(box[2] - box[3])
    left_h = np.linalg.norm(box[3] - box[0])
    right_h = np.linalg.norm(box[2] - box[1])
    width_avg = float((top_w + bot_w) / 2)
    height_avg = float((left_h + right_h) / 2)
    return cx, cy, width_avg, height_avg


def transform_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Applies a 3x3 homography matrix to a set of 2D points.
    Reshapes input to (N, 1, 2) for cv2.perspectiveTransform, then flattens back.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, H)
    return transformed.reshape(-1, 2).astype(np.float32)


def clip_rect(rect: Tuple[float, float, float, float], w: int, h: int, pad: int = 0) -> Tuple[int, int, int, int]:
    """
    Clips a rectangle to image boundaries with optional padding.
    Ensures all coordinates are valid integers within [0, w] and [0, h].
    """
    l, t, r, b = rect
    x1 = max(0, int(math.floor(l - pad)))
    y1 = max(0, int(math.floor(t - pad)))
    x2 = min(w, int(math.ceil(r + pad)))
    y2 = min(h, int(math.ceil(b + pad)))
    return (x1, y1, x2, y2)


def iou_rect(a: BBox, b: BBox) -> float:
    """
    Computes Intersection over Union for two axis-aligned rectangles.
    Returns 0.0 if rectangles do not overlap.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    return inter / max(1.0, union)


def rect_area(rect: BBox) -> float:
    """Returns the area of an axis-aligned rectangle."""
    l, t, r, b = rect
    width = max(0.0, r - l)
    height = max(0.0, b - t)
    return width * height


def token_rect(token: Token) -> BBox:
    """
    Extracts the axis-aligned bounding box from a Token object.
    Always accepts Token, never raw np.ndarray, for type consistency.
    """
    pts = np.asarray(token.box, dtype=np.float32).reshape(4, 2)
    x1 = float(pts[:, 0].min())
    y1 = float(pts[:, 1].min())
    x2 = float(pts[:, 0].max())
    y2 = float(pts[:, 1].max())
    return (x1, y1, x2, y2)


def rect_inter_area(a: BBox, b: BBox) -> float:
    """Computes the intersection area of two rectangles."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    return iw * ih


def union_token_box(tokens: List[Token]) -> np.ndarray:
    """
    Computes the minimal bounding box covering a list of Tokens.
    Always accepts List[Token], never raw List[np.ndarray], for type consistency.
    """
    if not tokens:
        return np.zeros((4, 2), dtype=np.float32)
    # Extract box arrays from Token objects
    all_pts = np.vstack([np.asarray(t.box, dtype=np.float32).reshape(4, 2) for t in tokens])
    x1 = float(all_pts[:, 0].min())
    y1 = float(all_pts[:, 1].min())
    x2 = float(all_pts[:, 0].max())
    y2 = float(all_pts[:, 1].max())
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def rect_to_list(rect: BBox) -> List[float]:
    """Rounds rectangle coordinates to 2 decimal places for JSON serialization."""
    return [round(float(x), 2) for x in rect]