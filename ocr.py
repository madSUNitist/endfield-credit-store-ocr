#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

核心流程：
1. 在原图上检测商品卡片四边形，不依赖屏幕边缘，因此屏幕不全也能矫正；
2. 用多张卡片四角 RANSAC 拟合同一个 UI 平面并整图透视矫正；
3. 在矫正图上先生成“白色/灰色卡片主体 body slot”，再只向下局部检测名称栏；
4. 裁切每个商品卡片和商品图区域；
5. OCR 默认使用 PaddleOCR，解析：商品名、现价、原价、折扣、数量、售罄；
6. 支持 --no-ocr，仅测试透视矫正和卡片切分。

安装 OCR：
  pip install opencv-python numpy rapidfuzz paddleocr paddlepaddle


"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import unicodedata
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Iterable, Any

import cv2
import numpy as np
from rapidfuzz import fuzz, process

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = ImageDraw = ImageFont = None

DEFAULT_ITEM_NAMES = []  # v14: 默认不再内置物品字典；优先使用 --refs 文件名自适应生成。

UID_MIN_LEN = 5   # UID 不固定为 10 位；低于 5 位很容易和延迟/价格/次数混淆。
UID_MAX_LEN = 20  # 常见游戏 UID 不会超过这个长度；更长通常是 OCR 把别的数字粘进来了。

# slot 由卡片检测得到时使用；商品图像区域先单独裁出，暂不解析。
ITEM_IMAGE_ROI = (0.06, 0.06, 0.94, 0.68)
NAME_ROI       = (0.00, 0.82, 1.00, 1.00)
DISCOUNT_ROI   = (0.55, -0.03, 1.02, 0.18)
PRICE_ROI      = (0.54, 0.70, 1.02, 0.97)  # 价格 ROI 再整体下移：避开 x10/x30/x2000 数量徽标
QUANTITY_ROI   = (0.28, 0.45, 0.78, 0.72)
SOLDOUT_ROI    = (0.12, 0.20, 0.88, 0.78)

@dataclass
class Token:
    text: str
    box: np.ndarray
    score: float = 1.0
    source: str = "ocr"
    cx: float = 0.0
    cy: float = 0.0
    w: float = 0.0
    h: float = 0.0
    slot_id: Optional[int] = None

    def __post_init__(self):
        self.box = np.asarray(self.box, dtype=np.float32).reshape(4, 2)
        self.cx, self.cy, self.w, self.h = box_center_size(self.box)

@dataclass
class Slot:
    id: int
    row: int
    col: int
    rect: tuple[float, float, float, float]  # BODY rect only: white/gray card body, never includes name bar
    namebar_rect: Optional[tuple[float, float, float, float]] = None
    full_rect: Optional[tuple[float, float, float, float]] = None
    source: str = "card"
    name: Optional[str] = None
    name_score: Optional[float] = None
    tokens: list[Token] = field(default_factory=list)

    def __post_init__(self):
        # Keep rect as the body coordinate system.  OCR/price/discount/quantity
        # ROIs are normalized against rect, not against the downward namebar
        # extension.  This is the key difference from v6.
        if self.full_rect is None:
            if self.namebar_rect is None:
                self.full_rect = self.rect
            else:
                l1, t1, r1, b1 = self.rect
                l2, t2, r2, b2 = self.namebar_rect
                self.full_rect = (min(l1, l2), min(t1, t2), max(r1, r2), max(b1, b2))

    def norm_xy(self, t: Token) -> tuple[float, float]:
        # Normalized BODY coordinates.  A token in the namebar will have ny > 1,
        # which is intentional; price/quantity/discount ROIs remain unaffected
        # by whether a namebar exists.
        l, top, r, b = self.rect
        return (t.cx - l) / max(1.0, r - l), (t.cy - top) / max(1.0, b - top)

    def contains_token_center(self, t: Token, margin: float = 4.0) -> bool:
        l, top, r, b = self.full_rect or self.rect
        return l - margin <= t.cx <= r + margin and top - margin <= t.cy <= b + margin

# ---------------- text ----------------
def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("％", "%")
    s = s.replace("－", "-").replace("—", "-").replace("–", "-").replace("−", "-")
    s = s.replace("×", "x").replace("＊", "*").replace("￥", "")
    s = re.sub(r"\s+", "", s)
    return s

def normalize_num_text(s: str) -> str:
    s = normalize_text(s)
    table = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "|": "1", "S": "5", "B": "8"})
    return s.translate(table)

def clean_name(s: str) -> str:
    s = normalize_text(s)
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", s)

def has_chinese(s: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in s)

# ---------------- geometry ----------------
def order_quad_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def box_center_size(box: np.ndarray) -> tuple[float, float, float, float]:
    box = np.asarray(box, dtype=np.float32).reshape(4, 2)
    cx = float(box[:, 0].mean())
    cy = float(box[:, 1].mean())
    top_w = np.linalg.norm(box[1] - box[0])
    bot_w = np.linalg.norm(box[2] - box[3])
    left_h = np.linalg.norm(box[3] - box[0])
    right_h = np.linalg.norm(box[2] - box[1])
    return cx, cy, float((top_w + bot_w) / 2), float((left_h + right_h) / 2)

def transform_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2).astype(np.float32)

def clip_rect(rect, w, h, pad=0):
    l,t,r,b = rect
    return (max(0, int(math.floor(l-pad))), max(0, int(math.floor(t-pad))),
            min(w, int(math.ceil(r+pad))), min(h, int(math.ceil(b+pad))))

def iou_rect(a, b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1)
    inter=iw*ih
    aa=max(0,ax2-ax1)*max(0,ay2-ay1); bb=max(0,bx2-bx1)*max(0,by2-by1)
    return inter/max(1, aa+bb-inter)


def rect_area(rect) -> float:
    l, t, r, b = rect
    return float(max(0.0, r - l) * max(0.0, b - t))

def token_rect(t: Token) -> tuple[float, float, float, float]:
    pts = np.asarray(t.box, dtype=np.float32).reshape(4, 2)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return (float(x1), float(y1), float(x2), float(y2))

def rect_inter_area(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return float(max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1))

def union_token_box(tokens: list[Token]) -> np.ndarray:
    if not tokens:
        return np.zeros((4, 2), dtype=np.float32)
    pts = np.vstack([t.box for t in tokens]).astype(np.float32)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

def rect_to_list(rect) -> list[float]:
    return [round(float(x), 2) for x in rect]

# ---------------- perspective / card detection ----------------
def detect_card_quads(image: np.ndarray, debug_path: Optional[str] = None) -> list[dict]:
    """Detect item-card quadrilaterals on original photo/screenshot. Used for UI-plane homography."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    e1 = cv2.Canny(blur, 40, 120)
    e2 = cv2.Canny(blur, 20, 80)       # dark sold-out cards / camera photo edges
    edges = cv2.bitwise_or(e1, e2)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    quads: list[dict] = []
    img_area = h * w
    for c in contours:
        area = cv2.contourArea(c)
        if not (img_area * 0.004 < area < img_area * 0.09):
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        cx, cy = x + bw/2, y + bh/2
        # 排除顶部 tab、底部按钮等，但不要过严：屏幕拍照可能有黑边/外壳。
        if cy < h * 0.10 or cy > h * 0.86:
            continue
        if bw < w * 0.035 or bh < h * 0.075:
            continue
        ar = bw / max(1, bh)
        if not (0.45 < ar < 1.35):
            continue
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.03 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
        else:
            pts = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)
        pts = order_quad_points(pts)
        qcx, qcy, qw, qh = box_center_size(pts)
        qar = qw / max(1.0, qh)
        if not (0.45 < qar < 1.25):
            continue
        quads.append({"pts": pts, "cx": qcx, "cy": qcy, "area": float(area), "rect": (x, y, bw, bh)})

    quads.sort(key=lambda q: q["area"], reverse=True)
    kept: list[dict] = []
    for q in quads:
        if all(math.hypot(q["cx"]-k["cx"], q["cy"]-k["cy"]) > min(w,h)*0.055 for k in kept):
            kept.append(q)
    kept = sorted(kept, key=lambda q: (q["cy"], q["cx"]))

    if debug_path:
        vis = image.copy()
        for i,q in enumerate(kept):
            pts=q["pts"].astype(np.int32)
            cv2.polylines(vis,[pts],True,(0,255,0),3)
            cv2.putText(vis,str(i),(int(q["cx"]),int(q["cy"])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        cv2.imwrite(debug_path, vis)
    return kept

def group_quads_rows(quads: list[dict]) -> list[list[dict]]:
    if not quads:
        return []
    hs = [box_center_size(q["pts"])[3] for q in quads]
    tol = max(30.0, float(np.median(hs)) * 0.50)
    rows: list[list[dict]] = []
    for q in sorted(quads, key=lambda z: z["cy"]):
        for row in rows:
            if abs(q["cy"] - float(np.mean([r["cy"] for r in row]))) <= tol:
                row.append(q); break
        else:
            rows.append([q])
    for r,row in enumerate(rows):
        row.sort(key=lambda z:z["cx"])
        # Do not blindly enumerate columns. If one card was not detected in the
        # original image, a large horizontal gap must remain a large gap in the
        # canonical grid; otherwise homography compresses the whole row and the
        # missing card can never be recovered after rectification.
        xs = [float(q["cx"]) for q in row]
        gaps = [b - a for a, b in zip(xs, xs[1:]) if b > a]
        if gaps:
            cut = float(np.percentile(gaps, 60))
            small = [g for g in gaps if g <= cut]
            pitch = float(np.median(small or gaps))
        else:
            pitch = 1.0
        col = 0
        for i,q in enumerate(row):
            if i == 0:
                col = 0
            else:
                gap = xs[i] - xs[i-1]
                col += max(1, int(round(gap / max(1.0, pitch))))
            q["row"] = r; q["col"] = col
    return rows

def rectify_by_card_plane(image: np.ndarray, quads: list[dict], max_output_side: int = 3000) -> tuple[np.ndarray, np.ndarray, dict]:
    h, w = image.shape[:2]
    rows = group_quads_rows(quads)
    if len(quads) < 4 or not rows or max(len(r) for r in rows) < 2:
        return image.copy(), np.eye(3, dtype=np.float64), {"used": False, "reason": "not enough card quads", "cards_detected": len(quads)}

    out_card_w = 260.0
    out_card_h = 330.0
    gap_x = 20.0
    pitch_y = 380.0
    margin_x = 180.0
    margin_y = 190.0
    src_pts, dst_pts = [], []
    for row in rows:
        for q in row:
            c = q["col"]; r = q["row"]
            x = margin_x + c * (out_card_w + gap_x)
            y = margin_y + r * pitch_y
            dst_quad = np.array([[x,y],[x+out_card_w,y],[x+out_card_w,y+out_card_h],[x,y+out_card_h]], dtype=np.float32)
            src_pts.extend(q["pts"]); dst_pts.extend(dst_quad)
    H, inliers = cv2.findHomography(np.array(src_pts,np.float32), np.array(dst_pts,np.float32), cv2.RANSAC, 5.0)
    if H is None:
        return image.copy(), np.eye(3,dtype=np.float64), {"used": False, "reason": "homography failed", "cards_detected": len(quads)}

    corners = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)
    tr = transform_points(H, corners)
    mn = tr.min(axis=0) - 50
    mx = tr.max(axis=0) + 50
    out_w = int(math.ceil(mx[0]-mn[0])); out_h = int(math.ceil(mx[1]-mn[1]))
    scale = 1.0
    if max(out_w, out_h) > max_output_side:
        scale = max_output_side / max(out_w, out_h)
    T = np.array([[scale,0,-mn[0]*scale],[0,scale,-mn[1]*scale],[0,0,1]], dtype=np.float64)
    H_full = T @ H
    out_w = max(1, int(out_w*scale)); out_h = max(1, int(out_h*scale))
    rectified = cv2.warpPerspective(image, H_full, (out_w,out_h), borderValue=(245,245,245))
    return rectified, H_full, {
        "used": True,
        "cards_detected": len(quads),
        "rows_detected": [len(r) for r in rows],
        "inliers": int(inliers.sum()) if inliers is not None else None,
        "output_size": [out_w,out_h],
    }

# ---------------- slots after rectification ----------------
@dataclass
class GridModel:
    card_w: float
    body_h: float
    x_pitch: float
    row_tops: list[float]
    row_centers: list[list[float]]
    bar_h: float = 0.0
    source: str = "white_body_then_namebar"


def robust_percentile(values, p: float, default: float = 1.0) -> float:
    arr = np.asarray([float(v) for v in values if np.isfinite(v) and v > 0], dtype=np.float32)
    if arr.size == 0:
        return float(default)
    return float(np.percentile(arr, p))


def robust_median(values, default: float = 1.0) -> float:
    arr = np.asarray([float(v) for v in values if np.isfinite(v) and v > 0], dtype=np.float32)
    if arr.size == 0:
        return float(default)
    return float(np.median(arr))


def reject_size_outliers(values: list[float]) -> list[float]:
    """Data-driven size filter; no resolution-specific pixel boundary."""
    vals = [float(v) for v in values if np.isfinite(v) and v > 0]
    if len(vals) < 4:
        return vals
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = max(1.0, float(q3 - q1))
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    kept = [v for v in vals if lo <= v <= hi]
    return kept or vals


def projected_rects_from_quads(quads: list[dict], H_full: np.ndarray, rect_shape) -> list[dict]:
    """Project original card quadrilaterals onto the rectified image.

    These rectangles are used as approximate WHITE CARD BODY anchors only. They
    are never trusted as final boxes, because a contour may include a bottom tab,
    miss a name strip, or merge with a neighboring dark region.
    """
    h, w = rect_shape[:2]
    out = []
    for q in quads:
        pts = transform_points(H_full, q["pts"])
        l, t = pts.min(axis=0)
        r, b = pts.max(axis=0)
        l, t, r, b = clip_rect((l, t, r, b), w, h, pad=0)
        if r <= l or b <= t:
            continue
        out.append({
            "pts": pts,
            "rect": (float(l), float(t), float(r), float(b)),
            "cx": float((l + r) / 2),
            "cy": float((t + b) / 2),
            "w": float(r - l),
            "h": float(b - t),
            "raw": q,
        })
    return out



def detect_rectified_card_edge_rects(rectified: np.ndarray, model: GridModel) -> list[dict]:
    """Detect card/body rectangles directly on the rectified image.

    This supplements the original-photo quad detector. It is deliberately driven
    by the learned card scale, not by fixed pixel boundaries. We only use these
    rectangles as WHITE/BODY anchors; final slots still keep a learned fixed
    width and extend downward only with a local name-bar search.
    """
    if model.card_w <= 1 or model.body_h <= 1:
        return []
    H, W = rectified.shape[:2]
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)

    # Edge contours are more reliable than thresholding "white", because the UI
    # background can also be pale. RETR_LIST keeps nested full-card contours.
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    e1 = cv2.Canny(blur, 24, 92)
    e2 = cv2.Canny(blur, 12, 56)
    edges = cv2.bitwise_or(e1, e2)
    k = max(3, int(round(model.card_w * 0.018)))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    raw = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw <= 0 or bh <= 0:
            continue
        # All limits are relative to the learned card size. This rejects footer
        # buttons/top tabs without using screen-position hardcoding.
        if not (model.card_w * 0.62 <= bw <= model.card_w * 1.32):
            continue
        if not (model.body_h * 0.55 <= bh <= model.body_h * 1.35):
            continue
        ar = bw / max(1.0, bh)
        expected_ar = model.card_w / max(1.0, model.body_h)
        if not (expected_ar * 0.55 <= ar <= expected_ar * 1.85):
            continue
        fill = cv2.contourArea(c) / max(1.0, bw * bh)
        if fill < 0.42:
            continue
        # Prefer clean outer rectangles over small object contours inside card.
        score = fill * 2.0 - abs(bw - model.card_w) / max(1.0, model.card_w) - 0.35 * abs(bh - model.body_h) / max(1.0, model.body_h)
        raw.append({
            "rect": (float(x), float(y), float(x + bw), float(y + bh)),
            "cx": float(x + bw / 2), "cy": float(y + bh / 2),
            "w": float(bw), "h": float(bh), "source": "rectified_edge", "score": float(score),
        })

    # NMS by card center. If there are body-only and body+namebar contours for
    # the same card, keep one anchor; the final bottom will be decided later by
    # local namebar detection, so the exact height here is not trusted.
    raw.sort(key=lambda z: z.get("score", 0.0), reverse=True)
    kept = []
    for rec in raw:
        if all(math.hypot(rec["cx"] - k["cx"], rec["cy"] - k["cy"]) > model.card_w * 0.32 for k in kept):
            kept.append(rec)
    return sorted(kept, key=lambda z: (z["cy"], z["cx"]))


def merge_anchor_rects(rects: list[dict], card_w: float) -> list[dict]:
    """Merge projected and rectified-edge anchors without allowing giant boxes."""
    if not rects:
        return []
    # Projected anchors are useful for dark sold-out cards; edge anchors recover
    # missed white cards. Keep the best per physical card center.
    for r in rects:
        r.setdefault("score", 1.0 if r.get("source") == "rectified_edge" else 1.15)
    rects = sorted(rects, key=lambda z: z.get("score", 1.0), reverse=True)
    kept = []
    for r in rects:
        if all(math.hypot(r["cx"] - k["cx"], r["cy"] - k["cy"]) > card_w * 0.34 for k in kept):
            kept.append(r)
    return sorted(kept, key=lambda z: (z["cy"], z["cx"]))


def complete_interior_grid_gaps(rects: list[dict], model: GridModel, rect_shape) -> list[dict]:
    """Fill only obvious interior gaps in a row.

    This recovers a card whose contour was missed between two detected cards,
    but does not invent cards beyond the left/right visible extent of a row.
    """
    if not rects or model.x_pitch <= 1 or model.card_w <= 1:
        return rects
    H, W = rect_shape[:2]
    rows = group_rects_rows(rects)
    out = list(rects)
    for row in rows:
        if len(row) < 2:
            continue
        row = sorted(row, key=lambda z: z["cx"])
        row_top = robust_median([x["rect"][1] for x in row], row[0]["rect"][1])
        for a, b in zip(row, row[1:]):
            gap = b["cx"] - a["cx"]
            if gap <= model.x_pitch * 1.45:
                continue
            n_missing = int(round(gap / model.x_pitch)) - 1
            if n_missing <= 0 or n_missing > 3:
                continue
            for k in range(1, n_missing + 1):
                cx = a["cx"] + k * gap / (n_missing + 1)
                l, r = cx - model.card_w / 2, cx + model.card_w / 2
                t, bb = row_top, row_top + model.body_h
                l2, t2, r2, b2 = clip_rect((l, t, r, bb), W, H, pad=0)
                if r2 <= l2 or b2 <= t2:
                    continue
                out.append({
                    "rect": (float(l2), float(t2), float(r2), float(b2)),
                    "cx": float((l2 + r2) / 2), "cy": float((t2 + b2) / 2),
                    "w": float(r2 - l2), "h": float(b2 - t2),
                    "source": "grid_gap_completion", "score": 0.55,
                })
    return merge_anchor_rects(out, model.card_w)

def group_rects_rows(rects: list[dict]) -> list[list[dict]]:
    if not rects:
        return []
    hs = reject_size_outliers([r["h"] for r in rects])
    med_h = robust_median(hs, default=100.0)
    tol = med_h * 0.55
    rows: list[list[dict]] = []
    for rec in sorted(rects, key=lambda z: z["cy"]):
        for row in rows:
            if abs(rec["cy"] - float(np.median([x["cy"] for x in row]))) <= tol:
                row.append(rec)
                break
        else:
            rows.append([rec])
    for row in rows:
        row.sort(key=lambda z: z["cx"])
    return rows


def _estimate_pitch(xs: list[float], default: float) -> float:
    xs = sorted(float(x) for x in xs)
    gaps = [b - a for a, b in zip(xs, xs[1:]) if b > a]
    if not gaps:
        return default
    # Missing cards create 2*pitch gaps. Use the lower part of the gap distribution.
    cut = robust_percentile(gaps, 60, robust_median(gaps, default))
    small = [g for g in gaps if g <= cut]
    return robust_median(small or gaps, default)


def estimate_initial_model(projected_rects: list[dict]) -> GridModel:
    """Learn card width, row tops and a BODY-height hint from detected cards.

    Height is intentionally the lower cluster of observed heights. Some contours
    already include the bottom name strip and some do not; using the lower cluster
    makes the first pass search start from the white card body, then we explicitly
    add the name strip only if it is detected below.
    """
    if not projected_rects:
        return GridModel(1.0, 1.0, 1.0, [], [])
    widths = reject_size_outliers([r["w"] for r in projected_rects])
    heights = reject_size_outliers([r["h"] for r in projected_rects])
    card_w = robust_median(widths, 1.0)
    h_cut = robust_percentile(heights, 60, robust_median(heights, card_w * 1.25))
    lower_h = [h for h in heights if h <= h_cut]
    body_h_hint = robust_median(lower_h or heights, card_w * 1.25)

    rows = group_rects_rows(projected_rects)
    row_tops: list[float] = []
    row_centers: list[list[float]] = []
    all_xs: list[float] = []
    for row in rows:
        row_tops.append(robust_median([x["rect"][1] for x in row], row[0]["rect"][1]))
        xs = [float(x["cx"]) for x in row]
        row_centers.append(sorted(xs))
        all_xs.extend(xs)
    x_pitch = _estimate_pitch(all_xs, card_w * 1.06)
    if x_pitch < card_w * 0.80:
        x_pitch = card_w * 1.03
    return GridModel(card_w=card_w, body_h=body_h_hint, x_pitch=x_pitch, row_tops=row_tops, row_centers=row_centers)


def detect_local_namebar_band(rectified: np.ndarray, body_rect: tuple[float, float, float, float]) -> Optional[tuple[float, float, float, float, float]]:
    """Find a bottom name strip *only if it touches the detected white-card body*.

    Important: the input ``body_rect`` is the white/gray card body without the
    name strip. We scan a very small vertical window around its bottom edge, not
    the whole area below the card. This prevents footer buttons / refresh buttons
    from being swallowed as a fake name bar when a card really has no bottom name
    strip.

    Returns (left, top, right, bottom, confidence-like score).
    """
    H, W = rectified.shape[:2]
    l, t, r, b = body_rect
    card_w = max(1.0, r - l)
    body_h = max(1.0, b - t)

    # Only look directly around the lower edge of the white body.  Everything is
    # scale-relative.  This is deliberately much tighter than the v5 search.
    x1 = max(0, int(round(l + card_w * 0.015)))
    x2 = min(W, int(round(r - card_w * 0.015)))
    pre = max(2, int(round(body_h * 0.045)))
    post = max(8, int(round(body_h * 0.20)))
    y1 = max(0, int(round(b - pre)))
    y2 = min(H, int(round(b + post)))
    if x2 <= x1 or y2 <= y1:
        return None

    roi = rectified[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Adaptive but conservative: the name bar is genuinely dark; pale card
    # texture and the gray page background must not become positive.
    thr = min(138.0, max(30.0, float(np.percentile(gray, 30))))
    dark_thr = max(thr, 108.0)
    # 名字栏常是中深灰，不一定比局部 30 分位还暗；用局部阈值和绝对暗阈值的较大者。
    # 因为这里只在卡片主体底边附近搜索，所以绝对阈值不会把远处 footer 吞进来。
    dark = gray <= dark_thr
    row_ratio = dark.mean(axis=1)

    win = max(3, int(round(body_h * 0.014)))
    smoothed = np.convolve(row_ratio, np.ones(win, dtype=np.float32) / win, mode="same")

    runs = []
    start = None
    density_threshold = 0.38
    for i, v in enumerate(smoothed):
        if v > density_threshold and start is None:
            start = i
        if start is not None and (v <= density_threshold or i == len(smoothed) - 1):
            end = i if v <= density_threshold else i + 1
            gh = end - start
            gy1 = y1 + start
            gy2 = y1 + end

            # Name strip should be short and should begin at / just below the
            # detected body bottom.  A footer button has a much larger vertical
            # gap, so it is rejected here.
            max_gap = max(5.0, body_h * 0.045)
            if gy1 > b + max_gap or gy2 < b - max_gap:
                start = None
                continue
            if not (body_h * 0.035 <= gh <= body_h * 0.17):
                start = None
                continue

            band = dark[start:end, :]
            if band.size == 0:
                start = None
                continue
            # Rectangular strip: most columns should be dark in the band.  This
            # rejects small dark icons/text and rounded footer buttons.
            col_coverage = float((band.mean(axis=0) > 0.45).mean())
            density = float(smoothed[start:end].max()) if end > start else float(smoothed[start])
            if col_coverage < 0.46:
                start = None
                continue
            score = density + 0.35 * col_coverage - 0.35 * abs(gy1 - b) / max_gap
            runs.append((float(l), float(gy1), float(r), float(gy2), float(score)))
            start = None

    if not runs:
        return None

    # Prefer the band that touches the body edge, not the lowest band.
    runs.sort(key=lambda z: (abs(z[1] - b), -z[4]))
    return runs[0]


def refine_body_height_with_namebars(rectified: np.ndarray, projected_rects: list[dict], model: GridModel) -> GridModel:
    """Two-pass geometry: first find strips, then infer true white-body height."""
    rows = group_rects_rows(projected_rects)
    body_h_samples = []
    bar_h_samples = []
    for row_idx, row in enumerate(rows):
        row_top = model.row_tops[row_idx] if row_idx < len(model.row_tops) else robust_median([x["rect"][1] for x in row], row[0]["rect"][1])
        for rec in row:
            cx = rec["cx"]
            body_guess = (cx - model.card_w / 2, row_top, cx + model.card_w / 2, row_top + model.body_h)
            bar = detect_local_namebar_band(rectified, body_guess)
            if bar is None:
                continue
            body_h = bar[1] - row_top
            bar_h = bar[3] - bar[1]
            if model.body_h * 0.55 <= body_h <= model.body_h * 1.20:
                body_h_samples.append(body_h)
            if model.body_h * 0.015 <= bar_h <= model.body_h * 0.20:
                bar_h_samples.append(bar_h)

    if len(body_h_samples) >= 2:
        body_h = robust_median(body_h_samples, model.body_h)
    else:
        body_h = model.body_h
    bar_h = robust_median(bar_h_samples, model.body_h * 0.09) if bar_h_samples else model.body_h * 0.0
    return GridModel(model.card_w, body_h, model.x_pitch, model.row_tops, model.row_centers, bar_h=bar_h, source=model.source)


def detect_global_namebar_components(rectified: np.ndarray, model: GridModel) -> list[dict]:
    """Find extra name bars for cards that the quad detector missed.

    This is only a supplement. Components must be close to an already learned row
    and to the expected bottom of a white body; footer buttons and top tabs are
    therefore rejected without hardcoded screen regions.
    """
    if not model.row_tops or model.card_w <= 1 or model.body_h <= 1:
        return []
    H, W = rectified.shape[:2]
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    thr = min(145.0, max(35.0, float(np.percentile(gray, 25))))
    dark = (gray < thr).astype(np.uint8) * 255
    kx = max(3, int(round(model.card_w * 0.12)))
    ky = max(1, int(round(model.body_h * 0.015)))
    mask = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((kx, ky), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((max(3, kx // 3), max(1, ky)), np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if not (model.card_w * 0.65 <= bw <= model.card_w * 1.25):
            continue
        if not (model.body_h * 0.025 <= bh <= model.body_h * 0.20):
            continue
        cx = x + bw / 2
        cy = y + bh / 2
        best_row = None
        best_d = 1e18
        for ri, row_top in enumerate(model.row_tops):
            expected = row_top + model.body_h + max(model.bar_h, model.body_h * 0.08) / 2
            d = abs(cy - expected)
            if d < best_d:
                best_d = d
                best_row = ri
        if best_row is None or best_d > model.body_h * 0.20:
            continue
        cands.append({"row": best_row, "cx": float(cx), "rect": (float(x), float(y), float(x + bw), float(y + bh))})

    # NMS by row and horizontal center.
    cands.sort(key=lambda z: (z["row"], z["cx"]))
    kept = []
    for c in cands:
        if all(not (c["row"] == k["row"] and abs(c["cx"] - k["cx"]) < model.card_w * 0.35) for k in kept):
            kept.append(c)
    return kept


def _renumber_slots(slots: list[Slot]) -> list[Slot]:
    if not slots:
        return []
    # Regroup by vertical center; then order by x. This removes any bad column
    # number inherited from a partial row.
    rows: list[list[Slot]] = []
    med_h = robust_median([s.rect[3] - s.rect[1] for s in slots], 100.0)
    tol = med_h * 0.45
    for s in sorted(slots, key=lambda z: (z.rect[1] + z.rect[3]) / 2):
        cy = (s.rect[1] + s.rect[3]) / 2
        for row in rows:
            rcy = float(np.median([(x.rect[1] + x.rect[3]) / 2 for x in row]))
            if abs(cy - rcy) <= tol:
                row.append(s)
                break
        else:
            rows.append([s])
    out = []
    sid = 0
    for ri, row in enumerate(rows):
        row.sort(key=lambda z: (z.rect[0] + z.rect[2]) / 2)
        for ci, s in enumerate(row):
            s.id = sid
            s.row = ri
            s.col = ci
            out.append(s)
            sid += 1
    return out


def build_slots_after_rectification(rectified: np.ndarray, quads: list[dict], H_full: np.ndarray) -> tuple[list[Slot], dict]:
    projected = projected_rects_from_quads(quads, H_full, rectified.shape)
    if not projected:
        return [], {"projected_rects": 0, "rectified_edge_rects": 0, "final_slots": 0, "reason": "no projected card anchors"}

    # First learn scale from the homography-projected card anchors. Then detect
    # card rectangles again on the rectified image to recover missed white cards.
    initial = estimate_initial_model(projected)
    edge_rects = detect_rectified_card_edge_rects(rectified, initial)
    anchors0 = merge_anchor_rects(projected + edge_rects, initial.card_w)
    initial2 = estimate_initial_model(anchors0)
    anchors0 = complete_interior_grid_gaps(anchors0, initial2, rectified.shape)

    model = refine_body_height_with_namebars(rectified, anchors0, initial2)
    anchors = complete_interior_grid_gaps(anchors0, model, rectified.shape)
    H, W = rectified.shape[:2]
    rows = group_rects_rows(anchors)

    slots: list[Slot] = []
    sid = 0
    for ri, row in enumerate(rows):
        row_top = model.row_tops[ri] if ri < len(model.row_tops) else robust_median([x["rect"][1] for x in row], row[0]["rect"][1])
        for rec in sorted(row, key=lambda z: z["cx"]):
            cx = rec["cx"]
            l = cx - model.card_w / 2
            r = cx + model.card_w / 2
            # Use the actually detected white-card anchor bottom as the body
            # bottom.  Do not use a globally extended body height here: if a card
            # has no name bar, scanning far below its true body will catch footer
            # buttons.  Fall back to the learned height only if the anchor is an
            # obvious outlier.
            # BODY bottom is the learned white/gray body height, not the raw
            # contour bottom.  The raw contour may already include the black
            # namebar, especially for high-contrast cards; using it would make
            # the body ROI vertically inconsistent.
            body_rect = (l, row_top, r, row_top + model.body_h)
            bar = detect_local_namebar_band(rectified, body_rect)

            # rect is clipped BODY ONLY.  namebar_rect is a separate downward
            # extension.  Never enlarge rect itself, otherwise every body ROI
            # gets vertically squashed on cards with a namebar and no-namebar
            # cards become impossible to model correctly.
            l2, t2, r2, b2 = clip_rect(body_rect, W, H, pad=2)
            if r2 > l2 and b2 > t2:
                nb = None
                source = "white_body_only_no_namebar"
                if bar is not None:
                    bl, bt, br, bb, _score = bar
                    nl, nt, nr, nbottom = clip_rect((l, bt, r, bb), W, H, pad=1)
                    if nr > nl and nbottom > nt:
                        nb = (float(nl), float(nt), float(nr), float(nbottom))
                        source = "white_body_plus_local_namebar"
                slots.append(Slot(sid, ri, 0, (float(l2), float(t2), float(r2), float(b2)), namebar_rect=nb, source=source))
                sid += 1

    # Supplement only with namebars that fit the learned grid and are not matched
    # by an existing card. This recovers dark sold-out cards or weak white frames.
    global_bars = detect_global_namebar_components(rectified, model)
    for gb in global_bars:
        row = gb["row"]
        cx = gb["cx"]
        if any(s.row == row and abs(((s.rect[0] + s.rect[2]) / 2) - cx) < model.card_w * 0.45 for s in slots):
            continue
        if row >= len(model.row_tops):
            continue
        row_top = model.row_tops[row]
        l = cx - model.card_w / 2
        r = cx + model.card_w / 2
        # This supplement creates a normal BODY slot plus a separate namebar.
        body = clip_rect((l, row_top, r, row_top + model.body_h), W, H, pad=2)
        nb = clip_rect((l, gb["rect"][1], r, gb["rect"][3]), W, H, pad=1)
        l2, t2, r2, b2 = body
        nl, nt, nr, nbottom = nb
        if r2 > l2 and b2 > t2 and nr > nl and nbottom > nt:
            slots.append(Slot(len(slots), row, 0, (float(l2), float(t2), float(r2), float(b2)), namebar_rect=(float(nl), float(nt), float(nr), float(nbottom)), source="namebar_supplement"))

    slots = _renumber_slots(slots)

    meta = {
        "projected_rects": len(projected),
        "rectified_edge_rects": len(edge_rects),
        "anchor_rects_after_merge": len(anchors),
        "global_namebar_candidates": len(global_bars),
        "final_slots": len(slots),
        "learned_card_w": round(float(model.card_w), 2),
        "learned_body_h": round(float(model.body_h), 2),
        "learned_bar_h": round(float(model.bar_h), 2),
        "learned_x_pitch": round(float(model.x_pitch), 2),
        "row_tops": [round(float(x), 2) for x in model.row_tops],
    }
    return slots, meta


# Backward-compatible wrappers for older calls/tests.
def slots_from_projected_quads(quads: list[dict], H_full: np.ndarray, rect_shape) -> list[Slot]:
    rects = projected_rects_from_quads(quads, H_full, rect_shape)
    model = estimate_initial_model(rects)
    return _renumber_slots([
        Slot(i, 0, 0, clip_rect((r["cx"] - model.card_w / 2, r["rect"][1], r["cx"] + model.card_w / 2, r["rect"][1] + model.body_h), rect_shape[1], rect_shape[0], pad=2), source="projected_body")
        for i, r in enumerate(rects)
    ])


def slots_from_name_bars(rectified: np.ndarray) -> list[Slot]:
    return []


def extend_projected_slots_to_namebars(projected: list[Slot], bars: list[Slot], rect_shape) -> list[Slot]:
    return projected


def merge_slots(primary: list[Slot], secondary: list[Slot]) -> list[Slot]:
    return primary

# ---------------- OCR ----------------

def flatten_paddle_result(obj) -> Iterable:
    if isinstance(obj, (list, tuple)) and len(obj) == 2:
        box, rec = obj
        if isinstance(box, (list, tuple)) and isinstance(rec, (list, tuple)) and len(rec) >= 2 and isinstance(rec[0], str):
            yield obj; return
    if isinstance(obj, (list, tuple)):
        for x in obj:
            yield from flatten_paddle_result(x)

def create_paddle_ocr(fast: bool = True):
    logging.getLogger("ppocr").setLevel(logging.ERROR)
    logging.getLogger("paddleocr").setLevel(logging.ERROR)
    os.environ.setdefault("FLAGS_minloglevel", "2")
    from paddleocr import PaddleOCR
    # 兼容 PaddleOCR 2.x / 3.x。透视矫正后默认关闭角度分类，速度明显快。
    # 如果用户图源旋转严重，可以把 --paddle-angle-cls 打开。
    if fast:
        tries = [
            dict(lang="ch", use_angle_cls=False, show_log=False, det_limit_side_len=1600, det_db_box_thresh=0.30, det_db_unclip_ratio=1.7, rec_batch_num=16),
            dict(lang="ch", use_angle_cls=False, show_log=False, det_limit_side_len=1600),
            dict(lang="ch", use_textline_orientation=False),
            dict(lang="ch"),
        ]
    else:
        tries = [
            dict(lang="ch", use_angle_cls=True, show_log=False, det_limit_side_len=1800, det_db_box_thresh=0.25, det_db_unclip_ratio=1.8, rec_batch_num=12),
            dict(lang="ch", use_angle_cls=True, show_log=False),
            dict(lang="ch", use_textline_orientation=True),
            dict(lang="ch"),
        ]
    last=None
    for kw in tries:
        try:
            return PaddleOCR(**kw)
        except Exception as e:
            last=e
    raise last

def ocr_paddle_image(ocr, img: np.ndarray, offset=(0,0), source="paddle", cls: bool = False, scale_back: float = 1.0) -> list[Token]:
    """Run PaddleOCR on an ndarray and return boxes in the parent image coordinates.

    cls defaults to False.  This is important: when PaddleOCR was initialized
    without an angle classifier, calling ocr(..., cls=True) prints the annoying
    warning "Since the angle classifier is not initialized...".  The shop UI
    is already perspective-rectified, so angle classification is unnecessary.
    """
    # Silence ppocr's logger noise while still allowing real Python exceptions.
    logging.getLogger("ppocr").setLevel(logging.ERROR)
    logging.getLogger("paddleocr").setLevel(logging.ERROR)
    try:
        raw = ocr.ocr(img, cls=bool(cls))
    except TypeError:
        raw = ocr.ocr(img)
    tokens=[]; ox,oy=offset
    for line in flatten_paddle_result(raw):
        box, rec = line
        text, score = rec[0], float(rec[1])
        if not text or score < 0.25:
            continue
        b=np.array(box,np.float32)
        if scale_back and abs(scale_back - 1.0) > 1e-6:
            b = b / float(scale_back)
        b[:,0]+=ox; b[:,1]+=oy
        tokens.append(Token(text=text, box=b, score=score, source=source))
    return tokens

def _ocr_crop_with_offset(
    ocr,
    rectified: np.ndarray,
    rect: tuple[float,float,float,float],
    source: str,
    pad: int = 4,
    cls: bool = False,
    upscale: float = 1.0,
) -> list[Token]:
    h,w = rectified.shape[:2]
    l,t,r,b = clip_rect(rect, w, h, pad=pad)
    crop = rectified[t:b, l:r]
    if crop.size == 0:
        return []
    scale_back = 1.0
    if upscale and upscale > 1.01:
        crop = cv2.resize(crop, None, fx=float(upscale), fy=float(upscale), interpolation=cv2.INTER_CUBIC)
        scale_back = float(upscale)
    return ocr_paddle_image(ocr, crop, offset=(l,t), source=source, cls=cls, scale_back=scale_back)

def _body_roi_rect(slot: Slot, roi: tuple[float,float,float,float]) -> tuple[float,float,float,float]:
    l,t,r,b = slot.rect
    sw=max(1.0,r-l); sh=max(1.0,b-t)
    return (l+roi[0]*sw, t+roi[1]*sh, l+roi[2]*sw, t+roi[3]*sh)

def _slot_needs_fallback(slot: Slot, item_names: list[str]) -> tuple[bool, list[tuple[float,float,float,float,str]]]:
    """Decide whether local OCR is worth paying for after the one-pass full OCR.

    We keep this deliberately conservative: fast mode should be one OCR pass;
    smart mode only rescans regions that are likely to change the result.
    """
    rects: list[tuple[float,float,float,float,str]] = []
    name, _, name_occluded = parse_name(slot, item_names)
    price, _, price_present = parse_prices(slot, item_names)
    sold = parse_sold_out(slot)

    if slot.namebar_rect is not None and name is None and not name_occluded:
        rects.append((*slot.namebar_rect, "paddle_namebar"))

    # Price panel exists on most non-empty cards. If the price parser sees no
    # price tokens, rescan only the price ROI instead of the whole card.
    if price is None and not sold:
        rects.append((*_body_roi_rect(slot, PRICE_ROI), "paddle_price_roi"))

    # Sold-out overlay and discount are visually small/dark; if the card is dark
    # or has very few tokens, rescan the full card once.
    if sold and len(slot.tokens) < 3:
        rects.append((*(slot.full_rect or slot.rect), "paddle_soldout_card"))

    return bool(rects), rects

def collect_smart_fallback_rects(tokens: list[Token], slots: list[Slot], item_names: list[str], rectified_shape) -> list[tuple[float,float,float,float,str]]:
    # tokens must already be assigned to slots before calling this.
    rects: list[tuple[float,float,float,float,str]] = []
    for slot in slots:
        need, rs = _slot_needs_fallback(slot, item_names)
        if need:
            rects.extend(rs)

    # Global small fallback: UID and refresh are outside cards. Scan only footer
    # if anchors were not found by the full-image OCR.
    H,W = rectified_shape[:2]
    uid_roi = default_uid_footer_roi(rectified_shape)
    if parse_uid(tokens, rectified_shape, uid_roi=uid_roi) is None:
        # Bottom-left UID is tiny/faint; rescan only the narrow UID strip, not the whole footer.
        rects.append((*uid_roi, "paddle_uid_tiny_roi"))
    if parse_refresh(tokens) is None:
        rects.append((W*0.42, H*0.72, W, H, "paddle_footer_refresh"))

    # Merge near-identical rectangles, but keep source label for debugging.
    out=[]
    for r in rects:
        x1,y1,x2,y2,src = r
        if x2 <= x1 or y2 <= y1:
            continue
        dup=False
        for q in out:
            qx1,qy1,qx2,qy2,_ = q
            inter=rect_inter_area((x1,y1,x2,y2),(qx1,qy1,qx2,qy2))
            area=min(rect_area((x1,y1,x2,y2)), rect_area((qx1,qy1,qx2,qy2)))
            if area>0 and inter/area>0.82:
                dup=True; break
        if not dup:
            out.append(r)
    return out

def ocr_paddle(
    rectified_path: str,
    rectified: np.ndarray,
    slots: list[Slot],
    item_names: list[str],
    mode: str = "fast",
    use_angle_cls: bool = False,
) -> tuple[list[Token], dict]:
    """Run PaddleOCR.

    mode:
      fast  = one OCR pass on the rectified full image.
      smart = full image once, then only missing/suspicious local ROIs.
      full  = full image + every full card crop; slow, only for debugging.
    """
    ocr=create_paddle_ocr(fast=not use_angle_cls)
    tokens=ocr_paddle_image(ocr, rectified, source="paddle_full", cls=use_angle_cls)
    meta={"mode": mode, "full_passes": 1, "crop_passes": 0, "fallback_sources": []}

    if mode not in {"fast", "smart", "full"}:
        raise RuntimeError(f"unknown ocr mode: {mode}")

    if mode == "fast":
        # Keep fast mode cheap: no per-card OCR.  But UID and remaining-refresh
        # are tiny footer texts and are often missed in the full pass, so do at
        # most two small upscaled footer rescans.
        H, W = rectified.shape[:2]
        global_rects: list[tuple[float,float,float,float,str]] = []
        uid_roi = default_uid_footer_roi(rectified.shape)
        if parse_uid(tokens, rectified.shape, uid_roi=uid_roi) is None:
            global_rects.append((*uid_roi, "paddle_uid_tiny_roi"))
        if parse_refresh(tokens) is None:
            global_rects.append((W*0.38, H*0.70, W, H, "paddle_footer_refresh"))
        for x1,y1,x2,y2,src in global_rects:
            try:
                tokens.extend(_ocr_crop_with_offset(ocr, rectified, (x1,y1,x2,y2), source=src, pad=5, cls=use_angle_cls, upscale=(3.0 if "uid_tiny" in src else 2.0)))
                meta["crop_passes"] += 1
                meta["fallback_sources"].append(src)
            except Exception:
                continue
        return deduplicate_tokens(tokens), meta

    crop_rects: list[tuple[float,float,float,float,str]] = []
    if mode == "full":
        for s in slots:
            crop_rects.append((*(s.full_rect or s.rect), "paddle_card_full"))
    else:
        assign_tokens_to_slots(tokens, slots)
        crop_rects = collect_smart_fallback_rects(tokens, slots, item_names, rectified.shape)

    for x1,y1,x2,y2,src in crop_rects:
        try:
            tokens.extend(_ocr_crop_with_offset(ocr, rectified, (x1,y1,x2,y2), source=src, pad=5, cls=use_angle_cls, upscale=(3.0 if "uid_tiny" in src else (2.0 if "footer" in src else 1.0))))
            meta["crop_passes"] += 1
            meta["fallback_sources"].append(src)
        except Exception:
            # Keep the full-image result even if a fallback crop fails.
            continue
    return deduplicate_tokens(tokens), meta


def _token_clean_for_dedup(t: Token) -> str:
    s = normalize_num_text(t.text)
    s = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff%/:：\-]", "", s)
    return s

def _prefer_token(a: Token, b: Token) -> Token:
    """Choose the better representative of two overlapping OCR readings."""
    ca, cb = _token_clean_for_dedup(a), _token_clean_for_dedup(b)
    # When one reading is a fragment of the other, prefer the longer text even
    # if its confidence is slightly lower.  This handles 120 + (1,20).
    if len(ca) != len(cb):
        longer = a if len(ca) > len(cb) else b
        shorter = b if longer is a else a
        if shorter.score <= longer.score + 0.35:
            return longer
    # Otherwise prefer confidence, then area.
    if abs(a.score - b.score) > 0.03:
        return a if a.score > b.score else b
    return a if rect_area(token_rect(a)) >= rect_area(token_rect(b)) else b

def _tokens_equivalent_or_fragment(a: Token, b: Token) -> bool:
    ca, cb = _token_clean_for_dedup(a), _token_clean_for_dedup(b)
    if not ca or not cb:
        return False
    ra, rb = token_rect(a), token_rect(b)
    inter = rect_inter_area(ra, rb)
    if inter <= 0:
        return False
    aa, ab = rect_area(ra), rect_area(rb)
    small_cover = inter / max(1.0, min(aa, ab))
    big_cover = inter / max(1.0, max(aa, ab))
    center_close = math.hypot(a.cx - b.cx, a.cy - b.cy) <= max(10.0, (a.h + b.h) * 0.85)

    if ca == cb and (small_cover > 0.45 or center_close):
        return True

    da = re.sub(r"\D", "", ca)
    db = re.sub(r"\D", "", cb)
    if da and db:
        # Numeric fragment: 1 / 20 inside 120, or duplicated full/card OCR.
        if (da in db or db in da) and small_cover > 0.55:
            return True
        if fuzz.ratio(da, db) >= 88 and (small_cover > 0.45 or center_close):
            return True

    # Text fragment: keep 协议棱柱组 over 协议棱柱 if they occupy the same box.
    if (ca in cb or cb in ca) and min(len(ca), len(cb)) >= 2 and small_cover > 0.58:
        return True
    if fuzz.ratio(ca, cb) >= 88 and (small_cover > 0.50 or (center_close and big_cover > 0.25)):
        return True
    return False

def deduplicate_tokens(tokens: list[Token]) -> list[Token]:
    """Collapse full-image/per-card OCR duplicates and contained fragments.

    Paddle often returns several mutually overlapping readings for the same
    glyphs, especially when OCR is run both on the rectified image and on each
    card crop.  Examples: ``120`` plus ``1`` and ``20``.  A pure exact-string
    de-dup is not enough; downstream price/UID parsing must see one logical
    block.
    """
    clusters: list[list[Token]] = []
    ordered_tokens = sorted(
        tokens,
        key=lambda z: (len(_token_clean_for_dedup(z)), rect_area(token_rect(z)), z.score),
        reverse=True,
    )
    for t in ordered_tokens:
        if not normalize_text(t.text):
            continue
        placed = False
        for cl in clusters:
            if any(_tokens_equivalent_or_fragment(t, u) for u in cl):
                cl.append(t)
                placed = True
                break
        if not placed:
            clusters.append([t])

    out: list[Token] = []
    for cl in clusters:
        best = cl[0]
        for u in cl[1:]:
            best = _prefer_token(best, u)
        out.append(best)

    out.sort(key=lambda z: (z.cy, z.cx, -z.score))
    return out

def ocr_tesseract(image_path: str) -> list[Token]:
    import pytesseract
    img=cv2.imread(image_path)
    data=pytesseract.image_to_data(img, lang="chi_sim+eng", config="--psm 11", output_type=pytesseract.Output.DICT)
    tokens=[]
    for i,text in enumerate(data.get("text",[])):
        text=str(text).strip()
        if not text: continue
        try: score=float(data["conf"][i])/100.0
        except Exception: score=0.0
        if score<0.20: continue
        x,y,bw,bh=int(data["left"][i]),int(data["top"][i]),int(data["width"][i]),int(data["height"][i])
        if bw<=0 or bh<=0: continue
        box=np.array([[x,y],[x+bw,y],[x+bw,y+bh],[x,y+bh]], np.float32)
        tokens.append(Token(text, box, score, "tesseract"))
    return tokens

def load_tokens_json(path: str) -> list[Token]:
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    arr=data.get("tokens", data if isinstance(data,list) else [])
    out=[]
    for x in arr:
        if "box" in x: box=np.array(x["box"],np.float32)
        else:
            l,t,r,b=x["rect"]; box=np.array([[l,t],[r,t],[r,b],[l,b]],np.float32)
        out.append(Token(x["text"], box, float(x.get("score",1.0)), x.get("source","json")))
    return out

# ---------------- assign / parse ----------------
def assign_tokens_to_slots(tokens: list[Token], slots: list[Slot]) -> None:
    for s in slots: s.tokens.clear()
    for t in tokens:
        best=None; best_margin=1e18
        for s in slots:
            if s.contains_token_center(t, margin=4.0):
                l,top,r,b=s.rect
                margin=abs(t.cx-(l+r)/2)+abs(t.cy-(top+b)/2)
                if margin<best_margin: best=s; best_margin=margin
        if best is not None:
            t.slot_id=best.id; best.tokens.append(t)

def tokens_in_region(slot: Slot, roi) -> list[Token]:
    # BODY-normalized region.  This must never use full_rect.
    nx1,ny1,nx2,ny2=roi
    return [t for t in slot.tokens if nx1 <= slot.norm_xy(t)[0] <= nx2 and ny1 <= slot.norm_xy(t)[1] <= ny2]

def tokens_in_namebar(slot: Slot) -> list[Token]:
    if slot.namebar_rect is None:
        return []
    l,t,r,b = slot.namebar_rect
    return [x for x in slot.tokens if l - 3 <= x.cx <= r + 3 and t - 3 <= x.cy <= b + 3]

def match_item_name(text: str, item_names: list[str], threshold: float = 68.0) -> tuple[Optional[str], float]:
    text_clean=clean_name(text)
    if not text_clean: return None,0.0
    mapping={clean_name(n):n for n in item_names}
    m=process.extractOne(text_clean, list(mapping.keys()), scorer=fuzz.ratio)
    if not m: return None,0.0
    best,score,_=m
    return (mapping[best], float(score)) if score>=threshold else (None,float(score))

def parse_name(slot: Slot, item_names: list[str]) -> tuple[Optional[str], Optional[float], bool]:
    # Name is read from the separate namebar.  If there is no local namebar, do
    # not reinterpret the bottom of the body as a fake namebar; return None.
    candidates=[]
    namebar_toks = sorted(tokens_in_namebar(slot), key=lambda t:(t.cy,t.cx))
    for t in namebar_toks:
        if has_chinese(t.text):
            candidates.append((t.text, t.score*10+12.0))
    if namebar_toks:
        lines=group_tokens_lines(namebar_toks, 1.5)
        for line in lines:
            raw="".join(t.text for t in sorted(line,key=lambda z:z.cx))
            if has_chinese(raw):
                candidates.append((raw, float(np.mean([t.score for t in line]))*10+14.0))

    # Sold-out overlay may cover the namebar; only then allow a weak whole-slot
    # fallback to recover the item name from visible text.
    sold=parse_sold_out(slot)
    if sold and not candidates:
        for t in slot.tokens:
            if has_chinese(t.text) and "售" not in t.text:
                candidates.append((t.text, t.score*10))

    best=(None,0.0)
    for raw, base in candidates:
        if "售" in raw and len(clean_name(raw))<=3:
            continue
        name,score=match_item_name(raw,item_names,threshold=62.0)
        if name and score+base>best[1]:
            best=(name,score+base)
    return best[0], (None if best[0] is None else round(best[1]/120.0,4)), bool(sold and best[0] is None)

def parse_sold_out(slot: Slot) -> bool:
    joined=normalize_text("".join(t.text for t in slot.tokens))
    if any(x in joined for x in ["售罄","售馨","已售罄"]): return True
    for t in tokens_in_region(slot, SOLDOUT_ROI) + slot.tokens:
        s=clean_name(t.text)
        if fuzz.ratio(s,"售罄")>=60 or fuzz.ratio(s,"售馨")>=60: return True
    return False

def parse_discount(slot: Slot) -> Optional[int]:
    candidates=tokens_in_region(slot, DISCOUNT_ROI)
    scan=[normalize_num_text(t.text) for t in candidates]
    scan.append(normalize_num_text("".join(t.text for t in candidates)))
    vals=[]
    for s in scan:
        # -95% 被识别成 -95元 时，只有右上角 ROI 才允许按折扣处理。
        for m in re.finditer(r"-?\s*(\d{1,2})\s*(?:%|元|折)?", s):
            v=int(m.group(1))
            if 1<=v<=99: vals.append(v)
    return max(vals) if vals else None

def parse_quantity(slot: Slot) -> Optional[int]:
    toks=sorted(tokens_in_region(slot, QUANTITY_ROI)+slot.tokens, key=lambda t:(t.cy,t.cx))
    for t in toks:
        s=normalize_num_text(t.text).replace("*","x")
        m=re.search(r"[xX]\s*(\d{1,7})", s)
        if m: return int(m.group(1))
    for a,b in zip(toks,toks[1:]):
        if normalize_num_text(a.text).lower()=='x' and re.fullmatch(r"\d{1,7}", normalize_num_text(b.text)):
            if abs(a.cy-b.cy)<max(a.h,b.h)*2.0: return int(normalize_num_text(b.text))
    return None

def _numeric_value_candidates_from_tokens(tokens: list[Token]) -> list[tuple[int, Token, str]]:
    """Return numeric readings and also joined adjacent fragments on the same line."""
    out: list[tuple[int, Token, str]] = []
    toks = sorted(tokens, key=lambda t: (t.cy, t.cx))
    for t in toks:
        s = normalize_num_text(t.text)
        for m in re.finditer(r"\d{1,5}", s):
            v = int(m.group(0))
            out.append((v, t, "single"))

    # If OCR split 120 into 1 + 20 and no reliable composite survived, add a
    # joined candidate.  Only join same-line, close, digit-only fragments.
    lines = group_tokens_lines(toks, 1.25)
    for line in lines:
        parts = []
        for t in sorted(line, key=lambda z: z.cx):
            s = normalize_num_text(t.text)
            if re.fullmatch(r"\d{1,3}", s):
                parts.append(t)
        if len(parts) < 2:
            continue
        run = [parts[0]]
        for prev, cur in zip(parts, parts[1:]):
            gap = cur.cx - prev.cx - (prev.w + cur.w) / 2
            if abs(cur.cy - prev.cy) <= max(cur.h, prev.h) * 0.75 and gap <= max(cur.h, prev.h) * 1.15:
                run.append(cur)
            else:
                if len(run) >= 2:
                    raw = "".join(normalize_num_text(x.text) for x in run)
                    if re.fullmatch(r"\d{2,5}", raw):
                        fake = Token(raw, union_token_box(run), max(x.score for x in run), "joined_numeric")
                        out.append((int(raw), fake, "joined"))
                run = [cur]
        if len(run) >= 2:
            raw = "".join(normalize_num_text(x.text) for x in run)
            if re.fullmatch(r"\d{2,5}", raw):
                fake = Token(raw, union_token_box(run), max(x.score for x in run), "joined_numeric")
                out.append((int(raw), fake, "joined"))
    return out

def _dedupe_numeric_values(cands: list[tuple[int, Token, str]]) -> list[tuple[int, Token, str]]:
    # Remove fragments when a larger overlapping numeric block exists.
    keep = [True] * len(cands)
    for i, (vi, ti, si) in enumerate(cands):
        if not keep[i]:
            continue
        di = str(vi)
        ri = token_rect(ti)
        for j, (vj, tj, sj) in enumerate(cands):
            if i == j:
                continue
            dj = str(vj)
            rj = token_rect(tj)
            inter = rect_inter_area(ri, rj)
            if inter <= 0:
                continue
            cover_i = inter / max(1.0, rect_area(ri))
            # vi is a fragment of vj, e.g. 1/20 inside 120.
            if len(dj) > len(di) and di in dj and cover_i > 0.55:
                keep[i] = False
                break
    out = [c for c, k in zip(cands, keep) if k]
    # Unique by value and approximate vertical position; keep the highest score.
    final: list[tuple[int, Token, str]] = []
    for cand in sorted(out, key=lambda x: x[1].score, reverse=True):
        v, t, src = cand
        if any(v == ov and math.hypot(t.cx - ot.cx, t.cy - ot.cy) < max(10.0, t.h + ot.h) for ov, ot, _ in final):
            continue
        final.append(cand)
    return final

def parse_prices(slot: Slot, item_names: list[str]) -> tuple[Optional[int], Optional[int], bool]:
    price_tokens=[]
    # Strict lower-right price ROI. If this region has no price-like tokens,
    # return price_panel_present=False instead of guessing from elsewhere.
    for t in tokens_in_region(slot, PRICE_ROI):
        nx, ny = slot.norm_xy(t)
        s=normalize_num_text(t.text)
        if not s or '%' in s or '/' in s: continue
        if re.search(r"[xX]\s*\d+",s): continue
        if any(ch in s for ch in "售罄馨"): continue
        if match_item_name(t.text,item_names,70)[0]: continue
        # A bare number near the left/center is more likely to be quantity.
        if nx < 0.58 and not re.search(r"[币信信用元₵$]", t.text):
            continue
        price_tokens.append(t)

    cands = _dedupe_numeric_values(_numeric_value_candidates_from_tokens(price_tokens))
    vals = sorted(set(v for v, _, _ in cands if 1 <= v <= 99999))
    if not vals:
        return None,None,False
    if len(vals)==1:
        return vals[0], None, True
    return vals[0], vals[-1], True

def group_tokens_lines(tokens: list[Token], tol_factor: float=1.8):
    if not tokens: return []
    med_h=float(np.median([max(1,t.h) for t in tokens])); tol=max(10,med_h*tol_factor)
    rows=[]
    for t in sorted(tokens,key=lambda z:z.cy):
        for row in rows:
            if abs(t.cy-np.mean([x.cy for x in row]))<=tol:
                row.append(t); break
        else: rows.append([t])
    for row in rows: row.sort(key=lambda z:z.cx)
    return rows

def _line_text(line: list[Token]) -> str:
    return "".join(t.text for t in sorted(line, key=lambda z:z.cx))

def _is_refresh_anchor_text(s: str) -> bool:
    c = clean_name(s)
    # Do not accept arbitrary A/B.  The UI has other ratios; this parser is
    # allowed to fire only when the Chinese label for refresh remaining count is
    # present or very close under OCR noise.
    if "剩余次数" in c:
        return True
    if ("剩余" in c and "次" in c) or ("刷新" in c and "次" in c):
        return True
    return fuzz.partial_ratio(c, "剩余次数") >= 72 or fuzz.partial_ratio(c, "剩余刷新次数") >= 72

def parse_refresh(tokens: list[Token]) -> Optional[dict]:
    best = None
    for line in group_tokens_lines(tokens,2.2):
        raw = _line_text(line)
        if not _is_refresh_anchor_text(raw):
            continue
        s = normalize_num_text(raw)
        m = re.search(r"(\d{1,3})\s*/\s*(\d{1,3})", s)
        if not m:
            # Try only the tokens near/right of the Chinese anchor on this line.
            ordered = sorted(line, key=lambda z:z.cx)
            anchor_i = 0
            for i, t in enumerate(ordered):
                if _is_refresh_anchor_text(t.text):
                    anchor_i = i
                    break
            tail = normalize_num_text("".join(t.text for t in ordered[anchor_i:]))
            m = re.search(r"(\d{1,3})\s*/\s*(\d{1,3})", tail)
        if not m:
            continue
        cand = {
            "remaining": int(m.group(1)),
            "total": int(m.group(2)),
            "text": m.group(0),
            "anchor_text": raw,
            "confidence": 0.96,
        }
        best = cand
        break
    return best


def default_uid_footer_roi(image_shape) -> tuple[float, float, float, float]:
    """Small UID search region in rectified-image coordinates.

    The old versions rescanned almost the whole lower-left footer. That made OCR
    glue latency/battery/other footer text into the UID.  Here we only use the
    narrow bottom-left UID strip.  It is deliberately independent of any assumed
    UID length.
    """
    H, W = image_shape[:2]
    return (0.0, H * 0.875, W * 0.34, H * 0.995)


def rect_contains_point(rect, x: float, y: float, margin: float = 0.0) -> bool:
    l, t, r, b = rect
    return l - margin <= x <= r + margin and t - margin <= y <= b + margin


def _uid_anchor_match(s: str):
    # Require UID-like anchor, not generic ID, to avoid eating unrelated IDs.
    # Common OCR variants: UID, U1D, U|D, U D, UlD.
    return re.search(r"U\s*[I1l|]\s*D", normalize_num_text(s), flags=re.IGNORECASE)


def _digit_runs(s: str) -> list[str]:
    return re.findall(r"\d+", normalize_num_text(s))


def _pick_uid_digits_from_runs(runs: list[str], prefer: str = "first") -> Optional[str]:
    """Pick one UID-looking digit block without assuming a fixed length.

    v14 wrongly preferred/cut to 10 digits.  Some accounts are not 10 digits,
    so v15 accepts any standalone 5..20 digit block.  Runs longer than
    UID_MAX_LEN are considered OCR-glued garbage; if this happens after an UID
    anchor we keep the side closest to the anchor, but mark the source as less
    confident elsewhere.
    """
    runs = [r for r in runs if r]
    if not runs:
        return None

    valid = [r for r in runs if UID_MIN_LEN <= len(r) <= UID_MAX_LEN]
    if valid:
        return valid[0] if prefer != "last" else valid[-1]

    long_runs = [r for r in runs if len(r) > UID_MAX_LEN]
    if long_runs:
        r = long_runs[0] if prefer != "last" else long_runs[-1]
        return r[-UID_MAX_LEN:] if prefer == "last" else r[:UID_MAX_LEN]
    return None


def _uid_candidate_digits(s: str, prefer: str = "last") -> Optional[str]:
    s = normalize_num_text(s)
    if "/" in s or "%" in s:
        return None
    return _pick_uid_digits_from_runs(_digit_runs(s), prefer=prefer)


def _uid_token_rank(t: Token, uid: str, image_shape=None, uid_roi=None) -> float:
    score = 0.0
    if UID_MIN_LEN <= len(uid) <= UID_MAX_LEN:
        score += 0.8
    if image_shape is not None:
        H, W = image_shape[:2]
        # True UID is tiny text in the bottom-left footer.
        if t.cy > 0.86 * H:
            score += 0.25
        if t.cx < 0.34 * W:
            score += 0.25
        if t.h < max(18, H * 0.040):
            score += 0.12
    if uid_roi is not None and rect_contains_point(uid_roi, t.cx, t.cy, margin=max(8.0, t.h * 2.0)):
        score += 0.45
    return score


def _token_span_rect_horizontal(t: Token, start_frac: float, end_frac: float) -> tuple[float, float, float, float]:
    """Approximate a sub-span bbox inside a horizontal OCR token."""
    x1, y1, x2, y2 = token_rect(t)
    start_frac = max(0.0, min(1.0, float(start_frac)))
    end_frac = max(start_frac, min(1.0, float(end_frac)))
    return (x1 + (x2 - x1) * start_frac, y1, x1 + (x2 - x1) * end_frac, y2)


def uid_debug_candidates(tokens: list[Token], image_shape=None, limit: int = 8) -> list[str]:
    """Small debug helper shown in the overlay when UID is missing/suspicious."""
    uid_roi = default_uid_footer_roi(image_shape) if image_shape is not None else None
    cands = []
    for t in sorted(tokens, key=lambda z: (z.cy, z.cx)):
        s = normalize_num_text(t.text)
        anchor_m = _uid_anchor_match(s)
        d = None
        if anchor_m:
            d = _uid_candidate_digits(s[anchor_m.end():], prefer="first")
        if d is None:
            d = _uid_candidate_digits(s, prefer="last")
        in_uid_roi = True
        if uid_roi is not None:
            in_uid_roi = rect_contains_point(uid_roi, t.cx, t.cy, margin=max(8.0, t.h * 2.0))
        if anchor_m or (d and in_uid_roi):
            tag = "anchor" if anchor_m else "num"
            cands.append(f"{tag}:{t.text}->{d or '?'}")
        if len(cands) >= limit:
            break
    return cands


def parse_uid(tokens: list[Token], image_shape=None, uid_roi=None) -> Optional[dict]:
    """Parse UID as a separate tiny footer block.

    v15 changes:
      * UID length is flexible: any 5..20 digit run is accepted;
      * fallback numeric search is restricted to a small bottom-left UID ROI;
      * result carries both uid bbox and uid_roi so debug can show exactly what
        was used.
    """
    if image_shape is not None and uid_roi is None:
        uid_roi = default_uid_footer_roi(image_shape)

    # 1) UID anchor and digits in the same token. Only inspect text AFTER the
    # anchor; this prevents preceding latency such as `61ms` from contaminating.
    one_token = []
    for t in sorted(tokens, key=lambda z: (z.cy, z.cx)):
        s = normalize_num_text(t.text)
        m_anchor = _uid_anchor_match(s)
        if not m_anchor:
            continue
        tail = s[m_anchor.end():]
        uid = _uid_candidate_digits(tail, prefer="first")
        if uid:
            # Approximate bbox of the UID part rather than the whole OCR line.
            # If the OCR token is already just UID:digits, this is nearly the
            # same box; if it contains leading junk, the debug bbox becomes much
            # closer to the real UID text.
            token_len = max(1, len(s))
            digit_start = s.find(uid, m_anchor.end())
            if digit_start >= 0:
                bbox = _token_span_rect_horizontal(t, digit_start / token_len, (digit_start + len(uid)) / token_len)
            else:
                bbox = token_rect(t)
            one_token.append((_uid_token_rank(t, uid, image_shape, uid_roi), t, uid, tail, bbox))
    if one_token:
        one_token.sort(key=lambda x: (-x[0], x[1].cx))
        score, t, uid, tail, bbox = one_token[0]
        return {
            "uid": uid,
            "text": f"UID:{uid}",
            "raw_text": t.text,
            "tail": tail,
            "bbox": rect_to_list(bbox),
            "roi_bbox": rect_to_list(uid_roi) if uid_roi is not None else None,
            "confidence": round(float(0.90 + min(0.08, score / 20)), 3),
            "source": "uid_one_token_anchor",
        }

    # 2) UID anchor token followed by digit token(s) nearby on the same visual line.
    for line in group_tokens_lines(tokens, 2.0):
        ordered = sorted(line, key=lambda z: z.cx)
        for i, t in enumerate(ordered):
            s_t = normalize_num_text(t.text)
            m_anchor = _uid_anchor_match(s_t)
            if not m_anchor:
                continue
            used = [t]
            pieces: list[str] = []
            tail_digits = _uid_candidate_digits(s_t[m_anchor.end():], prefer="first")
            if tail_digits:
                pieces.append(tail_digits)
            last = t
            for u in ordered[i + 1:]:
                if abs(u.cy - t.cy) > max(t.h, u.h) * 1.45:
                    continue
                gap = u.cx - last.cx - (u.w + last.w) / 2
                if gap > max(t.h, u.h) * 4.0 and pieces:
                    break
                su = normalize_num_text(u.text)
                d = _uid_candidate_digits(su, prefer="first")
                if d:
                    pieces.append(d)
                    used.append(u)
                    last = u
                    uid_joined = "".join(pieces)
                    uid = _pick_uid_digits_from_runs([uid_joined], prefer="first")
                    if uid:
                        return {
                            "uid": uid,
                            "text": f"UID:{uid}",
                            "raw_text": "|".join(x.text for x in used),
                            "bbox": rect_to_list(token_rect(Token("", union_token_box(used), 1.0, "uid_block"))),
                            "roi_bbox": rect_to_list(uid_roi) if uid_roi is not None else None,
                            "confidence": 0.92,
                            "source": "uid_anchor_plus_digits",
                        }
                    # Keep collecting until either valid length appears or it is clearly too long.
                    if len(uid_joined) > UID_MAX_LEN:
                        break
                elif pieces:
                    break

    # 3) Bottom-left numeric fallback. Strictly location-gated by the tiny UID ROI.
    # This catches cases where OCR sees only the number and misses `UID:`.
    if uid_roi is not None:
        cands = []
        for t in tokens:
            if not rect_contains_point(uid_roi, t.cx, t.cy, margin=max(6.0, t.h * 1.5)):
                continue
            uid = _uid_candidate_digits(t.text, prefer="last")
            if not uid:
                continue
            raw_norm = normalize_num_text(t.text).lower()
            penalty = 0.25 if ("ms" in raw_norm or "/" in raw_norm or "%" in raw_norm) else 0.0
            score = 0.65 + _uid_token_rank(t, uid, image_shape, uid_roi) * 0.12 - penalty
            cands.append((score, t.cx, uid, t))
        if cands:
            cands.sort(key=lambda x: (-x[0], x[1]))
            score, _, uid, t = cands[0]
            return {
                "uid": uid,
                "text": f"UID:{uid}",
                "raw_text": t.text,
                "bbox": rect_to_list(token_rect(t)),
                "roi_bbox": rect_to_list(uid_roi),
                "confidence": round(float(score), 3),
                "source": "uid_tiny_roi_numeric_fallback",
            }

    return None

def slot_result(slot: Slot, item_names: list[str]) -> dict:
    name, conf, occluded = parse_name(slot,item_names)
    price, original, price_panel_present = parse_prices(slot,item_names)
    return {
        "id": slot.id, "row":slot.row, "col":slot.col, "slot_source":slot.source,
        "bbox": [round(float(x),2) for x in slot.rect],  # body bbox, not including namebar
        "full_bbox": [round(float(x),2) for x in (slot.full_rect or slot.rect)],
        "namebar_bbox": None if slot.namebar_rect is None else [round(float(x),2) for x in slot.namebar_rect],
        "has_namebar": slot.namebar_rect is not None,
        "name": name, "name_confidence": conf, "name_occluded": occluded,
        "discount_percent": parse_discount(slot),
        "sold_out": parse_sold_out(slot),
        "quantity": parse_quantity(slot),
        "price": price, "original_price": original, "price_panel_present": price_panel_present,
        "ocr_texts": [{"text":t.text,"score":round(float(t.score),4),"nx":round(slot.norm_xy(t)[0],3),"ny":round(slot.norm_xy(t)[1],3),"source":t.source} for t in sorted(slot.tokens,key=lambda z:(z.cy,z.cx))]
    }

# ---------------- in-memory icon matcher for no-namebar cards ----------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

@dataclass
class RefItem:
    name: str
    path: Path
    bgr: np.ndarray
    lab: np.ndarray
    mask255: np.ndarray
    edge01: np.ndarray

@dataclass
class MatchCardItem:
    name: str
    full_bgr: np.ndarray
    roi_bgr: np.ndarray
    roi_lab: np.ndarray
    edge01: np.ndarray
    valid01: np.ndarray

def list_images(folder: Path, recursive: bool = False) -> list[Path]:
    if not folder.exists():
        return []
    it = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMG_EXTS and not p.name.startswith("._"))

def read_ref(path: Path, max_side: int = 110, pad: int = 4) -> RefItem:
    rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if rgba is None:
        raise FileNotFoundError(path)
    if rgba.ndim == 2:
        bgr0 = cv2.cvtColor(rgba, cv2.COLOR_GRAY2BGR)
        alpha0 = (rgba > 5).astype(np.uint8) * 255
    elif rgba.shape[2] == 4:
        bgr0 = rgba[:, :, :3]
        alpha0 = rgba[:, :, 3]
    else:
        bgr0 = rgba[:, :, :3]
        gray0 = cv2.cvtColor(bgr0, cv2.COLOR_BGR2GRAY)
        alpha0 = (gray0 > 5).astype(np.uint8) * 255

    ys, xs = np.where(alpha0 > 20)
    if len(xs) == 0:
        raise ValueError(f"empty foreground: {path}")
    y1, y2 = max(0, ys.min() - pad), min(alpha0.shape[0], ys.max() + pad + 1)
    x1, x2 = max(0, xs.min() - pad), min(alpha0.shape[1], xs.max() + pad + 1)
    bgr = bgr0[y1:y2, x1:x2]
    alpha = alpha0[y1:y2, x1:x2]

    scale = max_side / max(bgr.shape[:2])
    if scale < 1:
        new_size = (max(1, int(round(bgr.shape[1] * scale))), max(1, int(round(bgr.shape[0] * scale))))
        bgr = cv2.resize(bgr, new_size, interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, new_size, interpolation=cv2.INTER_NEAREST)

    mask255 = (alpha > 20).astype(np.uint8) * 255
    if min(mask255.shape) >= 5:
        mask255 = cv2.erode(mask255, np.ones((3, 3), np.uint8), 1)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edge = cv2.Canny(gray, 40, 120, L2gradient=True)
    edge01 = ((edge > 0) & (mask255 > 0)).astype(np.uint8)
    edge01 = cv2.dilate(edge01, np.ones((2, 2), np.uint8), 1)
    return RefItem(path.stem, path, bgr, lab, mask255, edge01)

def build_ignore_mask(roi_bgr: np.ndarray) -> np.ndarray:
    h, w = roi_bgr.shape[:2]
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    ignore = np.zeros((h, w), np.uint8)
    ignore[: int(0.20 * h), int(0.62 * w) :] = 1       # discount
    ignore[int(0.82 * h) :, :] = 1                     # price/name area inside body crop

    bright_white = ((hsv[:, :, 2] > 175) & (hsv[:, :, 1] < 85)).astype(np.uint8)
    bright_white[: int(0.28 * h), :] = 0
    bright_white[int(0.85 * h) :, :] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bright_white, connectivity=8)
    mean_gray = float(gray.mean())
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if area < 20:
            continue
        cx, cy = x + ww / 2, y + hh / 2
        is_count_badge = (0.35 * h < cy < 0.83 * h and 10 <= ww <= 115 and 7 <= hh <= 60)
        is_sold_overlay = (mean_gray < 155 and 0.25 * h < cy < 0.72 * h and 5 <= ww <= 155 and 5 <= hh <= 75)
        if is_count_badge or is_sold_overlay:
            ignore[y : y + hh, x : x + ww] = 1
    return cv2.dilate(ignore, np.ones((7, 7), np.uint8), 1)

def read_card_from_bgr(full: np.ndarray, name: str = "slot", roi_width: int = 160) -> MatchCardItem:
    if full is None or full.size == 0:
        raise ValueError("empty card crop")
    H, W = full.shape[:2]
    x1, x2 = int(0.02 * W), int(0.98 * W)
    y1, y2 = int(0.07 * H), int(0.76 * H)
    roi = full[y1:y2, x1:x2]
    if roi.size == 0:
        roi = full.copy()
    scale = roi_width / max(1, roi.shape[1])
    roi = cv2.resize(roi, (roi_width, max(1, int(round(roi.shape[0] * scale)))), interpolation=cv2.INTER_AREA)
    ignore = build_ignore_mask(roi)
    valid01 = (ignore == 0).astype(np.uint8)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edge01 = (cv2.Canny(gray, 35, 110, L2gradient=True) > 0).astype(np.uint8)
    edge01[ignore > 0] = 0
    return MatchCardItem(name, full, roi, lab, edge01, valid01)

def fast_lab_ncc(ref: RefItem, card: MatchCardItem, scales: Iterable[float]):
    H, W = card.roi_lab.shape[:2]
    bg_lab = np.median(card.roi_lab[: max(1, min(20, H)), : max(1, min(20, W))].reshape(-1, 3), axis=0).astype(np.uint8)
    best_score = -9.0
    best_info = None
    for s in scales:
        th = max(8, int(round(ref.lab.shape[0] * s)))
        tw = max(8, int(round(ref.lab.shape[1] * s)))
        if th > H or tw > W:
            continue
        tmpl = cv2.resize(ref.lab, (tw, th), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        mask = cv2.resize(ref.mask255, (tw, th), interpolation=cv2.INTER_NEAREST) > 0
        tmpl2 = tmpl.copy()
        tmpl2[~mask] = bg_lab
        weights = (0.55, 1.0, 1.0)
        res_sum = None
        for ch, wt in enumerate(weights):
            res = cv2.matchTemplate(card.roi_lab[:, :, ch], tmpl2[:, :, ch], cv2.TM_CCOEFF_NORMED)
            res = np.nan_to_num(res, nan=-2.0, posinf=-2.0, neginf=-2.0)
            res_sum = res * wt if res_sum is None else res_sum + res * wt
        res_sum /= sum(weights)
        _, maxv, _, maxloc = cv2.minMaxLoc(res_sum)
        if maxv > best_score:
            best_score = float(maxv)
            best_info = (float(s), (int(maxloc[0]), int(maxloc[1])), (int(tw), int(th)))
    return best_score, best_info

def partial_canny_score(ref_edge: np.ndarray, ref_mask01: np.ndarray, card: MatchCardItem, stride: int = 4):
    H, W = card.edge01.shape
    th, tw = ref_edge.shape[:2]
    if th > H or tw > W:
        return -1.0, None
    target_dil = cv2.dilate(card.edge01, np.ones((3, 3), np.uint8), 1)
    tmpl_dil = cv2.dilate(ref_edge, np.ones((3, 3), np.uint8), 1)
    best_score = -1.0
    best_loc = None
    for y in range(0, H - th + 1, stride):
        for x in range(0, W - tw + 1, stride):
            valid = card.valid01[y : y + th, x : x + tw]
            te_visible = ref_edge & valid
            n_te = int(te_visible.sum())
            if n_te < 12:
                continue
            target_patch = card.edge01[y : y + th, x : x + tw]
            target_in_obj = target_patch & ref_mask01 & valid
            n_target = int(target_in_obj.sum())
            recall = float((te_visible & target_dil[y : y + th, x : x + tw]).sum()) / n_te
            precision = 0.0 if n_target <= 5 else float((target_in_obj & tmpl_dil).sum()) / n_target
            score = 0.72 * recall + 0.28 * precision
            if score > best_score:
                best_score = score
                best_loc = (x, y)
    return best_score, best_loc

def canny_fallback(ref: RefItem, card: MatchCardItem, scales: Iterable[float], stride: int = 4):
    H, W = card.edge01.shape
    best_score = -1.0
    best_info = None
    for s in scales:
        th = max(8, int(round(ref.edge01.shape[0] * s)))
        tw = max(8, int(round(ref.edge01.shape[1] * s)))
        if th > H or tw > W:
            continue
        te = cv2.resize(ref.edge01, (tw, th), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        tm = cv2.resize((ref.mask255 > 0).astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        if int(te.sum()) < 12:
            continue
        score, loc = partial_canny_score(te, tm, card, stride=stride)
        if score > best_score:
            best_score = float(score)
            best_info = (float(s), loc or (0, 0), (int(tw), int(th)))
    return best_score, best_info

def load_ref_items(ref_dir: Optional[str | Path], recursive: bool = False) -> list[RefItem]:
    if ref_dir is None:
        return []
    paths = list_images(Path(ref_dir), recursive=recursive)
    refs = []
    for p in paths:
        try:
            refs.append(read_ref(p))
        except Exception:
            continue
    return refs

def match_card_bgr_to_refs(card_bgr: np.ndarray, refs: list[RefItem], name: str = "slot") -> Optional[dict]:
    if not refs or card_bgr is None or card_bgr.size == 0:
        return None
    card = read_card_from_bgr(card_bgr, name=name)
    lab_scales = np.linspace(0.62, 1.32, 8)
    edge_scales = np.linspace(0.60, 1.35, 10)
    lab_scores = []
    for ref in refs:
        score, info = fast_lab_ncc(ref, card, lab_scales)
        lab_scores.append((score, ref, info))
    lab_scores.sort(key=lambda x: x[0], reverse=True)
    best_score, best_ref, best_info = lab_scores[0]
    second_score = lab_scores[1][0] if len(lab_scores) > 1 else -9.0
    method = "fast_LAB_NCC"
    chosen_score, chosen_ref, chosen_info = best_score, best_ref, best_info
    if best_score < 0.45:
        edge_scores = []
        for ref in refs:
            escore, einfo = canny_fallback(ref, card, edge_scales, stride=4)
            edge_scores.append((escore, ref, einfo))
        edge_scores.sort(key=lambda x: x[0], reverse=True)
        chosen_score, chosen_ref, chosen_info = edge_scores[0]
        method = "partial_Canny_fallback"
    return {
        "name": chosen_ref.name,
        "method": method,
        "score": round(float(chosen_score), 4),
        "lab_best": best_ref.name,
        "lab_score": round(float(best_score), 4),
        "lab_margin": round(float(best_score - second_score), 4),
        "lab_top5": [(r.name, round(float(s), 4)) for s, r, _ in lab_scores[:5]],
    }

# ---------------- compact pipeline / batch ----------------
def _unique_item_names(names: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for n in names:
        n = str(n).strip()
        if not n:
            continue
        key = clean_name(n) or normalize_text(n)
        if key in seen:
            continue
        seen.add(key)
        out.append(n)
    return out

def load_item_names(path: Optional[str]) -> list[str]:
    if path is None:
        return DEFAULT_ITEM_NAMES[:]
    names = [x.strip() for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    return _unique_item_names(names)

def load_item_names_from_refs_or_file(items: Optional[str], refs: list[RefItem]) -> list[str]:
    # v14: whenever --refs is supplied, the OCR name whitelist is derived only
    # from reference image stems.  This makes the dictionary adapt to the user's
    # current item library and prevents stale hard-coded names from matching.
    if refs:
        return _unique_item_names([r.name for r in refs])
    return load_item_names(items)

def crop_rect_img(img: np.ndarray, rect, pad: int = 0) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = clip_rect(rect, w, h, pad=pad)
    return img[y1:y2, x1:x2]

def compact_uid(uid_obj: Optional[dict]) -> Optional[str]:
    if not uid_obj:
        return None
    return uid_obj.get("uid")

def compact_refresh(refresh_obj: Optional[dict]) -> Optional[dict]:
    if not refresh_obj:
        return None
    return {"remaining": refresh_obj.get("remaining"), "total": refresh_obj.get("total")}

def result_item_from_slot(slot: Slot, item_names: list[str], refs: list[RefItem], rectified: np.ndarray) -> dict:
    name, conf, occluded = parse_name(slot, item_names)
    name_source = "ocr_namebar" if name is not None else None
    match_info = None
    # Only no-namebar slots are classified by icon/template matching. Cards with
    # a real namebar keep the OCR name path; this avoids using visual matching as
    # a second source when the UI already provides text.
    if slot.namebar_rect is None and refs:
        card_bgr = crop_rect_img(rectified, slot.rect, pad=2)  # body only; no file write
        match_info = match_card_bgr_to_refs(card_bgr, refs, name=f"slot_{slot.id}")
        if match_info is not None:
            name = match_info["name"]
            conf = match_info["score"]
            name_source = "icon_match_no_namebar"
            occluded = False

    price, original, price_panel_present = parse_prices(slot, item_names)
    item = {
        "name": name,
        "name_confidence": conf,
        "name_source": name_source,
        "name_occluded": bool(occluded),
        "discount_percent": parse_discount(slot),
        "price": price,
        "original_price": original,
        "price_panel_present": bool(price_panel_present),
        "quantity": parse_quantity(slot),
        "sold_out": parse_sold_out(slot),
    }
    # Keep batch JSON compact: no coordinates and no large OCR/debug payload.
    return item

# ---------------- debug drawing with Chinese text ----------------

def _find_cjk_font() -> Optional[str]:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None

def _load_debug_font(size: int):
    if ImageFont is None:
        return None
    font_path = _find_cjk_font()
    try:
        if font_path:
            return ImageFont.truetype(font_path, size=size)
        return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()

def _pil_text_bbox(draw, xy, text, font):
    try:
        return draw.textbbox(xy, text, font=font)
    except Exception:
        w = len(str(text)) * 10
        h = 18
        x, y = xy
        return (x, y, x + w, y + h)

def _draw_text_box(draw, xy, text: str, font, fill=(255, 255, 0), bg=(0, 0, 0), pad: int = 4):
    x, y = int(xy[0]), int(xy[1])
    bbox = _pil_text_bbox(draw, (x, y), text, font)
    x1, y1, x2, y2 = bbox
    draw.rectangle((x1 - pad, y1 - pad, x2 + pad, y2 + pad), fill=bg)
    draw.text((x, y), text, font=font, fill=fill)
    return (x1 - pad, y1 - pad, x2 + pad, y2 + pad)

def draw_final_debug(
    rectified: np.ndarray,
    slots: list[Slot],
    items: list[dict],
    out_path: str,
    uid_obj: Optional[dict] = None,
    refresh_obj: Optional[dict] = None,
    tokens: Optional[list[Token]] = None,
):
    """Write exactly one debug image per input image.

    Green rectangle = card body, used by price/discount/quantity ROIs.
    Cyan rectangle  = separate namebar extension, never used to scale price ROI.
    Text is drawn with PIL so Chinese item names render correctly.
    """
    vis = rectified.copy()
    H, W = vis.shape[:2]

    if Image is None or ImageDraw is None:
        # Fallback: no Chinese support, but still draw rectangles.
        font = cv2.FONT_HERSHEY_SIMPLEX
        for slot, item in zip(slots, items):
            l, t, r, b = [int(round(x)) for x in slot.rect]
            cv2.rectangle(vis, (l, t), (r, b), (0, 220, 0), 2)
            if slot.namebar_rect is not None:
                nl, nt, nr, nb = [int(round(x)) for x in slot.namebar_rect]
                cv2.rectangle(vis, (nl, nt), (nr, nb), (255, 255, 0), 2)
            text = f"{item.get('name') or '?'} {item.get('discount_percent')} {item.get('price')}"
            cv2.putText(vis, text, (l + 2, max(20, t + 22)), font, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        uid_roi = uid_obj.get("roi_bbox") if uid_obj and uid_obj.get("roi_bbox") else rect_to_list(default_uid_footer_roi(rectified.shape))
        if uid_roi:
            ul, ut, ur, ub = [int(round(float(x))) for x in uid_roi]
            cv2.rectangle(vis, (ul, ut), (ur, ub), (0, 150, 255), 2)
        if uid_obj and uid_obj.get("bbox"):
            bl, bt, br, bb = [int(round(float(x))) for x in uid_obj.get("bbox")]
            cv2.rectangle(vis, (bl, bt), (br, bb), (255, 0, 255), 2)
        header = f"UID={uid_obj.get('uid') if uid_obj else None}  refresh={refresh_obj.get('remaining') if refresh_obj else None}/{refresh_obj.get('total') if refresh_obj else None}"
        cv2.putText(vis, header, (10, 26), font, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_path, vis)
        return

    # Draw geometry using OpenCV first, then text using PIL.
    for slot in slots:
        l, t, r, b = [int(round(x)) for x in slot.rect]
        cv2.rectangle(vis, (l, t), (r, b), (0, 220, 0), 2)
        if slot.namebar_rect is not None:
            nl, nt, nr, nb = [int(round(x)) for x in slot.namebar_rect]
            cv2.rectangle(vis, (nl, nt), (nr, nb), (255, 255, 0), 2)

    # UID debug: orange = tiny UID OCR/search ROI; magenta = parsed UID text bbox.
    uid_roi = None
    if uid_obj and uid_obj.get("roi_bbox"):
        uid_roi = uid_obj.get("roi_bbox")
    else:
        uid_roi = rect_to_list(default_uid_footer_roi(rectified.shape))
    if uid_roi:
        ul, ut, ur, ub = [int(round(float(x))) for x in uid_roi]
        cv2.rectangle(vis, (ul, ut), (ur, ub), (0, 150, 255), 2)
    if uid_obj and uid_obj.get("bbox"):
        bl, bt, br, bb = [int(round(float(x))) for x in uid_obj.get("bbox")]
        cv2.rectangle(vis, (bl, bt), (br, bb), (255, 0, 255), 2)

    img = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    header_font = _load_debug_font(max(14, min(24, W // 78)))
    card_font = _load_debug_font(max(11, min(18, W // 118)))

    uid_text = uid_obj.get("uid") if uid_obj else None
    uid_src = uid_obj.get("source") if uid_obj else "missing"
    uid_raw = ""
    if uid_obj and uid_obj.get("raw_text"):
        raw = str(uid_obj.get("raw_text"))
        if len(raw) > 28:
            raw = raw[:28] + "…"
        uid_raw = f" raw:{raw}"
    if refresh_obj:
        refresh_text = f"{refresh_obj.get('remaining')}/{refresh_obj.get('total')}"
    else:
        refresh_text = "None"
    cand_text = ""
    if tokens is not None and (uid_obj is None or uid_obj.get("source", "").endswith("fallback")):
        cands = uid_debug_candidates(tokens, rectified.shape, limit=3)
        if cands:
            cand_text = "  候选:" + ", ".join(cands)
    header = f"UID:{uid_text if uid_text else 'None'} [{uid_src}]{uid_raw}  剩余:{refresh_text}{cand_text}"
    _draw_text_box(draw, (10, 10), header, header_font, fill=(255, 255, 0), bg=(0, 0, 0), pad=4)

    for slot, item in zip(slots, items):
        l, t, r, b = [int(round(x)) for x in slot.rect]
        name = item.get("name") or "?"
        disc = item.get("discount_percent")
        price = item.get("price")
        disc_s = f"-{disc}%" if disc is not None else "无折扣"
        price_s = f"价:{price}" if price is not None else "无价格"
        text = f"{name}  {disc_s}  {price_s}"
        tx = max(4, min(l + 3, W - 60))
        ty = max(42, t + 4)
        _draw_text_box(draw, (tx, ty), text, card_font, fill=(255, 255, 0), bg=(0, 0, 0), pad=2)

    out_bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, out_bgr)

def recognize_one_compact(
    image_path: str | Path,
    refs: list[RefItem],
    item_names: list[str],
    backend: str = "paddle",
    no_ocr: bool = False,
    ocr_json: Optional[str] = None,
    ocr_mode: str = "fast",
    paddle_angle_cls: bool = False,
    debug_path: Optional[str | Path] = None,
    include_meta: bool = False,
) -> dict:
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    quads = detect_card_quads(image, debug_path=None)
    rectified, Hmat, rect_meta = rectify_by_card_plane(image, quads)
    slots, slot_meta = build_slots_after_rectification(rectified, quads, Hmat)

    tokens: list[Token] = []
    ocr_meta: dict[str, Any] = {"skipped": bool(no_ocr), "mode": None, "full_passes": 0, "crop_passes": 0}
    if not no_ocr:
        if ocr_json:
            tokens = load_tokens_json(ocr_json)
            ocr_meta = {"mode": "json", "full_passes": 0, "crop_passes": 0, "tokens_loaded": len(tokens)}
        elif backend == "paddle":
            # No temp file is needed; PaddleOCR receives the rectified ndarray.
            tokens, ocr_meta = ocr_paddle("", rectified, slots, item_names, mode=ocr_mode, use_angle_cls=paddle_angle_cls)
        elif backend == "tesseract":
            tmp = str(Path("/tmp") / f"shop_rectified_{image_path.stem}.jpg")
            cv2.imwrite(tmp, rectified)
            tokens = ocr_tesseract(tmp)
            ocr_meta = {"mode": "tesseract_full", "full_passes": 1, "crop_passes": 0}
        else:
            raise RuntimeError(f"unknown backend: {backend}")
    assign_tokens_to_slots(tokens, slots)

    items = [result_item_from_slot(s, item_names, refs, rectified) for s in sorted(slots, key=lambda z: (z.row, z.col))]
    uid_roi = default_uid_footer_roi(rectified.shape)
    uid_obj = parse_uid(tokens, rectified.shape, uid_roi=uid_roi) if tokens else None
    refresh_obj = parse_refresh(tokens) if tokens else None
    out = {
        "image": str(image_path),
        "uid": compact_uid(uid_obj),
        "refresh": compact_refresh(refresh_obj),
        "items": items,
    }
    if include_meta:
        out["meta"] = {
            "detected_cards": len(items),
            "original_card_quads": len(quads),
            "ocr": ocr_meta,
        }
    if debug_path is not None:
        draw_final_debug(rectified, sorted(slots, key=lambda z: (z.row, z.col)), items, str(debug_path), uid_obj=uid_obj, refresh_obj=refresh_obj, tokens=tokens)
    return out

def collect_input_images(input_path: Path, recursive: bool = False) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return list_images(input_path, recursive=recursive)
    raise FileNotFoundError(input_path)

def main():
    ap = argparse.ArgumentParser(description="Batch shop OCR + no-namebar icon matching. Output is a JSON array.")
    ap.add_argument("input", type=Path, help="single image or folder of images")
    ap.add_argument("--refs", type=Path, default=None, help="folder of clean item images; file stem is item name; also used as the adaptive item-name dictionary")
    ap.add_argument("--items", default=None, help="optional item-name whitelist txt; ignored when --refs is provided")
    ap.add_argument("--out", type=Path, default=Path("shop_batch_result.json"), help="output JSON array path")
    ap.add_argument("--debug-dir", type=Path, default=None, help="if set, write exactly one final overlay debug image per input image")
    ap.add_argument("--backend", choices=["paddle", "tesseract"], default="paddle")
    ap.add_argument("--ocr-mode", choices=["fast", "smart", "full"], default="fast", help="fast=one full-image OCR; smart=small fallback ROIs; full=slow debug")
    ap.add_argument("--paddle-angle-cls", action="store_true", help="turn on Paddle angle classifier; slower and usually unnecessary after rectification")
    ap.add_argument("--no-ocr", action="store_true", help="skip OCR; useful for testing detection/icon matching only")
    ap.add_argument("--ocr-json", default=None, help="OCR tokens in rectified-image coordinates; only sensible for a single image")
    ap.add_argument("--recursive", action="store_true", help="scan image folders recursively")
    ap.add_argument("--include-meta", action="store_true", help="include detection/OCR metadata in JSON output; off by default")
    args = ap.parse_args()

    refs = load_ref_items(args.refs, recursive=args.recursive)
    item_names = load_item_names_from_refs_or_file(args.items, refs)
    images = collect_input_images(args.input, recursive=args.recursive)
    if args.debug_dir:
        args.debug_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for idx, img_path in enumerate(images):
        dbg = None
        if args.debug_dir:
            dbg = args.debug_dir / f"{img_path.stem}_debug.jpg"
        try:
            res = recognize_one_compact(
                img_path,
                refs=refs,
                item_names=item_names,
                backend=args.backend,
                no_ocr=args.no_ocr,
                ocr_json=args.ocr_json if len(images) == 1 else None,
                ocr_mode=args.ocr_mode,
                paddle_angle_cls=args.paddle_angle_cls,
                debug_path=dbg,
                include_meta=args.include_meta,
            )
            results.append(res)
            print(f"[{idx+1}/{len(images)}] {img_path.name}: cards={len(res.get('items', []))} uid={res['uid']} refresh={res['refresh']}")
        except Exception as e:
            results.append({"image": str(img_path), "error": str(e), "uid": None, "refresh": None, "items": []})
            print(f"[{idx+1}/{len(images)}] {img_path.name}: ERROR {e}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
