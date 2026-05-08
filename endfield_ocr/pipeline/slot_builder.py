# endfield_ocr/pipeline/slot_builder.py
"""
Slot generation, namebar detection, and grid modeling after perspective rectification.
Learns card scale dynamically from the rectified image. Separates card body rectangles
from namebar extensions to prevent ROI normalization inconsistencies.
All intermediate anchors are stored as dictionaries for pipeline flexibility.
"""
import cv2
import numpy as np
import math
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass

from ..models import Slot
from ..utils.geometry import clip_rect, transform_points


@dataclass
class GridModel:
    """
    Learned geometric model for the shop UI grid.
    Stores dynamically inferred card dimensions, row alignment, and pitch.
    """
    card_w: float
    body_h: float
    x_pitch: float
    row_tops: List[float]
    row_centers: List[List[float]]
    bar_h: float = 0.0
    source: str = "white_body_then_namebar"


def robust_percentile(values: List[float], p: float, default: float = 1.0) -> float:
    """
    Computes the p-th percentile of a list, ignoring non-finite and non-positive values.
    Returns default if the input list contains no valid numbers.
    """
    arr = np.asarray([float(v) for v in values if np.isfinite(v) and v > 0], dtype=np.float32)
    if arr.size == 0:
        return float(default)
    return float(np.percentile(arr, p))


def robust_median(values: List[float], default: float = 1.0) -> float:
    """
    Computes the median of a list, ignoring non-finite and non-positive values.
    Returns default if the input list contains no valid numbers.
    """
    arr = np.asarray([float(v) for v in values if np.isfinite(v) and v > 0], dtype=np.float32)
    if arr.size == 0:
        return float(default)
    return float(np.median(arr))


def reject_size_outliers(values: List[float]) -> List[float]:
    """
    Data-driven size filter using IQR. Rejects extreme outliers without hardcoded pixel boundaries.
    Returns original list if fewer than 4 samples are provided.
    """
    vals = [float(v) for v in values if np.isfinite(v) and v > 0]
    if len(vals) < 4:
        return vals
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = max(1.0, float(q3 - q1))
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    kept = [v for v in vals if lower_bound <= v <= upper_bound]
    return kept if kept else vals


def projected_rects_from_quads(quads: List[Dict], H_full: np.ndarray, rect_shape: Tuple[int, int]) -> List[Dict]:
    """
    Projects original card quadrilaterals onto the rectified image.
    These rectangles are used as approximate WHITE CARD BODY anchors only.
    They are never trusted as final boxes due to potential contour inaccuracies.
    """
    h, w = rect_shape[:2]
    out: List[Dict] = []
    for q in quads:
        pts = transform_points(H_full, q["pts"])
        min_pt = pts.min(axis=0)
        max_pt = pts.max(axis=0)
        l, t = min_pt[0], min_pt[1]
        r, b = max_pt[0], max_pt[1]
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


def detect_rectified_card_edge_rects(rectified: np.ndarray, model: GridModel) -> List[Dict]:
    """
    Detects card/body rectangles directly on the rectified image using edge contours.
    Supplements the original-photo quad detector. Driven entirely by learned card scale.
    Only used as white/body anchors; final slot dimensions are fixed later.
    """
    if model.card_w <= 1 or model.body_h <= 1:
        return []
    h, w = rectified.shape[:2]
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    e1 = cv2.Canny(blur, 24, 92)
    e2 = cv2.Canny(blur, 12, 56)
    edges = cv2.bitwise_or(e1, e2)
    k = max(3, int(round(model.card_w * 0.018)))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    raw: List[Dict] = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw <= 0 or bh <= 0:
            continue
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
        score = fill * 2.0
        score -= abs(bw - model.card_w) / max(1.0, model.card_w)
        score -= 0.35 * abs(bh - model.body_h) / max(1.0, model.body_h)
        raw.append({
            "rect": (float(x), float(y), float(x + bw), float(y + bh)),
            "cx": float(x + bw / 2),
            "cy": float(y + bh / 2),
            "w": float(bw),
            "h": float(bh),
            "source": "rectified_edge",
            "score": float(score),
        })
    raw.sort(key=lambda z: float(z.get("score", 0.0)), reverse=True)
    kept: List[Dict] = []
    for rec in raw:
        is_duplicate = False
        for k_rect in kept:
            dist = math.hypot(rec["cx"] - k_rect["cx"], rec["cy"] - k_rect["cy"])
            threshold = model.card_w * 0.32
            if dist <= threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(rec)
    return sorted(kept, key=lambda z: (z["cy"], z["cx"]))


def merge_anchor_rects(rects: List[Dict], card_w: float) -> List[Dict]:
    """
    Merges projected and rectified-edge anchors without allowing giant boxes.
    Keeps the best representative per physical card center.
    """
    if not rects:
        return []
    for r in rects:
        if "score" not in r:
            if r.get("source") == "rectified_edge":
                r["score"] = 1.0
            else:
                r["score"] = 1.15
    rects = sorted(rects, key=lambda z: float(z.get("score", 1.0)), reverse=True)
    kept: List[Dict] = []
    for r in rects:
        is_duplicate = False
        for k in kept:
            dist = math.hypot(r["cx"] - k["cx"], r["cy"] - k["cy"])
            threshold = card_w * 0.34
            if dist <= threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(r)
    return sorted(kept, key=lambda z: (z["cy"], z["cx"]))


def complete_interior_grid_gaps(rects: List[Dict], model: GridModel, rect_shape: Tuple[int, int]) -> List[Dict]:
    """
    Fills only obvious interior gaps in a row.
    Recovers a card whose contour was missed between two detected cards,
    but does not invent cards beyond the visible left/right extent of a row.
    """
    if not rects or model.x_pitch <= 1 or model.card_w <= 1:
        return rects
    h, w = rect_shape[:2]
    rows = group_rects_rows(rects)
    out: List[Dict] = list(rects)
    for row in rows:
        if len(row) < 2:
            continue
        row = sorted(row, key=lambda z: z["cx"])
        row_top = robust_median([x["rect"][1] for x in row], default=row[0]["rect"][1])
        for a, b in zip(row, row[1:]):
            gap = b["cx"] - a["cx"]
            if gap <= model.x_pitch * 1.45:
                continue
            n_missing = int(round(gap / model.x_pitch)) - 1
            if n_missing <= 0 or n_missing > 3:
                continue
            for k in range(1, n_missing + 1):
                cx = a["cx"] + k * gap / (n_missing + 1)
                l = cx - model.card_w / 2
                r = cx + model.card_w / 2
                t = row_top
                bb = row_top + model.body_h
                l2, t2, r2, b2 = clip_rect((l, t, r, bb), w, h, pad=0)
                if r2 <= l2 or b2 <= t2:
                    continue
                out.append({
                    "rect": (float(l2), float(t2), float(r2), float(b2)),
                    "cx": float((l2 + r2) / 2),
                    "cy": float((t2 + b2) / 2),
                    "w": float(r2 - l2),
                    "h": float(b2 - t2),
                    "source": "grid_gap_completion",
                    "score": 0.55,
                })
    return merge_anchor_rects(out, model.card_w)


def group_rects_rows(rects: List[Dict]) -> List[List[Dict]]:
    """
    Groups anchor rectangles into horizontal rows based on vertical center clustering.
    Uses robust median to compute tolerance, then sorts each row left-to-right.
    """
    if not rects:
        return []
    hs = reject_size_outliers([r["h"] for r in rects])
    med_h = robust_median(hs, default=100.0)
    tol = med_h * 0.55
    rows: List[List[Dict]] = []
    for rec in sorted(rects, key=lambda z: z["cy"]):
        placed = False
        for row in rows:
            row_centers = [float(x["cy"]) for x in row]
            row_mean = float(np.median(row_centers))
            if abs(rec["cy"] - row_mean) <= tol:
                row.append(rec)
                placed = True
                break
        if not placed:
            rows.append([rec])
    for row in rows:
        row.sort(key=lambda z: z["cx"])
    return rows


def _estimate_pitch(xs: List[float], default: float) -> float:
    """
    Estimates horizontal grid pitch from card centers.
    Uses lower percentiles of gaps to ignore 2x-pitch gaps caused by missing cards.
    """
    xs_sorted = sorted(float(x) for x in xs)
    gaps = []
    for i in range(len(xs_sorted) - 1):
        gap = xs_sorted[i + 1] - xs_sorted[i]
        if gap > 0:
            gaps.append(gap)
    if not gaps:
        return default
    cut = robust_percentile(gaps, 60, robust_median(gaps, default))
    small_gaps = [g for g in gaps if g <= cut]
    return robust_median(small_gaps or gaps, default)


def estimate_initial_model(projected_rects: List[Dict]) -> GridModel:
    """
    Learns card width, row tops, and a BODY-height hint from detected cards.
    Uses the lower cluster of heights to avoid accidentally measuring namebars as part of the body.
    """
    if not projected_rects:
        return GridModel(1.0, 1.0, 1.0, [], [], 0.0, "empty")
    widths = reject_size_outliers([r["w"] for r in projected_rects])
    heights = reject_size_outliers([r["h"] for r in projected_rects])
    card_w = robust_median(widths, 1.0)
    h_cut = robust_percentile(heights, 60, robust_median(heights, card_w * 1.25))
    lower_h = [h for h in heights if h <= h_cut]
    body_h_hint = robust_median(lower_h or heights, card_w * 1.25)
    rows = group_rects_rows(projected_rects)
    row_tops: List[float] = []
    row_centers: List[List[float]] = []
    all_xs: List[float] = []
    for row in rows:
        row_tops.append(robust_median([x["rect"][1] for x in row], default=row[0]["rect"][1]))
        xs = [float(x["cx"]) for x in row]
        row_centers.append(sorted(xs))
        all_xs.extend(xs)
    x_pitch = _estimate_pitch(all_xs, card_w * 1.06)
    if x_pitch < card_w * 0.80:
        x_pitch = card_w * 1.03
    return GridModel(card_w=card_w, body_h=body_h_hint, x_pitch=x_pitch, row_tops=row_tops, row_centers=row_centers)


def detect_local_namebar_band(rectified: np.ndarray, body_rect: Tuple[float, float, float, float]) -> Optional[Tuple[float, float, float, float, float]]:
    """
    Finds a bottom name strip ONLY if it touches the detected white-card body.
    Scans a very tight vertical window around the body bottom edge to avoid footer buttons.
    Returns (left, top, right, bottom, confidence_score) or None.
    """
    h, w = rectified.shape[:2]
    l, t, r, b = body_rect
    card_w = max(1.0, r - l)
    body_h = max(1.0, b - t)
    x1 = max(0, int(round(l + card_w * 0.015)))
    x2 = min(w, int(round(r - card_w * 0.015)))
    pre = max(2, int(round(body_h * 0.045)))
    post = max(8, int(round(body_h * 0.20)))
    y1 = max(0, int(round(b - pre)))
    y2 = min(h, int(round(b + post)))
    if x2 <= x1 or y2 <= y1:
        return None
    roi = rectified[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thr = min(138.0, max(30.0, float(np.percentile(gray, 30))))
    dark_thr = max(thr, 108.0)
    dark = gray <= dark_thr
    row_ratio = dark.mean(axis=1)
    win = max(3, int(round(body_h * 0.014)))
    smoothed = np.convolve(row_ratio, np.ones(win, dtype=np.float32) / win, mode="same")
    runs: List[Tuple[float, float, float, float, float]] = []
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
    runs.sort(key=lambda z: (abs(z[1] - b), -z[4]))
    return runs[0]


def refine_body_height_with_namebars(rectified: np.ndarray, projected_rects: List[Dict], model: GridModel) -> GridModel:
    """
    Two-pass geometry: finds strips first, then infers true white-body height.
    Updates model.body_h and model.bar_h based on localized namebar detections.
    """
    rows = group_rects_rows(projected_rects)
    body_h_samples: List[float] = []
    bar_h_samples: List[float] = []
    for row_idx, row in enumerate(rows):
        if row_idx < len(model.row_tops):
            row_top = model.row_tops[row_idx]
        else:
            row_top = robust_median([x["rect"][1] for x in row], default=row[0]["rect"][1])
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
        body_h = robust_median(body_h_samples, default=model.body_h)
    else:
        body_h = model.body_h
    if bar_h_samples:
        bar_h = robust_median(bar_h_samples, default=model.body_h * 0.09)
    else:
        bar_h = model.body_h * 0.0
    return GridModel(model.card_w, body_h, model.x_pitch, model.row_tops, model.row_centers, bar_h=bar_h, source=model.source)


def detect_global_namebar_components(rectified: np.ndarray, model: GridModel) -> List[Dict]:
    """
    Finds extra name bars for cards that the quad detector missed.
    Only supplements the grid. Components must be close to learned rows and expected body bottoms.
    """
    if not model.row_tops or model.card_w <= 1 or model.body_h <= 1:
        return []
    h, w = rectified.shape[:2]
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    thr = min(145.0, max(35.0, float(np.percentile(gray, 25))))
    dark = (gray < thr).astype(np.uint8) * 255
    kx = max(3, int(round(model.card_w * 0.12)))
    ky = max(1, int(round(model.body_h * 0.015)))
    mask = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((kx, ky), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((max(3, kx // 3), max(1, ky)), np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands: List[Dict[str, Any]] = []
    
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if not (model.card_w * 0.65 <= bw <= model.card_w * 1.25):
            continue
        if not (model.body_h * 0.025 <= bh <= model.body_h * 0.20):
            continue
        cx = x + bw / 2.0
        cy = y + bh / 2.0
        best_row = None
        best_d = 1e18
        for ri, row_top in enumerate(model.row_tops):
            expected = row_top + model.body_h + max(model.bar_h, model.body_h * 0.08) / 2.0
            d = abs(cy - expected)
            if d < best_d:
                best_d = d
                best_row = ri
        if best_row is None or best_d > model.body_h * 0.20:
            continue
        
        rect_dict: Dict[str, Any] = {
            "row": best_row,
            "cx": float(cx),
            "rect": (float(x), float(y), float(x + bw), float(y + bh))
        }
        cands.append(rect_dict)
        
    # NMS by row and horizontal center.
    cands.sort(key=lambda z: (z["row"], z["cx"]))
    kept: List[Dict[str, Any]] = []
    for cand in cands:
        is_duplicate = False
        for k in kept:
            same_row = cand["row"] == k["row"]
            close_cx = abs(cand["cx"] - k["cx"]) < model.card_w * 0.35
            if same_row and close_cx:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(cand)
            
    return kept

def _renumber_slots(slots: List[Slot]) -> List[Slot]:
    """
    Regroups slots by vertical center, then sorts left-to-right.
    Removes bad column numbers inherited from partial rows and resets IDs sequentially.
    """
    if not slots:
        return []
    rows: List[List[Slot]] = []
    med_h = robust_median([s.rect[3] - s.rect[1] for s in slots], default=100.0)
    tol = med_h * 0.45
    for s in sorted(slots, key=lambda z: (z.rect[1] + z.rect[3]) / 2.0):
        cy = (s.rect[1] + s.rect[3]) / 2.0
        placed = False
        for row in rows:
            rcy = float(np.median([(x.rect[1] + x.rect[3]) / 2.0 for x in row]))
            if abs(cy - rcy) <= tol:
                row.append(s)
                placed = True
                break
        if not placed:
            rows.append([s])
    out: List[Slot] = []
    sid = 0
    for ri, row in enumerate(rows):
        row.sort(key=lambda z: (z.rect[0] + z.rect[2]) / 2.0)
        for ci, s in enumerate(row):
            s.id = sid
            s.row = ri
            s.col = ci
            out.append(s)
            sid += 1
    return out


def build_slots_after_rectification(rectified: np.ndarray, quads: List[Dict], H_full: np.ndarray) -> Tuple[List[Slot], Dict]:
    """
    Main entry point. Builds Slot objects from projected quads and learned geometry.
    Returns (list_of_slots, metadata_dict_for_debug).
    """
    projected = projected_rects_from_quads(quads, H_full, rectified.shape)
    if not projected:
        return [], {
            "projected_rects": 0,
            "rectified_edge_rects": 0,
            "final_slots": 0,
            "reason": "no projected card anchors"
        }
    initial = estimate_initial_model(projected)
    edge_rects = detect_rectified_card_edge_rects(rectified, initial)
    anchors0 = merge_anchor_rects(projected + edge_rects, initial.card_w)
    initial2 = estimate_initial_model(anchors0)
    anchors0 = complete_interior_grid_gaps(anchors0, initial2, rectified.shape)
    model = refine_body_height_with_namebars(rectified, anchors0, initial2)
    anchors = complete_interior_grid_gaps(anchors0, model, rectified.shape)
    h, w = rectified.shape[:2]
    rows = group_rects_rows(anchors)
    slots: List[Slot] = []
    sid = 0
    for ri, row in enumerate(rows):
        if ri < len(model.row_tops):
            row_top = model.row_tops[ri]
        else:
            row_top = robust_median([x["rect"][1] for x in row], default=row[0]["rect"][1])
        for rec in sorted(row, key=lambda z: z["cx"]):
            cx = rec["cx"]
            l = cx - model.card_w / 2
            r = cx + model.card_w / 2
            body_rect = (l, row_top, r, row_top + model.body_h)
            bar = detect_local_namebar_band(rectified, body_rect)
            l2, t2, r2, b2 = clip_rect(body_rect, w, h, pad=2)
            nb = None
            source = "white_body_only_no_namebar"
            if bar is not None:
                bl, bt, br, bb, _score = bar
                nl, nt, nr, nbottom = clip_rect((l, bt, r, bb), w, h, pad=1)
                if nr > nl and nbottom > nt:
                    nb = (float(nl), float(nt), float(nr), float(nbottom))
                    source = "white_body_plus_local_namebar"
            if r2 > l2 and b2 > t2:
                slots.append(Slot(sid, ri, 0, (float(l2), float(t2), float(r2), float(b2)), namebar_rect=nb, source=source))
                sid += 1
    global_bars = detect_global_namebar_components(rectified, model)
    for gb in global_bars:
        row = gb["row"]
        cx = gb["cx"]
        already_exists = False
        for s in slots:
            if s.row == row:
                center_diff = abs(((s.rect[0] + s.rect[2]) / 2.0) - cx)
                if center_diff < model.card_w * 0.45:
                    already_exists = True
                    break
        if already_exists:
            continue
        if row >= len(model.row_tops):
            continue
        row_top = model.row_tops[row]
        l = cx - model.card_w / 2
        r = cx + model.card_w / 2
        body = clip_rect((l, row_top, r, row_top + model.body_h), w, h, pad=2)
        nb = clip_rect((l, gb["rect"][1], r, gb["rect"][3]), w, h, pad=1)
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