# endfield_ocr/pipeline/detector.py
"""
Card quadrilateral detection and full-image perspective rectification.
Detects item-card boundaries on the raw image, filters by geometric constraints,
and computes a global homography to warp the entire screenshot into a canonical UI layout.
"""
from pathlib import Path

import cv2
import numpy as np
import math
from typing import Optional, Dict, List, Tuple


from ..utils.geometry import order_quad_points, box_center_size, transform_points
from ..config import DetectionConfig

def detect_card_quads(image: np.ndarray, config: DetectionConfig, debug_path: Optional[Path] = None) -> List[Dict]:
    """
    Detect item-card quadrilaterals on the original photo/screenshot.
    Uses multi-threshold Canny, contour area/aspect-ratio filtering,
    convex hull approximation, and NMS to isolate stable UI card anchors.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, config.gaussian_blur_kernel, 0)

    # Dual-threshold Canny to catch both bright white borders and dark sold-out cards
    (low1, high1), (low2, high2) = config.canny_thresholds
    edges_high = cv2.Canny(blur, low1, high1)
    edges_low = cv2.Canny(blur, low2, high2)
    edges = cv2.bitwise_or(edges_high, edges_low)
    kernel = np.ones((config.dilation_kernel_size, config.dilation_kernel_size), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Optional debug: save edge image
    if debug_path:
        debug_dir = debug_path.parent
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_path.with_suffix('.edges.png')), edges)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    quads: List[Dict] = []
    img_area = h * w
    debug_vis = image.copy() if debug_path else None

    for c in contours:
        area = cv2.contourArea(c)
        # Filter by relative area: ignore tiny artifacts and full-screen backgrounds
        if area <= img_area * config.card_area_ratio[0] or area >= img_area * config.card_area_ratio[1]:
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        cx = x + bw / 2.0
        cy = y + bh / 2.0

        # Exclude top navigation tabs and bottom system buttons, but leave margin for camera borders
        if cy < h * config.exclude_top_ratio or cy > h * config.exclude_bottom_ratio:
            continue
        if bw < w * config.min_bbox_width_ratio or bh < h * config.min_bbox_height_ratio:
            continue

        ar = bw / max(1.0, bh)
        if ar <= config.card_aspect_ratio[0] or ar >= config.card_aspect_ratio[1]:
            continue

        # Fit polygon to convex hull
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, config.approx_epsilon_ratio * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
        else:
            pts = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)

        pts = order_quad_points(pts)
        qcx, qcy, qw, qh = box_center_size(pts)
        qar = qw / max(1.0, qh)

        if qar <= config.quad_aspect_ratio[0] or qar >= config.quad_aspect_ratio[1]:
            continue

        quads.append({
            "pts": pts,
            "cx": qcx,
            "cy": qcy,
            "area": float(area),
            "rect": (x, y, bw, bh)
        })

        if debug_vis is not None:
            cv2.drawContours(debug_vis, [c], -1, (0, 255, 0), 2)

    if debug_vis is not None and debug_path:
        cv2.imwrite(str(debug_path.with_suffix('.contours.png')), debug_vis)

    # Sort by area descending, then apply NMS to remove overlapping duplicates
    quads.sort(key=lambda q: q["area"], reverse=True)
    kept: List[Dict] = []

    for q in quads:
        is_duplicate = False
        for k in kept:
            dist = math.hypot(q["cx"] - k["cx"], q["cy"] - k["cy"])
            threshold = min(w, h) * config.nms_distance_ratio
            if dist <= threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(q)

    # Final sort: top-to-bottom, then left-to-right
    kept.sort(key=lambda q: (q["cy"], q["cx"]))

    # Debug: draw final quads
    if debug_path and kept:
        debug_final = image.copy()
        for q in kept:
            pts = q["pts"].reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(debug_final, [pts], True, (0, 0, 255), 3)
        cv2.imwrite(str(debug_path.with_suffix('.quads.png')), debug_final)

    return kept

def group_quads_rows(quads: List[Dict]) -> List[List[Dict]]:
    """
    Groups detected card quadrilaterals into horizontal rows.
    Assigns canonical row and column indices based on Y-clustering and dynamic X-pitch.
    Preserves missing-card gaps to prevent homography compression artifacts.
    """
    if not quads:
        return []
        
    heights = [box_center_size(q["pts"])[3] for q in quads]
    med_height = float(np.median(heights))
    tol = max(30.0, med_height * 0.50)
    
    rows: List[List[Dict]] = []
    for q in sorted(quads, key=lambda z: z["cy"]):
        placed = False
        for row in rows:
            row_mean_cy = float(np.mean([r["cy"] for r in row]))
            if abs(q["cy"] - row_mean_cy) <= tol:
                row.append(q)
                placed = True
                break
        if not placed:
            rows.append([q])
            
    # Assign row and column indices per group
    for r_idx, row in enumerate(rows):
        row.sort(key=lambda z: z["cx"])
        xs = [float(q["cx"]) for q in row]
        
        # Compute horizontal gaps between adjacent cards
        gaps = []
        for i in range(len(xs) - 1):
            gap = xs[i + 1] - xs[i]
            if gap > 0:
                gaps.append(gap)
                
        if gaps:
            cut = float(np.percentile(gaps, 60))
            small_gaps = [g for g in gaps if g <= cut]
            pitch = float(np.median(small_gaps or gaps))
        else:
            pitch = 1.0
            
        col = 0
        for i, q in enumerate(row):
            if i == 0:
                col = 0
            else:
                gap = xs[i] - xs[i - 1]
                col_increment = max(1, int(round(gap / max(1.0, pitch))))
                col += col_increment
            q["row"] = r_idx
            q["col"] = col
            
    return rows


def rectify_by_card_plane(image: np.ndarray, quads: List[Dict], max_output_side: int = 3000) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Computes a global homography from detected card corners and warps the full image.
    Uses RANSAC for robustness against outliers and scales the output to prevent OOM.
    Returns (rectified_image, full_homography_matrix, metadata_dict).
    """
    h, w = image.shape[:2]
    rows = group_quads_rows(quads)
    
    # Minimum anchor requirement for stable homography
    if len(quads) < 4 or not rows or max(len(r) for r in rows) < 2:
        return image.copy(), np.eye(3, dtype=np.float64), {
            "used": False, 
            "reason": "not enough card quads", 
            "cards_detected": len(quads)
        }
        
    # Target canonical layout dimensions
    out_card_w = 260.0
    out_card_h = 330.0
    gap_x = 20.0
    pitch_y = 380.0
    margin_x = 180.0
    margin_y = 190.0
    
    src_pts: List[np.ndarray] = []
    dst_pts: List[np.ndarray] = []
    
    for row in rows:
        for q in row:
            c = q["col"]
            r = q["row"]
            x = margin_x + c * (out_card_w + gap_x)
            y = margin_y + r * pitch_y
            dst_quad = np.array([
                [x, y],
                [x + out_card_w, y],
                [x + out_card_w, y + out_card_h],
                [x, y + out_card_h]
            ], dtype=np.float32)
            src_pts.append(q["pts"])
            dst_pts.append(dst_quad)
            
    src_array = np.array(src_pts, dtype=np.float32).reshape(-1, 2)
    dst_array = np.array(dst_pts, dtype=np.float32).reshape(-1, 2)
    
    H, inliers = cv2.findHomography(src_array, dst_array, cv2.RANSAC, 5.0)
    
    if H is None:
        return image.copy(), np.eye(3, dtype=np.float64), {
            "used": False, 
            "reason": "homography failed", 
            "cards_detected": len(quads)
        }
        
    # Transform original image corners to estimate output canvas size
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    tr = transform_points(H, corners)
    
    min_pt = tr.min(axis=0) - 50
    max_pt = tr.max(axis=0) + 50
    
    out_w = int(math.ceil(max_pt[0] - min_pt[0]))
    out_h = int(math.ceil(max_pt[1] - min_pt[1]))
    
    # Scale down if output exceeds memory limit
    scale = 1.0
    if max(out_w, out_h) > max_output_side:
        scale = max_output_side / max(out_w, out_h)
        
    # Build translation + scaling matrix T
    T = np.array([
        [scale, 0, -min_pt[0] * scale],
        [0, scale, -min_pt[1] * scale],
        [0, 0, 1]
    ], dtype=np.float64)
    
    H_full = T @ H
    out_w = max(1, int(out_w * scale))
    out_h = max(1, int(out_h * scale))
    
    rectified = cv2.warpPerspective(image, H_full, (out_w, out_h), borderValue=(245, 245, 245))
    
    inlier_count = int(inliers.sum()) if inliers is not None else None
    
    return rectified, H_full, {
        "used": True,
        "cards_detected": len(quads),
        "rows_detected": [len(r) for r in rows],
        "inliers": inlier_count,
        "output_size": [out_w, out_h]
    }