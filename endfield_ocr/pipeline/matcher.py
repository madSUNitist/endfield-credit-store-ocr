# endfield_ocr/pipeline/matcher.py
"""
Icon-based name matching for cards without a detected namebar.
Uses multi-scale LAB-NCC template matching with dynamic ignore masks.
Falls back to partial Canny edge scoring if color matching confidence is low.
All functions are stateless except for reference image caching.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Iterable, Tuple, Any
from dataclasses import dataclass

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class RefItem:
    """
    Preprocessed reference item image for template matching.
    Contains BGR, LAB, binary mask, and dilated edge map.
    """
    name: str
    path: Path
    bgr: np.ndarray
    lab: np.ndarray
    mask255: np.ndarray
    edge01: np.ndarray


@dataclass
class MatchCardItem:
    """
    Preprocessed cropped card region for matching.
    Contains ignore mask (valid01), LAB representation, and edge map.
    """
    name: str
    full_bgr: np.ndarray
    roi_bgr: np.ndarray
    roi_lab: np.ndarray
    edge01: np.ndarray
    valid01: np.ndarray


def list_images(folder: Path, recursive: bool = False) -> List[Path]:
    """
    Scans a directory for image files supported by the matcher.
    Ignores macOS metadata files starting with '._'.
    """
    if not folder.exists():
        return []
        
    if recursive:
        iterator = folder.rglob("*")
    else:
        iterator = folder.iterdir()
        
    results: List[Path] = []
    for p in iterator:
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMG_EXTS:
            continue
        if p.name.startswith("._"):
            continue
        results.append(p)
        
    results.sort()
    return results


def read_ref(path: Path, max_side: int = 110, pad: int = 4) -> RefItem:
    """
    Loads a reference image, extracts foreground alpha mask, crops to content,
    resizes to max_side, and generates LAB + Canny edge representations.
    """
    rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if rgba is None:
        raise FileNotFoundError(f"Cannot read reference image: {path}")
        
    # Separate BGR and alpha channels based on image depth
    if rgba.ndim == 2:
        bgr0 = cv2.cvtColor(rgba, cv2.COLOR_GRAY2BGR)
        alpha0 = (rgba > 5).astype(np.uint8) * 255
    elif rgba.shape[2] == 4:
        bgr0 = rgba[:, :, :3]
        alpha0 = rgba[:, :, 3].astype(np.uint8)
    else:
        bgr0 = rgba[:, :, :3]
        gray0 = cv2.cvtColor(bgr0, cv2.COLOR_BGR2GRAY)
        alpha0 = (gray0 > 5).astype(np.uint8) * 255
        
    # Find bounding box of visible foreground
    ys, xs = np.where(alpha0 > 20)
    if len(xs) == 0:
        raise ValueError(f"Empty foreground in reference image: {path}")
        
    y1 = max(0, int(ys.min()) - pad)
    y2 = min(int(alpha0.shape[0]), int(ys.max()) + pad + 1)
    x1 = max(0, int(xs.min()) - pad)
    x2 = min(int(alpha0.shape[1]), int(xs.max()) + pad + 1)
    
    bgr = bgr0[y1:y2, x1:x2]
    alpha = alpha0[y1:y2, x1:x2]
    
    # Resize if reference is too large
    h, w = bgr.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        
    # Generate binary foreground mask
    mask255: np.ndarray = (alpha > 20).astype(np.uint8) * 255
    if min(mask255.shape) >= 5:
        mask255 = cv2.erode(mask255, np.ones((3, 3), np.uint8), iterations=1)
        
    # cv2.cvtColor returns generic np.ndarray; do not annotate with specific dtype
    lab: np.ndarray = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    
    # Generate edge map only within foreground mask
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 40, 120, L2gradient=True)
    edge01: np.ndarray = ((edges > 0) & (mask255 > 0)).astype(np.uint8)
    edge01 = cv2.dilate(edge01, np.ones((2, 2), np.uint8), iterations=1)
    
    return RefItem(
        name=path.stem,
        path=path,
        bgr=bgr,
        lab=lab,
        mask255=mask255,
        edge01=edge01
    )


def build_ignore_mask(roi_bgr: np.ndarray) -> np.ndarray:
    """
    Builds a binary mask marking UI elements that should be ignored during matching.
    Excludes: discount badge (top-right), price/name area (bottom), 
    count badges (center), and sold-out overlays (dark center rectangles).
    """
    h, w = roi_bgr.shape[:2]
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    
    ignore = np.zeros((h, w), dtype=np.uint8)
    
    # Hard-ignore top-right discount area
    top_cutoff = int(0.20 * h)
    left_cutoff = int(0.62 * w)
    ignore[:top_cutoff, left_cutoff:] = 1
    
    # Hard-ignore bottom price/name extension inside body crop
    bottom_start = int(0.82 * h)
    ignore[bottom_start:, :] = 1
    
    # Detect bright white UI badges
    bright_white = ((hsv[:, :, 2] > 175) & (hsv[:, :, 1] < 85)).astype(np.uint8)
    bright_white[: int(0.28 * h), :] = 0
    bright_white[int(0.85 * h) :, :] = 0
    
    # Analyze connected components of bright regions
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bright_white, connectivity=8)
    mean_gray = float(gray.mean())
    
    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        
        if area < 20:
            continue
            
        cx = x + ww / 2.0
        cy = y + hh / 2.0
        
        is_count_badge = (
            (0.35 * h < cy < 0.83 * h) and 
            (10 <= ww <= 115) and 
            (7 <= hh <= 60)
        )
        is_sold_overlay = (
            (mean_gray < 155) and 
            (0.25 * h < cy < 0.72 * h) and 
            (5 <= ww <= 155) and 
            (5 <= hh <= 75)
        )
        
        if is_count_badge or is_sold_overlay:
            ignore[y : y + hh, x : x + ww] = 1
            
    # Dilate ignore mask to create safe margins around UI clutter
    ignore = cv2.dilate(ignore, np.ones((7, 7), np.uint8), iterations=1).astype(np.uint8)
    return ignore


def read_card_from_bgr(full: np.ndarray, name: str = "slot", roi_width: int = 160) -> MatchCardItem:
    """
    Crops and preprocesses a detected card region for matching.
    Extracts item area, applies ignore mask, and computes LAB/edge representations.
    """
    if full is None or full.size == 0:
        raise ValueError("Empty card crop provided for matching")
        
    h_full, w_full = full.shape[:2]
    
    # Crop to standard item image ROI (approx 2% to 98% width, 7% to 76% height)
    x1 = int(0.02 * w_full)
    x2 = int(0.98 * w_full)
    y1 = int(0.07 * h_full)
    y2 = int(0.76 * h_full)
    roi = full[y1:y2, x1:x2]
    
    if roi.size == 0:
        roi = full.copy()
        
    # Scale width to roi_width, preserve aspect ratio
    h_roi, w_roi = roi.shape[:2]
    scale = roi_width / max(1, w_roi)
    new_w = roi_width
    new_h = max(1, int(round(h_roi * scale)))
    roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    ignore: np.ndarray = build_ignore_mask(roi)
    valid01: np.ndarray = (ignore == 0).astype(np.uint8)
    roi_lab: np.ndarray = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    
    # CLAHE + blur for stable edge extraction under varying lighting
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 35, 110, L2gradient=True)
    edge01: np.ndarray = (edges > 0).astype(np.uint8)
    edge01[ignore > 0] = 0
    
    return MatchCardItem(
        name=name,
        full_bgr=full,
        roi_bgr=roi,
        roi_lab=roi_lab,
        edge01=edge01,
        valid01=valid01
    )


def fast_lab_ncc(ref: RefItem, card: MatchCardItem, scales: Iterable[float]) -> Tuple[float, Optional[Tuple[float, Tuple[int, int], Tuple[int, int]]]]:
    """
    Fast multi-scale LAB color space template matching.
    Replaces masked template pixels with card background color to prevent edge leakage.
    Returns (best_score, best_info) where info contains (scale, top_left_loc, template_size).
    """
    h_card, w_card = card.roi_lab.shape[:2]
    bg_lab = np.median(card.roi_lab[: max(1, min(20, h_card)), : max(1, min(20, w_card))].reshape(-1, 3), axis=0).astype(np.uint8)
    
    best_score = -9.0
    best_info: Optional[Tuple[float, Tuple[int, int], Tuple[int, int]]] = None
    weights = (0.55, 1.0, 1.0)
    weight_sum = sum(weights)
    
    for s in scales:
        th = max(8, int(round(ref.lab.shape[0] * s)))
        tw = max(8, int(round(ref.lab.shape[1] * s)))
        
        if th > h_card or tw > w_card:
            continue
            
        tmpl = cv2.resize(ref.lab, (tw, th), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR)
        mask = cv2.resize(ref.mask255, (tw, th), interpolation=cv2.INTER_NEAREST) > 0
        
        # Replace background with card average to avoid color mismatch at edges
        tmpl2 = tmpl.copy()
        tmpl2[~mask] = bg_lab
        
        res_sum: Optional[np.ndarray] = None
        for ch in range(3):
            res = cv2.matchTemplate(card.roi_lab[:, :, ch], tmpl2[:, :, ch], cv2.TM_CCOEFF_NORMED)
            res = np.nan_to_num(res, nan=-2.0, posinf=-2.0, neginf=-2.0)
            
            if res_sum is None:
                res_sum = res * weights[ch]
            else:
                res_sum = res_sum + res * weights[ch]
                
        if res_sum is not None:
            res_sum /= weight_sum
            _, maxv, _, maxloc = cv2.minMaxLoc(res_sum)
            
            if maxv > best_score:
                best_score = float(maxv)
                best_info = (float(s), (int(maxloc[0]), int(maxloc[1])), (int(tw), int(th)))
                
    return best_score, best_info


def partial_canny_score(ref_edge: np.ndarray, ref_mask01: np.ndarray, card: MatchCardItem, stride: int = 4) -> Tuple[float, Optional[Tuple[int, int]]]:
    """
    Evaluates edge match quality at a single scale using recall and precision.
    Recall: how many reference edges are covered by card edges.
    Precision: how many valid card edges match the reference template.
    """
    h_card, w_card = card.edge01.shape
    th, tw = ref_edge.shape[:2]
    
    if th > h_card or tw > w_card:
        return -1.0, None
        
    target_dil = cv2.dilate(card.edge01, np.ones((3, 3), np.uint8), iterations=1)
    tmpl_dil = cv2.dilate(ref_edge, np.ones((3, 3), np.uint8), iterations=1)
    
    best_score = -1.0
    best_loc: Optional[Tuple[int, int]] = None
    
    for y in range(0, h_card - th + 1, stride):
        for x in range(0, w_card - tw + 1, stride):
            valid = card.valid01[y : y + th, x : x + tw]
            te_visible = ref_edge & valid
            n_te = int(te_visible.sum())
            
            if n_te < 12:
                continue
                
            target_patch = card.edge01[y : y + th, x : x + tw]
            target_in_obj = target_patch & ref_mask01 & valid
            n_target = int(target_in_obj.sum())
            
            overlap = te_visible & target_dil[y : y + th, x : x + tw].astype(np.bool)
            recall = float(overlap.sum()) / n_te
            
            precision = 0.0
            if n_target > 5:
                match = target_in_obj & tmpl_dil.astype(np.bool)
                precision = float(match.sum()) / n_target
                
            score = 0.72 * recall + 0.28 * precision
            
            if score > best_score:
                best_score = score
                best_loc = (x, y)
                
    return best_score, best_loc


def canny_fallback(ref: RefItem, card: MatchCardItem, scales: Iterable[float], stride: int = 4) -> Tuple[float, Optional[Tuple[float, Tuple[int, int], Tuple[int, int]]]]:
    """
    Multi-scale Canny edge fallback when LAB-NCC confidence is too low.
    Slower but robust against color shifts or lighting differences.
    """
    h_card, w_card = card.edge01.shape
    best_score = -1.0
    best_info: Optional[Tuple[float, Tuple[int, int], Tuple[int, int]]] = None
    
    for s in scales:
        th = max(8, int(round(ref.edge01.shape[0] * s)))
        tw = max(8, int(round(ref.edge01.shape[1] * s)))
        
        if th > h_card or tw > w_card:
            continue
            
        te = cv2.resize(ref.edge01, (tw, th), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        tm = cv2.resize((ref.mask255 > 0).astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        
        if int(te.sum()) < 12:
            continue
            
            score, loc = partial_canny_score(te, tm, card, stride=stride)
            
            if score > best_score:
                best_score = float(score)
                if loc is not None:
                    best_info = (float(s), loc, (int(tw), int(th)))
                else:
                    best_info = (float(s), (0, 0), (int(tw), int(th)))
                    
    return best_score, best_info


def load_ref_items(ref_dir: Optional[str | Path], recursive: bool = False) -> List[RefItem]:
    """
    Loads all reference images from a directory into preprocessed RefItem objects.
    Silently skips corrupted or unreadable files.
    """
    if ref_dir is None:
        return []
        
    paths = list_images(Path(ref_dir), recursive=recursive)
    refs: List[RefItem] = []
    
    for p in paths:
        try:
            refs.append(read_ref(p))
        except Exception:
            continue
            
    return refs


def match_card_bgr_to_refs(card_bgr: np.ndarray, refs: List[RefItem], name: str = "slot") -> Optional[Dict[str, Any]]:
    """
    Main matching entry point.
    Attempts fast LAB-NCC across multiple scales. Falls back to Canny if best score < 0.45.
    Returns a dictionary with match name, method, confidence score, and top-5 LAB candidates.
    """
    if not refs or card_bgr is None or card_bgr.size == 0:
        return None
        
    card = read_card_from_bgr(card_bgr, name=name)
    lab_scales = np.linspace(0.62, 1.32, 8)
    edge_scales = np.linspace(0.60, 1.35, 10)
    
    # Phase 1: Fast LAB-NCC matching
    lab_scores: List[Tuple[float, RefItem, Optional[Tuple]]] = []
    for ref in refs:
        score, info = fast_lab_ncc(ref, card, lab_scales)
        lab_scores.append((score, ref, info))
        
    lab_scores.sort(key=lambda x: x[0], reverse=True)
    best_score, best_ref, best_info = lab_scores[0]
    
    second_score = lab_scores[1][0] if len(lab_scores) > 1 else -9.0
    method = "fast_LAB_NCC"
    chosen_score = best_score
    chosen_ref = best_ref
    chosen_info = best_info
    
    # Phase 2: Canny fallback if confidence is insufficient
    if best_score < 0.45:
        edge_scores: List[Tuple[float, RefItem, Optional[Tuple]]] = []
        for ref in refs:
            escore, einfo = canny_fallback(ref, card, edge_scales, stride=4)
            edge_scores.append((escore, ref, einfo))
            
        edge_scores.sort(key=lambda x: x[0], reverse=True)
        if edge_scores:
            chosen_score, chosen_ref, chosen_info = edge_scores[0]
            method = "partial_Canny_fallback"
            
    lab_top5: List[Tuple[str, float]] = []
    for s, r, _ in lab_scores[:5]:
        lab_top5.append((r.name, round(float(s), 4)))
        
    return {
        "name": chosen_ref.name,
        "method": method,
        "score": round(float(chosen_score), 4),
        "lab_best": best_ref.name,
        "lab_score": round(float(best_score), 4),
        "lab_margin": round(float(best_score - second_score), 4),
        "lab_top5": lab_top5,
    }