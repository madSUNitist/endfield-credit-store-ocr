# endfield_ocr/pipeline/parser.py
"""
Field extraction, token assignment, and deduplication logic.
Parses structured UI fields (name, price, discount, quantity, UID, refresh, sold-out)
from OCR tokens spatially assigned to card slots. All ROI filtering uses body-normalized coordinates.
"""
import re
import math
import numpy as np
from typing import Optional, List, Dict, Tuple, Any
from rapidfuzz import fuzz, process

from ..models import Token, Slot
from ..utils.text import normalize_text, normalize_num_text, clean_name, has_chinese, group_tokens_lines
from ..utils.geometry import (
    rect_inter_area, rect_area, token_rect, union_token_box, rect_to_list
)
from ..config import ROIConfig, UIDConfig

ROI_DISCOUNT = ROIConfig.discount
ROI_PRICE = ROIConfig.price
ROI_QUANTITY = ROIConfig.quantity
ROI_SOLDOUT = ROIConfig.sold_out

UID_MIN_LEN = UIDConfig.min_length
UID_MAX_LEN = UIDConfig.max_length


def assign_tokens_to_slots(tokens: List[Token], slots: List[Slot]) -> None:
    """
    Assigns each OCR token to the closest slot by center proximity.
    Clears previous assignments first to ensure idempotent behavior on fallback OCR passes.
    """
    for s in slots:
        s.tokens.clear()
        
    for t in tokens:
        best_slot = None
        best_margin = 1e18
        
        for s in slots:
            if s.contains_token_center(t, margin=4.0):
                l, top, r, b = s.rect
                margin = abs(t.cx - (l + r) / 2.0) + abs(t.cy - (top + b) / 2.0)
                if margin < best_margin:
                    best_slot = s
                    best_margin = margin
                    
        if best_slot is not None:
            t.slot_id = best_slot.id
            best_slot.tokens.append(t)


def tokens_in_region(slot: Slot, roi: Tuple[float, float, float, float]) -> List[Token]:
    """
    Filters tokens whose normalized center falls within the given BODY-relative ROI.
    ROI format: (nx1, ny1, nx2, ny2) where values are typically 0.0 to 1.0.
    """
    nx1, ny1, nx2, ny2 = roi
    filtered_tokens: List[Token] = []
    for t in slot.tokens:
        nx, ny = slot.norm_xy(t)
        if nx1 <= nx <= nx2 and ny1 <= ny <= ny2:
            filtered_tokens.append(t)
    return filtered_tokens


def tokens_in_namebar(slot: Slot) -> List[Token]:
    """Returns tokens whose centers fall within the namebar rectangle."""
    if slot.namebar_rect is None:
        return []
        
    l, t, r, b = slot.namebar_rect
    filtered_tokens: List[Token] = []
    for tok in slot.tokens:
        if l - 3.0 <= tok.cx <= r + 3.0 and t - 3.0 <= tok.cy <= b + 3.0:
            filtered_tokens.append(tok)
    return filtered_tokens


def match_item_name(text: str, item_names: List[str], threshold: float = 68.0) -> Tuple[Optional[str], float]:
    """
    Fuzzy matches cleaned OCR text against a whitelist of canonical item names.
    Uses rapidfuzz for fast ratio scoring. Returns (matched_name, raw_score) or (None, 0.0).
    """
    text_clean = clean_name(text)
    if not text_clean:
        return None, 0.0
        
    name_mapping = {clean_name(n): n for n in item_names}
    match_result = process.extractOne(text_clean, list(name_mapping.keys()), scorer=fuzz.ratio)
    
    if match_result is None:
        return None, 0.0
        
    best_key, score, _ = match_result
    if score >= threshold:
        return name_mapping[best_key], float(score)
    else:
        return None, float(score)


def parse_name(slot: Slot, item_names: List[str]) -> Tuple[Optional[str], Optional[float], bool]:
    """
    Extracts item name from the namebar. Falls back to full-slot tokens only if sold-out overlay occludes the namebar.
    Returns (name, confidence, is_occluded_by_soldout).
    """
    candidates: List[Tuple[str, float]] = []
    
    namebar_toks = sorted(tokens_in_namebar(slot), key=lambda t: (t.cy, t.cx))
    for t in namebar_toks:
        if has_chinese(t.text):
            base_score = t.score * 10.0 + 12.0
            candidates.append((t.text, base_score))
            
    if namebar_toks:
        lines = group_tokens_lines(namebar_toks, tol_factor=1.5)
        for line in lines:
            raw = "".join(t.text for t in sorted(line, key=lambda z: z.cx))
            if has_chinese(raw):
                avg_score = float(np.mean([t.score for t in line]))
                base_score = avg_score * 10.0 + 14.0
                candidates.append((raw, base_score))
                
    is_sold_out = parse_sold_out(slot)
    if is_sold_out and not candidates:
        for t in slot.tokens:
            if has_chinese(t.text) and "售" not in t.text:
                base_score = t.score * 10.0
                candidates.append((t.text, base_score))
                
    best: Tuple[Optional[str], float] = (None, 0.0)
    for raw_text, base in candidates:
        if "售" in raw_text and len(clean_name(raw_text)) <= 3:
            continue
        matched_name, score = match_item_name(raw_text, item_names, threshold=62.0)
        if matched_name is not None and (score + base) > best[1]:
            best = (matched_name, score + base)
            
    final_confidence = None
    if best[0] is not None:
        final_confidence = round(best[1] / 120.0, 4)
        
    is_occluded = bool(is_sold_out and best[0] is None)
    return best[0], final_confidence, is_occluded


def parse_sold_out(slot: Slot) -> bool:
    """Detects sold-out status by checking for specific keywords or high fuzzy ratio in the overlay ROI."""
    joined_text = normalize_text("".join(t.text for t in slot.tokens))
    sold_keywords = ["售罄", "售馨", "已售罄"]
    for keyword in sold_keywords:
        if keyword in joined_text:
            return True
            
    overlay_tokens = tokens_in_region(slot, ROI_SOLDOUT)
    all_tokens = overlay_tokens + slot.tokens
    for t in all_tokens:
        cleaned = clean_name(t.text)
        if fuzz.ratio(cleaned, "售罄") >= 60:
            return True
        if fuzz.ratio(cleaned, "售馨") >= 60:
            return True
            
    return False


def parse_discount(slot: Slot) -> Optional[int]:
    """Extracts discount percentage (1-99) from the top-right discount ROI."""
    region_tokens = tokens_in_region(slot, ROI_DISCOUNT)
    scanned_texts: List[str] = []
    for t in region_tokens:
        scanned_texts.append(normalize_num_text(t.text))
    joined_text = normalize_num_text("".join(t.text for t in region_tokens))
    scanned_texts.append(joined_text)
    
    values: List[int] = []
    for s in scanned_texts:
        for m in re.finditer(r"-?\s*(\d{1,2})\s*(?:%|元|折)?", s):
            v = int(m.group(1))
            if 1 <= v <= 99:
                values.append(v)
                
    if values:
        return max(values)
    return None


def parse_quantity(slot: Slot) -> Optional[int]:
    """Extracts quantity (e.g., x10, x2000) from the center badge ROI."""
    region_tokens = tokens_in_region(slot, ROI_QUANTITY)
    sorted_toks = sorted(region_tokens + slot.tokens, key=lambda t: (t.cy, t.cx))
    
    for t in sorted_toks:
        s = normalize_num_text(t.text).replace("*", "x")
        m = re.search(r"[xX]\s*(\d{1,7})", s)
        if m:
            return int(m.group(1))
            
    for a, b in zip(sorted_toks, sorted_toks[1:]):
        norm_a = normalize_num_text(a.text).lower()
        norm_b = normalize_num_text(b.text)
        if norm_a == "x" and re.fullmatch(r"\d{1,7}", norm_b):
            vertical_diff = abs(a.cy - b.cy)
            max_height = max(a.h, b.h)
            if vertical_diff < max_height * 2.0:
                return int(norm_b)
                
    return None


def _numeric_value_candidates_from_tokens(tokens: List[Token]) -> List[Tuple[int, Token, str]]:
    """
    Returns numeric readings from individual tokens.
    Also joins adjacent same-line digit fragments to reconstruct split numbers (e.g., '1' + '20' -> '120').
    """
    candidates: List[Tuple[int, Token, str]] = []
    sorted_toks = sorted(tokens, key=lambda t: (t.cy, t.cx))
    
    for t in sorted_toks:
        s = normalize_num_text(t.text)
        for m in re.finditer(r"\d{1,5}", s):
            v = int(m.group(0))
            candidates.append((v, t, "single"))
            
    lines = group_tokens_lines(sorted_toks, tol_factor=1.25)
    for line in lines:
        parts: List[Token] = []
        for t in sorted(line, key=lambda z: z.cx):
            s = normalize_num_text(t.text)
            if re.fullmatch(r"\d{1,3}", s):
                parts.append(t)
                
        if len(parts) < 2:
            continue
            
        current_run = [parts[0]]
        for prev, cur in zip(parts, parts[1:]):
            gap = cur.cx - prev.cx - (prev.w + cur.w) / 2.0
            cy_diff = abs(cur.cy - prev.cy)
            max_h = max(cur.h, prev.h)
            if cy_diff <= max_h * 0.75 and gap <= max_h * 1.15:
                current_run.append(cur)
            else:
                if len(current_run) >= 2:
                    raw_joined = "".join(normalize_num_text(x.text) for x in current_run)
                    if re.fullmatch(r"\d{2,5}", raw_joined):
                        fake_token = Token(
                            text=raw_joined,
                            box=union_token_box(current_run),
                            score=max(x.score for x in current_run),
                            source="joined_numeric"
                        )
                        candidates.append((int(raw_joined), fake_token, "joined"))
                current_run = [cur]
                
        if len(current_run) >= 2:
            raw_joined = "".join(normalize_num_text(x.text) for x in current_run)
            if re.fullmatch(r"\d{2,5}", raw_joined):
                fake_token = Token(
                    text=raw_joined,
                    box=union_token_box(current_run),
                    score=max(x.score for x in current_run),
                    source="joined_numeric"
                )
                candidates.append((int(raw_joined), fake_token, "joined"))
                
    return candidates


def _dedupe_numeric_values(cands: List[Tuple[int, Token, str]]) -> List[Tuple[int, Token, str]]:
    """
    Removes numeric fragments when a larger overlapping numeric block exists.
    Keeps unique values by approximate position, preferring higher confidence.
    """
    keep_mask = [True] * len(cands)
    for i, (vi, ti, si) in enumerate(cands):
        if not keep_mask[i]:
            continue
            
        di = str(vi)
        ri = token_rect(ti)
        for j, (vj, tj, sj) in enumerate(cands):
            if i == j:
                continue
            dj = str(vj)
            rj = token_rect(tj)
            inter = rect_inter_area(ri, rj)
            if inter <= 0.0:
                continue
                
            cover_i = inter / max(1.0, rect_area(ri))
            if len(dj) > len(di) and di in dj and cover_i > 0.55:
                keep_mask[i] = False
                break
                
    filtered = [c for c, k in zip(cands, keep_mask) if k]
    final: List[Tuple[int, Token, str]] = []
    sorted_by_score = sorted(filtered, key=lambda x: x[1].score, reverse=True)
    
    for cand in sorted_by_score:
        v, t, src = cand
        is_duplicate = False
        for ov, ot, _ in final:
            dist = math.hypot(t.cx - ot.cx, t.cy - ot.cy)
            threshold = max(10.0, t.h + ot.h)
            if v == ov and dist < threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            final.append(cand)
            
    return final


def parse_prices(slot: Slot, item_names: List[str]) -> Tuple[Optional[int], Optional[int], bool]:
    """
    Extracts current and original prices from the strict bottom-right price ROI.
    Returns (price, original_price, price_panel_present).
    """
    price_tokens: List[Token] = []
    region_tokens = tokens_in_region(slot, ROI_PRICE)
    
    for t in region_tokens:
        nx, ny = slot.norm_xy(t)
        s = normalize_num_text(t.text)
        if not s or "%" in s or "/" in s:
            continue
        if re.search(r"[xX]\s*\d+", s):
            continue
        if any(ch in s for ch in "售罄馨"):
            continue
        if match_item_name(t.text, item_names, 70)[0] is not None:
            continue
        if nx < 0.58 and not re.search(r"[币信信用元₵$]", t.text):
            continue
        price_tokens.append(t)
        
    candidates = _dedupe_numeric_values(_numeric_value_candidates_from_tokens(price_tokens))
    values = sorted(set(v for v, _, _ in candidates if 1 <= v <= 99999))
    
    if not values:
        return None, None, False
    if len(values) == 1:
        return values[0], None, True
    return values[0], values[-1], True


def _is_refresh_anchor_text(s: str) -> bool:
    """Checks if text matches the Chinese anchor for remaining refresh count."""
    cleaned = clean_name(s)
    if "剩余次数" in cleaned:
        return True
    if ("剩余" in cleaned and "次" in cleaned) or ("刷新" in cleaned and "次" in cleaned):
        return True
    if fuzz.partial_ratio(cleaned, "剩余次数") >= 72:
        return True
    if fuzz.partial_ratio(cleaned, "剩余刷新次数") >= 72:
        return True
    return False


def parse_refresh(tokens: List[Token]) -> Optional[Dict[str, Any]]:
    """Parses remaining/total refresh count by locating the anchor text and extracting X/Y digits."""
    lines = group_tokens_lines(tokens, tol_factor=2.2)
    for line in lines:
        raw_text = "".join(t.text for t in sorted(line, key=lambda z: z.cx))
        if not _is_refresh_anchor_text(raw_text):
            continue
            
        normalized = normalize_num_text(raw_text)
        m = re.search(r"(\d{1,3})\s*/\s*(\d{1,3})", normalized)
        if not m:
            ordered = sorted(line, key=lambda z: z.cx)
            anchor_index = 0
            for i, t in enumerate(ordered):
                if _is_refresh_anchor_text(t.text):
                    anchor_index = i
                    break
            tail_text = normalize_num_text("".join(t.text for t in ordered[anchor_index:]))
            m = re.search(r"(\d{1,3})\s*/\s*(\d{1,3})", tail_text)
            if not m:
                continue
                
        return {
            "remaining": int(m.group(1)),
            "total": int(m.group(2)),
            "text": m.group(0),
            "anchor_text": raw_text,
            "confidence": 0.96
        }
        
    return None


def default_uid_footer_roi(image_shape: Tuple[int, int]) -> Tuple[float, float, float, float]:
    """Returns the narrow bottom-left UID search region in rectified-image coordinates."""
    h, w = image_shape[:2]
    return (0.0, h * 0.875, w * 0.34, h * 0.995)


def rect_contains_point(rect: Tuple[float, float, float, float], x: float, y: float, margin: float = 0.0) -> bool:
    """Checks if a point (x, y) falls within a rectangle with optional padding."""
    l, t, r, b = rect
    return l - margin <= x <= r + margin and t - margin <= y <= b + margin


def _uid_anchor_match(s: str) -> Optional[re.Match]:
    """Searches for UID-like anchor text (UID, U1D, U|D, etc.) ignoring case."""
    return re.search(r"U\s*[I1l|]\s*D", normalize_num_text(s), flags=re.IGNORECASE)


def _digit_runs(s: str) -> List[str]:
    """Extracts all contiguous digit sequences from normalized text."""
    return re.findall(r"\d+", normalize_num_text(s))


def _pick_uid_digits_from_runs(runs: List[str], prefer: str = "first") -> Optional[str]:
    """Selects one UID-looking digit block without assuming a fixed length."""
    cleaned_runs = [r for r in runs if r]
    if not cleaned_runs:
        return None
        
    valid_runs = [r for r in cleaned_runs if UID_MIN_LEN <= len(r) <= UID_MAX_LEN]
    if valid_runs:
        if prefer != "last":
            return valid_runs[0]
        else:
            return valid_runs[-1]
            
    long_runs = [r for r in cleaned_runs if len(r) > UID_MAX_LEN]
    if long_runs:
        r = long_runs[0] if prefer != "last" else long_runs[-1]
        if prefer == "last":
            return r[-UID_MAX_LEN:]
        else:
            return r[:UID_MAX_LEN]
            
    return None


def _uid_candidate_digits(s: str, prefer: str = "last") -> Optional[str]:
    """Extracts UID digits from a string, ignoring ratio/percent patterns."""
    s = normalize_num_text(s)
    if "/" in s or "%" in s:
        return None
    return _pick_uid_digits_from_runs(_digit_runs(s), prefer=prefer)


def _uid_token_rank(t: Token, uid: str, image_shape: Optional[Tuple[int, int]] = None, uid_roi: Optional[Tuple[float, float, float, float]] = None) -> float:
    """Scores a token's likelihood of being the true UID based on length, position, and ROI containment."""
    score = 0.0
    if UID_MIN_LEN <= len(uid) <= UID_MAX_LEN:
        score += 0.8
        
    if image_shape is not None:
        h, w = image_shape[:2]
        if t.cy > 0.86 * h:
            score += 0.25
        if t.cx < 0.34 * w:
            score += 0.25
        if t.h < max(18, h * 0.040):
            score += 0.12
        if uid_roi is not None and rect_contains_point(uid_roi, t.cx, t.cy, margin=max(8.0, t.h * 2.0)):
            score += 0.45
            
    return score


def _token_span_rect_horizontal(t: Token, start_frac: float, end_frac: float) -> Tuple[float, float, float, float]:
    """Approximates a sub-span bounding box inside a horizontal OCR token."""
    x1, y1, x2, y2 = token_rect(t)
    start_frac = max(0.0, min(1.0, float(start_frac)))
    end_frac = max(start_frac, min(1.0, float(end_frac)))
    sub_x1 = x1 + (x2 - x1) * start_frac
    sub_x2 = x1 + (x2 - x1) * end_frac
    return (sub_x1, y1, sub_x2, y2)


def parse_uid(tokens: List[Token], image_shape: Optional[Tuple[int, int]] = None, uid_roi: Optional[Tuple[float, float, float, float]] = None) -> Optional[Dict[str, Any]]:
    """
    Parses UID using a 3-stage fallback strategy:
    1. Anchor + digits in the same token.
    2. Anchor token followed by adjacent digit tokens on the same visual line.
    3. Strict bottom-left numeric fallback within the tiny UID ROI.
    """
    if image_shape is not None and uid_roi is None:
        uid_roi = default_uid_footer_roi(image_shape)
        
    # Stage 1: Same token anchor + digits
    single_token_candidates: List[Tuple[float, Token, str, str, Tuple[float, float, float, float]]] = []
    for t in sorted(tokens, key=lambda z: (z.cy, z.cx)):
        s = normalize_num_text(t.text)
        anchor_match = _uid_anchor_match(s)
        if not anchor_match:
            continue
            
        tail = s[anchor_match.end():]
        uid = _uid_candidate_digits(tail, prefer="first")
        if uid:
            token_len = max(1, len(s))
            digit_start = s.find(uid, anchor_match.end())
            if digit_start >= 0:
                bbox = _token_span_rect_horizontal(t, digit_start / token_len, (digit_start + len(uid)) / token_len)
            else:
                bbox = token_rect(t)
                
            rank = _uid_token_rank(t, uid, image_shape, uid_roi)
            single_token_candidates.append((rank, t, uid, tail, bbox))
            
    if single_token_candidates:
        single_token_candidates.sort(key=lambda x: (-x[0], x[1].cx))
        rank, t, uid, tail, bbox = single_token_candidates[0]
        return {
            "uid": uid,
            "text": f"UID:{uid}",
            "raw_text": t.text,
            "tail": tail,
            "bbox": rect_to_list(bbox),
            "roi_bbox": rect_to_list(uid_roi) if uid_roi is not None else None,
            "confidence": round(float(0.90 + min(0.08, rank / 20)), 3),
            "source": "uid_one_token_anchor"
        }
        
    # Stage 2: Multi-token anchor + digits on same line
    lines = group_tokens_lines(tokens, tol_factor=2.0)
    for line in lines:
        ordered = sorted(line, key=lambda z: z.cx)
        for i, t in enumerate(ordered):
            s_t = normalize_num_text(t.text)
            anchor_match = _uid_anchor_match(s_t)
            if not anchor_match:
                continue
                
            used_tokens = [t]
            pieces: List[str] = []
            tail_digits = _uid_candidate_digits(s_t[anchor_match.end():], prefer="first")
            if tail_digits:
                pieces.append(tail_digits)
                
            last_token = t
            for u in ordered[i + 1:]:
                cy_diff = abs(u.cy - t.cy)
                max_h = max(t.h, u.h)
                if cy_diff > max_h * 1.45:
                    break
                    
                gap = u.cx - last_token.cx - (u.w + last_token.w) / 2.0
                if gap > max_h * 4.0 and pieces:
                    break
                    
                su = normalize_num_text(u.text)
                d = _uid_candidate_digits(su, prefer="first")
                if d:
                    pieces.append(d)
                    used_tokens.append(u)
                    last_token = u
                    
            uid_joined = "".join(pieces)
            uid = _pick_uid_digits_from_runs([uid_joined], prefer="first")
            if uid:
                return {
                    "uid": uid,
                    "text": f"UID:{uid}",
                    "raw_text": "|".join(x.text for x in used_tokens),
                    "bbox": rect_to_list(token_rect(Token("", union_token_box(used_tokens), 1.0, "uid_block"))),
                    "roi_bbox": rect_to_list(uid_roi) if uid_roi is not None else None,
                    "confidence": 0.92,
                    "source": "uid_anchor_plus_digits"
                }
                
            if len(uid_joined) > UID_MAX_LEN:
                break
            elif pieces:
                break
                
    # Stage 3: Strict bottom-left numeric fallback
    if uid_roi is not None:
        fallback_candidates: List[Tuple[float, float, str, Token]] = []
        for t in tokens:
            if not rect_contains_point(uid_roi, t.cx, t.cy, margin=max(6.0, t.h * 1.5)):
                continue
                
            uid = _uid_candidate_digits(t.text, prefer="last")
            if not uid:
                continue
                
            raw_norm = normalize_num_text(t.text).lower()
            penalty = 0.25 if ("ms" in raw_norm or "/" in raw_norm or "%" in raw_norm) else 0.0
            rank = 0.65 + _uid_token_rank(t, uid, image_shape, uid_roi) * 0.12 - penalty
            fallback_candidates.append((rank, t.cx, uid, t))
            
        if fallback_candidates:
            fallback_candidates.sort(key=lambda x: (-x[0], x[1]))
            rank, _, uid, t = fallback_candidates[0]
            return {
                "uid": uid,
                "text": f"UID:{uid}",
                "raw_text": t.text,
                "bbox": rect_to_list(token_rect(t)),
                "roi_bbox": rect_to_list(uid_roi),
                "confidence": round(float(rank), 3),
                "source": "uid_tiny_roi_numeric_fallback"
            }
            
    return None


def _token_clean_for_dedup(t: Token) -> str:
    """Cleans token text for deduplication comparison, keeping alphanumerics, CJK, and common symbols."""
    s = normalize_num_text(t.text)
    s = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff%/:：\-]", "", s)
    return s


def _prefer_token(a: Token, b: Token) -> Token:
    """Chooses the better representative of two overlapping OCR readings."""
    ca = _token_clean_for_dedup(a)
    cb = _token_clean_for_dedup(b)
    if len(ca) != len(cb):
        longer = a if len(ca) > len(cb) else b
        shorter = b if longer is a else a
        if shorter.score <= longer.score + 0.35:
            return longer
            
    score_diff = abs(a.score - b.score)
    if score_diff > 0.03:
        return a if a.score > b.score else b
        
    area_a = rect_area(token_rect(a))
    area_b = rect_area(token_rect(b))
    return a if area_a >= area_b else b


def _tokens_equivalent_or_fragment(a: Token, b: Token) -> bool:
    """Checks if two tokens represent the same text or if one is a fragment of the other."""
    ca = _token_clean_for_dedup(a)
    cb = _token_clean_for_dedup(b)
    if not ca or not cb:
        return False
        
    ra = token_rect(a)
    rb = token_rect(b)
    inter = rect_inter_area(ra, rb)
    if inter <= 0.0:
        return False
        
    aa = rect_area(ra)
    ab = rect_area(rb)
    small_cover = inter / max(1.0, min(aa, ab))
    big_cover = inter / max(1.0, max(aa, ab))
    center_close = math.hypot(a.cx - b.cx, a.cy - b.cy) <= max(10.0, (a.h + b.h) * 0.85)
    
    if ca == cb and (small_cover > 0.45 or center_close):
        return True
        
    da = re.sub(r"\D", "", ca)
    db = re.sub(r"\D", "", cb)
    if da and db:
        if (da in db or db in da) and small_cover > 0.55:
            return True
        if fuzz.ratio(da, db) >= 88 and (small_cover > 0.45 or center_close):
            return True
            
    if (ca in cb or cb in ca) and min(len(ca), len(cb)) >= 2 and small_cover > 0.58:
        return True
    if fuzz.ratio(ca, cb) >= 88 and (small_cover > 0.50 or (center_close and big_cover > 0.25)):
        return True
        
    return False


def deduplicate_tokens(tokens: List[Token]) -> List[Token]:
    """
    Collapses full-image/per-card OCR duplicates and contained fragments.
    Clusters overlapping tokens, prefers longer/higher-score/cleaner representatives.
    """
    clusters: List[List[Token]] = []
    ordered = sorted(
        tokens,
        key=lambda z: (len(_token_clean_for_dedup(z)), rect_area(token_rect(z)), z.score),
        reverse=True
    )
    
    for t in ordered:
        if not normalize_text(t.text):
            continue
            
        placed = False
        for cl in clusters:
            is_equivalent = False
            for u in cl:
                if _tokens_equivalent_or_fragment(t, u):
                    is_equivalent = True
                    break
            if is_equivalent:
                cl.append(t)
                placed = True
                break
                
        if not placed:
            clusters.append([t])
            
    result: List[Token] = []
    for cl in clusters:
        best = cl[0]
        for u in cl[1:]:
            best = _prefer_token(best, u)
        result.append(best)
        
    result.sort(key=lambda z: (z.cy, z.cx, -z.score))
    return result

def collect_smart_fallback_rects(
    slots: list[Slot],
    item_names: list[str],
    image_shape: tuple[int, int],
) -> list[tuple[float, float, float, float, str]]:
    """
    Determine which local regions need a second OCR pass after the global full-image OCR.
    Returns a list of (x1, y1, x2, y2, source) rectangles in the rectified image coordinates.
    """
    rects: list[tuple[float, float, float, float, str]] = []
    H, W = image_shape[:2]

    for s in slots:
        name, _, name_occluded = parse_name(s, item_names)
        price, _, price_present = parse_prices(s, item_names)
        sold = parse_sold_out(s)

        if s.namebar_rect is not None and name is None and not name_occluded:
            rects.append((*s.namebar_rect, "paddle_namebar"))

        if price is None and not sold:
            l, t, r, b = s.rect
            sw, sh = r - l, b - t
            roi = ROI_PRICE  # (0.54, 0.70, 1.02, 0.97)
            x1 = l + roi[0] * sw
            y1 = t + roi[1] * sh
            x2 = l + roi[2] * sw
            y2 = t + roi[3] * sh
            rects.append((x1, y1, x2, y2, "paddle_price_roi"))

        if sold and len(s.tokens) < 3:
            rects.append((*(s.full_rect or s.rect), "paddle_soldout_card"))

    # Global UID and refresh fallback rectangles
    uid_roi = default_uid_footer_roi(image_shape)
    rects.append((*uid_roi, "paddle_uid_tiny_roi"))
    rects.append((W * 0.42, H * 0.72, W, H, "paddle_footer_refresh"))

    # Merge overlapping rectangles (optional but reduces redundant OCR)
    merged: list[tuple[float, float, float, float, str]] = []
    for rect in rects:
        x1, y1, x2, y2, src = rect
        if x2 <= x1 or y2 <= y1:
            continue
        is_dup = False
        for m in merged:
            mx1, my1, mx2, my2, _ = m
            inter = rect_inter_area((x1, y1, x2, y2), (mx1, my1, mx2, my2))
            area = min(rect_area((x1, y1, x2, y2)), rect_area((mx1, my1, mx2, my2)))
            if area > 0 and inter / area > 0.82:
                is_dup = True
                break
        if not is_dup:
            merged.append(rect)
    return merged