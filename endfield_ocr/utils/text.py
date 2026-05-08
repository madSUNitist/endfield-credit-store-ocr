# endfield_ocr/utils/text.py
"""
Text normalization, cleaning, Chinese detection, and line grouping helpers.
Designed for robust handling of OCR output noise, symbol variants, and spatial layout.
"""
import re
import unicodedata
import numpy as np
from typing import List, Any


def normalize_text(s: str) -> str:
    """
    Applies NFKC normalization and replaces common OCR symbol variants.
    Strips all whitespace for consistent downstream parsing.
    """
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("％", "%")
    s = s.replace("－", "-").replace("—", "-").replace("–", "-").replace("−", "-")
    s = s.replace("×", "x").replace("＊", "*").replace("￥", "")
    return re.sub(r"\s+", "", s)


def normalize_num_text(s: str) -> str:
    """
    Normalizes text and maps common numeric OCR errors to digits.
    Example: O->0, I->1, l->1, |->1, S->5, B->8.
    """
    s = normalize_text(s)
    table = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "|": "1", "S": "5", "B": "8"})
    return s.translate(table)


def clean_name(s: str) -> str:
    """Removes punctuation/symbols, keeping only CJK, Latin letters, and digits."""
    s = normalize_text(s)
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", s)


def has_chinese(s: str) -> bool:
    """Checks if a string contains at least one CJK character."""
    return any("\u4e00" <= ch <= "\u9fff" for ch in s)


def group_tokens_lines(tokens: List[Any], tol_factor: float = 1.8) -> List[List[Any]]:
    """
    Groups OCR tokens into horizontal lines based on vertical center proximity.
    Tokens within `tol_factor * median_height` of each other's Y-center are merged.
    Each resulting line is sorted left-to-right by X-center.
    """
    if not tokens:
        return []
    heights = [max(1.0, float(t.h)) for t in tokens]
    med_h = float(np.median(heights))
    tol = max(10.0, med_h * tol_factor)
    rows: List[List[Any]] = []
    for t in sorted(tokens, key=lambda z: z.cy):
        placed = False
        for row in rows:
            row_centers = [float(x.cy) for x in row]
            row_mean = float(np.mean(row_centers))
            if abs(t.cy - row_mean) <= tol:
                row.append(t)
                placed = True
                break
        if not placed:
            rows.append([t])
    for row in rows:
        row.sort(key=lambda z: z.cx)
    return rows