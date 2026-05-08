# endfield_ocr/utils/visualize.py
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Sequence, Union

# Try to import PIL for Chinese text support
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

def _get_chinese_font(size: int = 20):
    """Find a suitable Chinese font file for the current OS."""
    font_paths = [
        # Windows
        "C:/Windows/Fonts/msyh.ttc",      # Microsoft YaHei
        "C:/Windows/Fonts/simhei.ttf",    # SimHei
        "C:/Windows/Fonts/simsun.ttc",    # SimSun
        # Linux
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for p in font_paths:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size) # type: ignore
            except:
                continue
    # Fallback to default PIL font (may not support Chinese, but better than nothing)
    return ImageFont.load_default() # type: ignore

def draw_boxes(
    image: np.ndarray,
    boxes: Sequence[Union[np.ndarray, Tuple[float, float, float, float]]],
    labels: Optional[List[str]] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """
    Draw boxes and labels on image.
    If any label contains Chinese characters and PIL is available, use PIL for correct rendering.
    """
    # Check if we need PIL for Chinese text
    need_pil = False
    if labels and PIL_AVAILABLE:
        for label in labels:
            if label and any('\u4e00' <= ch <= '\u9fff' for ch in label):
                need_pil = True
                break

    if need_pil:
        # Convert to RGB for PIL
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # type: ignore
        draw = ImageDraw.Draw(pil_img) # type: ignore
        font = _get_chinese_font(size=int(14 * font_scale))

        for i, box in enumerate(boxes):
            # Draw the box (rectangle or polygon)
            label_coords = None
            if isinstance(box, np.ndarray) and box.shape == (4, 2):
                pts_pil = box.reshape(-1, 2).astype(np.int32).tolist()
                draw.polygon(pts_pil, outline=(color[2], color[1], color[0]), width=thickness)
                # Estimate label position: center of bounding box
                xs = [p[0] for p in pts_pil]
                ys = [p[1] for p in pts_pil]
                if xs and ys:
                    label_x = sum(xs) // len(xs)
                    label_y = sum(ys) // len(ys)
                    label_coords = (label_x, label_y)
            else:
                try:
                    x1, y1, x2, y2 = [int(round(v)) for v in box]
                    draw.rectangle([x1, y1, x2, y2], outline=(color[2], color[1], color[0]), width=thickness)
                    label_coords = (x1, y1 - 5)
                except:
                    continue

            if labels and i < len(labels) and label_coords is not None:
                text = labels[i]
                label_x, label_y = label_coords
                # Draw text background for better readability
                try:
                    bbox = draw.textbbox((label_x, label_y), text, font=font)
                except AttributeError:
                    # Older Pillow may not have textbbox; fallback to textsize
                    try:
                        text_width, text_height = draw.textsize(text, font=font) # type: ignore
                        bbox = (label_x, label_y, label_x + text_width, label_y + text_height)
                    except:
                        bbox = (label_x, label_y, label_x + 10*len(text), label_y + 20)
                draw.rectangle(bbox, fill=(0, 0, 0))
                draw.text((label_x, label_y), text, fill=(color[2], color[1], color[0]), font=font)

        # Convert back to BGR
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    else:
        # Original OpenCV drawing (fast, no Chinese support)
        vis = image.copy()
        for i, box in enumerate(boxes):
            label_coords = None
            if isinstance(box, np.ndarray) and box.shape == (4, 2):
                pts = box.reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(vis, [pts], True, color, thickness)
                if labels and i < len(labels):
                    cx = int(pts[:,0,0].mean())
                    cy = int(pts[:,0,1].mean())
                    label_coords = (cx, cy)
            else:
                try:
                    x1, y1, x2, y2 = [int(round(v)) for v in box]
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
                    label_coords = (x1, y1 - 5)
                except:
                    continue

            if labels and i < len(labels) and label_coords is not None:
                text = labels[i]
                lx, ly = label_coords
                cv2.putText(vis, text, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, color, thickness)

        return vis

def save_debug_image(debug_dir: Optional[Path], name: str, image: np.ndarray) -> None:
    """Saves image to debug directory if debug_dir is provided."""
    if debug_dir is None:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dir / f"{name}.png"
    cv2.imwrite(str(path), image)