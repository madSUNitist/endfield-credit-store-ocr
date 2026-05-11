# endfield_ocr/processor.py
"""
Main pipeline orchestrator for the Endfield Shop OCR system.
Replaces the monolithic recognize_one_compact() and main() with a 
streaming, memory-safe, batch-ready class designed for ~10GB datasets.
All logic is explicitly expanded. No compressed one-liners.
"""
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Iterable, Iterator, Callable, Any, List

from .config import PipelineConfig
from .models import Token, ShopResult, ItemResult
from .backend import (
    OCRBackend, 
    RemotePaddleOCRBackend, 
    PaddleOCRBackend
)
from .pipeline.detector import detect_card_quads, rectify_by_card_plane
from .pipeline.slot_builder import build_slots_after_rectification
from .pipeline.parser import (
    assign_tokens_to_slots, 
    deduplicate_tokens, 
    default_uid_footer_roi, 
    parse_uid, 
    parse_refresh, 
    parse_name, 
    parse_prices, 
    parse_discount, 
    parse_quantity, 
    parse_sold_out
)
from .pipeline.matcher import load_ref_items, match_card_bgr_to_refs
from .utils.image import load_image_safe, maybe_resize, crop_rect_img
from .utils.geometry import box_center_size, clip_rect
from .utils.visualize import draw_boxes, save_debug_image

logger = logging.getLogger(__name__)


class ShopOCRProcessor:
    """
    Thread-unsafe orchestrator. Instantiate once per process or thread.
    Lazy-initializes heavy dependencies (PaddleOCR, reference icons) to minimize
    startup latency and memory footprint during batch processing.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        refs_dir: Optional[str | Path] = None,
        item_names: Optional[List[str]] = None,
    ):
        """
        Initialize the processor with centralized configuration.
        
        Args:
            config: Global pipeline thresholds and OCR settings.
            refs_dir: Directory containing clean item reference images for icon matching.
            item_names: Optional whitelist for OCR name validation. Overridden by `refs_dir` stems.
        """
        self.config = config or PipelineConfig()
        self.refs_dir = Path(refs_dir) if refs_dir else None
        self.item_names = item_names or []
        
        # Internal state
        self._refs_cache: List[Any] = []
        self._ocr_backend: Optional[OCRBackend] = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization hook. Runs once per instance lifecycle."""
        if self._initialized:
            return None
        
        if self.refs_dir is not None:
            self._refs_cache = load_ref_items(self.refs_dir, self.config.recursive_refs)
            
            if self._refs_cache:
                self.item_names.extend([ref.name for ref in self._refs_cache])
                
        self._initialized = True

    def _get_ocr(self) -> OCRBackend:
        """Returns or creates the PaddleOCR backend singleton."""
        if self._ocr_backend is None:
            logger.info("Initializing PaddleOCR backend (lazy load)...")
            if self.config.ocr.use_remote_backend:
                self._ocr_backend = RemotePaddleOCRBackend(self.config.ocr)
            else:
                self._ocr_backend = PaddleOCRBackend(self.config.ocr)
            
        return self._ocr_backend

    def _load_image(self, image: np.ndarray | str | Path) -> np.ndarray:
        """Safely load image and apply max-side downscaling to prevent VRAM/OOM spikes."""
        if isinstance(image, np.ndarray):
            img = image
        else:
            img = load_image_safe(image)
        
        if self.config.max_input_side is not None:
            img = maybe_resize(img, max_side=self.config.max_input_side)
            
        return img
    
    def _crop_ocr_and_offset(
        self,
        ocr_backend: OCRBackend, 
        rectified: np.ndarray,
        rect: tuple[float, float, float, float],
        source: str,
        pad: int = 4,
        upscale: float = 1.0,
    ) -> list[Token]:
        """
        Crop a region from rectified image, run OCR, and translate coordinates back.
        Returns list of Token objects in the original rectified image coordinates.
        """
        h, w = rectified.shape[:2]
        l, t, r, b = clip_rect(rect, w, h, pad=pad)
        logger.debug(f"Crop region: ({l},{t},{r},{b}) from rect {rect}")
        if r <= l or b <= t:
            logger.warning(f"Invalid crop region for {source}: {rect} -> ({l},{t},{r},{b})")
            return []
        crop = rectified[t:b, l:r]
        logger.debug(f"Crop shape: {crop.shape}")
        if crop.size == 0:
            return []
        
        # Optionally upscale for tiny text
        scale_back = 1.0
        if upscale > 1.01:
            crop = cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
            scale_back = upscale
        # Run OCR (backend handles scale_back? No, we adjust box coordinates manually)
        tokens = ocr_backend.recognize(crop, source=source)
        # Translate coordinates
        for tok in tokens:
            if scale_back != 1.0:
                tok.box = tok.box / scale_back
            tok.box[:, 0] += l
            tok.box[:, 1] += t
            
            tok.cx, tok.cy, tok.w, tok.h = box_center_size(tok.box)
            
        return tokens

    def process_image(self, image: np.ndarray | str | Path) -> ShopResult:
        """
        Process a single image through the full pipeline:
        Load → Resize → Detect → Rectify → Build Slots → OCR → Parse → Match → Return.
        
        Args:
            image: Numpy array (BGR) or file path.
            
        Returns:
            ShopResult containing parsed items, UID, refresh count, and metadata.
        """
        self._ensure_initialized()
        
        if isinstance(image, (str, Path)):
            image_path = str(image)
        else:
            image_path = "in_memory"

        # 1. Load & safely downscale
        img = self._load_image(image)
        h_orig, w_orig = img.shape[:2]

        # 2. Perspective detection & rectification
        quads = detect_card_quads(img, self.config.detection, self.config.debug_save_dir)
        if self.config.debug_save_dir is not None:
            # Draw quads on original image
            quad_boxes = [q["pts"] for q in quads]
            vis_quads = draw_boxes(img, quad_boxes, color=(0, 255, 0), thickness=5)
            save_debug_image(self.config.debug_save_dir, "01_original_with_quads", vis_quads) # type: ignore
        
        rectified, h_mat, rect_meta = rectify_by_card_plane(
            img, 
            quads, 
            self.config.detection.max_output_side
        )
        if self.config.debug_save_dir is not None:
            save_debug_image(self.config.debug_save_dir, "02_rectified_full", rectified) # type: ignore
        
        # 3. Grid slot generation & local namebar search
        slots, slot_meta = build_slots_after_rectification(rectified, quads, h_mat)
        if self.config.debug_save_dir is not None:
            body_boxes = [s.rect for s in slots]
            namebar_boxes = [s.namebar_rect for s in slots if s.namebar_rect]
            vis_slots = rectified.copy()
            # Draw body in green
            body_vis = draw_boxes(vis_slots, body_boxes, color=(0,255,0), thickness=5)
            # Draw namebar in blue (if present)
            if namebar_boxes:
                body_vis = draw_boxes(body_vis, namebar_boxes, color=(255,0,0), thickness=5)
            # Optionally add row/col labels
            labels = [f"{s.row},{s.col}" for s in slots]
            body_vis = draw_boxes(body_vis, body_boxes, labels=labels, color=(0,255,0), thickness=5)
            save_debug_image(self.config.debug_save_dir, "03_rectified_slots", body_vis) # type: ignore

        # 4. OCR execution (lazy backend ensures model loads only once)
        tokens: List[Token] = []
        ocr_meta: dict[str, Any] = {
            "skipped": bool(self.config.skip_ocr), 
            "mode": self.config.ocr.mode,
            "full_passes": 0,
            "crop_passes": 0,
        }

        if not self.config.skip_ocr:
            ocr = self._get_ocr()
            mode = self.config.ocr.mode

            # Always do a full-image OCR first (fast mode does only this)
            full_tokens = ocr.recognize(rectified, source="paddle_full")
            tokens = full_tokens.copy()
            ocr_meta["full_passes"] = 1
            
            if self.config.debug_save_dir is not None:
                vis_crop = draw_boxes(
                    rectified,
                    [t.box for t in tokens],
                    labels=[f"{t.text} ({t.score:.2f})" for t in tokens],
                    color=(0, 0, 255),
                    thickness=5,
                    font_scale=1.0
                )
                cv2.imwrite(str(self.config.debug_save_dir / "04_rectified_ocr.png"), vis_crop)

            if mode == "fast":
                # Fast: only full image, plus optional UID/refresh fallback
                H, W = rectified.shape[:2]
                uid_roi = default_uid_footer_roi(rectified.shape)
                if parse_uid(tokens, self.config.uid, rectified.shape, uid_roi=uid_roi) is None:
                    uid_tokens = self._crop_ocr_and_offset(ocr, rectified, uid_roi, "paddle_uid_tiny_roi", pad=5, upscale=3.0)
                    tokens.extend(uid_tokens)
                    ocr_meta["crop_passes"] += 1
                if parse_refresh(tokens) is None:
                    refresh_roi = (W * 0.38, H * 0.70, W, H)
                    ref_tokens = self._crop_ocr_and_offset(ocr, rectified, refresh_roi, "paddle_footer_refresh", pad=5, upscale=2.0)
                    tokens.extend(ref_tokens)
                    ocr_meta["crop_passes"] += 1

            elif mode == "full":
                crop_images = []
                crop_metas = []  # (slot, l, t, crop_rect)
                for s in slots:
                    crop_rect = s.full_rect if s.full_rect else s.rect
                    h_img, w_img = rectified.shape[:2]
                    l, t, r, b = clip_rect(crop_rect, w_img, h_img, pad=4)
                    if r <= l or b <= t:
                        continue
                    crop = rectified[t:b, l:r]
                    if crop.size == 0:
                        continue
                    crop_images.append(crop)
                    crop_metas.append((s, l, t, crop_rect))

                if crop_images:
                    sources = [f"paddle_slot_{s.id}" for s, _, _, _ in crop_metas]
                    for idx, ((s, l, t, _), crop_tokens) in enumerate(zip(crop_metas, ocr.recognize_iter(crop_images, sources=sources))):
                        # Main flow: translate tokens in-place to full image coordinates
                        for tok in crop_tokens:
                            tok.box[:, 0] += l
                            tok.box[:, 1] += t
                            tok.cx, tok.cy, tok.w, tok.h = box_center_size(tok.box)
                            tok.slot_id = s.id
                        tokens.extend(crop_tokens)
                        ocr_meta["crop_passes"] += 1

                        # Debug drawing (skip if debug directory not set)
                        if self.config.debug_save_dir is None:
                            continue
                        debug_dir = self.config.debug_save_dir / f"slot_{s.id:03d}"
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(debug_dir / "crop.png"), crop_images[idx])

                        # Create temporary tokens with translated-back coordinates for drawing
                        temp_tokens = []
                        for tok in crop_tokens:
                            box_back = tok.box - np.array([l, t], dtype=np.float32)
                            temp_tok = Token(
                                text=tok.text,
                                box=box_back,
                                score=tok.score,
                                source=tok.source,
                            )
                            temp_tokens.append(temp_tok)

                        if not temp_tokens:
                            continue
                        vis_crop = draw_boxes(
                            crop_images[idx],
                            [t.box for t in temp_tokens],
                            labels=[f"{t.text} ({t.score:.2f})" for t in temp_tokens],
                            color=(0, 0, 255),
                            thickness=5,
                            font_scale=1.0
                        )
                        cv2.imwrite(str(debug_dir / "crop_ocr.png"), vis_crop)

                tokens = deduplicate_tokens(tokens)

            elif mode == "smart":
                # Smart: full image + targeted fallback for missing fields
                # First assign tokens to slots to know what's missing
                assign_tokens_to_slots(tokens, slots)
                # Collect fallback rectangles
                from .pipeline.parser import collect_smart_fallback_rects
                fallback_rects = collect_smart_fallback_rects(slots, self.item_names, rectified.shape, self.config.roi)
                # For each fallback rect, do local OCR
                for x1, y1, x2, y2, src in fallback_rects:
                    if src == "paddle_uid_tiny_roi" and parse_uid(tokens, self.config.uid, rectified.shape) is not None:
                        continue
                    if src == "paddle_footer_refresh" and parse_refresh(tokens) is not None:
                        continue
                    upscale = 3.0 if "uid" in src else (2.0 if "footer" in src else 1.0)
                    new_tokens = self._crop_ocr_and_offset(ocr, rectified, (x1, y1, x2, y2), src, pad=5, upscale=upscale)
                    tokens.extend(new_tokens)
                    ocr_meta["crop_passes"] += 1
                # Reassign tokens to slots after adding new ones
                assign_tokens_to_slots(tokens, slots)
                tokens = deduplicate_tokens(tokens)

            else:
                # Unknown mode, fallback to fast
                logger.warning(f"Unknown OCR mode '{mode}', falling back to 'fast'")
                tokens = full_tokens

            ocr_meta["tokens_found"] = len(tokens)

        # 5. Spatial assignment: attach tokens to their parent card slot
        assign_tokens_to_slots(tokens, slots)

        # 6. Parse UI fields & fallback to icon matching for missing namebars
        parsed_items: List[ItemResult] = []
        
        for s in slots:
            name, name_conf, name_occ = parse_name(s, self.item_names, self.config.roi)
            price, orig_price, price_present = parse_prices(s, self.item_names, self.config.roi)
            
            name_source = "ocr_namebar" if name is not None else None
            match_info = None
            
            if name is None and self._refs_cache:
                card_bgr = crop_rect_img(rectified, s.rect, pad=2)
                match_info = match_card_bgr_to_refs(card_bgr, self._refs_cache, name=f"slot_{s.id}")
                
                if match_info is not None:
                    name = match_info["name"]
                    name_conf = match_info["score"]
                    name_source = "icon_match_no_namebar"
                    name_occ = False

            parsed_items.append(ItemResult(
                id=s.id, 
                row=s.row, 
                col=s.col,
                name=name, 
                name_confidence=name_conf, 
                name_source=name_source,
                name_occluded=bool(name_occ),
                price=price, 
                original_price=orig_price, 
                price_panel_present=bool(price_present),
                discount_percent=parse_discount(s, self.config.roi),
                quantity=parse_quantity(s, self.config.roi),
                sold_out=parse_sold_out(s, self.config.roi),
            ))
            

            if self.config.debug_save_dir:
                body_crop = crop_rect_img(rectified, s.rect, pad=2)
                save_debug_image(self.config.debug_save_dir, f"slot_{s.id:03d}_body", body_crop) # type: ignore
                if s.namebar_rect:
                    nb_crop = crop_rect_img(rectified, s.namebar_rect, pad=2)
                    save_debug_image(self.config.debug_save_dir, f"slot_{s.id:03d}_namebar", nb_crop) # type: ignore

        # 7. Extract global footer metadata (UID, remaining refreshes)
        uid_obj = None
        if tokens:
            uid_obj = parse_uid(tokens, self.config.uid, rectified.shape)
            
        refresh_obj = None
        if tokens:
            refresh_obj = parse_refresh(tokens)

        uid_value = None
        if uid_obj is not None:
            uid_value = uid_obj.get("uid")
            
        refresh_remaining = None
        refresh_total = None
        refresh_remaining_time = None
        if refresh_obj is not None:
            refresh_remaining = refresh_obj.get("remaining")
            refresh_total = refresh_obj.get("total")
            refresh_remaining_time = refresh_obj.get("remaining_time")
            if refresh_remaining_time is not None:
                refresh_remaining_time = refresh_remaining_time.get("total_minutes")

        return ShopResult(
            image_path=image_path,
            items=parsed_items,
            uid=uid_value,
            refresh_remaining=refresh_remaining,
            refresh_remaining_time=refresh_remaining_time, 
            refresh_total=refresh_total,
            meta={
                "original_shape": [h_orig, w_orig],
                "cards_detected": len(quads),
                "slots_built": len(slots),
                "tokens_found": len(tokens),
                "rectification_used": rect_meta.get("used", False),
                "ocr": ocr_meta,
            },
        )

    def process_batch(
        self,
        image_paths: Iterable[str | Path],
        *,
        on_progress: Optional[Callable[[int, Optional[int], ShopResult], None]] = None,
        on_error: Optional[Callable[[Path, Exception], bool]] = None,
    ) -> Iterator[ShopResult]:
        """
        Stream-processing generator for large datasets.
        Yields results one-by-one to keep peak memory usage constant.
        
        Args:
            image_paths: Generator or list of file paths.
            on_progress: Callback(index, total_count_or_None, result).
            on_error: Callback(failed_path, exception). Return True to skip and continue.
            
        Yields:
            ShopResult for each successfully processed image.
        """
        total = None
        if hasattr(image_paths, "__len__"):
            total = len(image_paths) # type: ignore[arg-type]

        for idx, path in enumerate(image_paths):
            try:
                result = self.process_image(path)
                if on_progress is not None:
                    on_progress(idx, total, result)
                yield result
                
            except Exception as e:
                logger.error(f"Batch processing failed at index {idx} ({path}): {e}", exc_info=True)
                
                if on_error is not None:
                    path_obj = Path(path) if isinstance(path, (str, Path)) else Path("unknown")
                    if on_error(path_obj, e):
                        continue
                        
                raise

    def close(self) -> None:
        """Explicitly release heavy resources (PaddleOCR model, caches, numpy arrays)."""
        if self._ocr_backend is not None:
            self._ocr_backend.close()
            self._ocr_backend = None
            
        self._refs_cache.clear()
        self._initialized = False

    def __enter__(self) -> "ShopOCRProcessor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()