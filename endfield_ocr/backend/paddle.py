# endfield_ocr/backend/paddle.py
"""
PaddleOCR v3 Backend Implementation.
Fully adapted to PaddleOCR 3.x API using predict() and predict_iter().
"""
import logging
import numpy as np
from typing import Optional, List, Any, Dict, Iterable, Iterator

from paddleocr import PaddleOCR  # type: ignore[import-untyped]

from .ocr_base import OCRBackend
from ..config import OCRConfig
from ..models import Token

logger = logging.getLogger(__name__)


class PaddleOCRBackend(OCRBackend):
    """PaddleOCR v3 wrapper using predict()/predict_iter() with PaddleX dict output."""

    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()
        self._client: Optional[PaddleOCR] = None
        logging.getLogger("ppocr").setLevel(logging.ERROR)
        logging.getLogger("paddleocr").setLevel(logging.ERROR)

    @property
    def client(self) -> PaddleOCR:
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self) -> PaddleOCR:
        kwargs: Dict[str, Any] = {
            "lang": self.config.language,
            "use_textline_orientation": self.config.use_textline_orientation,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
        }
        if self.config.text_det_limit_side_len is not None:
            kwargs["text_det_limit_side_len"] = self.config.text_det_limit_side_len
        if self.config.text_det_box_thresh is not None:
            kwargs["text_det_box_thresh"] = self.config.text_det_box_thresh
        if self.config.text_det_unclip_ratio is not None:
            kwargs["text_det_unclip_ratio"] = self.config.text_det_unclip_ratio
        if self.config.text_recognition_batch_size is not None:
            kwargs["text_recognition_batch_size"] = self.config.text_recognition_batch_size

        logger.info("Initializing PaddleOCR v3 backend with strict parameters...")
        try:
            return PaddleOCR(**kwargs)
        except TypeError as e:
            logger.critical(f"PaddleOCR v3 parameter mismatch: {e}")
            raise

    def recognize(self, image: np.ndarray, source: str = "paddle") -> List[Token]:
        """
        Run OCR on a single image using predict().
        Returns a list of Token objects.
        """
        try:
            raw_results = self.client.predict(image)
        except Exception as e:
            logger.error(f"PaddleOCR v3 inference failed: {e}")
            return []

        tokens = self._parse_results(raw_results, source=source)
        if not tokens:
            logger.warning(f"OCR returned 0 tokens for {source} (image shape {image.shape})")
        return tokens

    def recognize_iter(
        self,
        images: Iterable[np.ndarray],
        sources: Optional[Iterable[str]] = None,
    ) -> Iterator[List[Token]]:
        """
        Batch OCR using predict_iter() for streaming processing.
        
        Args:
            images: Iterable of BGR images (numpy arrays). Can be a list or generator.
            sources: Optional iterable of source strings for each image.
                     If None, "paddle_batch" is used for all images.
                     Must have same length as images (if both are iterables).
        
        Yields:
            List[Token] for each input image in order.
        
        Example:
            processor = PaddleOCRBackend()
            crops = [crop1, crop2, ...]
            sources = [f"slot_{i}" for i in range(len(crops))]
            for tokens in processor.recognize_iter(crops, sources):
                process(tokens)
        """
        # Prepare source iterator: if sources is None, create an infinite default iterator
        if sources is None:
            def default_src():
                while True:
                    yield "paddle_batch"
            src_iter = default_src()
        else:
            src_iter = iter(sources)

        try:
            # predict_iter returns a generator that yields results one by one
            for result in self.client.predict_iter(images):
                src = next(src_iter)  # Get corresponding source
                # Each result is a dict (same format as single page from predict)
                tokens = self._parse_results([result], source=src)
                yield tokens
        except Exception as e:
            logger.error(f"PaddleOCR v3 streaming inference failed: {e}")
            # Re-raise to let caller handle the error
            raise

    def close(self) -> None:
        self._client = None

    @staticmethod
    def _parse_results(raw_results: Any, source: str) -> List[Token]:
        """
        Parse PaddleOCR v3 predict()/predict_iter() output (PaddleX dict format).
        Input structure: list of dicts, each dict contains:
            'rec_texts': list[str]
            'rec_scores': list[float]
            'rec_polys': list[np.ndarray] of shape (4,2) int16
        Returns list of Token objects.
        """
        tokens: List[Token] = []
        if not raw_results or not isinstance(raw_results, list):
            return tokens

        for page_result in raw_results:            
            if not isinstance(page_result, dict):
                continue

            texts = page_result.get("rec_texts", [])
            scores = page_result.get("rec_scores", [])
            polys = page_result.get("rec_polys", [])

            if not texts or not polys:
                continue

            for idx, (text, poly) in enumerate(zip(texts, polys)):
                # Filter empty text
                if not text or len(text.strip()) == 0:
                    continue
                try:
                    conf = float(scores[idx]) if idx < len(scores) else 0.0
                except (TypeError, ValueError):
                    conf = 0.0
                if conf < 0.25:
                    continue

                # Convert polygon to float32 (4,2)
                try:
                    if isinstance(poly, np.ndarray):
                        box = poly.astype(np.float32).reshape(4, 2)
                    else:
                        box = np.asarray(poly, dtype=np.float32).reshape(4, 2)
                except Exception as e:
                    logger.debug(f"Failed to parse polygon: {e}")
                    continue

                tokens.append(Token(text=text, box=box, score=conf, source=source))

        return tokens