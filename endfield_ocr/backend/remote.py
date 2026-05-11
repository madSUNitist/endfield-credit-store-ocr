# remote_paddle.py
"""
Remote PP-OCRv5 backend using the AIStudio HTTP API.
Compatible with the existing OCRBackend interface.
"""

import time
import base64
import logging
from typing import List, Optional, Iterable, Iterator, Dict, Any

import cv2
import numpy as np
import requests

from .ocr_base import OCRBackend
from ..config import OCRConfig
from ..models import Token

logger = logging.getLogger(__name__)


class RemotePaddleOCRBackend(OCRBackend):
    """
    OCR backend that calls the PP-OCRv5 service deployed on AIStudio.
    The API returns a JSON with 'prunedResult' containing:
        - rec_texts: list of recognized strings
        - rec_scores: list of confidence scores
        - rec_polys: list of polygons (4 points each)
        - rec_boxes: list of axis-aligned rectangles (x1,y1,x2,y2)
    """

    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()
        self.api_url = "https://mfv5nbu4h3u0m0x2.aistudio-app.com/ocr"
        self.token = config.remote_api_token # type: ignore
        self.timeout = config.timeout # type: ignore

    def recognize(self, image: np.ndarray, source: str = "remote") -> List[Token]:
        # Encode image as JPEG base64 (same as before)
        success, encoded = cv2.imencode('.jpg', image)
        if not success:
            logger.error("Failed to encode image to JPEG")
            return []
        img_base64 = base64.b64encode(encoded).decode('ascii')

        payload = {
            "file": img_base64,
            "fileType": 1,
            "useDocOrientationClassify": False,
            "useDocUnwarping": False,
            "useTextlineOrientation": False,
        }
        headers = {
            "Authorization": f"token {self.token}",
            "Content-Type": "application/json"
        }

        data = {}
        max_retries, base_delay = self.config.max_retries, self.config.base_delay
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                break  # Success, exit retry loop
            except Exception as e:
                logger.error(f"Remote OCR request failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    # Last attempt failed
                    return []
                # Exponential backoff before retry
                delay = base_delay * (2 ** attempt)
                logger.info(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)

        # Parse response (same as before)
        result = data.get("result", {})
        ocr_results = result.get("ocrResults", [])
        if not ocr_results:
            logger.warning("No OCR results in response")
            return []

        pruned = ocr_results[0].get("prunedResult", {})
        if not isinstance(pruned, dict):
            logger.error(f"Unexpected prunedResult type: {type(pruned)}")
            return []

        return self._parse_pruned_result(pruned, source)

    def recognize_iter(
        self,
        images: Iterable[np.ndarray],
        sources: Optional[Iterable[str]] = None,
    ) -> Iterator[List[Token]]:
        """
        Process multiple images sequentially (the remote API does not support true batching).
        Yields a list of tokens for each input image.
        """
        if sources is None:
            # Generate default source names
            sources = [f"remote_{i}" for i in range(len(list(images)))]
        for img, src in zip(images, sources):
            yield self.recognize(img, src)

    def close(self) -> None:
        """Nothing to clean up for a remote backend."""
        pass

    @staticmethod
    def _parse_pruned_result(pruned: Dict[str, Any], source: str) -> List[Token]:
        """
        Parse the 'prunedResult' dictionary into a list of Token objects.
        Uses polygons if available, otherwise falls back to bounding boxes.
        """
        tokens = []

        texts = pruned.get("rec_texts", [])
        scores = pruned.get("rec_scores", [])
        polys = pruned.get("rec_polys", [])
        boxes = pruned.get("rec_boxes", [])  # axis-aligned [x1,y1,x2,y2]

        if not texts:
            return tokens

        # Determine which geometry data to use
        use_poly = bool(polys)
        geo_list = polys if use_poly else boxes

        if not geo_list:
            logger.warning("No geometry data (rec_polys or rec_boxes) in response")
            return tokens

        for idx, text in enumerate(texts):
            # Skip empty or whitespace-only text
            if not text or not text.strip():
                continue

            # Get confidence, default to 0.0 if missing
            conf = float(scores[idx]) if idx < len(scores) else 0.0
            if conf < 0.25:  # filter low-confidence detections
                continue

            # Get geometry
            if idx >= len(geo_list):
                break

            geo = geo_list[idx]

            try:
                if use_poly:
                    # rec_polys format: list of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    box = np.array(geo, dtype=np.float32).reshape(4, 2)
                else:
                    # rec_boxes format: [x1, y1, x2, y2] -> convert to 4-point polygon
                    x1, y1, x2, y2 = map(float, geo)
                    box = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            except Exception as e:
                logger.debug(f"Failed to parse geometry for token '{text}': {e}")
                continue

            tokens.append(Token(
                text=text.strip(),
                box=box,
                score=conf,
                source=source
            ))

        return tokens