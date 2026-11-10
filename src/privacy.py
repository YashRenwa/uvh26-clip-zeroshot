"""Redaction for every figure that shows image pixels.

redact(): OpenCV's bundled Haar frontal-face cascade -> Gaussian blur, plus manual boxes from
analysis/redact.csv (stem, x, y, w, h, note) in original-frame pixels, for readable number plates.
Yash checks every report figure by eye and adds manual boxes where needed.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.common import ROOT

REDACT_CSV = ROOT / "analysis" / "redact.csv"


@lru_cache(maxsize=1)
def _face_cascade() -> cv2.CascadeClassifier:
    c = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
    assert not c.empty(), "Haar cascade not found in cv2.data.haarcascades"
    return c


def manual_boxes(stem: str | None, path: Path = REDACT_CSV) -> list[tuple[int, int, int, int]]:
    """Manual redaction boxes (x, y, w, h) in original-frame pixels for one image."""
    if stem is None or not path.exists():
        return []
    df = pd.read_csv(path, dtype={"stem": str})
    return [tuple(int(v) for v in r) for r in df.loc[df["stem"] == stem, ["x", "y", "w", "h"]].to_numpy()]


def _blur(img: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    H, W = img.shape[:2]
    x1, y1, x2, y2 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x2 <= x1 or y2 <= y1:
        return
    k = max(3, (max(x2 - x1, y2 - y1) // 2) | 1)
    img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (k, k), 0)


def redact(img: np.ndarray, stem: str | None = None, origin: tuple[float, float] = (0, 0),
           scale: float = 1.0) -> np.ndarray:
    """Blur detected faces and manual boxes. Returns a copy; the input is not changed.

    img: RGB uint8, either a full frame or a crop/thumbnail of frame `stem`.
    origin, scale: the frame pixel at img[0, 0] and img pixels per frame pixel, so manual boxes
    given in frame pixels land in the right place.
    """
    out = np.ascontiguousarray(img).copy()
    gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
    for x, y, w, h in _face_cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(12, 12)):
        _blur(out, int(x), int(y), int(w), int(h))
    ox, oy = origin
    for x, y, w, h in manual_boxes(stem):
        _blur(out, round((x - ox) * scale), round((y - oy) * scale), round(w * scale) + 1, round(h * scale) + 1)
    return out
