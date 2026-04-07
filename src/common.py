from __future__ import annotations
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
LOGGER = logging.getLogger('sitepulse_ai')

def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s', datefmt='%H:%M:%S')

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')

def ensure_parent(path: Optional[str]) -> None:
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def clamp_bbox(bbox: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return (x1, y1, x2, y2)

def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = (max(ax1, bx1), max(ay1, by1))
    ix2, iy2 = (min(ax2, bx2), min(ay2, by2))
    iw, ih = (max(0, ix2 - ix1), max(0, iy2 - iy1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)

def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def normalize_detector_label(raw: str) -> str:
    s = raw.strip().lower().replace('_', ' ').replace('-', ' ')
    s = ' '.join(s.split())
    if s in {'excavator', 'excavators'}:
        return 'excavator'
    if s in {'dump truck', 'dumptruck', 'dump trucks'}:
        return 'dump truck'
    return s

def normalize_parts_label(raw: str) -> str:
    s = raw.strip().lower().replace('_', ' ').replace('-', ' ')
    s = ' '.join(s.split())
    if s in {'arm', 'stick'}:
        return 'arm'
    if s in {'bucket', 'bucket tip', 'bucket face'}:
        return 'bucket'
    return s

def compute_hs_hist(crop_bgr: np.ndarray) -> np.ndarray:
    if crop_bgr.size == 0:
        return np.zeros((16 * 16,), dtype=np.float32)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
    return hist

def hist_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None or a.size == 0 or (b.size == 0):
        return 0.0
    score = float(cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_CORREL))
    return max(0.0, min(1.0, (score + 1.0) / 2.0))
