from __future__ import annotations
from typing import Any, List
import cv2
import numpy as np
from ultralytics import YOLO
from .common import clamp_bbox, normalize_parts_label
from .models import PartDetection

class ExcavatorPartsDetector:

    def __init__(self, model_path: str, device: Any=0, imgsz: int=640, conf: float=0.15):
        self.model = YOLO(model_path)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf

    def detect(self, crop_bgr: np.ndarray) -> List[PartDetection]:
        results = self.model.predict(source=crop_bgr, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)
        result = results[0]
        boxes = getattr(result, 'boxes', None)
        if boxes is None or boxes.xyxy is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy), dtype=np.float32)
        cls_ids = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=np.int32)
        masks_np = None
        if getattr(result, 'masks', None) is not None and result.masks.data is not None:
            masks_np = result.masks.data.cpu().numpy()
        h, w = crop_bgr.shape[:2]
        output: List[PartDetection] = []
        for i, (box, conf, cls_id) in enumerate(zip(xyxy, confs, cls_ids)):
            raw_name = result.names.get(int(cls_id), str(cls_id))
            label = normalize_parts_label(raw_name)
            if label not in {'arm', 'bucket'}:
                continue
            x1, y1, x2, y2 = map(int, box.tolist())
            local_mask = None
            if masks_np is not None and i < len(masks_np):
                m = masks_np[i]
                if m.shape[:2] != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                local_mask = m > 0.5
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            output.append(PartDetection(label=label, bbox=(x1, y1, x2, y2), conf=float(conf), mask=local_mask, center=center))
        return output
