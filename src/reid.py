from __future__ import annotations
import math
from collections import Counter
from typing import Dict, List, Optional, Tuple
from .common import bbox_center, bbox_iou, euclidean, hist_similarity
from .models import AssetState, MachineDetection

class ReIDManager:

    def __init__(self, fps: float, max_missing_frames: int=45, min_similarity: float=0.58, max_center_dist_ratio: float=0.18):
        self.fps = max(fps, 1.0)
        self.max_missing_frames = max_missing_frames
        self.min_similarity = min_similarity
        self.max_center_dist_ratio = max_center_dist_ratio
        self.assets: Dict[str, AssetState] = {}
        self.tracker_to_asset: Dict[int, str] = {}
        self.label_counters: Counter[str] = Counter()

    def _new_asset_id(self, label: str) -> str:
        self.label_counters[label] += 1
        prefix = 'EXC' if label == 'excavator' else 'DTR'
        return f'{prefix}-{self.label_counters[label]:03d}'

    def _new_asset(self, det: MachineDetection, frame_idx: int) -> AssetState:
        asset_id = self._new_asset_id(det.label)
        asset = AssetState(asset_id=asset_id, label=det.label, tracker_id=det.tracker_id, bbox=det.bbox, first_seen_frame=frame_idx, last_seen_frame=frame_idx, appearance_hist=det.appearance_hist)
        self.assets[asset_id] = asset
        self.tracker_to_asset[det.tracker_id] = asset_id
        return asset

    def begin_frame(self) -> None:
        for asset in self.assets.values():
            asset.is_visible = False

    def finish_frame(self) -> None:
        stale_tracker_ids: List[int] = []
        for tracker_id, asset_id in self.tracker_to_asset.items():
            asset = self.assets.get(asset_id)
            if asset is None or not asset.is_visible:
                stale_tracker_ids.append(tracker_id)
        for tracker_id in stale_tracker_ids:
            self.tracker_to_asset.pop(tracker_id, None)
        for asset in self.assets.values():
            if not asset.is_visible:
                asset.missed_frames += 1

    def _candidate_score(self, asset: AssetState, det: MachineDetection, frame_shape: Tuple[int, int, int]) -> float:
        frame_h, frame_w = frame_shape[:2]
        frame_diag = max(1.0, math.hypot(frame_w, frame_h))
        app_sim = hist_similarity(asset.appearance_hist, det.appearance_hist)
        center_dist = euclidean(bbox_center(asset.bbox), bbox_center(det.bbox))
        center_sim = max(0.0, 1.0 - center_dist / (frame_diag * self.max_center_dist_ratio))
        iou_sim = bbox_iou(asset.bbox, det.bbox)
        return 0.55 * app_sim + 0.25 * center_sim + 0.2 * iou_sim

    def assign(self, det: MachineDetection, frame_idx: int, frame_shape: Tuple[int, int, int]) -> AssetState:
        asset_id = self.tracker_to_asset.get(det.tracker_id)
        if asset_id is not None and asset_id in self.assets:
            asset = self.assets[asset_id]
            if asset.label == det.label and (not asset.is_visible):
                asset.tracker_id = det.tracker_id
                asset.bbox = det.bbox
                asset.last_seen_frame = frame_idx
                asset.appearance_hist = det.appearance_hist
                asset.missed_frames = 0
                asset.is_visible = True
                return asset
        best_asset: Optional[AssetState] = None
        best_score = -1.0
        for asset in self.assets.values():
            if asset.label != det.label:
                continue
            if asset.is_visible:
                continue
            if asset.missed_frames > self.max_missing_frames:
                continue
            score = self._candidate_score(asset, det, frame_shape)
            if score > best_score:
                best_score = score
                best_asset = asset
        if best_asset is not None and best_score >= self.min_similarity:
            best_asset.tracker_id = det.tracker_id
            best_asset.bbox = det.bbox
            best_asset.last_seen_frame = frame_idx
            best_asset.appearance_hist = det.appearance_hist
            best_asset.missed_frames = 0
            best_asset.is_visible = True
            self.tracker_to_asset[det.tracker_id] = best_asset.asset_id
            return best_asset
        return self._new_asset(det, frame_idx)
