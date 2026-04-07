from __future__ import annotations
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np
from .activity_classifier import DumpTruckActivityClassifier, ExcavatorActivityClassifier
from .common import bbox_center, clamp_bbox, euclidean, utc_now_iso
from .models import AssetState, MotionFeatures, PartDetection

class MotionFusionEngine:

    def __init__(self, fps: float):
        self.fps = max(fps, 1.0)
        self.dt = 1.0 / self.fps
        self.timeline_s = 0.0
        self.T_BODY_MOVE = 0.75
        self.T_PART_MOVE = 0.65
        self.T_BED_MOVE = 0.8
        self.T_IDLE = 0.12
        self.T_CENTER_MOVE = 1.0
        self.T_ACTIVE_RATIO = 0.015
        self.N_ACTIVE = 2
        self.N_IDLE = 18
        self._classifiers: Dict[str, Any] = {}

    def tick(self) -> None:
        self.timeline_s += self.dt

    @staticmethod
    def _roi_stats(flow_roi: np.ndarray, motion_threshold: float) -> Tuple[float, float, float, float]:
        if flow_roi.size == 0:
            return (0.0, 0.0, 0.0, 0.0)
        fx = flow_roi[..., 0]
        fy = flow_roi[..., 1]
        mag = np.sqrt(fx * fx + fy * fy)
        active_ratio = float(np.mean(mag > motion_threshold))
        return (float(np.mean(mag)), float(np.mean(fx)), float(np.mean(fy)), active_ratio)

    @staticmethod
    def _masked_flow_stats(flow_roi: np.ndarray, mask: Optional[np.ndarray], motion_threshold: float) -> Tuple[float, float, float, float]:
        if flow_roi.size == 0 or mask is None:
            return (0.0, 0.0, 0.0, 0.0)
        if mask.shape != flow_roi.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (flow_roi.shape[1], flow_roi.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        if not np.any(mask):
            return (0.0, 0.0, 0.0, 0.0)
        fx = flow_roi[..., 0][mask]
        fy = flow_roi[..., 1][mask]
        if fx.size == 0:
            return (0.0, 0.0, 0.0, 0.0)
        mag = np.sqrt(fx * fx + fy * fy)
        active_ratio = float(np.mean(mag > motion_threshold))
        return (float(np.mean(mag)), float(np.mean(fx)), float(np.mean(fy)), active_ratio)

    def _ensure_classifier(self, asset: AssetState) -> None:
        if asset.asset_id not in self._classifiers:
            if asset.label == 'excavator':
                self._classifiers[asset.asset_id] = ExcavatorActivityClassifier(self.fps)
            else:
                self._classifiers[asset.asset_id] = DumpTruckActivityClassifier(self.fps)

    def _center_speed(self, asset: AssetState, bbox: Tuple[int, int, int, int]) -> float:
        center = bbox_center(bbox)
        if asset.last_center is None:
            asset.last_center = center
            return 0.0
        speed = euclidean(center, asset.last_center)
        asset.last_center = center
        return speed

    def _compute_excavator_motion(self, asset: AssetState, flow: np.ndarray, parts: List[PartDetection]) -> MotionFeatures:
        x1, y1, x2, y2 = asset.bbox
        crop_flow = flow[y1:y2, x1:x2]
        h, w = crop_flow.shape[:2]
        center_speed = self._center_speed(asset, asset.bbox)
        if h < 2 or w < 2:
            return MotionFeatures(center_speed_px=center_speed)
        arm_mask = np.zeros((h, w), dtype=bool)
        bucket_mask = np.zeros((h, w), dtype=bool)
        for part in parts:
            if part.mask is not None and part.mask.shape == (h, w):
                if part.label == 'arm':
                    arm_mask |= part.mask
                elif part.label == 'bucket':
                    bucket_mask |= part.mask
            else:
                px1, py1, px2, py2 = clamp_bbox(part.bbox, w, h)
                if part.label == 'arm':
                    arm_mask[py1:py2, px1:px2] = True
                elif part.label == 'bucket':
                    bucket_mask[py1:py2, px1:px2] = True
        lower_mask = np.zeros((h, w), dtype=bool)
        lower_mask[int(0.55 * h):, :] = True
        body_mask = lower_mask & ~(arm_mask | bucket_mask)
        body_mag, body_fx, body_fy, body_ratio = self._masked_flow_stats(crop_flow, body_mask, self.T_BODY_MOVE)
        arm_mag, arm_fx, arm_fy, arm_ratio = self._masked_flow_stats(crop_flow, arm_mask, self.T_PART_MOVE)
        bucket_mag, bucket_fx, bucket_fy, bucket_ratio = self._masked_flow_stats(crop_flow, bucket_mask, self.T_PART_MOVE * 0.9)
        if np.any(bucket_mask):
            ys, xs = np.where(bucket_mask)
            bucket_y_norm = float(np.mean(ys) / max(h - 1, 1))
            bucket_x_norm = float(np.mean(xs) / max(w - 1, 1))
        else:
            bucket_y_norm = 1.0
            bucket_x_norm = 0.5
        if bucket_mag > 0:
            part_mag = 0.35 * arm_mag + 0.65 * bucket_mag
            part_fx = 0.35 * arm_fx + 0.65 * bucket_fx
            part_fy = 0.35 * arm_fy + 0.65 * bucket_fy
        else:
            part_mag = arm_mag
            part_fx = arm_fx
            part_fy = arm_fy
        part_ratio = max(arm_ratio, bucket_ratio)
        motion_source = 'none'
        if bucket_mag > self.T_PART_MOVE * 0.9 and body_mag < self.T_BODY_MOVE:
            motion_source = 'arm_only'
        elif part_mag > self.T_PART_MOVE and body_mag >= self.T_BODY_MOVE:
            motion_source = 'body_and_arm'
        elif body_mag >= self.T_BODY_MOVE or center_speed >= self.T_CENTER_MOVE:
            motion_source = 'body_only'
        return MotionFeatures(body_mag=body_mag, body_fx=body_fx, body_fy=body_fy, body_active_ratio=body_ratio, part_mag=part_mag, part_fx=part_fx, part_fy=part_fy, part_active_ratio=part_ratio, arm_mag=arm_mag, arm_fx=arm_fx, arm_fy=arm_fy, bucket_mag=bucket_mag, bucket_fx=bucket_fx, bucket_fy=bucket_fy, bucket_x_norm=bucket_x_norm, bucket_y_norm=bucket_y_norm, center_speed_px=center_speed, motion_source_candidate=motion_source, has_part_detections=bool(parts))

    def _compute_dump_truck_motion(self, asset: AssetState, flow: np.ndarray) -> MotionFeatures:
        x1, y1, x2, y2 = asset.bbox
        crop_flow = flow[y1:y2, x1:x2]
        h, w = crop_flow.shape[:2]
        center_speed = self._center_speed(asset, asset.bbox)
        if h < 2 or w < 2:
            return MotionFeatures(center_speed_px=center_speed)
        split_y = int(0.45 * h)
        bed_roi = crop_flow[:split_y, :]
        body_roi = crop_flow[split_y:, :]
        body_mag, body_fx, body_fy, body_ratio = self._roi_stats(body_roi, self.T_BODY_MOVE)
        part_mag, part_fx, part_fy, part_ratio = self._roi_stats(bed_roi, self.T_BED_MOVE)
        motion_source = 'none'
        if part_mag > self.T_BED_MOVE and body_mag < self.T_BODY_MOVE:
            motion_source = 'bed_only'
        elif part_mag > self.T_BED_MOVE and body_mag >= self.T_BODY_MOVE:
            motion_source = 'body_and_bed'
        elif body_mag >= self.T_BODY_MOVE or center_speed >= self.T_CENTER_MOVE:
            motion_source = 'body_only'
        return MotionFeatures(body_mag=body_mag, body_fx=body_fx, body_fy=body_fy, body_active_ratio=body_ratio, part_mag=part_mag, part_fx=part_fx, part_fy=part_fy, part_active_ratio=part_ratio, center_speed_px=center_speed, motion_source_candidate=motion_source, has_part_detections=False)

    def _attach_dump_context(self, motion: MotionFeatures, dump_ctx: Optional[Dict[str, Any]]) -> MotionFeatures:
        if not dump_ctx:
            return motion
        motion.has_near_dumptruck = bool(dump_ctx.get('has_near_dumptruck', False))
        motion.bucket_global_cx = float(dump_ctx.get('bucket_global_cx', -1.0))
        motion.bucket_global_cy = float(dump_ctx.get('bucket_global_cy', -1.0))
        motion.truck_center_cx = float(dump_ctx.get('truck_center_cx', -1.0))
        motion.truck_center_cy = float(dump_ctx.get('truck_center_cy', -1.0))
        motion.bucket_to_truck_center_norm = float(dump_ctx.get('bucket_to_truck_center_norm', 999.0))
        motion.bucket_in_truck_box = bool(dump_ctx.get('bucket_in_truck_box', False))
        motion.bucket_in_truck_core = bool(dump_ctx.get('bucket_in_truck_core', False))
        return motion

    @staticmethod
    def _majority_vote(window: Sequence[str], default: str) -> str:
        if not window:
            return default
        return Counter(window).most_common(1)[0][0]

    def _update_dwell(self, asset: AssetState) -> None:
        if asset.current_state == 'INACTIVE':
            if asset.dwell_start_s is None:
                asset.dwell_start_s = self.timeline_s
        elif asset.dwell_start_s is not None:
            asset.max_dwell_s = max(asset.max_dwell_s, self.timeline_s - asset.dwell_start_s)
            asset.dwell_start_s = None

    def _fuse_excavator(self, asset: AssetState, motion: MotionFeatures) -> None:
        motion_active = motion.bucket_mag > self.T_PART_MOVE or motion.arm_mag > self.T_PART_MOVE or motion.part_active_ratio > self.T_ACTIVE_RATIO or (motion.body_mag > self.T_BODY_MOVE) or (motion.center_speed_px > self.T_CENTER_MOVE)
        motion_idle = motion.bucket_mag < self.T_IDLE and motion.arm_mag < self.T_IDLE and (motion.body_mag < self.T_IDLE) and (motion.center_speed_px < self.T_CENTER_MOVE * 0.3) and (motion.part_active_ratio < 0.008) and (motion.body_active_ratio < 0.008)
        if motion_active:
            asset.active_counter += 1
            asset.idle_counter = 0
        elif motion_idle:
            asset.idle_counter += 1
            asset.active_counter = 0
        else:
            asset.idle_counter = max(0, asset.idle_counter - 1)
        if asset.active_counter >= self.N_ACTIVE:
            asset.current_state = 'ACTIVE'
        if asset.idle_counter >= self.N_IDLE:
            asset.current_state = 'INACTIVE'
        if asset.current_state == 'ACTIVE':
            asset.motion_source = motion.motion_source_candidate if motion.motion_source_candidate != 'none' else 'arm_only'
        else:
            asset.motion_source = 'none'
        self._ensure_classifier(asset)
        clf = self._classifiers[asset.asset_id]
        asset.current_activity = clf.update(motion, asset.current_state == 'ACTIVE')
        self._update_dwell(asset)

    def _fuse_dump_truck(self, asset: AssetState, motion: MotionFeatures) -> None:
        motion_active = motion.center_speed_px > self.T_CENTER_MOVE and motion.body_mag > self.T_BODY_MOVE * 0.6 or motion.part_mag > self.T_BED_MOVE or motion.body_active_ratio > self.T_ACTIVE_RATIO
        motion_idle = motion.part_mag < self.T_IDLE and motion.body_mag < self.T_IDLE and (motion.center_speed_px < self.T_CENTER_MOVE * 0.3) and (motion.body_active_ratio < 0.008)
        if motion_active:
            asset.active_counter += 1
            asset.idle_counter = 0
        elif motion_idle:
            asset.idle_counter += 1
            asset.active_counter = 0
        else:
            asset.idle_counter = max(0, asset.idle_counter - 1)
        if asset.active_counter >= self.N_ACTIVE:
            asset.current_state = 'ACTIVE'
        if asset.idle_counter >= self.N_IDLE:
            asset.current_state = 'INACTIVE'
        if asset.current_state == 'ACTIVE':
            asset.motion_source = motion.motion_source_candidate if motion.motion_source_candidate != 'none' else 'body_only'
        else:
            asset.motion_source = 'none'
        asset.current_activity = 'WAITING' if asset.current_state == 'ACTIVE' else 'MOVING'
        self._update_dwell(asset)

    def update(self, asset: AssetState, flow: np.ndarray, parts: Optional[List[PartDetection]], dump_ctx: Optional[Dict[str, Any]]=None) -> MotionFeatures:
        if asset.label == 'excavator':
            motion = self._compute_excavator_motion(asset, flow, parts or [])
            motion = self._attach_dump_context(motion, dump_ctx)
            self._fuse_excavator(asset, motion)
        else:
            motion = self._compute_dump_truck_motion(asset, flow)
            self._fuse_dump_truck(asset, motion)
        asset.tracked_time_s += self.dt
        if asset.current_state == 'ACTIVE':
            asset.active_time_s += self.dt
        else:
            asset.idle_time_s += self.dt
        return motion

    def build_payload(self, asset: AssetState, frame_idx: int, detector_conf: float, motion: MotionFeatures, parts: Optional[List[PartDetection]]=None) -> Dict[str, Any]:
        payload = {'timestamp': utc_now_iso(), 'frame_idx': frame_idx, 'asset_id': asset.asset_id, 'tracker_id': asset.tracker_id, 'equipment_type': asset.label, 'bbox': [int(v) for v in asset.bbox], 'detector_confidence': round(detector_conf, 4), 'current_state': asset.current_state, 'current_activity': asset.current_activity, 'motion_source': asset.motion_source, 'tracked_time_s': round(asset.tracked_time_s, 3), 'active_time_s': round(asset.active_time_s, 3), 'idle_time_s': round(asset.idle_time_s, 3), 'current_dwell_s': round(asset.current_dwell_s(self.timeline_s), 3), 'max_dwell_s': round(max(asset.max_dwell_s, asset.current_dwell_s(self.timeline_s)), 3), 'utilization_pct': round(asset.utilization_pct, 2), 'motion_features': {'body_mag': round(motion.body_mag, 4), 'part_mag': round(motion.part_mag, 4), 'arm_mag': round(motion.arm_mag, 4), 'bucket_mag': round(motion.bucket_mag, 4), 'bucket_x_norm': round(motion.bucket_x_norm, 4), 'bucket_y_norm': round(motion.bucket_y_norm, 4), 'center_speed_px': round(motion.center_speed_px, 4), 'has_part_detections': motion.has_part_detections, 'has_near_dumptruck': motion.has_near_dumptruck, 'bucket_to_truck_center_norm': round(motion.bucket_to_truck_center_norm, 4), 'bucket_in_truck_box': motion.bucket_in_truck_box, 'bucket_in_truck_core': motion.bucket_in_truck_core}}
        if parts:
            payload['parts'] = [{'label': p.label, 'bbox_local': [int(v) for v in p.bbox], 'confidence': round(p.conf, 4)} for p in parts]
        return payload
