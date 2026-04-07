from __future__ import annotations
import argparse
import json
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO
from .common import LOGGER, clamp_bbox, compute_hs_hist, ensure_parent, normalize_detector_label
from .detectors import ExcavatorPartsDetector
from .emitters import CompositeEmitter, EventEmitter, JsonlEmitter, KafkaJsonEmitter
from .fusion import MotionFusionEngine
from .models import MachineDetection, PartDetection
from .reid import ReIDManager
from .rendering import color_for_label, draw_box_with_text, draw_parts, draw_professional_dashboard

class VideoProcessor:

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.detector = YOLO(args.detector_model)
        self.parts_detector = ExcavatorPartsDetector(args.parts_model, args.device, args.parts_imgsz, args.parts_conf)
        emitters: List[EventEmitter] = []
        if args.events_jsonl:
            emitters.append(JsonlEmitter(args.events_jsonl))
        if args.kafka_bootstrap_servers and args.kafka_topic:
            emitters.append(KafkaJsonEmitter(args.kafka_bootstrap_servers, args.kafka_topic))
        self.emitter = CompositeEmitter(emitters) if emitters else None

    def _extract_tracks(self, result: Any, frame: np.ndarray) -> List[MachineDetection]:
        tracks: List[MachineDetection] = []
        boxes = getattr(result, 'boxes', None)
        if boxes is None or boxes.xyxy is None or len(boxes) == 0:
            return tracks
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy), dtype=np.float32)
        cls_ids = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=np.int32)
        track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.arange(len(xyxy), dtype=np.int32)
        h, w = frame.shape[:2]
        for i, (box, conf, cls_id, track_id) in enumerate(zip(xyxy, confs, cls_ids, track_ids)):
            raw_name = self.detector.names.get(int(cls_id), str(cls_id))
            label = normalize_detector_label(raw_name)
            if label not in {'excavator', 'dump truck'}:
                continue
            x1, y1, x2, y2 = map(int, box.tolist())
            bbox = clamp_bbox((x1, y1, x2, y2), w, h)
            crop = frame[y1:y2, x1:x2]
            tracks.append(MachineDetection(tracker_id=int(track_id if track_id >= 0 else i), label=label, bbox=bbox, conf=float(conf), appearance_hist=compute_hs_hist(crop)))
        return tracks

    def _compute_excavator_dump_context(self, excavator_track: MachineDetection, parts: Optional[List[PartDetection]], all_tracks: List[MachineDetection]) -> Dict[str, Any]:
        out = {'has_near_dumptruck': False, 'bucket_global_cx': -1.0, 'bucket_global_cy': -1.0, 'truck_center_cx': -1.0, 'truck_center_cy': -1.0, 'bucket_to_truck_center_norm': 999.0, 'bucket_in_truck_box': False, 'bucket_in_truck_core': False}
        if excavator_track.label != 'excavator' or not parts:
            return out
        bucket_parts = [p for p in parts if p.label == 'bucket']
        if not bucket_parts:
            return out
        bucket = max(bucket_parts, key=lambda p: p.conf)
        ex1, ey1, ex2, ey2 = excavator_track.bbox
        if bucket.mask is not None and np.any(bucket.mask):
            ys, xs = np.where(bucket.mask)
            bucket_cx_local = float(np.mean(xs))
            bucket_cy_local = float(np.mean(ys))
        else:
            bx1, by1, bx2, by2 = bucket.bbox
            bucket_cx_local = (bx1 + bx2) / 2.0
            bucket_cy_local = (by1 + by2) / 2.0
        bucket_cx = ex1 + bucket_cx_local
        bucket_cy = ey1 + bucket_cy_local
        out['bucket_global_cx'] = bucket_cx
        out['bucket_global_cy'] = bucket_cy
        dump_trucks = [t for t in all_tracks if t.label == 'dump truck']
        if not dump_trucks:
            return out
        best_truck = None
        best_dist = float('inf')
        for truck in dump_trucks:
            tx1, ty1, tx2, ty2 = truck.bbox
            tcx = (tx1 + tx2) / 2.0
            tcy = (ty1 + ty2) / 2.0
            d = math.hypot(bucket_cx - tcx, bucket_cy - tcy)
            if d < best_dist:
                best_dist = d
                best_truck = truck
        if best_truck is None:
            return out
        tx1, ty1, tx2, ty2 = best_truck.bbox
        tcx = (tx1 + tx2) / 2.0
        tcy = (ty1 + ty2) / 2.0
        tw = max(1.0, tx2 - tx1)
        th = max(1.0, ty2 - ty1)
        out['has_near_dumptruck'] = True
        out['truck_center_cx'] = tcx
        out['truck_center_cy'] = tcy
        out['bucket_to_truck_center_norm'] = best_dist / max(tw, th)
        in_box = tx1 <= bucket_cx <= tx2 and ty1 <= bucket_cy <= ty2
        out['bucket_in_truck_box'] = in_box
        core_margin_x = 0.22 * tw
        core_margin_y = 0.22 * th
        core_x1 = tx1 + core_margin_x
        core_x2 = tx2 - core_margin_x
        core_y1 = ty1 + core_margin_y
        core_y2 = ty2 - core_margin_y
        in_core = core_x1 <= bucket_cx <= core_x2 and core_y1 <= bucket_cy <= core_y2
        out['bucket_in_truck_core'] = in_core
        return out

    @staticmethod
    def _safe_crop(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        return frame[y1:y2, x1:x2].copy()

    def _build_summary(self, reid: ReIDManager, fusion: MotionFusionEngine) -> Dict[str, Any]:
        machines = []
        for asset in sorted(reid.assets.values(), key=lambda a: (a.label, a.asset_id)):
            current_dwell = asset.current_dwell_s(fusion.timeline_s)
            machines.append({'asset_id': asset.asset_id, 'tracker_id': asset.tracker_id, 'equipment_type': asset.label, 'first_seen_frame': asset.first_seen_frame, 'last_seen_frame': asset.last_seen_frame, 'tracked_time_s': round(asset.tracked_time_s, 3), 'active_time_s': round(asset.active_time_s, 3), 'idle_time_s': round(asset.idle_time_s, 3), 'current_dwell_s': round(current_dwell, 3), 'max_dwell_s': round(max(asset.max_dwell_s, current_dwell), 3), 'utilization_pct': round(asset.utilization_pct, 2), 'final_state': asset.current_state, 'final_activity': asset.current_activity})
        return {'generated_at': utc_now_iso(), 'machines': machines, 'totals': {'machine_count': len(machines), 'tracked_time_s': round(sum((m['tracked_time_s'] for m in machines)), 3), 'active_time_s': round(sum((m['active_time_s'] for m in machines)), 3), 'idle_time_s': round(sum((m['idle_time_s'] for m in machines)), 3), 'current_dwell_s_total': round(sum((m['current_dwell_s'] for m in machines)), 3), 'max_dwell_s_overall': round(max((m['max_dwell_s'] for m in machines), default=0.0), 3)}}

    def process(self) -> Dict[str, Any]:
        cap = cv2.VideoCapture(self.args.source)
        if not cap.isOpened():
            raise RuntimeError(f'Failed to open video: {self.args.source}')
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 1e-06:
            fps = 25.0
            LOGGER.warning('FPS could not be read from video. Falling back to 25 FPS.')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        LOGGER.info('Opened video | fps=%.2f | size=%dx%d | frames=%d', fps, width, height, total_frames)
        writer = None
        if self.args.output_video:
            ensure_parent(self.args.output_video)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(self.args.output_video, fourcc, fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f'Failed to create output video: {self.args.output_video}')
        ret, prev_frame = cap.read()
        if not ret:
            raise RuntimeError('Video contains no frames.')
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        fusion = MotionFusionEngine(fps)
        reid = ReIDManager(fps=fps, max_missing_frames=self.args.reid_max_missing_frames, min_similarity=self.args.reid_min_similarity, max_center_dist_ratio=self.args.reid_max_center_dist_ratio)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            fusion.tick()
            reid.begin_frame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            results = self.detector.track(source=frame, persist=True, tracker=self.args.tracker, conf=self.args.detector_conf, iou=self.args.detector_iou, imgsz=self.args.detector_imgsz, device=self.args.device, verbose=False)
            result = results[0]
            detections = self._extract_tracks(result, frame)
            if len(detections) == 0:
                result = self.detector.predict(source=frame, conf=self.args.detector_conf, imgsz=self.args.detector_imgsz, device=self.args.device, verbose=False)[0]
                detections = self._extract_tracks(result, frame)
            if self.args.verbose and frame_idx % 30 == 0:
                raw_boxes = 0 if getattr(result, 'boxes', None) is None else len(result.boxes)
                LOGGER.info('frame %d: raw_boxes=%d, detections=%d', frame_idx, raw_boxes, len(detections))
            visible_assets: List[AssetState] = []
            for det in detections:
                asset = reid.assign(det, frame_idx, frame.shape)
                crop = self._safe_crop(frame, det.bbox)
                parts: Optional[List[PartDetection]] = None
                dump_ctx: Optional[Dict[str, Any]] = None
                if det.label == 'excavator':
                    parts = self.parts_detector.detect(crop)
                    dump_ctx = self._compute_excavator_dump_context(det, parts, detections)
                motion = fusion.update(asset, flow, parts, dump_ctx)
                visible_assets.append(asset)
                payload = fusion.build_payload(asset, frame_idx, det.conf, motion, parts)
                if self.emitter is not None:
                    self.emitter.emit(payload)
                overlay_lines = [f'{asset.asset_id} | trk {det.tracker_id} | {det.conf:.2f}', f'{asset.current_state} | {asset.current_activity} | Dwell {asset.current_dwell_s(fusion.timeline_s):.1f}s', f'Idle total {asset.idle_time_s:.1f}s | Util {asset.utilization_pct:.1f}%', f'Arm {motion.arm_mag:.2f} | Bucket {motion.bucket_mag:.2f} | TruckCore {int(motion.bucket_in_truck_core)}']
                draw_box_with_text(frame, det.bbox, overlay_lines, color_for_label(det.label))
                if parts:
                    draw_parts(frame, det.bbox, parts)
            reid.finish_frame()
            draw_professional_dashboard(frame, visible_assets, fusion.timeline_s)
            if writer is not None:
                writer.write(frame)
            if self.args.display:
                cv2.imshow('Equipment Utilization Prototype', frame)
                key = cv2.waitKey(1) & 255
                if key == 27 or key == ord('q'):
                    LOGGER.info('Interrupted by user.')
                    break
            prev_gray = gray
        cap.release()
        if writer is not None:
            writer.release()
        if self.args.display:
            cv2.destroyAllWindows()
        if self.emitter is not None:
            self.emitter.close()
        summary = self._build_summary(reid, fusion)
        if self.args.summary_json:
            ensure_parent(self.args.summary_json)
            with open(self.args.summary_json, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            LOGGER.info('Saved summary JSON to %s', self.args.summary_json)
        LOGGER.info('Processing complete. Assets tracked: %d', len(summary['machines']))
        return summary
