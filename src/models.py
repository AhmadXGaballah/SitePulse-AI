from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple
import numpy as np

class PartDetection:
    label: str
    bbox: Tuple[int, int, int, int]
    conf: float
    mask: Optional[np.ndarray] = None
    center: Tuple[float, float] = (0.0, 0.0)

class MachineDetection:
    tracker_id: int
    label: str
    bbox: Tuple[int, int, int, int]
    conf: float
    appearance_hist: Optional[np.ndarray] = None

class MotionFeatures:
    body_mag: float = 0.0
    body_fx: float = 0.0
    body_fy: float = 0.0
    body_active_ratio: float = 0.0
    part_mag: float = 0.0
    part_fx: float = 0.0
    part_fy: float = 0.0
    part_active_ratio: float = 0.0
    arm_mag: float = 0.0
    arm_fx: float = 0.0
    arm_fy: float = 0.0
    bucket_mag: float = 0.0
    bucket_fx: float = 0.0
    bucket_fy: float = 0.0
    bucket_x_norm: float = 0.5
    bucket_y_norm: float = 1.0
    center_speed_px: float = 0.0
    motion_source_candidate: str = 'none'
    has_part_detections: bool = False
    has_near_dumptruck: bool = False
    bucket_global_cx: float = -1.0
    bucket_global_cy: float = -1.0
    truck_center_cx: float = -1.0
    truck_center_cy: float = -1.0
    bucket_to_truck_center_norm: float = 999.0
    bucket_in_truck_box: bool = False
    bucket_in_truck_core: bool = False

class AssetState:
    asset_id: str
    label: str
    tracker_id: int
    bbox: Tuple[int, int, int, int]
    first_seen_frame: int
    last_seen_frame: int
    current_state: str = 'INACTIVE'
    current_activity: str = 'WAITING'
    motion_source: str = 'none'
    tracked_time_s: float = 0.0
    active_time_s: float = 0.0
    idle_time_s: float = 0.0
    dwell_start_s: Optional[float] = None
    max_dwell_s: float = 0.0
    last_center: Optional[Tuple[float, float]] = None
    active_counter: int = 0
    idle_counter: int = 0
    activity_window: Deque[str] = field(default_factory=lambda: deque(maxlen=7))
    state_window: Deque[str] = field(default_factory=lambda: deque(maxlen=7))
    appearance_hist: Optional[np.ndarray] = None
    missed_frames: int = 0
    is_visible: bool = False

    @property
    def utilization_pct(self) -> float:
        if self.tracked_time_s <= 1e-09:
            return 0.0
        return 100.0 * self.active_time_s / self.tracked_time_s

    def current_dwell_s(self, timeline_s: float) -> float:
        if self.current_state != 'INACTIVE' or self.dwell_start_s is None:
            return 0.0
        return max(0.0, timeline_s - self.dwell_start_s)
