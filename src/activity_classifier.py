from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional
import numpy as np
COMMIT_SCORE = 0.58
HOLD_SCORE = 0.34
HYSTERESIS_FRAMES = 8
EXC_ARM_T = 0.5
EXC_BUCKET_T = 0.48
EXC_BODY_T = 0.8
EXC_SPEED_T = 1.0
EXC_MIN_FRAMES: Dict[str, int] = {'DIGGING': 5, 'SWINGING_LOADING': 4, 'DUMPING': 5, 'WAITING': 22}
DTR_BODY_T = 0.8
DTR_BED_T = 0.8
DTR_SPEED_T = 1.8
DTR_SPEED_CONFIRM_T = 2.6
DTR_MIN_FRAMES: Dict[str, int] = {'MOVING': 8, 'WAITING': 28}

class _Evidence:
    label: str
    commit_threshold: float
    hold_threshold: float
    min_frames: int
    _window: deque = field(default_factory=lambda: deque(maxlen=60))
    streak: int = 0
    below_hold: int = 0

    def push(self, score: float) -> None:
        self._window.append(score)
        if score >= self.commit_threshold:
            self.streak += 1
            self.below_hold = 0
        else:
            self.streak = 0
            if score < self.hold_threshold:
                self.below_hold += 1
            else:
                pass

    @property
    def ready_to_commit(self) -> bool:
        return self.streak >= self.min_frames

    @property
    def should_exit(self) -> bool:
        return self.below_hold >= HYSTERESIS_FRAMES

    @property
    def recent_mean(self) -> float:
        if not self._window:
            return 0.0
        tail = list(self._window)[-10:]
        return float(sum(tail) / len(tail))

    def reset_exit_counter(self) -> None:
        self.below_hold = 0

class ExcavatorActivityClassifier:
    VALID_TRANSITIONS: Dict[str, FrozenSet[str]] = {'WAITING': frozenset({'DIGGING', 'SWINGING_LOADING'}), 'DIGGING': frozenset({'SWINGING_LOADING', 'WAITING'}), 'SWINGING_LOADING': frozenset({'DUMPING', 'DIGGING', 'WAITING'}), 'DUMPING': frozenset({'SWINGING_LOADING', 'WAITING'})}

    def __init__(self, fps: float) -> None:
        self.fps = max(fps, 1.0)
        self.current = 'WAITING'
        self._ev: Dict[str, _Evidence] = {lbl: _Evidence(label=lbl, commit_threshold=COMMIT_SCORE, hold_threshold=HOLD_SCORE, min_frames=EXC_MIN_FRAMES[lbl]) for lbl in ('WAITING', 'DIGGING', 'SWINGING_LOADING', 'DUMPING')}

    @staticmethod
    def _safe_dir(component: float, magnitude: float) -> float:
        return component / (abs(magnitude) + 1e-06)

    def _score_digging(self, m) -> float:
        if m.arm_mag < EXC_ARM_T * 0.28:
            return 0.0
        if abs(m.arm_fx) > abs(m.arm_fy) * 1.4:
            return 0.0
        arm_down_ratio = max(0.0, self._safe_dir(m.arm_fy, m.arm_mag))
        arm_vert = abs(m.arm_fy) / (abs(m.arm_fy) + abs(m.arm_fx) + 1e-06)
        pos_s = float(np.clip((m.bucket_y_norm - 0.45) / 0.4, 0.0, 1.0))
        arm_mag_s = float(np.clip((m.arm_mag - EXC_ARM_T * 0.28) / EXC_ARM_T, 0.0, 1.0))
        bucket_corr = 0.0
        if m.bucket_mag > EXC_BUCKET_T * 0.2:
            bucket_corr = max(0.0, self._safe_dir(m.bucket_fy, m.bucket_mag))
        score = 0.35 * arm_down_ratio + 0.28 * arm_vert + 0.18 * pos_s + 0.12 * arm_mag_s + 0.07 * bucket_corr
        return float(np.clip(score, 0.0, 1.0))

    def _score_swinging(self, m) -> float:
        if m.arm_mag < EXC_ARM_T * 0.28 and m.part_mag < EXC_BUCKET_T * 0.28:
            return 0.0
        if m.bucket_y_norm > 0.65 and m.arm_fy > 0 and (abs(m.arm_fy) > abs(m.arm_fx) * 0.8):
            return 0.0
        arm_horiz = 0.0
        if m.arm_mag > EXC_ARM_T * 0.28:
            arm_fx_ratio = abs(m.arm_fx) / (abs(m.arm_fx) + abs(m.arm_fy) + 1e-06)
            if abs(m.arm_fx) > abs(m.arm_fy) * 1.1:
                arm_horiz = arm_fx_ratio
            else:
                arm_horiz = 0.0
        bucket_horiz = 0.0
        if m.bucket_mag > EXC_BUCKET_T * 0.28:
            bfx_ratio = abs(m.bucket_fx) / (abs(m.bucket_fx) + abs(m.bucket_fy) + 1e-06)
            if abs(m.bucket_fx) > abs(m.bucket_fy) * 1.0:
                bucket_horiz = bfx_ratio
        part_s = float(np.clip((m.part_mag - EXC_BUCKET_T * 0.28) / EXC_BUCKET_T, 0.0, 1.0))
        score = 0.5 * arm_horiz + 0.28 * bucket_horiz + 0.22 * part_s
        return float(np.clip(score, 0.0, 1.0))

    def _score_dumping(self, m) -> float:
        if m.bucket_mag < EXC_BUCKET_T * 0.22 and m.arm_mag < EXC_ARM_T * 0.22:
            return 0.0
        if m.bucket_y_norm > 0.55:
            return 0.0
        if m.arm_fy > 0 and abs(m.arm_fy) > abs(m.arm_fx) * 0.7:
            return 0.0
        if abs(m.arm_fx) > abs(m.arm_fy) * 1.8 and (not getattr(m, 'bucket_in_truck_box', False)) and (not getattr(m, 'bucket_in_truck_core', False)):
            return 0.0
        if not getattr(m, 'has_near_dumptruck', False):
            return 0.0
        pos_s = float(np.clip((0.45 - m.bucket_y_norm) / 0.35, 0.0, 1.0))
        if m.bucket_mag > 1e-06:
            bucket_upward = max(0.0, self._safe_dir(-m.bucket_fy, m.bucket_mag))
        else:
            bucket_upward = 0.2 * pos_s
        arm_s = float(np.clip(m.arm_mag / EXC_ARM_T, 0.0, 1.0))
        dist_norm = float(getattr(m, 'bucket_to_truck_center_norm', 999.0))
        center_s = float(np.clip(1.0 - dist_norm / 0.4, 0.0, 1.0))
        in_box_s = 1.0 if getattr(m, 'bucket_in_truck_box', False) else 0.0
        in_core_s = 1.0 if getattr(m, 'bucket_in_truck_core', False) else 0.0
        truck_s = max(center_s, 0.85 * in_box_s, 1.0 * in_core_s)
        if truck_s < 0.4:
            return 0.0
        score = 0.3 * pos_s + 0.22 * bucket_upward + 0.14 * arm_s + 0.34 * truck_s
        return float(np.clip(score, 0.0, 1.0))

    def _score_waiting(self, m) -> float:
        arm_idle = max(0.0, 1.0 - m.arm_mag / (EXC_ARM_T * 0.35))
        bucket_idle = max(0.0, 1.0 - m.bucket_mag / (EXC_BUCKET_T * 0.4))
        body_idle = max(0.0, 1.0 - m.body_mag / (EXC_BODY_T * 0.6))
        speed_idle = max(0.0, 1.0 - m.center_speed_px / (EXC_SPEED_T * 0.45))
        score = (arm_idle * bucket_idle * body_idle * speed_idle) ** 0.25
        return float(np.clip(score, 0.0, 1.0))

    def update(self, m, machine_is_active: bool) -> str:
        if not machine_is_active:
            for lbl in ('DIGGING', 'SWINGING_LOADING', 'DUMPING'):
                self._ev[lbl].push(0.0)
            self._ev['WAITING'].push(1.0)
            if self.current != 'WAITING' and self._ev['WAITING'].streak >= 4:
                self.current = 'WAITING'
                self._ev['WAITING'].reset_exit_counter()
            return self.current
        scores: Dict[str, float] = {'DIGGING': self._score_digging(m), 'SWINGING_LOADING': self._score_swinging(m), 'DUMPING': self._score_dumping(m), 'WAITING': self._score_waiting(m)}
        for lbl, s in scores.items():
            self._ev[lbl].push(s)
        cur_ev = self._ev[self.current]
        current_exiting = cur_ev.should_exit
        allowed = self.VALID_TRANSITIONS.get(self.current, frozenset())
        best_label: Optional[str] = None
        best_mean: float = -1.0
        for lbl in allowed:
            ev = self._ev[lbl]
            if ev.ready_to_commit and ev.recent_mean > best_mean:
                best_mean = ev.recent_mean
                best_label = lbl
        if current_exiting:
            if best_label is not None:
                self.current = best_label
                self._ev[self.current].reset_exit_counter()
        elif best_label is not None and best_mean > cur_ev.recent_mean + 0.2:
            self.current = best_label
            self._ev[self.current].reset_exit_counter()
        return self.current

class DumpTruckActivityClassifier:
    VALID_TRANSITIONS: Dict[str, FrozenSet[str]] = {'WAITING': frozenset({'MOVING'}), 'MOVING': frozenset({'WAITING'})}

    def __init__(self, fps: float) -> None:
        self.fps = max(fps, 1.0)
        self.current = 'WAITING'
        self._ev: Dict[str, _Evidence] = {lbl: _Evidence(label=lbl, commit_threshold=COMMIT_SCORE, hold_threshold=HOLD_SCORE, min_frames=DTR_MIN_FRAMES[lbl]) for lbl in ('WAITING', 'MOVING')}

    def _score_moving(self, m) -> float:
        speed_s = float(np.clip((m.center_speed_px - DTR_SPEED_T) / (DTR_SPEED_CONFIRM_T - DTR_SPEED_T), 0.0, 1.0))
        body_s = float(np.clip((m.body_mag - DTR_BODY_T) / DTR_BODY_T, 0.0, 1.0))
        if m.center_speed_px >= DTR_SPEED_CONFIRM_T:
            return float(np.clip(0.7 * speed_s + 0.3 * body_s, 0.0, 1.0))
        if speed_s < 0.1:
            return 0.0
        return float(np.clip(0.65 * speed_s + 0.35 * body_s, 0.0, 1.0))

    def _score_waiting(self, m) -> float:
        speed_idle = max(0.0, 1.0 - m.center_speed_px / (DTR_SPEED_T * 0.45))
        body_idle = max(0.0, 1.0 - m.body_mag / (DTR_BODY_T * 0.65))
        bed_idle = max(0.0, 1.0 - m.part_mag / (DTR_BED_T * 0.65))
        score = (speed_idle * body_idle * bed_idle) ** (1.0 / 3.0)
        return float(np.clip(score, 0.0, 1.0))

    def update(self, m, machine_is_active: bool) -> str:
        if not machine_is_active:
            self._ev['MOVING'].push(0.0)
            self._ev['WAITING'].push(1.0)
            if self.current != 'WAITING' and self._ev['WAITING'].streak >= 5:
                self.current = 'WAITING'
                self._ev['WAITING'].reset_exit_counter()
            return self.current
        scores: Dict[str, float] = {'MOVING': self._score_moving(m), 'WAITING': self._score_waiting(m)}
        for lbl, s in scores.items():
            self._ev[lbl].push(s)
        allowed = self.VALID_TRANSITIONS.get(self.current, frozenset())
        cur_ev = self._ev[self.current]
        exiting = cur_ev.should_exit
        best_label: Optional[str] = None
        best_mean: float = -1.0
        for lbl in allowed:
            ev = self._ev[lbl]
            if ev.ready_to_commit and ev.recent_mean > best_mean:
                best_mean = ev.recent_mean
                best_label = lbl
        if exiting:
            if best_label is not None:
                self.current = best_label
                self._ev[self.current].reset_exit_counter()
        elif best_label is not None and best_mean > cur_ev.recent_mean + 0.2:
            self.current = best_label
            self._ev[self.current].reset_exit_counter()
        return self.current
