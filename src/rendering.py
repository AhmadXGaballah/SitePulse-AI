from __future__ import annotations
from typing import List, Tuple
import cv2
import numpy as np
from .models import AssetState, PartDetection

def color_for_label(label: str) -> Tuple[int, int, int]:
    if label == 'excavator':
        return (60, 180, 75)
    if label == 'dump truck':
        return (0, 165, 255)
    return (220, 220, 220)

def draw_box_with_text(frame: np.ndarray, bbox: Tuple[int, int, int, int], lines: List[str], color: Tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    line_height = 18
    max_width = 0
    for line in lines:
        (w, _), _ = cv2.getTextSize(line, font, scale, thickness)
        max_width = max(max_width, w)
    box_height = line_height * len(lines) + 10
    overlay_y1 = max(0, y1 - box_height)
    overlay_y2 = y1
    overlay_x2 = min(frame.shape[1] - 1, x1 + max_width + 14)
    cv2.rectangle(frame, (x1, overlay_y1), (overlay_x2, overlay_y2), color, -1)
    for i, line in enumerate(lines, start=1):
        text_y = overlay_y1 + i * line_height - 4
        cv2.putText(frame, line, (x1 + 6, text_y), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

def draw_parts(frame: np.ndarray, parent_bbox: Tuple[int, int, int, int], parts: List[PartDetection]) -> None:
    px1, py1, px2, py2 = parent_bbox
    crop_h = py2 - py1
    crop_w = px2 - px1
    colors = {'arm': (255, 255, 0), 'bucket': (0, 255, 255)}
    overlay = frame.copy()
    for part in parts:
        color = colors.get(part.label, (255, 255, 255))
        if part.mask is not None:
            mask = part.mask
            if mask.shape != (crop_h, crop_w):
                mask = cv2.resize(mask.astype(np.uint8), (crop_w, crop_h), interpolation=cv2.INTER_NEAREST).astype(bool)
            crop_overlay = overlay[py1:py2, px1:px2]
            crop_overlay[mask] = color
            mask_u8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            shifted = [cnt + np.array([[[px1, py1]]], dtype=cnt.dtype) for cnt in contours]
            cv2.drawContours(overlay, shifted, -1, color, 2)
            ys, xs = np.where(mask)
            if len(xs) > 0 and len(ys) > 0:
                cx = int(px1 + np.mean(xs))
                cy = int(py1 + np.mean(ys))
                cv2.putText(overlay, f'{part.label} {part.conf:.2f}', (cx, max(16, cy - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)
        else:
            x1, y1, x2, y2 = part.bbox
            gx1, gy1, gx2, gy2 = (px1 + x1, py1 + y1, px1 + x2, py1 + y2)
            cv2.rectangle(overlay, (gx1, gy1), (gx2, gy2), color, 2)
            cv2.putText(overlay, f'{part.label} {part.conf:.2f}', (gx1, max(16, gy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

def draw_professional_dashboard(frame: np.ndarray, visible_assets: List[AssetState], timeline_s: float) -> None:
    h, w = frame.shape[:2]
    panel_w = min(560, w - 20)
    panel_x = 10
    panel_y = 10
    row_h = 26
    rows = min(6, len(visible_assets))
    panel_h = 168 + rows * row_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (20, 22, 28), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    header_color = (240, 240, 240)
    muted = (175, 185, 195)
    accent = (110, 215, 255)
    active_assets = sum((1 for a in visible_assets if a.current_state == 'ACTIVE'))
    idle_assets = sum((1 for a in visible_assets if a.current_state == 'INACTIVE'))
    fleet_util = np.mean([a.utilization_pct for a in visible_assets]).item() if visible_assets else 0.0
    longest_dwell = max([a.current_dwell_s(timeline_s) for a in visible_assets], default=0.0)
    cv2.putText(frame, 'Site Equipment Operations Monitor', (panel_x + 16, panel_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.78, header_color, 2, cv2.LINE_AA)
    cv2.putText(frame, 'Activity, Dwell Time, and Persistent Asset Identity', (panel_x + 16, panel_y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.46, muted, 1, cv2.LINE_AA)
    cards = [('Visible', f'{len(visible_assets)}'), ('Active', f'{active_assets}'), ('Waiting', f'{idle_assets}'), ('Fleet Util', f'{fleet_util:.1f}%'), ('Longest Dwell', f'{longest_dwell:.1f}s')]
    card_y = panel_y + 68
    card_w = 102
    for i, (title, value) in enumerate(cards):
        x1 = panel_x + 14 + i * (card_w + 8)
        x2 = x1 + card_w
        cv2.rectangle(frame, (x1, card_y), (x2, card_y + 48), (40, 44, 54), -1)
        cv2.putText(frame, title, (x1 + 8, card_y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, muted, 1, cv2.LINE_AA)
        cv2.putText(frame, value, (x1 + 8, card_y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.58, accent, 2, cv2.LINE_AA)
    table_y = card_y + 66
    cv2.putText(frame, 'Asset', (panel_x + 16, table_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, muted, 1, cv2.LINE_AA)
    cv2.putText(frame, 'Type', (panel_x + 102, table_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, muted, 1, cv2.LINE_AA)
    cv2.putText(frame, 'State', (panel_x + 188, table_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, muted, 1, cv2.LINE_AA)
    cv2.putText(frame, 'Activity', (panel_x + 260, table_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, muted, 1, cv2.LINE_AA)
    cv2.putText(frame, 'Dwell', (panel_x + 392, table_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, muted, 1, cv2.LINE_AA)
    cv2.putText(frame, 'Util', (panel_x + 474, table_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, muted, 1, cv2.LINE_AA)
    ranked = sorted(visible_assets, key=lambda a: (a.current_dwell_s(timeline_s), a.asset_id), reverse=True)[:rows]
    for idx, asset in enumerate(ranked, start=1):
        y = table_y + idx * row_h
        state_color = (100, 230, 120) if asset.current_state == 'ACTIVE' else (90, 190, 255)
        cv2.putText(frame, asset.asset_id, (panel_x + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, header_color, 1, cv2.LINE_AA)
        cv2.putText(frame, asset.label, (panel_x + 102, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, header_color, 1, cv2.LINE_AA)
        cv2.putText(frame, asset.current_state, (panel_x + 188, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, state_color, 1, cv2.LINE_AA)
        cv2.putText(frame, asset.current_activity, (panel_x + 260, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, header_color, 1, cv2.LINE_AA)
        cv2.putText(frame, f'{asset.current_dwell_s(timeline_s):.1f}s', (panel_x + 392, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, header_color, 1, cv2.LINE_AA)
        cv2.putText(frame, f'{asset.utilization_pct:.0f}%', (panel_x + 474, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, header_color, 1, cv2.LINE_AA)
