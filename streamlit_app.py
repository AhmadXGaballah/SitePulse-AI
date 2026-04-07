from __future__ import annotations
import json
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import streamlit as st
st.set_page_config(page_title='SitePulse AI', page_icon='🛰️', layout='wide', initial_sidebar_state='expanded')
NEON_CSS = '\n<style>\n:root {\n    --bg-0: #070b14;\n    --bg-1: #0d1324;\n    --card: rgba(19, 29, 54, 0.74);\n    --stroke: rgba(120, 170, 255, 0.18);\n    --text-0: #f5f8ff;\n    --text-1: #c9d6f2;\n    --text-2: #8ea3c7;\n    --cyan: #63e6ff;\n}\nhtml, body, [class*="css"] {\n    font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;\n}\n.stApp {\n    background:\n        radial-gradient(1200px 600px at 10% 0%, rgba(99,230,255,0.08), transparent 60%),\n        radial-gradient(1000px 500px at 100% 10%, rgba(179,136,255,0.10), transparent 50%),\n        linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 55%, #070b14 100%);\n    color: var(--text-0);\n}\n[data-testid="stSidebar"] {\n    background: linear-gradient(180deg, rgba(10,14,26,0.97), rgba(13,19,36,0.97));\n    border-right: 1px solid rgba(110,168,255,0.15);\n}\n.block-container { padding-top: 1.15rem; padding-bottom: 2rem; }\n.hero {\n    background: linear-gradient(135deg, rgba(16,24,46,0.88), rgba(11,18,35,0.78));\n    border: 1px solid var(--stroke);\n    border-radius: 24px;\n    padding: 24px 26px;\n    box-shadow: 0 12px 42px rgba(0,0,0,0.28);\n    backdrop-filter: blur(18px);\n}\n.hero h1 { margin: 0; font-size: 2.15rem; line-height: 1.05; letter-spacing: -0.02em; }\n.hero p { margin: 10px 0 0; color: var(--text-1); font-size: 1rem; }\n.subtle { color: var(--text-2); font-size: 0.92rem; }\n.kpi-card {\n    background: var(--card);\n    border: 1px solid var(--stroke);\n    border-radius: 22px;\n    padding: 16px 18px;\n    box-shadow: 0 12px 36px rgba(0,0,0,0.18);\n    backdrop-filter: blur(16px);\n}\n.kpi-label { color: var(--text-2); font-size: 0.86rem; text-transform: uppercase; letter-spacing: 0.08em; }\n.kpi-value { color: var(--text-0); font-weight: 800; font-size: 1.8rem; margin-top: 6px; }\n.kpi-foot { color: var(--text-1); font-size: 0.88rem; margin-top: 6px; }\n.section-title { color: var(--text-0); font-size: 1.06rem; font-weight: 700; margin: 0 0 10px 0; }\n.asset-card {\n    background: linear-gradient(180deg, rgba(18,27,50,0.88), rgba(11,17,32,0.92));\n    border: 1px solid var(--stroke);\n    border-radius: 20px;\n    padding: 14px 16px;\n    margin-bottom: 12px;\n}\n.asset-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }\n.asset-title { color: var(--text-0); font-weight: 700; font-size: 1rem; }\n.asset-type { color: var(--text-2); font-size: 0.85rem; }\n.pill {\n    display: inline-block;\n    padding: 4px 10px;\n    border-radius: 999px;\n    font-size: 0.78rem;\n    font-weight: 700;\n    letter-spacing: 0.03em;\n    border: 1px solid rgba(255,255,255,0.08);\n    margin-left: 6px;\n}\n.pill-active { background: rgba(85,239,196,0.16); color: #89ffda; }\n.pill-inactive { background: rgba(255,209,102,0.14); color: #ffe39a; }\n.pill-digging, .pill-moving, .pill-swinging_loading, .pill-dumping, .pill-waiting { background: rgba(110,168,255,0.12); color: #b8d1ff; }\n.metric-row { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 8px 12px; margin-top: 10px; }\n.metric-box { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); border-radius: 14px; padding: 10px 12px; }\n.metric-box .label { color: var(--text-2); font-size: 0.78rem; }\n.metric-box .value { color: var(--text-0); font-size: 1rem; font-weight: 700; margin-top: 4px; }\n.run-badge { display: inline-block; padding: 6px 10px; border-radius: 999px; background: rgba(99,230,255,0.10); color: var(--cyan); border: 1px solid rgba(99,230,255,0.18); font-size: 0.80rem; font-weight: 700; }\n</style>\n'
DEFAULT_BACKEND = 'src/sitepulse_ai/cli.py'

def inject_css() -> None:
    st.markdown(NEON_CSS, unsafe_allow_html=True)

def fmt_seconds(value: float) -> str:
    value = max(0.0, float(value))
    if value < 60:
        return f'{value:.1f}s'
    minutes, sec = divmod(int(value), 60)
    if minutes < 60:
        return f'{minutes}m {sec:02d}s'
    hours, minutes = divmod(minutes, 60)
    return f'{hours}h {minutes:02d}m'

def save_uploaded_file(uploaded_file, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / uploaded_file.name
    with open(out, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return out

def read_json_lines(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    return events

def load_summary(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None

def build_asset_dataframe(events: List[Dict[str, Any]]) -> pd.DataFrame:
    latest: Dict[str, Dict[str, Any]] = {}
    for e in events:
        aid = e.get('asset_id') or e.get('equipment_id')
        if not aid:
            continue
        latest[aid] = {'asset_id': aid, 'tracker_id': e.get('tracker_id', '-'), 'equipment_type': e.get('equipment_type') or e.get('equipment_class') or 'unknown', 'current_state': e.get('current_state') or 'UNKNOWN', 'current_activity': e.get('current_activity') or 'UNKNOWN', 'motion_source': e.get('motion_source') or 'none', 'tracked_time_s': float(e.get('tracked_time_s', 0.0) or 0.0), 'active_time_s': float(e.get('active_time_s', 0.0) or 0.0), 'idle_time_s': float(e.get('idle_time_s', 0.0) or 0.0), 'current_dwell_s': float(e.get('current_dwell_s', 0.0) or 0.0), 'max_dwell_s': float(e.get('max_dwell_s', 0.0) or 0.0), 'utilization_pct': float(e.get('utilization_pct', 0.0) or 0.0), 'frame_idx': int(e.get('frame_idx', 0) or 0)}
    if not latest:
        return pd.DataFrame(columns=['asset_id', 'tracker_id', 'equipment_type', 'current_state', 'current_activity', 'motion_source', 'tracked_time_s', 'active_time_s', 'idle_time_s', 'current_dwell_s', 'max_dwell_s', 'utilization_pct', 'frame_idx'])
    return pd.DataFrame(latest.values()).sort_values(['current_dwell_s', 'asset_id'], ascending=[False, True]).reset_index(drop=True)

def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {'visible': 0, 'active': 0, 'waiting': 0, 'fleet_util': 0.0, 'longest_dwell': 0.0, 'working_total': 0.0, 'idle_total': 0.0}
    active = int((df['current_state'] == 'ACTIVE').sum())
    waiting = int((df['current_state'] != 'ACTIVE').sum())
    return {'visible': int(len(df)), 'active': active, 'waiting': waiting, 'fleet_util': float(df['utilization_pct'].mean()), 'longest_dwell': float(df['current_dwell_s'].max()), 'working_total': float(df['active_time_s'].sum()), 'idle_total': float(df['idle_time_s'].sum())}

def pill_class(label: str, kind: str='activity') -> str:
    label = (label or 'unknown').strip().lower().replace('/', '_').replace('-', '_')
    if kind == 'state':
        return 'pill-active' if label == 'active' else 'pill-inactive'
    return f'pill-{label}'

def render_kpis(df: pd.DataFrame) -> None:
    kpis = compute_kpis(df)
    cols = st.columns(6)
    cards = [('Visible', f"{kpis['visible']}", 'Tracked assets in the current scene'), ('Active', f"{kpis['active']}", 'Assets currently classified as working'), ('Waiting', f"{kpis['waiting']}", 'Assets currently inactive or idle'), ('Fleet Util', f"{kpis['fleet_util']:.1f}%", 'Average utilization across visible assets'), ('Working Time', fmt_seconds(kpis['working_total']), 'Cumulative active time'), ('Longest Dwell', fmt_seconds(kpis['longest_dwell']), 'Longest current idle streak')]
    for col, (title, value, foot) in zip(cols, cards):
        with col:
            st.markdown(f"<div class='kpi-card'><div class='kpi-label'>{title}</div><div class='kpi-value'>{value}</div><div class='kpi-foot'>{foot}</div></div>", unsafe_allow_html=True)

def render_asset_cards(df: pd.DataFrame) -> None:
    st.markdown("<div class='section-title'>Live Machine Cards</div>", unsafe_allow_html=True)
    if df.empty:
        st.info('No machine events have arrived yet.')
        return
    cols = st.columns(2)
    for i, row in df.iterrows():
        with cols[i % 2]:
            st.markdown(f"<div class='asset-card'><div class='asset-header'><div><div class='asset-title'>{row['asset_id']}</div><div class='asset-type'>{row['equipment_type']} · tracker {row['tracker_id']}</div></div><div><span class='pill {pill_class(str(row['current_state']), 'state')}'>{row['current_state']}</span><span class='pill {pill_class(str(row['current_activity']))}'>{row['current_activity']}</span></div></div><div class='metric-row'><div class='metric-box'><div class='label'>Current Dwell</div><div class='value'>{fmt_seconds(row['current_dwell_s'])}</div></div><div class='metric-box'><div class='label'>Utilization</div><div class='value'>{row['utilization_pct']:.1f}%</div></div><div class='metric-box'><div class='label'>Working Time</div><div class='value'>{fmt_seconds(row['active_time_s'])}</div></div><div class='metric-box'><div class='label'>Idle Time</div><div class='value'>{fmt_seconds(row['idle_time_s'])}</div></div><div class='metric-box'><div class='label'>Tracked Time</div><div class='value'>{fmt_seconds(row['tracked_time_s'])}</div></div><div class='metric-box'><div class='label'>Motion Source</div><div class='value'>{row['motion_source']}</div></div></div></div>", unsafe_allow_html=True)

def render_live_table(df: pd.DataFrame) -> None:
    st.markdown("<div class='section-title'>Live Asset Table</div>", unsafe_allow_html=True)
    if df.empty:
        st.info('No live machine state available yet.')
        return
    show = df[['asset_id', 'equipment_type', 'current_state', 'current_activity', 'current_dwell_s', 'active_time_s', 'idle_time_s', 'tracked_time_s', 'utilization_pct', 'motion_source']].copy()
    show.columns = ['Asset', 'Type', 'State', 'Activity', 'Current Dwell (s)', 'Working (s)', 'Idle (s)', 'Tracked (s)', 'Util %', 'Motion']
    st.dataframe(show, use_container_width=True, hide_index=True)

def build_command(python_exec: str, backend_script: str, detector_model: str, parts_model: str, source_video: str, output_video: str, events_jsonl: str, summary_json: str, args: Dict[str, Any]) -> List[str]:
    cmd = [python_exec, backend_script, '--detector-model', detector_model, '--parts-model', parts_model, '--source', source_video, '--output-video', output_video, '--events-jsonl', events_jsonl, '--summary-json', summary_json, '--device', str(args['device']), '--detector-imgsz', str(args['detector_imgsz']), '--parts-imgsz', str(args['parts_imgsz']), '--detector-conf', str(args['detector_conf']), '--detector-iou', str(args['detector_iou']), '--parts-conf', str(args['parts_conf']), '--reid-max-missing-frames', str(args['reid_max_missing_frames']), '--reid-min-similarity', str(args['reid_min_similarity']), '--reid-max-center-dist-ratio', str(args['reid_max_center_dist_ratio'])]
    if args.get('verbose'):
        cmd.append('--verbose')
    return cmd

def tail_file(path: Path, last_n: int=100) -> str:
    if not path.exists():
        return ''
    try:
        return '\n'.join(path.read_text(encoding='utf-8', errors='ignore').splitlines()[-last_n:])
    except Exception:
        return ''

def prepare_video_for_streamlit(video_path: Path) -> Path:
    if not video_path.exists():
        return video_path
    converted = video_path.with_name(video_path.stem + '_web.mp4')
    try:
        subprocess.run(['ffmpeg', '-y', '-i', str(video_path), '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(converted)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        if converted.exists() and converted.stat().st_size > 0:
            return converted
    except Exception:
        pass
    return video_path

def stream_run_ui(backend_script: str, detector_model: str, parts_model: str, uploaded_video, run_args: Dict[str, Any]) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    run_root = Path(tempfile.gettempdir()) / 'sitepulse_runs' / datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    run_root.mkdir(parents=True, exist_ok=True)
    source_video = save_uploaded_file(uploaded_video, run_root)
    output_video = run_root / 'processed.mp4'
    events_jsonl = run_root / 'events.jsonl'
    summary_json = run_root / 'summary.json'
    log_file = run_root / 'pipeline.log'
    cmd = build_command(sys.executable, backend_script, detector_model, parts_model, str(source_video), str(output_video), str(events_jsonl), str(summary_json), run_args)
    status_slot = st.empty()
    command_slot = st.empty()
    metrics_slot = st.empty()
    cards_slot = st.empty()
    table_slot = st.empty()
    logs_slot = st.empty()
    with open(log_file, 'w', encoding='utf-8') as lf:
        process = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(Path(backend_script).resolve().parent))
    started_at = time.time()
    while process.poll() is None:
        elapsed = time.time() - started_at
        events = read_json_lines(events_jsonl)
        df = build_asset_dataframe(events)
        status_slot.markdown(f"<span class='run-badge'>RUNNING · {elapsed:.1f}s · {run_root.name}</span>", unsafe_allow_html=True)
        command_slot.markdown(f"<div class='kpi-card'><div class='section-title'>Pipeline Command</div><code>{' '.join((shlex.quote(x) for x in cmd))}</code></div>", unsafe_allow_html=True)
        with metrics_slot.container():
            render_kpis(df)
        with cards_slot.container():
            render_asset_cards(df)
        with table_slot.container():
            render_live_table(df)
        with logs_slot.container():
            st.markdown("<div class='section-title'>Pipeline Logs</div>", unsafe_allow_html=True)
            st.code(tail_file(log_file), language='bash')
        time.sleep(1.0)
    events = read_json_lines(events_jsonl)
    df = build_asset_dataframe(events)
    rc = process.returncode
    badge = 'COMPLETED' if rc == 0 else f'FAILED ({rc})'
    status_slot.markdown(f"<span class='run-badge'>{badge} · {run_root.name}</span>", unsafe_allow_html=True)
    with metrics_slot.container():
        render_kpis(df)
    with cards_slot.container():
        render_asset_cards(df)
    with table_slot.container():
        render_live_table(df)
    with logs_slot.container():
        st.markdown("<div class='section-title'>Pipeline Logs</div>", unsafe_allow_html=True)
        st.code(tail_file(log_file), language='bash')
    return (output_video, events_jsonl, summary_json)

def main() -> None:
    inject_css()
    st.markdown("<div class='hero'><h1>SitePulse AI</h1><p>Equipment utilization, activity intelligence, dwell-time monitoring, and persistent machine identity.</p><div class='subtle'>Run the backend, watch the live state, and review the processed output in one screen.</div></div>", unsafe_allow_html=True)
    st.write('')
    with st.sidebar:
        st.markdown('## Control Plane')
        backend_script = st.text_input('Backend script path', value=DEFAULT_BACKEND)
        detector_model = st.text_input('Detector model path', value='/path/to/detector.pt')
        parts_model = st.text_input('Parts model path', value='/path/to/arm_bucket.pt')
        device = st.selectbox('Device', ['cpu', 'mps', '0'], index=1)
        st.markdown('### Inference')
        detector_imgsz = st.slider('Detector image size', 512, 1280, 960, 32)
        parts_imgsz = st.slider('Parts image size', 320, 960, 640, 32)
        detector_conf = st.slider('Detector confidence', 0.01, 0.9, 0.15, 0.01)
        detector_iou = st.slider('Detector IOU', 0.1, 0.9, 0.5, 0.01)
        parts_conf = st.slider('Parts confidence', 0.01, 0.9, 0.15, 0.01)
        st.markdown('### Re-ID')
        reid_max_missing_frames = st.slider('Max missing frames', 5, 180, 45, 1)
        reid_min_similarity = st.slider('Min similarity', 0.1, 0.95, 0.58, 0.01)
        reid_max_center_dist_ratio = st.slider('Max center distance ratio', 0.05, 0.5, 0.18, 0.01)
        verbose = st.checkbox('Verbose logs', value=True)
    run_args = {'device': device, 'detector_imgsz': detector_imgsz, 'parts_imgsz': parts_imgsz, 'detector_conf': detector_conf, 'detector_iou': detector_iou, 'parts_conf': parts_conf, 'reid_max_missing_frames': reid_max_missing_frames, 'reid_min_similarity': reid_min_similarity, 'reid_max_center_dist_ratio': reid_max_center_dist_ratio, 'verbose': verbose}
    st.markdown('### Video Processing Launcher')
    uploaded_video = st.file_uploader('Upload a construction video clip', type=['mp4', 'mov', 'avi', 'mkv'])
    if st.button('Launch Ops Run', type='primary', use_container_width=True):
        if uploaded_video is None:
            st.error('Upload a video first.')
            return
        if not Path(backend_script).exists():
            st.error(f'Backend script not found: {backend_script}')
            return
        out_video, events_jsonl, summary_json = stream_run_ui(backend_script, detector_model, parts_model, uploaded_video, run_args)
        st.divider()
        if out_video and out_video.exists():
            st.markdown('### Processed Video Feed')
            web_video = prepare_video_for_streamlit(out_video)
            with open(web_video, 'rb') as f:
                video_bytes = f.read()
            st.video(video_bytes)
            st.download_button('Download Processed Video', data=video_bytes, file_name=web_video.name, mime='video/mp4', use_container_width=True)
        if summary_json and summary_json.exists():
            summary = load_summary(summary_json)
            if summary is not None:
                st.markdown('### Summary JSON')
                st.json(summary)
        if events_jsonl and events_jsonl.exists():
            st.success(f'Artifacts saved under: {events_jsonl.parent}')
if __name__ == '__main__':
    main()
