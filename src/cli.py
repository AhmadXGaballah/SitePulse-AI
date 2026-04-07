from __future__ import annotations
import argparse
import json
from .common import LOGGER, configure_logging
from .pipeline import VideoProcessor

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Equipment monitoring with dwell time, Re-ID, and motion analysis')
    parser.add_argument('--detector-model', required=True, type=str, help='Path to machine detector weights (bounding boxes)')
    parser.add_argument('--parts-model', required=True, type=str, help='Path to excavator ARM/BUCKET segmentation weights')
    parser.add_argument('--source', required=True, type=str, help='Path to input video')
    parser.add_argument('--output-video', default=None, type=str, help='Path to output annotated video')
    parser.add_argument('--events-jsonl', default=None, type=str, help='Path to JSONL event stream')
    parser.add_argument('--summary-json', default=None, type=str, help='Path to summary JSON')
    parser.add_argument('--tracker', default='bytetrack.yaml', type=str, help='Ultralytics tracker config')
    parser.add_argument('--device', default=0, help="Inference device, e.g. 0, 'cpu', 'mps'")
    parser.add_argument('--detector-imgsz', default=960, type=int)
    parser.add_argument('--parts-imgsz', default=640, type=int)
    parser.add_argument('--detector-conf', default=0.15, type=float)
    parser.add_argument('--detector-iou', default=0.5, type=float)
    parser.add_argument('--parts-conf', default=0.15, type=float)
    parser.add_argument('--reid-max-missing-frames', default=45, type=int)
    parser.add_argument('--reid-min-similarity', default=0.58, type=float)
    parser.add_argument('--reid-max-center-dist-ratio', default=0.18, type=float)
    parser.add_argument('--display', action='store_true', help='Show live annotated frames')
    parser.add_argument('--verbose', action='store_true', help='Enable debug logs')
    parser.add_argument('--kafka-bootstrap-servers', default=None, type=str)
    parser.add_argument('--kafka-topic', default=None, type=str)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    LOGGER.info('Starting equipment monitoring pipeline')
    LOGGER.info('Detector model: %s', args.detector_model)
    LOGGER.info('Parts model: %s', args.parts_model)
    LOGGER.info('Source: %s', args.source)
    processor = VideoProcessor(args)
    summary = processor.process()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
if __name__ == '__main__':
    main()
