#!/usr/bin/env python3
"""Command-line entry point for multi-object tracking + cross-camera ReID."""
import argparse
import os
import sys
import time
import cv2
from loguru import logger

from src.cross_camera_matcher import CrossCameraMatcher
from src.detector import ObjectDetector
from src.reid import ReIDEmbedder
from src.tracker import MultiObjectTracker
from src.utils import draw_tracks, ensure_dir, init_csv_writer, load_config, open_video_writer


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 + DeepSORT + ResNet-50 cross-camera ReID")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--inputs", nargs="+", required=True, help="Video paths; use 0 for webcam")
    parser.add_argument("--camera-ids", nargs="+", required=True, help="One ID per input")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def process_single_camera(video_path, camera_id, cfg, detector, tracker_factory,
                          reid_embedder, matcher, output_dir, max_frames=None):
    source = 0 if str(video_path) == "0" else video_path
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input source: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 25.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Could not read video dimensions: {video_path}")

    tracker = tracker_factory()
    ensure_dir(output_dir)
    video_out_path = os.path.join(output_dir, f"{camera_id}_tracked.mp4")
    csv_out_path = os.path.join(output_dir, f"{camera_id}_tracks.csv")
    writer = open_video_writer(video_out_path, fps, width, height) if cfg["io"]["save_video"] else None
    csv_file, csv_writer = init_csv_writer(csv_out_path) if cfg["io"]["save_annotations"] else (None, None)

    local_to_global = {}
    frame_idx = 0
    frame_skip = max(1, int(cfg["runtime"].get("frame_skip", 1)))
    t0 = time.time()
    logger.info(f"[{camera_id}] Processing: {video_path}")

    try:
        while max_frames is None or frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            detections = detector.detect(frame)
            tracks = tracker.update(frame, detections)

            for track_id, x1, y1, x2, y2, cls_id in tracks:
                x1c = max(0, min(width, int(x1)))
                y1c = max(0, min(height, int(y1)))
                x2c = max(0, min(width, int(x2)))
                y2c = max(0, min(height, int(y2)))
                if x2c <= x1c or y2c <= y1c:
                    continue
                crop = frame[y1c:y2c, x1c:x2c]
                embedding = reid_embedder.extract(crop)
                global_id, score = matcher.match_or_register(embedding, class_id=cls_id)
                local_to_global[track_id] = global_id

                if csv_writer:
                    csv_writer.writerow([frame_idx, camera_id, track_id, global_id,
                                         round(float(x1), 2), round(float(y1), 2),
                                         round(float(x2), 2), round(float(y2), 2), cls_id])

            if writer:
                writer.write(draw_tracks(frame, tracks, local_to_global))
            frame_idx += 1
    finally:
        cap.release()
        if writer:
            writer.release()
        if csv_file:
            csv_file.close()

    elapsed = time.time() - t0
    logger.info(f"[{camera_id}] Done: {frame_idx} frames, {frame_idx / max(elapsed, 1e-6):.1f} FPS")
    if writer:
        logger.info(f"Video: {video_out_path}")
    if csv_writer:
        logger.info(f"CSV:   {csv_out_path}")


def main():
    args = parse_args()
    if len(args.inputs) != len(args.camera_ids):
        raise SystemExit("Error: --inputs and --camera-ids must contain the same number of values.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("Error: --max-frames must be greater than 0.")
    if not os.path.isfile(args.config):
        raise SystemExit(f"Error: config file not found: {args.config}")

    cfg = load_config(args.config)
    output_dir = args.output_dir or cfg["io"]["output_dir"]
    ensure_dir(output_dir)
    logger.remove()
    logger.add(sys.stderr, level=cfg["runtime"].get("log_level", "INFO"))

    detector = ObjectDetector(**{
        "model_name": cfg["detector"]["model_name"],
        "confidence_threshold": cfg["detector"]["confidence_threshold"],
        "iou_threshold": cfg["detector"]["iou_threshold"],
        "classes": cfg["detector"]["classes"],
        "device": cfg["detector"]["device"],
    })

    def tracker_factory():
        return MultiObjectTracker(**cfg["tracker"])

    reid_embedder = ReIDEmbedder(
        model_name=cfg["reid"]["model_name"],
        embedding_dim=cfg["reid"]["embedding_dim"],
        device=cfg["detector"]["device"],
    )
    matcher = CrossCameraMatcher(match_threshold=cfg["reid"]["match_threshold"])

    for video_path, camera_id in zip(args.inputs, args.camera_ids):
        process_single_camera(video_path, camera_id, cfg, detector, tracker_factory,
                              reid_embedder, matcher, output_dir, args.max_frames)
    logger.info(f"All streams processed. Results: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
