"""Per-direction video processing with OpenCV + YOLO."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src import config


@dataclass
class DirectionResult:
    direction: str
    video_path: str
    avg_count: float          # average vehicles per sampled frame
    max_count: int
    class_counts: dict        # average per vehicle type
    frames_sampled: int
    annotated_frame: np.ndarray


def find_video(direction: str, videos_dir: Path = config.VIDEOS_DIR) -> Path:
    """Find videos/<direction>.<ext> (e.g. south.mp4)."""
    for ext in config.SUPPORTED_EXTENSIONS:
        path = videos_dir / f"{direction.lower()}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No video for {direction}. Put '{direction.lower()}.mp4' in {videos_dir}"
    )


def _resize(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= config.MAX_FRAME_WIDTH:
        return frame
    scale = config.MAX_FRAME_WIDTH / w
    return cv2.resize(frame, (config.MAX_FRAME_WIDTH, int(h * scale)))


def _sample_frames(cap: cv2.VideoCapture, num_samples: int):
    """Yield evenly spaced frames from the video."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total > 0:
        indices = np.unique(np.linspace(0, total - 1, num_samples).astype(int))
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok:
                yield frame
    else:  # frame count unknown: read sequentially, take every 15th frame
        step, i, yielded = 15, 0, 0
        while yielded < num_samples:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                yield frame
                yielded += 1
            i += 1


def analyze_video(direction: str, detector,
                  num_samples: int = config.NUM_SAMPLE_FRAMES) -> DirectionResult:
    """Run YOLO on sampled frames of one direction's video."""
    path = find_video(direction)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {path}")

    totals = []
    class_sum = {name: 0 for name in config.VEHICLE_CLASSES.values()}
    best_frame, best_count = None, -1

    try:
        for frame in _sample_frames(cap, num_samples):
            annotated, counts = detector.detect(_resize(frame))
            n = sum(counts.values())
            totals.append(n)
            for name, value in counts.items():
                class_sum[name] += value
            if n > best_count:          # keep the frame with most detections
                best_count, best_frame = n, annotated
    finally:
        cap.release()

    if not totals:
        raise RuntimeError(f"No readable frames in video: {path}")

    frames = len(totals)
    return DirectionResult(
        direction=direction,
        video_path=str(path),
        avg_count=round(float(np.mean(totals)), 1),
        max_count=int(max(totals)),
        class_counts={k: round(v / frames, 1) for k, v in class_sum.items()},
        frames_sampled=frames,
        annotated_frame=best_frame,
    )


def analyze_all(detector, num_samples: int = config.NUM_SAMPLE_FRAMES):
    """Process all four directions separately.

    Returns (results, errors). A failure in one direction does not stop the others.
    """
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results, errors = {}, {}
    for direction in config.DIRECTIONS:
        try:
            result = analyze_video(direction, detector, num_samples)
            results[direction] = result
            cv2.imwrite(str(config.RESULTS_DIR / f"{direction.lower()}_detected.jpg"),
                        result.annotated_frame)
        except (FileNotFoundError, RuntimeError) as exc:
            errors[direction] = str(exc)
    return results, errors
