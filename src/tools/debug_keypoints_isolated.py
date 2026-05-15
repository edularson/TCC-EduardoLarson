"""
debug_keypoints_isolated.py
Teste isolado do FieldDetector (PnLCalib) num frame único.
"""
import sys
import cv2
import numpy as np
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "PnLCalib"))
sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "PnLCalib" / "utils"))
sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "PnLCalib" / "model"))

from modules.field_detection import FieldDetector


def test_field_detector(video_path: str = None, frame_idx: int = 0):
    print("=" * 60)
    print("FieldDetector Isolated Test")
    print("=" * 60)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    detector = FieldDetector()
    print(f"\nDevice: {detector.device}")
    print(f"use_wp_calib: {detector.use_wp_calib}")
    print(f"kp_threshold: {detector.kp_threshold}")
    print(f"line_threshold: {detector.line_threshold}")

    if video_path is None:
        video_path = "src/data/SPAINxCROATIA2.mp4"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nVideo: {video_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        print(f"ERROR: Cannot read frame {frame_idx}")
        cap.release()
        return

    print(f"\nTesting frame {frame_idx}: {frame.shape}")

    print("\nRunning get_homography...")
    H, quality = detector.get_homography(frame)
    print(f"  H: {'None' if H is None else 'computed'}")
    print(f"  quality: {quality:.3f}")

    cv2.imwrite("src/data/debug_keypoints.jpg", frame)

    # ── múltiplos frames ──────────────────────────────────────────────────────
    print("\n--- Testing multiple frames ---")
    test_frames = [0, 50, 100, 200, 300, 400, 500]
    results = []
    for fidx in test_frames:
        if fidx >= total:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, f = cap.read()
        if ret:
            H, q = detector.get_homography(f)
            results.append((fidx, q, H is not None))

    print("\nQuality per frame:")
    for fidx, q, has_H in results:
        bar = "#" * int(q * 20) + "." * (20 - int(q * 20))
        status = "OK" if has_H else "NONE"
        print(f"  Frame {fidx:3d}: [{bar}] {q:.2f} ({status})")

    # ── thresholds mais baixos ────────────────────────────────────────────────
    print("\n--- Testing with lower thresholds ---")
    print("Temporarily lowering thresholds to see if model produces any output...")

    detector.kp_threshold = 0.05
    detector.line_threshold = 0.20

    test_frames2 = [0, 50, 100, 200]
    results2 = []
    for fidx in test_frames2:
        if fidx >= total:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, f = cap.read()
        if ret:
            H, q = detector.get_homography(f)
            results2.append((fidx, q, H is not None))

    print("\nQuality per frame (thresholds kp=0.05, line=0.20):")
    for fidx, q, has_H in results2:
        bar = "#" * int(q * 20) + "." * (20 - int(q * 20))
        status = "OK" if has_H else "NONE"
        print(f"  Frame {fidx:3d}: [{bar}] {q:.2f} ({status})")

    cap.release()
    print("\nDone.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Isolated FieldDetector test")
    parser.add_argument("--video", "-v", type=str, default=None,
                        help="Video path (default: src/data/SPAINxCROATIA2.mp4)")
    parser.add_argument("--frame", "-f", type=int, default=0,
                        help="Frame index to test (default: 0)")
    args = parser.parse_args()

    test_field_detector(args.video, args.frame)