#!/usr/bin/env python3
"""Run Google's MediaPipe Hand Landmarker sample pipeline for comparison."""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
import urllib.request

import cv2
import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_MODEL = pathlib.Path("models/hand_landmarker/hand_landmarker.task")
HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Tron landmarks against Google's MediaPipe Hand Landmarker."
    )
    parser.add_argument("--model", type=pathlib.Path, default=DEFAULT_MODEL)
    parser.add_argument("--download-model", action="store_true")
    parser.add_argument("--input", type=pathlib.Path)
    parser.add_argument("--camera-index", type=int)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("/tmp/tron-google-hands"))
    parser.add_argument("--mode", choices=("image", "video"), default="image")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--min-hand-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-hand-presence-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    return parser.parse_args()


def ensure_model(path: pathlib.Path, download: bool) -> None:
    if path.exists():
        return
    if not download:
        raise FileNotFoundError(
            f"{path} is missing; rerun with --download-model or pass --model"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {MODEL_URL} -> {path}", file=sys.stderr)
    urllib.request.urlretrieve(MODEL_URL, path)


def load_mediapipe():
    try:
        import mediapipe as mp
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Python package 'mediapipe' is missing. Install it in your environment, "
            "for example: python3 -m pip install mediapipe"
        ) from error
    return mp


def create_landmarker(mp, args: argparse.Namespace):
    base_options = mp.tasks.BaseOptions(model_asset_path=str(args.model))
    running_mode = (
        mp.tasks.vision.RunningMode.IMAGE
        if args.mode == "image"
        else mp.tasks.vision.RunningMode.VIDEO
    )
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=running_mode,
        num_hands=1,
        min_hand_detection_confidence=args.min_hand_detection_confidence,
        min_hand_presence_confidence=args.min_hand_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def image_paths(path: pathlib.Path) -> list[pathlib.Path]:
    if path.is_file():
        return [path]
    suffixes = {".bmp", ".jpg", ".jpeg", ".png", ".ppm", ".webp"}
    return sorted(p for p in path.iterdir() if p.suffix.lower() in suffixes)


def mp_image(mp, bgr: np.ndarray):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def detect(landmarker, mp, bgr: np.ndarray, mode: str, timestamp_ms: int):
    image = mp_image(mp, bgr)
    if mode == "image":
        return landmarker.detect(image)
    return landmarker.detect_for_video(image, timestamp_ms)


def draw_result(bgr: np.ndarray, result) -> np.ndarray:
    out = bgr.copy()
    h, w = out.shape[:2]
    if not result.hand_landmarks:
        return out

    points = result.hand_landmarks[0]
    pixels = [
        (
            int(round(point.x * (w - 1))),
            int(round(point.y * (h - 1))),
        )
        for point in points
    ]
    for a, b in HAND_CONNECTIONS:
        cv2.line(out, pixels[a], pixels[b], (0, 230, 255), 1, cv2.LINE_AA)
    for x, y in pixels:
        cv2.drawMarker(
            out,
            (x, y),
            (255, 255, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=8,
            thickness=1,
            line_type=cv2.LINE_AA,
        )

    if result.handedness and result.handedness[0]:
        category = result.handedness[0][0]
        cv2.putText(
            out,
            f"{category.category_name} {category.score:.3f}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return out


def write_result(path: pathlib.Path, frame_id: int, bgr: np.ndarray, result) -> None:
    path.mkdir(parents=True, exist_ok=True)
    suffix = "nohand"
    if result.handedness and result.handedness[0]:
        score = result.handedness[0][0].score
        suffix = f"h{score:.3f}"
    cv2.imwrite(str(path / f"google-hand-{frame_id:08}-{suffix}.png"), bgr)


def run_images(landmarker, mp, args: argparse.Namespace) -> None:
    if args.input is None:
        raise SystemExit("--input is required unless --camera-index is used")
    paths = image_paths(args.input)
    if not paths:
        raise SystemExit(f"no input images found in {args.input}")

    output_id = 0
    index = 0
    while args.max_frames is None or output_id < args.max_frames:
        path = paths[index]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"skip unreadable image {path}", file=sys.stderr)
            index = (index + 1) % len(paths)
            continue
        if args.mirror:
            bgr = cv2.flip(bgr, 1)
        result = detect(landmarker, mp, bgr, args.mode, output_id * 33)
        out = draw_result(bgr, result)
        write_result(args.output, output_id, out, result)
        if args.show:
            cv2.imshow("google-hand-landmarker", out)
            key = cv2.waitKey(0)
            if key == 27:
                break
            if key in (ord("a"), ord("A"), 81, 2424832):
                index = (index - 1) % len(paths)
            else:
                index = (index + 1) % len(paths)
        else:
            index = (index + 1) % len(paths)
        output_id += 1


def run_capture(landmarker, mp, args: argparse.Namespace) -> None:
    source = args.camera_index if args.camera_index is not None else str(args.input)
    loop_input = args.camera_index is None and args.input is not None
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"failed to open video source {source!r}")
    start = time.monotonic()
    frame_id = 0
    while args.max_frames is None or frame_id < args.max_frames:
        ok, bgr = cap.read()
        if not ok:
            if loop_input:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        if args.mirror:
            bgr = cv2.flip(bgr, 1)
        timestamp_ms = int((time.monotonic() - start) * 1000.0)
        result = detect(landmarker, mp, bgr, "video", timestamp_ms)
        out = draw_result(bgr, result)
        write_result(args.output, frame_id, out, result)
        if args.show:
            cv2.imshow("google-hand-landmarker", out)
            if cv2.waitKey(1) == 27:
                break
        frame_id += 1
    cap.release()


def main() -> None:
    args = parse_args()
    ensure_model(args.model, args.download_model)
    mp = load_mediapipe()
    with create_landmarker(mp, args) as landmarker:
        if args.camera_index is not None or (args.input and not args.input.is_dir() and args.mode == "video"):
            run_capture(landmarker, mp, args)
        else:
            run_images(landmarker, mp, args)


if __name__ == "__main__":
    main()
