#!/usr/bin/env python3
"""Score CourtKeyNet against hand-annotated court corners.

Companion to ``annotate_court_corners_offframe.py``. Where ``ckn_corner_eval.py`` scores
the detector against ShuttleSet's recorded homographies in a 1280x720 reference
space, this scores it against hand annotations, which have no reference space:
error is measured in the video's own native pixels.

For each annotated frame it reads the frame, runs the detector, and takes the
per-corner Euclidean error in native pixels: the detected quad's contractual
TL TR BR BL slot order (src/courtkeynet/PROVENANCE.md) against the hand quad's
``corner_label`` slot order, corner to same-named corner, with no re-sorting on
either side. That makes the score slot-wise: a detection that genuinely emits
the wrong slot order scores its own honest per-slot error rather than being
geometrically re-paired with its nearest hand corner. Gate signals (peak,
entropy, flags, pass) are carried through so the same reporting as the
reference-homography eval applies.

Run from a checkout so the detector import resolves (``--repo-root``); works on
CPU or CUDA.

Usage::

    python src/courtkeynet/validation_scripts/score_hand_corners.py \\
        --annotations-csv hand_corners.csv --video path/to/clip.mp4 \\
        --out-csv hand_scores.csv --device cpu --resize-mode pad
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Sibling import: this script runs from its own directory (python puts the
# script's dir on sys.path), and the test file loads it by file path, where
# __file__ still resolves. Matches annotate_court_corners_offframe.py's route,
# which keeps this module importable without pulling in the courtkeynet
# package's torch-loading __init__.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from court_landmarks import CORNER_NAMES  # noqa: E402


# --- Pure helpers: error (unit-tested) --------------------------------------

def per_corner_error(quad_a: np.ndarray, quad_b: np.ndarray) -> np.ndarray:
    """:return: (4,) Euclidean per-corner distance; both quads must already be TL TR BR BL.

    :param quad_a: (4, 2) xy, TL TR BR BL
    :param quad_b: (4, 2) xy, TL TR BR BL, same pixel space as ``quad_a``
    """
    return np.linalg.norm(np.asarray(quad_a, dtype=np.float64) - np.asarray(quad_b, dtype=np.float64), axis=1)


def hand_quad_px(frame_rows: pd.DataFrame, width: int, height: int) -> np.ndarray:
    """Rebuild one frame's hand corners in native pixels, ordered TL TR BR BL.

    Normalised xy times the current video extent, so the annotation stays valid
    even if it was clicked on a differently-sized copy of the same video. Rows
    are ordered by ``corner_label``, the annotator's own tl/tr/br/bl call, which
    matches the detector's slot contract by name. Geometric re-sorting is gone
    because it mispairs off-frame corners: an extrapolated corner can sit nearer
    another corner's pixel position than its own (a hard-left camera pan puts BL
    nearer the origin than TL).

    :param frame_rows: the 4 CSV rows for one frame (any row order), with a
        corner_label column holding exactly one each of tl, tr, br, bl
    :param width: native video width in pixels
    :param height: native video height in pixels
    :return: (4, 2) float64 xy in native pixels, ordered TL TR BR BL
    :raises ValueError: no corner_label column, or its 4 values are not exactly
        one each of tl/tr/br/bl (wrong row count, a duplicate, or a gap)
    """
    if "corner_label" not in frame_rows.columns:
        raise ValueError("corners CSV has no corner_label column; annotate with the off-frame tool")
    labels = frame_rows["corner_label"].tolist()
    if sorted(labels) != sorted(CORNER_NAMES):
        raise ValueError(f"expected exactly one row each of {CORNER_NAMES}, got corner_label {sorted(labels)}")
    ordered = frame_rows.set_index("corner_label").loc[list(CORNER_NAMES)]
    xs = ordered["x_norm"].to_numpy(dtype=np.float64) * width
    ys = ordered["y_norm"].to_numpy(dtype=np.float64) * height
    return np.column_stack([xs, ys])


def rows_for_video(annotations: pd.DataFrame, video: Path) -> pd.DataFrame:
    """:return: the annotation rows whose video basename matches ``video``.

    Matching on basename tolerates the annotation CSV storing a different path
    prefix (e.g. annotated on one host, scored on another).

    :param annotations: the full hand-annotation table
    :param video: the video being scored
    """
    names = annotations["video"].map(lambda path: Path(str(path)).name)
    return annotations[names == video.name]


# --- Frame IO (used only by main) ------------------------------------------

def read_frames(video_path: Path, frame_indices: list[int]) -> tuple[dict[int, np.ndarray], int, int]:
    """Read specific frames by index; a failed seek/decode drops that frame.

    :param video_path: the video file
    :param frame_indices: which frames to read
    :return: ({frame_idx: BGR frame}, width, height)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"cannot open {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames: dict[int, np.ndarray] = {}
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames[idx] = frame
    cap.release()
    return frames, width, height


def report(results: pd.DataFrame, video: Path, resize_mode: str) -> None:
    """Print the per-corner and pooled error summary in the ckn_corner_eval voice.

    :param results: the per-frame results table
    :param video: the scored video, for the header line
    :param resize_mode: the detector resize mode, for the header line
    """
    errors = results["err_mean_px"]
    pooled_median = errors.median()
    pooled_p90 = np.percentile(errors, 90)
    pass_rate = results["passed"].mean()
    print(f"\n{video.name} ({resize_mode}): n={len(results)} frames  "
          f"err median {pooled_median:.1f} / p90 {pooled_p90:.1f} px  "
          f"gate pass {pass_rate:.1%}")
    per_corner = "  ".join(
        f"{name.upper()} {results[f'err_{name}_px'].median():.1f}" for name in CORNER_NAMES
    )
    print(f"  per-corner median px: {per_corner}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--annotations-csv", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."), help="Checkout root for the detector import.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device (default cpu).")
    parser.add_argument("--resize-mode", choices=("pad", "squash"), default="pad", help="Detector resize (default pad).")
    parser.add_argument("--batch", type=int, default=10, help="Frames per detector forward pass (default 10).")
    parser.add_argument("--out-csv", type=Path, required=True, help="Per-frame results.")
    args = parser.parse_args()

    if not args.video.exists():
        sys.exit(f"ERROR: --video does not exist: {args.video}")
    if not args.annotations_csv.exists():
        sys.exit(f"ERROR: --annotations-csv does not exist: {args.annotations_csv}")

    # Import inside main so the pure functions above are importable without torch.
    sys.path.insert(0, str(args.repo_root.resolve()))
    from src.courtkeynet.wrapper import CourtKeyNetDetector

    annotations = pd.read_csv(args.annotations_csv)
    video_rows = rows_for_video(annotations, args.video)
    if video_rows.empty:
        sys.exit(f"ERROR: no annotation rows for {args.video.name} in {args.annotations_csv}")
    frame_indices = sorted({int(frame) for frame in video_rows["frame"]})

    frames, width, height = read_frames(args.video, frame_indices)
    if not frames:
        sys.exit(f"ERROR: could not read any of the {len(frame_indices)} annotated frames from {args.video}")

    detector = CourtKeyNetDetector(device=args.device, resize_mode=args.resize_mode)
    order = sorted(frames)
    print(f"scoring {len(order)} frames of {args.video.name} ({width}x{height}, {args.resize_mode})")

    result_rows = []
    for chunk_start in range(0, len(order), args.batch):
        chunk = order[chunk_start : chunk_start + args.batch]
        detections = detector.detect_batch([frames[idx] for idx in chunk])
        for frame_idx, detection in zip(chunk, detections):
            hand = hand_quad_px(video_rows[video_rows["frame"] == frame_idx], width, height)
            err = per_corner_error(detection.corners_px, hand)
            row = {
                "video": args.video.name,
                "frame": frame_idx,
                "err_mean_px": float(err.mean()),
                "err_max_px": float(err.max()),
                "peak_min": float(detection.peak.min()),
                "peak_median": float(np.median(detection.peak)),
                "entropy_max": float(detection.entropy.max()),
                "entropy_median": float(np.median(detection.entropy)),
                "passed": detection.passed,
                "flags": "|".join(detection.flags),
            }
            for corner_i, name in enumerate(CORNER_NAMES):
                row[f"err_{name}_px"] = float(err[corner_i])
            result_rows.append(row)

    results = pd.DataFrame(result_rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv} ({len(results)} rows)")
    report(results, args.video, args.resize_mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
