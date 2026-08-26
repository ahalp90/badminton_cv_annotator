"""Build a tracker-validity trial from existing human hallucination reviews."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .build_trials import (
    CYAN,
    DEVELOPMENT_VIDEOS,
    EVENT_FPS,
    GOLD,
    VIDEO_IDS,
    _draw_ring,
    _event_records,
    _letterbox,
    _load_npy_xz,
    _load_video,
    _open_writer,
    _sha256,
    _verify_clip,
    _write_json,
    balanced_select,
    shift_window,
)
from .score_trials import TRUTH_SCHEMA
from .trial_schema import EXPECTED_FRAMES, HEIGHT, MANIFEST_SCHEMA, WIDTH, load_manifest


def _write_track_clip(
    video: Any,
    centre_frame: int,
    target_start: int,
    target_end: int,
    path: Path,
    *,
    slow_target: bool,
    zoom_target: bool,
    clean_target_replay: bool,
) -> tuple[int, int]:
    context_frames = round(video.fps * 2.0) if slow_target else EXPECTED_FRAMES
    start, end = shift_window(centre_frame, context_frames, len(video.track))
    source_frames = track_source_frames(
        start,
        end,
        target_start,
        target_end,
        slow_target=slow_target,
        clean_target_replay=clean_target_replay,
    )
    capture = cv2.VideoCapture(str(video.source_path))
    writer = _open_writer(path, EVENT_FPS)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"could not open source video {video.source_path}")
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        crop_bounds = (
            tracker_crop_bounds(
                video.track,
                target_start,
                target_end,
                frame_width,
                frame_height,
            )
            if zoom_target
            else None
        )
        for output_frame, source_frame in enumerate(source_frames):
            capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"failed to read source frame {source_frame}")
            target_start_output = 10 if clean_target_replay else EXPECTED_FRAMES // 2
            target_replay = slow_target and output_frame >= target_start_output
            clean_replay = clean_target_replay and 10 <= output_frame < 30
            if target_replay and crop_bounds is not None:
                left, top, right, bottom = crop_bounds
                composed = cv2.resize(
                    frame[top:bottom, left:right],
                    (WIDTH, HEIGHT),
                    interpolation=cv2.INTER_LINEAR,
                )
                scale_x = WIDTH / (right - left)
                scale_y = HEIGHT / (bottom - top)
                offset_x = -left * scale_x
                offset_y = -top * scale_y
            else:
                composed, scale, offset_x, offset_y = _letterbox(
                    frame, WIDTH, HEIGHT
                )
                scale_x = scale
                scale_y = scale
            if video.track[source_frame, 2] > 0 and not clean_replay:
                _draw_ring(
                    composed,
                    float(video.track[source_frame, 0]) * frame_width * scale_x
                    + offset_x,
                    float(video.track[source_frame, 1]) * frame_height * scale_y
                    + offset_y,
                )
            if target_replay or (
                not slow_target and target_start <= source_frame < target_end
            ):
                cv2.rectangle(composed, (2, 2), (WIDTH - 3, HEIGHT - 3), GOLD, 3)
            if clean_replay:
                section = "CLEAN ZOOMED TARGET"
            elif target_replay and zoom_target and clean_target_replay:
                section = "MARKED ZOOMED TARGET"
            elif target_replay and zoom_target:
                section = "ZOOMED SLOW TARGET"
            elif target_replay:
                section = "SLOW TARGET REPLAY"
            else:
                section = "CONTEXT"
            label = (
                f"{section} | NO MARKER"
                if clean_replay
                else f"{section} | CYAN RING = TRACKER CLAIM"
            )
            cv2.putText(
                composed,
                label,
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                CYAN,
                1,
                cv2.LINE_AA,
            )
            writer.write(composed)
    finally:
        writer.release()
        capture.release()
    _verify_clip(path, EVENT_FPS)
    return start, end


def tracker_crop_bounds(
    track: np.ndarray,
    target_start: int,
    target_end: int,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    """Return a fixed three-times crop centred on visible target claims."""
    target = track[target_start:target_end]
    visible = target[:, 2] > 0
    if not visible.any():
        raise ValueError("target interval has no visible tracker claims")
    centre_x = float(np.median(target[visible, 0])) * frame_width
    centre_y = float(np.median(target[visible, 1])) * frame_height
    crop_width = max(1, round(frame_width / 3))
    crop_height = max(1, round(crop_width * HEIGHT / WIDTH))
    if crop_height > frame_height:
        crop_height = max(1, round(frame_height / 3))
        crop_width = max(1, round(crop_height * WIDTH / HEIGHT))
    left = min(max(round(centre_x - crop_width / 2), 0), frame_width - crop_width)
    top = min(max(round(centre_y - crop_height / 2), 0), frame_height - crop_height)
    return left, top, left + crop_width, top + crop_height


def track_source_frames(
    start: int,
    end: int,
    target_start: int,
    target_end: int,
    *,
    slow_target: bool,
    clean_target_replay: bool = False,
) -> list[int]:
    """Return native frames for ordinary context or a half-speed target replay."""
    if not slow_target:
        return list(range(start, end))
    context_count = 10 if clean_target_replay else EXPECTED_FRAMES // 2
    context = np.linspace(start, end, num=context_count, endpoint=False, dtype=int)
    if clean_target_replay:
        target = np.linspace(
            target_start,
            target_end,
            num=20,
            endpoint=False,
            dtype=int,
        )
        return [*map(int, context), *map(int, target), *map(int, target)]
    target = np.linspace(
        target_start,
        target_end,
        num=EXPECTED_FRAMES - context_count,
        endpoint=False,
        dtype=int,
    )
    return [*map(int, context), *map(int, target)]


def _positive_records(videos: Sequence[Any], count: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for video in videos:
        for record in _event_records(video, "filtered_contacts"):
            truth = record["truth"]
            if truth["usable_at_5"] and truth["event_role"] == "later-stroke":
                candidates.append(record)
    return balanced_select(candidates, count)


def build_track_trials(
    artifacts_root: Path,
    repo_root: Path,
    scene_labels_dir: Path,
    review_path: Path,
    output_dir: Path,
    *,
    video_names: tuple[str, ...],
    expected_negative_cases: int,
    positive_cases: int,
    slow_target: bool,
    zoom_target: bool,
    review_track: Path | None,
    clean_target_replay: bool,
) -> None:
    """Render separated human-negative and structural-positive track cases."""
    if zoom_target and not slow_target:
        raise ValueError("zoomed target view requires slow target replay")
    if clean_target_replay and not zoom_target:
        raise ValueError("clean target replay requires the zoomed target view")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    clips_dir = output_dir / "inference" / "clips"
    scoring_dir = output_dir / "scoring"
    clips_dir.mkdir(parents=True)
    scoring_dir.mkdir()
    shots_path = repo_root / "training/data/shuttleset/annotations/shots_master.csv"
    shots = pd.read_csv(shots_path)
    videos = [
        _load_video(artifacts_root, scene_labels_dir, shots, name)
        for name in video_names
    ]
    if review_track is not None:
        if len(videos) != 1:
            raise ValueError("a reviewed track override requires exactly one video")
        reviewed_track = _load_npy_xz(review_track)
        if len(reviewed_track) != len(videos[0].track):
            raise ValueError("reviewed track override has the wrong frame count")
        videos = [replace(videos[0], track=reviewed_track, track_path=review_track)]
    by_name = {video.name: video for video in videos}
    review = pd.read_csv(review_path)
    negatives = review[
        review["fixture"].isin(video_names)
        & (review["human_decision"] == "hallucination")
        & (review["human_confidence"] == "high")
    ]
    if len(negatives) != expected_negative_cases:
        raise ValueError(
            f"expected {expected_negative_cases} hallucinations, found {len(negatives)}"
        )

    records: list[dict[str, Any]] = []
    for row in negatives.itertuples(index=False):
        records.append(
            {
                "video_id": str(row.fixture),
                "centre_frame": (int(row.start_frame) + int(row.stop_frame_exclusive))
                // 2,
                "target_start": int(row.start_frame),
                "target_end": int(row.stop_frame_exclusive),
                "sample_id": str(row.sample_id),
                "tracker_real": False,
                "truth_source": "human-visual-review",
            }
        )
    for record in _positive_records(videos, positive_cases):
        frame = int(record["candidate_frame"])
        records.append(
            {
                "video_id": str(record["video_id"]),
                "centre_frame": frame,
                "target_start": frame - 4,
                "target_end": frame + 5,
                "sample_id": "structural-later-contact",
                "tracker_real": True,
                "truth_source": "shuttleset-near-contact-proxy",
            }
        )
    records.sort(key=lambda row: (str(row["video_id"]), int(row["centre_frame"])))

    manifest_cases: list[dict[str, Any]] = []
    truth_cases: list[dict[str, Any]] = []
    for record in records:
        video = by_name[record["video_id"]]
        centre = int(record["centre_frame"])
        case_id = f"track-{video.name}-f{centre:06d}"
        clip_path = clips_dir / f"{case_id}.mp4"
        start, end = _write_track_clip(
            video,
            centre,
            int(record["target_start"]),
            int(record["target_end"]),
            clip_path,
            slow_target=slow_target,
            zoom_target=zoom_target,
            clean_target_replay=clean_target_replay,
        )
        target = video.track[int(record["target_start"]):int(record["target_end"])]
        manifest_cases.append(
            {
                "case_id": case_id,
                "kind": "track",
                "video_id": video.name,
                "clip_path": str(clip_path.resolve()),
                "source_start_frame": start,
                "source_end_frame": end,
                "candidate_frame": centre,
                "sample_fps": EVENT_FPS,
                "expected_frames": EXPECTED_FRAMES,
                "width": WIDTH,
                "height": HEIGHT,
                "pipeline_priors": {
                    "target_interval_frames": len(target),
                    "tracker_visible_fraction": float((target[:, 2] > 0).mean()),
                    "target_view": (
                        "clean-then-marked-zoom"
                        if clean_target_replay
                        else "tracker-centred-zoom"
                        if zoom_target
                        else "full-frame"
                    ),
                },
            }
        )
        truth_cases.append(
            {
                "case_id": case_id,
                "kind": "track",
                "video_id": video.name,
                "tracker_real": bool(record["tracker_real"]),
                "truth_source": str(record["truth_source"]),
                "sample_id": str(record["sample_id"]),
            }
        )

    manifest_path = output_dir / "inference" / "manifest.json"
    truth_path = scoring_dir / "truth.json"
    _write_json(manifest_path, {"schema": MANIFEST_SCHEMA, "cases": manifest_cases})
    _write_json(truth_path, {"schema": TRUTH_SCHEMA, "cases": truth_cases})
    load_manifest(manifest_path)
    provenance = {
        "schema": "vlm-track-provenance/0.1",
        "settings": {
            "videos": video_names,
            "negative_cases": len(negatives),
            "positive_proxy_cases": positive_cases,
            "frames": EXPECTED_FRAMES,
            "width": WIDTH,
            "height": HEIGHT,
            "slow_target": slow_target,
            "zoom_target": zoom_target,
            "clean_target_replay": clean_target_replay,
        },
        "inputs": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in (
                shots_path,
                review_path,
                *(() if review_track is None else (review_track,)),
            )
        ],
        "outputs": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in sorted(clips_dir.glob("*.mp4"))
        ],
    }
    _write_json(scoring_dir / "provenance.json", provenance)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--scene-labels-dir", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--video",
        action="append",
        choices=tuple(VIDEO_IDS),
        help="video to include; defaults to the two development videos",
    )
    parser.add_argument("--expected-negative-cases", type=int, default=12)
    parser.add_argument("--review-track", type=Path)
    parser.add_argument("--positive-cases", type=int, default=12)
    parser.add_argument("--slow-target", action="store_true")
    parser.add_argument("--zoom-target", action="store_true")
    parser.add_argument("--clean-target-replay", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    build_track_trials(
        args.artifacts_root,
        args.repo_root,
        args.scene_labels_dir,
        args.review,
        args.out,
        video_names=(
            DEVELOPMENT_VIDEOS if args.video is None else tuple(args.video)
        ),
        expected_negative_cases=args.expected_negative_cases,
        positive_cases=args.positive_cases,
        slow_target=args.slow_target,
        zoom_target=args.zoom_target,
        review_track=args.review_track,
        clean_target_replay=args.clean_target_replay,
    )
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
