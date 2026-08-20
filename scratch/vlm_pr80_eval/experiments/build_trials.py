"""Build truth-blind contact and broadcast clips from fixed Issue 103 artefacts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import lzma
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise, product
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from annotator.calibration.scoring import RallyBoundary, classify_all, load_gt_rallies

from .score_trials import TRUTH_SCHEMA
from .trial_schema import EXPECTED_FRAMES, HEIGHT, MANIFEST_SCHEMA, WIDTH, load_manifest

DEVELOPMENT_VIDEOS = ("sset_01", "sset_15")
VIDEO_IDS = {"sset_01": 1, "sset_15": 15, "sset_21": 21}
EVENT_FPS = 25.0
EVENT_TARGET_RADIUS = round(10 * EVENT_FPS / 30)
BROADCAST_FPS = 2.5
EVENT_SECONDS = 2.0
BROADCAST_TARGET_SECONDS = 10.0
BROADCAST_CONTEXT_SECONDS = 20.0
DENSE_BROADCAST_TARGET_SECONDS = 4.0
DENSE_BROADCAST_TARGET_FRAMES = 30
CYAN = (255, 255, 0)
GOLD = (0, 215, 255)


@dataclass(frozen=True)
class VideoData:
    """All inputs used to select and render one development video's cases."""

    name: str
    numeric_id: int
    fps: float
    source_path: Path
    result_path: Path
    track_path: Path
    bboxes_path: Path
    kps_path: Path
    court_present_path: Path
    raw_mask_path: Path
    definitive_mask_path: Path
    scene_labels_path: Path
    result: dict[str, Any]
    track: np.ndarray
    bboxes: np.ndarray
    kps: np.ndarray
    court_present: np.ndarray
    raw_mask: np.ndarray
    definitive_mask: np.ndarray
    scene_labels: pd.DataFrame
    gt_rallies: list[Any]
    frame_sides: dict[int, str]


def shift_window(centre: int, length: int, total_frames: int) -> tuple[int, int]:
    """Return a fixed-length half-open window, shifted rather than shortened at edges."""
    if length < 1 or length > total_frames:
        raise ValueError("window length must fit inside the video")
    start = centre - length // 2
    start = min(max(start, 0), total_frames - length)
    return start, start + length


def distance_stratum(distance_base30: float) -> str:
    """Classify a candidate by the frozen base-30 timing bands."""
    if distance_base30 <= 5:
        return "positive"
    if distance_base30 <= 15:
        return "boundary"
    return "negative"


def _evenly_spaced(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count >= len(rows):
        return rows
    indices = np.linspace(0, len(rows) - 1, num=count)
    chosen = sorted({math.floor(value + 0.5) for value in indices})
    if len(chosen) < count:
        for index in range(len(rows)):
            if index not in chosen:
                chosen.append(index)
            if len(chosen) == count:
                break
    return [rows[index] for index in sorted(chosen[:count])]


def balanced_select(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Select across videos first, then spread cases through each video in time."""
    if count < 1 or count > len(rows):
        raise ValueError("selection count must be positive and available")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["video_id"])].append(row)
    for group in grouped.values():
        group.sort(key=lambda row: int(row["sort_frame"]))

    names = sorted(grouped)
    quotas = {name: count // len(names) for name in names}
    for name in names[: count % len(names)]:
        quotas[name] += 1
    selected: list[dict[str, Any]] = []
    shortfall = 0
    for name in names:
        available = grouped[name]
        take = min(quotas[name], len(available))
        selected.extend(_evenly_spaced(available, take))
        shortfall += quotas[name] - take
    if shortfall:
        selected_ids = {id(row) for row in selected}
        remainder = [row for row in rows if id(row) not in selected_ids]
        remainder.sort(key=lambda row: (str(row["video_id"]), int(row["sort_frame"])))
        selected.extend(_evenly_spaced(remainder, shortfall))
    return sorted(
        selected, key=lambda row: (str(row["video_id"]), int(row["sort_frame"]))
    )


def _load_json_gz(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return payload


def _load_npy_xz(path: Path) -> np.ndarray:
    with lzma.open(path, "rb") as stream:
        return np.load(stream)


def _single_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"expected one {pattern!r} under {directory}, found {len(matches)}"
        )
    return matches[0]


def _video_paths(
    artifacts_root: Path, scene_labels_dir: Path, name: str
) -> dict[str, Path]:
    stages = artifacts_root / "stages"
    return {
        "source": _single_file(stages / "tracknet_input" / name, "*.avi"),
        "result": stages / "annotation" / name / "annotator_result.json.gz",
        "track": stages / "shuttle" / name / "shuttle_track.npy.xz",
        "bboxes": stages / "pose" / name / "pose_bboxes.npy.xz",
        "kps": stages / "pose" / name / "pose_kps.npy.xz",
        "court": stages / "court" / name / "court_present.npy.xz",
        "raw_mask": stages / "annotation" / name / "raw_replay_mask.npy.xz",
        "definitive_mask": stages
        / "annotation"
        / name
        / "definitive_exclusion_mask.npy.xz",
        "scene": scene_labels_dir / f"{name}_broadcast_timeline_labels.csv.gz",
    }


def _load_video(
    artifacts_root: Path,
    scene_labels_dir: Path,
    shots_master: pd.DataFrame,
    name: str,
) -> VideoData:
    paths = _video_paths(artifacts_root, scene_labels_dir, name)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    payload = _load_json_gz(paths["result"])
    result = payload["result"]
    track = _load_npy_xz(paths["track"])
    bboxes = _load_npy_xz(paths["bboxes"])
    kps = _load_npy_xz(paths["kps"])
    court_present = _load_npy_xz(paths["court"])
    raw_mask = _load_npy_xz(paths["raw_mask"])
    definitive_mask = _load_npy_xz(paths["definitive_mask"])
    lengths = {
        len(track),
        len(bboxes),
        len(kps),
        len(court_present),
        len(raw_mask),
        len(definitive_mask),
    }
    if len(lengths) != 1:
        raise ValueError(
            f"{name}: frame-aligned inputs have different lengths: {sorted(lengths)}"
        )
    fps = float(pd.read_csv(paths["scene"])["fps"].iloc[0])
    if fps not in {25.0, 30.0}:
        raise ValueError(f"{name}: trial renderer supports 25 or 30 FPS, got {fps}")
    capture = cv2.VideoCapture(str(paths["source"]))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"could not open source video {paths['source']}")
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        source_frames = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if abs(source_fps - fps) > 0.01:
        raise ValueError(
            f"{name}: source video is {source_fps} FPS but labels are {fps} FPS"
        )
    aligned_frames = next(iter(lengths))
    if source_frames != aligned_frames:
        raise ValueError(
            f"{name}: source video has {source_frames} frames but arrays have {aligned_frames}"
        )
    numeric_id = VIDEO_IDS[name]
    per_video = shots_master[shots_master["vid"] == numeric_id]
    frame_sides = {
        int(frame): str(side)
        for frame, side in zip(per_video["frame_num"], per_video["player_side"])
    }
    return VideoData(
        name=name,
        numeric_id=numeric_id,
        fps=fps,
        source_path=paths["source"],
        result_path=paths["result"],
        track_path=paths["track"],
        bboxes_path=paths["bboxes"],
        kps_path=paths["kps"],
        court_present_path=paths["court"],
        raw_mask_path=paths["raw_mask"],
        definitive_mask_path=paths["definitive_mask"],
        scene_labels_path=paths["scene"],
        result=result,
        track=track,
        bboxes=bboxes,
        kps=kps,
        court_present=court_present,
        raw_mask=raw_mask,
        definitive_mask=definitive_mask,
        scene_labels=pd.read_csv(paths["scene"]),
        gt_rallies=load_gt_rallies(shots_master, numeric_id),
        frame_sides=frame_sides,
    )


def _covered_gt_by_span(
    video: VideoData,
) -> tuple[list[tuple[RallyBoundary, int | None]], dict[int, list[int]]]:
    spans = [tuple(span) for span in video.result["spans"]]
    classifications = classify_all(spans, video.gt_rallies)
    by_span: dict[int, list[int]] = defaultdict(list)
    for gt_index, (category, span_index) in enumerate(classifications):
        if category is RallyBoundary.COVERED and span_index is not None:
            by_span[span_index].append(gt_index)
    return classifications, by_span


def _neighbour_seconds(
    frames: list[int], index: int, fps: float
) -> tuple[float | None, float | None]:
    previous = None if index == 0 else (frames[index] - frames[index - 1]) / fps
    following = (
        None if index + 1 == len(frames) else (frames[index + 1] - frames[index]) / fps
    )
    return previous, following


def _event_records(video: VideoData, event_source: str) -> list[dict[str, Any]]:
    _, gt_by_span = _covered_gt_by_span(video)
    contacts_by_span: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for contact in video.result[event_source]:
        contacts_by_span[int(contact["rally_id"])].append(contact)
    records: list[dict[str, Any]] = []
    for span_index, contacts in contacts_by_span.items():
        span = tuple(map(int, video.result["spans"][span_index]))
        if (
            len(gt_by_span.get(span_index, [])) != 1
            or len(_overlapping_gt_indices(video, span)) != 1
        ):
            continue
        gt = video.gt_rallies[gt_by_span[span_index][0]]
        contacts.sort(key=lambda row: int(row["contact_frame"]))
        frames = [int(row["contact_frame"]) for row in contacts]
        for index, contact in enumerate(contacts):
            frame = int(contact["contact_frame"])
            nearest_frame = min(
                gt.stroke_frames, key=lambda value: (abs(value - frame), value)
            )
            distance_base30 = abs(nearest_frame - frame) * 30.0 / video.fps
            previous, following = _neighbour_seconds(frames, index, video.fps)
            side = video.frame_sides[nearest_frame]
            ball_round = gt.stroke_frames.index(nearest_frame) + 1
            expected_actor = None
            if distance_base30 <= 15:
                expected_actor = "top" if side == "Top" else "bottom"
            records.append(
                {
                    "video_id": video.name,
                    "sort_frame": frame,
                    "span_index": span_index,
                    "candidate_frame": frame,
                    "stratum": distance_stratum(distance_base30),
                    "priors": {
                        "wrist_near": contact["wrist_near"],
                        "proximity_ok": contact["proximity_ok"],
                        "suppressed": contact["suppressed"],
                        "court_present": bool(video.court_present[frame]),
                        "raw_masked": bool(video.raw_mask[frame]),
                        "definitive_masked": bool(video.definitive_mask[frame]),
                        "track_visible": bool(video.track[frame, 2] > 0),
                        "seconds_from_previous_raw_candidate": previous,
                        "seconds_to_next_raw_candidate": following,
                    },
                    "truth": {
                        "stratum": distance_stratum(distance_base30),
                        "nearest_gt_frame": int(nearest_frame),
                        "distance_to_gt_base30": distance_base30,
                        "usable_at_5": distance_base30 <= 5,
                        "usable_at_10": distance_base30 <= 10,
                        "usable_at_15": distance_base30 <= 15,
                        "expected_actor": expected_actor,
                        "set_id": gt.set_id,
                        "rally": gt.rally,
                        "ball_round": ball_round,
                        "event_role": "serve" if ball_round == 1 else "later-stroke",
                    },
                }
            )
    return records


def _scene_fractions(labels: pd.DataFrame, start: int, end: int) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for row in labels.itertuples(index=False):
        overlap = max(
            0, min(end, int(row.end_frame)) - max(start, int(row.start_frame))
        )
        if overlap:
            counts[str(row.truth)] += overlap
    length = end - start
    if sum(counts.values()) != length:
        raise ValueError(f"scene labels cover {sum(counts.values())}/{length} frames")
    return {name: counts[name] / length for name in sorted(counts)}


def _overlapping_gt_indices(video: VideoData, span: tuple[int, int]) -> list[int]:
    start, end = span
    return [
        index
        for index, rally in enumerate(video.gt_rallies)
        if rally.extent[0] < end and rally.extent[1] >= start
    ]


def broadcast_control_kind(
    *, one_to_one: bool, live_fraction: float, replay_cutaway_fraction: float
) -> str | None:
    """Keep only controls whose visual scene truth answers the prompt unambiguously."""
    if one_to_one and live_fraction >= 0.8:
        return "positive"
    if replay_cutaway_fraction >= 0.6:
        return "negative"
    return None


def strict_broadcast_control_kind(
    *, one_to_one: bool, scene_fractions: dict[str, float]
) -> str | None:
    """Keep only pure live or pure replay/cutaway controls."""
    if one_to_one and scene_fractions == {"live": 1.0}:
        return "positive"
    if scene_fractions in ({"replay": 1.0}, {"cutaway": 1.0}):
        return "negative"
    return None


def _broadcast_records(
    video: VideoData,
    *,
    dense_target: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    spans = [tuple(span) for span in video.result["spans"]]
    _, gt_by_span = _covered_gt_by_span(video)
    raw_contacts = Counter(int(row["rally_id"]) for row in video.result["contacts"])
    filtered_contacts = Counter(
        int(row["rally_id"]) for row in video.result["filtered_contacts"]
    )
    total_frames = len(video.track)
    target_seconds = (
        DENSE_BROADCAST_TARGET_SECONDS
        if dense_target
        else BROADCAST_TARGET_SECONDS
    )
    target_length = round(video.fps * target_seconds)
    context_length = int(video.fps * BROADCAST_CONTEXT_SECONDS)
    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    for span_index, span in enumerate(spans):
        midpoint = (span[0] + span[1]) // 2
        target_start, target_end = shift_window(midpoint, target_length, total_frames)
        context_start, context_end = shift_window(
            midpoint, context_length, total_frames
        )
        scene_fractions = _scene_fractions(video.scene_labels, target_start, target_end)
        live_fraction = scene_fractions.get("live", 0.0) + scene_fractions.get(
            "live-non-standard", 0.0
        )
        dominant_scene = max(scene_fractions, key=scene_fractions.get)
        overlapping = _overlapping_gt_indices(video, span)
        one_to_one = len(gt_by_span.get(span_index, [])) == 1 and len(overlapping) == 1
        if not overlapping:
            boundary_class = "spurious"
        elif len(overlapping) > 1:
            boundary_class = "merged"
        elif one_to_one:
            boundary_class = "one-to-one"
        else:
            boundary_class = "partial"
        replay_cutaway_fraction = scene_fractions.get(
            "replay", 0.0
        ) + scene_fractions.get("cutaway", 0.0)
        control_kind = (
            strict_broadcast_control_kind(
                one_to_one=one_to_one,
                scene_fractions=scene_fractions,
            )
            if dense_target
            else broadcast_control_kind(
                one_to_one=one_to_one,
                live_fraction=live_fraction,
                replay_cutaway_fraction=replay_cutaway_fraction,
            )
        )
        if control_kind is None:
            continue
        valid_rally = control_kind == "positive"
        mapped_gt = video.gt_rallies[gt_by_span[span_index][0]] if one_to_one else None
        record = {
            "video_id": video.name,
            "sort_frame": midpoint,
            "span_index": span_index,
            "source_start_frame": context_start,
            "source_end_frame": context_end,
            "target_start_frame": target_start,
            "target_end_frame": target_end,
            "priors": {
                "span_duration_seconds": (span[1] - span[0]) / video.fps,
                "raw_contact_count": raw_contacts[span_index],
                "filtered_contact_count": filtered_contacts[span_index],
                "court_present_fraction": float(
                    np.mean(video.court_present[target_start:target_end])
                ),
                "raw_mask_fraction": float(
                    np.mean(video.raw_mask[target_start:target_end])
                ),
                "definitive_mask_fraction": float(
                    np.mean(video.definitive_mask[target_start:target_end])
                ),
                "track_visible_fraction": float(
                    np.mean(video.track[target_start:target_end, 2] > 0)
                ),
                "sampling_layout": (
                    "dense-four-second-target"
                    if dense_target
                    else "uniform-twenty-second"
                ),
            },
            "truth": {
                "valid_rally": valid_rally,
                "dominant_scene_truth": dominant_scene,
                "scene_fractions": scene_fractions,
                "boundary_class": boundary_class,
                "mapped_set_id": None if mapped_gt is None else mapped_gt.set_id,
                "mapped_rally": None if mapped_gt is None else mapped_gt.rally,
            },
        }
        (positives if valid_rally else negatives).append(record)
    return positives, negatives


def _letterbox(
    frame: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, float, int, int]:
    source_height, source_width = frame.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    resized = cv2.resize(
        frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    offset_x = (width - resized_width) // 2
    offset_y = (height - resized_height) // 2
    canvas[
        offset_y : offset_y + resized_height, offset_x : offset_x + resized_width
    ] = resized
    return canvas, scale, offset_x, offset_y


def _draw_ring(image: np.ndarray, x: float, y: float) -> None:
    if math.isfinite(x) and math.isfinite(y):
        cv2.circle(image, (round(x), round(y)), 8, CYAN, 2, lineType=cv2.LINE_AA)


def event_frame_in_target(source_frame: int, candidate_frame: int) -> bool:
    """Return whether one 25 FPS frame is inside the ±10 base-30 target."""
    return abs(source_frame - candidate_frame) <= EVENT_TARGET_RADIUS


def _open_writer(path: Path, fps: float) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {path}")
    return writer


def _verify_clip(path: Path, fps: float) -> None:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"could not reopen clip {path}")
        observed = (
            int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        if observed != (EXPECTED_FRAMES, WIDTH, HEIGHT):
            raise ValueError(f"{path}: clip geometry/count is {observed}")
        if abs(capture.get(cv2.CAP_PROP_FPS) - fps) > 0.01:
            raise ValueError(f"{path}: clip FPS differs from {fps}")
    finally:
        capture.release()


def _write_event_clip(
    video: VideoData, candidate_frame: int, path: Path
) -> tuple[int, int]:
    start, end = shift_window(candidate_frame, EXPECTED_FRAMES, len(video.track))
    capture = cv2.VideoCapture(str(video.source_path))
    writer = _open_writer(path, EVENT_FPS)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"could not open source video {video.source_path}")
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        for source_frame in range(start, end):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"failed to read source frame {source_frame}")
            composed, scale, offset_x, offset_y = _letterbox(frame, WIDTH, HEIGHT)
            cv2.putText(
                composed,
                "TOP / FAR",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                GOLD,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                composed,
                "BOTTOM / NEAR",
                (8, HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                GOLD,
                1,
                cv2.LINE_AA,
            )
            shuttle_x = float(video.track[source_frame, 0]) * frame_width
            shuttle_y = float(video.track[source_frame, 1]) * frame_height
            if video.track[source_frame, 2] > 0:
                _draw_ring(
                    composed,
                    shuttle_x * scale + offset_x,
                    shuttle_y * scale + offset_y,
                )
            if event_frame_in_target(source_frame, candidate_frame):
                cv2.rectangle(composed, (2, 2), (WIDTH - 3, HEIGHT - 3), GOLD, 3)
            writer.write(composed)
    finally:
        writer.release()
        capture.release()
    _verify_clip(path, EVENT_FPS)
    return start, end


def broadcast_source_frames(
    start: int,
    end: int,
    target_start: int,
    target_end: int,
) -> tuple[list[int], range]:
    """Sample sparse context around a dense four-second target."""
    if not start <= target_start < target_end <= end:
        raise ValueError("broadcast target must sit inside its context")
    if target_end - target_start < DENSE_BROADCAST_TARGET_FRAMES:
        raise ValueError("broadcast target is too short for unique dense samples")
    before_count = 10
    after_count = 10
    if target_start == start:
        before_count = 0
        after_count = 20
    elif target_end == end:
        before_count = 20
        after_count = 0

    def sample(interval_start: int, interval_end: int, count: int) -> list[int]:
        if count == 0:
            return []
        if interval_end - interval_start < count:
            raise ValueError("broadcast context is too short for unique samples")
        return [
            int(frame)
            for frame in np.linspace(
                interval_start,
                interval_end,
                num=count,
                endpoint=False,
                dtype=int,
            )
        ]

    before = sample(start, target_start, before_count)
    target = sample(
        target_start,
        target_end,
        DENSE_BROADCAST_TARGET_FRAMES,
    )
    after = sample(target_end, end, after_count)
    source_frames = before + target + after
    if len(source_frames) != EXPECTED_FRAMES:
        raise AssertionError("dense broadcast sampler did not produce 50 frames")
    if source_frames != sorted(set(source_frames)):
        raise ValueError("dense broadcast source frames must be unique and ordered")
    target_outputs = range(len(before), len(before) + len(target))
    return source_frames, target_outputs


def _write_broadcast_clip(
    video: VideoData,
    start: int,
    end: int,
    target_start: int,
    target_end: int,
    path: Path,
    *,
    dense_target: bool,
) -> list[int]:
    if dense_target:
        source_frames, target_outputs = broadcast_source_frames(
            start, end, target_start, target_end
        )
    else:
        source_frames = [
            int(frame)
            for frame in np.linspace(
                start, end, num=EXPECTED_FRAMES, endpoint=False, dtype=int
            )
        ]
        target_outputs = range(0)
    capture = cv2.VideoCapture(str(video.source_path))
    writer = _open_writer(path, BROADCAST_FPS)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"could not open source video {video.source_path}")
        for output_frame, source_frame in enumerate(source_frames):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(source_frame))
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"failed to read source frame {source_frame}")
            composed, _, _, _ = _letterbox(frame, WIDTH, HEIGHT)
            if output_frame in target_outputs or (
                not dense_target and target_start <= source_frame < target_end
            ):
                cv2.rectangle(composed, (2, 2), (WIDTH - 3, HEIGHT - 3), GOLD, 3)
            writer.write(composed)
    finally:
        writer.release()
        capture.release()
    _verify_clip(path, BROADCAST_FPS)
    return source_frames


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _select_event_records(
    videos: list[VideoData],
    count: int,
    event_source: str,
    event_spans: set[tuple[str, int]] | None = None,
) -> list[dict[str, Any]]:
    if count == 0 and event_spans is None:
        return []
    all_records = [
        record for video in videos for record in _event_records(video, event_source)
    ]
    if event_spans is not None:
        selected = [
            record
            for record in all_records
            if (str(record["video_id"]), int(record["span_index"])) in event_spans
        ]
        observed = {
            (str(record["video_id"]), int(record["span_index"]))
            for record in selected
        }
        missing = event_spans - observed
        if missing:
            raise ValueError(f"event spans have no eligible candidates: {sorted(missing)}")
        return sorted(
            selected,
            key=lambda record: (str(record["video_id"]), int(record["sort_frame"])),
        )
    if count % 3:
        raise ValueError("event case count must be divisible by three strata")
    per_stratum = count // 3
    selected: list[dict[str, Any]] = []
    for stratum in ("positive", "negative", "boundary"):
        candidates = [record for record in all_records if record["stratum"] == stratum]
        selected.extend(balanced_select(candidates, per_stratum))
    return selected


def _select_broadcast_records(
    videos: list[VideoData], count: int, *, dense_target: bool
) -> list[dict[str, Any]]:
    if count == 0:
        return []
    if count % 2:
        raise ValueError("broadcast case count must be even")
    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    for video in videos:
        video_positive, video_negative = _broadcast_records(
            video, dense_target=dense_target
        )
        positives.extend(video_positive)
        negatives.extend(video_negative)
    selected_positives = balanced_select(positives, count // 2)
    if dense_target:
        selected_negatives: list[dict[str, Any]] = []
        negative_count = count // 2
        scenes = ("replay", "cutaway")
        groups = [(video.name, scene) for video in videos for scene in scenes]
        available = Counter(
            (
                str(record["video_id"]),
                str(record["truth"]["dominant_scene_truth"]),
            )
            for record in negatives
        )
        allocations = [
            counts
            for counts in product(
                *[
                    range(min(available[group], negative_count) + 1)
                    for group in groups
                ]
            )
            if sum(counts) == negative_count
        ]
        if not allocations:
            raise ValueError(f"too few pure broadcast negatives: {available}")

        def allocation_cost(counts: tuple[int, ...]) -> tuple[Any, ...]:
            chosen = dict(zip(groups, counts))
            video_totals = [
                sum(chosen[(video.name, scene)] for scene in scenes)
                for video in videos
            ]
            scene_totals = [
                sum(chosen[(video.name, scene)] for video in videos)
                for scene in scenes
            ]
            return (
                max(video_totals) - min(video_totals),
                max(scene_totals) - min(scene_totals),
                counts,
            )

        chosen_counts = min(
            allocations,
            key=allocation_cost,
        )
        for (video_name, scene), scene_count in zip(groups, chosen_counts):
            candidates = [
                record
                for record in negatives
                if record["video_id"] == video_name
                and record["truth"]["dominant_scene_truth"] == scene
            ]
            selected_negatives.extend(_evenly_spaced(candidates, scene_count))
    else:
        selected_negatives = balanced_select(negatives, count // 2)
    selected = selected_positives + selected_negatives
    for video in videos:
        intervals = sorted(
            (
                int(record["target_start_frame"]),
                int(record["target_end_frame"]),
            )
            for record in selected
            if record["video_id"] == video.name
        )
        if any(
            left_end > right_start
            for (_, left_end), (right_start, _) in pairwise(intervals)
        ):
            raise ValueError(f"selected broadcast targets overlap in {video.name}")
    return selected


def build_trials(
    artifacts_root: Path,
    repo_root: Path,
    scene_labels_dir: Path,
    output_dir: Path,
    *,
    event_cases: int,
    broadcast_cases: int,
    event_source: str,
    event_spans: set[tuple[str, int]] | None = None,
    dense_broadcast_target: bool = False,
) -> None:
    """Select cases, render clips, and write separated inference and truth records."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    inference_dir = output_dir / "inference"
    scoring_dir = output_dir / "scoring"
    clips_dir = inference_dir / "clips"
    clips_dir.mkdir(parents=True)
    scoring_dir.mkdir()
    shots_master_path = (
        repo_root / "training/data/shuttleset/annotations/shots_master.csv"
    )
    shots_master = pd.read_csv(shots_master_path)
    videos = [
        _load_video(artifacts_root, scene_labels_dir, shots_master, name)
        for name in DEVELOPMENT_VIDEOS
    ]
    by_name = {video.name: video for video in videos}

    manifest_cases: list[dict[str, Any]] = []
    truth_cases: list[dict[str, Any]] = []
    for record in _select_event_records(
        videos, event_cases, event_source, event_spans
    ):
        video = by_name[record["video_id"]]
        candidate_frame = int(record["candidate_frame"])
        case_id = f"event-{video.name}-f{candidate_frame:06d}"
        clip_path = clips_dir / f"{case_id}.mp4"
        start, end = _write_event_clip(video, candidate_frame, clip_path)
        manifest_cases.append(
            {
                "case_id": case_id,
                "kind": "event",
                "video_id": video.name,
                "clip_path": str(clip_path.resolve()),
                "source_start_frame": start,
                "source_end_frame": end,
                "candidate_frame": candidate_frame,
                "sample_fps": EVENT_FPS,
                "expected_frames": EXPECTED_FRAMES,
                "width": WIDTH,
                "height": HEIGHT,
                "pipeline_priors": record["priors"],
            }
        )
        truth_cases.append(
            {
                "case_id": case_id,
                "kind": "event",
                "video_id": video.name,
                **record["truth"],
            }
        )

    broadcast_frame_maps: dict[str, list[int]] = {}
    for record in _select_broadcast_records(
        videos, broadcast_cases, dense_target=dense_broadcast_target
    ):
        video = by_name[record["video_id"]]
        span_index = int(record["span_index"])
        case_id = f"broadcast-{video.name}-r{span_index:03d}"
        clip_path = clips_dir / f"{case_id}.mp4"
        start = int(record["source_start_frame"])
        end = int(record["source_end_frame"])
        target_start = int(record["target_start_frame"])
        target_end = int(record["target_end_frame"])
        broadcast_frame_maps[case_id] = _write_broadcast_clip(
            video,
            start,
            end,
            target_start,
            target_end,
            clip_path,
            dense_target=dense_broadcast_target,
        )
        manifest_cases.append(
            {
                "case_id": case_id,
                "kind": "broadcast",
                "video_id": video.name,
                "clip_path": str(clip_path.resolve()),
                "source_start_frame": start,
                "source_end_frame": end,
                "candidate_frame": None,
                "sample_fps": BROADCAST_FPS,
                "expected_frames": EXPECTED_FRAMES,
                "width": WIDTH,
                "height": HEIGHT,
                "pipeline_priors": record["priors"],
            }
        )
        truth_cases.append(
            {
                "case_id": case_id,
                "kind": "broadcast",
                "video_id": video.name,
                **record["truth"],
            }
        )

    manifest = {"schema": MANIFEST_SCHEMA, "cases": manifest_cases}
    truth = {"schema": TRUTH_SCHEMA, "cases": truth_cases}
    _write_json(inference_dir / "manifest.json", manifest)
    _write_json(scoring_dir / "truth.json", truth)
    load_manifest(inference_dir / "manifest.json")

    input_paths = [shots_master_path]
    for video in videos:
        input_paths.extend(
            (
                video.source_path,
                video.result_path,
                video.track_path,
                video.bboxes_path,
                video.kps_path,
                video.court_present_path,
                video.raw_mask_path,
                video.definitive_mask_path,
                video.scene_labels_path,
            )
        )
    provenance = {
        "schema": "vlm-cleanup-provenance/0.1",
        "settings": {
            "development_videos": DEVELOPMENT_VIDEOS,
            "event_cases": event_cases,
            "event_source": event_source,
            "event_spans": None
            if event_spans is None
            else [f"{video_id}:{span}" for video_id, span in sorted(event_spans)],
            "broadcast_cases": broadcast_cases,
            "event_fps": EVENT_FPS,
            "broadcast_fps": BROADCAST_FPS,
            "broadcast_target_seconds": BROADCAST_TARGET_SECONDS,
            "broadcast_context_seconds": BROADCAST_CONTEXT_SECONDS,
            "dense_broadcast_target": dense_broadcast_target,
            "expected_frames": EXPECTED_FRAMES,
            "width": WIDTH,
            "height": HEIGHT,
        },
        "inputs": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in input_paths
        ],
        "outputs": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(clips_dir.glob("*.mp4"))
        ],
        "broadcast_frame_maps": broadcast_frame_maps,
        "software": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(Path(__file__).parent.iterdir())
            if path.suffix in {".py", ".sh"}
        ],
    }
    _write_json(scoring_dir / "provenance.json", provenance)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--scene-labels-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--event-cases", type=int, default=24)
    parser.add_argument(
        "--event-source",
        choices=("contacts", "filtered_contacts"),
        default="contacts",
    )
    parser.add_argument("--broadcast-cases", type=int, default=12)
    parser.add_argument("--dense-broadcast-target", action="store_true")
    parser.add_argument(
        "--event-span",
        action="append",
        metavar="VIDEO:SPAN",
        help="include every eligible event candidate in this span; repeat as needed",
    )
    return parser


def parse_event_span(value: str) -> tuple[str, int]:
    """Parse one explicit video and natural span index."""
    try:
        video_id, raw_span = value.rsplit(":", maxsplit=1)
        span = int(raw_span)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("event span must be VIDEO:SPAN") from exc
    if video_id not in DEVELOPMENT_VIDEOS or span < 0:
        raise argparse.ArgumentTypeError("event span must name a development video and non-negative index")
    return video_id, span


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    event_spans = (
        None
        if args.event_span is None
        else {parse_event_span(value) for value in args.event_span}
    )
    build_trials(
        args.artifacts_root,
        args.repo_root,
        args.scene_labels_dir,
        args.out,
        event_cases=args.event_cases,
        broadcast_cases=args.broadcast_cases,
        event_source=args.event_source,
        event_spans=event_spans,
        dense_broadcast_target=args.dense_broadcast_target,
    )
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
