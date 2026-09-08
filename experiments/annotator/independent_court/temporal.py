"""Small image-space person tracker for the independent court experiment.

The tracker operates on sampled detector output before any court geometry is
fit.  It records only detections that were observed in a sample; gaps remain
gaps in each track.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Any


@dataclass
class Track:
    """A sequence of person detections associated through sampled frames."""

    track_id: int
    sample_indices: list[int]
    detection_indices: list[int]
    feet_px: list[list[float]]
    box_heights: list[float]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_samples(samples: list[dict[str, Any]]) -> None:
    if not isinstance(samples, list):
        raise TypeError("samples must be a list")
    previous_timestamp: float | None = None
    for sample_index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise TypeError(f"sample {sample_index} must be a dictionary")
        required = {"timestamp_seconds", "bboxes", "scores"}
        if not required.issubset(sample):
            raise ValueError(f"sample {sample_index} is missing a required field")
        timestamp = sample["timestamp_seconds"]
        if not _is_number(timestamp):
            raise ValueError(f"sample {sample_index} timestamp_seconds must be finite")
        timestamp = float(timestamp)
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ValueError("sample timestamps must be strictly increasing")
        previous_timestamp = timestamp

        bboxes = sample["bboxes"]
        scores = sample["scores"]
        if not isinstance(bboxes, list) or not isinstance(scores, list):
            raise TypeError(f"sample {sample_index} bboxes and scores must be lists")
        if len(bboxes) != len(scores):
            raise ValueError(f"sample {sample_index} bboxes and scores must have equal lengths")
        for detection_index, (bbox, score) in enumerate(zip(bboxes, scores)):
            if not isinstance(bbox, list) or len(bbox) != 4 or not all(_is_number(value) for value in bbox):
                raise ValueError(f"sample {sample_index} bbox {detection_index} must be a finite xyxy list")
            x1, y1, x2, y2 = (float(value) for value in bbox)
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"sample {sample_index} bbox {detection_index} must have positive size")
            if not _is_number(score):
                raise ValueError(f"sample {sample_index} score {detection_index} must be finite")


def _validate_tracking_settings(score_min: float, max_gap_seconds: float, max_step_heights: float) -> None:
    if not _is_number(score_min):
        raise ValueError("score_min must be finite")
    if not _is_number(max_gap_seconds) or max_gap_seconds < 0:
        raise ValueError("max_gap_seconds must be finite and non-negative")
    if not _is_number(max_step_heights) or max_step_heights < 0:
        raise ValueError("max_step_heights must be finite and non-negative")


def track_people(
    samples: list[dict[str, Any]],
    *,
    score_min: float = 0.2,
    max_gap_seconds: float = 0.5,
    max_step_heights: float = 1.0,
) -> list[Track]:
    """Associate person detections by nearest gated footpoint distance.

    :param samples: Ordered sampled detections with native ``xyxy`` boxes.
    :param score_min: Strict lower bound for a detection score.
    :param max_gap_seconds: Maximum time since a track's last observation.
    :param max_step_heights: Maximum distance as a multiple of mean box height.
    :return: Tracks in creation order, including tracks with one observation.
    """
    _validate_tracking_settings(score_min, max_gap_seconds, max_step_heights)
    _validate_samples(samples)

    tracks: list[Track] = []
    last_seen: dict[int, float] = {}

    for sample_index, sample in enumerate(samples):
        timestamp = float(sample["timestamp_seconds"])
        current_detections: list[tuple[int, list[float], float]] = []
        for detection_index, (bbox, score) in enumerate(zip(sample["bboxes"], sample["scores"])):
            if float(score) <= score_min:
                continue
            x1, y1, x2, y2 = (float(value) for value in bbox)
            current_detections.append((detection_index, [(x1 + x2) / 2, y2], y2 - y1))

        eligible_ids = [
            track.track_id
            for track in tracks
            if timestamp - last_seen[track.track_id] <= max_gap_seconds
        ]
        candidate_matches: list[tuple[float, int, int]] = []
        for track_id in eligible_ids:
            track = tracks[track_id]
            previous_foot = track.feet_px[-1]
            previous_height = track.box_heights[-1]
            for detection_index, foot, height in current_detections:
                distance = math.dist(previous_foot, foot)
                gate = max_step_heights * (previous_height + height) / 2
                if distance <= gate:
                    candidate_matches.append((distance, track_id, detection_index))
        candidate_matches.sort(key=lambda match: (match[0], match[1], match[2]))

        matched_track_ids: set[int] = set()
        matched_detection_indices: set[int] = set()
        detections_by_index = {detection[0]: detection for detection in current_detections}
        for _, track_id, detection_index in candidate_matches:
            if track_id in matched_track_ids or detection_index in matched_detection_indices:
                continue
            _, foot, height = detections_by_index[detection_index]
            track = tracks[track_id]
            track.sample_indices.append(sample_index)
            track.detection_indices.append(detection_index)
            track.feet_px.append(foot.copy())
            track.box_heights.append(height)
            last_seen[track_id] = timestamp
            matched_track_ids.add(track_id)
            matched_detection_indices.add(detection_index)

        for detection_index, foot, height in current_detections:
            if detection_index in matched_detection_indices:
                continue
            track_id = len(tracks)
            track = Track(track_id, [sample_index], [detection_index], [foot.copy()], [height])
            tracks.append(track)
            last_seen[track_id] = timestamp

    return tracks


def summarise_track(track: Track, sample_count: int) -> dict[str, Any]:
    """Return movement and coverage statistics for one track."""
    path_length = sum(
        math.dist(previous, current)
        for previous, current in zip(track.feet_px, track.feet_px[1:])
    )
    displacement = math.dist(track.feet_px[0], track.feet_px[-1])
    median_height = float(median(track.box_heights))
    return {
        "id": track.track_id,
        "observations": len(track.sample_indices),
        "observed_fraction": len(track.sample_indices) / sample_count,
        "path_length_px": path_length,
        "displacement_px": displacement,
        "median_foot_px": [
            float(median(foot[0] for foot in track.feet_px)),
            float(median(foot[1] for foot in track.feet_px)),
        ],
        "median_height_px": median_height,
        "path_length_heights": path_length / median_height,
    }


def pair_presence(first: Track, second: Track, sample_count: int) -> dict[str, Any]:
    """Measure literal two-player and one-or-more-player frame coverage."""
    first_samples = set(first.sample_indices)
    second_samples = set(second.sample_indices)
    two_count = len(first_samples & second_samples)
    one_or_more_count = len(first_samples | second_samples)
    two_fraction = two_count / sample_count
    one_or_more_fraction = one_or_more_count / sample_count
    zero_frames = sample_count - one_or_more_count
    return {
        "two_fraction": two_fraction,
        "one_or_more_fraction": one_or_more_fraction,
        "zero_frames": zero_frames,
        "passes_literal": two_fraction >= 0.5 and zero_frames == 0,
    }
