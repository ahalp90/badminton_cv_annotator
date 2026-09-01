"""Load the frozen 47-video predictions without opening any label file."""

from __future__ import annotations

import gzip
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from scratch.contact_det.scripts.score_contact_rallies import (
    FixedEvent,
    FixedSpan,
    fixed_spans_from_evidence,
)
from scratch.contact_det_full_ds_fit.scripts.experiment_config import (
    VideoSpec,
    load_development_split,
)
from scratch.contact_det_full_ds_fit.scripts.inpaint_shuttleset22_tracks import (
    VIDEO_IDS,
)
from scratch.contact_det_full_ds_fit.scripts.score_training_videos import (
    load_score_groups,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PREDICTIONS = (
    REPO_ROOT
    / "scratch/contact_det_full_ds_fit/raw/shuttleset22-test-predictions/combined_predictions.json.gz"
)
PREDICTION_SCHEMA = "shuttleset22-contact-predictions-combined/1"
DEVELOPMENT_SPLIT = REPO_ROOT / "scratch/contact_det_full_ds_fit/records/shuttleset_development_split.json"
SCORE_GROUPS = REPO_ROOT / "scratch/contact_det_full_ds_fit/records/training_video_score_groups.json"
TRAINING_PREDICTIONS = (
    REPO_ROOT
    / "scratch/contact_det_full_ds_fit/raw/training_rally_start_inputs/rally_start_training_inputs.json.gz"
)
VALIDATION_PREDICTIONS = (
    REPO_ROOT
    / "scratch/contact_det_full_ds_fit/raw/validation_rally_start_inputs/rally_start_validation_inputs.json.gz"
)


@dataclass(frozen=True)
class FrozenPredictionPack:
    """Saved prediction rows needed by the follow-up scorers."""

    path: Path
    source_commit: str
    payload: Mapping[str, Any]
    videos: tuple[Mapping[str, Any], ...]
    events_by_fixture: Mapping[str, tuple[FixedEvent, ...]]
    spans: tuple[FixedSpan, ...]


@dataclass(frozen=True)
class DevelopmentPredictionPack:
    """Out-of-fold development predictions from all five video groups."""

    paths: tuple[Path, Path]
    videos: tuple[VideoSpec, ...]
    group_by_fixture: Mapping[str, str]
    events_by_fixture: Mapping[str, tuple[FixedEvent, ...]]
    spans: tuple[FixedSpan, ...]


def read_json(path: Path) -> dict[str, Any]:
    """Read a plain or gzip-compressed JSON object."""
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def normalise_side(value: object, fixture: str, frame: int) -> str | None:
    if value is None:
        return None
    if value not in {"Top", "Bot"}:
        raise ValueError(f"{fixture}/{frame}: unknown player side {value!r}")
    return str(value)


def _events(fixture: str, raw_contacts: object) -> tuple[FixedEvent, ...]:
    if not isinstance(raw_contacts, list):
        raise TypeError(f"{fixture}: contacts must be a list")
    previous_frame = -1
    events: list[FixedEvent] = []
    for raw_contact in raw_contacts:
        if not isinstance(raw_contact, dict):
            raise TypeError(f"{fixture}: each contact must be an object")
        frame = int(raw_contact["frame"])
        if frame <= previous_frame:
            raise ValueError(f"{fixture}: contacts are not in frame order")
        previous_frame = frame
        score = float(raw_contact["contact_score"])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"{fixture}/{frame}: contact score is outside zero to one")
        events.append(
            FixedEvent(
                fixture=fixture,
                frame=frame,
                timing_score=score,
                predicted_side=normalise_side(
                    raw_contact.get("predicted_side"), fixture, frame
                ),
            )
        )
    return tuple(events)


def load_frozen_test_predictions(
    path: Path = DEFAULT_PREDICTIONS,
) -> FrozenPredictionPack:
    """Load the saved test predictions and rebuild their fixed event streams."""
    payload = read_json(path)
    if payload.get("schema") != PREDICTION_SCHEMA or payload.get("status") != "complete":
        raise ValueError("Frozen prediction record is incomplete or has another schema")
    raw_videos = payload.get("videos")
    video_ids = payload.get("video_ids")
    if not isinstance(raw_videos, list) or not isinstance(video_ids, list):
        raise TypeError("Frozen prediction videos and video IDs must be lists")
    if tuple(video_ids) != VIDEO_IDS or len(raw_videos) != len(VIDEO_IDS):
        raise ValueError("Frozen predictions do not cover the fixed 47-video test set")

    videos: list[Mapping[str, Any]] = []
    events_by_fixture: dict[str, tuple[FixedEvent, ...]] = {}
    evidence_fixtures: list[dict[str, object]] = []
    for expected_video_id, raw_video in zip(VIDEO_IDS, raw_videos, strict=True):
        if not isinstance(raw_video, dict):
            raise TypeError("Each frozen prediction video must be an object")
        fixture = str(raw_video["fixture"])
        if raw_video.get("video_id") != expected_video_id or fixture != str(expected_video_id):
            raise ValueError(f"Video {expected_video_id}: saved identity differs")
        if fixture in events_by_fixture:
            raise ValueError(f"Duplicate frozen prediction fixture {fixture}")
        raw_contacts = raw_video.get("contacts")
        raw_spans = raw_video.get("spans")
        if not isinstance(raw_contacts, list) or not isinstance(raw_spans, list):
            raise TypeError(f"{fixture}: contacts and spans must be lists")

        videos.append(raw_video)
        events_by_fixture[fixture] = _events(fixture, raw_contacts)
        evidence_fixtures.append({"fixture": fixture, "spans": raw_spans})

    spans = fixed_spans_from_evidence(
        {"fixtures": evidence_fixtures},
        events_by_fixture,
    )
    source_commit = payload.get("source_commit")
    if not isinstance(source_commit, str):
        raise TypeError("Frozen prediction source commit must be a string")
    return FrozenPredictionPack(
        path=path,
        source_commit=source_commit,
        payload=MappingProxyType(payload),
        videos=tuple(videos),
        events_by_fixture=MappingProxyType(events_by_fixture),
        spans=spans,
    )


def load_development_predictions(
    training_path: Path = TRAINING_PREDICTIONS,
    validation_path: Path = VALIDATION_PREDICTIONS,
) -> DevelopmentPredictionPack:
    """Load the 40 predictions made by contact models that did not train on each video."""
    split = load_development_split(DEVELOPMENT_SPLIT)
    score_groups = load_score_groups(SCORE_GROUPS, split)
    expected = {video.fixture: video for video in split.videos}
    training = read_json(training_path)
    validation = read_json(validation_path)
    if (
        training.get("schema") != "contact-rally-start-training-inputs/1"
        or validation.get("schema") != "contact-rally-start-validation-inputs/1"
        or training.get("status") != "complete"
        or validation.get("status") != "complete"
    ):
        raise ValueError("Development prediction inputs are incomplete or have another schema")
    if training.get("labels_read") is not False or validation.get("labels_read") is not False:
        raise ValueError("Development prediction inputs were not made label-free")
    raw_training = training.get("videos")
    raw_validation = validation.get("videos")
    if not isinstance(raw_training, list) or not isinstance(raw_validation, list):
        raise TypeError("Development prediction videos must be lists")
    expected_validation = [video.fixture for video in split.validation_videos]
    if validation.get("validation_videos") != expected_validation:
        raise ValueError("Validation prediction fixtures differ from the fixed split")

    expected_group_by_fixture = {
        video.fixture: group_name
        for group_name, score_group in score_groups.items()
        for video in score_group.scored_videos
    }

    events_by_fixture: dict[str, tuple[FixedEvent, ...]] = {}
    group_by_fixture: dict[str, str] = {}
    evidence_fixtures: list[dict[str, object]] = []
    prediction_rows = [(raw_video, None) for raw_video in raw_training]
    prediction_rows.extend((raw_video, "V") for raw_video in raw_validation)
    for raw_video, default_group in prediction_rows:
        if not isinstance(raw_video, dict):
            raise TypeError("Each development prediction video must be an object")
        identity = raw_video.get("video") if default_group is None else raw_video
        if not isinstance(identity, dict):
            raise TypeError("Development video identity must be an object")
        fixture = str(identity["fixture"])
        video = expected.get(fixture)
        if video is None or int(identity["video_id"]) != video.video_id or float(identity["fps"]) != video.fps:
            raise ValueError(f"{fixture}: development video identity differs")
        if fixture in events_by_fixture:
            raise ValueError(f"Duplicate development fixture {fixture}")
        group = default_group if default_group is not None else str(raw_video["group"])
        if group not in {"A", "B", "C", "D", "V"}:
            raise ValueError(f"{fixture}: unknown development group {group}")
        if default_group is None:
            expected_group = expected_group_by_fixture.get(fixture)
            if group != expected_group:
                raise ValueError(f"{fixture}: saved score group differs")
            expected_training = tuple(
                video.fixture for video in score_groups[group].training_videos
            )
            if tuple(raw_video.get("model_training_videos", ())) != expected_training:
                raise ValueError(f"{fixture}: contact model training videos differ")
        elif video not in split.validation_videos:
            raise ValueError(f"{fixture}: expected a fixed validation video")
        raw_spans = raw_video.get("spans")
        if not isinstance(raw_spans, list):
            raise TypeError(f"{fixture}: spans must be a list")
        events_by_fixture[fixture] = _events(fixture, raw_video.get("kept_contacts"))
        group_by_fixture[fixture] = group
        evidence_fixtures.append({"fixture": fixture, "spans": raw_spans})

    if set(events_by_fixture) != set(expected):
        raise ValueError("Development prediction fixture coverage differs")
    spans = fixed_spans_from_evidence({"fixtures": evidence_fixtures}, events_by_fixture)
    return DevelopmentPredictionPack(
        paths=(training_path, validation_path),
        videos=split.videos,
        group_by_fixture=MappingProxyType(group_by_fixture),
        events_by_fixture=MappingProxyType(events_by_fixture),
        spans=spans,
    )
