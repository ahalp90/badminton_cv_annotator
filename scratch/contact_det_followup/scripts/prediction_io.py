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
from scratch.contact_det_full_ds_fit.scripts.inpaint_shuttleset22_tracks import (
    VIDEO_IDS,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PREDICTIONS = (
    REPO_ROOT
    / "scratch/contact_det_full_ds_fit/raw/shuttleset22-test-predictions/combined_predictions.json.gz"
)
PREDICTION_SCHEMA = "shuttleset22-contact-predictions-combined/1"


@dataclass(frozen=True)
class FrozenPredictionPack:
    """Saved prediction rows needed by the follow-up scorers."""

    path: Path
    source_commit: str
    payload: Mapping[str, Any]
    videos: tuple[Mapping[str, Any], ...]
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
        videos.append(raw_video)
        events_by_fixture[fixture] = tuple(events)
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
