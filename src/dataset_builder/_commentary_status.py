"""Persist per-video commentary-cleaning status needed by safe resume."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from dataset_builder.selection import COMMENTARY_FAILED
from dataset_builder.vision import load_json_gz, save_json_gz


_SCHEMA = "commentary-cleaning-status/0.1"


def save_cleaning_statuses(
    path: Path,
    statuses: Mapping[str, str],
    selected_video_ids: Sequence[str],
) -> Path:
    """Save only selected failed-cleaning statuses and validate the result."""
    selected = set(selected_video_ids)
    persisted = {
        video_id: status
        for video_id, status in statuses.items()
        if video_id in selected
    }
    saved = save_json_gz(path, {"schema": _SCHEMA, "statuses": persisted})
    load_cleaning_statuses(saved, selected_video_ids)
    return saved


def load_cleaning_statuses(
    path: Path,
    selected_video_ids: Sequence[str],
) -> dict[str, str]:
    """Load a cleaning-status artifact bound to the current selection."""
    payload = load_json_gz(path)
    if set(payload) != {"schema", "statuses"}:
        raise ValueError("commentary cleaning status fields differ")
    if payload["schema"] != _SCHEMA:
        raise ValueError("commentary cleaning status schema differs")
    selected = set(selected_video_ids)
    statuses: dict[str, str] = {}
    for video_id, status in _object(payload["statuses"]).items():
        if not video_id or video_id in {".", ".."} or "/" in video_id or "\\" in video_id:
            raise ValueError("commentary cleaning status video_id is not path-safe")
        if video_id not in selected:
            raise ValueError("commentary cleaning status references an unselected video")
        if status != COMMENTARY_FAILED:
            raise ValueError("commentary cleaning status must record a failed boundary")
        statuses[video_id] = status
    return statuses


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("commentary cleaning statuses must be an object")
    return value
