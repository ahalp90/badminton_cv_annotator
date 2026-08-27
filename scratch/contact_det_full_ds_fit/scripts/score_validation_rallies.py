"""Score the saved validation contacts after every prediction is fixed."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

import numpy as np

from scratch.contact_det.scripts.score_contact_rallies import (
    DEFAULT_CONFIDENCE_REQUIREMENTS,
    FixedEvent,
    fixed_spans_from_evidence,
    normalise_rallies,
    score_strict_rallies,
    unassigned_events,
)
from scratch.contact_det_full_ds_fit.scripts.baseline_config import FIXED_RUN_IDS
from scratch.contact_det_full_ds_fit.scripts.experiment_config import (
    DevelopmentSplit,
    load_development_split,
    verify_accepted_development_split,
)
from scratch.contact_det_full_ds_fit.scripts.save_validation_rally_predictions import (
    CENTRE_FEATURE_FIELDS,
    PREDICTION_SCHEMA,
)
from scratch.contact_det_full_ds_fit.scripts.score_contact_baseline import (
    ContactLabels,
    _match_contacts,
    _scaled_frames,
    contact_counts,
)

RALLY_RESULT_SCHEMA = "full-dataset-contact-rally-result/1"
SOURCE_COMMIT = re.compile(r"^[0-9a-f]{7,40}$")
PRIMARY_TOLERANCE_AT_30_FPS = 10
SENSITIVITY_TOLERANCE_AT_30_FPS = 5
EXPECTED_RALLY_COUNT = 668
EXPECTED_CONTACT_COUNT = 5696
TIMING_COLUMNS = ("vid", "set_id", "rally", "frame_num")
SIDE_COLUMNS = TIMING_COLUMNS + ("player_side",)
PREDICTION_FIELDS = {
    "schema",
    "status",
    "source_commit",
    "labels_read",
    "menu_result_file",
    "menu_result_sha256",
    "split_file",
    "split_sha256",
    "raw_feature_record_file",
    "raw_feature_record_sha256",
    "contact_label_file",
    "contact_label_sha256",
    "validation_videos",
    "centre_feature_fields_checked",
    "videos",
    "runs",
}
VIDEO_FIELDS = {
    "fixture",
    "video_id",
    "fps",
    "frame_count",
    "spans",
    "replayed_contact_count",
    "input_files",
}
INPUT_FILE_FIELDS = {"role", "filename", "size_bytes", "sha256"}
SPAN_FIELDS = {"span_id", "start_frame", "end_frame"}
RUN_FIELDS = {"run_id", "videos"}
RUN_VIDEO_FIELDS = {"fixture", "contacts"}
CONTACT_FIELDS = {"frame", "timing_score", "predicted_side", "span_id"}
EXPECTED_INPUT_ROLES = (
    "shuttle_track",
    "pose_kps",
    "pose_bboxes",
    "pose_scores",
    "pose_kp_scores",
    "pose_ndet",
    "court_evidence",
    "court_keep_vote",
    "court_present",
    "annotation",
)

TableReader = Callable[..., Any]


@dataclass(frozen=True)
class SavedRunPredictions:
    """One fixed model run, grouped by validation video."""

    run_id: str
    events_by_fixture: Mapping[str, tuple[FixedEvent, ...]]


@dataclass(frozen=True)
class VerifiedRallyPredictions:
    """A checked prediction file ready for later label reads."""

    path: Path
    payload: Mapping[str, Any]
    split: DevelopmentSplit
    spans_by_fixture: Mapping[str, tuple[dict[str, int], ...]]
    runs: tuple[SavedRunPredictions, ...]


@dataclass(frozen=True)
class TimingLabels:
    """Contact frames and rallies loaded without player-side labels."""

    rallies: Mapping[str, tuple[Any, ...]]
    identities: frozenset[tuple[int, str, int, int]]
    frames: Mapping[str, np.ndarray]
    first_contacts: Mapping[str, frozenset[int]]


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be an object with string keys")
    return value


def _check_fields(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields differ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    source_path = Path(path)
    if source_path.name.endswith(".gz"):
        with gzip.open(source_path, "rt", encoding="utf-8") as source:
            value = json.load(source)
    else:
        value = json.loads(source_path.read_text(encoding="utf-8"))
    return dict(_mapping(value, "saved rally predictions"))


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.partial")
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    if destination.name.endswith(".gz"):
        with temporary.open("wb") as raw, gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0
        ) as zipped:
            zipped.write(encoded)
    else:
        temporary.write_bytes(encoded)
    os.replace(temporary, destination)


def _check_bound_file(
    payload: Mapping[str, Any],
    path: Path,
    filename_field: str,
    hash_field: str,
    label: str,
) -> None:
    if payload[filename_field] != Path(path).name:
        raise ValueError(f"{label} filename differs")
    expected_hash = payload[hash_field]
    if not isinstance(expected_hash, str) or _sha256(path) != expected_hash:
        raise ValueError(f"{label} file hash differs")


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer of at least {minimum}")
    return value


def _saved_spans(value: object, fixture: str) -> tuple[dict[str, int], ...]:
    if not isinstance(value, list):
        raise TypeError(f"{fixture}: spans must be a list")
    spans: list[dict[str, int]] = []
    previous_end = -1
    for expected_id, raw_span in enumerate(value):
        span = _mapping(raw_span, f"{fixture}: span {expected_id}")
        _check_fields(span, SPAN_FIELDS, f"{fixture}: span {expected_id}")
        span_id = _integer(span["span_id"], f"{fixture}: span ID")
        start = _integer(span["start_frame"], f"{fixture}: span start")
        end = _integer(span["end_frame"], f"{fixture}: span end", minimum=1)
        if span_id != expected_id or end <= start or start < previous_end:
            raise ValueError(f"{fixture}: rally spans differ")
        spans.append({"span_id": span_id, "start_frame": start, "end_frame": end})
        previous_end = end
    return tuple(spans)


def _expected_span_id(frame: int, spans: Sequence[Mapping[str, int]]) -> int | None:
    matches = [
        int(span["span_id"])
        for span in spans
        if int(span["start_frame"]) <= frame < int(span["end_frame"])
    ]
    if len(matches) > 1:
        raise ValueError(f"frame {frame} belongs to overlapping rally spans")
    return matches[0] if matches else None


def _saved_videos(
    payload: Mapping[str, Any], split: DevelopmentSplit
) -> dict[str, tuple[dict[str, int], ...]]:
    raw_videos = payload["videos"]
    expected_names = [video.fixture for video in split.validation_videos]
    if not isinstance(raw_videos, list) or len(raw_videos) != len(expected_names):
        raise ValueError("saved validation videos differ")
    spans_by_fixture: dict[str, tuple[dict[str, int], ...]] = {}
    for video, raw_video in zip(split.validation_videos, raw_videos, strict=True):
        saved = _mapping(raw_video, f"{video.fixture}: saved video")
        _check_fields(saved, VIDEO_FIELDS, f"{video.fixture}: saved video")
        if (
            saved["fixture"] != video.fixture
            or saved["video_id"] != video.video_id
            or saved["fps"] != video.fps
        ):
            raise ValueError(f"{video.fixture}: saved video details differ")
        frame_count = _integer(saved["frame_count"], f"{video.fixture}: frame count", minimum=1)
        _integer(saved["replayed_contact_count"], f"{video.fixture}: replayed contact count")
        raw_files = saved["input_files"]
        if not isinstance(raw_files, list) or len(raw_files) != len(EXPECTED_INPUT_ROLES):
            raise ValueError(f"{video.fixture}: checked input files differ")
        roles: list[str] = []
        for raw_file in raw_files:
            file_record = _mapping(raw_file, f"{video.fixture}: input file")
            _check_fields(file_record, INPUT_FILE_FIELDS, f"{video.fixture}: input file")
            role = file_record["role"]
            filename = file_record["filename"]
            if (
                not isinstance(role, str)
                or not role
                or role in roles
                or not isinstance(filename, str)
                or PurePosixPath(filename).name != filename
                or type(file_record["size_bytes"]) is not int
                or file_record["size_bytes"] <= 0
                or not isinstance(file_record["sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", file_record["sha256"]) is None
            ):
                raise ValueError(f"{video.fixture}: checked input file differs")
            roles.append(role)
        if tuple(roles) != EXPECTED_INPUT_ROLES:
            raise ValueError(f"{video.fixture}: checked input file order differs")
        spans = _saved_spans(saved["spans"], video.fixture)
        if any(span["end_frame"] > frame_count for span in spans):
            raise ValueError(f"{video.fixture}: rally span exceeds the video frame count")
        spans_by_fixture[video.fixture] = spans
    if payload["validation_videos"] != expected_names:
        raise ValueError("saved validation video order differs")
    return spans_by_fixture


def _normalise_side(value: object, label: str) -> str | None:
    if value is None:
        return None
    if value == "Top":
        return "Top"
    if value in {"Bot", "Bottom"}:
        return "Bot"
    raise ValueError(f"{label}: player side differs")


def _saved_runs(
    payload: Mapping[str, Any],
    split: DevelopmentSplit,
    spans_by_fixture: Mapping[str, Sequence[Mapping[str, int]]],
) -> tuple[SavedRunPredictions, ...]:
    raw_runs = payload["runs"]
    if not isinstance(raw_runs, list) or len(raw_runs) != len(FIXED_RUN_IDS):
        raise ValueError("saved run list differs")
    expected_names = [video.fixture for video in split.validation_videos]
    frame_counts = {
        str(saved_video["fixture"]): int(saved_video["frame_count"])
        for saved_video in payload["videos"]
    }
    side_by_identity: dict[tuple[str, int], str | None] = {}
    runs: list[SavedRunPredictions] = []
    for expected_run_id, raw_run in zip(FIXED_RUN_IDS, raw_runs, strict=True):
        run = _mapping(raw_run, f"{expected_run_id}: saved run")
        _check_fields(run, RUN_FIELDS, f"{expected_run_id}: saved run")
        if run["run_id"] != expected_run_id:
            raise ValueError("saved run order differs")
        raw_videos = run["videos"]
        if not isinstance(raw_videos, list) or len(raw_videos) != len(expected_names):
            raise ValueError(f"{expected_run_id}: saved videos differ")
        events_by_fixture: dict[str, tuple[FixedEvent, ...]] = {}
        for video_name, raw_video in zip(expected_names, raw_videos, strict=True):
            saved_video = _mapping(raw_video, f"{expected_run_id}/{video_name}")
            _check_fields(saved_video, RUN_VIDEO_FIELDS, f"{expected_run_id}/{video_name}")
            if saved_video["fixture"] != video_name:
                raise ValueError(f"{expected_run_id}: saved video order differs")
            raw_contacts = saved_video["contacts"]
            if not isinstance(raw_contacts, list):
                raise TypeError(f"{expected_run_id}/{video_name}: contacts must be a list")
            events: list[FixedEvent] = []
            previous_frame = -1
            for raw_contact in raw_contacts:
                contact = _mapping(raw_contact, f"{expected_run_id}/{video_name}: contact")
                _check_fields(contact, CONTACT_FIELDS, f"{expected_run_id}/{video_name}: contact")
                frame = _integer(contact["frame"], f"{expected_run_id}/{video_name}: frame")
                if frame <= previous_frame or frame >= frame_counts[video_name]:
                    raise ValueError(f"{expected_run_id}/{video_name}: contact frames differ")
                previous_frame = frame
                score = contact["timing_score"]
                if (
                    isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or not math.isfinite(float(score))
                    or not 0.0 <= float(score) <= 1.0
                ):
                    raise ValueError(f"{expected_run_id}/{video_name}: contact score differs")
                side = _normalise_side(
                    contact["predicted_side"], f"{expected_run_id}/{video_name}/{frame}"
                )
                span_id = contact["span_id"]
                if span_id is not None and type(span_id) is not int:
                    raise ValueError(f"{expected_run_id}/{video_name}/{frame}: span ID differs")
                if span_id != _expected_span_id(frame, spans_by_fixture[video_name]):
                    raise ValueError(f"{expected_run_id}/{video_name}/{frame}: span ID differs")
                identity = (video_name, frame)
                if identity in side_by_identity and side_by_identity[identity] != side:
                    raise ValueError(f"{video_name}/{frame}: player-side answers differ between runs")
                side_by_identity[identity] = side
                events.append(FixedEvent(video_name, frame, float(score), side))
            events_by_fixture[video_name] = tuple(events)
        runs.append(SavedRunPredictions(expected_run_id, MappingProxyType(events_by_fixture)))
    return tuple(runs)


def _check_against_raw_feature_record(
    payload: Mapping[str, Any],
    raw_feature_record_path: Path,
    split: DevelopmentSplit,
) -> None:
    raw_record = _mapping(
        json.loads(Path(raw_feature_record_path).read_text(encoding="utf-8")),
        "raw feature record",
    )
    raw_videos = raw_record.get("videos")
    if not isinstance(raw_videos, list):
        raise TypeError("raw feature video records must be a list")
    by_name: dict[str, Mapping[str, Any]] = {}
    for raw_video in raw_videos:
        feature_video = _mapping(raw_video, "raw feature video record")
        identity = _mapping(feature_video.get("video"), "raw feature video identity")
        name = identity.get("name")
        if not isinstance(name, str) or name in by_name:
            raise ValueError("raw feature video identities differ")
        by_name[name] = feature_video
    for video, saved_raw in zip(split.validation_videos, payload["videos"], strict=True):
        saved = _mapping(saved_raw, f"{video.fixture}: saved video")
        feature_video = by_name.get(video.fixture)
        if feature_video is None:
            raise ValueError(f"{video.fixture}: raw feature record is missing")
        feature_files = feature_video.get("input_files")
        if not isinstance(feature_files, list):
            raise TypeError(f"{video.fixture}: raw feature input files must be a list")
        expected_files = [
            file_record
            for file_record in feature_files
            if isinstance(file_record, Mapping)
            and file_record.get("role") in EXPECTED_INPUT_ROLES
        ]
        if saved["input_files"] != expected_files:
            raise ValueError(f"{video.fixture}: saved input files differ from raw features")
        summary = _mapping(feature_video.get("feature_summary"), f"{video.fixture}: feature summary")
        if (
            saved["frame_count"] != summary.get("frame_count")
            or len(saved["spans"]) != summary.get("rally_span_count")
        ):
            raise ValueError(f"{video.fixture}: saved video counts differ from raw features")


def _check_replayed_contact_counts(
    payload: Mapping[str, Any],
    split: DevelopmentSplit,
    runs: Sequence[SavedRunPredictions],
) -> None:
    raw_videos = payload["videos"]
    for video, raw_video in zip(split.validation_videos, raw_videos, strict=True):
        saved_video = _mapping(raw_video, f"{video.fixture}: saved video")
        distinct_frames = {
            event.frame
            for saved_run in runs
            for event in saved_run.events_by_fixture[video.fixture]
        }
        if saved_video["replayed_contact_count"] != len(distinct_frames):
            raise ValueError(f"{video.fixture}: replayed contact count differs")


def load_validation_rally_predictions(
    prediction_path: Path,
    menu_result_path: Path,
    split_path: Path,
    raw_feature_record_path: Path,
    shots_master_path: Path,
) -> VerifiedRallyPredictions:
    """Check every saved prediction before any label row is read."""
    payload = _read_json(prediction_path)
    _check_fields(payload, PREDICTION_FIELDS, "saved rally predictions")
    if (
        payload["schema"] != PREDICTION_SCHEMA
        or payload["status"] != "complete"
        or payload["labels_read"] is not False
        or not isinstance(payload["source_commit"], str)
        or SOURCE_COMMIT.fullmatch(payload["source_commit"]) is None
    ):
        raise ValueError("saved rally predictions are not complete")
    _check_bound_file(
        payload, menu_result_path, "menu_result_file", "menu_result_sha256", "menu result"
    )
    _check_bound_file(payload, split_path, "split_file", "split_sha256", "split")
    _check_bound_file(
        payload,
        raw_feature_record_path,
        "raw_feature_record_file",
        "raw_feature_record_sha256",
        "raw feature record",
    )
    _check_bound_file(
        payload,
        shots_master_path,
        "contact_label_file",
        "contact_label_sha256",
        "contact label",
    )
    if payload["centre_feature_fields_checked"] != list(CENTRE_FEATURE_FIELDS):
        raise ValueError("checked centre-frame feature list differs")
    split = load_development_split(split_path)
    verify_accepted_development_split(split)
    spans_by_fixture = _saved_videos(payload, split)
    _check_against_raw_feature_record(payload, raw_feature_record_path, split)
    runs = _saved_runs(payload, split, spans_by_fixture)
    _check_replayed_contact_counts(payload, split, runs)
    return VerifiedRallyPredictions(
        Path(prediction_path),
        MappingProxyType(payload),
        split,
        MappingProxyType(spans_by_fixture),
        runs,
    )


def _label_rows(
    shots_master_path: Path,
    split: DevelopmentSplit,
    columns: Sequence[str],
    table_reader: TableReader,
) -> tuple[Any, frozenset[tuple[int, str, int, int]]]:
    table = table_reader(Path(shots_master_path), usecols=list(columns))
    if set(table.columns) != set(columns):
        raise ValueError("contact label columns differ")
    validation_ids = {video.video_id for video in split.validation_videos}
    rows = table[table["vid"].isin(validation_ids)].copy()
    if rows.empty or rows[list(TIMING_COLUMNS)].isna().any().any():
        raise ValueError("validation contact timing labels are missing")
    if rows.duplicated(subset=list(TIMING_COLUMNS)).any():
        raise ValueError("validation contact timing identities are repeated")
    if rows.duplicated(subset=["vid", "frame_num"]).any():
        raise ValueError("validation video frame identities are repeated")
    identities: set[tuple[int, str, int, int]] = set()
    for row in rows.itertuples(index=False):
        values = (row.vid, row.rally, row.frame_num)
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) for value in values):
            raise ValueError("validation contact timing numbers must be integers")
        if not isinstance(row.set_id, str) or not row.set_id:
            raise ValueError("validation contact set IDs differ")
        identities.add((int(row.vid), row.set_id, int(row.rally), int(row.frame_num)))
    frozen_identities = frozenset(identities)
    if {identity[0] for identity in frozen_identities} != validation_ids:
        raise ValueError("validation contact timing videos differ")
    return rows, frozen_identities


def load_timing_labels(
    shots_master_path: Path,
    split: DevelopmentSplit,
    table_reader: TableReader,
) -> TimingLabels:
    """Read contact timing columns without reading player-side labels."""
    from annotator.calibration.scoring import load_gt_rallies

    rows, identities = _label_rows(shots_master_path, split, TIMING_COLUMNS, table_reader)
    raw = {
        video.fixture: load_gt_rallies(rows, video.video_id)
        for video in split.validation_videos
    }
    rallies = normalise_rallies(raw)
    rally_count = sum(len(video_rallies) for video_rallies in rallies.values())
    contact_count = sum(
        len(rally.frames) for video_rallies in rallies.values() for rally in video_rallies
    )
    if rally_count != EXPECTED_RALLY_COUNT or contact_count != EXPECTED_CONTACT_COUNT:
        raise ValueError("validation rally or contact total differs")
    frames = {
        video.fixture: np.asarray(
            [frame for rally in rallies[video.fixture] for frame in rally.frames],
            dtype=np.int32,
        )
        for video in split.validation_videos
    }
    first_contacts = {
        video.fixture: frozenset(rally.frames[0] for rally in rallies[video.fixture])
        for video in split.validation_videos
    }
    return TimingLabels(
        MappingProxyType(rallies),
        identities,
        MappingProxyType(frames),
        MappingProxyType(first_contacts),
    )


def load_player_side_labels(
    shots_master_path: Path,
    split: DevelopmentSplit,
    timing_identities: frozenset[tuple[int, str, int, int]],
    table_reader: TableReader,
) -> dict[tuple[str, int], str]:
    """Read player sides after the fixed contacts, scores and sides are loaded."""
    rows, identities = _label_rows(shots_master_path, split, SIDE_COLUMNS, table_reader)
    if identities != timing_identities or rows["player_side"].isna().any():
        raise ValueError("player-side label identities differ from timing labels")
    fixture_by_id = {video.video_id: video.fixture for video in split.validation_videos}
    sides: dict[tuple[str, int], str] = {}
    for row in rows.itertuples(index=False):
        fixture = fixture_by_id[int(row.vid)]
        frame = int(row.frame_num)
        side = _normalise_side(row.player_side, f"{fixture}/{frame} player-side label")
        if side is None:
            raise ValueError(f"{fixture}/{frame}: player-side label is missing")
        sides[(fixture, frame)] = side
    if len(sides) != len(timing_identities):
        raise ValueError("player-side label count differs")
    return sides


def _evidence(verified: VerifiedRallyPredictions) -> dict[str, object]:
    return {
        "fixtures": [
            {"fixture": video.fixture, "spans": list(verified.spans_by_fixture[video.fixture])}
            for video in verified.split.validation_videos
        ]
    }


def _per_video_rally_totals(primary: Mapping[str, Any]) -> list[dict[str, object]]:
    by_fixture: dict[str, list[Mapping[str, Any]]] = {}
    for raw_span in primary["spans"]:
        span = _mapping(raw_span, "whole-rally result span")
        by_fixture.setdefault(str(span["fixture"]), []).append(span)
    output: list[dict[str, object]] = []
    for fixture, spans in by_fixture.items():
        kept = sum(bool(span["kept"]) for span in spans)
        correct = sum(bool(span["fully_correct"]) for span in spans)
        output.append(
            {
                "fixture": fixture,
                "rally_span_count": len(spans),
                "rallies_kept": kept,
                "fully_correct_rallies": correct,
                "kept_rally_accuracy": correct / kept if kept else None,
            }
        )
    return output


def _failure_counts(primary: Mapping[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for raw_span in primary["spans"]:
        span = _mapping(raw_span, "whole-rally result span")
        for reason in span["rejection_reasons"]:
            counts[str(reason)] += 1
    return dict(sorted(counts.items()))


def _player_side_metrics(
    saved_run: SavedRunPredictions,
    timing: TimingLabels,
    sides: Mapping[tuple[str, int], str],
    split: DevelopmentSplit,
    tolerance_at_30_fps: int,
) -> dict[str, object]:
    matched = 0
    answered = 0
    correct = 0
    for video in split.validation_videos:
        events = saved_run.events_by_fixture[video.fixture]
        predicted_frames = np.asarray([event.frame for event in events], dtype=np.int32)
        matches = _match_contacts(
            timing.frames[video.fixture],
            predicted_frames,
            _scaled_frames(tolerance_at_30_fps, video.fps),
        )
        matched += len(matches)
        for contact_index, prediction_index, _offset in matches:
            predicted_side = events[prediction_index].predicted_side
            if predicted_side is None:
                continue
            answered += 1
            contact_frame = int(timing.frames[video.fixture][contact_index])
            if predicted_side == sides[(video.fixture, contact_frame)]:
                correct += 1
    return {
        "tolerance_at_30_fps": tolerance_at_30_fps,
        "timing_matched_contacts": matched,
        "player_side_answers": answered,
        "correct_player_sides": correct,
        "answer_rate_for_timing_matches": answered / matched if matched else None,
        "accuracy_when_answered": correct / answered if answered else None,
        "correct_rate_for_timing_matches": correct / matched if matched else None,
    }


def _score_run(
    saved_run: SavedRunPredictions,
    verified: VerifiedRallyPredictions,
    timing: TimingLabels,
    sides: Mapping[tuple[str, int], str],
) -> dict[str, object]:
    spans = fixed_spans_from_evidence(_evidence(verified), saved_run.events_by_fixture)
    unassigned = unassigned_events(spans, saved_run.events_by_fixture)
    fps_by_fixture = {
        video.fixture: video.fps for video in verified.split.validation_videos
    }
    primary = score_strict_rallies(
        spans,
        timing.rallies,
        sides,
        fps_by_fixture,
        tolerance_base30=PRIMARY_TOLERANCE_AT_30_FPS,
        requirements=DEFAULT_CONFIDENCE_REQUIREMENTS,
        detail_requirement=0.0,
    )
    sensitivity = score_strict_rallies(
        spans,
        timing.rallies,
        sides,
        fps_by_fixture,
        tolerance_base30=SENSITIVITY_TOLERANCE_AT_30_FPS,
        requirements=DEFAULT_CONFIDENCE_REQUIREMENTS,
        detail_requirement=0.0,
    )
    predictions = {
        fixture: np.asarray([event.frame for event in events], dtype=np.int32)
        for fixture, events in saved_run.events_by_fixture.items()
    }
    contact_labels = ContactLabels(
        timing.frames,
        timing.first_contacts,
        {fixture: len(rallies) for fixture, rallies in timing.rallies.items()},
    )
    timing_metrics = {
        str(tolerance): contact_counts(
            contact_labels,
            predictions,
            verified.split.validation_videos,
            tolerance,
        )
        for tolerance in (5, 10, 15)
    }
    events = [
        event
        for video_events in saved_run.events_by_fixture.values()
        for event in video_events
    ]
    answered = sum(event.predicted_side is not None for event in events)
    return {
        "run_id": saved_run.run_id,
        "contact_timing": timing_metrics,
        "player_side": {
            "predicted_contacts": len(events),
            "answered_contacts": answered,
            "answer_rate": answered / len(events) if events else None,
            "by_timing_tolerance": {
                str(tolerance): _player_side_metrics(
                    saved_run,
                    timing,
                    sides,
                    verified.split,
                    tolerance,
                )
                for tolerance in (5, 10, 15)
            },
        },
        "primary": primary,
        "sensitivity": sensitivity,
        "per_video_whole_rallies": _per_video_rally_totals(primary),
        "failure_reason_counts": _failure_counts(primary),
        "unassigned_events": [asdict(event) for event in unassigned],
    }


def score_validation_rallies(
    prediction_path: Path,
    menu_result_path: Path,
    split_path: Path,
    raw_feature_record_path: Path,
    shots_master_path: Path,
    output_path: Path,
    source_commit: str,
    *,
    table_reader: TableReader | None = None,
) -> Path:
    """Check predictions, read timing then player sides, and save all nine scores."""
    destination = Path(output_path)
    _write_json(
        destination,
        {
            "schema": RALLY_RESULT_SCHEMA,
            "status": "running",
            "source_commit": source_commit,
        },
    )
    if SOURCE_COMMIT.fullmatch(source_commit) is None:
        raise ValueError("source commit must be a short or full Git commit")
    verified = load_validation_rally_predictions(
        prediction_path,
        menu_result_path,
        split_path,
        raw_feature_record_path,
        shots_master_path,
    )
    if table_reader is None:
        import pandas as pd

        table_reader = pd.read_csv
    timing = load_timing_labels(shots_master_path, verified.split, table_reader)
    if _sha256(shots_master_path) != verified.payload["contact_label_sha256"]:
        raise ValueError("contact label file changed during the timing-label read")
    sides = load_player_side_labels(
        shots_master_path,
        verified.split,
        timing.identities,
        table_reader,
    )
    if _sha256(shots_master_path) != verified.payload["contact_label_sha256"]:
        raise ValueError("contact label file changed during the player-side read")
    result = {
        "schema": RALLY_RESULT_SCHEMA,
        "status": "complete",
        "source_commit": source_commit,
        "labels_read_after_predictions_fixed": True,
        "timing_columns_read_first": list(TIMING_COLUMNS),
        "player_side_columns_read_second": list(SIDE_COLUMNS),
        "validation_videos": [video.fixture for video in verified.split.validation_videos],
        "rally_count": sum(len(rallies) for rallies in timing.rallies.values()),
        "contact_count": sum(len(frames) for frames in timing.frames.values()),
        "inputs": {
            "rally_prediction_file": Path(prediction_path).name,
            "rally_prediction_sha256": _sha256(prediction_path),
            "menu_result_file": Path(menu_result_path).name,
            "menu_result_sha256": _sha256(menu_result_path),
            "split_file": Path(split_path).name,
            "split_sha256": _sha256(split_path),
            "raw_feature_record_file": Path(raw_feature_record_path).name,
            "raw_feature_record_sha256": _sha256(raw_feature_record_path),
            "contact_label_file": Path(shots_master_path).name,
            "contact_label_sha256": _sha256(shots_master_path),
        },
        "runs": [
            _score_run(saved_run, verified, timing, sides) for saved_run in verified.runs
        ],
    }
    _write_json(destination, result)
    return destination


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--menu-result", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--raw-feature-record", type=Path, required=True)
    parser.add_argument("--shots-master", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    output = score_validation_rallies(
        arguments.predictions,
        arguments.menu_result,
        arguments.split,
        arguments.raw_feature_record,
        arguments.shots_master,
        arguments.output,
        arguments.source_commit,
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
