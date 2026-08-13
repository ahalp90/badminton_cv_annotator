"""Run the current annotator over a hard-coded fixture list.

The command currently runs builds and reads or writes outputs for the hard-coded
fixture list in the shared contract.  This maintained module is the conceptual
skeleton for a true end-to-end runner in the near future.  It deliberately keeps
the fixed eight-configuration measurement procedural and does not provide a
general experiment-axis or resume framework.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
import platform
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, NamedTuple, cast

import cv2
import numpy as np

import annotator.calibration.gt_scoring as gt_scoring_module
from annotator.artifact_io import (
    atomic_gzip_text_writer,
    load_npy,
    save_npy_xz,
    write_gzip_bytes,
    write_json_object,
)
from annotator.calibration.fixtures import (
    FIXTURES,
    REPO_ROOT,
    SHARED_FILES,
    FilePin,
    Fixture,
    fixtures_root,
    verify_file,
)
from annotator.calibration.gt_scoring import (
    canonical_tolerance,
    flatten_metrics,
    load_gt_tables,
    score_video,
)
from annotator.calibration.scoring import (
    GtRally,
    safe_f1,
    strict_contact_rows,
    wide_edge_contact_rows,
)
from annotator.config import BaseAnnotatorConfig, ResolvedAnnotatorConfig
from annotator.experiment_records import clean_run, human_bytes, utc_run_directory, write_summary_and_report
from annotator.court_evidence import (
    COURT_SCENE_SAMPLE_LIMIT,
    DETECTOR_RESOLUTION,
    PERSON_COURT_MARGIN,
    SCENE_VALID_MIN_FRACTION,
    CourtConsensusError,
    CourtEvidenceResult,
    CourtSceneRecord,
    build_detected_court_evidence,
    build_raw_cut_intervals,
    build_static_court_evidence,
    detect_scene_evidence,
)
from annotator.point_winner import Landing, SHIPPED_LANDING_FILTER_OPTIONS
from annotator.resolve import resolve
from annotator.run_video import AnnotatorResult, LandingHorizonRow, RunCapture, run_video
from annotator.types import DeadMaskMode
from courtkeynet.wrapper import CONFIG_PATH, CourtKeyNetDetector


PARENTS = (
    "static_shuttleset_homography",
    "detected_ckn_opencv_consensus",
)
LANDING_HORIZONS = (1.0, 2.0, 3.0)
REF_ERR_PX = 3.5
BASE_ANNOTATOR_CONFIG = BaseAnnotatorConfig()
LANDING_OPTIONS = SHIPPED_LANDING_FILTER_OPTIONS

COURT_SCENES_COLUMNS = (
    "video_id", "case_id", "court_parent", "scene_index", "start_frame", "end_frame", "n_frames",
    "sampled_frame_indices", "quad_source",
    "raw_tl_x", "raw_tl_y", "raw_tr_x", "raw_tr_y", "raw_br_x", "raw_br_y", "raw_bl_x", "raw_bl_y",
    "peak_tl", "peak_tr", "peak_br", "peak_bl",
    "corner_source_tl", "corner_source_tr", "corner_source_br", "corner_source_bl",
    "fallback_reproj_line_px", "fallback_reproj_anchor_px",
    "fallback_gate_line_frac", "fallback_gate_anchor_frac",
    "fallback_n_lines_used", "fallback_n_correspondences", "fallback_max_sagitta_px",
    "exactly_two_frame_count", "exactly_two_frame_fraction", "scene_valid",
    "consensus_distance_px", "consensus_flagged",
    "active_tl_x", "active_tl_y", "active_tr_x", "active_tr_y",
    "active_br_x", "active_br_y", "active_bl_x", "active_bl_y",
)
LANDING_HORIZON_COLUMNS = (
    "rally_id", "horizon_seconds", "horizon_frames", "final_contact_frame",
    "requested_end_frame", "safe_end_frame", "effective_end_frame", "closure_reasons",
    "strict_landing_frame", "strict_landing_x_norm", "strict_landing_y_norm",
    "strict_landing_half", "strict_landing_at_border", "strict_landing_net_ender",
    "capped_landing_frame", "capped_landing_x_norm", "capped_landing_y_norm",
    "capped_landing_half", "capped_landing_at_border", "capped_landing_net_ender",
    "strict_verdict", "strict_winner", "strict_verdict_source",
    "capped_verdict", "capped_winner", "capped_verdict_source",
    "landing_changed", "winner_changed",
)
STRICT_CONTACT_COLUMNS = (
    "rally_id", "tolerance_base30", "tolerance_frames", "row_kind",
    "gt_frame", "candidate_frame", "offset_frames",
)
WIDE_CONTACT_COLUMNS = (
    "window_id", "rally_id", "edge", "window_start", "window_end", "row_kind",
    "gt_frame", "candidate_frame", "offset_frames",
)

_EXPECTED_CASES = (
    ("sset_01/tracknet-stride-8", "sset_01", 8, "nonoverlap", 25.0, 154393),
    ("sset_01/tracknet-stride-1", "sset_01", 1, "weight", 25.0, 154393),
    ("sset_15/tracknet-stride-8", "sset_15", 8, "nonoverlap", 25.0, 149487),
    ("sset_21/tracknet-stride-8", "sset_21", 8, "nonoverlap", 30.0, 100349),
)
_PRODUCER_KEYS = (
    "video:sset_01", "video:sset_15", "video:sset_21",
    "pose:sset_01", "pose:sset_15", "pose:sset_21",
    "track:sset_01/tracknet-stride-8", "track:sset_01/tracknet-stride-1",
    "track:sset_15/tracknet-stride-8", "track:sset_21/tracknet-stride-8",
    "courtkeynet_config", "courtkeynet_weights",
)


@dataclass(frozen=True)
class FixedCase:
    """One fixed video and shuttle-track case in the measurement matrix."""

    case_id: str
    fixture_name: str
    tracknet_stride: int
    tracknet_producer_mode: str
    fps: float
    n_frames: int


CASES = tuple(FixedCase(*values) for values in _EXPECTED_CASES)


@dataclass(frozen=True)
class InputManifest:
    """Strict input description for the fixed measurement."""

    schema_version: int
    videos: dict[str, FilePin]
    track_overrides: dict[str, FilePin]
    courtkeynet_config: FilePin
    courtkeynet_weights: FilePin
    producers: dict[str, str]


@dataclass
class CaseData:
    """Loaded shared arrays and regenerated scene partition for one fixed case."""

    fixed: FixedCase
    fixture: Fixture
    video_pin: FilePin
    track_pin: FilePin
    bboxes_pin: FilePin
    scores_pin: FilePin
    kps_pin: FilePin
    ndet_pin: FilePin
    video_path: Path
    track: np.ndarray
    bboxes: np.ndarray
    scores: np.ndarray
    kps: np.ndarray
    ndet: np.ndarray
    raw_cuts: list[tuple[int, int]]
    raw_cuts_artifact: dict[str, Any] | None = None
    status: str = "not_run"
    failure_path: Path | None = None


@dataclass
class ConfigurationState:
    """Terminal state and retained data for one parent/case configuration."""

    fixed: FixedCase
    parent: str
    fixture: Fixture
    case: CaseData
    directory: Path
    inputs: list[dict[str, Any]]
    started_at_utc: str = ""
    started_clock: float = 0.0
    result: AnnotatorResult | None = None
    court_result: CourtEvidenceResult | None = None
    capture: RunCapture | None = None
    resolved_config: ResolvedAnnotatorConfig | None = None
    status: str = "not_run"
    failure_path: Path | None = None
    manifest_path: Path | None = None


@dataclass
class RunDriver:
    """State for one fixed run, retained until terminal manifests are written."""

    manifest_path: Path
    output_root: Path
    device: str
    command: tuple[str, ...]
    started_at_utc: str
    started_clock: float
    input_manifest: InputManifest | None = None
    input_manifest_bytes: bytes = b""
    master: Any = None
    homo_df: Any = None
    courts: dict[Any, Any] | None = None
    resolution: Any = None
    source_commit: str = ""
    resolved_device: str = ""
    detector: object | None = None
    detector_factory: Callable[..., object] = CourtKeyNetDetector
    cases: dict[str, CaseData] | None = None
    configurations: list[ConfigurationState] | None = None
    setup_failure_path: Path | None = None
    scoring_failure_path: Path | None = None
    run_log_path: Path | None = None
    input_manifest_output_path: Path | None = None


class VideoMetadata(NamedTuple):
    """Metadata probed from one pinned video."""

    fps: float
    n_frames: int
    width: int
    height: int


def utc_now() -> str:
    """Return an RFC 3339 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _artifact_record(run_root: Path, path: Path) -> dict[str, Any]:
    """Hash one fully closed output artefact."""
    if not path.is_file():
        raise ValueError(f"closed artefact is missing: {path}")
    relative = path.resolve().relative_to(run_root.resolve())
    data = path.read_bytes()
    return {"path": relative.as_posix(), "md5": _md5_bytes(data), "bytes": len(data)}


def _json_ready(value: object, *, top_level: bool = False) -> object:
    """Convert the annotator's tuples, dataclasses and NumPy values to JSON values."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_ready(getattr(value, field.name)) for field in fields(value)}
    if hasattr(value, "_asdict"):
        return {key: _json_ready(item) for key, item in value._asdict().items()}
    if isinstance(value, Mapping):
        items = [(str(key), _json_ready(item)) for key, item in value.items()]
        if not top_level:
            if items and all(_integer_like(key) for key, _item in items):
                items.sort(key=lambda pair: int(pair[0]))
            else:
                items.sort(key=lambda pair: pair[0])
        return dict(items)
    if isinstance(value, frozenset) or isinstance(value, set):
        return [_json_ready(item) for item in sorted(value)]
    if isinstance(value, tuple) or isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("JSON value is not finite")
        return value
    raise TypeError(f"cannot serialise {type(value).__name__}")


def _integer_like(value: str) -> bool:
    try:
        int(value)
    except ValueError:
        return False
    return True


def _write_json(path: Path, value: object) -> None:
    payload = _json_ready(value, top_level=True)
    if not isinstance(payload, Mapping):
        raise TypeError("top-level JSON artifact must be an object")
    write_json_object(path, payload)


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return _csv_value(value.item())
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("CSV value is not finite")
    return value


def _write_rows(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    with atomic_gzip_text_writer(path, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in columns})


def _relative_pin_path(pin: FilePin) -> str:
    return pin.path.as_posix()


def _pin_from_json(value: object, field: str) -> FilePin:
    if not isinstance(value, dict) or set(value) != {"path", "md5", "root"}:
        raise ValueError(f"{field} must contain exactly path, md5 and root")
    raw_path, raw_md5, raw_root = value["path"], value["md5"], value["root"]
    if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path:
        raise ValueError(f"{field}.path must be a relative POSIX path")
    path = PurePosixPath(raw_path)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"{field}.path must not escape its root")
    if not isinstance(raw_md5, str) or len(raw_md5) != 32 or raw_md5.lower() != raw_md5:
        raise ValueError(f"{field}.md5 must be lower-case hexadecimal MD5")
    try:
        int(raw_md5, 16)
    except ValueError as error:
        raise ValueError(f"{field}.md5 must be lower-case hexadecimal MD5") from error
    if raw_root not in {"fixtures", "repo"}:
        raise ValueError(f"{field}.root must be fixtures or repo")
    return FilePin(Path(raw_path), raw_md5, cast(Any, raw_root))


def parse_input_manifest(data: bytes | str | Mapping[str, object]) -> InputManifest:
    """Parse the strict input manifest schema."""
    payload: object
    if isinstance(data, Mapping):
        payload = data
    else:
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as error:
            raise ValueError("input manifest is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("input manifest must be a JSON object")
    expected = {"schema_version", "videos", "track_overrides", "courtkeynet_config", "courtkeynet_weights", "producers"}
    if set(payload) != expected:
        missing = sorted(expected.difference(payload))
        unknown = sorted(set(payload).difference(expected))
        raise ValueError(f"input manifest fields differ; missing={missing}, unknown={unknown}")
    if payload["schema_version"] != 1 or isinstance(payload["schema_version"], bool):
        raise ValueError("input manifest schema_version must be integer 1")
    videos_raw = payload["videos"]
    if not isinstance(videos_raw, dict) or set(videos_raw) != {"sset_01", "sset_15", "sset_21"}:
        raise ValueError("videos must contain exactly the three fixed fixture names")
    tracks_raw = payload["track_overrides"]
    if not isinstance(tracks_raw, dict) or set(tracks_raw) != {"sset_01/tracknet-stride-1"}:
        raise ValueError("track_overrides must contain exactly the stride-1 sset_01 case")
    producers_raw = payload["producers"]
    if not isinstance(producers_raw, dict) or set(producers_raw) != set(_PRODUCER_KEYS):
        raise ValueError("producers keys do not match the fixed consumed roles")
    producers: dict[str, str] = {}
    for key in _PRODUCER_KEYS:
        value = producers_raw[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"producer {key} must be a non-empty string")
        producers[key] = value.strip()
    videos = {key: _pin_from_json(value, f"videos.{key}") for key, value in videos_raw.items()}
    track_overrides = {key: _pin_from_json(value, f"track_overrides.{key}") for key, value in tracks_raw.items()}
    if any(pin.root != "fixtures" for pin in videos.values()):
        raise ValueError("video pins must use the fixtures root")
    if any(pin.root != "fixtures" for pin in track_overrides.values()):
        raise ValueError("track override pins must use the fixtures root")
    courtkeynet_config = _pin_from_json(payload["courtkeynet_config"], "courtkeynet_config")
    courtkeynet_weights = _pin_from_json(payload["courtkeynet_weights"], "courtkeynet_weights")
    if courtkeynet_config.root != "repo":
        raise ValueError("courtkeynet_config must use the repo root")
    if courtkeynet_weights.root != "fixtures":
        raise ValueError("courtkeynet_weights must use the fixtures root")
    return InputManifest(1, videos, track_overrides, courtkeynet_config, courtkeynet_weights, producers)


def verify_selected_pins(pins: Sequence[FilePin]) -> None:
    """Verify exactly the selected pins, without loading a fixture's old file bundle."""
    for pin in pins:
        _pin_path(pin)
        verify_file(pin)


def _pin_path(pin: FilePin) -> Path:
    root = fixtures_root() if pin.root == "fixtures" else REPO_ROOT
    resolved_root = root.resolve()
    resolved_path = (resolved_root / pin.path).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"pinned path escapes its {pin.root} root: {pin.path}") from error
    return resolved_path


def _pin_record(
    pin: FilePin,
    producer: str,
    shape: Sequence[int] | None = None,
    dtype: str | None = None,
) -> dict[str, Any]:
    return {
        "role": "",
        "path": _relative_pin_path(pin),
        "md5": pin.md5,
        "producer": producer,
        "shape": list(shape) if shape is not None else None,
        "dtype": dtype,
    }


def _array_pin_record(role: str, pin: FilePin, producer: str, array: np.ndarray) -> dict[str, Any]:
    record = _pin_record(pin, producer, array.shape, str(array.dtype))
    record["role"] = role
    return record


def _plain_pin_record(role: str, pin: FilePin, producer: str) -> dict[str, Any]:
    record = _pin_record(pin, producer)
    record["role"] = role
    return record


def _fixture_by_name() -> dict[str, Fixture]:
    return {fixture.name: fixture for fixture in FIXTURES}


def _load_array(pin: FilePin) -> np.ndarray:
    return np.load(_pin_path(pin), allow_pickle=False)


def _validate_arrays(fixed: FixedCase, track: np.ndarray, bboxes: np.ndarray, scores: np.ndarray,
                    kps: np.ndarray, ndet: np.ndarray) -> None:
    n_frames = fixed.n_frames
    if track.shape != (n_frames, 3):
        raise ValueError(f"{fixed.case_id}: track shape {track.shape} != {(n_frames, 3)}")
    if bboxes.ndim != 3 or bboxes.shape[0] != n_frames or bboxes.shape[2] != 4:
        raise ValueError(f"{fixed.case_id}: bboxes shape {bboxes.shape} is invalid")
    n_slots = bboxes.shape[1]
    if scores.shape != (n_frames, n_slots):
        raise ValueError(f"{fixed.case_id}: scores shape {scores.shape} is invalid")
    if kps.shape != (n_frames, n_slots, 17, 2):
        raise ValueError(f"{fixed.case_id}: kps shape {kps.shape} is invalid")
    if ndet.shape != (n_frames,) or not np.issubdtype(ndet.dtype, np.integer):
        raise ValueError(f"{fixed.case_id}: ndet shape or dtype is invalid")
    for name, array in (("track", track), ("bboxes", bboxes), ("scores", scores), ("kps", kps)):
        if np.isinf(array).any():
            raise ValueError(f"{fixed.case_id}: {name} contains infinity")
    if not np.isfinite(track).all():
        raise ValueError(f"{fixed.case_id}: track must be finite")
    if (ndet < 0).any() or (ndet > n_slots).any():
        raise ValueError(f"{fixed.case_id}: ndet is outside [0, n_slots]")
    finite_score_count = np.isfinite(scores).sum(axis=1)
    if not np.array_equal(ndet.astype(np.int64), finite_score_count.astype(np.int64)):
        raise ValueError(f"{fixed.case_id}: ndet does not equal the finite-score count")
    active_slots = np.arange(n_slots)[None, :] < ndet[:, None]
    if not np.isfinite(bboxes[active_slots]).all() or not np.isfinite(kps[active_slots]).all():
        raise ValueError(f"{fixed.case_id}: active pose detections must be finite")


def probe_video(video_path: Path) -> VideoMetadata:
    """Read fixed metadata from a video without running inference."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"could not open video {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        n_frames = int(round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
        width = int(round(float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))))
        height = int(round(float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    finally:
        capture.release()
    return VideoMetadata(fps, n_frames, width, height)


def validate_video_metadata(fixed: FixedCase, metadata: VideoMetadata) -> None:
    if not math.isclose(metadata.fps, fixed.fps, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"{fixed.case_id}: video FPS {metadata.fps} != {fixed.fps}")
    if metadata.n_frames != fixed.n_frames:
        raise ValueError(f"{fixed.case_id}: video frame count {metadata.n_frames} != {fixed.n_frames}")
    if (metadata.width, metadata.height) != DETECTOR_RESOLUTION:
        raise ValueError(
            f"{fixed.case_id}: video dimensions {(metadata.width, metadata.height)} "
            f"!= {DETECTOR_RESOLUTION}"
        )


def _raw_cut_rows(raw_cuts: Sequence[tuple[int, int]]) -> list[dict[str, int]]:
    return [
        {"scene_index": index, "start_frame": start, "end_frame": end}
        for index, (start, end) in enumerate(raw_cuts)
    ]


def _save_mask(path: Path, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.bool_)
    if values.ndim != 1:
        raise ValueError("saved masks must be one-dimensional")
    save_npy_xz(path, np.ascontiguousarray(values))


def _corner_values(corners: np.ndarray | None) -> list[object]:
    if corners is None:
        return [None] * 8
    array = np.asarray(corners, dtype=float)
    return [float(value) for value in array.reshape(-1)]


def _scene_row(record: CourtSceneRecord) -> dict[str, object]:
    raw = _corner_values(record.raw_corners_px)
    active = _corner_values(record.active_corners_native_px)
    peaks = [None] * 4 if record.raw_peaks is None else [float(value) for value in record.raw_peaks]
    corner_sources = [None] * 4 if record.raw_corner_source is None else list(record.raw_corner_source)
    diagnostics = record.fallback_diagnostics
    diagnostic_values = [None] * 7 if diagnostics is None else [
        float(diagnostics.reproj_line_px), float(diagnostics.reproj_anchor_px),
        float(diagnostics.gate_line_frac), float(diagnostics.gate_anchor_frac),
        int(diagnostics.n_lines_used), int(diagnostics.n_correspondences), float(diagnostics.max_sagitta_px),
    ]
    return dict(zip(
        COURT_SCENES_COLUMNS,
        [
            record.video_id, record.case_id, record.parent, record.scene_index, record.start_frame,
            record.end_frame, record.end_frame - record.start_frame,
            json.dumps(list(record.sampled_frame_indices), separators=(",", ":")), record.raw_source,
            raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
            peaks[0], peaks[1], peaks[2], peaks[3],
            corner_sources[0], corner_sources[1], corner_sources[2], corner_sources[3],
            diagnostic_values[0], diagnostic_values[1], diagnostic_values[2], diagnostic_values[3],
            diagnostic_values[4], diagnostic_values[5], diagnostic_values[6],
            record.exactly_two_count, record.exactly_two_fraction, record.scene_valid,
            record.consensus_distance_px, record.consensus_flag,
            active[0], active[1], active[2], active[3], active[4], active[5], active[6], active[7],
        ],
    ))


def _landing_fields(landing: Landing | None) -> list[object]:
    if landing is None:
        return [None] * 6
    return [
        landing.frame, landing.norm[0], landing.norm[1], landing.half,
        landing.at_border, landing.net_ender,
    ]


def _horizon_row(row: LandingHorizonRow) -> dict[str, object]:
    strict_winner = row.strict_winner
    capped_winner = row.capped_winner
    strict_verdict = row.strict_verdict
    capped_verdict = row.capped_verdict
    strict_landing = _landing_fields(row.strict_landing)
    capped_landing = _landing_fields(row.capped_landing)
    return dict(zip(
        LANDING_HORIZON_COLUMNS,
        [
            row.rally_id, row.horizon_seconds, row.horizon_frames, row.final_contact_frame,
            row.requested_end_frame, row.safe_end_frame, row.effective_end_frame,
            "+".join(row.closure_reasons),
            *strict_landing, *capped_landing,
            getattr(strict_verdict, "verdict", None), strict_winner, getattr(strict_verdict, "verdict_source", None),
            getattr(capped_verdict, "verdict", None), capped_winner, getattr(capped_verdict, "verdict_source", None),
            row.landing_changed, row.winner_changed,
        ],
    ))


def _strict_metrics(rows: Sequence[Mapping[str, object]], tolerance_base30: int, fps: float) -> dict[str, object]:
    selected = [row for row in rows if row["tolerance_base30"] == tolerance_base30]
    matched = [row for row in selected if row["row_kind"] == "matched"]
    unmatched_gt = [row for row in selected if row["row_kind"] == "unmatched_gt"]
    unmatched_candidate = [row for row in selected if row["row_kind"] == "unmatched_candidate"]
    gt_count = len(matched) + len(unmatched_gt)
    candidate_count = len(matched) + len(unmatched_candidate)
    matched_count = len(matched)
    precision = matched_count / candidate_count if candidate_count else None
    recall = matched_count / gt_count if gt_count else None
    f1 = None if precision is None or recall is None else safe_f1(precision, recall)
    offsets = [abs(int(row["offset_frames"])) for row in matched]
    mean_offset = float(np.mean(offsets)) if offsets else None
    return {
        "tolerance_frames": next((row["tolerance_frames"] for row in selected), None),
        "gt_count": gt_count,
        "candidate_count": candidate_count,
        "matched_count": matched_count,
        "unmatched_gt_count": len(unmatched_gt),
        "unmatched_candidate_count": len(unmatched_candidate),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_abs_offset_frames": mean_offset,
        "mean_abs_offset_seconds": mean_offset / fps if mean_offset is not None else None,
    }


def _landing_metrics(rows: Sequence[LandingHorizonRow]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for horizon in LANDING_HORIZONS:
        selected = [row for row in rows if row.horizon_seconds == horizon]
        result[f"seconds_{int(horizon)}"] = {
            "eligible_rally_count": len(selected),
            "horizon_bound_count": sum("horizon_cap" in row.closure_reasons for row in selected),
            "landing_changed_count": sum(row.landing_changed for row in selected),
            "winner_changed_count": sum(row.winner_changed for row in selected),
        }
    return result


def _configuration_values(dead_mask_mode: DeadMaskMode) -> dict[str, object]:
    return {
        "base_annotator_config": "BaseAnnotatorConfig()",
        "dead_mask_mode": dead_mask_mode.value,
        "landing_filter_options": {
            "settle_win": LANDING_OPTIONS.settle_win,
            "settle_thr": LANDING_OPTIONS.settle_thr,
            "settle_min": LANDING_OPTIONS.settle_min,
            "carry_win": LANDING_OPTIONS.carry_win,
            "carry_thr": LANDING_OPTIONS.carry_thr,
        },
        "ref_err_px": REF_ERR_PX,
        "injected_positions": False,
        "injected_spans": False,
        "injected_contacts": False,
        "serve_start": None,
        "court_invalid_is_excluded": True,
        "person_margin": PERSON_COURT_MARGIN,
        "scene_threshold": SCENE_VALID_MIN_FRACTION,
        "court_samples": COURT_SCENE_SAMPLE_LIMIT,
        "landing_horizons_s": list(LANDING_HORIZONS),
    }


def _device_record(requested: str, resolved: str) -> dict[str, str]:
    return {"requested": requested, "resolved": resolved}


def _source_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    )
    commit = result.stdout.strip().lower()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("git commit is not a 40-character hexadecimal id")
    return commit


def _require_clean_source_tree() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    )
    if status.stdout.strip():
        raise ValueError("tracked source tree has staged or unstaged changes")


def validate_output_root(output_root: Path) -> Path:
    resolved = output_root.expanduser().resolve()
    if resolved.exists():
        raise ValueError("output root already exists")
    return resolved


def _configuration_path(root: Path, parent: str, case_id: str) -> Path:
    return root / parent / Path(case_id)


def _failure_payload(
    scope: str,
    case_id: str | None,
    parent: str | None,
    stage: str,
    error: BaseException,
) -> dict[str, object]:
    import traceback

    return {
        "schema_version": 1,
        "scope": scope,
        "case_id": case_id,
        "court_parent": parent,
        "stage": stage,
        "exception_type": type(error).__name__,
        "message": str(error),
        "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
        "occurred_at_utc": utc_now(),
    }


def _write_failure(
    path: Path,
    scope: str,
    case_id: str | None,
    parent: str | None,
    stage: str,
    error: BaseException,
) -> None:
    _write_json(path, _failure_payload(scope, case_id, parent, stage, error))


def _shared_pins_by_fixture() -> dict[str, tuple[FilePin, ...]]:
    result: dict[str, list[FilePin]] = {fixture.name: [] for fixture in FIXTURES}
    for pin in SHARED_FILES:
        for fixture in FIXTURES:
            if pin.path.parent == fixture.gt_set_dir:
                result[fixture.name].append(pin)
    return {key: tuple(value) for key, value in result.items()}


def verify_eligible_gt_files() -> dict[str, tuple[FilePin, ...]]:
    """Verify all per-set CSV membership and hashes at the scoring boundary."""
    by_fixture = _shared_pins_by_fixture()
    for fixture in FIXTURES:
        directory = (REPO_ROOT / fixture.gt_set_dir).resolve()
        expected = {pin.path.name for pin in by_fixture[fixture.name]}
        actual = {path.name for path in directory.glob("*.csv")}
        if actual != expected:
            raise ValueError(f"{fixture.name}: GT CSV set {sorted(actual)} != {sorted(expected)}")
        verify_selected_pins(by_fixture[fixture.name])
    return by_fixture


def _input_records(case: CaseData, manifest: InputManifest) -> list[dict[str, Any]]:
    fixture = case.fixture
    producer = manifest.producers
    records = [
        _plain_pin_record("video", case.video_pin, producer[f"video:{fixture.name}"]),
        _array_pin_record("track", case.track_pin, producer[f"track:{case.fixed.case_id}"], case.track),
        _array_pin_record("bboxes", case.bboxes_pin, producer[f"pose:{fixture.name}"], case.bboxes),
        _array_pin_record("scores", case.scores_pin, producer[f"pose:{fixture.name}"], case.scores),
        _array_pin_record("kps", case.kps_pin, producer[f"pose:{fixture.name}"], case.kps),
        _array_pin_record("ndet", case.ndet_pin, producer[f"pose:{fixture.name}"], case.ndet),
    ]
    records.extend([
        _plain_pin_record("resolution_csv", SHARED_FILES[2], "repository calibration table"),
        _plain_pin_record("homography_csv", SHARED_FILES[1], "repository calibration table"),
    ])
    return records


def _load_case(fixed: FixedCase, manifest: InputManifest, fixture: Fixture) -> CaseData:
    video_pin = manifest.videos[fixture.name]
    track_pin = fixture.files[0]
    if fixed.tracknet_stride == 1:
        track_pin = manifest.track_overrides[fixed.case_id]
    bboxes_pin, scores_pin, kps_pin, ndet_pin = fixture.files[1], fixture.files[2], fixture.files[3], fixture.files[5]
    verify_selected_pins((video_pin, track_pin, bboxes_pin, scores_pin, kps_pin, ndet_pin))
    track, bboxes, scores, kps, ndet = (
        _load_array(track_pin), _load_array(bboxes_pin), _load_array(scores_pin),
        _load_array(kps_pin), _load_array(ndet_pin),
    )
    _validate_arrays(fixed, track, bboxes, scores, kps, ndet)
    video_path = _pin_path(video_pin)
    validate_video_metadata(fixed, probe_video(video_path))
    raw_cuts = build_raw_cut_intervals(video_path, fixed.n_frames, fixed.fps)
    return CaseData(
        fixed, fixture, video_pin, track_pin, bboxes_pin, scores_pin, kps_pin, ndet_pin,
        video_path, track, bboxes, scores, kps, ndet, raw_cuts,
    )


def _write_raw_cuts(case: CaseData, root: Path) -> None:
    path = root / "shared" / case.fixed.case_id / "raw_cuts.csv.gz"
    _write_rows(path, ("scene_index", "start_frame", "end_frame"), _raw_cut_rows(case.raw_cuts))
    case.raw_cuts_artifact = _artifact_record(root, path)
    case.status = "succeeded"


def _write_scene_evidence(directory: Path, court_result: CourtEvidenceResult) -> None:
    _write_rows(
        directory / "court_scenes.csv.gz",
        COURT_SCENES_COLUMNS,
        (_scene_row(record) for record in court_result.scene_records),
    )
    _write_rows(
        directory / "scene_rows.csv.gz",
        ("video_id", "start_frame", "end_frame", "upleft_x", "upleft_y", "upright_x", "upright_y",
         "downleft_x", "downleft_y", "downright_x", "downright_y"),
        court_result.inputs.homography_rows.to_dict("records") if court_result.inputs is not None else (),
    )
    _save_mask(directory / "keep_vote.npy.xz", court_result.keep_vote)
    _save_mask(directory / "court_present.npy.xz", court_result.court_present)


def _write_annotations(directory: Path, result: AnnotatorResult) -> None:
    _write_json(directory / "annotations.json.gz", result)


def _write_landing_horizons(directory: Path, rows: Sequence[LandingHorizonRow]) -> None:
    ordered = sorted(rows, key=lambda row: (row.rally_id, row.horizon_seconds))
    _write_rows(
        directory / "landing_horizons.csv.gz",
        LANDING_HORIZON_COLUMNS,
        (_horizon_row(row) for row in ordered),
    )


def _write_scoring_outputs(
    directory: Path,
    fixture: Fixture,
    case: CaseData,
    result: AnnotatorResult,
    capture: RunCapture,
    master: Any,
    court_info: dict[str, object],
) -> dict[str, object]:
    rallies = _gt_rallies_for_fixture(master, fixture)
    strict_rows = strict_contact_rows(result.spans, result.filtered_contacts, rallies, case.fixed.fps)
    wide_rows = wide_edge_contact_rows(rallies, result.filtered_contacts, case.fixed.fps, case.fixed.n_frames)
    _write_rows(directory / "strict_contacts.csv.gz", STRICT_CONTACT_COLUMNS, strict_rows)
    _write_rows(directory / "wide_edge_contacts.csv.gz", WIDE_CONTACT_COLUMNS, wide_rows)
    scoring = score_video(
        fixture, result, master, {fixture.video_id: court_info}, canonical_tolerance(case.fixed.fps)
    )
    metrics = {
        "schema_version": 1,
        "configuration_id": f"{directory.parent.parent.name}/{case.fixed.case_id}",
        "existing_calibration": flatten_metrics(scoring),
        "strict_contacts": {
            "base30_5": _strict_metrics(strict_rows, 5, case.fixed.fps),
            "base30_10": _strict_metrics(strict_rows, 10, case.fixed.fps),
        },
        "court_valid_fraction": None,
        "definitive_exclusion_fraction": float(np.asarray(capture.definitive_exclusion_mask, dtype=bool).mean()),
        "landing_horizons": _landing_metrics(capture.landing_horizon_rows),
    }
    # The court-valid value comes from the parent evidence, not from the exclusion mask.
    court_present = load_npy(directory / "court_present.npy.xz")
    metrics["court_valid_fraction"] = float(court_present.mean())
    _write_json(directory / "metrics.json.gz", metrics)
    return metrics


def _gt_rallies_for_fixture(master: Any, fixture: Fixture) -> list[GtRally]:
    from annotator.calibration.scoring import load_gt_rallies

    return load_gt_rallies(master, fixture.video_id)


def _configuration_manifest(
    state: ConfigurationState,
    driver: RunDriver,
    failure_record: dict[str, Any] | None,
) -> dict[str, object]:
    if state.resolved_config is None:
        raise ValueError(f"{state.fixed.case_id}: resolved annotator configuration is missing")
    finished = utc_now()
    artefacts = []
    for path in sorted(state.directory.rglob("*")):
        if path.is_file() and path.name != "manifest.json.gz":
            artefacts.append(_artifact_record(driver.output_root, path))
    artefacts.sort(key=lambda record: record["path"])
    inputs = list(state.inputs)
    if state.parent == "detected_ckn_opencv_consensus" and driver.input_manifest is not None:
        inputs.extend([
            _plain_pin_record("courtkeynet_config", driver.input_manifest.courtkeynet_config,
                              driver.input_manifest.producers["courtkeynet_config"]),
            _plain_pin_record("courtkeynet_weights", driver.input_manifest.courtkeynet_weights,
                              driver.input_manifest.producers["courtkeynet_weights"]),
        ])
    if state.result is not None and driver.input_manifest is not None and driver.scoring_failure_path is None:
        # These roles are added only after the one global GT membership check.
        pins = sorted(_shared_pins_by_fixture()[state.fixture.name], key=lambda pin: pin.path.as_posix())
        inputs.append(_plain_pin_record("shots_master", SHARED_FILES[0], "ShuttleSet ground truth"))
        inputs.extend(_plain_pin_record(pin.path.as_posix(), pin, "ShuttleSet ground truth") for pin in pins)
    for index, record in enumerate(inputs):
        if not record.get("role"):
            record["role"] = str(record.get("path", index))
    inputs.sort(key=lambda record: [
        "video", "track", "bboxes", "scores", "kps", "ndet", "resolution_csv", "homography_csv",
        "courtkeynet_config", "courtkeynet_weights", "shots_master",
    ].index(record["role"]) if record["role"] in {
        "video", "track", "bboxes", "scores", "kps", "ndet", "resolution_csv", "homography_csv",
        "courtkeynet_config", "courtkeynet_weights", "shots_master",
    } else 99)
    return {
        "schema_version": 1,
        "configuration_id": f"{state.parent}/{state.fixed.case_id}",
        "status": state.status,
        "case_id": state.fixed.case_id,
        "court_parent": state.parent,
        "video_id": state.fixture.video_id,
        "tracknet_stride": state.fixed.tracknet_stride,
        "tracknet_producer_mode": state.fixed.tracknet_producer_mode,
        "source_commit": driver.source_commit,
        "command": list(driver.command),
        "device": _device_record(driver.device, driver.resolved_device),
        "started_at_utc": state.started_at_utc,
        "finished_at_utc": finished,
        "elapsed_seconds": max(0.0, time.monotonic() - state.started_clock),
        "configuration": _configuration_values(state.resolved_config.dead_mask_mode),
        "resolved_annotator_config": state.resolved_config,
        "inputs": inputs,
        "shared_artifacts": [state.case.raw_cuts_artifact] if state.case.raw_cuts_artifact else [],
        "artifacts": artefacts,
        "failure": failure_record,
    }


def _write_terminal_configuration_manifest(state: ConfigurationState, driver: RunDriver,
                                           failure_record: dict[str, Any] | None = None) -> None:
    manifest_path = state.directory / "manifest.json.gz"
    _write_json(manifest_path, _configuration_manifest(state, driver, failure_record))
    state.manifest_path = manifest_path


def _run_one_configuration(state: ConfigurationState, driver: RunDriver) -> None:
    state.started_at_utc = utc_now()
    state.started_clock = time.monotonic()
    state.resolved_config = resolve(BASE_ANNOTATOR_CONFIG, state.fixed.fps)
    if state.case.status != "succeeded":
        state.status = "failed"
        state.failure_path = state.case.failure_path
        if state.failure_path is None:
            raise ValueError(f"{state.fixed.case_id}: failed case has no failure record")
        _write_terminal_configuration_manifest(
            state, driver, _artifact_record(driver.output_root, state.failure_path)
        )
        return
    state.directory.mkdir(parents=True, exist_ok=True)
    parent = state.parent
    case = state.case
    try:
        if parent == "static_shuttleset_homography":
            court_result = build_static_court_evidence(
                case.fixed.case_id, parent, case.fixture.video_id, driver.homo_df, case.fixture.resolution,
                case.raw_cuts, case.bboxes, case.scores, case.ndet,
                gate_resolution_table=driver.resolution, ref_err_px=REF_ERR_PX,
            )
        else:
            if driver.detector is None:
                raise ValueError("detected parent has no shared CourtKeyNet detector")
            evidence = detect_scene_evidence(case.video_path, case.raw_cuts, driver.detector)
            court_result = build_detected_court_evidence(
                case.fixed.case_id, parent, case.fixture.video_id, case.fixture.resolution, case.raw_cuts,
                evidence, case.bboxes, case.scores, case.ndet,
                gate_resolution_table=driver.resolution, ref_err_px=REF_ERR_PX,
            )
        state.court_result = court_result
        _write_scene_evidence(state.directory, court_result)
    except CourtConsensusError as error:
        state.court_result = error.result
        _write_scene_evidence_partial(state.directory, error.result)
        failure_path = state.directory / "failure.json.gz"
        _write_failure(failure_path, "configuration", case.fixed.case_id, parent, "court_consensus", error)
        state.failure_path = failure_path
        state.status = "failed"
        _write_terminal_configuration_manifest(state, driver, _artifact_record(driver.output_root, failure_path))
        return
    except Exception as error:
        failure_path = state.directory / "failure.json.gz"
        _write_failure(failure_path, "configuration", case.fixed.case_id, parent, "court_evidence", error)
        state.failure_path = failure_path
        state.status = "failed"
        _write_terminal_configuration_manifest(state, driver, _artifact_record(driver.output_root, failure_path))
        return

    try:
        if court_result.inputs is None:
            raise ValueError("court evidence has no operational inputs")
        capture = RunCapture()
        state.capture = capture
        input_values = court_result.inputs
        gate_court_info = {str(case.fixture.video_id): input_values.court_info}
        cut_frames = [end for _start, end in case.raw_cuts[:-1]]
        result = run_video(
            case.track, case.bboxes, case.scores, case.kps, case.ndet,
            fps=case.fixed.fps,
            base=BASE_ANNOTATOR_CONFIG,
            landing_options=LANDING_OPTIONS,
            net_band=input_values.net_band,
            resolution=input_values.resolution,
            video_id=case.fixture.video_id,
            court_info=input_values.court_info,
            homo_df=driver.homo_df if parent == "static_shuttleset_homography" else None,
            gate_court_info=gate_court_info,
            gate_resolution_table=input_values.gate_resolution_table,
            ref_err_px=REF_ERR_PX,
            raw_exclusion_mask=None,
            court_present=court_result.court_present,
            homography_rows=input_values.homography_rows,
            cut_frames=cut_frames,
            keep_vote=court_result.keep_vote,
            court_invalid_is_excluded=True,
            landing_error_band_m=input_values.landing_error_band_m,
            landing_horizons_s=LANDING_HORIZONS,
            capture=capture,
        )
        state.result = result
        if capture.raw_exclusion_mask is None or capture.definitive_exclusion_mask is None:
            raise ValueError("run_video did not capture both exclusion masks")
        _save_mask(state.directory / "raw_replay_mask.npy.xz", capture.raw_exclusion_mask)
        _save_mask(state.directory / "definitive_exclusion_mask.npy.xz", capture.definitive_exclusion_mask)
        _write_annotations(state.directory, result)
        _write_landing_horizons(state.directory, capture.landing_horizon_rows)
    except Exception as error:
        failure_path = state.directory / "failure.json.gz"
        _write_failure(failure_path, "configuration", case.fixed.case_id, parent, "inference", error)
        state.failure_path = failure_path
        state.status = "failed"
        _write_terminal_configuration_manifest(state, driver, _artifact_record(driver.output_root, failure_path))
        return
    state.status = "inference_only"


def _write_scene_evidence_partial(directory: Path, result: CourtEvidenceResult) -> None:
    _write_rows(
        directory / "court_scenes.csv.gz",
        COURT_SCENES_COLUMNS,
        (_scene_row(record) for record in result.scene_records),
    )
    _save_mask(directory / "keep_vote.npy.xz", result.keep_vote)
    _save_mask(directory / "court_present.npy.xz", result.court_present)


def _configuration_summary(state: ConfigurationState, driver: RunDriver) -> dict[str, object]:
    failure = _artifact_record(driver.output_root, state.failure_path) if state.failure_path else None
    manifest = _artifact_record(driver.output_root, state.manifest_path) if state.manifest_path else None
    return {
        "configuration_id": f"{state.parent}/{state.fixed.case_id}",
        "status": state.status,
        "manifest": manifest,
        "failure": failure,
    }


def _not_run_configuration_summaries() -> list[dict[str, object]]:
    return [
        {
            "configuration_id": f"{parent}/{fixed.case_id}",
            "status": "not_run",
            "manifest": None,
            "failure": None,
        }
        for parent in PARENTS
        for fixed in CASES
    ]


def _environment(driver: RunDriver) -> dict[str, object]:
    import importlib.metadata
    import torch

    def version(package: str) -> str | None:
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "requested_device": driver.device,
        "resolved_device": driver.resolved_device,
        "cuda_available": bool(torch.cuda.is_available()),
        "packages": {
            "numpy": version("numpy"), "pandas": version("pandas"), "opencv": cv2.__version__,
            "scenedetect": version("scenedetect"), "torch": version("torch"), "safetensors": version("safetensors"),
        },
    }


def _run_manifest(
    driver: RunDriver,
    exit_code: int,
    setup_failure: dict[str, Any] | None,
) -> dict[str, object]:
    states = driver.configurations or []
    resolved_dead_mask_modes: set[DeadMaskMode] = set()
    for state in states:
        if state.resolved_config is not None:
            resolved_dead_mask_modes.add(state.resolved_config.dead_mask_mode)
    if not resolved_dead_mask_modes:
        for case in CASES:
            resolved_dead_mask_modes.add(
                resolve(BASE_ANNOTATOR_CONFIG, case.fps).dead_mask_mode
            )
    if len(resolved_dead_mask_modes) != 1:
        values = ", ".join(sorted(mode.value for mode in resolved_dead_mask_modes))
        raise ValueError(f"fixed cases resolved to multiple dead-mask modes: {values}")
    dead_mask_mode = next(iter(resolved_dead_mask_modes))
    successful_inference = sum(state.status in {"inference_only", "succeeded"} for state in states)
    succeeded = sum(state.status == "succeeded" for state in states)
    if setup_failure is not None:
        status = "failed"
    elif successful_inference == 0:
        status = "failed"
    elif succeeded != len(states):
        status = "partial_failure"
    else:
        status = "succeeded"
    cases = []
    for case in CASES:
        data = driver.cases.get(case.case_id) if driver.cases else None
        cases.append({
            "case_id": case.case_id,
            "status": data.status if data else "not_run",
            "raw_cuts": data.raw_cuts_artifact if data else None,
            "failure": _artifact_record(driver.output_root, data.failure_path) if data and data.failure_path else None,
        })
    configurations = (
        _not_run_configuration_summaries()
        if setup_failure is not None
        else [_configuration_summary(state, driver) for state in states]
    )
    return {
        "schema_version": 1,
        "run_id": driver.output_root.name,
        "status": status,
        "source_commit": driver.source_commit,
        "command": list(driver.command),
        "started_at_utc": driver.started_at_utc,
        "finished_at_utc": utc_now(),
        "elapsed_seconds": max(0.0, time.monotonic() - driver.started_clock),
        "input_manifest_source": str(driver.manifest_path.resolve()),
        "input_manifest": _artifact_record(driver.output_root, driver.input_manifest_output_path)
        if driver.input_manifest_output_path else None,
        "run_log": _artifact_record(driver.output_root, driver.run_log_path) if driver.run_log_path else None,
        "configuration": _configuration_values(dead_mask_mode),
        "environment": _environment(driver),
        "cases": cases,
        "configurations": configurations,
        "setup_failure": _artifact_record(driver.output_root, driver.setup_failure_path)
        if driver.setup_failure_path else None,
        "scoring_failure": _artifact_record(driver.output_root, driver.scoring_failure_path)
        if driver.scoring_failure_path else None,
        "exit_code": exit_code,
    }


def _make_detector(driver: RunDriver) -> None:
    if driver.input_manifest is None:
        raise ValueError("input manifest is missing")
    weight_path = _pin_path(driver.input_manifest.courtkeynet_weights)
    detector = driver.detector_factory(weights_path=weight_path, device=driver.device)
    driver.detector = detector
    driver.resolved_device = str(getattr(detector, "device", driver.device))


def _setup(driver: RunDriver) -> None:
    driver.source_commit = _source_commit()
    _require_clean_source_tree()
    manifest = parse_input_manifest(driver.input_manifest_bytes)
    driver.input_manifest = manifest
    verify_selected_pins((manifest.courtkeynet_config, manifest.courtkeynet_weights))
    expected_config_path = CONFIG_PATH.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    if manifest.courtkeynet_config.path.as_posix() != expected_config_path:
        raise ValueError("courtkeynet_config does not point to the current repository CONFIG_PATH")
    # The setup retains the master for scoring, while geometry tables enter inference.
    # The existing helper shares one pin tuple with its old calibration loader. Restrict its
    # verification view for this call so eligible per-set CSVs stay at the post-inference gate.
    original_shared_files = gt_scoring_module.SHARED_FILES
    # NB NOT THREADSAFE
    gt_scoring_module.SHARED_FILES = tuple(SHARED_FILES[:3])
    try:
        master, homo_df, courts, resolution = load_gt_tables()  # Non-sset runs need a no-GT path.
    finally:
        gt_scoring_module.SHARED_FILES = original_shared_files
    driver.master, driver.homo_df, driver.courts, driver.resolution = master, homo_df, courts, resolution
    driver.resolved_device = driver.device
    if driver.input_manifest_output_path is None:
        raise ValueError("input manifest output path is missing")
    cases: dict[str, CaseData] = {}
    fixture_map = _fixture_by_name()
    for fixed in CASES:
        fixture = fixture_map[fixed.fixture_name]
        try:
            cases[fixed.case_id] = _load_case(fixed, manifest, fixture)
            _write_raw_cuts(cases[fixed.case_id], driver.output_root)
        except Exception as error:
            case = CaseData(
                fixed,
                fixture,
                manifest.videos[fixture.name],
                fixture.files[0],
                fixture.files[1],
                fixture.files[2],
                fixture.files[3],
                fixture.files[5],
                Path(),
                np.empty((0, 3)),
                np.empty((0, 0, 4)),
                np.empty((0, 0)),
                np.empty((0, 0, 17, 2)),
                np.empty((0,), dtype=np.int64),
                [],
                status="failed",
            )
            failure_path = driver.output_root / "shared" / fixed.case_id / "failure.json.gz"
            _write_failure(failure_path, "configuration", fixed.case_id, None, "shared_case", error)
            case.failure_path = failure_path
            cases[fixed.case_id] = case
    driver.cases = cases
    _make_detector(driver)
    driver.configurations = [
        ConfigurationState(
            fixed, parent, fixture_map[fixed.fixture_name], cases[fixed.case_id],
            _configuration_path(driver.output_root, parent, fixed.case_id),
            _input_records(cases[fixed.case_id], manifest) if cases[fixed.case_id].status == "succeeded" else [],
        )
        for parent in PARENTS for fixed in CASES
    ]


def _score_configurations(driver: RunDriver) -> None:
    if driver.configurations is None or driver.master is None or driver.courts is None:
        raise ValueError("scoring state is incomplete")
    try:
        verify_eligible_gt_files()
    except Exception as error:
        path = driver.output_root / "scoring_failure.json.gz"
        _write_failure(path, "scoring", None, None, "gt_verification", error)
        driver.scoring_failure_path = path
        for state in driver.configurations:
            if state.status == "inference_only":
                state.status = "inference_only"
                _write_terminal_configuration_manifest(state, driver, _artifact_record(driver.output_root, path))
        return
    for state in driver.configurations:
        if (
            state.status != "inference_only"
            or state.result is None
            or state.capture is None
            or state.court_result is None
        ):
            continue
        try:
            if state.court_result.inputs is None:
                raise ValueError("successful inference has no court inputs")
            _write_scoring_outputs(
                state.directory, state.fixture, state.case, state.result, state.capture,
                driver.master, state.court_result.inputs.court_info,
            )
            state.status = "succeeded"
            _write_terminal_configuration_manifest(state, driver)
        except Exception as error:
            path = state.directory / "failure.json.gz"
            _write_failure(path, "configuration", state.fixed.case_id, state.parent, "scoring", error)
            state.failure_path = path
            state.status = "failed"
            _write_terminal_configuration_manifest(state, driver, _artifact_record(driver.output_root, path))


def _write_initial_run_files(driver: RunDriver) -> None:
    input_manifest_path = driver.output_root / "input_manifest.json.gz"
    write_gzip_bytes(input_manifest_path, driver.input_manifest_bytes)
    driver.input_manifest_output_path = input_manifest_path

    run_log_path = driver.output_root / "run.log"
    with run_log_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"started_at_utc={driver.started_at_utc}\n")
    driver.run_log_path = run_log_path


def run_annotator_measurement(manifest_path: Path, output_root: Path, device: str = "cpu",
                              *, command: Sequence[str] | None = None,
                              detector_factory: Callable[..., object] = CourtKeyNetDetector) -> int:
    """Run the fixed eight-configuration measurement and return its exit code."""
    resolved_output = validate_output_root(output_root)
    resolved_manifest = manifest_path.expanduser().resolve()
    input_bytes = resolved_manifest.read_bytes()
    resolved_output.mkdir(parents=True, exist_ok=False)
    driver = RunDriver(
        resolved_manifest, resolved_output, device,
        tuple(command or (
            sys.executable,
            "-m",
            "annotator.e2e_court_annotator",
            "--manifest",
            str(manifest_path),
            "--device",
            device,
        )),
        utc_now(), time.monotonic(), input_manifest_bytes=input_bytes, detector_factory=detector_factory,
    )
    setup_failure: dict[str, Any] | None = None
    exit_code = 0
    try:
        _write_initial_run_files(driver)
        _setup(driver)
    except Exception as error:
        path = resolved_output / "setup_failure.json.gz"
        _write_failure(path, "setup", None, None, "setup", error)
        driver.setup_failure_path = path
        exit_code = 1
        setup_failure = _failure_payload("setup", None, None, "setup", error)
    else:
        for state in driver.configurations or ():
            _run_one_configuration(state, driver)
        _score_configurations(driver)
        failed = any(state.status == "failed" for state in driver.configurations or ())
        if failed or any(state.status != "succeeded" for state in driver.configurations or ()):
            exit_code = 3
    try:
        _write_json(resolved_output / "manifest.json.gz", _run_manifest(driver, exit_code, setup_failure))
    except Exception as error:
        print(f"could not write terminal run manifest: {error}", file=sys.stderr)
        return 1
    return exit_code


def _run_cli_measurement(manifest_path: Path, device: str, command: Sequence[str]) -> int:
    """Run the real CLI into its one timestamped in-repository directory."""
    output_root = utc_run_directory()
    if output_root.exists():
        raise ValueError(f"annotator run directory already exists: {output_root}")
    print(f"Annotator run directory: {output_root}")
    exit_code = run_annotator_measurement(manifest_path, output_root, device, command=command)
    if exit_code != 0:
        return exit_code
    try:
        _summary_path, _report_path, compressed = write_summary_and_report(output_root)
        archive = clean_run(output_root)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"measurement completed in {output_root}, but reporting or cleaning failed: {error}", file=sys.stderr)
        return 1
    print(f"Annotator run directory: {output_root}")
    if archive is not None:
        print(f"Pre-clean backup: {archive}")
    print(
        f"Compressed masks and arrays: {compressed['file_count']} NPY.XZ files "
        f"({human_bytes(compressed['total_bytes'])}). Git will preserve them with the run."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the fixed eight-configuration court annotator measurement")
    parser.add_argument("--manifest", type=Path, required=True, metavar="INPUT_MANIFEST")
    parser.add_argument("--device", default="cpu", metavar="DEVICE")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = sys.argv if argv is None else (sys.argv[0], *argv)
    try:
        return _run_cli_measurement(args.manifest, args.device, command)
    except (OSError, ValueError) as error:
        build_parser().error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
