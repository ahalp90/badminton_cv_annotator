"""Whole-video vision adapters and full automatic annotation persistence."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import gzip
import json
import lzma
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import TYPE_CHECKING, cast
from uuid import uuid4

import numpy as np

from annotator.shuttle_track import validate_shuttle_track
from dataset_builder._pose_process import (
    POSE_CHILD_STEM,
    load_raw_pose_mapping,
    pose_subprocess_environment,
    resolve_pose_executable,
)
from dataset_builder.shuttle_quality import (
    ShuttleQualitySummary,
    summarize_shuttle_quality,
)

if TYPE_CHECKING:
    import pandas as pd

    from annotator.config import BaseAnnotatorConfig
    from annotator.court_evidence import CourtEvidenceResult, CourtInputs
    from annotator.point_winner import LandingFilterOptions
    from annotator.run_video import AnnotatorResult
    from annotator.video_metadata import VideoMetadata
    from bst_x.pipeline.shuttle_extractor import WholeVideoShuttle


ANNOTATOR_RESULT_SCHEMA = "annotator-result/0.1"
COURT_EVIDENCE_SCHEMA = "court-evidence/0.1"
RAW_REPLAY_MASK_FILENAME = "raw_replay_mask.npy.xz"
DEFINITIVE_EXCLUSION_MASK_FILENAME = "definitive_exclusion_mask.npy.xz"
ANNOTATOR_RESULT_FILENAME = "annotator_result.json.gz"
SHUTTLE_QUALITY_FILENAME = "shuttle_quality.json.gz"
TRACK_FILENAME = "shuttle_track.npy.xz"
COURT_EVIDENCE_FILENAME = "court_evidence.json.gz"
COURT_KEEP_VOTE_FILENAME = "court_keep_vote.npy.xz"
COURT_PRESENT_FILENAME = "court_present.npy.xz"
POSE_FILENAMES = {
    "kps": "pose_kps.npy.xz", "bboxes": "pose_bboxes.npy.xz", "scores": "pose_scores.npy.xz",
    "kp_scores": "pose_kp_scores.npy.xz", "ndet": "pose_ndet.npy.xz",
}
_POSE_CHILD_COMMAND = "_extract-rtmlib-pose"

@dataclass(frozen=True)
class PoseArrays:
    """Five frame-aligned arrays emitted by the canonical RTMLib extractor."""

    kps: np.ndarray
    bboxes: np.ndarray
    scores: np.ndarray
    kp_scores: np.ndarray
    ndet: np.ndarray


@dataclass(frozen=True)
class PoseArtifacts:
    """Canonical compressed paths for one pose extraction."""

    kps: Path
    bboxes: Path
    scores: Path
    kp_scores: Path
    ndet: Path

    def as_mapping(self) -> dict[str, Path]:
        """Return stable manifest names mapped to their paths."""
        return {
            "pose_kps": self.kps, "pose_bboxes": self.bboxes, "pose_scores": self.scores,
            "pose_kp_scores": self.kp_scores, "pose_ndet": self.ndet,
        }


@dataclass(frozen=True)
class PoseExtraction:
    """Validated pose arrays, their persisted artefacts, and executed command."""

    arrays: PoseArrays
    artifacts: PoseArtifacts
    command: tuple[str, ...]


@dataclass(frozen=True)
class CourtVision:
    """Raw scene intervals and their detected CourtKeyNet evidence."""

    raw_cuts: tuple[tuple[int, int], ...]
    evidence: CourtEvidenceResult
    artifacts: CourtArtifacts | None = None


@dataclass(frozen=True)
class CourtArtifacts:
    """Persisted operational and provenance outputs for detected court evidence."""

    evidence: Path
    keep_vote: Path
    court_present: Path

    def as_mapping(self) -> dict[str, Path]:
        """Return stable manifest names mapped to their paths."""
        return {
            "court_evidence": self.evidence, "court_keep_vote": self.keep_vote, "court_present": self.court_present,
        }


@dataclass(frozen=True)
class AnnotationRun:
    """Full primitive result plus the two distinct captured masks."""

    video_id: str
    result: AnnotatorResult
    raw_replay_mask: np.ndarray
    definitive_exclusion_mask: np.ndarray
    shuttle_quality: ShuttleQualitySummary


@dataclass(frozen=True)
class AnnotationArtifacts:
    """Persisted outputs of one full annotation run."""

    result: Path
    raw_replay_mask: Path
    definitive_exclusion_mask: Path
    shuttle_quality: Path

    def as_mapping(self) -> dict[str, Path]:
        """Return stable manifest names mapped to their paths."""
        return {
            "annotator_result": self.result,
            "raw_replay_mask": self.raw_replay_mask,
            "definitive_exclusion_mask": self.definitive_exclusion_mask,
            "shuttle_quality": self.shuttle_quality,
        }


@dataclass(frozen=True)
class AnnotationOutput:
    """One completed full annotation and its compressed artefacts."""

    run: AnnotationRun
    artifacts: AnnotationArtifacts


def save_npy_xz(path: Path, values: np.ndarray) -> Path:
    """Atomically store one NumPy array as XZ-compressed ``.npy`` bytes."""
    destination = Path(path)
    if not destination.name.endswith(".npy.xz"):
        raise ValueError(f"compressed NumPy path must end in .npy.xz: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        with lzma.open(temporary, "wb", format=lzma.FORMAT_XZ, preset=9) as handle:
            np.save(handle, np.asarray(values), allow_pickle=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_npy_xz(path: Path) -> np.ndarray:
    """Load one XZ-compressed NumPy array without permitting pickle data."""
    source = Path(path)
    with lzma.open(source, "rb", format=lzma.FORMAT_XZ) as handle:
        return np.load(handle, allow_pickle=False)


def save_json_gz(path: Path, payload: Mapping[str, object]) -> Path:
    """Atomically store deterministic UTF-8 JSON in a gzip member."""
    destination = Path(path)
    if not destination.name.endswith(".json.gz"):
        raise ValueError(f"compressed JSON path must end in .json.gz: {destination}")
    encoded = json.dumps(
        _json_ready(payload),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    _atomic_write_bytes(destination, gzip.compress(encoded, compresslevel=9, mtime=0))
    return destination


def load_json_gz(path: Path) -> dict[str, object]:
    """Load a gzip-compressed JSON object."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise ValueError(f"compressed JSON must contain an object: {path}")
    return payload


def convert_tracknet_csv_stage(
    csv_path: Path,
    *,
    video_id: str,
    metadata: VideoMetadata,
    output_path: Path,
) -> WholeVideoShuttle:
    """Convert and persist a strict whole-video TrackNet CSV."""
    from bst_x.pipeline.shuttle_extractor import whole_video_csv_to_shuttle

    shuttle = whole_video_csv_to_shuttle(
        csv_path,
        video_id=video_id,
        frame_count=metadata.frame_count,
        width=metadata.width,
        height=metadata.height,
    )
    save_npy_xz(output_path, shuttle.track)
    return shuttle


def pose_artifact_paths(output_dir: Path) -> PoseArtifacts:
    """Return the five canonical compressed pose paths under ``output_dir``."""
    root = Path(output_dir)
    return PoseArtifacts(**{name: root / filename for name, filename in POSE_FILENAMES.items()})


def extract_rtmlib_pose_stage(
    *,
    metadata: VideoMetadata,
    output_dir: Path,
    interpreter: str | Path,
    device: str = "cuda",
    n_max: int = 16,
) -> PoseExtraction:
    """Run the canonical pose producer in its configured interpreter."""
    if isinstance(n_max, bool) or not isinstance(n_max, int) or not 0 < n_max <= 127:
        raise ValueError(f"n_max must be an integer in [1, 127], got {n_max!r}")
    executable = resolve_pose_executable(interpreter)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rtmlib-", dir=root) as raw_dir_text:
        raw_dir = Path(raw_dir_text)
        command = rtmlib_pose_command(
            executable=executable,
            video_path=metadata.source_path,
            raw_output_dir=raw_dir,
            device=device,
            n_max=n_max,
        )
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            cwd=Path(__file__).resolve().parents[2],
            env=pose_subprocess_environment(),
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(
                f"RTMLib pose subprocess exited with status {completed.returncode}: {detail}"
            )
        arrays = PoseArrays(**load_raw_pose_mapping(raw_dir, POSE_CHILD_STEM))
    validate_pose_arrays(arrays, metadata.frame_count)
    artifacts = save_pose_arrays(root, arrays, metadata.frame_count)
    return PoseExtraction(arrays=arrays, artifacts=artifacts, command=tuple(command))


def rtmlib_pose_command(
    *,
    executable: Path,
    video_path: Path,
    raw_output_dir: Path,
    device: str,
    n_max: int,
) -> list[str]:
    """Build the auditable child command for one whole-video pose extract."""
    return [
        os.fspath(executable),
        "-m",
        "dataset_builder.vision",
        _POSE_CHILD_COMMAND,
        "--video",
        os.fspath(video_path),
        "--output-dir",
        os.fspath(raw_output_dir),
        "--device",
        device,
        "--n-max",
        str(n_max),
    ]


def validate_pose_arrays(arrays: PoseArrays, frame_count: int) -> None:
    """Validate raw pose shapes, dtypes, padding, and canonical frame count."""
    bboxes = np.asarray(arrays.bboxes)
    scores = np.asarray(arrays.scores)
    kps = np.asarray(arrays.kps)
    kp_scores = np.asarray(arrays.kp_scores)
    ndet = np.asarray(arrays.ndet)
    if bboxes.ndim != 3 or bboxes.shape[0] != frame_count or bboxes.shape[2] != 4:
        raise ValueError(f"pose bboxes shape {bboxes.shape} is invalid for {frame_count} frames")
    n_slots = bboxes.shape[1]
    if n_slots <= 0:
        raise ValueError("pose arrays must contain at least one detection slot")
    expected = {
        "scores": (scores, (frame_count, n_slots)),
        "kps": (kps, (frame_count, n_slots, 17, 2)),
        "kp_scores": (kp_scores, (frame_count, n_slots, 17)),
    }
    for name, (values, shape) in expected.items():
        if values.shape != shape:
            raise ValueError(f"pose {name} shape {values.shape} != {shape}")
    for name, values in (
        ("bboxes", bboxes),
        ("scores", scores),
        ("kps", kps),
        ("kp_scores", kp_scores),
    ):
        if not np.issubdtype(values.dtype, np.floating):
            raise ValueError(f"pose {name} must have floating dtype, got {values.dtype}")
        if np.isinf(values).any():
            raise ValueError(f"pose {name} contains infinity")
    if ndet.shape != (frame_count,):
        raise ValueError(f"pose ndet shape {ndet.shape} != {(frame_count,)}")
    if np.issubdtype(ndet.dtype, np.bool_) or not np.issubdtype(ndet.dtype, np.integer):
        raise ValueError(f"pose ndet must have integer dtype, got {ndet.dtype}")
    if ((ndet < 0) | (ndet > n_slots)).any():
        raise ValueError(f"pose ndet values must be in [0, {n_slots}]")
    finite_score_count = np.isfinite(scores).sum(axis=1)
    if not np.array_equal(ndet.astype(np.int64), finite_score_count.astype(np.int64)):
        raise ValueError("pose ndet does not equal the finite-score count")
    active = np.arange(n_slots)[None, :] < ndet[:, None]
    if not np.array_equal(np.isfinite(scores), active):
        raise ValueError("pose finite scores must occupy exactly the leading ndet slots")
    for name, values in (("bboxes", bboxes), ("kps", kps), ("kp_scores", kp_scores)):
        if not np.isfinite(values[active]).all():
            raise ValueError(f"active pose {name} values must be finite")
        if not np.isnan(values[~active]).all():
            raise ValueError(f"inactive pose {name} values must be NaN padding")


def save_pose_arrays(output_dir: Path, arrays: PoseArrays, frame_count: int) -> PoseArtifacts:
    """Validate and atomically publish the five compressed pose arrays."""
    validate_pose_arrays(arrays, frame_count)
    artifacts = pose_artifact_paths(output_dir)
    for name, path in artifacts.as_mapping().items():
        field_name = name.removeprefix("pose_")
        save_npy_xz(path, cast(np.ndarray, getattr(arrays, field_name)))
    return artifacts


def load_pose_arrays(output_dir: Path, frame_count: int) -> PoseArrays:
    """Load and validate the five canonical compressed pose arrays."""
    artifacts = pose_artifact_paths(output_dir)
    arrays = PoseArrays(
        kps=load_npy_xz(artifacts.kps),
        bboxes=load_npy_xz(artifacts.bboxes),
        scores=load_npy_xz(artifacts.scores),
        kp_scores=load_npy_xz(artifacts.kp_scores),
        ndet=load_npy_xz(artifacts.ndet),
    )
    validate_pose_arrays(arrays, frame_count)
    return arrays


def build_detected_court_stage(
    *,
    video_id: str,
    metadata: VideoMetadata,
    pose: PoseArrays,
    detector: object,
    output_dir: Path,
    case_id: str | None = None,
    parent: str = "detected_ckn_opencv_consensus",
    ref_err_px: float = 3.5,
) -> CourtVision:
    """Build raw-cut and detected CourtKeyNet evidence with existing producers."""
    from annotator.court_evidence import (
        build_detected_court_evidence,
        build_raw_cut_intervals,
        detect_scene_evidence,
    )

    validate_pose_arrays(pose, metadata.frame_count)
    cuts = build_raw_cut_intervals(
        metadata.source_path,
        metadata.frame_count,
        float(metadata.fps),
    )
    scene_evidence = detect_scene_evidence(metadata.source_path, cuts, detector)
    resolution = (float(metadata.width), float(metadata.height))
    result = build_detected_court_evidence(
        case_id or video_id,
        parent,
        video_id,
        resolution,
        cuts,
        scene_evidence,
        pose.bboxes,
        pose.scores,
        pose.ndet,
        detector_resolution=resolution,
        ref_err_px=ref_err_px,
    )
    court = CourtVision(tuple(cuts), result)
    _validate_court_vision(
        cuts,
        result,
        metadata.frame_count,
        resolution,
    )
    artifacts = persist_court_vision(
        output_dir,
        video_id=video_id,
        court=court,
        frame_count=metadata.frame_count,
        resolution=resolution,
    )
    return CourtVision(court.raw_cuts, court.evidence, artifacts)


def persist_court_vision(
    output_dir: Path,
    *,
    video_id: str,
    court: CourtVision,
    frame_count: int,
    resolution: tuple[float, float],
) -> CourtArtifacts:
    """Persist operational court inputs, raw evidence, and frame masks."""
    _validate_court_vision(court.raw_cuts, court.evidence, frame_count, resolution)
    inputs = court.evidence.inputs
    assert inputs is not None
    root = Path(output_dir)
    payload = {
        "schema": COURT_EVIDENCE_SCHEMA,
        "video_id": video_id,
        "raw_cuts": [list(interval) for interval in court.raw_cuts],
        "inputs": _court_inputs_payload(inputs),
        "scene_records": court.evidence.scene_records,
        "consensus": court.evidence.consensus,
    }
    evidence_path = save_json_gz(root / COURT_EVIDENCE_FILENAME, payload)
    keep_vote_path = save_npy_xz(root / COURT_KEEP_VOTE_FILENAME, court.evidence.keep_vote)
    court_present_path = save_npy_xz(root / COURT_PRESENT_FILENAME, court.evidence.court_present)
    return CourtArtifacts(evidence_path, keep_vote_path, court_present_path)


def load_court_vision(
    output_dir: Path,
    *,
    video_id: str,
    frame_count: int,
    resolution: tuple[float, float],
) -> CourtVision:
    """Restore validated operational court evidence from compressed artefacts."""
    from annotator.court_evidence import CourtEvidenceResult
    from dataset_builder._court_codec import load_court_provenance

    root = Path(output_dir)
    payload = load_json_gz(root / COURT_EVIDENCE_FILENAME)
    expected = {"schema", "video_id", "raw_cuts", "inputs", "scene_records", "consensus"}
    if set(payload) != expected:
        raise ValueError("court evidence payload fields differ from court-evidence/0.1")
    if payload["schema"] != COURT_EVIDENCE_SCHEMA:
        raise ValueError(f"unsupported court evidence schema: {payload['schema']!r}")
    if payload["video_id"] != video_id:
        raise ValueError(
            f"court evidence video_id {payload['video_id']!r} does not match {video_id!r}"
        )
    raw_cuts = _raw_cuts_from_payload(payload["raw_cuts"])
    inputs = _court_inputs_from_payload(payload["inputs"])
    scene_records, consensus = load_court_provenance(
        payload["scene_records"],
        payload["consensus"],
        raw_cuts=raw_cuts,
        video_id=video_id,
    )
    keep_vote = load_npy_xz(root / COURT_KEEP_VOTE_FILENAME)
    court_present = load_npy_xz(root / COURT_PRESENT_FILENAME)
    _validated_mask(keep_vote, frame_count, "persisted court keep_vote")
    _validated_mask(court_present, frame_count, "persisted court_present")
    result = CourtEvidenceResult(
        inputs=inputs,
        scene_records=scene_records,
        keep_vote=keep_vote,
        court_present=court_present,
        consensus=consensus,
    )
    artifacts = CourtArtifacts(
        root / COURT_EVIDENCE_FILENAME,
        root / COURT_KEEP_VOTE_FILENAME,
        root / COURT_PRESENT_FILENAME,
    )
    court = CourtVision(raw_cuts, result, artifacts)
    _validate_court_vision(court.raw_cuts, court.evidence, frame_count, resolution)
    return court


def _court_inputs_payload(inputs: CourtInputs) -> dict[str, object]:
    return {
        "court_info": inputs.court_info,
        "gate_court_info": inputs.gate_court_info,
        "net_band": inputs.net_band,
        "resolution": inputs.resolution,
        "gate_resolution_table": _dataframe_payload(inputs.gate_resolution_table),
        "homography_rows": _dataframe_payload(inputs.homography_rows),
        "landing_error_band_m": inputs.landing_error_band_m,
        "active_corners_refpx": inputs.active_corners_refpx,
    }


def _court_inputs_from_payload(payload: object) -> CourtInputs:
    from annotator.court_evidence import CourtInputs

    record = _object_payload(payload, "court inputs")
    expected = {
        "court_info",
        "gate_court_info",
        "net_band",
        "resolution",
        "gate_resolution_table",
        "homography_rows",
        "landing_error_band_m",
        "active_corners_refpx",
    }
    if set(record) != expected:
        raise ValueError("court inputs payload fields differ")
    gate_payload = _object_payload(record["gate_court_info"], "gate_court_info")
    gate_court_info = {
        video_key: _court_info_from_payload(info, f"gate_court_info.{video_key}")
        for video_key, info in gate_payload.items()
    }
    return CourtInputs(
        court_info=_court_info_from_payload(record["court_info"], "court_info"),
        gate_court_info=gate_court_info,
        net_band=_float_pair(record["net_band"], "net_band"),
        resolution=_float_pair(record["resolution"], "resolution"),
        gate_resolution_table=_dataframe_from_payload(
            record["gate_resolution_table"],
            "gate_resolution_table",
        ),
        homography_rows=_dataframe_from_payload(record["homography_rows"], "homography_rows"),
        landing_error_band_m=_finite_float(
            record["landing_error_band_m"],
            "landing_error_band_m",
        ),
        active_corners_refpx=_float_array(
            record["active_corners_refpx"],
            "active_corners_refpx",
            (4, 2),
        ),
    )


def _dataframe_payload(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "columns": [str(column) for column in frame.columns],
        "index": list(frame.index),
        "data": frame.to_numpy(dtype=object).tolist(),
    }


def _dataframe_from_payload(payload: object, name: str) -> pd.DataFrame:
    import pandas as pd

    record = _object_payload(payload, name)
    if set(record) != {"columns", "index", "data"}:
        raise ValueError(f"{name} table payload fields differ")
    columns = record["columns"]
    index = record["index"]
    data = record["data"]
    if not isinstance(columns, list) or not all(isinstance(column, str) for column in columns):
        raise ValueError(f"{name}.columns must be a list of strings")
    if len(set(columns)) != len(columns):
        raise ValueError(f"{name}.columns must be unique")
    if not isinstance(index, list) or not isinstance(data, list) or len(index) != len(data):
        raise ValueError(f"{name} index and data row counts differ")
    if not all(isinstance(row, list) and len(row) == len(columns) for row in data):
        raise ValueError(f"{name} data rows do not match its columns")
    return pd.DataFrame(data, columns=columns, index=index)


def _court_info_from_payload(payload: object, name: str) -> dict[str, object]:
    record = _object_payload(payload, name)
    required = {"H", "border_L", "border_R", "border_U", "border_D"}
    if not required.issubset(record):
        raise ValueError(f"{name} is missing required court geometry")
    restored = dict(record)
    restored["H"] = _float_array(record["H"], f"{name}.H", (3, 3))
    for key in required.difference({"H"}):
        restored[key] = _finite_float(record[key], f"{name}.{key}")
    return restored


def _raw_cuts_from_payload(payload: object) -> tuple[tuple[int, int], ...]:
    if not isinstance(payload, list):
        raise ValueError("court raw_cuts must be a list")
    intervals: list[tuple[int, int]] = []
    for item in payload:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("each court raw cut must contain start and end")
        start, end = item
        if any(isinstance(value, bool) or not isinstance(value, int) for value in item):
            raise ValueError("court raw cut bounds must be integers")
        intervals.append((start, end))
    return tuple(intervals)


def _object_payload(payload: object, name: str) -> dict[str, object]:
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise ValueError(f"{name} must be an object with string keys")
    return payload


def _float_pair(payload: object, name: str) -> tuple[float, float]:
    if not isinstance(payload, list) or len(payload) != 2:
        raise ValueError(f"{name} must contain two numbers")
    return _finite_float(payload[0], f"{name}[0]"), _finite_float(payload[1], f"{name}[1]")


def _float_array(payload: object, name: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        values = np.asarray(payload, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numbers") from error
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    return values


def _finite_float(payload: object, name: str) -> float:
    if isinstance(payload, bool) or not isinstance(payload, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(payload)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def run_full_annotation_stage(
    *,
    video_id: str,
    metadata: VideoMetadata,
    track: np.ndarray,
    inpaint_fill_mask: np.ndarray,
    guard_codes: np.ndarray,
    pose: PoseArrays,
    court: CourtVision,
    output_dir: Path,
    base: BaseAnnotatorConfig | None = None,
    landing_options: LandingFilterOptions | None = None,
    ref_err_px: float = 3.5,
) -> AnnotationOutput:
    """Run the full annotator with replay masking and persist all primitives."""
    from annotator.config import BaseAnnotatorConfig
    from annotator.point_winner import SHIPPED_LANDING_FILTER_OPTIONS
    from annotator.run_video import RunCapture, run_video
    from annotator.types import DeadMaskMode

    effective_base = BaseAnnotatorConfig() if base is None else base
    if effective_base.dead_mask_mode is not DeadMaskMode.REPLAY:
        raise ValueError(
            "dataset-builder/0.1 requires BaseAnnotatorConfig.dead_mask_mode=DeadMaskMode.REPLAY"
        )
    effective_landing = (
        SHIPPED_LANDING_FILTER_OPTIONS if landing_options is None else landing_options
    )
    validate_shuttle_track(track, metadata.frame_count)
    shuttle_quality = summarize_shuttle_quality(
        track,
        inpaint_fill_mask,
        guard_codes,
        effective_base.rejected_grades,
    )
    validate_pose_arrays(pose, metadata.frame_count)
    _validate_court_vision(
        court.raw_cuts,
        court.evidence,
        metadata.frame_count,
        (float(metadata.width), float(metadata.height)),
    )
    court_inputs = court.evidence.inputs
    if court_inputs is None:
        raise ValueError("detected court evidence has no operational inputs")

    capture = RunCapture()
    cut_frames = [end for _start, end in court.raw_cuts[:-1]]
    result = run_video(
        track,
        pose.bboxes,
        pose.scores,
        pose.kps,
        pose.ndet,
        fps=float(metadata.fps),
        base=effective_base,
        landing_options=effective_landing,
        net_band=court_inputs.net_band,
        resolution=court_inputs.resolution,
        video_id=video_id,
        court_info=court_inputs.court_info,
        homo_df=None,
        gate_court_info=court_inputs.gate_court_info,
        gate_resolution_table=court_inputs.gate_resolution_table,
        ref_err_px=ref_err_px,
        raw_exclusion_mask=None,
        court_present=court.evidence.court_present,
        homography_rows=court_inputs.homography_rows,
        cut_frames=cut_frames,
        keep_vote=court.evidence.keep_vote,
        inpaint_codes=guard_codes,
        court_invalid_is_excluded=True,
        landing_error_band_m=court_inputs.landing_error_band_m,
        capture=capture,
    )
    raw_mask = _validated_mask(
        capture.raw_exclusion_mask,
        metadata.frame_count,
        "captured raw replay mask",
    )
    definitive_mask = _validated_mask(
        capture.definitive_exclusion_mask,
        metadata.frame_count,
        "captured definitive exclusion mask",
    )
    run = AnnotationRun(
        video_id=video_id,
        result=result,
        raw_replay_mask=raw_mask.copy(),
        definitive_exclusion_mask=definitive_mask.copy(),
        shuttle_quality=shuttle_quality,
    )
    artifacts = persist_annotation_run(output_dir, run, metadata.frame_count)
    return AnnotationOutput(run=run, artifacts=artifacts)


def persist_annotation_run(
    output_dir: Path,
    run: AnnotationRun,
    frame_count: int,
) -> AnnotationArtifacts:
    """Persist every primitive result field and both captured masks."""
    raw_mask = _validated_mask(run.raw_replay_mask, frame_count, "raw replay mask")
    definitive_mask = _validated_mask(
        run.definitive_exclusion_mask,
        frame_count,
        "definitive exclusion mask",
    )
    if run.shuttle_quality.frame_count != frame_count:
        raise ValueError("shuttle quality frame count differs from annotation frame count")
    root = Path(output_dir)
    raw_path = save_npy_xz(root / RAW_REPLAY_MASK_FILENAME, raw_mask)
    definitive_path = save_npy_xz(root / DEFINITIVE_EXCLUSION_MASK_FILENAME, definitive_mask)
    result_path = save_json_gz(
        root / ANNOTATOR_RESULT_FILENAME,
        annotation_result_payload(run.video_id, run.result),
    )
    quality_path = save_json_gz(
        root / SHUTTLE_QUALITY_FILENAME,
        run.shuttle_quality.to_payload(),
    )
    return AnnotationArtifacts(
        result=result_path,
        raw_replay_mask=raw_path,
        definitive_exclusion_mask=definitive_path,
        shuttle_quality=quality_path,
    )


def annotation_result_payload(video_id: str, result: AnnotatorResult) -> dict[str, object]:
    """Return a versioned payload containing every ``AnnotatorResult`` field."""
    from annotator.run_video import AnnotatorResult

    if not isinstance(video_id, str) or not video_id:
        raise ValueError("annotation video_id must be a non-empty string")
    if not isinstance(result, AnnotatorResult):
        raise TypeError(f"annotation result must be AnnotatorResult, got {type(result).__name__}")
    primitives = {name: _json_ready(getattr(result, name)) for name in AnnotatorResult._fields}
    if tuple(primitives) != AnnotatorResult._fields:
        raise AssertionError("annotation payload omitted an AnnotatorResult field")
    return {
        "schema": ANNOTATOR_RESULT_SCHEMA,
        "video_id": video_id,
        "result": primitives,
    }


def _validate_court_vision(
    raw_cuts: Sequence[tuple[int, int]],
    evidence: CourtEvidenceResult,
    frame_count: int,
    resolution: tuple[float, float],
) -> None:
    expected_start = 0
    for start, end in raw_cuts:
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (start, end)):
            raise ValueError("raw court cut bounds must be integers")
        if start != expected_start or end <= start or end > frame_count:
            raise ValueError("raw court cuts must tile the canonical frame range")
        expected_start = end
    if not raw_cuts or expected_start != frame_count:
        raise ValueError("raw court cuts must cover the canonical frame range")
    if len(evidence.scene_records) != len(raw_cuts):
        raise ValueError("court scene record count must match raw cuts")
    _validated_mask(evidence.keep_vote, frame_count, "court keep_vote")
    _validated_mask(evidence.court_present, frame_count, "court_present")
    if evidence.inputs is None:
        raise ValueError("court evidence has no operational inputs")
    if tuple(map(float, evidence.inputs.resolution)) != resolution:
        raise ValueError(
            f"court resolution {evidence.inputs.resolution} does not match canonical {resolution}"
        )


def _validated_mask(values: np.ndarray | None, frame_count: int, name: str) -> np.ndarray:
    if values is None:
        raise ValueError(f"{name} was not captured")
    array = np.asarray(values)
    if array.shape != (frame_count,) or array.dtype != np.dtype(bool):
        raise ValueError(
            f"{name} must be a one-dimensional boolean array of length {frame_count}, "
            f"got shape {array.shape} dtype {array.dtype}"
        )
    return np.ascontiguousarray(array)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_ready(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_ready(getattr(value, field.name)) for field in fields(value)}
    if hasattr(value, "_asdict"):
        return {str(key): _json_ready(item) for key, item in value._asdict().items()}
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [_json_ready(item) for item in sorted(value)]
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("structured annotation values must be finite")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot serialise structured annotation value {type(value).__name__}")


def _extract_pose_child(
    *,
    video_path: Path,
    output_dir: Path,
    device: str,
    n_max: int,
) -> int:
    from preparing_data.raw_extract import extract_one_clip
    from preparing_data.rtmlib_pose import RtmlibPoseExtractor

    if not video_path.is_file():
        raise FileNotFoundError(f"pose source video is not a regular file: {video_path}")
    if not 0 < n_max <= 127:
        raise ValueError(f"n_max must be in [1, 127], got {n_max}")
    output_dir.mkdir(parents=True, exist_ok=True)
    extractor = RtmlibPoseExtractor(device=device)
    succeeded = extract_one_clip(
        extractor,
        video_path,
        os.fspath(output_dir / POSE_CHILD_STEM),
        n_max,
        set(),
    )
    if not succeeded:
        raise RuntimeError(f"RTMLib decoded zero frames from {video_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pose_parser = subparsers.add_parser(_POSE_CHILD_COMMAND)
    pose_parser.add_argument("--video", type=Path, required=True)
    pose_parser.add_argument("--output-dir", type=Path, required=True)
    pose_parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    pose_parser.add_argument("--n-max", type=int, required=True)
    arguments = parser.parse_args(argv)
    if arguments.command != _POSE_CHILD_COMMAND:
        parser.error(f"unsupported command: {arguments.command}")
    return _extract_pose_child(
        video_path=arguments.video,
        output_dir=arguments.output_dir,
        device=arguments.device,
        n_max=arguments.n_max,
    )


if __name__ == "__main__":
    raise SystemExit(main())
