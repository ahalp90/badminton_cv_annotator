"""Load a complete baseline menu before any contact labels are read."""

from __future__ import annotations

import hashlib
import json
import lzma
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

import numpy as np

from scratch.contact_det.scripts.freeze_tree_contact_features import REGION_FIELDS
from scratch.contact_det_full_ds_fit.scripts.baseline_config import (
    FIXED_RUN_IDS,
    BaselineConfig,
    BaselineRun,
    MotionMode,
    load_baseline_config,
)
from scratch.contact_det_full_ds_fit.scripts.experiment_config import (
    DevelopmentSplit,
    load_development_split,
    verify_accepted_development_split,
)
from scratch.contact_det_full_ds_fit.scripts.feature_dataset import (
    VerifiedFeatureDataset,
    load_verified_feature_dataset,
)
from scratch.contact_det_full_ds_fit.scripts.run_baseline_menu import (
    MENU_RESULT_SCHEMA,
)
from scratch.contact_det_full_ds_fit.scripts.score_contact_baseline import (
    RESULT_FILE,
    RESULT_SCHEMA,
    SCORE_DTYPE,
    SCORE_FILE,
    predictions_for_settings,
)

MENU_FIELDS = {
    "schema",
    "status",
    "source_commit",
    "config_file",
    "config_sha256",
    "split_file",
    "split_sha256",
    "contact_label_file",
    "contact_label_sha256",
    "feature_records",
    "run_ids",
    "completed_runs",
}
FEATURE_RECORD_FIELDS = {"filename", "sha256"}
COMPLETED_RUN_FIELDS = {"run_id", "result_file", "result_sha256"}
RESULT_FIELDS = {
    "schema",
    "status",
    "run_id",
    "source_commit",
    "config_file",
    "config_sha256",
    "feature_record_file",
    "feature_record_sha256",
    "feature_source_commit",
    "split_file",
    "split_sha256",
    "contact_label_file",
    "contact_label_sha256",
    "training_videos",
    "validation_videos",
    "feature_row_count",
    "candidate_row_count",
    "feature_names",
    "model",
    "training_selection",
    "selected_score_cutoff",
    "selected_duplicate_distance_at_30_fps",
    "selection_metrics",
    "metrics",
    "videos",
    "validation_score_file",
    "validation_score_sha256",
    "validation_score_row_count",
}
MODEL_FIELDS = {"name", "kind", "settings", "class_weight", "class_weight_value"}
IDENTITY_FIELDS = ("fixture", "interval_id", "frame", "fps")
IDENTITY_DTYPE = np.dtype(
    [("fixture", "S16"), ("interval_id", "<i4"), ("frame", "<i4"), ("fps", "<f4")]
)
TRAINING_SELECTION_FIELDS = {
    "positive_radius_at_30_fps",
    "ignored_radius_at_30_fps",
    "nearby_negative_radius_at_30_fps",
    "negative_rule",
    "maximum_negatives_per_positive",
    "selected_row_count",
    "positive_row_count",
    "videos",
}
TRAINING_VIDEO_FIELDS = {"positive", "nearby_negative", "sampled_other_negative", "selected"}
METRIC_FIELDS = {
    "matched",
    "contact_count",
    "prediction_count",
    "precision",
    "recall",
    "f1",
    "first_contact_matched",
    "first_contact_count",
    "first_contact_recall",
    "other_contact_matched",
    "other_contact_count",
    "other_contact_recall",
    "median_absolute_frame_error",
}
VIDEO_RESULT_FIELDS = {"rally_count", "prediction_frames", "metrics"}
REPORT_TOLERANCES = ("5", "10", "15")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SOURCE_COMMIT = re.compile(r"^[0-9a-f]{7,40}$")


@dataclass(frozen=True)
class VerifiedBaselineRun:
    """One checked baseline result and the predictions it produces."""

    run: BaselineRun
    result_path: Path
    result: Mapping[str, Any]
    feature_record_path: Path
    score_rows: np.ndarray
    predictions: Mapping[str, np.ndarray]
    kept: np.ndarray

    @property
    def scores(self) -> np.ndarray:
        """Return the saved score rows."""
        return self.score_rows


@dataclass(frozen=True)
class VerifiedBaselineMenu:
    """The checked menu, feature rows and nine checked baseline results."""

    menu_path: Path
    menu: Mapping[str, Any]
    config: BaselineConfig
    split: DevelopmentSplit
    raw_features: VerifiedFeatureDataset
    runs: tuple[VerifiedBaselineRun, ...]

    @property
    def results(self) -> tuple[VerifiedBaselineRun, ...]:
        """Return the checked runs in the written menu order."""
        return self.runs

    @property
    def runs_by_id(self) -> Mapping[str, VerifiedBaselineRun]:
        """Return the checked runs by their fixed run ID."""
        return MappingProxyType({item.run.run_id: item for item in self.runs})


@dataclass(frozen=True)
class ValidationCandidates:
    """Small identity rows for validation candidates and the full candidate count."""

    identities: np.ndarray
    full_count: int


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be an object with string keys")
    return value


def _read_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(_mapping(value, label))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_fields(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields differ")


def _check_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a SHA-256 file hash")
    return value


def _check_source_commit(value: object, label: str) -> str:
    if not isinstance(value, str) or SOURCE_COMMIT.fullmatch(value) is None:
        raise ValueError(f"{label} is invalid")
    return value


def _check_filename(value: object, expected: str, label: str) -> None:
    if not isinstance(value, str) or PurePosixPath(value).name != value:
        raise ValueError(f"{label} must be a filename")
    if value != expected:
        raise ValueError(f"{label} differs")


def _check_file_hash(path: Path, saved_hash: object, label: str) -> None:
    expected_hash = _check_sha256(saved_hash, label)
    if _sha256(path) != expected_hash:
        raise ValueError(f"{label} differs from the file")


def _candidate_mask(rows: np.ndarray) -> np.ndarray:
    selected = np.zeros(len(rows), dtype=bool)
    for field in REGION_FIELDS:
        selected |= rows[field].astype(bool)
    return selected


def _validation_candidates(features: VerifiedFeatureDataset) -> ValidationCandidates:
    chunks: list[np.ndarray] = []
    full_count = 0
    validation_names = {video.fixture for video in features.split.validation_videos}
    for video in features.split.videos:
        start, end = features.video_ranges[video.fixture]
        video_rows = features.rows[start:end]
        selected = _candidate_mask(video_rows)
        full_count += int(selected.sum())
        if video.fixture not in validation_names:
            continue
        chosen = video_rows[selected]
        identities = np.empty(len(chosen), dtype=IDENTITY_DTYPE)
        for field in IDENTITY_FIELDS:
            identities[field] = chosen[field]
        chunks.append(identities)
    return ValidationCandidates(np.concatenate(chunks), full_count)


def _check_candidate_identities(
    raw_candidates: ValidationCandidates,
    common30_candidates: ValidationCandidates,
) -> None:
    raw_rows = raw_candidates.identities
    common30_rows = common30_candidates.identities
    if raw_candidates.full_count != common30_candidates.full_count or len(raw_rows) != len(
        common30_rows
    ):
        raise ValueError("raw and common30 validation candidates differ")
    for field in IDENTITY_FIELDS:
        if not np.array_equal(raw_rows[field], common30_rows[field]):
            raise ValueError("raw and common30 validation candidates differ")


def _check_score_identity(
    scores: np.ndarray,
    expected: np.ndarray,
    run_id: str,
) -> None:
    if len(scores) != len(expected):
        raise ValueError(f"{run_id}: score rows differ from validation candidates")
    for field in IDENTITY_FIELDS:
        if not np.array_equal(scores[field], expected[field]):
            raise ValueError(f"{run_id}: score identities differ from validation candidates")


def _load_scores(path: Path, run_id: str) -> np.ndarray:
    with lzma.open(path, "rb") as source:
        scores = np.load(source, allow_pickle=False)
    if not isinstance(scores, np.ndarray) or scores.ndim != 1 or scores.dtype != SCORE_DTYPE:
        raise ValueError(f"{run_id}: saved scores have the wrong fields")
    if not np.all(np.isfinite(scores["contact_score"])) or np.any(
        (scores["contact_score"] < 0.0) | (scores["contact_score"] > 1.0)
    ):
        raise ValueError(f"{run_id}: saved contact scores must be finite values from 0 to 1")
    return scores


def _check_number_in(value: object, choices: tuple[float, ...], label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} is not a number")
    number = float(value)
    if number not in choices:
        raise ValueError(f"{label} is not in the fixed menu")
    return number


def _check_distance_in(value: object, choices: tuple[int, ...], label: str) -> int:
    if type(value) is not int or value not in choices:
        raise ValueError(f"{label} is not in the fixed menu")
    return value


def _check_model(result: Mapping[str, Any], run: BaselineRun) -> None:
    model = _mapping(result["model"], f"{run.run_id}: model")
    _check_fields(model, MODEL_FIELDS, f"{run.run_id}: model")
    if model["name"] != run.model_name or model["kind"] != run.model_kind.value:
        raise ValueError(f"{run.run_id}: model differs from the fixed menu")
    settings = _mapping(model["settings"], f"{run.run_id}: model settings")
    if dict(settings) != dict(run.model_settings):
        raise ValueError(f"{run.run_id}: model settings differ from the fixed menu")
    if model["class_weight"] != run.class_weight or model["class_weight_value"] != run.class_weight_value:
        raise ValueError(f"{run.run_id}: class weight differs from the fixed menu")


def _check_video_lists(result: Mapping[str, Any], split: DevelopmentSplit, run_id: str) -> None:
    training_names = [video.fixture for video in split.training_videos]
    validation_names = [video.fixture for video in split.validation_videos]
    if result["training_videos"] != training_names:
        raise ValueError(f"{run_id}: training videos differ")
    if result["validation_videos"] != validation_names:
        raise ValueError(f"{run_id}: validation videos differ")


def _nonnegative_integer(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _check_training_selection(
    result: Mapping[str, Any],
    split: DevelopmentSplit,
    config: BaselineConfig,
    run: BaselineRun,
) -> None:
    run_id = run.run_id
    selection = _mapping(result["training_selection"], f"{run_id}: training selection")
    _check_fields(selection, TRAINING_SELECTION_FIELDS, f"{run_id}: training selection")
    expected_settings = {
        "positive_radius_at_30_fps": config.positive_radius_at_30_fps,
        "ignored_radius_at_30_fps": config.ignored_radius_at_30_fps,
        "nearby_negative_radius_at_30_fps": config.hard_negative_radius_at_30_fps,
        "negative_rule": run.negative_rule,
        "maximum_negatives_per_positive": run.negative_limit,
    }
    if any(selection[field] != value for field, value in expected_settings.items()):
        raise ValueError(f"{run_id}: training selection settings differ")
    videos = _mapping(selection.get("videos"), f"{run_id}: training selection videos")
    expected = {video.fixture for video in split.training_videos}
    if set(videos) != expected:
        raise ValueError(f"{run_id}: training selection videos differ")
    selected_total = 0
    positive_total = 0
    for video_name, raw_counts in videos.items():
        counts = _mapping(raw_counts, f"{run_id}: {video_name} training counts")
        _check_fields(counts, TRAINING_VIDEO_FIELDS, f"{run_id}: {video_name} training counts")
        positive = _nonnegative_integer(counts["positive"], f"{run_id}: {video_name} positive")
        nearby = _nonnegative_integer(
            counts["nearby_negative"], f"{run_id}: {video_name} nearby negative"
        )
        sampled = _nonnegative_integer(
            counts["sampled_other_negative"],
            f"{run_id}: {video_name} sampled other negative",
        )
        selected = _nonnegative_integer(counts["selected"], f"{run_id}: {video_name} selected")
        if selected != positive + nearby + sampled:
            raise ValueError(f"{run_id}: {video_name} training counts do not add up")
        positive_total += positive
        selected_total += selected
    if positive_total <= 0 or selected_total <= positive_total:
        raise ValueError(f"{run_id}: training totals must contain positive and negative rows")
    if selection["positive_row_count"] != positive_total or selection["selected_row_count"] != selected_total:
        raise ValueError(f"{run_id}: training totals differ from the per-video counts")


def _check_rate(value: object, expected: float, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    number = float(value)
    if not np.isfinite(number) or not np.isclose(number, expected, rtol=0.0, atol=1e-12):
        raise ValueError(f"{label} differs from its saved counts")


def _check_metrics(value: object, label: str) -> Mapping[str, Any]:
    metrics = _mapping(value, label)
    _check_fields(metrics, METRIC_FIELDS, label)
    matched = _nonnegative_integer(metrics["matched"], f"{label}.matched")
    contact_count = _nonnegative_integer(metrics["contact_count"], f"{label}.contact_count")
    prediction_count = _nonnegative_integer(
        metrics["prediction_count"], f"{label}.prediction_count"
    )
    first_matched = _nonnegative_integer(
        metrics["first_contact_matched"], f"{label}.first_contact_matched"
    )
    first_count = _nonnegative_integer(
        metrics["first_contact_count"], f"{label}.first_contact_count"
    )
    other_matched = _nonnegative_integer(
        metrics["other_contact_matched"], f"{label}.other_contact_matched"
    )
    other_count = _nonnegative_integer(
        metrics["other_contact_count"], f"{label}.other_contact_count"
    )
    if matched > min(contact_count, prediction_count):
        raise ValueError(f"{label}: matched count is too large")
    if first_count + other_count != contact_count or first_matched + other_matched != matched:
        raise ValueError(f"{label}: contact counts do not add up")
    if first_matched > first_count or other_matched > other_count:
        raise ValueError(f"{label}: matched contact count is too large")
    precision = matched / prediction_count if prediction_count else 0.0
    recall = matched / contact_count if contact_count else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    _check_rate(metrics["precision"], precision, f"{label}.precision")
    _check_rate(metrics["recall"], recall, f"{label}.recall")
    _check_rate(metrics["f1"], f1, f"{label}.f1")
    _check_rate(
        metrics["first_contact_recall"],
        first_matched / first_count if first_count else 0.0,
        f"{label}.first_contact_recall",
    )
    _check_rate(
        metrics["other_contact_recall"],
        other_matched / other_count if other_count else 0.0,
        f"{label}.other_contact_recall",
    )
    median_error = metrics["median_absolute_frame_error"]
    if median_error is not None and (
        isinstance(median_error, bool)
        or not isinstance(median_error, (int, float))
        or not np.isfinite(float(median_error))
        or float(median_error) < 0.0
    ):
        raise ValueError(f"{label}.median_absolute_frame_error is invalid")
    return metrics


def _check_prediction_frames(
    result: Mapping[str, Any],
    predictions: Mapping[str, np.ndarray],
    split: DevelopmentSplit,
    run_id: str,
) -> None:
    raw_videos = _mapping(result["videos"], f"{run_id}: videos")
    expected_names = {video.fixture for video in split.validation_videos}
    if set(raw_videos) != expected_names:
        raise ValueError(f"{run_id}: result videos differ")
    for video in split.validation_videos:
        video_result = _mapping(raw_videos[video.fixture], f"{run_id}: {video.fixture}")
        _check_fields(video_result, VIDEO_RESULT_FIELDS, f"{run_id}: {video.fixture}")
        if type(video_result["rally_count"]) is not int or video_result["rally_count"] <= 0:
            raise ValueError(f"{run_id}: {video.fixture} rally count must be positive")
        raw_frames = video_result.get("prediction_frames")
        if not isinstance(raw_frames, list) or any(type(frame) is not int for frame in raw_frames):
            raise ValueError(f"{run_id}: {video.fixture} prediction frames are invalid")
        saved_frames = np.asarray(raw_frames, dtype=np.int32)
        if not np.array_equal(saved_frames, predictions[video.fixture]):
            raise ValueError(f"{run_id}: {video.fixture} prediction frames differ")


def _check_result_metrics(
    result: Mapping[str, Any],
    split: DevelopmentSplit,
    predictions: Mapping[str, np.ndarray],
    config: BaselineConfig,
    run_id: str,
) -> None:
    raw_metrics = _mapping(result["metrics"], f"{run_id}: metrics")
    if set(raw_metrics) != set(REPORT_TOLERANCES):
        raise ValueError(f"{run_id}: metric tolerances differ")
    aggregate = {
        tolerance: _check_metrics(raw_metrics[tolerance], f"{run_id}: metrics at {tolerance}")
        for tolerance in REPORT_TOLERANCES
    }
    selection_metrics = _check_metrics(result["selection_metrics"], f"{run_id}: selection metrics")
    selected_tolerance = str(config.timing_tolerance_at_30_fps)
    if dict(selection_metrics) != dict(aggregate[selected_tolerance]):
        raise ValueError(f"{run_id}: selection metrics differ from the selected tolerance")

    raw_videos = _mapping(result["videos"], f"{run_id}: videos")
    count_fields = (
        "matched",
        "contact_count",
        "prediction_count",
        "first_contact_matched",
        "first_contact_count",
        "other_contact_matched",
        "other_contact_count",
    )
    for tolerance in REPORT_TOLERANCES:
        totals = {field: 0 for field in count_fields}
        for video in split.validation_videos:
            video_result = _mapping(raw_videos[video.fixture], f"{run_id}: {video.fixture}")
            video_metrics_by_tolerance = _mapping(
                video_result["metrics"], f"{run_id}: {video.fixture} metrics"
            )
            if set(video_metrics_by_tolerance) != set(REPORT_TOLERANCES):
                raise ValueError(f"{run_id}: {video.fixture} metric tolerances differ")
            metrics = _check_metrics(
                video_metrics_by_tolerance[tolerance],
                f"{run_id}: {video.fixture} metrics at {tolerance}",
            )
            if metrics["prediction_count"] != len(predictions[video.fixture]):
                raise ValueError(f"{run_id}: {video.fixture} prediction count differs")
            for field in count_fields:
                totals[field] += int(metrics[field])
        if any(aggregate[tolerance][field] != total for field, total in totals.items()):
            raise ValueError(f"{run_id}: aggregate metrics at {tolerance} do not add up")


def _check_same_feature_sources(
    raw_features: VerifiedFeatureDataset,
    common30_features: VerifiedFeatureDataset,
) -> None:
    if raw_features.record.get("source_commit") != common30_features.record.get("source_commit"):
        raise ValueError("raw and common30 feature source commits differ")
    raw_videos = raw_features.record.get("videos")
    common30_videos = common30_features.record.get("videos")
    if not isinstance(raw_videos, list) or not isinstance(common30_videos, list):
        raise TypeError("feature video records must be lists")
    if len(raw_videos) != len(common30_videos):
        raise ValueError("raw and common30 feature video records differ")
    for raw_video, common30_video in zip(raw_videos, common30_videos, strict=True):
        raw_record = _mapping(raw_video, "raw feature video record")
        common30_record = _mapping(common30_video, "common30 feature video record")
        if raw_record.get("video") != common30_record.get("video") or raw_record.get(
            "input_files"
        ) != common30_record.get("input_files"):
            raise ValueError("raw and common30 feature input files differ")


def _feature_record_for_run(
    run: BaselineRun,
    raw_path: Path,
    common30_path: Path,
    raw_features: VerifiedFeatureDataset,
    common30_features: VerifiedFeatureDataset,
) -> tuple[Path, VerifiedFeatureDataset]:
    if run.motion_mode is MotionMode.RAW_PER_FRAME:
        return raw_path, raw_features
    if run.motion_mode is MotionMode.BASE30_PER_FRAME:
        return common30_path, common30_features
    raise ValueError(f"{run.run_id}: motion mode is not supported")


def _check_child_result(
    result: Mapping[str, Any],
    run: BaselineRun,
    config: BaselineConfig,
    source_commit: str,
    config_path: Path,
    config_hash: str,
    split_path: Path,
    split_hash: str,
    label_path: Path,
    label_hash: str,
    feature_path: Path,
    feature_hash: str,
    features: VerifiedFeatureDataset,
    candidates: ValidationCandidates,
    split: DevelopmentSplit,
    score_path: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    _check_fields(result, RESULT_FIELDS, f"{run.run_id}: result")
    if result["schema"] != RESULT_SCHEMA or result["status"] != "complete":
        raise ValueError(f"{run.run_id}: result is not complete")
    if result["run_id"] != run.run_id:
        raise ValueError(f"{run.run_id}: result run ID differs")
    if _check_source_commit(result["source_commit"], f"{run.run_id}: source commit") != source_commit:
        raise ValueError(f"{run.run_id}: source commit differs")

    _check_filename(result["config_file"], config_path.name, f"{run.run_id}: config file")
    _check_file_hash(config_path, result["config_sha256"], f"{run.run_id}: config hash")
    _check_filename(result["split_file"], split_path.name, f"{run.run_id}: split file")
    _check_file_hash(split_path, result["split_sha256"], f"{run.run_id}: split hash")
    _check_filename(result["contact_label_file"], label_path.name, f"{run.run_id}: label file")
    _check_file_hash(label_path, result["contact_label_sha256"], f"{run.run_id}: label hash")
    _check_filename(result["feature_record_file"], feature_path.name, f"{run.run_id}: feature record")
    _check_file_hash(feature_path, result["feature_record_sha256"], f"{run.run_id}: feature record hash")
    if result["feature_source_commit"] != features.record["source_commit"]:
        raise ValueError(f"{run.run_id}: feature source commit differs")

    _check_video_lists(result, split, run.run_id)
    if result["feature_row_count"] != len(features.rows):
        raise ValueError(f"{run.run_id}: feature row count differs")
    if result["candidate_row_count"] != candidates.full_count:
        raise ValueError(f"{run.run_id}: candidate row count differs")
    if result["feature_names"] != list(features.model_input_fields):
        raise ValueError(f"{run.run_id}: feature names differ")
    _check_model(result, run)
    _check_training_selection(result, split, config, run)

    cutoff = _check_number_in(
        result["selected_score_cutoff"],
        tuple(config.score_cutoffs),
        f"{run.run_id}: score cutoff",
    )
    distance = _check_distance_in(
        result["selected_duplicate_distance_at_30_fps"],
        tuple(config.duplicate_distances_at_30_fps),
        f"{run.run_id}: duplicate distance",
    )
    _check_filename(result["validation_score_file"], SCORE_FILE, f"{run.run_id}: score file")
    _check_file_hash(score_path, result["validation_score_sha256"], f"{run.run_id}: score hash")
    scores = _load_scores(score_path, run.run_id)
    if result["validation_score_row_count"] != len(scores):
        raise ValueError(f"{run.run_id}: score row count differs")

    expected_scores = candidates.identities
    _check_score_identity(scores, expected_scores, run.run_id)
    predictions, kept = predictions_for_settings(
        scores,
        split.validation_videos,
        cutoff,
        distance,
    )
    if not np.array_equal(scores["kept"], kept):
        raise ValueError(f"{run.run_id}: saved kept flags differ")
    _check_prediction_frames(result, predictions, split, run.run_id)
    _check_result_metrics(result, split, predictions, config, run.run_id)
    return scores, predictions, kept


def load_completed_baseline_menu(
    menu_result_path: Path,
    config_path: Path,
    split_path: Path,
    raw_feature_record_path: Path,
    common30_feature_record_path: Path,
    shots_master_path: Path,
) -> VerifiedBaselineMenu:
    """Load and check all nine results without reading contact labels."""
    menu_file = Path(menu_result_path)
    config_file = Path(config_path)
    split_file = Path(split_path)
    raw_feature_file = Path(raw_feature_record_path)
    common30_feature_file = Path(common30_feature_record_path)
    label_file = Path(shots_master_path)

    menu = _read_object(menu_file, "menu result")
    _check_fields(menu, MENU_FIELDS, "menu result")
    if menu["schema"] != MENU_RESULT_SCHEMA or menu["status"] != "complete":
        raise ValueError("menu result is not complete")
    source_commit = _check_source_commit(menu["source_commit"], "menu source commit")

    _check_filename(menu["config_file"], config_file.name, "menu config file")
    config_hash = _sha256(config_file)
    if menu["config_sha256"] != config_hash:
        raise ValueError("menu config hash differs")
    _check_filename(menu["split_file"], split_file.name, "menu split file")
    split_hash = _sha256(split_file)
    if menu["split_sha256"] != split_hash:
        raise ValueError("menu split hash differs")
    _check_filename(menu["contact_label_file"], label_file.name, "menu label file")
    label_hash = _check_sha256(menu["contact_label_sha256"], "menu label hash")
    if _sha256(label_file) != label_hash:
        raise ValueError("menu label hash differs")

    feature_records = _mapping(menu["feature_records"], "menu feature records")
    expected_feature_records = {
        MotionMode.RAW_PER_FRAME.value: raw_feature_file,
        MotionMode.BASE30_PER_FRAME.value: common30_feature_file,
    }
    if set(feature_records) != set(expected_feature_records):
        raise ValueError("menu feature records differ")
    feature_hashes: dict[str, str] = {}
    for motion_mode, feature_file in expected_feature_records.items():
        record = _mapping(feature_records[motion_mode], f"menu {motion_mode} feature record")
        _check_fields(record, FEATURE_RECORD_FIELDS, f"menu {motion_mode} feature record")
        _check_filename(record["filename"], feature_file.name, f"menu {motion_mode} feature filename")
        feature_hash = _check_sha256(record["sha256"], f"menu {motion_mode} feature hash")
        if _sha256(feature_file) != feature_hash:
            raise ValueError(f"menu {motion_mode} feature hash differs")
        feature_hashes[motion_mode] = feature_hash

    if menu["run_ids"] != list(FIXED_RUN_IDS):
        raise ValueError("menu run IDs differ")
    completed = menu["completed_runs"]
    if not isinstance(completed, list) or len(completed) != len(FIXED_RUN_IDS):
        raise ValueError("menu completed runs differ")
    completed_paths: list[tuple[str, Path]] = []
    for expected_run_id, raw_completed in zip(FIXED_RUN_IDS, completed, strict=True):
        entry = _mapping(raw_completed, f"menu {expected_run_id} completion")
        _check_fields(entry, COMPLETED_RUN_FIELDS, f"menu {expected_run_id} completion")
        if entry["run_id"] != expected_run_id:
            raise ValueError("menu completed run order differs")
        expected_relative = PurePosixPath(expected_run_id) / RESULT_FILE
        if entry["result_file"] != expected_relative.as_posix():
            raise ValueError(f"{expected_run_id}: result path differs")
        result_hash = _check_sha256(entry["result_sha256"], f"{expected_run_id}: result hash")
        result_path = menu_file.parent.joinpath(*expected_relative.parts)
        if _sha256(result_path) != result_hash:
            raise ValueError(f"{expected_run_id}: result hash differs")
        completed_paths.append((expected_run_id, result_path))

    config = load_baseline_config(config_file)
    split = load_development_split(split_file)
    verify_accepted_development_split(split)
    raw_features = load_verified_feature_dataset(
        raw_feature_file,
        split_file,
        MotionMode.RAW_PER_FRAME.value,
    )
    common30_features = load_verified_feature_dataset(
        common30_feature_file,
        split_file,
        MotionMode.BASE30_PER_FRAME.value,
    )
    _check_same_feature_sources(raw_features, common30_features)
    raw_candidates = _validation_candidates(raw_features)
    common30_candidates = _validation_candidates(common30_features)
    _check_candidate_identities(raw_candidates, common30_candidates)

    verified_runs: list[VerifiedBaselineRun] = []
    for expected_run_id, result_path in completed_paths:
        run = next(item for item in config.runs if item.run_id == expected_run_id)
        feature_path, features = _feature_record_for_run(
            run,
            raw_feature_file,
            common30_feature_file,
            raw_features,
            common30_features,
        )
        candidates = raw_candidates if run.motion_mode is MotionMode.RAW_PER_FRAME else common30_candidates
        result = _read_object(result_path, f"{run.run_id} result")
        score_path = result_path.parent / SCORE_FILE
        scores, predictions, kept = _check_child_result(
            result,
            run,
            config,
            source_commit,
            config_file,
            config_hash,
            split_file,
            split_hash,
            label_file,
            label_hash,
            feature_path,
            feature_hashes[run.motion_mode.value],
            features,
            candidates,
            split,
            score_path,
        )
        verified_runs.append(
            VerifiedBaselineRun(
                run=run,
                result_path=result_path,
                result=MappingProxyType(result),
                feature_record_path=feature_path,
                score_rows=scores,
                predictions=MappingProxyType(predictions),
                kept=kept,
            )
        )

    return VerifiedBaselineMenu(
        menu_path=menu_file,
        menu=MappingProxyType(menu),
        config=config,
        split=split,
        raw_features=raw_features,
        runs=tuple(verified_runs),
    )
