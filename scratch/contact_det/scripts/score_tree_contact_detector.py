"""Fit and score tree contact baselines from a verified label-blind freeze."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import lzma
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from freeze_contact_evidence import FIXTURE_SPECS
from freeze_tree_contact_features import (
    FEATURE_FILENAME,
    FEATURE_SCHEMA,
    IDENTITY_FIELDS,
    MANIFEST_SCHEMA,
    MOTION_MODES,
    REGION_FIELDS,
    WINDOW_OFFSETS_BASE30,
    _feature_family_names,
    _write_npy_xz,
)

RESULTS_SCHEMA = "tree-contact-results/3"
CANDIDATE_SCORES_SCHEMA = "tree-contact-candidate-scores/1"
CANDIDATE_VARIANT = "region_v2/histogram_boosting/physics"
PHYSICS_WITHOUT_RAW_MOTION_VARIANT = "region_v2/histogram_boosting/physics_without_raw_motion"
PHYSICS_BASE30_MOTION_VARIANT = "region_v2/histogram_boosting/physics_base30_motion"
ALLOWED_VARIANTS = (
    CANDIDATE_VARIANT,
    PHYSICS_WITHOUT_RAW_MOTION_VARIANT,
    PHYSICS_BASE30_MOTION_VARIANT,
)
RAW_MOTION_MODE = "raw"
RAW_PER_FRAME_MOTION_MODE = "raw_per_frame"
BASE30_PER_FRAME_MOTION_MODE = "base30_per_frame"
ALLOWED_EXPLICIT_MOTION_MODES = tuple(MOTION_MODES)
FRAME_RATE_SENSITIVE_SIGNALS = (
    "shuttle_vx",
    "shuttle_vy",
    "shuttle_speed",
    "shuttle_impulse",
    "ankle_speed_top",
    "ankle_speed_bot",
)
TOLERANCES_BASE30 = (5, 10, 15)
POSITIVE_RADIUS_BASE30 = 1
IGNORE_RADIUS_BASE30 = 4
HARD_NEGATIVE_RADIUS_BASE30 = 15
NMS_RADIUS_BASE30 = 5
RANDOM_SEED = 20260824
# Easy negatives fill towards this total ratio. All hard negatives survive even
# if a future dataset contains more than the target.
MAX_NEGATIVE_RATIO = 12
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")

FEATURE_SETS = {
    "physics": ("physics", "missingness"),
    "physics_context": ("physics", "context", "missingness"),
    "context_only": ("context",),
    "missingness_only": ("missingness",),
}

CANDIDATE_BELOW_THRESHOLD = 0
CANDIDATE_NEARBY_DUPLICATE = 1
CANDIDATE_RETAINED = 2
CANDIDATE_DECISIONS = {
    "below_threshold": CANDIDATE_BELOW_THRESHOLD,
    "nearby_duplicate": CANDIDATE_NEARBY_DUPLICATE,
    "retained": CANDIDATE_RETAINED,
}
CANDIDATE_SCORE_DTYPE = np.dtype(
    [
        ("fixture", "S7"),
        ("interval_id", "<i2"),
        ("frame", "<i4"),
        ("timing_score", "<f8"),
        ("threshold", "<f8"),
        ("decision", "u1"),
    ]
)


@dataclass(frozen=True)
class VerifiedFeatures:
    """A verified label-blind feature table and its manifest."""

    manifest_path: Path
    manifest: dict[str, Any]
    rows: np.ndarray


@dataclass(frozen=True)
class GroundTruth:
    """Frame identities needed for labels and event scoring."""

    frames: dict[str, np.ndarray]
    serves: dict[str, set[int]]
    rally_count: int


@dataclass(frozen=True)
class VerifiedCandidateScores:
    """Held-out scores tied to one verified feature freeze and tree result."""

    manifest_path: Path
    manifest: dict[str, Any]
    rows: np.ndarray
    tree_result: dict[str, Any]


def _read_json_object(path: Path) -> dict[str, Any]:
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as source:
            value = json.load(source)
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_filename(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty filename")
    path = Path(value)
    if path.is_absolute() or len(path.parts) != 1 or path.name != value:
        raise ValueError(f"{name} must not contain a directory")
    return value


def _feature_families(value: object) -> dict[str, list[str]]:
    if not isinstance(value, Mapping) or set(value) != {"physics", "context", "missingness"}:
        raise ValueError("manifest feature families differ")
    parsed: dict[str, list[str]] = {}
    seen: set[str] = set()
    for family in ("physics", "context", "missingness"):
        names = value[family]
        if not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names):
            raise TypeError(f"manifest feature family {family} is malformed")
        duplicate_elsewhere = (set(names) - set(REGION_FIELDS)) & seen
        if len(names) != len(set(names)) or duplicate_elsewhere:
            raise ValueError(f"manifest feature family {family} has duplicate fields")
        parsed[family] = list(names)
        seen.update(names)
    return parsed


def _manifest_motion_mode(manifest: Mapping[str, Any]) -> str:
    """Return the freeze motion mode, retaining the legacy raw default."""
    if "motion_mode" not in manifest:
        return RAW_MOTION_MODE
    value = manifest["motion_mode"]
    if not isinstance(value, str) or value not in ALLOWED_EXPLICIT_MOTION_MODES:
        raise ValueError("feature manifest motion_mode differs")
    return value


def _validate_variant(variant: object) -> str:
    if not isinstance(variant, str) or variant not in ALLOWED_VARIANTS:
        raise ValueError(f"unknown tree contact variant: {variant!r}")
    return variant


def _validate_variant_for_manifest(manifest: Mapping[str, Any], variant: object) -> str:
    selected = _validate_variant(variant)
    motion_mode = _manifest_motion_mode(manifest)
    if selected == CANDIDATE_VARIANT and motion_mode not in {
        RAW_MOTION_MODE,
        RAW_PER_FRAME_MOTION_MODE,
    }:
        raise ValueError("baseline tree contact variant requires raw motion features")
    if selected == PHYSICS_WITHOUT_RAW_MOTION_VARIANT and motion_mode not in {
        RAW_MOTION_MODE,
        RAW_PER_FRAME_MOTION_MODE,
    }:
        raise ValueError("no-motion tree contact variant requires raw motion features")
    if selected == PHYSICS_BASE30_MOTION_VARIANT and motion_mode != BASE30_PER_FRAME_MOTION_MODE:
        raise ValueError("base30 tree contact variant requires base30_per_frame motion features")
    return selected


def _manifest_intervals(manifest: Mapping[str, Any], field: str) -> dict[str, tuple[tuple[int, int], ...]]:
    summaries = manifest.get("fixtures")
    if not isinstance(summaries, list):
        raise TypeError("feature fixture summaries are malformed")
    by_fixture: dict[str, tuple[tuple[int, int], ...]] = {}
    for summary in summaries:
        if not isinstance(summary, Mapping) or summary.get("fixture") not in FIXTURE_SPECS:
            raise ValueError("feature fixture summary is malformed")
        fixture = str(summary["fixture"])
        frame_count = summary.get("frame_count")
        raw_intervals = summary.get(field)
        if not isinstance(frame_count, int) or not isinstance(raw_intervals, list):
            raise TypeError(f"feature fixture {field} is malformed")
        intervals: list[tuple[int, int]] = []
        for value in raw_intervals:
            if not isinstance(value, list) or len(value) != 2 or not all(isinstance(bound, int) for bound in value):
                raise ValueError(f"feature fixture {field} is malformed")
            start, end = value
            if not 0 <= start < end <= frame_count:
                raise ValueError(f"feature fixture {field} lies outside the timeline")
            if intervals and start < intervals[-1][1]:
                raise ValueError(f"feature fixture {field} overlap")
            intervals.append((start, end))
        by_fixture[fixture] = tuple(intervals)
    if set(by_fixture) != set(FIXTURE_SPECS):
        raise ValueError("feature fixture summaries differ")
    return by_fixture


def verify_freeze(manifest_path: Path) -> VerifiedFeatures:
    """Verify the label-blind manifest and table before any GT import."""
    manifest_path = Path(manifest_path)
    manifest = _read_json_object(manifest_path)
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError("feature manifest schema differs")
    if manifest.get("labels_read") is not False:
        raise ValueError("feature manifest does not prove a label-blind freeze")
    if manifest.get("row_domain") != "eligible tracker intervals plus 45-base-30 serve pre-roll":
        raise ValueError("feature row domain differs")
    if manifest.get("model_search_surface") != "seeded region union":
        raise ValueError("model search surface differs")
    _manifest_motion_mode(manifest)
    if manifest.get("fixture_set") != list(FIXTURE_SPECS):
        raise ValueError("feature fixture set differs")
    filename = _relative_filename(manifest.get("feature_file"), "manifest.feature_file")
    expected_sha = manifest.get("feature_sha256")
    if not isinstance(expected_sha, str) or HEX_SHA256.fullmatch(expected_sha) is None:
        raise ValueError("feature SHA-256 is malformed")
    feature_path = manifest_path.parent / filename
    if feature_path.name != FEATURE_FILENAME or _sha256(feature_path) != expected_sha:
        raise ValueError("feature file identity or SHA-256 differs")
    families = _feature_families(manifest.get("feature_families"))
    if families != _feature_family_names():
        raise ValueError("feature families differ from the producer contract")
    if manifest.get("identity_fields") != list(IDENTITY_FIELDS):
        raise ValueError("feature identity fields differ")
    if manifest.get("region_fields") != list(REGION_FIELDS):
        raise ValueError("feature region fields differ")
    _manifest_intervals(manifest, "tracker_intervals")
    _manifest_intervals(manifest, "eligible_intervals")
    search_intervals = _manifest_intervals(manifest, "search_intervals")

    with lzma.open(feature_path, "rb") as source:
        rows = np.load(source, allow_pickle=False)
    if rows.ndim != 1 or rows.dtype.names is None:
        raise ValueError("feature table must be a one-dimensional structured array")
    required = set(IDENTITY_FIELDS) | set(REGION_FIELDS)
    for names in families.values():
        required.update(names)
    if set(rows.dtype.names) != required:
        raise ValueError("feature table fields differ from the manifest")
    if len(rows) != manifest.get("row_count"):
        raise ValueError("feature row count differs")
    fixtures = np.char.decode(rows["fixture"], "ascii")
    if set(fixtures) != set(FIXTURE_SPECS):
        raise ValueError("feature rows contain an unexpected fixture")
    for fixture, (_video_id, fps) in FIXTURE_SPECS.items():
        fixture_rows = rows[fixtures == fixture]
        if not np.all(fixture_rows["fps"] == fps):
            raise ValueError(f"{fixture}: feature row fps differs")
        intervals = search_intervals[fixture]
        expected_row_count = sum(end - start for start, end in intervals)
        if len(fixture_rows) != expected_row_count:
            raise ValueError(f"{fixture}: feature rows do not cover the search intervals")
        for interval_id, (start, end) in enumerate(intervals):
            interval_rows = fixture_rows[fixture_rows["interval_id"] == interval_id]
            if not np.array_equal(interval_rows["frame"], np.arange(start, end, dtype=np.int32)):
                raise ValueError(f"{fixture}: feature rows differ from search interval {interval_id}")
    identities = np.rec.fromarrays([fixtures, rows["interval_id"], rows["frame"]])
    if len(np.unique(identities)) != len(rows):
        raise ValueError("feature frame identities are duplicated")
    return VerifiedFeatures(manifest_path, manifest, rows)


def _load_ground_truth() -> GroundTruth:
    """Load timing labels only after feature verification has succeeded."""
    import pandas as pd

    from annotator.calibration.fixtures import REPO_ROOT as CALIBRATION_ROOT
    from annotator.calibration.fixtures import SHARED_FILES, verify_file
    from annotator.calibration.scoring import load_gt_rallies

    master_pin = next(pin for pin in SHARED_FILES if pin.path.name == "shots_master.csv")
    verify_file(master_pin)
    master = pd.read_csv(
        CALIBRATION_ROOT / master_pin.path,
        usecols=["vid", "set_id", "rally", "frame_num"],
    )
    frames: dict[str, np.ndarray] = {}
    serves: dict[str, set[int]] = {}
    rally_count = 0
    for fixture, (video_id, _fps) in FIXTURE_SPECS.items():
        rallies = load_gt_rallies(master, video_id)
        rally_count += len(rallies)
        frames[fixture] = np.asarray(
            [frame for rally in rallies for frame in rally.stroke_frames],
            dtype=np.int32,
        )
        serves[fixture] = {rally.stroke_frames[0] for rally in rallies}
    if rally_count != 292 or sum(len(values) for values in frames.values()) != 3128:
        raise ValueError("ground-truth fixture totals differ from the pinned three-fixture set")
    return GroundTruth(frames, serves, rally_count)


def _scaled_frames(base30: int, fps: float) -> int:
    from annotator.fps_constants import ScalingKind

    return int(ScalingKind.FRAME_COUNT.scale(base30, fps))


def _fixture_names(rows: np.ndarray) -> np.ndarray:
    return np.char.decode(rows["fixture"], "ascii")


def seeded_region_mask(rows: np.ndarray) -> np.ndarray:
    """Return rows selected by at least one label-blind region channel."""
    selected = np.zeros(len(rows), dtype=bool)
    for field in REGION_FIELDS:
        selected |= rows[field].astype(bool)
    return selected


def _result_variant(result: Mapping[str, Any]) -> str:
    if "selected_variant" not in result:
        return CANDIDATE_VARIANT
    selected_variant = result["selected_variant"]
    selected = _validate_variant(selected_variant)
    if selected == CANDIDATE_VARIANT:
        raise ValueError("baseline tree result must omit selected_variant")
    return selected


def verify_tree_result(
    path: Path,
    verified: VerifiedFeatures,
    variant: str | None = None,
) -> dict[str, Any]:
    """Verify the retained result identity and selected HGB fold structure."""
    result = _read_json_object(Path(path))
    result_variant = _result_variant(result)
    _validate_variant_for_manifest(verified.manifest, result_variant)
    if variant is not None and result_variant != _validate_variant(variant):
        raise ValueError("tree result selected_variant differs")
    expected = {
        "schema": RESULTS_SCHEMA,
        "source_commit": verified.manifest["source_commit"],
        "feature_sha256": verified.manifest["feature_sha256"],
        "row_count": len(verified.rows),
        "model_row_count": int(np.count_nonzero(seeded_region_mask(verified.rows))),
        "model_search_surface": "seeded_union",
    }
    for field, value in expected.items():
        if result.get(field) != value:
            raise ValueError(f"tree result {field} differs")

    try:
        physics_result = result["models"]["histogram_boosting"]["physics"]
        folds = physics_result["folds"]
    except (KeyError, TypeError) as error:
        raise ValueError("tree result does not contain the selected HGB physics variant") from error
    feature_count = physics_result.get("feature_count")
    expected_feature_count = len(_feature_names(verified.manifest, "physics", variant=result_variant))
    if isinstance(feature_count, bool) or not isinstance(feature_count, int) or feature_count != expected_feature_count:
        raise ValueError("tree result selected feature count differs")
    if not isinstance(folds, list) or len(folds) != len(FIXTURE_SPECS):
        raise ValueError("tree result selected folds differ")
    by_fixture: dict[str, Mapping[str, Any]] = {}
    for fold in folds:
        if not isinstance(fold, Mapping):
            raise TypeError("tree result selected fold is malformed")
        fixture = fold.get("test_fixture")
        if not isinstance(fixture, str) or fixture not in FIXTURE_SPECS or fixture in by_fixture:
            raise ValueError("tree result selected fold fixture differs")
        by_fixture[str(fixture)] = fold
    if set(by_fixture) != set(FIXTURE_SPECS):
        raise ValueError("tree result selected fold fixture set differs")
    for fixture, fold in by_fixture.items():
        expected_training = [name for name in FIXTURE_SPECS if name != fixture]
        threshold = fold.get("threshold")
        frames = fold.get("prediction_frames")
        if (
            fold.get("train_fixtures") != expected_training
            or isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or not 0.0 <= float(threshold) <= 1.0
            or not isinstance(frames, list)
            or any(isinstance(frame, bool) or not isinstance(frame, int) or frame < 0 for frame in frames)
            or fold.get("prediction_count") != len(frames)
            or frames != sorted(set(frames))
        ):
            raise ValueError(f"{fixture}: tree result selected fold differs")
    return result


def _nearest_distances(frames: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Return each frame's distance to the nearest sorted target frame."""
    if not len(targets):
        return np.full(len(frames), np.iinfo(np.int32).max, dtype=np.int32)
    positions = np.searchsorted(targets, frames)
    left_index = np.maximum(positions - 1, 0)
    right_index = np.minimum(positions, len(targets) - 1)
    left = np.abs(frames - targets[left_index])
    right = np.abs(frames - targets[right_index])
    return np.minimum(left, right).astype(np.int32)


def build_training_mask(
    rows: np.ndarray,
    fixtures: Sequence[str],
    ground_truth: GroundTruth,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select positives plus hard and sampled easy negatives on named fixtures."""
    names = _fixture_names(rows)
    selected = np.zeros(len(rows), dtype=bool)
    labels = np.zeros(len(rows), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    for fixture in fixtures:
        fixture_indices = np.flatnonzero(names == fixture)
        fixture_rows = rows[fixture_indices]
        fps = float(FIXTURE_SPECS[fixture][1])
        distances = _nearest_distances(fixture_rows["frame"], ground_truth.frames[fixture])
        positive = distances <= _scaled_frames(POSITIVE_RADIUS_BASE30, fps)
        ignored = (~positive) & (distances <= _scaled_frames(IGNORE_RADIUS_BASE30, fps))
        negative = ~positive & ~ignored
        hard = negative & (distances <= _scaled_frames(HARD_NEGATIVE_RADIUS_BASE30, fps))
        positive_count = int(positive.sum())
        negative_limit = MAX_NEGATIVE_RATIO * positive_count
        hard_indices = np.flatnonzero(hard)
        easy_indices = np.flatnonzero(negative & ~hard)
        easy_count = max(0, negative_limit - len(hard_indices))
        if len(easy_indices) > easy_count:
            easy_indices = rng.choice(easy_indices, size=easy_count, replace=False)
        local_selected = np.concatenate([np.flatnonzero(positive), hard_indices, easy_indices])
        selected[fixture_indices[local_selected]] = True
        labels[fixture_indices[np.flatnonzero(positive)]] = 1
    return selected, labels


def _base_feature_names(manifest: Mapping[str, Any], feature_set: str) -> list[str]:
    """Return the normal ordered columns for one model feature set."""
    families = _feature_families(manifest["feature_families"])
    names: list[str] = []
    for family in FEATURE_SETS[feature_set]:
        for name in families[family]:
            if name not in names:
                names.append(name)
    return names


def _feature_names(
    manifest: Mapping[str, Any],
    feature_set: str,
    *,
    variant: str = CANDIDATE_VARIANT,
) -> list[str]:
    """Return the columns for the selected HGB physics trial."""
    selected_variant = _validate_variant_for_manifest(manifest, variant)
    names = _base_feature_names(manifest, feature_set)
    if feature_set != "physics" or selected_variant != PHYSICS_WITHOUT_RAW_MOTION_VARIANT:
        return names
    dropped = {
        f"{signal}_t{offset:+d}"
        for signal in FRAME_RATE_SENSITIVE_SIGNALS
        for offset in WINDOW_OFFSETS_BASE30
    }
    return [name for name in names if name not in dropped]


def _matrix(rows: np.ndarray, names: Sequence[str]) -> np.ndarray:
    return np.column_stack([rows[name].astype(np.float32, copy=False) for name in names])


def _make_model(model_name: str) -> Any:
    if model_name == "histogram_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=180,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )
    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=160,
            max_depth=16,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
    raise ValueError(f"unknown model {model_name!r}")


def temporal_nms(
    frames: np.ndarray,
    intervals: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    radius: int,
) -> np.ndarray:
    """Keep the strongest thresholded frame within each temporal neighbourhood."""
    accepted: list[int] = []
    for interval_id in np.unique(intervals):
        local = np.flatnonzero((intervals == interval_id) & (probabilities >= threshold))
        order = sorted(local, key=lambda index: (-probabilities[index], frames[index]))
        kept: list[int] = []
        for index in order:
            if all(abs(int(frames[index]) - int(frames[other])) > radius for other in kept):
                kept.append(int(index))
        accepted.extend(kept)
    return np.asarray(sorted(accepted, key=lambda index: frames[index]), dtype=np.int32)


def _candidate_score_rows(
    rows: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    kept: np.ndarray,
) -> np.ndarray:
    """Retain aligned held-out scores and the current threshold/NMS decision."""
    if len(rows) != len(probabilities):
        raise ValueError("candidate rows and timing scores differ in length")
    decisions = np.full(len(rows), CANDIDATE_BELOW_THRESHOLD, dtype=np.uint8)
    decisions[probabilities >= threshold] = CANDIDATE_NEARBY_DUPLICATE
    decisions[kept] = CANDIDATE_RETAINED

    output = np.empty(len(rows), dtype=CANDIDATE_SCORE_DTYPE)
    output["fixture"] = rows["fixture"]
    output["interval_id"] = rows["interval_id"]
    output["frame"] = rows["frame"]
    output["timing_score"] = probabilities
    output["threshold"] = threshold
    output["decision"] = decisions
    return output


def _greedy_matches(gt_frames: np.ndarray, predictions: np.ndarray, tolerance: int) -> list[tuple[int, int, int]]:
    pairs = []
    for gt_index, gt_frame in enumerate(gt_frames):
        nearby = np.flatnonzero(np.abs(predictions - gt_frame) <= tolerance)
        for prediction_index in nearby:
            offset = int(predictions[prediction_index] - gt_frame)
            pairs.append((abs(offset), int(gt_frame), int(predictions[prediction_index]), gt_index, int(prediction_index), offset))
    pairs.sort()
    used_gt: set[int] = set()
    used_predictions: set[int] = set()
    matches: list[tuple[int, int, int]] = []
    for _distance, _gt_frame, _prediction_frame, gt_index, prediction_index, offset in pairs:
        if gt_index in used_gt or prediction_index in used_predictions:
            continue
        used_gt.add(gt_index)
        used_predictions.add(prediction_index)
        matches.append((gt_index, prediction_index, offset))
    return matches


def _event_counts(
    ground_truth: GroundTruth,
    predictions: Mapping[str, np.ndarray],
    tolerance_base30: int,
    fixtures: Sequence[str] | None = None,
) -> dict[str, int | float]:
    matched = 0
    gt_total = 0
    prediction_total = 0
    serve_matched = 0
    serve_total = 0
    nonserve_matched = 0
    nonserve_total = 0
    offsets: list[int] = []
    selected_fixtures = tuple(ground_truth.frames) if fixtures is None else tuple(fixtures)
    for fixture in selected_fixtures:
        gt_frames = ground_truth.frames[fixture]
        predicted = predictions.get(fixture, np.empty(0, dtype=np.int32))
        tolerance = _scaled_frames(tolerance_base30, FIXTURE_SPECS[fixture][1])
        matches = _greedy_matches(gt_frames, predicted, tolerance)
        matched += len(matches)
        gt_total += len(gt_frames)
        prediction_total += len(predicted)
        serve_frames = ground_truth.serves[fixture]
        fixture_serve_total = len(serve_frames)
        serve_total += fixture_serve_total
        nonserve_total += len(gt_frames) - fixture_serve_total
        for gt_index, _prediction_index, offset in matches:
            offsets.append(abs(offset))
            if int(gt_frames[gt_index]) in serve_frames:
                serve_matched += 1
            else:
                nonserve_matched += 1
    precision = matched / prediction_total if prediction_total else 0.0
    recall = matched / gt_total if gt_total else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "matched": matched,
        "ground_truth": gt_total,
        "predictions": prediction_total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "serve_matched": serve_matched,
        "serve_total": serve_total,
        "serve_recall": serve_matched / serve_total,
        "nonserve_matched": nonserve_matched,
        "nonserve_total": nonserve_total,
        "nonserve_recall": nonserve_matched / nonserve_total,
        "median_absolute_offset": float(np.median(offsets)) if offsets else None,
    }


def _threshold_candidates(probabilities: np.ndarray) -> np.ndarray:
    del probabilities
    return np.linspace(0.05, 0.95, 19)


def _choose_threshold(
    rows: np.ndarray,
    names: np.ndarray,
    probabilities: np.ndarray,
    validation_fixtures: Sequence[str],
    ground_truth: GroundTruth,
) -> tuple[float, dict[str, int | float]]:
    best_threshold = 0.5
    best_metrics: dict[str, int | float] | None = None
    best_key = (-math.inf, -math.inf, -math.inf, -math.inf)
    for threshold in _threshold_candidates(probabilities):
        predictions: dict[str, np.ndarray] = {}
        for fixture in validation_fixtures:
            fixture_indices = np.flatnonzero(names == fixture)
            fixture_rows = rows[fixture_indices]
            kept = temporal_nms(
                fixture_rows["frame"],
                fixture_rows["interval_id"],
                probabilities[fixture_indices],
                float(threshold),
                _scaled_frames(NMS_RADIUS_BASE30, FIXTURE_SPECS[fixture][1]),
            )
            predictions[fixture] = fixture_rows["frame"][kept]
        metrics = _event_counts(ground_truth, predictions, 5, validation_fixtures)
        key = (float(metrics["f1"]), float(metrics["recall"]), float(metrics["precision"]), float(threshold))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics
    assert best_metrics is not None
    return best_threshold, best_metrics


def _fit_predict(
    rows: np.ndarray,
    train_fixtures: Sequence[str],
    predict_fixtures: Sequence[str],
    ground_truth: GroundTruth,
    feature_names: Sequence[str],
    model_name: str,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    names = _fixture_names(rows)
    training_mask, labels = build_training_mask(rows, train_fixtures, ground_truth, seed=seed)
    predict_mask = np.isin(names, predict_fixtures)
    model = _make_model(model_name)
    model.fit(_matrix(rows[training_mask], feature_names), labels[training_mask])
    probabilities = model.predict_proba(_matrix(rows[predict_mask], feature_names))[:, 1]
    return np.flatnonzero(predict_mask), probabilities


def _outer_fold_with_candidate_scores(
    rows: np.ndarray,
    test_fixture: str,
    ground_truth: GroundTruth,
    feature_names: Sequence[str],
    model_name: str,
    *,
    retain_candidate_scores: bool = False,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray | None]:
    train_fixtures = [fixture for fixture in FIXTURE_SPECS if fixture != test_fixture]
    names = _fixture_names(rows)
    oof_probabilities = np.full(len(rows), np.nan, dtype=np.float64)
    for validation_fixture in train_fixtures:
        inner_train = [fixture for fixture in train_fixtures if fixture != validation_fixture]
        indices, probabilities = _fit_predict(
            rows,
            inner_train,
            [validation_fixture],
            ground_truth,
            feature_names,
            model_name,
            seed=RANDOM_SEED + list(FIXTURE_SPECS).index(validation_fixture),
        )
        oof_probabilities[indices] = probabilities
    train_mask = np.isin(names, train_fixtures)
    threshold, threshold_metrics = _choose_threshold(
        rows[train_mask],
        names[train_mask],
        oof_probabilities[train_mask],
        train_fixtures,
        ground_truth,
    )

    test_indices, test_probabilities = _fit_predict(
        rows,
        train_fixtures,
        [test_fixture],
        ground_truth,
        feature_names,
        model_name,
        seed=RANDOM_SEED + 100 + list(FIXTURE_SPECS).index(test_fixture),
    )
    test_rows = rows[test_indices]
    kept = temporal_nms(
        test_rows["frame"],
        test_rows["interval_id"],
        test_probabilities,
        threshold,
        _scaled_frames(NMS_RADIUS_BASE30, FIXTURE_SPECS[test_fixture][1]),
    )
    predictions = test_rows["frame"][kept].astype(np.int32)
    candidate_scores = (
        _candidate_score_rows(test_rows, test_probabilities, threshold, kept)
        if retain_candidate_scores
        else None
    )
    fold = {
        "test_fixture": test_fixture,
        "train_fixtures": train_fixtures,
        "threshold": threshold,
        "threshold_oof_metrics_at_5": threshold_metrics,
        "prediction_count": len(predictions),
        "prediction_frames": predictions.tolist(),
        "metrics": {
            str(tolerance): _event_counts(
                ground_truth,
                {test_fixture: predictions},
                tolerance,
                [test_fixture],
            )
            for tolerance in TOLERANCES_BASE30
        },
    }
    return fold, predictions, candidate_scores


def _outer_fold(
    rows: np.ndarray,
    test_fixture: str,
    ground_truth: GroundTruth,
    feature_names: Sequence[str],
    model_name: str,
) -> tuple[dict[str, Any], np.ndarray]:
    """Run one outer fold while preserving the existing two-value API."""
    fold, predictions, _candidate_scores = _outer_fold_with_candidate_scores(
        rows,
        test_fixture,
        ground_truth,
        feature_names,
        model_name,
    )
    return fold, predictions


def _region_coverage(
    gt_frames: np.ndarray,
    serve_frames: set[int],
    region_frames: np.ndarray,
    tolerance: int,
) -> dict[str, int | float]:
    distances = _nearest_distances(gt_frames, region_frames)
    covered = distances <= tolerance
    serves = np.asarray([int(frame) in serve_frames for frame in gt_frames], dtype=bool)
    nonserves = ~serves
    serve_covered = int(np.count_nonzero(covered & serves))
    nonserve_covered = int(np.count_nonzero(covered & nonserves))
    serve_total = int(serves.sum())
    nonserve_total = int(nonserves.sum())
    total = len(gt_frames)
    covered_total = int(covered.sum())
    return {
        "covered": covered_total,
        "total": total,
        "recall": covered_total / total,
        "serve_covered": serve_covered,
        "serve_total": serve_total,
        "serve_recall": serve_covered / serve_total,
        "nonserve_covered": nonserve_covered,
        "nonserve_total": nonserve_total,
        "nonserve_recall": nonserve_covered / nonserve_total,
    }


def _aggregate_region_coverage(rows: Sequence[Mapping[str, int | float]]) -> dict[str, int | float]:
    totals = {
        name: sum(int(row[name]) for row in rows)
        for name in ("covered", "total", "serve_covered", "serve_total", "nonserve_covered", "nonserve_total")
    }
    totals.update(
        {
            "recall": totals["covered"] / totals["total"],
            "serve_recall": totals["serve_covered"] / totals["serve_total"],
            "nonserve_recall": totals["nonserve_covered"] / totals["nonserve_total"],
        }
    )
    return totals


def _coverage_surface(
    region_frames_by_fixture: Mapping[str, np.ndarray],
    ground_truth: GroundTruth,
) -> dict[str, Any]:
    by_fixture: dict[str, dict[str, dict[str, int | float]]] = {}
    for fixture, gt_frames in ground_truth.frames.items():
        fixture_fps = FIXTURE_SPECS[fixture][1]
        by_fixture[fixture] = {
            str(tolerance): _region_coverage(
                gt_frames,
                ground_truth.serves[fixture],
                region_frames_by_fixture[fixture],
                0 if tolerance == 0 else _scaled_frames(tolerance, fixture_fps),
            )
            for tolerance in (0, *TOLERANCES_BASE30)
        }
    return {
        "strict": _aggregate_region_coverage([values["0"] for values in by_fixture.values()]),
        "operational": {
            str(tolerance): _aggregate_region_coverage(
                [values[str(tolerance)] for values in by_fixture.values()]
            )
            for tolerance in TOLERANCES_BASE30
        },
        "fixtures": by_fixture,
    }


def _region_ceiling(
    rows: np.ndarray,
    manifest: Mapping[str, Any],
    ground_truth: GroundTruth,
) -> dict[str, dict[str, Any]]:
    names = _fixture_names(rows)
    output: dict[str, dict[str, Any]] = {}
    region_sets = {"search_intervals": (), "seeded_union": tuple(REGION_FIELDS)}
    region_sets.update({name.removeprefix("region_"): (name,) for name in REGION_FIELDS})
    for label, fields in region_sets.items():
        region_frames_by_fixture: dict[str, np.ndarray] = {}
        for fixture in ground_truth.frames:
            fixture_rows = rows[names == fixture]
            in_region = np.ones(len(fixture_rows), dtype=bool) if not fields else np.zeros(len(fixture_rows), dtype=bool)
            for field in fields:
                in_region |= fixture_rows[field].astype(bool)
            region_frames_by_fixture[fixture] = np.sort(fixture_rows["frame"][in_region])
        output[label] = _coverage_surface(region_frames_by_fixture, ground_truth)

    tracker_intervals = _manifest_intervals(manifest, "tracker_intervals")
    tracker_frames = {
        fixture: np.concatenate([np.arange(start, end, dtype=np.int32) for start, end in intervals])
        for fixture, intervals in tracker_intervals.items()
    }
    output["court_present_tracker_intervals"] = _coverage_surface(tracker_frames, ground_truth)
    eligible_intervals = _manifest_intervals(manifest, "eligible_intervals")
    eligible_frames = {
        fixture: np.concatenate([np.arange(start, end, dtype=np.int32) for start, end in intervals])
        for fixture, intervals in eligible_intervals.items()
    }
    output["eligible_intervals"] = _coverage_surface(eligible_frames, ground_truth)
    return output


def score(
    verified: VerifiedFeatures,
    *,
    variant: str = CANDIDATE_VARIANT,
    retain_candidate_scores: bool = False,
) -> tuple[dict[str, Any], np.ndarray | None]:
    selected_variant = _validate_variant_for_manifest(verified.manifest, variant)
    ground_truth = _load_ground_truth()
    model_rows = verified.rows[seeded_region_mask(verified.rows)]
    candidate_chunks: list[np.ndarray] = []
    output: dict[str, Any] = {
        "schema": RESULTS_SCHEMA,
        "source_commit": verified.manifest["source_commit"],
        "feature_sha256": verified.manifest["feature_sha256"],
        "row_count": len(verified.rows),
        "model_row_count": len(model_rows),
        "model_search_surface": "seeded_union",
        "ground_truth_rallies": ground_truth.rally_count,
        "region_ceiling": _region_ceiling(verified.rows, verified.manifest, ground_truth),
        "models": {},
        "training": {
            "positive_radius_base30": POSITIVE_RADIUS_BASE30,
            "ignore_radius_base30": IGNORE_RADIUS_BASE30,
            "hard_negative_radius_base30": HARD_NEGATIVE_RADIUS_BASE30,
            "maximum_negative_ratio": MAX_NEGATIVE_RATIO,
            "nms_radius_base30": NMS_RADIUS_BASE30,
        },
    }
    if selected_variant != CANDIDATE_VARIANT:
        output["selected_variant"] = selected_variant
    for model_name in ("histogram_boosting", "random_forest"):
        model_output: dict[str, Any] = {}
        for feature_set in FEATURE_SETS:
            is_selected_trial = model_name == "histogram_boosting" and feature_set == "physics"
            if is_selected_trial:
                feature_names = _feature_names(
                    verified.manifest,
                    feature_set,
                    variant=selected_variant,
                )
            else:
                feature_names = _base_feature_names(verified.manifest, feature_set)
            predictions: dict[str, np.ndarray] = {}
            folds = []
            for test_fixture in FIXTURE_SPECS:
                capture_fold = (
                    retain_candidate_scores
                    and model_name == "histogram_boosting"
                    and feature_set == "physics"
                )
                fold, fold_predictions, fold_candidates = _outer_fold_with_candidate_scores(
                    model_rows,
                    test_fixture,
                    ground_truth,
                    feature_names,
                    model_name,
                    retain_candidate_scores=capture_fold,
                )
                folds.append(fold)
                predictions[test_fixture] = fold_predictions
                if fold_candidates is not None:
                    candidate_chunks.append(fold_candidates)
            metrics = {
                str(tolerance): _event_counts(ground_truth, predictions, tolerance)
                for tolerance in TOLERANCES_BASE30
            }
            model_output[feature_set] = {
                "feature_count": len(feature_names),
                "folds": folds,
                "metrics": metrics,
            }
            print(f"completed {model_name} / {feature_set}", flush=True)
        output["models"][model_name] = model_output
    candidate_scores = np.concatenate(candidate_chunks) if candidate_chunks else None
    if retain_candidate_scores:
        if candidate_scores is None or len(candidate_scores) != len(model_rows):
            raise AssertionError("selected held-out candidate export is incomplete")
        candidate_identities = candidate_scores[["fixture", "interval_id", "frame"]]
        model_identities = model_rows[["fixture", "interval_id", "frame"]]
        if not np.array_equal(candidate_identities, model_identities):
            raise AssertionError("selected held-out candidate order differs from the model rows")
    return output, candidate_scores


def write_results(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.name.endswith(".gz"):
        with destination.open("wb") as raw, gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0
        ) as zipped:
            zipped.write(encoded)
    else:
        destination.write_bytes(encoded)


def write_candidate_scores(
    candidate_path: Path,
    manifest_path: Path,
    rows: np.ndarray,
    verified: VerifiedFeatures,
    tree_result_path: Path,
    results: Mapping[str, Any],
) -> None:
    """Write one observational held-out score row per selected model candidate."""
    candidate_destination = Path(candidate_path)
    manifest_destination = Path(manifest_path)
    if candidate_destination.parent.resolve() != manifest_destination.parent.resolve():
        raise ValueError("candidate scores and their manifest must share a directory")
    if not candidate_destination.name.endswith(".npy.xz") or not manifest_destination.name.endswith(".json"):
        raise ValueError("candidate scores must use .npy.xz with a JSON manifest")
    if rows.dtype != CANDIDATE_SCORE_DTYPE:
        raise ValueError("candidate score fields differ")

    selected_variant = _result_variant(results)
    _validate_variant_for_manifest(verified.manifest, selected_variant)
    candidate_destination.parent.mkdir(parents=True, exist_ok=True)
    _write_npy_xz(candidate_destination, rows)
    folds = results["models"]["histogram_boosting"]["physics"]["folds"]
    fold_summaries = []
    for fold in folds:
        fixture = str(fold["test_fixture"])
        fixture_rows = rows[np.char.decode(rows["fixture"], "ascii") == fixture]
        fold_summaries.append(
            {
                "fixture": fixture,
                "threshold": float(fold["threshold"]),
                "nms_radius_frames": _scaled_frames(NMS_RADIUS_BASE30, FIXTURE_SPECS[fixture][1]),
                "candidate_count": len(fixture_rows),
                "prediction_count": int(np.count_nonzero(fixture_rows["decision"] == CANDIDATE_RETAINED)),
            }
        )
    manifest = {
        "schema": CANDIDATE_SCORES_SCHEMA,
        "export_kind": "observational held-out timing scores",
        "timing_labels_read": True,
        "player_side_labels_read": False,
        "source_commit": verified.manifest["source_commit"],
        "feature_sha256": verified.manifest["feature_sha256"],
        "tree_result_sha256": _sha256(tree_result_path),
        "variant": selected_variant,
        "fixture_set": list(FIXTURE_SPECS),
        "candidate_file": candidate_destination.name,
        "candidate_sha256": _sha256(candidate_destination),
        "row_count": len(rows),
        "fields": list(CANDIDATE_SCORE_DTYPE.names or ()),
        "decision_codes": CANDIDATE_DECISIONS,
        "nms_radius_base30": NMS_RADIUS_BASE30,
        "folds": fold_summaries,
    }
    manifest_destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_candidate_scores(
    manifest_path: Path,
    verified: VerifiedFeatures,
    tree_result_path: Path,
) -> VerifiedCandidateScores:
    """Verify the selected held-out score sidecar without reading player-side labels."""
    manifest_path = Path(manifest_path)
    manifest = _read_json_object(manifest_path)
    tree_result = verify_tree_result(tree_result_path, verified)
    selected_variant = _result_variant(tree_result)
    expected_metadata = {
        "schema": CANDIDATE_SCORES_SCHEMA,
        "export_kind": "observational held-out timing scores",
        "timing_labels_read": True,
        "player_side_labels_read": False,
        "source_commit": verified.manifest["source_commit"],
        "feature_sha256": verified.manifest["feature_sha256"],
        "tree_result_sha256": _sha256(tree_result_path),
        "variant": selected_variant,
        "fixture_set": list(FIXTURE_SPECS),
        "fields": list(CANDIDATE_SCORE_DTYPE.names or ()),
        "decision_codes": CANDIDATE_DECISIONS,
        "nms_radius_base30": NMS_RADIUS_BASE30,
    }
    for field, expected in expected_metadata.items():
        if manifest.get(field) != expected:
            raise ValueError(f"candidate score manifest {field} differs")

    filename = _relative_filename(manifest.get("candidate_file"), "candidate_file")
    candidate_path = manifest_path.parent / filename
    if manifest.get("candidate_sha256") != _sha256(candidate_path):
        raise ValueError("candidate score file SHA-256 differs")
    with lzma.open(candidate_path, "rb") as source:
        rows = np.load(source, allow_pickle=False)
    model_rows = verified.rows[seeded_region_mask(verified.rows)]
    if rows.ndim != 1 or rows.dtype != CANDIDATE_SCORE_DTYPE or len(rows) != len(model_rows):
        raise ValueError("candidate score table shape or fields differ")
    if manifest.get("row_count") != len(rows):
        raise ValueError("candidate score row count differs")
    if not np.array_equal(
        rows[["fixture", "interval_id", "frame"]],
        model_rows[["fixture", "interval_id", "frame"]],
    ):
        raise ValueError("candidate score identities differ from the model surface")
    if (
        not np.isfinite(rows["timing_score"]).all()
        or not np.isfinite(rows["threshold"]).all()
        or np.any((rows["timing_score"] < 0.0) | (rows["timing_score"] > 1.0))
        or np.any((rows["threshold"] < 0.0) | (rows["threshold"] > 1.0))
    ):
        raise ValueError("candidate timing scores or thresholds are invalid")
    if not np.isin(rows["decision"], list(CANDIDATE_DECISIONS.values())).all():
        raise ValueError("candidate decisions are invalid")

    fold_summaries = manifest.get("folds")
    if not isinstance(fold_summaries, list) or len(fold_summaries) != len(FIXTURE_SPECS):
        raise ValueError("candidate fold summaries differ")
    names = np.char.decode(rows["fixture"], "ascii")
    summaries_by_fixture = {
        str(summary.get("fixture")): summary
        for summary in fold_summaries
        if isinstance(summary, Mapping)
    }
    if set(summaries_by_fixture) != set(FIXTURE_SPECS):
        raise ValueError("candidate fold fixture set differs")
    result_folds = {
        str(fold["test_fixture"]): fold
        for fold in tree_result["models"]["histogram_boosting"]["physics"]["folds"]
    }
    for fixture, (_video_id, fps) in FIXTURE_SPECS.items():
        fixture_rows = rows[names == fixture]
        summary = summaries_by_fixture[fixture]
        threshold = summary.get("threshold")
        result_fold = result_folds[fixture]
        result_threshold = float(result_fold["threshold"])
        result_frames = np.asarray(result_fold["prediction_frames"], dtype=np.int32)
        retained_frames = fixture_rows[fixture_rows["decision"] == CANDIDATE_RETAINED]["frame"]
        radius = _scaled_frames(NMS_RADIUS_BASE30, fps)
        if (
            not isinstance(threshold, (int, float))
            or float(threshold) != result_threshold
            or summary.get("nms_radius_frames") != radius
            or summary.get("candidate_count") != len(fixture_rows)
            or summary.get("prediction_count") != len(retained_frames)
            or not np.all(fixture_rows["threshold"] == float(threshold))
            or not np.array_equal(retained_frames, result_frames)
        ):
            raise ValueError(f"{fixture}: candidate fold summary differs")
        kept = temporal_nms(
            fixture_rows["frame"],
            fixture_rows["interval_id"],
            fixture_rows["timing_score"],
            float(threshold),
            radius,
        )
        expected_decisions = _candidate_score_rows(
            fixture_rows,
            fixture_rows["timing_score"],
            float(threshold),
            kept,
        )["decision"]
        if not np.array_equal(fixture_rows["decision"], expected_decisions):
            raise ValueError(f"{fixture}: candidate decisions differ from threshold and NMS")
    return VerifiedCandidateScores(manifest_path, manifest, rows, tree_result)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--variant", choices=ALLOWED_VARIANTS, default=CANDIDATE_VARIANT)
    arguments = parser.parse_args(argv)
    if (arguments.candidate_output is None) != (arguments.candidate_manifest is None):
        parser.error("--candidate-output and --candidate-manifest must be provided together")
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    verified = verify_freeze(arguments.manifest)
    retain_candidate_scores = arguments.candidate_output is not None
    results, candidate_scores = score(
        verified,
        variant=arguments.variant,
        retain_candidate_scores=retain_candidate_scores,
    )
    write_results(arguments.output, results)
    print(f"wrote {arguments.output}")
    if candidate_scores is not None:
        if arguments.candidate_output is None or arguments.candidate_manifest is None:
            raise AssertionError("candidate output paths are missing")
        write_candidate_scores(
            arguments.candidate_output,
            arguments.candidate_manifest,
            candidate_scores,
            verified,
            arguments.output,
            results,
        )
        print(f"wrote {arguments.candidate_output}")
        print(f"wrote {arguments.candidate_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
