"""Score player attribution for the frozen contact-detector event streams.

All event frames and learned thresholds are fixed before this scorer reads the
ShuttleSet ``player_side`` column. The only event stream reproduced here is the
previously reported eligible-court-only HGB sensitivity.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import lzma
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MODULE_ROOT = Path(__file__).resolve().parent
CONTACT_DET_ROOT = MODULE_ROOT.parent
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import freeze_contact_evidence as evidence_freezer
import score_contact_evidence as evidence_scorer
import score_tree_contact_detector as tree_scorer

RESULTS_SCHEMA = "contact-player-attribution-score/1"
TOLERANCES_BASE30 = (5, 10, 15)
TIMING_COUNT_FIELDS = (
    "matched",
    "ground_truth",
    "predictions",
    "serve_matched",
    "serve_total",
    "nonserve_matched",
    "nonserve_total",
)
TREE_FEATURE_SETS = ("physics", "physics_context", "context_only", "missingness_only")
TREE_MODELS = ("histogram_boosting", "random_forest")
SELECTED_HGB_VARIANT = "region_v2/histogram_boosting/physics/shipped"


@dataclass(frozen=True)
class TreeFreeze:
    """One verified tree feature table and its retained result."""

    generation: str
    manifest: dict[str, Any]
    rows: np.ndarray
    result: dict[str, Any]


@dataclass(frozen=True)
class AttributedMatch:
    """One timing match with its predicted and labelled court half."""

    fixture: str
    gt_frame: int
    prediction_frame: int
    serve: bool
    predicted_half: str | None
    target_half: str


def _read_json(path: Path) -> dict[str, Any]:
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


def _normalise_half(value: object, name: str) -> str | None:
    if value is None:
        return None
    raw = str(getattr(value, "value", value))
    if raw == "Top":
        return "Top"
    if raw in {"Bot", "Bottom"}:
        return "Bot"
    raise ValueError(f"{name}: expected Top, Bot, Bottom, or null; found {value!r}")


def _load_v1_features(manifest_path: Path) -> tuple[dict[str, Any], np.ndarray]:
    """Verify the retained version-1 feature table against its own manifest."""
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != "tree-contact-feature-manifest/1":
        raise ValueError("region-v1 feature manifest schema differs")
    if manifest.get("feature_schema") != "tree-contact-features/1" or manifest.get("labels_read") is not False:
        raise ValueError("region-v1 feature freeze contract differs")
    if manifest.get("fixture_set") != list(evidence_freezer.FIXTURE_SPECS):
        raise ValueError("region-v1 fixture set differs")
    feature_name = manifest.get("feature_file")
    if not isinstance(feature_name, str) or Path(feature_name).name != feature_name:
        raise ValueError("region-v1 feature filename is malformed")
    feature_path = manifest_path.parent / feature_name
    if _sha256(feature_path) != manifest.get("feature_sha256"):
        raise ValueError("region-v1 feature SHA-256 differs")
    with lzma.open(feature_path, "rb") as source:
        rows = np.load(source, allow_pickle=False)
    if rows.ndim != 1 or rows.dtype.names is None or len(rows) != manifest.get("row_count"):
        raise ValueError("region-v1 feature table shape differs")
    expected_fields = set(manifest.get("identity_fields", [])) | set(manifest.get("region_fields", []))
    for names in manifest.get("feature_families", {}).values():
        expected_fields.update(names)
    if set(rows.dtype.names) != expected_fields:
        raise ValueError("region-v1 feature fields differ")
    fixture_names = np.char.decode(rows["fixture"], "ascii")
    summaries = {str(row["fixture"]): row for row in manifest.get("fixtures", [])}
    if set(fixture_names) != set(evidence_freezer.FIXTURE_SPECS) or set(summaries) != set(evidence_freezer.FIXTURE_SPECS):
        raise ValueError("region-v1 feature fixtures differ")
    for fixture, (_video_id, fps) in evidence_freezer.FIXTURE_SPECS.items():
        fixture_rows = rows[fixture_names == fixture]
        summary = summaries[fixture]
        if len(fixture_rows) != summary.get("row_count") or not np.all(fixture_rows["fps"] == fps):
            raise ValueError(f"{fixture}: region-v1 row alignment differs")
        if np.any(fixture_rows["frame"] < 0) or np.any(fixture_rows["frame"] >= summary.get("frame_count")):
            raise ValueError(f"{fixture}: region-v1 frame lies outside the source timeline")
        if len(np.unique(fixture_rows["frame"])) != len(fixture_rows):
            raise ValueError(f"{fixture}: region-v1 frame identities are duplicated")
    return manifest, rows


def _validate_tree_result(
    generation: str,
    manifest: Mapping[str, Any],
    rows: np.ndarray,
    result_path: Path,
) -> dict[str, Any]:
    result = _read_json(result_path)
    expected_schema = "tree-contact-results/3" if generation == "region_v2" else "tree-contact-results/1"
    if result.get("schema") != expected_schema:
        raise ValueError(f"{generation}: tree result schema differs")
    if result.get("feature_sha256") != manifest.get("feature_sha256"):
        raise ValueError(f"{generation}: result and feature SHA-256 differ")
    if result.get("source_commit") != manifest.get("source_commit") or result.get("row_count") != len(rows):
        raise ValueError(f"{generation}: result provenance differs")
    models = result.get("models")
    if not isinstance(models, Mapping) or set(models) != set(TREE_MODELS):
        raise ValueError(f"{generation}: retained model set differs")
    fixture_names = np.char.decode(rows["fixture"], "ascii")
    for model_name in TREE_MODELS:
        if not isinstance(models[model_name], Mapping) or set(models[model_name]) != set(TREE_FEATURE_SETS):
            raise ValueError(f"{generation}/{model_name}: retained feature sets differ")
        for feature_set in TREE_FEATURE_SETS:
            folds = models[model_name][feature_set].get("folds")
            if not isinstance(folds, list) or len(folds) != len(evidence_freezer.FIXTURE_SPECS):
                raise ValueError(f"{generation}/{model_name}/{feature_set}: folds differ")
            for fold in folds:
                fixture = fold.get("test_fixture")
                frames = fold.get("prediction_frames")
                if fixture not in evidence_freezer.FIXTURE_SPECS or not isinstance(frames, list):
                    raise ValueError(f"{generation}/{model_name}/{feature_set}: prediction frames are missing")
                if fold.get("prediction_count") != len(frames) or frames != sorted(set(frames)):
                    raise ValueError(f"{generation}/{model_name}/{feature_set}/{fixture}: prediction frames differ")
                available = rows[fixture_names == fixture]["frame"]
                if not np.isin(np.asarray(frames, dtype=np.int32), available).all():
                    raise ValueError(f"{generation}/{model_name}/{feature_set}/{fixture}: prediction is outside the freeze")
    return result


def _load_tree_freezes(arguments: argparse.Namespace) -> dict[str, TreeFreeze]:
    verified_v2 = tree_scorer.verify_freeze(arguments.region_v2_manifest)
    result_v2 = _validate_tree_result(
        "region_v2", verified_v2.manifest, verified_v2.rows, arguments.region_v2_results
    )
    v1_manifest, v1_rows = _load_v1_features(arguments.region_v1_manifest)
    result_v1 = _validate_tree_result("region_v1", v1_manifest, v1_rows, arguments.region_v1_results)
    return {
        "region_v2": TreeFreeze("region_v2", verified_v2.manifest, verified_v2.rows, result_v2),
        "region_v1": TreeFreeze("region_v1", v1_manifest, v1_rows, result_v1),
    }


def _prediction_frames(result: Mapping[str, Any], model_name: str, feature_set: str) -> dict[str, np.ndarray]:
    folds = result["models"][model_name][feature_set]["folds"]
    predictions = {
        str(fold["test_fixture"]): np.asarray(fold["prediction_frames"], dtype=np.int32)
        for fold in folds
    }
    if set(predictions) != set(evidence_freezer.FIXTURE_SPECS):
        raise ValueError("retained prediction fixture set differs")
    return predictions


def _all_retained_predictions(freezes: Mapping[str, TreeFreeze]) -> dict[str, dict[str, np.ndarray]]:
    variants: dict[str, dict[str, np.ndarray]] = {}
    for generation, freeze in freezes.items():
        for model_name in TREE_MODELS:
            for feature_set in TREE_FEATURE_SETS:
                key = f"{generation}/{model_name}/{feature_set}"
                variants[key] = _prediction_frames(freeze.result, model_name, feature_set)
    return variants


def _eligible_row_mask(rows: np.ndarray, manifest: Mapping[str, Any]) -> np.ndarray:
    fixture_names = np.char.decode(rows["fixture"], "ascii")
    selected = np.zeros(len(rows), dtype=bool)
    intervals = tree_scorer._manifest_intervals(manifest, "eligible_intervals")
    for fixture, fixture_intervals in intervals.items():
        fixture_mask = fixture_names == fixture
        for start, end in fixture_intervals:
            selected |= fixture_mask & (rows["frame"] >= start) & (rows["frame"] < end)
    return selected


def _eligible_hgb_predictions(
    freeze: TreeFreeze, ground_truth: tree_scorer.GroundTruth
) -> dict[str, np.ndarray]:
    rows = freeze.rows[tree_scorer.seeded_region_mask(freeze.rows) & _eligible_row_mask(freeze.rows, freeze.manifest)]
    feature_names = tree_scorer._feature_names(freeze.manifest, "physics")
    predictions: dict[str, np.ndarray] = {}
    for fixture in evidence_freezer.FIXTURE_SPECS:
        _fold, fixture_predictions = tree_scorer._outer_fold(
            rows,
            fixture,
            ground_truth,
            feature_names,
            "histogram_boosting",
        )
        predictions[fixture] = fixture_predictions
    return predictions


def _assert_timing_metrics(
    predictions: Mapping[str, np.ndarray],
    ground_truth: tree_scorer.GroundTruth,
    retained: Mapping[str, Any],
    name: str,
) -> None:
    for tolerance in TOLERANCES_BASE30:
        actual = tree_scorer._event_counts(ground_truth, predictions, tolerance)
        expected = retained["metrics"][str(tolerance)]
        differences = {
            field: (actual[field], expected[field])
            for field in TIMING_COUNT_FIELDS
            if actual[field] != expected[field]
        }
        if differences:
            raise AssertionError(f"{name}: retained timing counts changed at ±{tolerance}: {differences}")


def _verify_all_timing_counts(
    freezes: Mapping[str, TreeFreeze],
    predictions: Mapping[str, Mapping[str, np.ndarray]],
    ground_truth: tree_scorer.GroundTruth,
) -> None:
    for generation, freeze in freezes.items():
        for model_name in TREE_MODELS:
            for feature_set in TREE_FEATURE_SETS:
                name = f"{generation}/{model_name}/{feature_set}"
                retained = freeze.result["models"][model_name][feature_set]
                _assert_timing_metrics(predictions[name], ground_truth, retained, name)
                for fold in retained["folds"]:
                    fixture = str(fold["test_fixture"])
                    for tolerance in TOLERANCES_BASE30:
                        actual = tree_scorer._event_counts(
                            ground_truth,
                            {fixture: predictions[name][fixture]},
                            tolerance,
                            [fixture],
                        )
                        expected = fold["metrics"][str(tolerance)]
                        if any(actual[field] != expected[field] for field in TIMING_COUNT_FIELDS):
                            raise AssertionError(
                                f"{name}/{fixture}: retained fold timing counts changed at ±{tolerance}"
                            )


def _manifest_input_rows(manifest: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    rows: dict[tuple[str, str], Mapping[str, Any]] = {}
    for fixture_row in manifest.get("inputs", []):
        fixture = str(fixture_row["fixture"])
        for file_row in fixture_row["files"]:
            rows[(fixture, str(file_row["role"]))] = file_row
    return rows


def _verify_stage_inputs(data_root: Path, freezes: Mapping[str, TreeFreeze]) -> None:
    shared_roles = set(evidence_freezer._stage_paths(data_root, evidence_freezer.FixtureSpec("sset_01", 1, 25.0)))
    expected_by_generation = {
        generation: _manifest_input_rows(freeze.manifest) for generation, freeze in freezes.items()
    }
    for fixture, (video_id, fps) in evidence_freezer.FIXTURE_SPECS.items():
        spec = evidence_freezer.FixtureSpec(fixture, video_id, fps)
        paths = evidence_freezer._stage_paths(data_root, spec)
        for role in shared_roles:
            rows = [expected[(fixture, role)] for expected in expected_by_generation.values()]
            if len({str(row["sha256"]) for row in rows}) != 1:
                raise ValueError(f"{fixture}/{role}: region-v1 and region-v2 inputs differ")
            path = paths[role]
            expected = rows[0]
            if path.name != expected["filename"] or path.stat().st_size != expected["size_bytes"]:
                raise ValueError(f"{fixture}/{role}: local stage identity differs")
            if _sha256(path) != expected["sha256"]:
                raise ValueError(f"{fixture}/{role}: local stage SHA-256 differs")


def nearest_tracked_player(top_gap: float, bot_gap: float) -> str | None:
    """Return the closest sticky-tracked half, with Top winning an exact tie."""
    top_finite = math.isfinite(float(top_gap))
    bot_finite = math.isfinite(float(bot_gap))
    if not top_finite and not bot_finite:
        return None
    if not bot_finite or (top_finite and top_gap <= bot_gap):
        return "Top"
    return "Bot"


def _nearest_maps(
    freezes: Mapping[str, TreeFreeze],
    variants: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, dict[tuple[str, int], str | None]]:
    output: dict[str, dict[tuple[str, int], str | None]] = {}
    for name, predictions in variants.items():
        generation = name.split("/", maxsplit=1)[0]
        freeze = freezes[generation]
        fixture_names = np.char.decode(freeze.rows["fixture"], "ascii")
        attribution: dict[tuple[str, int], str | None] = {}
        for fixture, frames in predictions.items():
            fixture_rows = freeze.rows[fixture_names == fixture]
            indices = np.searchsorted(fixture_rows["frame"], frames)
            if np.any(indices >= len(fixture_rows)) or not np.array_equal(fixture_rows["frame"][indices], frames):
                raise ValueError(f"{name}/{fixture}: prediction-to-feature alignment differs")
            for frame, row in zip(frames, fixture_rows[indices], strict=True):
                attribution[(fixture, int(frame))] = nearest_tracked_player(
                    float(row["wrist_gap_top_t+0"]),
                    float(row["wrist_gap_bot_t+0"]),
                )
        output[name] = attribution
    return output


def _shipped_attribution_map(
    data_root: Path,
    freezes: Mapping[str, TreeFreeze],
    variants: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[tuple[str, int], str | None]:
    from annotator import point_winner

    _verify_stage_inputs(data_root, freezes)
    frames_by_fixture: dict[str, set[int]] = {fixture: set() for fixture in evidence_freezer.FIXTURE_SPECS}
    for predictions in variants.values():
        for fixture, frames in predictions.items():
            frames_by_fixture[fixture].update(int(frame) for frame in frames)

    output: dict[tuple[str, int], str | None] = {}
    v2_summaries = {str(row["fixture"]): row for row in freezes["region_v2"].manifest["fixtures"]}
    v1_summaries = {str(row["fixture"]): row for row in freezes["region_v1"].manifest["fixtures"]}
    for fixture, (video_id, fps) in evidence_freezer.FIXTURE_SPECS.items():
        spec = evidence_freezer.FixtureSpec(fixture, video_id, fps)
        track, pose, court, _segments, sticky, _annotation = evidence_freezer._load_inputs(data_root, spec)
        if len(track) != v2_summaries[fixture]["frame_count"] or len(track) != v1_summaries[fixture]["frame_count"]:
            raise ValueError(f"{fixture}: stage and feature frame counts differ")
        for generation, freeze in freezes.items():
            fixture_names = np.char.decode(freeze.rows["fixture"], "ascii")
            fixture_rows = freeze.rows[fixture_names == fixture]
            frames = fixture_rows["frame"]
            sticky_gaps = np.asarray(sticky.distances_per_slot[frames], dtype=np.float32)
            sticky_gaps[~np.isfinite(sticky_gaps)] = np.nan
            if not np.allclose(
                fixture_rows["wrist_gap_top_t+0"], sticky_gaps[:, 0], equal_nan=True
            ) or not np.allclose(
                fixture_rows["wrist_gap_bot_t+0"], sticky_gaps[:, 1], equal_nan=True
            ):
                raise ValueError(f"{fixture}: {generation} centre-frame wrist gaps differ from replay")
            expected_visibility = (track[frames, 2] == 1).astype(np.float32)
            expected_pose_top = (sticky.picks[frames, 0] >= 0).astype(np.float32)
            expected_pose_bot = (sticky.picks[frames, 1] >= 0).astype(np.float32)
            if not np.array_equal(fixture_rows["shuttle_visible_t+0"], expected_visibility):
                raise ValueError(f"{fixture}: {generation} centre-frame shuttle validity differs from replay")
            if not np.array_equal(fixture_rows["pose_valid_top_t+0"], expected_pose_top) or not np.array_equal(
                fixture_rows["pose_valid_bot_t+0"], expected_pose_bot
            ):
                raise ValueError(f"{fixture}: {generation} centre-frame pose validity differs from replay")
        court_inputs = court.evidence.inputs
        if court_inputs is None:
            raise ValueError(f"{fixture}: court inputs are unavailable")
        for frame in sorted(frames_by_fixture[fixture]):
            output[(fixture, frame)] = _normalise_half(
                point_winner.attribute_half(frame, track, sticky, pose.bboxes, court_inputs.net_band),
                f"{fixture}/{frame} shipped attribution",
            )
    return output


def _load_side_ground_truth() -> dict[tuple[str, int], str]:
    """Read the player-side labels after every prediction stream is frozen."""
    from annotator.calibration.gt_scoring import load_gt_tables

    master, _homography, _court_info, _resolution = load_gt_tables()
    sides: dict[tuple[str, int], str] = {}
    for fixture, (video_id, _fps) in evidence_freezer.FIXTURE_SPECS.items():
        rows = master[master["vid"] == video_id]
        if "player_side" not in rows:
            raise ValueError("ShuttleSet shots_master is missing player_side")
        for frame, raw_half in zip(rows["frame_num"], rows["player_side"], strict=True):
            key = (fixture, int(frame))
            half = _normalise_half(raw_half, f"{fixture}/{frame} player_side")
            if half is None or key in sides:
                raise ValueError(f"{fixture}/{frame}: player-side identity differs")
            sides[key] = half
    if len(sides) != 3128:
        raise ValueError(f"expected 3,128 player-side labels, found {len(sides)}")
    return sides


def _load_timing_ground_truth() -> tree_scorer.GroundTruth:
    """Load only contact timing columns for the eligible-court HGB rerun."""
    import pandas as pd

    from annotator.calibration.fixtures import REPO_ROOT as CALIBRATION_ROOT
    from annotator.calibration.fixtures import SHARED_FILES, verify_file
    from annotator.calibration.scoring import load_gt_rallies

    master_pin = next(pin for pin in SHARED_FILES if pin.path.name == "shots_master.csv")
    verify_file(master_pin)
    timing_table = pd.read_csv(
        CALIBRATION_ROOT / master_pin.path,
        usecols=["vid", "set_id", "rally", "frame_num"],
    )
    frames: dict[str, np.ndarray] = {}
    serves: dict[str, set[int]] = {}
    rally_count = 0
    for fixture, (video_id, _fps) in evidence_freezer.FIXTURE_SPECS.items():
        rallies = load_gt_rallies(timing_table, video_id)
        rally_count += len(rallies)
        frames[fixture] = np.asarray(
            [frame for rally in rallies for frame in rally.stroke_frames], dtype=np.int32
        )
        serves[fixture] = {rally.stroke_frames[0] for rally in rallies}
    if rally_count != 292 or sum(len(values) for values in frames.values()) != 3128:
        raise ValueError("timing-only ground-truth totals differ from the pinned fixture set")
    return tree_scorer.GroundTruth(frames, serves, rally_count)


def _metric_slice(
    matches: Sequence[AttributedMatch],
    ground_truth_count: int,
    prediction_count: int | None = None,
) -> dict[str, int | float | None]:
    timing_matches = len(matches)
    answered = sum(match.predicted_half is not None for match in matches)
    correct = sum(match.predicted_half == match.target_half for match in matches)
    timing_recall = timing_matches / ground_truth_count if ground_truth_count else None
    answer_coverage = answered / timing_matches if timing_matches else None
    side_accuracy = correct / answered if answered else None
    joint_recall = correct / ground_truth_count if ground_truth_count else None
    result: dict[str, int | float | None] = {
        "ground_truth_contacts": ground_truth_count,
        "timing_matches": timing_matches,
        "timing_recall": timing_recall,
        "side_answers": answered,
        "side_answer_coverage": answer_coverage,
        "correct_side_answers": correct,
        "side_accuracy": side_accuracy,
        "timing_and_correct_side_recall": joint_recall,
    }
    if prediction_count is not None:
        joint_precision = correct / prediction_count if prediction_count else None
        if joint_precision is None or joint_recall is None or joint_precision + joint_recall == 0:
            joint_f1 = 0.0 if joint_precision == 0 and joint_recall == 0 else None
        else:
            joint_f1 = 2 * joint_precision * joint_recall / (joint_precision + joint_recall)
        result.update(
            {
                "event_predictions": prediction_count,
                "joint_event_and_side_precision": joint_precision,
                "joint_event_and_side_recall": joint_recall,
                "joint_event_and_side_f1": joint_f1,
            }
        )
    return result


def _score_slices(
    matches: Sequence[AttributedMatch],
    ground_truth_counts: Mapping[str, int],
    prediction_count: int,
) -> dict[str, dict[str, int | float | None]]:
    return {
        "all": _metric_slice(matches, ground_truth_counts["all"], prediction_count),
        "non_serve": _metric_slice(
            [match for match in matches if not match.serve], ground_truth_counts["non_serve"]
        ),
        "serve": _metric_slice(
            [match for match in matches if match.serve], ground_truth_counts["serve"]
        ),
    }


def _ground_truth_counts(ground_truth: tree_scorer.GroundTruth, fixtures: Sequence[str]) -> dict[str, int]:
    all_count = sum(len(ground_truth.frames[fixture]) for fixture in fixtures)
    serve_count = sum(len(ground_truth.serves[fixture]) for fixture in fixtures)
    return {"all": all_count, "non_serve": all_count - serve_count, "serve": serve_count}


def _tree_matches(
    predictions: Mapping[str, np.ndarray],
    attribution: Mapping[tuple[str, int], str | None],
    sides: Mapping[tuple[str, int], str],
    ground_truth: tree_scorer.GroundTruth,
    tolerance_base30: int,
) -> list[AttributedMatch]:
    matches: list[AttributedMatch] = []
    for fixture, gt_frames in ground_truth.frames.items():
        fixture_predictions = predictions[fixture]
        tolerance = tree_scorer._scaled_frames(tolerance_base30, evidence_freezer.FIXTURE_SPECS[fixture][1])
        for gt_index, prediction_index, _offset in tree_scorer._greedy_matches(
            gt_frames, fixture_predictions, tolerance
        ):
            gt_frame = int(gt_frames[gt_index])
            prediction_frame = int(fixture_predictions[prediction_index])
            matches.append(
                AttributedMatch(
                    fixture,
                    gt_frame,
                    prediction_frame,
                    gt_frame in ground_truth.serves[fixture],
                    attribution[(fixture, prediction_frame)],
                    sides[(fixture, gt_frame)],
                )
            )
    return matches


def _fixture_scores(
    matches: Sequence[AttributedMatch],
    predictions: Mapping[str, np.ndarray],
    ground_truth: tree_scorer.GroundTruth,
) -> dict[str, dict[str, dict[str, int | float | None]]]:
    output: dict[str, dict[str, dict[str, int | float | None]]] = {}
    for fixture in evidence_freezer.FIXTURE_SPECS:
        fixture_matches = [match for match in matches if match.fixture == fixture]
        output[fixture] = _score_slices(
            fixture_matches,
            _ground_truth_counts(ground_truth, [fixture]),
            len(predictions[fixture]),
        )
    return output


def _score_tree_variants(
    variants: Mapping[str, Mapping[str, np.ndarray]],
    shipped: Mapping[tuple[str, int], str | None],
    nearest: Mapping[str, Mapping[tuple[str, int], str | None]],
    sides: Mapping[tuple[str, int], str],
    ground_truth: tree_scorer.GroundTruth,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    all_counts = _ground_truth_counts(ground_truth, list(evidence_freezer.FIXTURE_SPECS))
    for name, predictions in variants.items():
        control = name.endswith(("/context_only", "/missingness_only"))
        for method, attribution in (("shipped", shipped), ("nearest_tracked_player", nearest[name])):
            variant_name = f"{name}/{method}"
            tolerance_scores: dict[str, Any] = {}
            for tolerance in TOLERANCES_BASE30:
                matches = _tree_matches(predictions, attribution, sides, ground_truth, tolerance)
                tolerance_scores[str(tolerance)] = {
                    "pooled": _score_slices(
                        matches,
                        all_counts,
                        sum(len(frames) for frames in predictions.values()),
                    ),
                    "fixtures": _fixture_scores(matches, predictions, ground_truth),
                }
            output[variant_name] = {
                "source": "tree",
                "control": control,
                "attribution": method,
                "tolerances": tolerance_scores,
            }
    return output


def _heuristic_matches(
    raw_matches: Sequence[evidence_scorer.ContactMatch],
    field_name: str,
    sides: Mapping[tuple[str, int], str],
) -> list[AttributedMatch]:
    return [
        AttributedMatch(
            match.candidate.fixture,
            match.gt_frame,
            match.candidate.contact_frame,
            match.gt_index == 0,
            getattr(match.candidate, field_name),
            sides[(match.candidate.fixture, match.gt_frame)],
        )
        for match in raw_matches
    ]


def _score_heuristics(
    verified: evidence_scorer.VerifiedFreeze,
    retained_score: Mapping[str, Any],
    sides: Mapping[tuple[str, int], str],
) -> dict[str, Any]:
    from annotator.calibration.gt_scoring import load_gt_tables
    from annotator.calibration.scoring import load_gt_rallies

    master, _homography, _court_info, _resolution = load_gt_tables()
    evidence_by_fixture = {str(row["fixture"]): row for row in verified.evidence["fixtures"]}
    rallies_by_fixture = {
        fixture: load_gt_rallies(master, video_id)
        for fixture, (video_id, _fps) in evidence_freezer.FIXTURE_SPECS.items()
    }
    all_counts = {
        "all": sum(len(rally.stroke_frames) for rallies in rallies_by_fixture.values() for rally in rallies),
        "serve": sum(len(rallies) for rallies in rallies_by_fixture.values()),
    }
    all_counts["non_serve"] = all_counts["all"] - all_counts["serve"]
    output: dict[str, Any] = {}
    for event_variant, filtered in (("raw", False), ("filtered", True)):
        rows_by_fixture = {
            fixture: evidence_scorer._contact_rows(evidence_by_fixture[fixture], filtered=filtered)
            for fixture in evidence_freezer.FIXTURE_SPECS
        }
        for method, field_name in (("current", "current_half"), ("ankle_counterfactual", "ankle_half")):
            tolerance_scores: dict[str, Any] = {}
            for tolerance in TOLERANCES_BASE30:
                matches_by_fixture: dict[str, list[AttributedMatch]] = {}
                for fixture, (_video_id, fps) in evidence_freezer.FIXTURE_SPECS.items():
                    metrics, raw_matches = evidence_scorer._match_variant(
                        evidence_by_fixture[fixture],
                        rallies_by_fixture[fixture],
                        fps,
                        rows_by_fixture[fixture],
                        tolerance,
                    )
                    expected = retained_score["fixtures"][fixture][event_variant][str(tolerance)]
                    checks = {
                        "matched": (metrics["overall"]["matched"], expected["overall"]["matched"]),
                        "serve_matched": (metrics["serve"]["matched"], expected["serve"]["matched"]),
                        "non_serve_matched": (
                            metrics["non_serve"]["matched"],
                            expected["non_serve"]["matched"],
                        ),
                        "candidate_count": (metrics["candidate_count"], expected["candidate_count"]),
                    }
                    differences = {name: values for name, values in checks.items() if values[0] != values[1]}
                    if differences:
                        raise AssertionError(
                            f"heuristic/{event_variant}/{fixture}: retained timing counts changed "
                            f"at ±{tolerance}: {differences}"
                        )
                    matches_by_fixture[fixture] = _heuristic_matches(raw_matches, field_name, sides)
                matches = [match for fixture_matches in matches_by_fixture.values() for match in fixture_matches]
                fixtures: dict[str, Any] = {}
                for fixture in evidence_freezer.FIXTURE_SPECS:
                    fixture_rallies = rallies_by_fixture[fixture]
                    fixture_total = sum(len(rally.stroke_frames) for rally in fixture_rallies)
                    fixture_counts = {
                        "all": fixture_total,
                        "serve": len(fixture_rallies),
                        "non_serve": fixture_total - len(fixture_rallies),
                    }
                    fixtures[fixture] = _score_slices(
                        matches_by_fixture[fixture], fixture_counts, len(rows_by_fixture[fixture])
                    )
                tolerance_scores[str(tolerance)] = {
                    "pooled": _score_slices(
                        matches,
                        all_counts,
                        sum(len(rows) for rows in rows_by_fixture.values()),
                    ),
                    "fixtures": fixtures,
                }
            output[f"heuristic/{event_variant}/{method}"] = {
                "source": "heuristic",
                "control": method == "ankle_counterfactual",
                "attribution": method,
                "tolerances": tolerance_scores,
            }
    return output


def _percentage(value: object) -> str:
    return "—" if value is None else f"{100 * float(value):.1f}%"


def _table_row(name: str, score: Mapping[str, Mapping[str, object]]) -> list[str]:
    values = [name]
    for slice_name in ("all", "serve", "non_serve"):
        metrics = score[slice_name]
        values.extend(
            _percentage(metrics[field])
            for field in (
                "timing_recall",
                "side_answer_coverage",
                "side_accuracy",
                "timing_and_correct_side_recall",
            )
        )
    return values


def _print_table(rows: Sequence[Sequence[str]]) -> None:
    headers = [
        "variant",
        "timing recall",
        "side answer coverage",
        "side accuracy",
        "timing + correct-side recall",
        "serve timing recall",
        "serve side answer coverage",
        "serve side accuracy",
        "serve timing + correct-side recall",
        "non-serve timing recall",
        "non-serve side answer coverage",
        "non-serve side accuracy",
        "non-serve timing + correct-side recall",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _header in headers) + " |")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def print_tables(variants: Mapping[str, Any]) -> None:
    print("\n## Contact player attribution at ±10 base-30 frames\n")
    _print_table([_table_row(name, value["tolerances"]["10"]["pooled"]) for name, value in variants.items()])
    print("\n## Region-v2 HGB physical with shipped attribution, by fixture\n")
    selected = variants[SELECTED_HGB_VARIANT]["tolerances"]["10"]["fixtures"]
    _print_table([_table_row(fixture, selected[fixture]) for fixture in evidence_freezer.FIXTURE_SPECS])


def score(arguments: argparse.Namespace) -> dict[str, Any]:
    """Verify frozen events, reproduce one fixed sensitivity, then score sides."""
    evidence = evidence_scorer.verify_freeze(arguments.evidence_manifest)
    retained_evidence_score = _read_json(arguments.evidence_results)
    if retained_evidence_score.get("evidence_sha256") != evidence.manifest["evidence_sha256"]:
        raise ValueError("retained heuristic result and evidence SHA-256 differ")
    freezes = _load_tree_freezes(arguments)
    predictions = _all_retained_predictions(freezes)

    timing_ground_truth = _load_timing_ground_truth()
    _verify_all_timing_counts(freezes, predictions, timing_ground_truth)
    eligible_name = "region_v2/histogram_boosting/physics_eligible_only"
    predictions[eligible_name] = _eligible_hgb_predictions(freezes["region_v2"], timing_ground_truth)

    nearest = _nearest_maps(freezes, predictions)
    shipped = _shipped_attribution_map(arguments.data_root, freezes, predictions)
    sides = _load_side_ground_truth()

    variants = _score_heuristics(evidence, retained_evidence_score, sides)
    variants.update(_score_tree_variants(predictions, shipped, nearest, sides, timing_ground_truth))
    return {
        "schema": RESULTS_SCHEMA,
        "fixture_set": list(evidence_freezer.FIXTURE_SPECS),
        "tolerances_base30": list(TOLERANCES_BASE30),
        "event_frames_frozen_before_side_scoring": True,
        "timing_counts_match_retained_results": True,
        "eligible_court_only_reproduced_before_side_scoring": True,
        "inputs": {
            "contact_evidence_sha256": evidence.manifest["evidence_sha256"],
            "region_v1_feature_sha256": freezes["region_v1"].manifest["feature_sha256"],
            "region_v2_feature_sha256": freezes["region_v2"].manifest["feature_sha256"],
        },
        "variants": variants,
    }


def write_results(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name.endswith(".gz"):
        with path.open("wb") as raw, gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0
        ) as zipped:
            zipped.write(encoded)
    else:
        path.write_bytes(encoded)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    raw_root = CONTACT_DET_ROOT / "raw"
    parser.add_argument("--evidence-manifest", type=Path, default=raw_root / "contact_evidence_manifest.json")
    parser.add_argument(
        "--evidence-results", type=Path, default=raw_root / "contact_evidence_score.json.gz"
    )
    parser.add_argument(
        "--region-v2-manifest",
        type=Path,
        default=raw_root / "region_v2" / "run_a" / "tree_contact_features_manifest.json",
    )
    parser.add_argument(
        "--region-v2-results", type=Path, default=raw_root / "region_v2" / "tree_contact_results.json.gz"
    )
    parser.add_argument(
        "--region-v1-manifest", type=Path, default=raw_root / "tree_trial" / "tree_contact_features_manifest.json"
    )
    parser.add_argument(
        "--region-v1-results",
        type=Path,
        default=raw_root / "tree_trial" / "tree_contact_results_with_frames.json.gz",
    )
    parser.add_argument("--data-root", type=Path, default=raw_root / "region_v2_inputs")
    parser.add_argument("--output", type=Path, default=raw_root / "contact_player_attribution_score.json.gz")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    payload = score(arguments)
    write_results(arguments.output, payload)
    print_tables(payload["variants"])
    print(f"\nwrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
