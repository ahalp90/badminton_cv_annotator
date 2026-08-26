"""Replay the fixed Phase 2 contact decision trials from held-out scores.

The script never refits a model. It verifies the retained region-v2 feature
freeze, tree result and held-out candidate scores, then applies the five
pre-specified score and duplicate-removal choices. Every event stream and
Top/Bottom answer is fixed before timing or player-side labels load.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

MODULE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import score_contact_evidence as evidence_scorer
import score_contact_player_attribution as attribution_scorer
import score_contact_rallies as rally_scorer
import score_tree_contact_detector as tree_scorer

RESULTS_SCHEMA = "contact-decision-trials/1"
START_REGION_FIELD = "region_rally_start"

BASELINE_THRESHOLD = "baseline"
LOWER_THRESHOLD = "lower"
LOWER_START_THRESHOLD = "lower_start"


@dataclass(frozen=True)
class DecisionTrial:
    """One fixed held-out-score decision row."""

    trial_id: str
    threshold_mode: str
    nms_radius_base30: int


TRIALS = (
    DecisionTrial("B0", BASELINE_THRESHOLD, 5),
    DecisionTrial("T−", LOWER_THRESHOLD, 5),
    DecisionTrial("N−", BASELINE_THRESHOLD, 4),
    DecisionTrial("N+", BASELINE_THRESHOLD, 6),
    DecisionTrial("S−", LOWER_START_THRESHOLD, 5),
)


@dataclass(frozen=True)
class FrozenTrial:
    """A decision row after its events and player-side answers are fixed."""

    spec: DecisionTrial
    rows: np.ndarray
    predictions: dict[str, np.ndarray]
    spans: tuple[rally_scorer.FixedSpan, ...]
    unassigned: tuple[rally_scorer.FixedEvent, ...]


def _previous_threshold(threshold: float) -> float:
    """Return the exact preceding point in the scorer's frozen grid."""
    grid = tree_scorer._threshold_candidates(np.empty(0, dtype=np.float64))
    matches = np.flatnonzero(grid == threshold)
    if len(matches) != 1 or matches[0] == 0:
        raise ValueError(f"threshold has no preceding grid point: {threshold!r}")
    return float(grid[int(matches[0]) - 1])


def _next_threshold(threshold: float) -> float:
    """Return the exact following point in the scorer's frozen grid."""
    grid = tree_scorer._threshold_candidates(np.empty(0, dtype=np.float64))
    matches = np.flatnonzero(grid == threshold)
    if len(matches) != 1 or matches[0] == len(grid) - 1:
        raise ValueError(f"threshold has no following grid point: {threshold!r}")
    return float(grid[int(matches[0]) + 1])


def _model_rows(verified: tree_scorer.VerifiedFeatures) -> np.ndarray:
    return verified.rows[tree_scorer.seeded_region_mask(verified.rows)]


def _validate_row_alignment(candidate_rows: np.ndarray, model_rows: np.ndarray) -> None:
    """Require the stable candidate-to-feature identity before using flags."""
    if len(candidate_rows) != len(model_rows):
        raise ValueError("candidate and model row counts differ")
    identity = ["fixture", "interval_id", "frame"]
    if not np.array_equal(candidate_rows[identity], model_rows[identity]):
        raise ValueError("candidate and model row identities differ")
    if START_REGION_FIELD not in (model_rows.dtype.names or ()):
        raise ValueError(f"model rows do not contain {START_REGION_FIELD}")


def _temporal_nms_with_eligibility(
    frames: np.ndarray,
    intervals: np.ndarray,
    scores: np.ndarray,
    eligible: np.ndarray,
    radius: int,
) -> np.ndarray:
    """Apply the existing score/frame tie order to pre-qualified rows."""
    accepted: list[int] = []
    for interval_id in np.unique(intervals):
        local = np.flatnonzero((intervals == interval_id) & eligible)
        order = sorted(local, key=lambda index: (-scores[index], frames[index]))
        kept: list[int] = []
        for index in order:
            if all(abs(int(frames[index]) - int(frames[other])) > radius for other in kept):
                kept.append(int(index))
        accepted.extend(kept)
    return np.asarray(sorted(accepted, key=lambda index: frames[index]), dtype=np.int32)


def _effective_thresholds(
    candidate_rows: np.ndarray,
    model_rows: np.ndarray,
    spec: DecisionTrial,
) -> np.ndarray:
    thresholds = candidate_rows["threshold"].astype(np.float64, copy=True)
    fixture_names = rally_scorer._decode_rows_fixture(candidate_rows)
    start_rows = model_rows[START_REGION_FIELD].astype(bool)
    for fixture in tree_scorer.FIXTURE_SPECS:
        fixture_rows = fixture_names == fixture
        if not fixture_rows.any():
            continue
        baseline_values = np.unique(thresholds[fixture_rows])
        if len(baseline_values) != 1:
            raise ValueError(f"{fixture}: candidate rows do not have one fold threshold")
        lower = _previous_threshold(float(baseline_values[0]))
        if spec.threshold_mode == LOWER_THRESHOLD:
            thresholds[fixture_rows] = lower
        elif spec.threshold_mode == LOWER_START_THRESHOLD:
            thresholds[fixture_rows & start_rows] = lower
        elif spec.threshold_mode != BASELINE_THRESHOLD:
            raise ValueError(f"unknown threshold mode: {spec.threshold_mode!r}")
    return thresholds


def replay_trial(
    candidate_rows: np.ndarray,
    model_rows: np.ndarray,
    spec: DecisionTrial,
) -> np.ndarray:
    """Recompute one decision row from held-out scores without refitting."""
    _validate_row_alignment(candidate_rows, model_rows)
    output = candidate_rows.copy()
    effective_thresholds = _effective_thresholds(candidate_rows, model_rows, spec)
    output["threshold"] = effective_thresholds
    output["decision"] = tree_scorer.CANDIDATE_BELOW_THRESHOLD
    fixture_names = rally_scorer._decode_rows_fixture(output)
    scores = output["timing_score"]
    for fixture, (_video_id, fps) in tree_scorer.FIXTURE_SPECS.items():
        fixture_indices = np.flatnonzero(fixture_names == fixture)
        fixture_rows = output[fixture_indices]
        eligible = scores[fixture_indices] >= effective_thresholds[fixture_indices]
        output["decision"][fixture_indices[eligible]] = tree_scorer.CANDIDATE_NEARBY_DUPLICATE
        kept = _temporal_nms_with_eligibility(
            fixture_rows["frame"],
            fixture_rows["interval_id"],
            scores[fixture_indices],
            eligible,
            tree_scorer._scaled_frames(spec.nms_radius_base30, fps),
        )
        output["decision"][fixture_indices[kept]] = tree_scorer.CANDIDATE_RETAINED
    return output


def replay_all_trials(candidate_rows: np.ndarray, model_rows: np.ndarray) -> dict[str, np.ndarray]:
    """Replay the five frozen rows and require exact baseline identity."""
    rows_by_trial = {
        spec.trial_id: replay_trial(candidate_rows, model_rows, spec)
        for spec in TRIALS
    }
    baseline = rows_by_trial["B0"]
    if not np.array_equal(baseline, candidate_rows):
        raise ValueError("B0 replay differs from the verified baseline decisions")
    return rows_by_trial


def _prediction_frames(rows: np.ndarray) -> dict[str, np.ndarray]:
    fixture_names = rally_scorer._decode_rows_fixture(rows)
    return {
        fixture: rows[
            (fixture_names == fixture)
            & (rows["decision"] == tree_scorer.CANDIDATE_RETAINED)
        ]["frame"].astype(np.int32)
        for fixture in tree_scorer.FIXTURE_SPECS
    }


def _replay_attribution(
    arguments: argparse.Namespace,
    predictions_by_trial: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[tuple[str, int], str | None]:
    """Replay Top/Bottom once across the union of all fixed event streams."""
    freeze_arguments = argparse.Namespace(
        region_v2_manifest=arguments.feature_manifest,
        region_v2_results=arguments.tree_results,
        region_v1_manifest=arguments.region_v1_manifest,
        region_v1_results=arguments.region_v1_results,
    )
    freezes = attribution_scorer._load_tree_freezes(freeze_arguments)
    variants = {
        f"region_v2/decision/{trial_id}": dict(predictions)
        for trial_id, predictions in predictions_by_trial.items()
    }
    attribution = attribution_scorer._shipped_attribution_map(arguments.data_root, freezes, variants)
    expected = {
        (fixture, int(frame))
        for predictions in predictions_by_trial.values()
        for fixture, frames in predictions.items()
        for frame in frames
    }
    if set(attribution) != expected:
        raise ValueError("shipped attribution does not cover every decision-trial event")
    return attribution


def _ground_truth_from_rallies(
    rallies_by_fixture: Mapping[str, Sequence[rally_scorer.RallyReference]],
) -> tree_scorer.GroundTruth:
    """Reuse the timing-rally load for pooled contact metrics."""
    frames: dict[str, np.ndarray] = {}
    serves: dict[str, set[int]] = {}
    rally_count = 0
    for fixture in tree_scorer.FIXTURE_SPECS:
        rallies = tuple(rallies_by_fixture[fixture])
        rally_count += len(rallies)
        frames[fixture] = np.asarray(
            [frame for rally in rallies for frame in rally.frames],
            dtype=np.int32,
        )
        serves[fixture] = {rally.frames[0] for rally in rallies}
    return tree_scorer.GroundTruth(frames, serves, rally_count)


def _freeze_trial_streams(
    arguments: argparse.Namespace,
    evidence: evidence_scorer.VerifiedFreeze,
    candidate_rows: np.ndarray,
    model_rows: np.ndarray,
) -> tuple[FrozenTrial, ...]:
    """Fix all decisions, side answers, event lists and span assignments."""
    rows_by_trial = replay_all_trials(candidate_rows, model_rows)
    predictions_by_trial = {
        trial_id: _prediction_frames(rows)
        for trial_id, rows in rows_by_trial.items()
    }
    attribution = _replay_attribution(arguments, predictions_by_trial)
    frozen: list[FrozenTrial] = []
    for spec in TRIALS:
        rows = rows_by_trial[spec.trial_id]
        events = rally_scorer.retained_events_from_scores(rows, attribution)
        spans = rally_scorer.fixed_spans_from_evidence(evidence.evidence, events)
        frozen.append(
            FrozenTrial(
                spec,
                rows,
                predictions_by_trial[spec.trial_id],
                spans,
                rally_scorer.unassigned_events(spans, events),
            )
        )
    return tuple(frozen)


def _decision_counts(rows: np.ndarray) -> dict[str, dict[str, int]]:
    fixture_names = rally_scorer._decode_rows_fixture(rows)
    output: dict[str, dict[str, int]] = {}
    for fixture in tree_scorer.FIXTURE_SPECS:
        fixture_rows = rows[fixture_names == fixture]
        output[fixture] = {
            name: int(np.count_nonzero(fixture_rows["decision"] == code))
            for name, code in tree_scorer.CANDIDATE_DECISIONS.items()
        }
    return output


def _trial_result(
    trial: FrozenTrial,
    ground_truth: tree_scorer.GroundTruth,
    rallies_by_fixture: Mapping[str, Sequence[rally_scorer.RallyReference]],
    target_sides: Mapping[tuple[str, int], str],
) -> dict[str, Any]:
    fixture_rows = rally_scorer._decode_rows_fixture(trial.rows)
    fixture_details: dict[str, dict[str, Any]] = {}
    for fixture, (_video_id, fps) in tree_scorer.FIXTURE_SPECS.items():
        selected = trial.rows[fixture_rows == fixture]
        thresholds = np.unique(selected["threshold"])
        if trial.spec.threshold_mode == LOWER_THRESHOLD:
            baseline_threshold = _next_threshold(float(thresholds[0]))
        else:
            baseline_threshold = float(max(thresholds))
        fixture_details[fixture] = {
            "baseline_threshold": baseline_threshold,
            "effective_thresholds": [float(value) for value in thresholds],
            "nms_radius_frames": tree_scorer._scaled_frames(trial.spec.nms_radius_base30, fps),
            "prediction_count": len(trial.predictions[fixture]),
            "prediction_frames": trial.predictions[fixture].tolist(),
            "metrics": {
                str(tolerance): tree_scorer._event_counts(
                    ground_truth,
                    {fixture: trial.predictions[fixture]},
                    tolerance,
                    [fixture],
                )
                for tolerance in tree_scorer.TOLERANCES_BASE30
            },
        }
    fps_by_fixture = {
        fixture: fps
        for fixture, (_video_id, fps) in tree_scorer.FIXTURE_SPECS.items()
    }
    return {
        "id": trial.spec.trial_id,
        "policy": asdict(trial.spec),
        "decision_counts": _decision_counts(trial.rows),
        "fixtures": fixture_details,
        "metrics": {
            str(tolerance): tree_scorer._event_counts(
                ground_truth,
                trial.predictions,
                tolerance,
            )
            for tolerance in tree_scorer.TOLERANCES_BASE30
        },
        "primary": rally_scorer.score_strict_rallies(
            trial.spans,
            rallies_by_fixture,
            target_sides,
            fps_by_fixture,
            tolerance_base30=rally_scorer.PRIMARY_TOLERANCE_BASE30,
        ),
        "sensitivity": rally_scorer.score_strict_rallies(
            trial.spans,
            rallies_by_fixture,
            target_sides,
            fps_by_fixture,
            tolerance_base30=rally_scorer.SENSITIVITY_TOLERANCE_BASE30,
        ),
        "unassigned_event_count": len(trial.unassigned),
    }


def score(arguments: argparse.Namespace) -> dict[str, Any]:
    verified = tree_scorer.verify_freeze(arguments.feature_manifest)
    verified_candidates = tree_scorer.verify_candidate_scores(
        arguments.candidate_manifest,
        verified,
        arguments.tree_results,
    )
    if tree_scorer._result_variant(verified_candidates.tree_result) != tree_scorer.CANDIDATE_VARIANT:
        raise ValueError("decision trials require the retained baseline HGB physics scores")
    evidence = evidence_scorer.verify_freeze(arguments.evidence_manifest)
    trials = _freeze_trial_streams(
        arguments,
        evidence,
        verified_candidates.rows,
        _model_rows(verified),
    )

    # Every event, score, span assignment and Top/Bottom answer is fixed above.
    # Timing labels and player-side labels load only after that boundary.
    rallies_by_fixture = rally_scorer._load_timing_rallies()
    ground_truth = _ground_truth_from_rallies(rallies_by_fixture)
    target_sides = rally_scorer._load_side_labels()

    return {
        "schema": RESULTS_SCHEMA,
        "fixture_set": list(tree_scorer.FIXTURE_SPECS),
        "selected_variant": tree_scorer.CANDIDATE_VARIANT,
        "model_refit": False,
        "start_region_field": START_REGION_FIELD,
        "labels_read_after_predictions_fixed": True,
        "inputs": {
            "feature_manifest_sha256": tree_scorer._sha256(arguments.feature_manifest),
            "feature_sha256": verified.manifest["feature_sha256"],
            "tree_result_sha256": verified_candidates.manifest["tree_result_sha256"],
            "candidate_manifest_sha256": tree_scorer._sha256(arguments.candidate_manifest),
            "candidate_scores_sha256": verified_candidates.manifest["candidate_sha256"],
            "evidence_manifest_sha256": tree_scorer._sha256(arguments.evidence_manifest),
            "evidence_sha256": evidence.manifest["evidence_sha256"],
            "region_v1_manifest_sha256": tree_scorer._sha256(arguments.region_v1_manifest),
            "region_v1_result_sha256": tree_scorer._sha256(arguments.region_v1_results),
        },
        "trials": [
            _trial_result(trial, ground_truth, rallies_by_fixture, target_sides)
            for trial in trials
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    raw_root = rally_scorer.CONTACT_DET_ROOT / "raw"
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=raw_root / "region_v2" / "run_a" / "tree_contact_features_manifest.json",
    )
    parser.add_argument(
        "--tree-results",
        type=Path,
        default=raw_root / "region_v2" / "tree_contact_results.json.gz",
    )
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument(
        "--evidence-manifest",
        type=Path,
        default=raw_root / "contact_evidence_manifest.json",
    )
    parser.add_argument(
        "--region-v1-manifest",
        type=Path,
        default=raw_root / "tree_trial" / "tree_contact_features_manifest.json",
    )
    parser.add_argument(
        "--region-v1-results",
        type=Path,
        default=raw_root / "tree_trial" / "tree_contact_results_with_frames.json.gz",
    )
    parser.add_argument("--data-root", type=Path, default=raw_root / "region_v2_inputs")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    payload = score(arguments)
    rally_scorer.write_results(arguments.output, payload)
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
