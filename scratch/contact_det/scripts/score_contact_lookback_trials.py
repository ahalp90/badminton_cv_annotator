"""Replay the bounded serve-lookback contact decision experiment.

This scorer keeps the historical five-row decision output unchanged. It
replays only the selected controls (B0, N+ and S-) and the two pre-specified
serve-lookback threshold policies (L- and SL-). Every event frame, timing
score, duplicate-removal decision and Top/Bottom answer is fixed before
timing or player-side labels are loaded.

The two new policies lower the exact preceding point in the retained score
grid. L- lowers it only for region_serve_lookback rows. SL- lowers it for
the union of region_rally_start and region_serve_lookback rows.
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

import score_contact_decision_trials as decision_scorer
import score_contact_evidence as evidence_scorer
import score_contact_player_attribution as attribution_scorer
import score_contact_rallies as rally_scorer
import score_tree_contact_detector as tree_scorer

RESULTS_SCHEMA = "contact-lookback-trials/1"
START_REGION_FIELD = decision_scorer.START_REGION_FIELD
SERVE_LOOKBACK_REGION_FIELD = "region_serve_lookback"

BASELINE_THRESHOLD = decision_scorer.BASELINE_THRESHOLD
LOWER_START_THRESHOLD = decision_scorer.LOWER_START_THRESHOLD
LOWER_SERVE_LOOKBACK_THRESHOLD = "lower_serve_lookback"
LOWER_START_OR_SERVE_LOOKBACK_THRESHOLD = "lower_start_or_serve_lookback"


@dataclass(frozen=True)
class LookbackTrial:
    """One selected decision policy in the bounded lookback experiment."""

    trial_id: str
    threshold_mode: str
    nms_radius_base30: int


TRIALS = (
    LookbackTrial("B0", BASELINE_THRESHOLD, 5),
    LookbackTrial("N+", BASELINE_THRESHOLD, 6),
    LookbackTrial("S−", LOWER_START_THRESHOLD, 5),
    LookbackTrial("L−", LOWER_SERVE_LOOKBACK_THRESHOLD, 5),
    LookbackTrial("SL−", LOWER_START_OR_SERVE_LOOKBACK_THRESHOLD, 5),
)


@dataclass(frozen=True)
class FrozenTrial:
    """A trial after event frames and player-side answers are fixed."""

    spec: LookbackTrial
    rows: np.ndarray
    predictions: dict[str, np.ndarray]
    attribution: dict[tuple[str, int], str | None]
    spans: tuple[rally_scorer.FixedSpan, ...]
    unassigned: tuple[rally_scorer.FixedEvent, ...]


def _previous_threshold(threshold: float) -> float:
    """Return the exact preceding point in the retained score grid."""
    return decision_scorer._previous_threshold(threshold)


def _model_rows(verified: tree_scorer.VerifiedFeatures) -> np.ndarray:
    return decision_scorer._model_rows(verified)


def _validate_row_alignment(candidate_rows: np.ndarray, model_rows: np.ndarray) -> None:
    """Require stable candidate-to-feature identities and both region flags."""
    if len(candidate_rows) != len(model_rows):
        raise ValueError("candidate and model row counts differ")
    identity = ["fixture", "interval_id", "frame"]
    if not np.array_equal(candidate_rows[identity], model_rows[identity]):
        raise ValueError("candidate and model row identities differ")
    fields = model_rows.dtype.names or ()
    for field in (START_REGION_FIELD, SERVE_LOOKBACK_REGION_FIELD):
        if field not in fields:
            raise ValueError(f"model rows do not contain {field}")


def _temporal_nms_with_eligibility(
    frames: np.ndarray,
    intervals: np.ndarray,
    scores: np.ndarray,
    eligible: np.ndarray,
    radius: int,
) -> np.ndarray:
    """Apply the historical score/frame tie order to eligible rows."""
    return decision_scorer._temporal_nms_with_eligibility(
        frames,
        intervals,
        scores,
        eligible,
        radius,
    )


def _effective_thresholds(
    candidate_rows: np.ndarray,
    model_rows: np.ndarray,
    spec: LookbackTrial,
) -> np.ndarray:
    """Apply one exact lower-grid step to the selected region mask."""
    thresholds = candidate_rows["threshold"].astype(np.float64, copy=True)
    fixture_names = rally_scorer._decode_rows_fixture(candidate_rows)
    start_rows = model_rows[START_REGION_FIELD].astype(bool)
    lookback_rows = model_rows[SERVE_LOOKBACK_REGION_FIELD].astype(bool)
    if spec.threshold_mode == LOWER_START_THRESHOLD:
        lower_mask = start_rows
    elif spec.threshold_mode == LOWER_SERVE_LOOKBACK_THRESHOLD:
        lower_mask = lookback_rows
    elif spec.threshold_mode == LOWER_START_OR_SERVE_LOOKBACK_THRESHOLD:
        # The OR mask assigns one value once, including rows in both regions.
        lower_mask = start_rows | lookback_rows
    elif spec.threshold_mode == BASELINE_THRESHOLD:
        lower_mask = np.zeros(len(candidate_rows), dtype=bool)
    else:
        raise ValueError(f"unknown threshold mode: {spec.threshold_mode!r}")

    for fixture in tree_scorer.FIXTURE_SPECS:
        fixture_mask = fixture_names == fixture
        if not fixture_mask.any():
            continue
        baseline_values = np.unique(thresholds[fixture_mask])
        if len(baseline_values) != 1:
            raise ValueError(f"{fixture}: candidate rows do not have one fold threshold")
        if spec.threshold_mode != BASELINE_THRESHOLD:
            thresholds[fixture_mask & lower_mask] = _previous_threshold(
                float(baseline_values[0])
            )
    return thresholds


def replay_trial(
    candidate_rows: np.ndarray,
    model_rows: np.ndarray,
    spec: LookbackTrial,
) -> np.ndarray:
    """Recompute one lookback decision row from frozen held-out scores."""
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
    """Replay only the five bounded lookback rows and require exact B0 identity."""
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
    """Replay shipped Top/Bottom once across all fixed trial event streams."""
    freeze_arguments = argparse.Namespace(
        region_v2_manifest=arguments.feature_manifest,
        region_v2_results=arguments.tree_results,
        region_v1_manifest=arguments.region_v1_manifest,
        region_v1_results=arguments.region_v1_results,
    )
    freezes = attribution_scorer._load_tree_freezes(freeze_arguments)
    variants = {
        f"region_v2/lookback/{trial_id}": dict(predictions)
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
        raise ValueError("shipped attribution does not cover every lookback event")
    return attribution


def _freeze_trial_streams(
    arguments: argparse.Namespace,
    evidence: evidence_scorer.VerifiedFreeze,
    candidate_rows: np.ndarray,
    model_rows: np.ndarray,
) -> tuple[FrozenTrial, ...]:
    """Fix all event streams, side answers, and current-span assignments."""
    rows_by_trial = replay_all_trials(candidate_rows, model_rows)
    predictions_by_trial = {
        trial_id: _prediction_frames(rows)
        for trial_id, rows in rows_by_trial.items()
    }
    attribution = _replay_attribution(arguments, predictions_by_trial)
    frozen: list[FrozenTrial] = []
    for spec in TRIALS:
        rows = rows_by_trial[spec.trial_id]
        trial_attribution = {
            (fixture, int(frame)): attribution[(fixture, int(frame))]
            for fixture, frames in predictions_by_trial[spec.trial_id].items()
            for frame in frames
        }
        events = rally_scorer.retained_events_from_scores(rows, trial_attribution)
        spans = rally_scorer.fixed_spans_from_evidence(evidence.evidence, events)
        frozen.append(
            FrozenTrial(
                spec,
                rows,
                predictions_by_trial[spec.trial_id],
                trial_attribution,
                spans,
                rally_scorer.unassigned_events(spans, events),
            )
        )
    return tuple(frozen)


def _decision_counts(rows: np.ndarray) -> dict[str, dict[str, int]]:
    return decision_scorer._decision_counts(rows)


def _event_side_metrics(
    trial: FrozenTrial,
    ground_truth: tree_scorer.GroundTruth,
    target_sides: Mapping[tuple[str, int], str],
) -> dict[str, dict[str, Any]]:
    """Return pooled and per-fixture event-side metrics at all tolerances."""
    all_counts = attribution_scorer._ground_truth_counts(
        ground_truth,
        list(tree_scorer.FIXTURE_SPECS),
    )
    prediction_count = sum(len(frames) for frames in trial.predictions.values())
    scores: dict[str, dict[str, Any]] = {}
    for tolerance in attribution_scorer.TOLERANCES_BASE30:
        matches = attribution_scorer._tree_matches(
            trial.predictions,
            trial.attribution,
            target_sides,
            ground_truth,
            tolerance,
        )
        scores[str(tolerance)] = {
            "pooled": attribution_scorer._score_slices(matches, all_counts, prediction_count),
            "fixtures": attribution_scorer._fixture_scores(
                matches,
                trial.predictions,
                ground_truth,
            ),
        }
    return scores


def _fully_correct_span_ids(
    trial: FrozenTrial,
    rallies_by_fixture: Mapping[str, Sequence[rally_scorer.RallyReference]],
    target_sides: Mapping[tuple[str, int], str],
    fps_by_fixture: Mapping[str, float],
    confidence_requirement: float,
) -> list[str]:
    """Return stable complete-rally identities at the primary tolerance."""
    tolerance_frames = {
        fixture: evidence_scorer.scale_base30_frames(
            rally_scorer.PRIMARY_TOLERANCE_BASE30,
            fps,
        )
        for fixture, fps in fps_by_fixture.items()
    }
    identities = [
        f"{score.fixture}:{score.span_id}"
        for span in trial.spans
        for score in (
            rally_scorer.evaluate_span(
                span,
                rallies_by_fixture.get(span.fixture, ()),
                target_sides,
                tolerance_frames[span.fixture],
                confidence_requirement,
            ),
        )
        if score.fully_correct
    ]
    return sorted(identities)


def _strict_identity_deltas(
    trial_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Report new and lost complete-rally identities for fixed comparisons."""
    by_id = {str(result["id"]): result for result in trial_results}
    pairs = (
        ("L−", "B0"),
        ("SL−", "S−"),
        ("B0", "N+"),
        ("S−", "N+"),
        ("L−", "N+"),
        ("SL−", "N+"),
    )
    output: dict[str, Any] = {
        "tolerance_base30": rally_scorer.PRIMARY_TOLERANCE_BASE30,
        "confidence_requirements": [0.0, 0.9],
        "comparisons": {},
    }
    if not trial_results:
        return output
    for candidate_id, control_id in pairs:
        candidate = by_id[candidate_id]["strict_identity"]["fully_correct_span_ids"]
        control = by_id[control_id]["strict_identity"]["fully_correct_span_ids"]
        at_requirement: dict[str, Any] = {}
        for requirement in ("0.00", "0.90"):
            candidate_set = set(candidate[requirement])
            control_set = set(control[requirement])
            at_requirement[requirement] = {
                "candidate_count": len(candidate_set),
                "control_count": len(control_set),
                "new_in_candidate": sorted(candidate_set - control_set),
                "lost_from_control": sorted(control_set - candidate_set),
            }
        output["comparisons"][f"{candidate_id}_vs_{control_id}"] = {
            "candidate": candidate_id,
            "control": control_id,
            "at_requirement": at_requirement,
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
        fixture_details[fixture] = {
            "baseline_threshold": float(max(thresholds)),
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
    strict_identity = {
        "tolerance_base30": rally_scorer.PRIMARY_TOLERANCE_BASE30,
        "fully_correct_span_ids": {
            f"{requirement:.2f}": _fully_correct_span_ids(
                trial,
                rallies_by_fixture,
                target_sides,
                fps_by_fixture,
                requirement,
            )
            for requirement in (0.0, 0.9)
        },
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
        "event_side_metrics": {
            "attribution": "shipped",
            "tolerances": _event_side_metrics(trial, ground_truth, target_sides),
        },
        "strict_identity": strict_identity,
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
    """Verify frozen inputs, build the label-blind trials, then score labels."""
    verified = tree_scorer.verify_freeze(arguments.feature_manifest)
    verified_candidates = tree_scorer.verify_candidate_scores(
        arguments.candidate_manifest,
        verified,
        arguments.tree_results,
    )
    if tree_scorer._result_variant(verified_candidates.tree_result) != tree_scorer.CANDIDATE_VARIANT:
        raise ValueError("lookback trials require the retained baseline HGB physics scores")
    evidence = evidence_scorer.verify_freeze(arguments.evidence_manifest)
    trials = _freeze_trial_streams(
        arguments,
        evidence,
        verified_candidates.rows,
        _model_rows(verified),
    )

    # Timing and player-side labels are read only after all trial outputs are fixed.
    rallies_by_fixture = rally_scorer._load_timing_rallies()
    ground_truth = decision_scorer._ground_truth_from_rallies(rallies_by_fixture)
    target_sides = rally_scorer._load_side_labels()

    trial_results = [
        _trial_result(trial, ground_truth, rallies_by_fixture, target_sides)
        for trial in trials
    ]
    return {
        "schema": RESULTS_SCHEMA,
        "fixture_set": list(tree_scorer.FIXTURE_SPECS),
        "selected_variant": tree_scorer.CANDIDATE_VARIANT,
        "model_refit": False,
        "start_region_field": START_REGION_FIELD,
        "serve_lookback_region_field": SERVE_LOOKBACK_REGION_FIELD,
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
        "trials": trial_results,
        "strict_identity_deltas": _strict_identity_deltas(trial_results),
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
