"""Measure the frozen label-blind Phase 3 contact shortlist.

The shortlist keeps every selected N+ event and its strongest distinct nearby
alternative from the verified held-out scores. Its identities and size are
asserted before timing labels load. The shortlist is an overcomplete candidate
set, not a replacement detector event stream.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
import score_contact_rallies as rally_scorer
import score_tree_contact_detector as tree_scorer

RESULTS_SCHEMA = "contact-shortlist-coverage/1"
SELECTED_TRIAL_ID = "N+"
AREA_RADIUS_BASE30 = 10
EXCLUSION_RADIUS_BASE30 = 6
SCORE_TOLERANCES_BASE30 = (5, 10)
PRIMARY_TOLERANCE_BASE30 = 10

EXPECTED_ANCHOR_COUNT = 3_238
EXPECTED_ALTERNATIVE_SELECTION_COUNT = 3_238
EXPECTED_DISTINCT_ALTERNATIVE_COUNT = 3_203
EXPECTED_REPEATED_ALTERNATIVE_COUNT = 35
EXPECTED_ALTERNATIVES_ALREADY_ANCHORS = 136
EXPECTED_SHORTLIST_COUNT = 6_305
EXPECTED_NPLUS_PRIMARY_MATCHED = 2_825
EXPECTED_NPLUS_PRIMARY_MISSED = 303
REQUIRED_NEW_MATCHES = 152
MAX_SIZE_MULTIPLIER = 2.0


@dataclass(frozen=True)
class FrozenShortlist:
    """The selected events and alternative rows fixed before label loading."""

    nplus_rows: np.ndarray
    anchor_indices: np.ndarray
    alternative_indices: np.ndarray
    shortlist_indices: np.ndarray


def _selected_trial() -> decision_scorer.DecisionTrial:
    matches = [trial for trial in decision_scorer.TRIALS if trial.trial_id == SELECTED_TRIAL_ID]
    if len(matches) != 1:
        raise ValueError(f"expected one {SELECTED_TRIAL_ID} decision trial")
    trial = matches[0]
    if (
        trial.threshold_mode != decision_scorer.BASELINE_THRESHOLD
        or trial.nms_radius_base30 != EXCLUSION_RADIUS_BASE30
    ):
        raise ValueError("selected N+ decision policy differs from the shortlist contract")
    return trial


def _best_alternative(
    rows: np.ndarray,
    local_indices: np.ndarray,
    anchor_index: int,
    exclusion_radius: int,
) -> int:
    anchor_frame = int(rows["frame"][anchor_index])
    separated = local_indices[
        np.abs(rows["frame"][local_indices].astype(np.int64) - anchor_frame)
        > exclusion_radius
    ]
    if not len(separated):
        raise ValueError(f"anchor frame {anchor_frame} has no distinct shortlist alternative")
    return int(
        min(
            separated,
            key=lambda index: (-float(rows["timing_score"][index]), int(rows["frame"][index])),
        )
    )


def freeze_shortlist(candidate_rows: np.ndarray, model_rows: np.ndarray) -> FrozenShortlist:
    """Build the selected local shortlist without reading labels."""
    trials = decision_scorer.replay_all_trials(candidate_rows, model_rows)
    nplus_rows = trials[SELECTED_TRIAL_ID]
    fixture_names = rally_scorer._decode_rows_fixture(nplus_rows)
    anchor_indices = np.flatnonzero(
        nplus_rows["decision"] == tree_scorer.CANDIDATE_RETAINED
    ).astype(np.int32)
    alternative_indices: list[int] = []
    for fixture, (_video_id, fps) in tree_scorer.FIXTURE_SPECS.items():
        fixture_indices = np.flatnonzero(fixture_names == fixture)
        fixture_anchors = anchor_indices[fixture_names[anchor_indices] == fixture]
        area_radius = tree_scorer._scaled_frames(AREA_RADIUS_BASE30, fps)
        exclusion_radius = tree_scorer._scaled_frames(EXCLUSION_RADIUS_BASE30, fps)
        for anchor_index in fixture_anchors:
            anchor_frame = int(nplus_rows["frame"][anchor_index])
            anchor_interval = int(nplus_rows["interval_id"][anchor_index])
            local_indices = fixture_indices[
                (nplus_rows["interval_id"][fixture_indices] == anchor_interval)
                & (np.abs(nplus_rows["frame"][fixture_indices].astype(np.int64) - anchor_frame) <= area_radius)
            ]
            alternative_indices.append(
                _best_alternative(
                    nplus_rows,
                    local_indices,
                    int(anchor_index),
                    exclusion_radius,
                )
            )

    alternative_array = np.asarray(alternative_indices, dtype=np.int32)
    # Sorting source-row indices restores source order after the two sets unite.
    shortlist_indices = np.unique(np.concatenate([anchor_indices, alternative_array])).astype(
        np.int32
    )
    return FrozenShortlist(
        nplus_rows,
        anchor_indices,
        alternative_array,
        shortlist_indices,
    )


def validate_frozen_shortlist(frozen: FrozenShortlist) -> None:
    """Require the exact label-free pilot identity counts before scoring."""
    anchor_identities = set(map(tuple, frozen.nplus_rows[frozen.anchor_indices][
        ["fixture", "interval_id", "frame"]
    ].tolist()))
    alternative_identities = [
        tuple(row)
        for row in frozen.nplus_rows[frozen.alternative_indices][
            ["fixture", "interval_id", "frame"]
        ].tolist()
    ]
    distinct_alternatives = set(alternative_identities)
    shortlist_identities = set(map(tuple, frozen.nplus_rows[frozen.shortlist_indices][
        ["fixture", "interval_id", "frame"]
    ].tolist()))
    actual = {
        "anchors": len(frozen.anchor_indices),
        "alternative_selections": len(frozen.alternative_indices),
        "distinct_alternatives": len(distinct_alternatives),
        "repeated_alternative_selections": len(alternative_identities) - len(distinct_alternatives),
        "alternatives_already_anchors": len(distinct_alternatives & anchor_identities),
        "shortlist": len(frozen.shortlist_indices),
        "shortlist_identities": len(shortlist_identities),
    }
    expected = {
        "anchors": EXPECTED_ANCHOR_COUNT,
        "alternative_selections": EXPECTED_ALTERNATIVE_SELECTION_COUNT,
        "distinct_alternatives": EXPECTED_DISTINCT_ALTERNATIVE_COUNT,
        "repeated_alternative_selections": EXPECTED_REPEATED_ALTERNATIVE_COUNT,
        "alternatives_already_anchors": EXPECTED_ALTERNATIVES_ALREADY_ANCHORS,
        "shortlist": EXPECTED_SHORTLIST_COUNT,
        "shortlist_identities": EXPECTED_SHORTLIST_COUNT,
    }
    if actual != expected:
        raise ValueError(f"frozen shortlist identity counts differ: {actual}")


def _prediction_frames(rows: np.ndarray, selected_indices: np.ndarray) -> dict[str, np.ndarray]:
    selected = rows[selected_indices]
    fixture_names = rally_scorer._decode_rows_fixture(selected)
    predictions: dict[str, np.ndarray] = {}
    for fixture in tree_scorer.FIXTURE_SPECS:
        frames = selected["frame"][fixture_names == fixture].astype(np.int32)
        if len(frames) != len(np.unique(frames)):
            raise ValueError(f"{fixture}: shortlist contains duplicate source frames")
        predictions[fixture] = frames
    return predictions


def _coverage_counts(metrics: Mapping[str, int | float]) -> dict[str, int | float]:
    matched = int(metrics["matched"])
    candidates = int(metrics["predictions"])
    ground_truth = int(metrics["ground_truth"])
    return {
        "candidate_count": candidates,
        "matched_contacts": matched,
        "missed_contacts": ground_truth - matched,
        "coverage": float(metrics["recall"]),
        "unmatched_candidates": candidates - matched,
        "serve_contacts": int(metrics["serve_total"]),
        "serve_matched": int(metrics["serve_matched"]),
        "serve_coverage": float(metrics["serve_recall"]),
        "nonserve_contacts": int(metrics["nonserve_total"]),
        "nonserve_matched": int(metrics["nonserve_matched"]),
        "nonserve_coverage": float(metrics["nonserve_recall"]),
        "median_absolute_offset": metrics["median_absolute_offset"],
    }


def _score_predictions(
    ground_truth: tree_scorer.GroundTruth,
    predictions: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for tolerance in SCORE_TOLERANCES_BASE30:
        output[str(tolerance)] = {
            "pooled": _coverage_counts(
                tree_scorer._event_counts(ground_truth, predictions, tolerance)
            ),
            "fixtures": {
                fixture: _coverage_counts(
                    tree_scorer._event_counts(
                        ground_truth,
                        {fixture: predictions[fixture]},
                        tolerance,
                        [fixture],
                    )
                )
                for fixture in tree_scorer.FIXTURE_SPECS
            },
        }
    return output


def _matched_contact_identities(
    ground_truth: tree_scorer.GroundTruth,
    predictions: Mapping[str, np.ndarray],
    tolerance_base30: int,
) -> set[tuple[str, int]]:
    """Identify matched labelled contacts by fixture and ground-truth row."""
    identities: set[tuple[str, int]] = set()
    for fixture, (_video_id, fps) in tree_scorer.FIXTURE_SPECS.items():
        tolerance = tree_scorer._scaled_frames(tolerance_base30, fps)
        matches = tree_scorer._greedy_matches(
            ground_truth.frames[fixture],
            predictions[fixture],
            tolerance,
        )
        identities.update((fixture, gt_index) for gt_index, _prediction_index, _offset in matches)
    return identities


def _gate_result(anchor_count: int, shortlist_count: int, newly_matched: int) -> dict[str, Any]:
    size_multiplier = shortlist_count / anchor_count
    size_pass = size_multiplier <= MAX_SIZE_MULTIPLIER
    coverage_pass = newly_matched >= REQUIRED_NEW_MATCHES
    passed = size_pass and coverage_pass
    return {
        "maximum_size_multiplier": MAX_SIZE_MULTIPLIER,
        "required_newly_matched_contacts": REQUIRED_NEW_MATCHES,
        "size_pass": size_pass,
        "coverage_pass": coverage_pass,
        "pass": passed,
        "decision": "continue" if passed else "stop",
    }


def _result_payload(
    arguments: argparse.Namespace,
    verified: tree_scorer.VerifiedFeatures,
    verified_candidates: tree_scorer.VerifiedCandidateScores,
    frozen: FrozenShortlist,
    ground_truth: tree_scorer.GroundTruth,
) -> dict[str, Any]:
    anchor_predictions = _prediction_frames(frozen.nplus_rows, frozen.anchor_indices)
    shortlist_predictions = _prediction_frames(frozen.nplus_rows, frozen.shortlist_indices)
    anchor_scores = _score_predictions(ground_truth, anchor_predictions)
    shortlist_scores = _score_predictions(ground_truth, shortlist_predictions)
    anchor_primary = anchor_scores[str(PRIMARY_TOLERANCE_BASE30)]["pooled"]
    shortlist_primary = shortlist_scores[str(PRIMARY_TOLERANCE_BASE30)]["pooled"]
    if (
        anchor_primary["matched_contacts"] != EXPECTED_NPLUS_PRIMARY_MATCHED
        or anchor_primary["missed_contacts"] != EXPECTED_NPLUS_PRIMARY_MISSED
    ):
        raise ValueError("N+ primary coverage differs from the frozen Phase 2 result")

    added_candidates = len(frozen.shortlist_indices) - len(frozen.anchor_indices)
    net_matched = int(shortlist_primary["matched_contacts"]) - int(
        anchor_primary["matched_contacts"]
    )
    anchor_matches = _matched_contact_identities(
        ground_truth,
        anchor_predictions,
        PRIMARY_TOLERANCE_BASE30,
    )
    shortlist_matches = _matched_contact_identities(
        ground_truth,
        shortlist_predictions,
        PRIMARY_TOLERANCE_BASE30,
    )
    newly_matched = len(shortlist_matches - anchor_matches)
    lost_matches = len(anchor_matches - shortlist_matches)
    if newly_matched - lost_matches != net_matched:
        raise ValueError("matched-contact identity comparison differs from the pooled totals")
    size_multiplier = len(frozen.shortlist_indices) / len(frozen.anchor_indices)
    distinct_alternatives = np.unique(frozen.alternative_indices)
    return {
        "schema": RESULTS_SCHEMA,
        "fixture_set": list(tree_scorer.FIXTURE_SPECS),
        "selected_tree_variant": tree_scorer.CANDIDATE_VARIANT,
        "labels_read_after_shortlist_fixed": True,
        "inputs": {
            "feature_manifest_sha256": tree_scorer._sha256(arguments.feature_manifest),
            "feature_sha256": verified.manifest["feature_sha256"],
            "tree_result_sha256": verified_candidates.manifest["tree_result_sha256"],
            "candidate_manifest_sha256": tree_scorer._sha256(arguments.candidate_manifest),
            "candidate_scores_sha256": verified_candidates.manifest["candidate_sha256"],
        },
        "policy": {
            "anchor_trial": SELECTED_TRIAL_ID,
            "area_radius_base30": AREA_RADIUS_BASE30,
            "area_boundary": "inclusive",
            "same_interval_only": True,
            "alternative_exclusion_radius_base30": EXCLUSION_RADIUS_BASE30,
            "alternative_exclusion_boundary": "inclusive",
            "alternative_score_cutoff": None,
            "alternative_tie_break": "earlier_frame",
            "deduplication_identity": ["fixture", "interval_id", "frame"],
        },
        "label_free_counts": {
            "anchors": len(frozen.anchor_indices),
            "alternative_selections": len(frozen.alternative_indices),
            "distinct_alternatives": len(distinct_alternatives),
            "repeated_alternative_selections": len(frozen.alternative_indices)
            - len(distinct_alternatives),
            "alternatives_already_anchors": len(
                np.intersect1d(distinct_alternatives, frozen.anchor_indices)
            ),
            "shortlist": len(frozen.shortlist_indices),
            "added_candidates": added_candidates,
            "size_multiplier_over_nplus": size_multiplier,
        },
        "nplus": {"scores": anchor_scores},
        "shortlist": {"scores": shortlist_scores},
        "primary_comparison": {
            "tolerance_base30": PRIMARY_TOLERANCE_BASE30,
            "newly_matched_contacts": newly_matched,
            "lost_nplus_matched_contacts": lost_matches,
            "net_matched_contact_change": net_matched,
            "added_candidates": added_candidates,
            "added_candidates_per_newly_matched_contact": (
                added_candidates / newly_matched if newly_matched else None
            ),
            "additional_unmatched_candidates": int(shortlist_primary["unmatched_candidates"])
            - int(anchor_primary["unmatched_candidates"]),
        },
        "gate": _gate_result(
            len(frozen.anchor_indices),
            len(frozen.shortlist_indices),
            newly_matched,
        ),
    }


def score(arguments: argparse.Namespace) -> dict[str, Any]:
    verified = tree_scorer.verify_freeze(arguments.feature_manifest)
    verified_candidates = tree_scorer.verify_candidate_scores(
        arguments.candidate_manifest,
        verified,
        arguments.tree_results,
    )
    if tree_scorer._result_variant(verified_candidates.tree_result) != tree_scorer.CANDIDATE_VARIANT:
        raise ValueError("shortlist requires the retained baseline HGB physics scores")
    _selected_trial()
    frozen = freeze_shortlist(
        verified_candidates.rows,
        decision_scorer._model_rows(verified),
    )
    validate_frozen_shortlist(frozen)

    # All candidate identities, scores and counts are fixed above. Only timing
    # labels are needed for the coverage measurement below.
    ground_truth = tree_scorer._load_ground_truth()
    return _result_payload(arguments, verified, verified_candidates, frozen, ground_truth)


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
