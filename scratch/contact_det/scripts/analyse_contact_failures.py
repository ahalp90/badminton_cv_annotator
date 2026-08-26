"""Audit missed contacts in the frozen held-out HGB contact stream.

This is an observational pass. It verifies the label-blind feature, candidate
score, and contact-evidence freezes first, then loads timing labels only to
identify the contacts that the already-retained HGB rows missed. It never
changes the retained event stream.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from collections.abc import Mapping, Sequence
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

RESULTS_SCHEMA = "tree-contact-missed-contact-audit/1"
SELECTED_VARIANT = tree_scorer.CANDIDATE_VARIANT
PRIMARY_TOLERANCE_BASE30 = 10
SENSITIVITY_TOLERANCE_BASE30 = 5
AUDIT_PEAK_COUNT = 3
RETAINED_DECISION = tree_scorer.CANDIDATE_RETAINED
NMS_RADIUS_BASE30 = tree_scorer.NMS_RADIUS_BASE30

DECISION_NAMES = {value: name for name, value in tree_scorer.CANDIDATE_DECISIONS.items()}
EVIDENCE_DEFINITION = (
    "For each missed GT frame, inspect all frozen feature rows in the same fixture "
    "within the primary ±10 base-30-frame audit window. Shuttle is available when "
    "any row has shuttle_visible_t+0 > 0.5. Pose is available when either "
    "pose_valid_top_t+0 or pose_valid_bot_t+0 is > 0.5. Wrist is available when "
    "either wrist_valid_top_t+0 or wrist_valid_bot_t+0 is > 0.5."
)


def _decode_fixture_names(rows: np.ndarray) -> np.ndarray:
    """Decode the fixed-width fixture identity used by the structured tables."""
    if rows.dtype.names is None or "fixture" not in rows.dtype.names:
        raise ValueError("structured rows must contain a fixture field")
    values = rows["fixture"]
    if values.dtype.kind == "S":
        return np.char.decode(values, "ascii")
    if values.dtype.kind == "U":
        return values.astype(str)
    raise TypeError("fixture field must contain fixed-width text")


def _require_fields(rows: np.ndarray, fields: Sequence[str], name: str) -> None:
    """Raise a useful error when a frozen structured table lacks a field."""
    actual = set(rows.dtype.names or ())
    missing = sorted(set(fields) - actual)
    if missing:
        raise ValueError(f"{name} is missing fields: {missing}")


def scale_tolerance(base30: int, fps: float) -> int:
    """Scale a base-30 tolerance using the repository's frame-count rule."""
    if isinstance(base30, bool) or not isinstance(base30, int) or base30 < 0:
        raise ValueError("base30 tolerance must be a non-negative integer")
    if not math.isfinite(float(fps)) or fps <= 0:
        raise ValueError("fps must be a positive finite number")
    return tree_scorer._scaled_frames(base30, fps)


def _fixture_rows(rows: np.ndarray, fixture: str) -> np.ndarray:
    names = _decode_fixture_names(rows)
    return rows[names == fixture]


def retained_candidate_rows(candidate_rows: np.ndarray) -> np.ndarray:
    """Return the unchanged final events represented by decision code 2."""
    _require_fields(candidate_rows, ("decision", "frame"), "candidate rows")
    return candidate_rows[candidate_rows["decision"] == RETAINED_DECISION]


def _matched_gt_indices(
    gt_frames: Sequence[int] | np.ndarray,
    event_frames: Sequence[int] | np.ndarray,
    tolerance_frames: int,
) -> set[int]:
    """Return GT indexes matched one-to-one at an already-scaled tolerance."""
    if isinstance(tolerance_frames, bool) or not isinstance(tolerance_frames, int) or tolerance_frames < 0:
        raise ValueError("tolerance_frames must be a non-negative integer")
    gt_values = np.asarray(gt_frames, dtype=np.int32)
    event_values = np.asarray(event_frames, dtype=np.int32)
    return {gt_index for gt_index, _event_index, _offset in tree_scorer._greedy_matches(
        gt_values,
        event_values,
        tolerance_frames,
    )}


def _window_mask(rows: np.ndarray, centre_frame: int, tolerance_frames: int) -> np.ndarray:
    if isinstance(tolerance_frames, bool) or not isinstance(tolerance_frames, int) or tolerance_frames < 0:
        raise ValueError("tolerance_frames must be a non-negative integer")
    _require_fields(rows, ("frame",), "rows")
    frame_values = rows["frame"].astype(np.int64, copy=False)
    return np.abs(frame_values - int(centre_frame)) <= tolerance_frames


def evidence_availability(
    feature_rows: np.ndarray,
    centre_frame: int,
    tolerance_frames: int,
) -> dict[str, bool | int]:
    """Summarise centre-frame evidence visible in a frozen audit window.

    The rows are the verified feature freeze, not the model's retained subset.
    This keeps missing search-region seeds separate from missing physical
    evidence. Validity flags are deliberately read at ``t+0`` only.
    """
    fields = (
        "shuttle_visible_t+0",
        "pose_valid_top_t+0",
        "pose_valid_bot_t+0",
        "wrist_valid_top_t+0",
        "wrist_valid_bot_t+0",
    )
    _require_fields(feature_rows, fields, "feature rows")
    nearby = feature_rows[_window_mask(feature_rows, centre_frame, tolerance_frames)]

    def has_valid(field: str) -> bool:
        return bool(np.any(np.asarray(nearby[field], dtype=np.float64) > 0.5))

    return {
        "frozen_row_count": len(nearby),
        "shuttle": has_valid("shuttle_visible_t+0"),
        "pose": has_valid("pose_valid_top_t+0") or has_valid("pose_valid_bot_t+0"),
        "wrist": has_valid("wrist_valid_top_t+0") or has_valid("wrist_valid_bot_t+0"),
    }


def _candidate_record(row: np.void, centre_frame: int) -> dict[str, int | float | str]:
    """Convert one structured candidate row to stable JSON-compatible values."""
    decision = int(row["decision"])
    return {
        "frame": int(row["frame"]),
        "interval_id": int(row["interval_id"]),
        "distance_frames": int(row["frame"]) - int(centre_frame),
        "timing_score": float(row["timing_score"]),
        "threshold": float(row["threshold"]),
        "decision": decision,
        "decision_name": DECISION_NAMES[decision],
    }


def best_timing_peaks(
    candidate_rows: np.ndarray,
    centre_frame: int,
    tolerance_frames: int,
    nms_radius_frames: int,
    *,
    limit: int = AUDIT_PEAK_COUNT,
) -> list[dict[str, int | float | str]]:
    """Return the strongest raw-score peaks in a fixed window.

    NMS follows the current detector's interval boundary and strict-distance
    rule, but does not apply a score threshold. This preserves below-threshold
    alternatives for the observational failure table.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    if isinstance(nms_radius_frames, bool) or not isinstance(nms_radius_frames, int) or nms_radius_frames < 0:
        raise ValueError("nms_radius_frames must be a non-negative integer")
    _require_fields(
        candidate_rows,
        ("frame", "interval_id", "timing_score", "threshold", "decision"),
        "candidate rows",
    )
    nearby_indices = np.flatnonzero(_window_mask(candidate_rows, centre_frame, tolerance_frames))
    ranked = sorted(
        nearby_indices.tolist(),
        key=lambda index: (
            -float(candidate_rows[index]["timing_score"]),
            int(candidate_rows[index]["frame"]),
            int(candidate_rows[index]["interval_id"]),
        ),
    )
    selected: list[int] = []
    for index in ranked:
        row = candidate_rows[index]
        if all(
            int(row["interval_id"]) != int(candidate_rows[other]["interval_id"])
            or abs(int(row["frame"]) - int(candidate_rows[other]["frame"])) > nms_radius_frames
            for other in selected
        ):
            selected.append(index)
        if len(selected) == limit:
            break
    return [_candidate_record(candidate_rows[index], centre_frame) for index in selected]


def _best_candidate(
    candidate_rows: np.ndarray,
    centre_frame: int,
    tolerance_frames: int,
) -> dict[str, int | float | str] | None:
    nearby_indices = np.flatnonzero(_window_mask(candidate_rows, centre_frame, tolerance_frames))
    if not len(nearby_indices):
        return None
    best_index = min(
        nearby_indices.tolist(),
        key=lambda index: (
            -float(candidate_rows[index]["timing_score"]),
            int(candidate_rows[index]["frame"]),
            int(candidate_rows[index]["interval_id"]),
        ),
    )
    return _candidate_record(candidate_rows[best_index], centre_frame)


def _decision_counts(candidate_rows: np.ndarray, centre_frame: int, tolerance_frames: int) -> dict[str, int]:
    nearby = candidate_rows[_window_mask(candidate_rows, centre_frame, tolerance_frames)]
    return {
        name: int(np.count_nonzero(nearby["decision"] == code))
        for name, code in tree_scorer.CANDIDATE_DECISIONS.items()
    }


def missed_contact_row(
    fixture: str,
    frame: int,
    *,
    is_serve: bool,
    candidate_rows: np.ndarray,
    feature_rows: np.ndarray,
    handcrafted_found: bool,
    tolerance_frames: int,
    nms_radius_frames: int,
) -> dict[str, Any]:
    """Build one stable report row for a GT contact missed by retained HGB."""
    candidate_window = candidate_rows[_window_mask(candidate_rows, frame, tolerance_frames)]
    best = _best_candidate(candidate_rows, frame, tolerance_frames)
    decision_counts = _decision_counts(candidate_rows, frame, tolerance_frames)
    candidate_present = bool(len(candidate_window))
    peaks = best_timing_peaks(
        candidate_rows,
        frame,
        tolerance_frames,
        nms_radius_frames,
    )
    return {
        "fixture": fixture,
        "frame": int(frame),
        "contact_type": "serve" if is_serve else "exchange",
        "serve": is_serve,
        "seeded_candidate_within_tolerance": candidate_present,
        "candidate_count_within_tolerance": len(candidate_window),
        "best_timing_score": None if best is None else best["timing_score"],
        "fold_threshold": None if best is None else best["threshold"],
        "best_candidate": best,
        "nearby_below_threshold": decision_counts["below_threshold"] > 0,
        "nearby_duplicate": decision_counts["nearby_duplicate"] > 0,
        "nearby_retained": decision_counts["retained"] > 0,
        "nearby_decision_counts": decision_counts,
        "peaks": peaks,
        "evidence": evidence_availability(feature_rows, frame, tolerance_frames),
        "handcrafted_filtered_found": bool(handcrafted_found),
    }


def _validate_fixture_inputs(
    fixture: str,
    gt_frames: Sequence[int] | np.ndarray,
    serve_frames: set[int],
    feature_rows: np.ndarray,
    candidate_rows: np.ndarray,
    handcrafted_frames: Sequence[int] | np.ndarray,
) -> None:
    if len(gt_frames) == 0:
        raise ValueError(f"{fixture}: GT contacts are empty")
    if any(int(frame) < 0 for frame in gt_frames):
        raise ValueError(f"{fixture}: GT contact frame is negative")
    if not serve_frames.issubset({int(frame) for frame in gt_frames}):
        raise ValueError(f"{fixture}: serve frame is not a GT contact")
    _require_fields(candidate_rows, ("decision", "frame"), f"{fixture} candidate rows")
    _require_fields(feature_rows, ("frame",), f"{fixture} feature rows")
    if any(int(frame) < 0 for frame in handcrafted_frames):
        raise ValueError(f"{fixture}: handcrafted contact frame is negative")


def audit_fixture(
    fixture: str,
    fps: float,
    gt_frames: Sequence[int] | np.ndarray,
    serve_frames: set[int],
    feature_rows: np.ndarray,
    candidate_rows: np.ndarray,
    handcrafted_frames: Sequence[int] | np.ndarray,
    *,
    primary_tolerance_base30: int = PRIMARY_TOLERANCE_BASE30,
    sensitivity_tolerance_base30: int = SENSITIVITY_TOLERANCE_BASE30,
) -> dict[str, Any]:
    """Audit one fixture using fixed candidate and evidence rows."""
    gt_values = np.asarray(gt_frames, dtype=np.int32)
    handcrafted_values = np.asarray(handcrafted_frames, dtype=np.int32)
    _validate_fixture_inputs(
        fixture,
        gt_values,
        serve_frames,
        feature_rows,
        candidate_rows,
        handcrafted_values,
    )
    primary_tolerance = scale_tolerance(primary_tolerance_base30, fps)
    sensitivity_tolerance = scale_tolerance(sensitivity_tolerance_base30, fps)
    nms_radius = scale_tolerance(NMS_RADIUS_BASE30, fps)

    retained = retained_candidate_rows(candidate_rows)
    retained_frames = retained["frame"].astype(np.int32, copy=False)
    matched = _matched_gt_indices(gt_values, retained_frames, primary_tolerance)
    missed_indices = [index for index in range(len(gt_values)) if index not in matched]
    handcrafted_matched = _matched_gt_indices(gt_values, handcrafted_values, primary_tolerance)
    rows = [
        missed_contact_row(
            fixture,
            int(gt_values[index]),
            is_serve=int(gt_values[index]) in serve_frames,
            candidate_rows=candidate_rows,
            feature_rows=feature_rows,
            handcrafted_found=index in handcrafted_matched,
            tolerance_frames=primary_tolerance,
            nms_radius_frames=nms_radius,
        )
        for index in missed_indices
    ]

    sensitivity_matched = _matched_gt_indices(gt_values, retained_frames, sensitivity_tolerance)
    sensitivity_candidates = {
        index
        for index in range(len(gt_values))
        if bool(np.any(_window_mask(candidate_rows, int(gt_values[index]), sensitivity_tolerance)))
    }
    sensitivity_handcrafted = _matched_gt_indices(gt_values, handcrafted_values, sensitivity_tolerance)
    missed_serves = sum(row["serve"] for row in rows)
    missed_exchanges = len(rows) - missed_serves
    evidence_counts = {
        name: sum(bool(row["evidence"][name]) for row in rows)
        for name in ("shuttle", "pose", "wrist")
    }
    summary = {
        "fixture": fixture,
        "fps": float(fps),
        "primary_tolerance_base30": primary_tolerance_base30,
        "primary_tolerance_frames": primary_tolerance,
        "ground_truth_contacts": len(gt_values),
        "retained_events": len(retained_frames),
        "matched_contacts": len(matched),
        "missed_contacts": len(rows),
        "serve_contacts": int(sum(int(frame) in serve_frames for frame in gt_values)),
        "exchange_contacts": int(sum(int(frame) not in serve_frames for frame in gt_values)),
        "missed_serves": int(missed_serves),
        "missed_exchanges": int(missed_exchanges),
        "missed_with_seeded_candidate": sum(row["seeded_candidate_within_tolerance"] for row in rows),
        "missed_with_below_threshold": sum(row["nearby_below_threshold"] for row in rows),
        "missed_with_nearby_duplicate": sum(row["nearby_duplicate"] for row in rows),
        "missed_with_retained_candidate": sum(row["nearby_retained"] for row in rows),
        "missed_handcrafted_filtered_found": sum(row["handcrafted_filtered_found"] for row in rows),
        "missed_evidence_available": evidence_counts,
        "sensitivity": {
            "tolerance_base30": sensitivity_tolerance_base30,
            "tolerance_frames": sensitivity_tolerance,
            "matched_contacts": len(sensitivity_matched),
            "missed_contacts": len(gt_values) - len(sensitivity_matched),
            "seeded_candidate_coverage": len(sensitivity_candidates),
            "handcrafted_filtered_found": len(sensitivity_handcrafted),
        },
    }
    return {"summary": summary, "missed_contacts": rows}


def _evidence_fixture(evidence: Mapping[str, Any], fixture: str) -> Mapping[str, Any]:
    fixtures = evidence.get("fixtures")
    if not isinstance(fixtures, list):
        raise TypeError("evidence fixture list is malformed")
    for row in fixtures:
        if isinstance(row, Mapping) and row.get("fixture") == fixture:
            return row
    raise ValueError(f"{fixture}: evidence fixture is missing")


def _aggregate_summary(
    fixture_results: Mapping[str, Mapping[str, Any]],
    missed_contacts: Sequence[Mapping[str, Any]],
    ground_truth: tree_scorer.GroundTruth,
) -> dict[str, Any]:
    summaries = {fixture: result["summary"] for fixture, result in fixture_results.items()}
    primary_count_fields = (
        "ground_truth_contacts",
        "retained_events",
        "matched_contacts",
        "missed_contacts",
        "serve_contacts",
        "exchange_contacts",
        "missed_serves",
        "missed_exchanges",
        "missed_with_seeded_candidate",
        "missed_with_below_threshold",
        "missed_with_nearby_duplicate",
        "missed_with_retained_candidate",
        "missed_handcrafted_filtered_found",
    )
    primary = {field: sum(int(summary[field]) for summary in summaries.values()) for field in primary_count_fields}
    primary["tolerance_base30"] = PRIMARY_TOLERANCE_BASE30
    primary["tolerance_frames_by_fixture"] = {
        fixture: int(summary["primary_tolerance_frames"]) for fixture, summary in summaries.items()
    }
    primary["missed_evidence_available"] = {
        name: sum(int(summary["missed_evidence_available"][name]) for summary in summaries.values())
        for name in ("shuttle", "pose", "wrist")
    }
    sensitivity = {
        "tolerance_base30": SENSITIVITY_TOLERANCE_BASE30,
        "tolerance_frames_by_fixture": {
            fixture: int(summary["sensitivity"]["tolerance_frames"])
            for fixture, summary in summaries.items()
        },
        "matched_contacts": sum(int(summary["sensitivity"]["matched_contacts"]) for summary in summaries.values()),
        "missed_contacts": sum(int(summary["sensitivity"]["missed_contacts"]) for summary in summaries.values()),
        "seeded_candidate_coverage": sum(
            int(summary["sensitivity"]["seeded_candidate_coverage"]) for summary in summaries.values()
        ),
        "handcrafted_filtered_found": sum(
            int(summary["sensitivity"]["handcrafted_filtered_found"]) for summary in summaries.values()
        ),
    }
    all_gt = sum(len(values) for values in ground_truth.frames.values())
    if primary["ground_truth_contacts"] != all_gt:
        raise ValueError("audit GT total differs from the verified ground-truth fixture set")
    return {
        **primary,
        "sensitivity": sensitivity,
        "missed_serve_share": (
            primary["missed_serves"] / primary["missed_contacts"] if primary["missed_contacts"] else None
        ),
        "by_fixture": dict(summaries),
        "missed_contact_rows": len(missed_contacts),
    }


def build_audit_result(
    verified_features: tree_scorer.VerifiedFeatures,
    verified_candidates: tree_scorer.VerifiedCandidateScores,
    verified_evidence: evidence_scorer.VerifiedFreeze,
    ground_truth: tree_scorer.GroundTruth,
) -> dict[str, Any]:
    """Build the deterministic audit after all prediction freezes are verified."""
    candidate_rows = verified_candidates.rows
    fixture_results: dict[str, dict[str, Any]] = {}
    all_missed: list[dict[str, Any]] = []
    for fixture, (_video_id, fps) in evidence_freezer.FIXTURE_SPECS.items():
        feature_rows = _fixture_rows(verified_features.rows, fixture)
        fixture_candidates = _fixture_rows(candidate_rows, fixture)
        evidence_fixture = _evidence_fixture(verified_evidence.evidence, fixture)
        handcrafted = evidence_scorer._contact_rows(evidence_fixture, filtered=True)
        handcrafted_frames = np.asarray([row.contact_frame for row in handcrafted], dtype=np.int32)
        result = audit_fixture(
            fixture,
            fps,
            ground_truth.frames[fixture],
            ground_truth.serves[fixture],
            feature_rows,
            fixture_candidates,
            handcrafted_frames,
        )
        fixture_results[fixture] = result
        all_missed.extend(result["missed_contacts"])

    summary = _aggregate_summary(fixture_results, all_missed, ground_truth)
    return {
        "schema": RESULTS_SCHEMA,
        "selected_variant": SELECTED_VARIANT,
        "fixture_set": list(evidence_freezer.FIXTURE_SPECS),
        "primary_tolerance_base30": PRIMARY_TOLERANCE_BASE30,
        "sensitivity_tolerance_base30": SENSITIVITY_TOLERANCE_BASE30,
        "audit_window_definition": (
            "The primary window is the fixture's ±10 base-30-frame tolerance, scaled to its source fps."
        ),
        "nms_definition": (
            "Peaks are ranked by raw held-out timing score, ties by frame then interval, and separated "
            "within each interval by the existing strict NMS radius of ±5 base-30 frames."
        ),
        "evidence_definition": EVIDENCE_DEFINITION,
        "labels_read_after_prediction_freeze": True,
        "inputs": {
            "feature_sha256": verified_features.manifest["feature_sha256"],
            "candidate_scores_sha256": verified_candidates.manifest["candidate_sha256"],
            "tree_result_sha256": verified_candidates.manifest["tree_result_sha256"],
            "contact_evidence_sha256": verified_evidence.manifest["evidence_sha256"],
        },
        "summary": summary,
        "fixtures": fixture_results,
        "missed_contacts": all_missed,
    }


def audit(
    feature_manifest: Path,
    candidate_manifest: Path,
    tree_results: Path,
    evidence_manifest: Path,
) -> dict[str, Any]:
    """Verify frozen inputs, then run the observational missed-contact audit."""
    verified_features = tree_scorer.verify_freeze(feature_manifest)
    verified_candidates = tree_scorer.verify_candidate_scores(
        candidate_manifest,
        verified_features,
        tree_results,
    )
    verified_evidence = evidence_scorer.verify_freeze(evidence_manifest)
    # This is the first label import. All feature rows, sidecar scores,
    # decisions, and retained event identities were fixed above.
    ground_truth = tree_scorer._load_ground_truth()
    return build_audit_result(
        verified_features,
        verified_candidates,
        verified_evidence,
        ground_truth,
    )


def write_results(path: Path, payload: Mapping[str, object]) -> None:
    """Write deterministic plain JSON or gzip JSON."""
    encoded = (
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.name.endswith(".gz"):
        with destination.open("wb") as raw, gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0
        ) as zipped:
            zipped.write(encoded)
    else:
        destination.write_bytes(encoded)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    raw_root = CONTACT_DET_ROOT / "raw"
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=raw_root / "region_v2" / "run_a" / "tree_contact_features_manifest.json",
    )
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--tree-results",
        type=Path,
        default=raw_root / "region_v2" / "tree_contact_results.json.gz",
    )
    parser.add_argument(
        "--evidence-manifest",
        type=Path,
        default=raw_root / "contact_evidence_manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="optional uncompressed JSON file containing only the accessible summary counts",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    payload = audit(
        arguments.feature_manifest,
        arguments.candidate_manifest,
        arguments.tree_results,
        arguments.evidence_manifest,
    )
    write_results(arguments.output, payload)
    if arguments.summary_output is not None:
        write_results(arguments.summary_output, payload["summary"])
    print(f"wrote {arguments.output}")
    print(json.dumps(payload["summary"], sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
