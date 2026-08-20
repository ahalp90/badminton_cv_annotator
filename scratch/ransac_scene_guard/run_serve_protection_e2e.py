"""Replay the baseline and serve-protected RANSAC masks through the full pipeline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from enum import Enum
from pathlib import Path
from typing import Any

import freeze_serve_qualification as freeze
import numpy as np
import scene_aware_ransac as scene
import score_rally_ender_counterfactual as ender
import score_serve_protection as score

from annotator.calibration.fixtures import SSET_01, SSET_15, SSET_21, Fixture
from annotator.calibration.gt_scoring import (
    build_run_video_inputs,
    canonical_tolerance,
    flatten_metrics,
    score_video,
)
from annotator.calibration.scoring import load_gt_rallies, safe_f1, strict_contact_rows
from annotator.run_video import AnnotatorResult, RunCapture, run_video
from annotator.video_outcomes import LandingHorizonRow

DEFAULT_QUALIFICATION = Path(__file__).resolve().parent / "results/serve_qualification.json.gz"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "results/serve_protection_e2e_first_per_region.json.gz"
)
LANDING_HORIZONS_SECONDS = (1.0, 2.0, 3.0)
SOURCE_GENERATION_COMMIT = "189c5af58e45d23ae827dde516924194eb238e18"
SERVE_PROTECTION = "serve_qualified_bursts"
RALLY_ENDER_PROTECTION = "gt_qualified_rally_enders"
PROTECTION_SOURCES = (SERVE_PROTECTION, RALLY_ENDER_PROTECTION)


def release_fixture(
    fixture: Fixture,
    *,
    dead_mask: str,
    court_present: str,
    scene_rows: str,
) -> Fixture:
    """Point the maintained fixture schema at the exact PR98 source-run pins."""
    return replace(
        fixture,
        digests=replace(
            fixture.digests,
            dead_mask=dead_mask,
            court_present=court_present,
            scene_rows=scene_rows,
        ),
    )


PR98_SOURCE_FIXTURES = (
    release_fixture(
        SSET_01,
        dead_mask="70a2a4e9cbd7c6c02b497b468682c462",
        court_present="65f4e28d0556c0e5422f569ad4b69fac",
        scene_rows="7d781f33e29804ef8363bbbd1b60d772",
    ),
    release_fixture(
        SSET_15,
        dead_mask="281562f7933f1fd24301bdba48bb26b9",
        court_present="9b3ab966ef6d357a70ec4541410046a5",
        scene_rows="15f3d6751e75f3c68bab520186096c25",
    ),
    release_fixture(
        SSET_21,
        dead_mask="4d2dfde901ccb5253a54542e60585d71",
        court_present="0c51f0e894c3addfc576c140805fd96f",
        scene_rows="06386f0b6d604819c18b3dd1c3097bef",
    ),
)


def json_value(value: Any) -> Any:
    """Convert compact annotator values into deterministic JSON values."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def strict_metrics(
    rows: Sequence[Mapping[str, object]],
    tolerance_base30: int,
    fps: float,
) -> dict[str, object]:
    """Summarise strict one-to-one contact rows at one base-30 tolerance."""
    selected = [row for row in rows if row["tolerance_base30"] == tolerance_base30]
    matched = [row for row in selected if row["row_kind"] == "matched"]
    unmatched_gt = [row for row in selected if row["row_kind"] == "unmatched_gt"]
    unmatched_candidate = [
        row for row in selected if row["row_kind"] == "unmatched_candidate"
    ]
    gt_count = len(matched) + len(unmatched_gt)
    candidate_count = len(matched) + len(unmatched_candidate)
    precision = len(matched) / candidate_count if candidate_count else None
    recall = len(matched) / gt_count if gt_count else None
    f1 = None if precision is None or recall is None else safe_f1(precision, recall)
    offsets = [abs(int(row["offset_frames"])) for row in matched]
    mean_offset = float(np.mean(offsets)) if offsets else None
    return {
        "tolerance_frames": next(
            (row["tolerance_frames"] for row in selected),
            None,
        ),
        "gt_count": gt_count,
        "candidate_count": candidate_count,
        "matched_count": len(matched),
        "unmatched_gt_count": len(unmatched_gt),
        "unmatched_candidate_count": len(unmatched_candidate),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_abs_offset_frames": mean_offset,
        "mean_abs_offset_seconds": mean_offset / fps if mean_offset is not None else None,
    }


def landing_horizon_metrics(rows: Sequence[LandingHorizonRow]) -> dict[str, dict[str, int]]:
    """Summarise the fixed one-, two-, and three-second landing cutoffs."""
    result: dict[str, dict[str, int]] = {}
    for horizon in LANDING_HORIZONS_SECONDS:
        selected = [row for row in rows if row.horizon_seconds == horizon]
        result[f"seconds_{int(horizon)}"] = {
            "eligible_rally_count": len(selected),
            "horizon_bound_count": sum(
                "horizon_cap" in row.closure_reasons for row in selected
            ),
            "landing_changed_count": sum(row.landing_changed for row in selected),
            "winner_changed_count": sum(row.winner_changed for row in selected),
        }
    return result


def landing_signatures(result: AnnotatorResult) -> dict[str, Any]:
    """Retain exact landing choices, including same-half frame changes."""
    signatures: dict[str, Any] = {}
    for rally_id, landing in result.landings.items():
        signatures[str(rally_id)] = None if landing is None else {
            "frame": landing.frame,
            "norm": list(landing.norm),
            "half": landing.half.value,
            "at_border": landing.at_border,
            "net_ender": landing.net_ender,
        }
    return signatures


def final_contact_signatures(rows: Sequence[LandingHorizonRow]) -> dict[str, int]:
    """Return the final usable contact recorded by each eligible rally."""
    signatures: dict[str, int] = {}
    for row in rows:
        key = str(row.rally_id)
        prior = signatures.setdefault(key, row.final_contact_frame)
        if prior != row.final_contact_frame:
            raise AssertionError(f"rally {row.rally_id} has inconsistent horizon contacts")
    return signatures


def changed_signatures(
    baseline: Mapping[str, Any],
    protected: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return every key whose compact value differs across the two arms."""
    changed: list[dict[str, Any]] = []
    for key in sorted(set(baseline) | set(protected), key=int):
        before = baseline.get(key)
        after = protected.get(key)
        if before != after:
            changed.append({"rally_id": int(key), "baseline": before, "protected": after})
    return changed


def mask_pair(
    fixture: Fixture,
    track: np.ndarray,
    frozen: dict[str, Any],
    burst_policy: str,
    protection_source: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Rebuild the exact baseline and primary protected masks."""
    if freeze.array_content_sha256(track) != frozen.get("track_content_sha256"):
        raise ValueError(f"{fixture.name}: E2E track differs from frozen qualification")
    candidate_path = scene.ANALYSIS_ROOT / f"{fixture.name}_ransac_candidate.npy.xz"
    impulse_path = scene.ANALYSIS_ROOT / f"{fixture.name}_impulse_event_mask.npy.xz"
    expected_digests = score.EXPECTED_POLICY_INPUT_SHA256[fixture.name]
    score.verify_sha256(candidate_path, expected_digests["ransac_candidate"])
    score.verify_sha256(impulse_path, expected_digests["impulse_event_mask"])
    candidate = scene.read_npy_xz(candidate_path).astype(bool)
    impulse = scene.read_npy_xz(impulse_path).astype(bool)
    recurrence_codes, recurrence_info = scene.grade_track(track)
    if recurrence_info["detector_version"] != 4 or recurrence_info["halo_frames"] != 3:
        raise ValueError(f"{fixture.name}: recurrence baseline is not detector v4 with halo 3")
    baseline = scene.policy_masks(candidate, recurrence_codes, impulse)[score.BASELINE_POLICY]
    if protection_source == SERVE_PROTECTION:
        qualified_bursts = score.qualified_burst_frames(frozen, burst_policy)
        radius_frames = scene.scaled_contact_radius(score.PRIMARY_RADIUS_BASE30, fixture.fps)
        window = score.protection_window_mask(len(track), qualified_bursts, radius_frames)
    elif protection_source == RALLY_ENDER_PROTECTION:
        ender_path = scene.ANALYSIS_ROOT / f"{fixture.name}_tp_rally_ender_mask.npy.xz"
        score.verify_sha256(ender_path, ender.EXPECTED_ENDER_SHA256[fixture.name])
        ender_events = scene.read_npy_xz(ender_path).astype(bool)
        window = scene.variants.dilate(ender_events, ender.RADIUS_FRAMES)
    else:
        raise ValueError(f"unknown protection source {protection_source!r}")
    protected = score.subtract_protection(baseline, window)
    return baseline, protected, int(window.sum())


def run_arm(inputs: Any, mask: np.ndarray) -> tuple[AnnotatorResult, RunCapture, list[dict[str, object]]]:
    """Run one event-mask arm while keeping every upstream input fixed."""
    keyword = dict(inputs.keyword)
    keyword.pop("inpaint_codes")
    keyword["shuttle_hallucination_mask"] = mask
    capture = RunCapture()
    rejection_rows: list[dict[str, object]] = []
    keyword["capture"] = capture
    keyword["rejection_diagnostics"] = rejection_rows
    keyword["landing_horizons_s"] = LANDING_HORIZONS_SECONDS
    keyword["court_invalid_is_excluded"] = True
    result = run_video(*inputs.positional, **keyword)
    return result, capture, rejection_rows


def assert_fixed_upstream(
    baseline: AnnotatorResult,
    protected: AnnotatorResult,
    baseline_capture: RunCapture,
    protected_capture: RunCapture,
) -> None:
    """Prove the event-mask comparison did not move spans or contacts."""
    comparisons = {
        "spans": baseline.spans == protected.spans,
        "raw contacts": baseline.contacts == protected.contacts,
        "filtered contacts": baseline.filtered_contacts == protected.filtered_contacts,
        "per-rally contacts": baseline.filtered_by_rally == protected.filtered_by_rally,
        "raw exclusion mask": np.array_equal(
            baseline_capture.raw_exclusion_mask,
            protected_capture.raw_exclusion_mask,
        ),
        "definitive exclusion mask": np.array_equal(
            baseline_capture.definitive_exclusion_mask,
            protected_capture.definitive_exclusion_mask,
        ),
    }
    failed = [name for name, equal in comparisons.items() if not equal]
    if failed:
        raise AssertionError(f"event-mask replay changed fixed upstream values: {', '.join(failed)}")


def rejection_key(row: Mapping[str, object]) -> tuple[object, ...]:
    """Return the stable identity of one event-mask rejection row."""
    return tuple(
        row.get(field)
        for field in (
            "rule",
            "rally_id",
            "start_frame",
            "end_frame",
            "trigger_frame",
        )
    )


def changed_rally_rows(baseline_rows: Sequence[Any], protected_rows: Sequence[Any]) -> list[dict[str, Any]]:
    """Retain only GT-scored rally rows whose landing or winner output changed."""
    if len(baseline_rows) != len(protected_rows):
        raise AssertionError("baseline and protected GT rally rows differ in length")
    fields = (
        "gt_index",
        "set_id",
        "rally",
        "classification",
        "mapped_span",
        "landing_gt",
        "landing_pred",
        "landing_correct",
        "getpoint_gt",
        "getpoint_pred",
        "getpoint_correct",
    )
    changes: list[dict[str, Any]] = []
    for before, after in zip(baseline_rows, protected_rows):
        before_values = before._asdict()
        after_values = after._asdict()
        if (
            before_values["landing_pred"] == after_values["landing_pred"]
            and before_values["landing_correct"] == after_values["landing_correct"]
            and before_values["getpoint_pred"] == after_values["getpoint_pred"]
            and before_values["getpoint_correct"] == after_values["getpoint_correct"]
        ):
            continue
        changes.append(
            {
                "baseline": {field: json_value(before_values[field]) for field in fields},
                "protected": {field: json_value(after_values[field]) for field in fields},
            }
        )
    return changes


def arm_record(
    fixture: Fixture,
    result: AnnotatorResult,
    capture: RunCapture,
    scoring: Any,
    strict_rows: list[dict[str, object]],
    rejection_rows: list[dict[str, object]],
) -> dict[str, Any]:
    """Build the compact retained record for one full-chain arm."""
    landing_values = list(result.landings.values())
    return {
        "metrics": flatten_metrics(scoring),
        "strict_contacts": {
            f"base30_{tolerance}": strict_metrics(strict_rows, tolerance, fixture.fps)
            for tolerance in (5, 10, 15)
        },
        "rally_count": len(result.spans),
        "raw_contact_count": len(result.contacts),
        "filtered_contact_count": len(result.filtered_contacts),
        "landing_entry_count": len(landing_values),
        "landing_available_count": sum(landing is not None for landing in landing_values),
        "rejection_count": len(rejection_rows),
        "rejections": [json_value(row) for row in rejection_rows],
        "landings": landing_signatures(result),
        "final_contacts": final_contact_signatures(capture.landing_horizon_rows),
        "landing_horizons": landing_horizon_metrics(capture.landing_horizon_rows),
        "definitive_exclusion_fraction": float(
            np.asarray(capture.definitive_exclusion_mask, dtype=bool).mean()
        ),
    }


def run_fixture_replay(
    fixture: Fixture,
    frozen: dict[str, Any],
    burst_policy: str,
    protection_source: str,
) -> dict[str, Any]:
    """Run and compare both full-chain arms for one pinned fixture."""
    inputs = build_run_video_inputs(fixture)
    track = inputs.positional[0]
    if not isinstance(track, np.ndarray):
        raise TypeError(f"{fixture.name}: fixture track is not a numpy array")
    baseline_mask, protected_mask, window_frames = mask_pair(
        fixture,
        track,
        frozen,
        burst_policy,
        protection_source,
    )
    baseline_result, baseline_capture, baseline_rejections = run_arm(inputs, baseline_mask)
    protected_result, protected_capture, protected_rejections = run_arm(inputs, protected_mask)
    assert_fixed_upstream(
        baseline_result,
        protected_result,
        baseline_capture,
        protected_capture,
    )

    gt_rallies = load_gt_rallies(inputs.master, fixture.video_id)
    baseline_strict = strict_contact_rows(
        baseline_result.spans,
        baseline_result.filtered_contacts,
        gt_rallies,
        fixture.fps,
        (5, 10, 15),
    )
    protected_strict = strict_contact_rows(
        protected_result.spans,
        protected_result.filtered_contacts,
        gt_rallies,
        fixture.fps,
        (5, 10, 15),
    )
    if baseline_strict != protected_strict:
        raise AssertionError(f"{fixture.name}: strict contact rows changed in fixed-upstream replay")

    tolerance = canonical_tolerance(fixture.fps)
    baseline_scoring = score_video(
        fixture,
        baseline_result,
        inputs.master,
        inputs.courts,
        tolerance,
    )
    protected_scoring = score_video(
        fixture,
        protected_result,
        inputs.master,
        inputs.courts,
        tolerance,
    )
    baseline_rejection_keys = {rejection_key(row) for row in baseline_rejections}
    protected_rejection_keys = {rejection_key(row) for row in protected_rejections}
    added_rejections = protected_rejection_keys - baseline_rejection_keys
    if added_rejections:
        raise AssertionError(f"{fixture.name}: protection added event-mask rejections")
    baseline_record = arm_record(
        fixture,
        baseline_result,
        baseline_capture,
        baseline_scoring,
        baseline_strict,
        baseline_rejections,
    )
    protected_record = arm_record(
        fixture,
        protected_result,
        protected_capture,
        protected_scoring,
        protected_strict,
        protected_rejections,
    )
    changed_landings = changed_signatures(
        baseline_record["landings"],
        protected_record["landings"],
    )
    changed_final_contacts = changed_signatures(
        baseline_record["final_contacts"],
        protected_record["final_contacts"],
    )
    changed_gt_rallies = changed_rally_rows(
        baseline_scoring.rows,
        protected_scoring.rows,
    )
    scored_changed_spans = {
        int(change["protected"]["mapped_span"])
        for change in changed_gt_rallies
        if change["protected"]["mapped_span"] is not None
    }
    changed_output_spans = {
        int(change["rally_id"])
        for change in (*changed_landings, *changed_final_contacts)
    }
    unscored_changed_output_spans = sorted(changed_output_spans - scored_changed_spans)
    return {
        "video_id": fixture.video_id,
        "fps": fixture.fps,
        "frames": len(track),
        "window_frames": window_frames,
        "baseline_selected_frames": int(baseline_mask.sum()),
        "protected_selected_frames": int(protected_mask.sum()),
        "fixed_upstream_equal": True,
        "baseline": baseline_record,
        "protected": protected_record,
        "comparison": {
            "cleared_rejection_count": len(
                baseline_rejection_keys - protected_rejection_keys
            ),
            "added_rejection_count": 0,
            "changed_landings": changed_landings,
            "changed_final_contacts": changed_final_contacts,
            "changed_gt_rallies": changed_gt_rallies,
            "unscored_changed_output_rally_ids": unscored_changed_output_spans,
        },
    }


def safety_failures(fixtures: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return lost correct outcomes and changed outcomes that are not known correct."""
    failures: list[dict[str, Any]] = []
    for fixture_name, fixture in fixtures.items():
        comparison = fixture["comparison"]
        for change in comparison["changed_gt_rallies"]:
            before = change["baseline"]
            after = change["protected"]
            reasons: list[str] = []
            if before["landing_correct"] is True and after["landing_correct"] is not True:
                reasons.append("lost_correct_landing")
            if before["getpoint_correct"] is True and after["getpoint_correct"] is not True:
                reasons.append("lost_correct_winner")
            if (
                before["landing_pred"] != after["landing_pred"]
                and after["landing_pred"] is not None
                and after["landing_correct"] is not True
            ):
                reasons.append("new_or_changed_wrong_or_unscored_landing")
            if (
                before["getpoint_pred"] != after["getpoint_pred"]
                and after["getpoint_pred"] is not None
                and after["getpoint_correct"] is not True
            ):
                reasons.append("new_or_changed_wrong_or_unscored_winner")
            if reasons:
                failures.append({"fixture": fixture_name, "reasons": reasons, **change})
    return failures


def replay_decision(
    *,
    fixed_upstream: bool,
    added_rejections: int,
    failures: Sequence[Mapping[str, Any]],
    unscored_output_changes: int,
    protection_source: str,
) -> dict[str, Any]:
    """Keep diagnostic counterfactuals separate from deployable evidence."""
    passes_observed_output_screen = (
        fixed_upstream
        and added_rejections == 0
        and not failures
        and unscored_output_changes == 0
    )
    diagnostic_only = protection_source == RALLY_ENDER_PROTECTION
    return {
        "fixed_upstream_equal": fixed_upstream,
        "added_rejection_count": added_rejections,
        "safety_failure_count": len(failures),
        "unscored_output_change_count": unscored_output_changes,
        "passes_observed_output_screen": passes_observed_output_screen,
        "diagnostic_only": diagnostic_only,
        "deployable_evidence": not diagnostic_only,
        "passes_e2e_safety_screen": (
            passes_observed_output_screen and not diagnostic_only
        ),
    }


def run_replay(
    qualification_path: Path,
    burst_policy: str = score.FIRST_QUALIFIED_PER_REGION,
    protection_source: str = SERVE_PROTECTION,
) -> dict[str, Any]:
    """Run the frozen primary arm on every PR98 source-run fixture."""
    qualification = score.read_json_gz(qualification_path)
    if qualification.get("baseline_commit") != scene.BASELINE_SHA:
        raise ValueError("serve qualification was not frozen at the PR98 baseline")
    if qualification.get("configuration", {}).get("threshold_bh") != freeze.SERVE_THRESHOLD_BH:
        raise ValueError("serve qualification threshold differs from the frozen S2 plan")
    scene_baseline_proof = score.verify_scene_baseline_proof()
    fixture_results = {
        fixture.name: run_fixture_replay(
            fixture,
            qualification["fixtures"][fixture.name],
            burst_policy,
            protection_source,
        )
        for fixture in PR98_SOURCE_FIXTURES
    }
    failures = safety_failures(fixture_results)
    fixed_upstream = all(
        fixture["fixed_upstream_equal"] for fixture in fixture_results.values()
    )
    added_rejections = sum(
        fixture["comparison"]["added_rejection_count"]
        for fixture in fixture_results.values()
    )
    unscored_output_changes = sum(
        len(fixture["comparison"]["unscored_changed_output_rally_ids"])
        for fixture in fixture_results.values()
    )
    return {
        "schema_version": 1,
        "baseline_commit": scene.BASELINE_SHA,
        "status": "fixed-upstream full-chain replay on the PR98 source-run fixtures",
        "configuration": {
            "fixture_profile": "UNE 189c5af static ShuttleSet homography stride-8",
            "fixture_source_commit": SOURCE_GENERATION_COMMIT,
            "baseline_policy": score.BASELINE_POLICY,
            "protection_source": protection_source,
            "threshold_bh": (
                freeze.SERVE_THRESHOLD_BH
                if protection_source == SERVE_PROTECTION
                else None
            ),
            "burst_policy": (
                burst_policy if protection_source == SERVE_PROTECTION else None
            ),
            "protection_radius_base30": (
                score.PRIMARY_RADIUS_BASE30
                if protection_source == SERVE_PROTECTION
                else None
            ),
            "rally_ender_radius_frames": (
                ender.RADIUS_FRAMES
                if protection_source == RALLY_ENDER_PROTECTION
                else None
            ),
            "landing_horizons_seconds": list(LANDING_HORIZONS_SECONDS),
            "court_invalid_is_excluded": True,
        },
        "input_digests": {
            "qualification_sha256": score.verify_sha256(
                qualification_path,
                score.QUALIFICATION_SHA256,
            ),
            "scene_baseline_proof": scene_baseline_proof,
            "shots_master_md5": scene.SHOTS_MASTER_MD5,
            "policy_input_sha256": score.EXPECTED_POLICY_INPUT_SHA256,
            "rally_ender_mask_sha256": (
                ender.EXPECTED_ENDER_SHA256
                if protection_source == RALLY_ENDER_PROTECTION
                else None
            ),
            "fixture_md5": {
                fixture.name: {
                    pin.path.as_posix(): pin.md5
                    for pin in fixture.run_video_files
                }
                for fixture in PR98_SOURCE_FIXTURES
            },
        },
        "decision": replay_decision(
            fixed_upstream=fixed_upstream,
            added_rejections=added_rejections,
            failures=failures,
            unscored_output_changes=unscored_output_changes,
            protection_source=protection_source,
        ),
        "safety_failures": failures,
        "fixtures": fixture_results,
    }


def parse_args() -> argparse.Namespace:
    """Parse the frozen qualification and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=Path, default=DEFAULT_QUALIFICATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--burst-policy",
        choices=score.BURST_POLICIES,
        default=score.FIRST_QUALIFIED_PER_REGION,
    )
    parser.add_argument(
        "--protection-source",
        choices=PROTECTION_SOURCES,
        default=SERVE_PROTECTION,
    )
    return parser.parse_args()


def main() -> None:
    """Run and retain the fixed-upstream full-chain replay."""
    args = parse_args()
    result = run_replay(
        args.qualification,
        args.burst_policy,
        args.protection_source,
    )
    scene.write_json_gz(args.output, result)
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    for fixture_name, fixture in result["fixtures"].items():
        comparison = fixture["comparison"]
        print(
            f"{fixture_name}: cleared {comparison['cleared_rejection_count']} rejections; "
            f"changed {len(comparison['changed_gt_rallies'])} GT rally outcomes"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
