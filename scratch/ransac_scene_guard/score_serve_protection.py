"""Score fixed serve-burst protection windows against the PR 98 RANSAC baseline."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import freeze_serve_qualification as freeze
import numpy as np
import scene_aware_ransac as scene

DEFAULT_QUALIFICATION = Path(__file__).resolve().parent / "results/serve_qualification.json.gz"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "results/serve_protection_first_per_region.json.gz"
)
BASELINE_POLICY = "recurrence_v4_clean_impulse_veto_radius3"
PRIMARY_RADIUS_BASE30 = 10
PROTECTION_RADII_BASE30 = (5, 10, 15)
EXPECTED_BASELINE_SELECTED = 11_660
QUALIFICATION_SHA256 = "3957894d367d2090f481ca98d62e59a62637cb403819be56b1775af860ca8914"
ALL_QUALIFIED_BURSTS = "all_qualified_bursts"
FIRST_QUALIFIED_PER_REGION = "first_qualified_per_region"
BURST_POLICIES = (ALL_QUALIFIED_BURSTS, FIRST_QUALIFIED_PER_REGION)
SCENE_PROOF_PATH = Path(__file__).resolve().parent / "results/scene_aware_ransac.json.gz"
SCENE_PROOF_SHA256 = "6b2e05d9fbcc0057317bb7fed61977d1dfe0ec6b1f126fd82cd098a33765798c"
EXPECTED_POLICY_INPUT_SHA256 = {
    "sset_01": {
        "ransac_candidate": "d4681a119c1b7a95fc5457bfbac0a74bcdfa3ade10a5cf7471796c26b45ef2a3",
        "impulse_event_mask": "04375804644a78f953b22dfc5398b49cd90b5663def486462122505718c17d4b",
    },
    "sset_15": {
        "ransac_candidate": "23b79aa0b05183bdab3138a1fade061b9d1d4ed7246104946109258d2562ad10",
        "impulse_event_mask": "d5de2b36eb3a22df6968c2e346fa0976bc2a39fb29820fa2d601264aa93c6093",
    },
    "sset_21": {
        "ransac_candidate": "8f1f85be1a589cbd4df0747c27a9d3bec6f4d6841be371459f145c103146a78c",
        "impulse_event_mask": "736a671dfaa374cf04db6e4ff8adb4cacbe66484b2970c6825fe84a8a229ed44",
    },
}


def read_json_gz(path: Path) -> dict[str, Any]:
    """Read and validate a compressed JSON object."""
    with gzip.open(path, "rt", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return payload


def verify_sha256(path: Path, expected: str) -> str:
    """Fail when a retained experiment input differs from its frozen bytes."""
    actual = scene.file_digest(path, "sha256")
    if actual != expected:
        raise ValueError(f"{path}: SHA-256 {actual} != frozen {expected}")
    return actual


def verify_scene_baseline_proof() -> dict[str, Any]:
    """Verify the retained source rerun that proved the stored RANSAC masks."""
    proof_sha256 = verify_sha256(SCENE_PROOF_PATH, SCENE_PROOF_SHA256)
    proof = read_json_gz(SCENE_PROOF_PATH)
    expected_verification = {
        "recurrence_baseline": "detector v4, halo 3, recomputed from each pinned track",
        "source_fields_exact": True,
        "stored_candidate_masks_exact": True,
        "stored_integer_vote_fields_exact": True,
        "stored_six_decimal_float_fields_within_rounding": True,
    }
    if proof.get("baseline_commit") != scene.BASELINE_SHA:
        raise ValueError("scene baseline proof was not run at the PR 98 commit")
    if proof.get("baseline_verification") != expected_verification:
        raise ValueError("scene baseline proof does not establish exact stored RANSAC masks")
    return {
        "path": str(SCENE_PROOF_PATH.relative_to(scene.variants.REPO)),
        "sha256": proof_sha256,
        "verification": expected_verification,
    }


def load_contact_roles() -> tuple[
    dict[int, set[int]],
    dict[int, set[int]],
    dict[int, set[int]],
]:
    """Load all, first, and final ShuttleSet contact frames."""
    path = scene.variants.REPO / "training/data/shuttleset/annotations/shots_master.csv"
    if scene.file_digest(path, "md5") != scene.SHOTS_MASTER_MD5:
        raise ValueError("ShuttleSet shots_master.csv differs from the pinned fixture input")
    contacts: dict[int, set[int]] = defaultdict(set)
    rallies: dict[tuple[int, str, int], list[tuple[int, int]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            video_id = int(row["vid"])
            if video_id not in scene.variants.FIXTURES.values():
                continue
            frame = int(row["frame_num"])
            ball_round = int(row["ball_round"])
            contacts[video_id].add(frame)
            rallies[(video_id, row["set_id"], int(row["rally"]))].append(
                (ball_round, frame)
            )

    first_contacts: dict[int, set[int]] = defaultdict(set)
    final_contacts: dict[int, set[int]] = defaultdict(set)
    for (video_id, _set_id, _rally), rows in rallies.items():
        ordered = sorted(rows)
        first_contacts[video_id].add(ordered[0][1])
        final_contacts[video_id].add(ordered[-1][1])
    return contacts, first_contacts, final_contacts


def protection_window_mask(
    frame_count: int,
    qualified_bursts: list[int],
    radius: int,
) -> np.ndarray:
    """Build clipped inclusive windows around qualified burst starts."""
    if frame_count < 0 or radius < 0:
        raise ValueError("frame_count and radius must be nonnegative")
    mask = np.zeros(frame_count, dtype=bool)
    for burst in qualified_bursts:
        if not 0 <= burst < frame_count:
            raise ValueError(f"burst frame {burst} is outside [0, {frame_count})")
        start = max(0, burst - radius)
        stop = min(frame_count, burst + radius + 1)
        mask[start:stop] = True
    return mask


def qualified_burst_frames(frozen: dict[str, Any], policy: str) -> list[int]:
    """Select frozen qualified bursts under one predeclared consumption rule."""
    if policy not in BURST_POLICIES:
        raise ValueError(f"unknown burst policy {policy!r}")
    rows = frozen.get("bursts")
    if not isinstance(rows, list):
        raise TypeError("frozen fixture does not contain a burst list")
    qualified = [row for row in rows if row["qualified"]]
    if policy == ALL_QUALIFIED_BURSTS:
        return [int(row["burst_start"]) for row in qualified]

    selected: list[int] = []
    seen_regions: set[int] = set()
    for row in qualified:
        region_index = int(row["region_index"])
        if region_index in seen_regions:
            continue
        seen_regions.add(region_index)
        selected.append(int(row["burst_start"]))
    return selected


def subtract_protection(baseline: np.ndarray, window: np.ndarray) -> np.ndarray:
    """Clear protected frames without ever adding a rejection candidate."""
    if baseline.dtype != np.bool_ or window.dtype != np.bool_ or baseline.shape != window.shape:
        raise ValueError("baseline and window must be aligned boolean masks")
    protected = baseline & ~window
    if np.any(protected & ~baseline):
        raise AssertionError("serve protection added a rejection candidate")
    return protected


def event_metrics(mask: np.ndarray, events: set[int], fps: float) -> dict[str, Any]:
    """Return exact and fixed-margin measures for one event role."""
    metrics = scene.mask_metrics(mask, events, set(), [], fps)
    return {
        "exact": metrics["exact_contacts"],
        "tolerances": metrics["contact_tolerances"],
    }


def stress_span_rows(
    spans: list[tuple[int, int]],
    baseline: np.ndarray,
    protected: np.ndarray,
) -> list[dict[str, Any]]:
    """Record every positive span so a lost baseline hit stays visible."""
    rows: list[dict[str, Any]] = []
    for span_index, (start, stop) in enumerate(spans):
        baseline_frames = (np.flatnonzero(baseline[start:stop]) + start).tolist()
        protected_frames = (np.flatnonzero(protected[start:stop]) + start).tolist()
        rows.append(
            {
                "span_index": span_index,
                "start_frame": start,
                "stop_frame_exclusive": stop,
                "baseline_frames": baseline_frames,
                "protected_frames": protected_frames,
                "baseline_hit": bool(baseline_frames),
                "protected_hit": bool(protected_frames),
            }
        )
    return rows


def primary_decision(
    baseline: dict[str, Any],
    protected: dict[str, Any],
    first_baseline: dict[str, Any],
    first_protected: dict[str, Any],
) -> dict[str, Any]:
    """Apply the predeclared mask screen to the base-30 ±10 arm."""
    selected_before = baseline["selected_frames"]
    selected_after = protected["selected_frames"]
    risk_before = baseline["contact_tolerances"]["base30_10"]["contacts_with_candidate"]
    risk_after = protected["contact_tolerances"]["base30_10"]["contacts_with_candidate"]
    frame_reduction = (selected_before - selected_after) / selected_before
    risk_reduction = (risk_before - risk_after) / risk_before
    first_exact_rescued = first_baseline["exact"] - first_protected["exact"]
    first_near_rescued = (
        first_baseline["tolerances"]["base30_10"]["contacts_with_candidate"]
        - first_protected["tolerances"]["base30_10"]["contacts_with_candidate"]
    )
    spans_preserved = (
        protected["visual_positive_spans_hit"]
        == baseline["visual_positive_spans_hit"]
    )
    first_contact_rescued = first_exact_rescued > 0 or first_near_rescued > 0
    contact_enrichment_pass = risk_reduction >= 2.0 * frame_reduction
    return {
        "stress_spans_preserved": spans_preserved,
        "first_exact_conflicts_rescued": first_exact_rescued,
        "first_contacts_within_base30_10_rescued": first_near_rescued,
        "first_contact_rescued": first_contact_rescued,
        "selected_frame_reduction_fraction": frame_reduction,
        "all_contact_base30_10_risk_reduction_fraction": risk_reduction,
        "contact_reduction_at_least_twice_frame_reduction": contact_enrichment_pass,
        "passes_mask_screen": (
            spans_preserved and first_contact_rescued and contact_enrichment_pass
        ),
    }


def run_score(
    qualification_path: Path,
    burst_policy: str = FIRST_QUALIFIED_PER_REGION,
) -> dict[str, Any]:
    """Score every fixed protection radius against contacts and stress spans."""
    verify_sha256(qualification_path, QUALIFICATION_SHA256)
    qualification = read_json_gz(qualification_path)
    if qualification.get("schema_version") != 1:
        raise ValueError("unsupported serve-qualification schema")
    if qualification.get("baseline_commit") != scene.BASELINE_SHA:
        raise ValueError("serve qualification was not frozen at the PR 98 baseline")
    expected_configuration = {
        "threshold_bh": freeze.SERVE_THRESHOLD_BH,
        "mode": "trim",
        "close": None,
        "stillness_threshold_bh": None,
        "serve_setup_window": "[burst-lookback, burst)",
        "track_source": "unmasked pinned track",
    }
    if qualification.get("configuration") != expected_configuration:
        raise ValueError("serve qualification configuration differs from the frozen S2 plan")
    firewall = qualification.get("label_firewall")
    if firewall != {"shots_master_read": False, "visual_hallucination_spans_read": False}:
        raise ValueError("serve qualification does not carry the expected label firewall")
    scene_baseline_proof = verify_scene_baseline_proof()

    contacts, first_contacts, final_contacts = load_contact_roles()
    fixture_results: dict[str, Any] = {}
    baseline_total = scene.empty_total_metrics()
    first_baseline_total = {"exact": 0, "tolerances": {}}
    policy_input_digests: dict[str, dict[str, str]] = {}
    radius_totals: dict[str, dict[str, Any]] = {}
    for radius_base30 in PROTECTION_RADII_BASE30:
        key = f"base30_{radius_base30}"
        radius_totals[key] = {
            "protected": scene.empty_total_metrics(),
            "cleared": scene.empty_total_metrics(),
            "first_protected": {"exact": 0, "tolerances": {}},
        }

    for spec in scene.audit.FIXTURES:
        frozen = qualification["fixtures"].get(spec.name)
        if not isinstance(frozen, dict):
            raise TypeError(f"qualification lacks a valid {spec.name} object")
        track = scene.read_npy_xz(spec.raw_path)
        if freeze.array_content_sha256(track) != frozen.get("track_content_sha256"):
            raise ValueError(f"{spec.name}: local scoring track differs from remote qualification")

        candidate_path = scene.ANALYSIS_ROOT / f"{spec.name}_ransac_candidate.npy.xz"
        impulse_path = scene.ANALYSIS_ROOT / f"{spec.name}_impulse_event_mask.npy.xz"
        expected_digests = EXPECTED_POLICY_INPUT_SHA256[spec.name]
        policy_input_digests[spec.name] = {
            "ransac_candidate_sha256": verify_sha256(
                candidate_path,
                expected_digests["ransac_candidate"],
            ),
            "impulse_event_mask_sha256": verify_sha256(
                impulse_path,
                expected_digests["impulse_event_mask"],
            ),
        }
        candidate = scene.read_npy_xz(candidate_path).astype(bool)
        recurrence_codes, recurrence_info = scene.grade_track(track)
        if recurrence_info["detector_version"] != 4 or recurrence_info["halo_frames"] != 3:
            raise ValueError(f"{spec.name}: recurrence baseline is not detector v4 with halo 3")
        impulse_events = scene.read_npy_xz(impulse_path).astype(bool)
        baseline = scene.policy_masks(candidate, recurrence_codes, impulse_events)[BASELINE_POLICY]
        spans = scene.variants.load_visual_spans(spec.name)
        baseline_metrics = scene.mask_metrics(
            baseline,
            contacts[spec.video_id],
            final_contacts[spec.video_id],
            spans,
            spec.fps,
        )
        first_baseline = event_metrics(baseline, first_contacts[spec.video_id], spec.fps)
        scene.add_to_total(baseline_total, baseline_metrics)
        add_event_totals(first_baseline_total, first_baseline)

        qualified_bursts = qualified_burst_frames(frozen, burst_policy)
        radius_results: dict[str, Any] = {}
        for radius_base30 in PROTECTION_RADII_BASE30:
            key = f"base30_{radius_base30}"
            radius_frames = scene.scaled_contact_radius(radius_base30, spec.fps)
            window = protection_window_mask(len(track), qualified_bursts, radius_frames)
            protected = subtract_protection(baseline, window)
            cleared = baseline & window
            protected_metrics = scene.mask_metrics(
                protected,
                contacts[spec.video_id],
                final_contacts[spec.video_id],
                spans,
                spec.fps,
            )
            cleared_metrics = scene.mask_metrics(
                cleared,
                contacts[spec.video_id],
                final_contacts[spec.video_id],
                spans,
                spec.fps,
            )
            first_protected = event_metrics(
                protected,
                first_contacts[spec.video_id],
                spec.fps,
            )
            scene.add_to_total(radius_totals[key]["protected"], protected_metrics)
            scene.add_to_total(radius_totals[key]["cleared"], cleared_metrics)
            add_event_totals(radius_totals[key]["first_protected"], first_protected)
            radius_results[key] = {
                "radius_frames": radius_frames,
                "window_frames": int(window.sum()),
                "protected": protected_metrics,
                "cleared": cleared_metrics,
                "first_contacts": {
                    "baseline": first_baseline,
                    "protected": first_protected,
                },
                "positive_spans": stress_span_rows(spans, baseline, protected),
            }

        fixture_results[spec.name] = {
            "video_id": spec.video_id,
            "fps": spec.fps,
            "frames": len(track),
            "qualified_burst_count": len(qualified_bursts),
            "baseline": baseline_metrics,
            "first_contacts_baseline": first_baseline,
            "radii": radius_results,
        }

    if baseline_total["selected_frames"] != EXPECTED_BASELINE_SELECTED:
        raise AssertionError(
            f"baseline selected {baseline_total['selected_frames']} frames, "
            f"expected {EXPECTED_BASELINE_SELECTED}"
        )
    primary_key = f"base30_{PRIMARY_RADIUS_BASE30}"
    decision = primary_decision(
        baseline_total,
        radius_totals[primary_key]["protected"],
        first_baseline_total,
        radius_totals[primary_key]["first_protected"],
    )
    return {
        "schema_version": 1,
        "baseline_commit": scene.BASELINE_SHA,
        "status": "analysis-only; protection only subtracts RANSAC rejection candidates",
        "configuration": {
            "baseline_policy": BASELINE_POLICY,
            "threshold_bh": freeze.SERVE_THRESHOLD_BH,
            "burst_policy": burst_policy,
            "protection_radii_base30": list(PROTECTION_RADII_BASE30),
            "primary_radius_base30": PRIMARY_RADIUS_BASE30,
            "window_policy": "inclusive symmetric radius around each qualified burst start",
        },
        "input_digests": {
            "qualification_file": qualification_path.name,
            "qualification_sha256": hashlib.sha256(qualification_path.read_bytes()).hexdigest(),
            "shots_master_md5": scene.SHOTS_MASTER_MD5,
            "policy_inputs": policy_input_digests,
            "scene_baseline_proof": scene_baseline_proof,
        },
        "baseline": baseline_total,
        "first_contacts_baseline": first_baseline_total,
        "radii": radius_totals,
        "primary_decision": decision,
        "fixtures": fixture_results,
    }


def add_event_totals(total: dict[str, Any], metrics: dict[str, Any]) -> None:
    """Add exact and fixed-margin event counts in place."""
    total["exact"] += metrics["exact"]
    for tolerance, values in metrics["tolerances"].items():
        target = total["tolerances"].setdefault(
            tolerance,
            {
                "contacts_with_candidate": 0,
                "candidate_frames_near_contacts": 0,
            },
        )
        target["contacts_with_candidate"] += values["contacts_with_candidate"]
        target["candidate_frames_near_contacts"] += values["candidate_frames_near_contacts"]


def print_summary(result: dict[str, Any]) -> None:
    """Print the compact fixed-radius comparison."""
    baseline = result["baseline"]
    print("radius_base30,frames_before,frames_after,exact_contacts_before,exact_contacts_after,first_exact_before,first_exact_after,spans_before,spans_after")
    for radius_base30 in PROTECTION_RADII_BASE30:
        key = f"base30_{radius_base30}"
        protected = result["radii"][key]["protected"]
        first = result["radii"][key]["first_protected"]
        print(
            f"{radius_base30},"
            f"{baseline['selected_frames']},"
            f"{protected['selected_frames']},"
            f"{baseline['exact_contacts']},"
            f"{protected['exact_contacts']},"
            f"{result['first_contacts_baseline']['exact']},"
            f"{first['exact']},"
            f"{baseline['visual_positive_spans_hit']},"
            f"{protected['visual_positive_spans_hit']}"
        )
    print(json.dumps(result["primary_decision"], indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    """Parse the frozen qualification and retained result paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=Path, default=DEFAULT_QUALIFICATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--burst-policy",
        choices=BURST_POLICIES,
        default=FIRST_QUALIFIED_PER_REGION,
    )
    return parser.parse_args()


def main() -> None:
    """Score and retain the fixed serve protection arms."""
    args = parse_args()
    result = run_score(args.qualification, args.burst_policy)
    scene.write_json_gz(args.output, result)
    print_summary(result)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
