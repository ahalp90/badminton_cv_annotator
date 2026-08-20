"""Score the fixed GT-qualified rally-ending veto defined in PR 98."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import scene_aware_ransac as scene
import score_serve_protection as score

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results/rally_ender_counterfactual.json.gz"
RADIUS_FRAMES = 3
EXPECTED_ENDER_SHA256 = {
    "sset_01": "b1961319fb3dedbde0df0b4765cf360c9c67c9333c6a9df2c00eaa7d3da040ca",
    "sset_15": "c90a6c1cc5de3d764c41f263db8053a872a101643ab33ca176e520c7e82532b9",
    "sset_21": "53916161c42748499775b649f7023b46a6b65c9fd9529050c7fb86e7ef14f4d7",
}


def fixture_masks(spec: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the exact baseline, protected mask, and ender window."""
    track = scene.read_npy_xz(spec.raw_path)
    candidate_path = scene.ANALYSIS_ROOT / f"{spec.name}_ransac_candidate.npy.xz"
    impulse_path = scene.ANALYSIS_ROOT / f"{spec.name}_impulse_event_mask.npy.xz"
    ender_path = scene.ANALYSIS_ROOT / f"{spec.name}_tp_rally_ender_mask.npy.xz"
    expected_policy = score.EXPECTED_POLICY_INPUT_SHA256[spec.name]
    score.verify_sha256(candidate_path, expected_policy["ransac_candidate"])
    score.verify_sha256(impulse_path, expected_policy["impulse_event_mask"])
    score.verify_sha256(ender_path, EXPECTED_ENDER_SHA256[spec.name])
    candidate = scene.read_npy_xz(candidate_path).astype(bool)
    impulse = scene.read_npy_xz(impulse_path).astype(bool)
    enders = scene.read_npy_xz(ender_path).astype(bool)
    recurrence_codes, recurrence_info = scene.grade_track(track)
    if recurrence_info["detector_version"] != 4 or recurrence_info["halo_frames"] != 3:
        raise ValueError(f"{spec.name}: recurrence baseline is not detector v4 with halo 3")
    baseline = scene.policy_masks(candidate, recurrence_codes, impulse)[score.BASELINE_POLICY]
    window = scene.variants.dilate(enders, RADIUS_FRAMES)
    protected = score.subtract_protection(baseline, window)
    return baseline, protected, window


def run_score() -> dict[str, Any]:
    """Score the fixed counterfactual on all three fixtures."""
    contacts, _first_contacts, final_contacts = score.load_contact_roles()
    baseline_total = scene.empty_total_metrics()
    protected_total = scene.empty_total_metrics()
    fixture_results: dict[str, Any] = {}
    ender_events = 0
    for spec in scene.audit.FIXTURES:
        baseline, protected, window = fixture_masks(spec)
        spans = scene.variants.load_visual_spans(spec.name)
        baseline_metrics = scene.mask_metrics(
            baseline,
            contacts[spec.video_id],
            final_contacts[spec.video_id],
            spans,
            spec.fps,
        )
        protected_metrics = scene.mask_metrics(
            protected,
            contacts[spec.video_id],
            final_contacts[spec.video_id],
            spans,
            spec.fps,
        )
        scene.add_to_total(baseline_total, baseline_metrics)
        scene.add_to_total(protected_total, protected_metrics)
        ender_path = scene.ANALYSIS_ROOT / f"{spec.name}_tp_rally_ender_mask.npy.xz"
        fixture_enders = scene.read_npy_xz(ender_path).astype(bool)
        ender_events += int(fixture_enders.sum())
        fixture_results[spec.name] = {
            "video_id": spec.video_id,
            "fps": spec.fps,
            "ender_events": int(fixture_enders.sum()),
            "window_frames": int(window.sum()),
            "baseline": baseline_metrics,
            "protected": protected_metrics,
            "cleared_frames": int(np.count_nonzero(baseline & ~protected)),
        }

    if baseline_total["selected_frames"] != score.EXPECTED_BASELINE_SELECTED:
        raise AssertionError("rally-ending counterfactual did not reproduce the baseline")
    stress_preserved = (
        protected_total["visual_positive_spans_hit"]
        == baseline_total["visual_positive_spans_hit"]
    )
    cleared_frames = baseline_total["selected_frames"] - protected_total["selected_frames"]
    return {
        "schema_version": 1,
        "baseline_commit": scene.BASELINE_SHA,
        "status": "GT-qualified fixture counterfactual; not deployable",
        "configuration": {
            "baseline_policy": score.BASELINE_POLICY,
            "protection": "inclusive ±3 frames around GT-qualified rally-ending events",
            "radius_frames": RADIUS_FRAMES,
        },
        "input_digests": {
            "shots_master_md5": scene.SHOTS_MASTER_MD5,
            "ender_mask_sha256": EXPECTED_ENDER_SHA256,
            "policy_input_sha256": score.EXPECTED_POLICY_INPUT_SHA256,
        },
        "baseline": baseline_total,
        "protected": protected_total,
        "decision": {
            "ender_event_count": ender_events,
            "cleared_frame_count": cleared_frames,
            "stress_spans_preserved": stress_preserved,
            "run_fixed_upstream_e2e": cleared_frames > 0 and stress_preserved,
        },
        "fixtures": fixture_results,
    }


def parse_args() -> argparse.Namespace:
    """Parse the retained result path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Run and retain the rally-ending counterfactual score."""
    args = parse_args()
    result = run_score()
    scene.write_json_gz(args.output, result)
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
