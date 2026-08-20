"""Compare the pinned RANSAC audit with scene-boundary-aware model windows.

This is an analysis-only experiment. It reuses the original fitter and changes
only whether a scheduled 16-frame model window may cross a raw scene cut.

Run from the repository root::

    ~/.venvs/badminton-cicd/bin/python \
        scratch/ransac_scene_guard/scene_aware_ransac.py
"""

from __future__ import annotations

# This standalone script adds the audit and project folders before importing them.
# ruff: noqa: E402

import argparse
import csv
import gzip
import hashlib
import io
import json
import lzma
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = REPO_ROOT / "docs/scraper_pipeline/inpaint_hallucination_fix"
ANALYSIS_ROOT = AUDIT_ROOT / "analysis"
E2E_RUN_ROOT = REPO_ROOT / "experiments/annotator/runs/20260730-041328"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results/scene_aware_ransac.json.gz"
BASELINE_SHA = "3c217d4cc8ab3698e825315a218735322f938a00"
CONTACT_TOLERANCES_BASE30 = (5, 10, 15)
IMPULSE_VETO_RADIUS_FRAMES = 3
SHOTS_MASTER_MD5 = "39cdc201057050abfe4c6f8770734fde"

if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import audit_production_variants as variants
import audit_tracks as audit

from annotator.fps_constants import ScalingKind
from annotator.inpaint_guard import NO_FLAG, grade_track


@dataclass(frozen=True)
class WindowDiagnostics:
    """Counts for one complete pass over the scheduled model windows."""

    scheduled: int = 0
    cut_crossing_scheduled: int = 0
    cut_crossing_skipped: int = 0
    masked: int = 0
    cut_crossing_masked: int = 0
    fit_failed: int = 0
    cut_crossing_fit_failed: int = 0
    accepted: int = 0
    cut_crossing_accepted: int = 0


def file_digest(path: Path, algorithm: str) -> str:
    """Return a streaming file digest.

    :param path: File to hash.
    :param algorithm: Name accepted by :func:`hashlib.new`.
    :return: Lower-case hexadecimal digest.
    """
    digest = hashlib.new(algorithm)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_npy_xz(path: Path) -> np.ndarray:
    """Load an XZ-wrapped NumPy array without pickle."""
    with lzma.open(path, "rb") as source:
        value = np.load(source, allow_pickle=False)
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{path}: expected a NumPy array")
    return value


def load_raw_track_digests() -> dict[str, str]:
    """Load the pinned SHA-256 digest for each compressed audit track."""
    path = AUDIT_ROOT / "raw_manifest.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as source:
        payload = json.load(source)
    fixtures = payload.get("fixtures") if isinstance(payload, dict) else None
    if not isinstance(fixtures, list):
        raise TypeError(f"{path}: expected a fixtures list")
    digests: dict[str, str] = {}
    for row in fixtures:
        if not isinstance(row, dict):
            raise TypeError(f"{path}: fixture row is not an object")
        fixture = row.get("fixture")
        digest = row.get("stored_track_sha256")
        if not isinstance(fixture, str) or not isinstance(digest, str):
            raise TypeError(f"{path}: fixture digest row is incomplete")
        digests[fixture] = digest
    return digests


def load_scene_boundaries(spec: audit.FixtureSpec, n_frames: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Load and validate internal raw scene boundaries for one fixture.

    :param spec: Pinned RANSAC fixture description.
    :param n_frames: Track length that the scene intervals must cover.
    :return: Internal cut frames and compact provenance.
    """
    manifest_path = REPO_ROOT / spec.source_manifest
    with manifest_path.open(encoding="utf-8") as source:
        manifest = json.load(source)
    artifacts = manifest.get("shared_artifacts") if isinstance(manifest, dict) else None
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise ValueError(f"{manifest_path}: expected one shared raw-cut artifact")
    artifact = artifacts[0]
    if not isinstance(artifact, dict):
        raise TypeError(f"{manifest_path}: raw-cut artifact is not an object")
    relative_path = artifact.get("path")
    expected_md5 = artifact.get("md5")
    if not isinstance(relative_path, str) or not isinstance(expected_md5, str):
        raise TypeError(f"{manifest_path}: raw-cut artifact lacks path or MD5")
    cuts_path = E2E_RUN_ROOT / relative_path
    actual_md5 = file_digest(cuts_path, "md5")
    if actual_md5 != expected_md5:
        raise ValueError(f"{spec.name}: raw-cut MD5 differs from its fixture manifest")

    with cuts_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["scene_index", "start_frame", "end_frame"]:
            raise ValueError(f"{cuts_path}: unexpected columns {reader.fieldnames!r}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{cuts_path}: raw cuts are empty")

    intervals: list[tuple[int, int]] = []
    expected_start = 0
    for expected_index, row in enumerate(rows):
        scene_index = int(row["scene_index"])
        start = int(row["start_frame"])
        stop = int(row["end_frame"])
        if scene_index != expected_index:
            raise ValueError(f"{cuts_path}: scene indexes are not consecutive")
        if start != expected_start or stop <= start:
            raise ValueError(f"{cuts_path}: intervals are not positive and contiguous")
        intervals.append((start, stop))
        expected_start = stop
    if expected_start != n_frames:
        raise ValueError(f"{cuts_path}: final scene ends at {expected_start}, expected {n_frames}")

    boundaries = np.asarray([start for start, _stop in intervals[1:]], dtype=np.int64)
    provenance = {
        "path": str(cuts_path.relative_to(REPO_ROOT)),
        "md5": actual_md5,
        "scene_count": len(intervals),
        "internal_cut_count": len(boundaries),
        "minimum_scene_frames": min(stop - start for start, stop in intervals),
        "maximum_scene_frames": max(stop - start for start, stop in intervals),
    }
    return boundaries, provenance


def window_crosses_cut(start: int, stop: int, boundaries: np.ndarray) -> bool:
    """Return whether half-open ``[start, stop)`` contains an internal cut."""
    next_index = int(np.searchsorted(boundaries, start, side="right"))
    return next_index < len(boundaries) and int(boundaries[next_index]) < stop


def finish_ransac_fields(
    masked: np.ndarray,
    eligible_windows: np.ndarray,
    outlier_votes: np.ndarray,
    maximum_residual: np.ndarray,
) -> dict[str, np.ndarray]:
    """Derive the per-frame vote fraction and final candidate mask."""
    minimum_votes = np.maximum(1, (eligible_windows + 1) // 2)
    candidate = ~masked & (eligible_windows > 0) & (outlier_votes >= minimum_votes)
    outlier_fraction = np.divide(
        outlier_votes,
        eligible_windows,
        out=np.zeros(len(masked), dtype=np.float64),
        where=eligible_windows > 0,
    )
    return {
        "eligible_windows": eligible_windows,
        "outlier_votes": outlier_votes,
        "outlier_fraction": outlier_fraction,
        "maximum_residual_px": maximum_residual,
        "candidate": candidate,
        "masked": masked,
    }


def run_ransac_with_boundaries(
    track: np.ndarray,
    boundaries: np.ndarray,
    *,
    exclude_cut_crossing: bool,
) -> tuple[dict[str, np.ndarray], WindowDiagnostics]:
    """Run the original fitter with an optional raw-cut window gate.

    :param track: Dense ``(frames, 3+)`` normalised shuttle track.
    :param boundaries: First frame of each scene after the opening scene.
    :param exclude_cut_crossing: Skip a model window before fitting when it
        crosses an internal scene boundary.
    :return: Original per-frame RANSAC fields and window counts.
    """
    points = audit.pixel_points(track)
    masked = np.all(track[:, :2] == 0, axis=1)
    eligible_windows = np.zeros(len(track), dtype=np.int16)
    outlier_votes = np.zeros(len(track), dtype=np.int16)
    maximum_residual = np.zeros(len(track), dtype=np.float64)
    triples = audit.ransac_triples(20260731)
    frame_offsets = np.arange(audit.WINDOW, dtype=np.float64)
    design = np.column_stack(
        (
            np.ones(audit.WINDOW, dtype=np.float64),
            frame_offsets,
            frame_offsets**2,
        )
    )
    sample_solvers = np.linalg.inv(design[triples])

    counts = {
        "scheduled": 0,
        "cut_crossing_scheduled": 0,
        "cut_crossing_skipped": 0,
        "masked": 0,
        "cut_crossing_masked": 0,
        "fit_failed": 0,
        "cut_crossing_fit_failed": 0,
        "accepted": 0,
        "cut_crossing_accepted": 0,
    }
    for start in range(0, len(track) - audit.WINDOW + 1, audit.WINDOW_STEP):
        stop = start + audit.WINDOW
        crosses_cut = window_crosses_cut(start, stop, boundaries)
        counts["scheduled"] += 1
        counts["cut_crossing_scheduled"] += int(crosses_cut)
        if exclude_cut_crossing and crosses_cut:
            counts["cut_crossing_skipped"] += 1
            continue

        window_masked = masked[start:stop]
        if window_masked.any():
            counts["masked"] += 1
            counts["cut_crossing_masked"] += int(crosses_cut)
            continue
        residuals = audit.fit_quadratic_ransac(
            points[start:stop],
            design,
            triples,
            sample_solvers,
            audit.JITTER_RADIUS_PX,
        )
        if residuals is None:
            counts["fit_failed"] += 1
            counts["cut_crossing_fit_failed"] += int(crosses_cut)
            continue

        counts["accepted"] += 1
        counts["cut_crossing_accepted"] += int(crosses_cut)
        window_slice = slice(start, stop)
        eligible_windows[window_slice] += 1
        outlier_votes[window_slice] += residuals > audit.JITTER_RADIUS_PX
        maximum_residual[window_slice] = np.maximum(maximum_residual[window_slice], residuals)

    fields = finish_ransac_fields(masked, eligible_windows, outlier_votes, maximum_residual)
    return fields, WindowDiagnostics(**counts)


def assert_baseline_matches(
    fixture: str,
    track: np.ndarray,
    baseline: dict[str, np.ndarray],
) -> None:
    """Prove the unchanged arm matches source and tracked audit fields."""
    source = audit.run_ransac(track, seed=20260731, jitter_radius_px=audit.JITTER_RADIUS_PX)
    for field in source:
        if not np.array_equal(baseline[field], source[field]):
            raise AssertionError(f"{fixture}: unchanged arm differs from source field {field}")

    stored_candidate = read_npy_xz(ANALYSIS_ROOT / f"{fixture}_ransac_candidate.npy.xz").astype(bool)
    if not np.array_equal(baseline["candidate"], stored_candidate):
        raise AssertionError(f"{fixture}: unchanged arm differs from stored candidate mask")

    stored_fields = variants.load_frame_fields(fixture)
    if not np.array_equal(baseline["eligible_windows"], stored_fields["eligible"].astype(np.int16)):
        raise AssertionError(f"{fixture}: eligible-window fields differ from the stored audit")
    if not np.array_equal(baseline["outlier_votes"], stored_fields["votes"].astype(np.int16)):
        raise AssertionError(f"{fixture}: outlier-vote fields differ from the stored audit")
    for current_name, stored_name in (
        ("outlier_fraction", "fraction"),
        ("maximum_residual_px", "residual"),
    ):
        if not np.allclose(
            baseline[current_name], stored_fields[stored_name], atol=5.1e-7, rtol=0.0
        ):
            raise AssertionError(f"{fixture}: {current_name} differs from the rounded stored audit")


def scaled_contact_radius(base30_frames: int, fps: float) -> int:
    """Scale one base-30 contact tolerance with the annotator's half-up rule."""
    return int(ScalingKind.FRAME_COUNT.scale(base30_frames, fps))


def frames_near_events(mask: np.ndarray, events: Iterable[int], radius: int) -> int:
    """Count selected frames inside the union of event neighbourhoods."""
    event_mask = np.zeros(len(mask), dtype=bool)
    event_list = list(events)
    event_mask[event_list] = True
    return int((mask & variants.dilate(event_mask, radius)).sum())


def mask_metrics(
    mask: np.ndarray,
    contacts: set[int],
    final_contacts: set[int],
    spans: list[tuple[int, int]],
    fps: float,
) -> dict[str, Any]:
    """Measure one candidate policy against the fixed safety evidence."""
    metrics = variants.metric(mask, contacts, final_contacts, spans)
    metrics.pop("contacts_within_radius3")
    metrics.pop("final_contacts_within_radius3")
    tolerances: dict[str, dict[str, int]] = {}
    for base30_frames in CONTACT_TOLERANCES_BASE30:
        radius = scaled_contact_radius(base30_frames, fps)
        contacts_with_candidate = 0
        for frame in contacts:
            contacts_with_candidate += int(
                mask[max(0, frame - radius) : min(len(mask), frame + radius + 1)].any()
            )
        finals_with_candidate = 0
        for frame in final_contacts:
            finals_with_candidate += int(
                mask[max(0, frame - radius) : min(len(mask), frame + radius + 1)].any()
            )
        tolerances[f"base30_{base30_frames}"] = {
            "radius_frames": radius,
            "contacts_with_candidate": contacts_with_candidate,
            "final_contacts_with_candidate": finals_with_candidate,
            "candidate_frames_near_contacts": frames_near_events(mask, contacts, radius),
            "candidate_frames_near_final_contacts": frames_near_events(mask, final_contacts, radius),
        }
    metrics["contact_tolerances"] = tolerances
    return metrics


def policy_masks(
    candidate: np.ndarray,
    recurrence_codes: np.ndarray,
    impulse_events: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build the three predeclared policies for the scene comparison."""
    guard_clean = candidate & (recurrence_codes == NO_FLAG)
    impulse_neighbourhood = variants.dilate(impulse_events, IMPULSE_VETO_RADIUS_FRAMES)
    return {
        "raw_candidate": candidate,
        "recurrence_v4_clean": guard_clean,
        "recurrence_v4_clean_impulse_veto_radius3": guard_clean & ~impulse_neighbourhood,
    }


def compare_policy(
    baseline: np.ndarray,
    scene_aware: np.ndarray,
    contacts: set[int],
    final_contacts: set[int],
    spans: list[tuple[int, int]],
    fps: float,
) -> dict[str, Any]:
    """Return baseline, scene-aware, removed, and added policy measures."""
    removed = baseline & ~scene_aware
    added = scene_aware & ~baseline
    return {
        "baseline": mask_metrics(baseline, contacts, final_contacts, spans, fps),
        "scene_aware": mask_metrics(scene_aware, contacts, final_contacts, spans, fps),
        "removed": mask_metrics(removed, contacts, final_contacts, spans, fps),
        "added": mask_metrics(added, contacts, final_contacts, spans, fps),
        "net_selected_frames": int(scene_aware.sum() - baseline.sum()),
    }


def span_rows(
    spans: list[tuple[int, int]],
    baseline_policies: dict[str, np.ndarray],
    scene_policies: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Record each stress span separately so coverage cannot hide a loss."""
    rows: list[dict[str, Any]] = []
    for index, (start, stop) in enumerate(spans):
        policies: dict[str, dict[str, Any]] = {}
        for policy in baseline_policies:
            baseline_frames = np.flatnonzero(baseline_policies[policy][start:stop]) + start
            scene_frames = np.flatnonzero(scene_policies[policy][start:stop]) + start
            policies[policy] = {
                "baseline_frames": baseline_frames.tolist(),
                "scene_aware_frames": scene_frames.tolist(),
                "baseline_hit": bool(len(baseline_frames)),
                "scene_aware_hit": bool(len(scene_frames)),
            }
        rows.append({
            "span_index": index,
            "start_frame": start,
            "stop_frame_exclusive": stop,
            "policies": policies,
        })
    return rows


def empty_total_metrics() -> dict[str, Any]:
    """Create one aggregate metric accumulator without fixture-specific radii."""
    return {
        "selected_frames": 0,
        "visual_positive_frames": 0,
        "visual_positive_spans_hit": 0,
        "exact_contacts": 0,
        "exact_final_contacts": 0,
        "contact_tolerances": {
            f"base30_{value}": {
                "contacts_with_candidate": 0,
                "final_contacts_with_candidate": 0,
                "candidate_frames_near_contacts": 0,
                "candidate_frames_near_final_contacts": 0,
            }
            for value in CONTACT_TOLERANCES_BASE30
        },
    }


def add_to_total(total: dict[str, Any], metrics: dict[str, Any]) -> None:
    """Add one fixture's metrics to an aggregate in place."""
    for field in (
        "selected_frames",
        "visual_positive_frames",
        "visual_positive_spans_hit",
        "exact_contacts",
        "exact_final_contacts",
    ):
        total[field] += metrics[field]
    for tolerance, values in metrics["contact_tolerances"].items():
        for field in (
            "contacts_with_candidate",
            "final_contacts_with_candidate",
            "candidate_frames_near_contacts",
            "candidate_frames_near_final_contacts",
        ):
            total["contact_tolerances"][tolerance][field] += values[field]


def initialise_policy_totals() -> dict[str, dict[str, Any]]:
    """Create aggregate slots for every policy and comparison arm."""
    arms = ("baseline", "scene_aware", "removed", "added")
    return {
        policy: {arm: empty_total_metrics() for arm in arms}
        for policy in (
            "raw_candidate",
            "recurrence_v4_clean",
            "recurrence_v4_clean_impulse_veto_radius3",
        )
    }


def write_json_gz(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic UTF-8 JSON in a gzip stream."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("wb") as raw_target,
        gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_target,
            compresslevel=9,
            mtime=0,
        ) as compressed,
        io.TextIOWrapper(compressed, encoding="utf-8") as text_target,
    ):
        json.dump(payload, text_target, indent=2, sort_keys=True)
        text_target.write("\n")


def run_experiment() -> dict[str, Any]:
    """Run and verify the complete three-fixture scene comparison."""
    expected_track_digests = load_raw_track_digests()
    contacts_by_video, final_contacts_by_video = variants.load_contact_frames()
    if file_digest(variants.REPO / "training/data/shuttleset/annotations/shots_master.csv", "md5") != SHOTS_MASTER_MD5:
        raise ValueError("ShuttleSet shots_master.csv differs from the pinned fixture input")

    fixture_results: dict[str, Any] = {}
    policy_totals = initialise_policy_totals()
    total_windows = {field: 0 for field in asdict(WindowDiagnostics())}
    input_digests: dict[str, Any] = {}

    for spec in audit.FIXTURES:
        track_path = spec.raw_path
        actual_track_sha256 = file_digest(track_path, "sha256")
        if actual_track_sha256 != expected_track_digests.get(spec.name):
            raise ValueError(f"{spec.name}: compressed track differs from raw_manifest.json.gz")
        track = read_npy_xz(track_path)
        if track.shape != (len(track), 3) or not np.isfinite(track).all():
            raise ValueError(f"{spec.name}: expected a finite (frames, 3) track")

        boundaries, cut_provenance = load_scene_boundaries(spec, len(track))
        baseline, baseline_windows = run_ransac_with_boundaries(
            track, boundaries, exclude_cut_crossing=False
        )
        assert_baseline_matches(spec.name, track, baseline)
        scene_aware, scene_windows = run_ransac_with_boundaries(
            track, boundaries, exclude_cut_crossing=True
        )
        if scene_windows.cut_crossing_skipped != baseline_windows.cut_crossing_scheduled:
            raise AssertionError(f"{spec.name}: scene arm did not skip every crossing window")

        recurrence_codes, recurrence_info = grade_track(track)
        if recurrence_info["detector_version"] != 4 or recurrence_info["halo_frames"] != 3:
            raise ValueError(f"{spec.name}: recurrence baseline is not detector v4 with halo 3")
        impulse_path = ANALYSIS_ROOT / f"{spec.name}_impulse_event_mask.npy.xz"
        impulse_events = read_npy_xz(impulse_path).astype(bool)
        if impulse_events.shape != (len(track),):
            raise ValueError(f"{spec.name}: impulse mask length differs from the track")

        contacts = contacts_by_video[spec.video_id]
        final_contacts = final_contacts_by_video[spec.video_id]
        spans = variants.load_visual_spans(spec.name)
        baseline_policies = policy_masks(baseline["candidate"], recurrence_codes, impulse_events)
        scene_policies = policy_masks(scene_aware["candidate"], recurrence_codes, impulse_events)
        comparisons: dict[str, Any] = {}
        for policy in baseline_policies:
            comparison = compare_policy(
                baseline_policies[policy],
                scene_policies[policy],
                contacts,
                final_contacts,
                spans,
                spec.fps,
            )
            comparisons[policy] = comparison
            for arm in ("baseline", "scene_aware", "removed", "added"):
                add_to_total(policy_totals[policy][arm], comparison[arm])

        for field, value in asdict(baseline_windows).items():
            total_windows[field] += value
        fixture_results[spec.name] = {
            "frames": len(track),
            "fps": spec.fps,
            "window_diagnostics": {
                "baseline": asdict(baseline_windows),
                "scene_aware": asdict(scene_windows),
            },
            "cut_provenance": cut_provenance,
            "current_recurrence_info": recurrence_info,
            "contact_count": len(contacts),
            "final_contact_count": len(final_contacts),
            "contact_radius_frames": {
                f"base30_{value}": scaled_contact_radius(value, spec.fps)
                for value in CONTACT_TOLERANCES_BASE30
            },
            "policies": comparisons,
            "positive_spans": span_rows(spans, baseline_policies, scene_policies),
        }
        input_digests[spec.name] = {
            "track_path": str(track_path.relative_to(REPO_ROOT)),
            "track_sha256": actual_track_sha256,
            "raw_cuts": cut_provenance,
            "impulse_mask_path": str(impulse_path.relative_to(REPO_ROOT)),
            "impulse_mask_sha256": file_digest(impulse_path, "sha256"),
            "visual_spans_sha256": file_digest(
                ANALYSIS_ROOT / f"{spec.name}_visual_hallucination_audit.csv.gz", "sha256"
            ),
        }

    for policy, arms in policy_totals.items():
        arms["net_selected_frames"] = (
            arms["scene_aware"]["selected_frames"] - arms["baseline"]["selected_frames"]
        )

    return {
        "schema_version": 1,
        "baseline_commit": BASELINE_SHA,
        "status": "analysis-only; positive spans are a stress set, not precision evidence",
        "configuration": {
            "resolution_px": [audit.FRAME_WIDTH, audit.FRAME_HEIGHT],
            "window_frames": audit.WINDOW,
            "window_step_frames": audit.WINDOW_STEP,
            "ransac_iterations": audit.RANSAC_ITERATIONS,
            "minimum_inliers": audit.MIN_RANSAC_INLIERS,
            "residual_threshold_px": audit.JITTER_RADIUS_PX,
            "seed": 20260731,
            "candidate_policy": "at least half of eligible windows vote outlier",
            "scene_policy": "exclude a model window when start < cut < stop",
            "impulse_veto_radius_frames": IMPULSE_VETO_RADIUS_FRAMES,
            "contact_tolerances_base30": list(CONTACT_TOLERANCES_BASE30),
        },
        "baseline_verification": {
            "source_fields_exact": True,
            "stored_candidate_masks_exact": True,
            "stored_integer_vote_fields_exact": True,
            "stored_six_decimal_float_fields_within_rounding": True,
            "recurrence_baseline": "detector v4, halo 3, recomputed from each pinned track",
        },
        "input_digests": {
            "shots_master_md5": SHOTS_MASTER_MD5,
            "fixtures": input_digests,
        },
        "window_totals_baseline": total_windows,
        "fixtures": fixture_results,
        "totals": {"policies": policy_totals},
    }


def print_summary(result: dict[str, Any]) -> None:
    """Print the main comparison without dumping the complete result."""
    print("policy,baseline,scene_aware,removed,added,exact_contacts_before,exact_contacts_after,spans_before,spans_after")
    for policy, arms in result["totals"]["policies"].items():
        print(
            f"{policy},"
            f"{arms['baseline']['selected_frames']},"
            f"{arms['scene_aware']['selected_frames']},"
            f"{arms['removed']['selected_frames']},"
            f"{arms['added']['selected_frames']},"
            f"{arms['baseline']['exact_contacts']},"
            f"{arms['scene_aware']['exact_contacts']},"
            f"{arms['baseline']['visual_positive_spans_hit']},"
            f"{arms['scene_aware']['visual_positive_spans_hit']}"
        )


def parse_args() -> argparse.Namespace:
    """Parse the optional retained-result path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Run, retain, and print the scene-aware comparison."""
    args = parse_args()
    result = run_experiment()
    write_json_gz(args.output, result)
    print_summary(result)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
