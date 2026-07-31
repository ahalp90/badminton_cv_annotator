"""Measure sidecar inpaint provenance agreement before and after event context."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import numpy as np

from audit_tracks import FIXTURES, FixtureSpec, load_fixture_track, load_sidecar_mask
from compressed_io import read_npy_xz, write_json_gz


def percentage(count: int, denominator: int) -> float:
    if denominator == 0:
        raise ValueError("coverage percentage requires a non-empty denominator")
    return 100.0 * count / denominator


def count_span_hits(
    mask: np.ndarray,
    valid: np.ndarray,
    spans: list[list[int]],
) -> int:
    hit_count = 0
    for start, stop in spans:
        if np.any(mask[start:stop] & valid[start:stop]):
            hit_count += 1
    return hit_count


def frame_metric(mask: np.ndarray, sidecar_mask: np.ndarray, valid: np.ndarray) -> dict[str, object]:
    denominator = int((sidecar_mask & valid).sum())
    count = int((sidecar_mask & valid & mask).sum())
    return {
        "count": count,
        "denominator": denominator,
        "percentage": percentage(count, denominator),
    }


def span_metric(
    mask: np.ndarray,
    valid: np.ndarray,
    spans: list[list[int]],
    coordinate_valid_span_count: int,
) -> dict[str, object]:
    count = count_span_hits(mask, valid, spans)
    return {
        "count": count,
        "denominator": coordinate_valid_span_count,
        "percentage": percentage(count, coordinate_valid_span_count),
    }


def measure_fixture(spec: FixtureSpec, analysis_dir: Path) -> dict[str, object]:
    track = load_fixture_track(spec)
    sidecar_mask, sidecar_payload = load_sidecar_mask(spec, len(track))
    spans = cast(list[list[int]], sidecar_payload["inpaint_selected"])

    guard_codes = read_npy_xz(analysis_dir / f"{spec.name}_guard_codes.npy.xz")
    uncaught = read_npy_xz(analysis_dir / f"{spec.name}_uncaught_mask.npy.xz")
    impulse = read_npy_xz(analysis_dir / f"{spec.name}_impulse_event_mask.npy.xz")
    tp_rally_ender = read_npy_xz(
        analysis_dir / f"{spec.name}_tp_rally_ender_mask.npy.xz"
    )
    valid = ~((track[:, 0] == 0) & (track[:, 1] == 0))

    baseline_guard = guard_codes != 0
    union_one = uncaught | sidecar_mask | (impulse & valid)
    union_two = uncaught | (impulse & valid) | (tp_rally_ender & valid)
    augmented_tag = baseline_guard | union_two

    coordinate_valid_span_count = 0
    for start, stop in spans:
        if np.any(valid[start:stop]):
            coordinate_valid_span_count += 1

    return {
        "fixture": spec.name,
        "sidecar": {
            "span_count": len(spans),
            "coordinate_valid_span_count": coordinate_valid_span_count,
            "selected_frame_count": int(sidecar_mask.sum()),
            "coordinate_valid_frame_count": int((sidecar_mask & valid).sum()),
        },
        "baseline_guard": {
            "frame_metric": frame_metric(baseline_guard, sidecar_mask, valid),
            "span_metric": span_metric(
                baseline_guard, valid, spans, coordinate_valid_span_count
            ),
        },
        "union_one": {
            "frame_metric": frame_metric(union_one, sidecar_mask, valid),
            "span_metric": span_metric(
                union_one, valid, spans, coordinate_valid_span_count
            ),
            "interpretation": (
                "100% for sidecar-selected valid frames and coordinate-valid "
                "sidecar spans by construction because Union 1 includes the "
                "sidecar mask"
            ),
        },
        "union_two": {
            "frame_metric": frame_metric(union_two, sidecar_mask, valid),
            "span_metric": span_metric(
                union_two, valid, spans, coordinate_valid_span_count
            ),
        },
        "augmented_guard_or_union_two": {
            "frame_metric": frame_metric(augmented_tag, sidecar_mask, valid),
            "span_metric": span_metric(
                augmented_tag, valid, spans, coordinate_valid_span_count
            ),
            "interpretation": (
                "baseline guard non-zero OR a Union 2 source; an exploratory "
                "evidence tag, not a production detector"
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workset",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="inpaint audit workset directory (default: this script's parent workset)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workset = args.workset.resolve()
    analysis_dir = workset / "analysis"
    summary = {
        "instrument": "measure_inpaint_coverage.py",
        "status": (
            "sidecar provenance agreement; not hallucination recall or detector "
            "precision"
        ),
        "fixture_order": [spec.name for spec in FIXTURES],
        "definitions": {
            "baseline_guard": (
                "a coordinate-valid sidecar frame has guard_code != 0, or a "
                "coordinate-valid sidecar span contains one"
            ),
            "union_one": (
                "uncaught | sidecar_inpaint | impulse; it is 100% for "
                "sidecar-selected valid frames and coordinate-valid sidecar "
                "spans by construction"
            ),
            "union_two": "uncaught | impulse | inductive_tp_rally_ender",
            "augmented_guard_or_union_two": (
                "baseline_guard OR union_two; a broader exploratory evidence tag"
            ),
            "span_denominator": (
                "sidecar inpaint spans with at least one coordinate-valid frame"
            ),
            "frame_denominator": "coordinate-valid frames selected by the sidecar",
        },
        "fixtures": [measure_fixture(spec, analysis_dir) for spec in FIXTURES],
    }
    write_json_gz(analysis_dir / "inpaint_coverage.json.gz", summary)
    for fixture in summary["fixtures"]:
        assert isinstance(fixture, dict)
        frame_metric_data = fixture["augmented_guard_or_union_two"]["frame_metric"]
        span_metric_data = fixture["augmented_guard_or_union_two"]["span_metric"]
        assert isinstance(frame_metric_data, dict)
        assert isinstance(span_metric_data, dict)
        print(
            f"{fixture['fixture']}: augmented frame {frame_metric_data['percentage']:.2f}% "
            f"and span {span_metric_data['percentage']:.2f}%"
        )


if __name__ == "__main__":
    main()
