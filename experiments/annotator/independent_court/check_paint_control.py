"""Check that legacy paint controls are reproduced by the current stripe code.

The checker remeasures archived starts and refits with the original and current
measurement modules.  It compares only the frozen legacy assignment and score
fields; it does not run a new detector or fit.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from zipfile import ZipFile

import cv2
import numpy as np
import scipy

from . import stripe_observations as stripes
from .assignment import Observations, prepare_observations
from .detector import CORNER_COURT_M
from .run_paint_refit import PROJECTION_ROUNDOFF_PX

HISTORICAL_MODULE_NAME = "experiments.annotator.independent_court._historical_stripes"
ZERO_TOLERANCE_PX = 0.0
FIT_SAMPLE_FIELDS = ("fragment_ids", "sample_ids", "intervals", "positions")


def load_historical_stripes(source: Path) -> ModuleType:
    """Load the supplied archived module under a package-qualified name."""
    spec = importlib.util.spec_from_file_location(HISTORICAL_MODULE_NAME, source)
    if spec is None or spec.loader is None:
        raise ImportError("Could not create a loader for the historical stripe module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[HISTORICAL_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def read_member(archive: ZipFile, name: str) -> dict:
    return json.loads(gzip.decompress(archive.read(name)))


def exact_equal(left: object, right: object) -> bool:
    if isinstance(left, (list, tuple, np.ndarray)) or isinstance(
        right, (list, tuple, np.ndarray)
    ):
        return np.array_equal(np.asarray(left), np.asarray(right))
    return bool(left == right)


def require_equal(case_id: str, label: str, left: object, right: object) -> None:
    if not exact_equal(left, right):
        raise ValueError(f"{case_id}: {label} changed")


def maximum_delta(left: object, right: object) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.size == 0 and right_array.size == 0:
        return 0.0
    return float(np.max(np.abs(left_array - right_array)))


def compare_scores(
    case_id: str, identifier: str, current: dict, historical: dict
) -> float:
    """Require the old and current fixed-assignment scores to be bitwise equal."""
    delta = 0.0
    for score_name in ("independent", "exclusive"):
        for metric in ("forward", "reverse"):
            current_value = current[score_name][metric]
            historical_value = historical[score_name][metric]
            delta = max(delta, maximum_delta(current_value, historical_value))
            require_equal(
                case_id,
                f"{identifier} {score_name}.{metric}",
                current_value,
                historical_value,
            )
    for metric in ("independent_per_marking", "exclusive_per_marking"):
        current_value = current[metric]
        historical_value = historical[metric]
        delta = max(delta, maximum_delta(current_value, historical_value))
        require_equal(
            case_id, f"{identifier} {metric}", current_value, historical_value
        )
    for field in ("marking", "position"):
        require_equal(
            case_id,
            f"{identifier} assignments.{field}",
            current["assignments"][field],
            historical["assignments"][field],
        )
    return delta


def native_observations(
    case: dict,
    record: dict,
) -> tuple[tuple[int, int], np.ndarray, Observations, np.ndarray]:
    dimensions = record["dimensions"]
    size = tuple(int(value) for value in record["working_size"])
    scale = np.array(
        [dimensions["width"] / size[0], dimensions["height"] / size[1]], dtype=float
    )
    segments = np.asarray(case["segments_px"], dtype=float).reshape(-1, 4) / np.tile(
        scale, 2
    )
    observations = prepare_observations(segments, size)
    return size, scale, observations, stripes.fragment_weights(observations)


def measured_score(
    module: ModuleType,
    corners_native: object,
    scale: np.ndarray,
    observations: Observations,
    weights: np.ndarray,
    size: tuple[int, int],
    assignment: dict,
    boundary_tolerance_px: float | None,
) -> dict:
    corners = np.asarray(corners_native, dtype=float)
    homography = cv2.getPerspectiveTransform(
        CORNER_COURT_M, (corners / scale).astype(np.float32)
    )
    if boundary_tolerance_px is None:
        evidence = module.measure(homography, observations, size)
    else:
        evidence = module.measure(
            homography, observations, size, module.SEGMENTS_M, boundary_tolerance_px
        )
    return module.score_model(evidence, weights, 3, assignment)


def check_fit_samples(case_id: str, start: dict, fixed_entry: dict) -> float:
    current = start.get("fit_samples")
    archived = fixed_entry["fit_samples"]
    if current is None:
        raise ValueError(f"{case_id}: legacy start has no fit_samples")
    for field in FIT_SAMPLE_FIELDS:
        require_equal(
            case_id,
            f"{start['id']} fit_samples.{field}",
            current[field],
            archived[field],
        )
    delta = maximum_delta(current["weights"], archived["weights"])
    if delta > 1e-12:
        raise ValueError(f"{case_id}: {start['id']} fit sample weights changed")
    return delta


def check_records(
    results: dict, marking: dict, fixed: dict, archived: dict, historical: ModuleType
) -> dict:
    result_records = {record["id"]: record for record in results["records"]}
    marking_cases = {case["id"]: case for case in marking["cases"]}
    fixed_records = {record["id"]: record for record in fixed["records"]}
    archived_records = {record["id"]: record for record in archived["records"]}
    if set(result_records) != set(marking_cases) or set(result_records) != set(
        fixed_records
    ):
        raise ValueError(
            "Results, marking inputs and fixed-refit records have different case IDs"
        )

    entries_compared = 0
    starts_compared = 0
    refits_compared = 0
    fits_compared = 0
    max_old_delta = 0.0
    max_score_delta = 0.0
    changed_scores = 0
    max_weights_delta = 0.0
    assignment_difference_ids = []

    for case_id, record in result_records.items():
        case = marking_cases[case_id]
        size, scale, observations, weights = native_observations(case, record)
        fixed_entries = {
            entry["id"]: entry for entry in fixed_records[case_id]["entries"]
        }
        archived_entries = {
            entry["id"]: entry for entry in archived_records[case_id]["entries"]
        }
        legacy_entries = [
            entry
            for entry in record["entries"]
            if entry["model"] == "legacy" and entry["stage"] in ("start", "refit")
        ]
        expected_ids = {
            entry["parent_id"]
            + ("/start" if entry["stage"] == "start" else "/fixed_position")
            for entry in legacy_entries
        }
        allowed_ids = {
            entry["id"]
            for entry in archived_entries.values()
            if entry["model"] in ("start", "fixed_position")
        }
        if expected_ids != allowed_ids:
            raise ValueError(f"{case_id}: legacy start/refit IDs differ from archive")

        current_starts = {
            entry["parent_id"]: entry["stripe"]["assignments"]
            for entry in legacy_entries
            if entry["stage"] == "start"
        }
        for parent_id, current_assignment in current_starts.items():
            archived_assignment = archived_entries[f"{parent_id}/start"]["stripe"][
                "assignments"
            ]
            if not all(
                exact_equal(current_assignment[field], archived_assignment[field])
                for field in ("marking", "position")
            ):
                assignment_difference_ids.append(f"{case_id}/{parent_id}")
        for entry in legacy_entries:
            entries_compared += 1
            if entry["stage"] == "start":
                starts_compared += 1
                archive_id = f"{entry['parent_id']}/start"
            else:
                refits_compared += 1
                archive_id = f"{entry['parent_id']}/fixed_position"
            previous = archived_entries[archive_id]
            require_equal(
                case_id,
                f"{entry['id']} eligibility",
                entry["eligible"],
                previous["eligible"],
            )
            assignment = current_starts[entry["parent_id"]]
            zero = measured_score(
                stripes,
                entry["corners_px"],
                scale,
                observations,
                weights,
                size,
                assignment,
                0.0,
            )
            old = measured_score(
                historical,
                entry["corners_px"],
                scale,
                observations,
                weights,
                size,
                assignment,
                None,
            )
            max_old_delta = max(
                max_old_delta, compare_scores(case_id, entry["id"], zero, old)
            )
            stable = measured_score(
                stripes,
                entry["corners_px"],
                scale,
                observations,
                weights,
                size,
                assignment,
                PROJECTION_ROUNDOFF_PX,
            )
            saved_score = entry["stripe"]["exclusive"]["score"]
            require_equal(
                case_id,
                f"{entry['id']} stable exclusive score",
                stable["exclusive"]["score"],
                saved_score,
            )
            stable_minus_zero = (
                stable["exclusive"]["score"] - zero["exclusive"]["score"]
            )
            max_score_delta = max(max_score_delta, stable_minus_zero)
            changed_scores += int(stable_minus_zero != 0.0)
            if entry["stage"] == "start":
                fixed_entry = fixed_entries[entry["parent_id"]]
                max_weights_delta = max(
                    max_weights_delta, check_fit_samples(case_id, entry, fixed_entry)
                )
                fits_compared += 1

    if max_old_delta != 0.0:
        raise ValueError("Historical and current zero-tolerance scores differ")
    return {
        "records_compared": len(result_records),
        "legacy_entries_compared": entries_compared,
        "legacy_starts_compared": starts_compared,
        "legacy_refits_compared": refits_compared,
        "fit_sample_records_compared": fits_compared,
        "max_old_source_vs_new_zero_delta": max_old_delta,
        "max_stable_minus_zero_score": max_score_delta,
        "changed_scores_count": changed_scores,
        "max_archive_fit_weights_delta": max_weights_delta,
        "assignment_differences_count": len(assignment_difference_ids),
        "assignment_difference_ids": assignment_difference_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, required=True, help="Paired run_paint_refit JSON gzip"
    )
    parser.add_argument(
        "--recorded",
        type=Path,
        required=True,
        help="Recorded player-guided archive directory",
    )
    parser.add_argument(
        "--historical-stripes",
        type=Path,
        required=True,
        help="a8c13ab stripe_observations.py",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSON gzip")
    args = parser.parse_args()

    results = json.loads(gzip.decompress(args.results.read_bytes()))
    historical = load_historical_stripes(args.historical_stripes)
    with ZipFile(args.recorded / "marking_refit_replay.zip") as archive:
        marking = read_member(archive, "marking_inputs.json.gz")
    with ZipFile(args.recorded / "stripe_diagnostics.zip") as archive:
        fixed = read_member(archive, "fixed_refit_v2.json.gz")
        archived = read_member(archive, "refit_selection_v2.json.gz")

    cv2.setNumThreads(1)
    counts = check_records(results, marking, fixed, archived, historical)
    output = {
        "schema": "paint-control-check/1",
        "development_data": True,
        "acceptance_evaluated": False,
        "historical_commit": "a8c13ab",
        "zero_boundary_tolerance_px": ZERO_TOLERANCE_PX,
        "stable_boundary_tolerance_px": PROJECTION_ROUNDOFF_PX,
        "fixed_assignment": "current recorded legacy start stripe assignments",
        "archive_members": [
            "marking_inputs.json.gz",
            "fixed_refit_v2.json.gz",
            "refit_selection_v2.json.gz",
        ],
        "versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "opencv": cv2.__version__,
        },
        **counts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        gzip.compress(
            json.dumps(output, allow_nan=False, separators=(",", ":")).encode(), mtime=0
        )
    )


if __name__ == "__main__":
    main()
