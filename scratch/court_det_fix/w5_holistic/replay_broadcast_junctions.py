"""Repair the historical broadcast junction rankings without regenerating courts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import sys
import zipfile
from itertools import combinations
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np

REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(REPOSITORY / "src"))

from experiments.annotator.independent_court import detector, junction_observations
from experiments.annotator.independent_court.assignment import prepare_observations
from experiments.annotator.independent_court.run_assignment import ACCURATE_PX
from experiments.annotator.independent_court.run_junction_selection import rank

ARCHIVE_INPUTS = "broadcast/inputs.json.gz"
ARCHIVE_RESULTS = "broadcast/results.json.gz"
MASK_POLICIES = ("no_mask", "median_two_of_three")
COORDINATE_TOLERANCE_PX = 1.0


def read_archive_json(archive: Path, member: str) -> dict[str, Any]:
    """Read one gzip-compressed JSON member from the historical archive."""
    with zipfile.ZipFile(archive) as bundle:
        return json.loads(gzip.decompress(bundle.read(member)))


def atomic_write_json_gz(path: Path, value: dict[str, Any]) -> None:
    """Write deterministic compressed JSON without exposing a partial result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(gzip.compress(json.dumps(value, allow_nan=False).encode(), mtime=0))
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    """Hash one input artefact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_legacy(directory: Path) -> ModuleType:
    """Load the historical case preparer from an explicit directory."""
    module_path = directory / "run_alignment.py"
    if not module_path.is_file():
        raise FileNotFoundError(module_path)
    sys.path.insert(0, str(directory.resolve()))
    spec = importlib.util.spec_from_file_location("historical_marking_refit_run_alignment", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def no_mask_boxes() -> np.ndarray:
    """Return the empty box set used by the unmasked comparison."""
    return np.empty((0, 4), dtype=float)


def _sample_boxes(sample: dict[str, Any], score_cutoff: float) -> np.ndarray:
    boxes = np.asarray(sample["bboxes"], dtype=float).reshape(-1, 4)
    scores = np.asarray(sample["scores"], dtype=float)
    if len(scores) != len(boxes):
        raise ValueError(f"Pose sample {sample['frame_index']} has different box and score counts")
    return boxes[scores >= score_cutoff]


def _validate_image_scale(boxes: np.ndarray, dimensions: dict[str, int], case_id: str) -> None:
    if not np.isfinite(boxes).all():
        raise ValueError(f"{case_id}: pose boxes contain non-finite coordinates")
    if len(boxes) == 0:
        return
    x1, y1, x2, y2 = boxes.T
    if np.any(x2 < x1) or np.any(y2 < y1):
        raise ValueError(f"{case_id}: pose boxes have reversed bounds")
    width = dimensions["width"]
    height = dimensions["height"]
    tolerance = COORDINATE_TOLERANCE_PX
    if np.any(x1 < -tolerance) or np.any(x2 > width + tolerance):
        raise ValueError(f"{case_id}: pose boxes do not use the {width}-pixel image width")
    if np.any(y1 < -tolerance) or np.any(y2 > height + tolerance):
        raise ValueError(f"{case_id}: pose boxes do not use the {height}-pixel image height")


def median_two_of_three_boxes(case: dict[str, Any]) -> np.ndarray:
    """Represent points covered in at least two median frames as rectangle intersections."""
    samples = case["pose_samples"]
    if len(samples) != 3:
        raise ValueError(f"{case['id']}: median mask requires exactly three pose samples")
    frame_indices = [sample["frame_index"] for sample in samples]
    if len(set(frame_indices)) != 3:
        raise ValueError(f"{case['id']}: median pose samples must come from three distinct frames")
    score_cutoff = float(case.get("provenance", {}).get("person_score_cutoff", 0.2))
    boxes_by_frame = [_sample_boxes(sample, score_cutoff) for sample in samples]
    for boxes in boxes_by_frame:
        _validate_image_scale(boxes, case["dimensions"], case["id"])

    intersections = []
    for first, second in combinations(boxes_by_frame, 2):
        if len(first) == 0 or len(second) == 0:
            continue
        left = np.maximum(first[:, None, 0], second[None, :, 0])
        top = np.maximum(first[:, None, 1], second[None, :, 1])
        right = np.minimum(first[:, None, 2], second[None, :, 2])
        bottom = np.minimum(first[:, None, 3], second[None, :, 3])
        overlapping = (right >= left) & (bottom >= top)
        if not overlapping.any():
            continue
        pair_intersections = np.stack((left, top, right, bottom), axis=-1)
        intersections.append(pair_intersections[overlapping])
    if not intersections:
        return no_mask_boxes()
    return np.unique(np.concatenate(intersections), axis=0)


def _ranking_inputs(
    record: dict[str, Any],
    prepared: dict[str, Any],
    observations: Any,
    paint_observations: Any,
    boxes: np.ndarray,
) -> list[dict[str, Any]]:
    size = prepared["size"]
    native_scale = prepared["native_scale"]
    entries = []
    for entry in record["entries"]:
        if entry["model"] != "legacy" or not entry["eligible"]:
            continue
        corners = np.asarray(entry["corners_px"], dtype=float) / native_scale
        homography = cv2.getPerspectiveTransform(detector.CORNER_COURT_M, corners.astype(np.float32))
        measured = junction_observations.measure(homography, observations, boxes, size)
        qualified = junction_observations.measure(homography, paint_observations, boxes, size)
        removed = 0
        for raw_site, painted_site in zip(measured["sites"], qualified["sites"], strict=True):
            for arm in raw_site["disagreements"]:
                if raw_site["expected"].get(arm) is not False:
                    continue
                support = raw_site["arms"].get(arm)
                if support is None or support < junction_observations.PRESENT_SUPPORT:
                    continue
                paint_support = painted_site["arms"].get(arm)
                if paint_support is None or paint_support < junction_observations.PRESENT_SUPPORT:
                    removed += 1
        complete = sum(site["usable"] and len(site["agreements"]) == 2 for site in measured["sites"])
        entries.append({
            "id": entry["id"],
            "stripe_score": entry["score"],
            "complete_agreements": complete,
            "disagreements": measured["disagreements"],
            "qualified_disagreements": measured["disagreements"] - removed,
            "usable_sites": measured["usable_sites"],
        })
    return entries


def measure_orders(
    record: dict[str, Any],
    prepared: dict[str, Any],
    observations: Any,
    paint_observations: Any,
    boxes: np.ndarray,
) -> dict[str, Any]:
    """Remeasure and rank the saved eligible legacy candidates under one mask."""
    entries = _ranking_inputs(record, prepared, observations, paint_observations, boxes)
    original = rank(entries)["complete_agreements_first"]
    qualified_entries = [{**entry, "disagreements": entry["qualified_disagreements"]} for entry in entries]
    paint_qualified = rank(qualified_entries)["complete_agreements_first"]
    return {"orders": {"original": original, "paint_qualified": paint_qualified}, "ranking_inputs": entries}


def validate_historical_orders(case_id: str, measured: dict[str, Any], expected: dict[str, Any]) -> None:
    """Fail before producing corrected evidence when the old run cannot be reproduced."""
    if measured["orders"] != expected:
        raise ValueError(f"{case_id}: current helpers do not reproduce the archived junction orders")


def _winner(order: list[str], entries: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if not order:
        return None
    entry = entries[order[0]]
    return {
        "id": entry["id"],
        "metrics": entry.get("metrics"),
        "corners_px": entry["corners_px"],
    }


def attach_winners(measured: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """Attach only saved metrics and corners for the two selected candidates."""
    source_entries = {entry["id"]: entry for entry in record["entries"]}
    measured["winners"] = {
        name: _winner(order, source_entries) for name, order in measured["orders"].items()
    }
    return measured


def empty_measurement() -> dict[str, Any]:
    """Represent an unchanged case with no eligible legacy candidates."""
    return {
        "orders": {"original": [], "paint_qualified": []},
        "ranking_inputs": [],
        "winners": {"original": None, "paint_qualified": None},
    }


def summarise(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise the fixed 18-reference comparison without scoring unverified views."""
    summary: dict[str, Any] = {
        "scoreable_case_count": sum(case["scoreable"] for case in cases),
        "unverified_case_ids": [case["id"] for case in cases if not case["scoreable"]],
        "empty_pool_case_ids": [case["id"] for case in cases if case["eligible_legacy_count"] == 0],
        "policies": {},
    }
    for policy in MASK_POLICIES:
        policy_summary: dict[str, Any] = {}
        for order_name in ("original", "paint_qualified"):
            accurate = 0
            for case in cases:
                if not case["scoreable"]:
                    continue
                winner = case["policies"][policy]["winners"][order_name]
                if winner is None:
                    continue
                accurate += winner["metrics"]["corner_max_error_px"] <= ACCURATE_PX
            policy_summary[order_name] = {
                "within_15_px": accurate,
                "denominator": summary["scoreable_case_count"],
            }
        comparable = 0
        same = 0
        for case in cases:
            winners = case["policies"][policy]["winners"]
            if winners["original"] is None or winners["paint_qualified"] is None:
                continue
            comparable += 1
            same += winners["original"]["id"] == winners["paint_qualified"]["id"]
        policy_summary["original_vs_paint_qualified"] = {
            "same_winner": same,
            "comparable_cases": comparable,
        }
        summary["policies"][policy] = policy_summary
    return summary


def process_case(
    case: dict[str, Any], record: dict[str, Any], image_root: Path, legacy: ModuleType,
) -> dict[str, Any]:
    """Reproduce the historical order, then measure both corrected mask policies."""
    eligible_count = sum(entry["model"] == "legacy" and entry["eligible"] for entry in record["entries"])
    if eligible_count == 0:
        historical = empty_measurement()
        validate_historical_orders(case["id"], historical, record["junction_orders"])
        return {
            "id": case["id"],
            "scoreable": record["scoreable"],
            "reference_status": record["reference_status"],
            "eligible_legacy_count": 0,
            "historical_order_reproduced": True,
            "policies": {policy: empty_measurement() for policy in MASK_POLICIES},
        }

    frame_path = image_root / case["image"]
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise ValueError(f"Cannot read image for {case['id']}: {case['image']}")
    dimensions = case["dimensions"]
    if frame.shape[:2] != (dimensions["height"], dimensions["width"]):
        raise ValueError(f"{case['id']}: image dimensions do not match the archived input")
    prepared = legacy.prepare_case(case)
    observations = prepare_observations(prepared["segments"], prepared["size"])
    working = cv2.resize(frame, prepared["size"])
    painted = detector._filter_painted_stripes(working, prepared["segments"])
    paint_observations = prepare_observations(painted, prepared["size"])

    historical = measure_orders(
        record, prepared, observations, paint_observations, np.asarray(case["bbox_px"], dtype=float),
    )
    validate_historical_orders(case["id"], historical, record["junction_orders"])
    policies = {
        "no_mask": attach_winners(
            measure_orders(record, prepared, observations, paint_observations, no_mask_boxes()), record,
        ),
        "median_two_of_three": attach_winners(
            measure_orders(
                record, prepared, observations, paint_observations, median_two_of_three_boxes(case),
            ),
            record,
        ),
    }
    return {
        "id": case["id"],
        "scoreable": record["scoreable"],
        "reference_status": record["reference_status"],
        "eligible_legacy_count": eligible_count,
        "historical_order_reproduced": True,
        "policies": policies,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cv2.setNumThreads(1)
    packed = read_archive_json(args.archive, ARCHIVE_INPUTS)
    saved = read_archive_json(args.archive, ARCHIVE_RESULTS)
    cases_by_id = {case["id"]: case for case in packed["cases"]}
    records_by_id = {record["id"]: record for record in saved["records"]}
    if set(cases_by_id) != set(records_by_id):
        raise ValueError("Archived broadcast inputs and results contain different case IDs")
    requested_ids = list(records_by_id) if args.ids is None else args.ids
    unknown_ids = sorted(set(requested_ids) - set(records_by_id))
    if unknown_ids:
        raise ValueError(f"Unknown case IDs: {unknown_ids}")

    legacy = load_legacy(args.legacy)
    cases = []
    for case_id in requested_ids:
        print(case_id, flush=True)
        cases.append(process_case(cases_by_id[case_id], records_by_id[case_id], args.image_root, legacy))

    output = {
        "schema": "broadcast-junction-box-repair/1",
        "development_data": True,
        "acceptance_evaluated": False,
        "policies": {
            "no_mask": "no spatial person mask",
            "median_two_of_three": "mask points covered by boxes in at least two of three median frames",
        },
        "provenance": {
            "archive": args.archive.name,
            "archive_sha256": sha256_file(args.archive),
        },
        "summary": summarise(cases),
        "cases": cases,
    }
    atomic_write_json_gz(args.output, output)
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
