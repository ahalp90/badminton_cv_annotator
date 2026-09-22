"""Extract saved fit scores and signed disagreement with the available references.

References are diagnostic inputs only. ShuttleSet grids are shared across each
video and are not scene-level ground truth; GX includes extrapolated corners.
"""

from __future__ import annotations

import argparse
import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

BASELINE = Path(
    "evidence/holistic_admission/directional_20260921_r5/"
    "w5_directional_20260921_r5_43"
)
SCORES = ("q_geom", "q_paint10", "q_geom_span_weighted", "q_paint10_span_weighted")
CORNER_NAMES = ("far_left", "far_right", "near_right", "near_left")
EDGE_NAMES = ("far", "right", "near", "left")
MAX_WORKING_DIMENSION = 960  # Frozen W5 measurement setting.


def read(path: Path) -> dict:
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def reference_disagreement(corners: list, reference: dict, scale: np.ndarray) -> dict:
    predicted = np.asarray(corners, dtype=float)
    expected = np.asarray(reference["corners_px"], dtype=float)
    direct = np.linalg.norm(predicted - expected, axis=1)
    rotated = np.linalg.norm(predicted[[2, 3, 0, 1]] - expected, axis=1)
    relabelled = bool(rotated.max() < direct.max())
    if relabelled:
        predicted = predicted[[2, 3, 0, 1]]
    delta = predicted - expected
    predicted_working, expected_working = predicted * scale, expected * scale
    centre = expected_working.mean(axis=0)
    edge_distances = {}
    for index, name in enumerate(EDGE_NAMES):
        next_index = (index + 1) % 4
        start, end = expected_working[[index, next_index]]
        tangent = end - start
        normal = np.array([-tangent[1], tangent[0]])
        normal /= np.linalg.norm(normal)
        if (centre - (start + end) / 2) @ normal < 0:
            normal = -normal
        distances = (predicted_working[[index, next_index]] - start) @ normal
        edge_distances[name] = {
            "endpoint_signed_distances": distances.tolist(),
            "mean_signed_distance": float(distances.mean()),
        }
    return {
        "relabelled_180": relabelled,
        "corner_order": CORNER_NAMES,
        "signed_corner_xy_native": delta.tolist(),
        "signed_corner_xy_working": (delta * scale).tolist(),
        "corner_distance_native": np.linalg.norm(delta, axis=1).tolist(),
        "corner_distance_working": np.linalg.norm(delta * scale, axis=1).tolist(),
        "edge_disagreement_working": edge_distances,
    }


def extract_case(job: dict) -> dict:
    row, comparison = job["manifest"], job["comparison"]
    required = {
        selection[gate]
        for selection in comparison["selections"].values()
        for gate in ("gated", "ungated")
        if selection[gate] is not None
    }
    if row["previous_w5_case"]:
        candidates = job["baseline_candidates"]
        optimisation = None
        native_size = np.asarray(job["native_size"], dtype=float)
        working_size = np.rint(native_size * min(1.0, MAX_WORKING_DIMENSION / max(native_size)))
    else:
        result = read(Path(job["result"]))
        candidates = result["review_candidates"]
        optimisation = result.get("measurement_optimisation")
        native_size = np.asarray(result["provenance"]["native_dimensions"], dtype=float)
        working_size = np.asarray(result["provenance"]["working_dimensions"], dtype=float)
        if job["native_size"] is not None:
            np.testing.assert_array_equal(native_size, job["native_size"])
    if required - candidates.keys():
        record = read(Path(comparison["record"]))
        for candidate in record["parents"] + record["valid_children"]:
            if candidate["origin_key"] in required:
                candidates[candidate["origin_key"]] = candidate
    scale = working_size / native_size
    extracted = {}
    for key in sorted(required):
        candidate = candidates[key]
        evidence = candidate.get("evidence", candidate)
        selected = {
            "origin_key": key,
            "corners_px": candidate["corners_px"],
            "source_memberships": candidate["source_memberships"],
            "scores": {name: evidence.get(name) for name in SCORES},
            "directional": evidence["directional"],
            "gates": candidate["gates"],
            "refit": candidate.get("refit"),
        }
        reference = job["reference"]
        if reference and reference.get("corners_px"):
            selected["reference_disagreement"] = reference_disagreement(
                candidate["corners_px"], reference, scale,
            )
        extracted[key] = selected
    reference_metadata = {
        key: value for key, value in job["reference"].items()
        if key not in ("landmarks", "landmark_names")
    }
    return {
        "case_id": row["case_id"],
        "group": row["group"],
        "arm": row["arm"],
        "video": row.get("video"),
        "previous_w5_case": row["previous_w5_case"],
        "view_status": row.get("view_status"),
        "control_label": row.get("reference_status"),
        "native_size": native_size.astype(int).tolist(),
        "working_size": working_size.astype(int).tolist(),
        "reference": reference_metadata,
        "population_counts": comparison["population_counts"],
        "selections": comparison["selections"],
        "candidates": extracted,
        "measurement_optimisation": optimisation,
        "restricted_rank_diagnostic": comparison.get("restricted_rank_diagnostic"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "wider_evaluation/runs/20260922"
    manifest = read(base / "manifest.json.gz")
    comparisons = {}
    for filename in ("regression_comparison.json.gz", "partial_comparison.json.gz"):
        comparisons.update({row["case_id"]: row for row in read(base / filename)["cases"]})
    baseline = json.loads((root / BASELINE / "review_candidates.json").read_text())
    pack_paths = {
        row["source_pack"]
        for row in manifest["cases"] if row["arm"] == "frozen_detector"
    }
    packs = {path: read(root / path) for path in pack_paths}
    jobs = []
    for row in manifest["cases"]:
        case_id = row["case_id"]
        if case_id not in comparisons:
            continue
        if row["arm"] == "frozen_detector":
            pack = packs[row["source_pack"]]
            source = next(case for case in pack["cases"] if case["id"] == case_id)
            dimensions = source["dimensions"]
            native_size = [dimensions["width"], dimensions["height"]]
            reference = pack["references"].get(case_id, {})
        else:
            native_size = None
            reference = {}
        required = {
            selection[gate]
            for selection in comparisons[case_id]["selections"].values()
            for gate in ("gated", "ungated")
        }
        compact = baseline.get(case_id, {}).get("candidates", {})
        jobs.append({
            "manifest": row, "comparison": comparisons[case_id],
            "reference": reference, "native_size": native_size,
            "baseline_candidates": {key: value for key, value in compact.items() if key in required},
            "result": str(base / "measured/results" / f"{case_id}.json.gz"),
        })
    with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as executor:
        rows = list(executor.map(extract_case, jobs))
    result = {
        "schema": "wider-numeric-fit/1", "workers": args.workers,
        "planned_case_count": len(manifest["cases"]), "complete": len(rows) == len(manifest["cases"]),
        "conventions": {
            "signed_xy": "image x right-positive, image y down-positive",
            "working_pixels": "native image aspect ratio retained; longest dimension capped at 960",
            "edge_sign": "positive towards reference court interior; mean of endpoint distances to reference edge",
            "corner_alignment": "direct or 180-degree relabelling, whichever minimises maximum corner distance",
            "broadcast_reference": "one static grid per video; valid only for standard view, not scene ground truth",
            "other_references": "GX records clicked/extrapolated corners; amateur corner provenance is unspecified in pack",
            "scores": "fit/support scores, not independent accuracy or confidence probabilities",
        },
        "cases": rows,
    }
    with gzip.open(base / "numeric_fit.json.gz", "wt") as stream:
        json.dump(result, stream, allow_nan=False, indent=2)
    print(json.dumps({"cases": len(rows), "complete": result["complete"]}))


if __name__ == "__main__":
    main()
