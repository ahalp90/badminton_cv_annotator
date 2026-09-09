"""Compare paint conventions on the same frozen parents and line observations.

This isolates measurement/refinement geometry. Generation, player/net evidence
and the legacy floor gate remain the recorded controls. It does not evaluate a
complete detector using physical paint geometry throughout.
"""

from __future__ import annotations

import argparse
import gzip
import importlib
import json
import sys
from pathlib import Path
from time import perf_counter
from types import ModuleType
from zipfile import ZipFile

import cv2
import numpy as np

from . import fixed_stripe_refit as fitting
from . import stripe_observations as stripes
from .assignment import Observations, prepare_observations
from .detector import CORNER_COURT_M, SEGMENTS_M
from .paint_geometry import CENTRE_SEGMENTS_M
from .run_assignment import attach_metrics, frozen_entries, read_replay
from .run_refit_selection import eligible

MODELS = {"legacy": SEGMENTS_M, "physical": CENTRE_SEGMENTS_M}
# Clipping and inverse/project round trips can move endpoints about 1e-12 pixels.
PROJECTION_ROUNDOFF_PX = 1e-7


def score(
    corners: np.ndarray, observations: Observations, weights: np.ndarray, size: tuple[int, int],
    scale: np.ndarray, centres: np.ndarray, assignment: dict | None = None,
) -> dict:
    homography = cv2.getPerspectiveTransform(CORNER_COURT_M, (corners / scale).astype(np.float32))
    measured = stripes.measure(homography, observations, size, centres, PROJECTION_ROUNDOFF_PX)
    return stripes.score_model(measured, weights, 3, assignment)


def run_case(case: dict, parents: list[dict], legacy: ModuleType) -> dict:
    """Keep all starts; fix fragment identities separately under each paint model."""
    started = perf_counter()
    prepared = legacy.prepare_case(case)
    size, scale = prepared["size"], prepared["native_scale"]
    observations = prepare_observations(prepared["segments"], size)
    weights = stripes.fragment_weights(observations)
    entries = []
    fit_cache = {}
    for parent in parents:
        if not parent["eligible"]:
            continue
        corners = np.asarray(parent["corners_px"])
        construction = case["candidates"][parent["source_index"]]
        starting = legacy.evidence(corners, case, prepared, construction)
        saved = parent["evidence"]
        if eligible(starting) != parent["eligible"] or starting["net_score"] != saved["net_score"]:
            raise ValueError(f"Starting gate/net evidence changed: {case['id']}/{parent['id']}")
        homography = cv2.getPerspectiveTransform(CORNER_COURT_M, (corners / scale).astype(np.float32))
        initial = fitting.initial_parameters(corners / scale, size)
        for model, centres in MODELS.items():
            start_score = score(corners, observations, weights, size, scale, centres)
            assigned = start_score["assignments"]
            constraints = fitting.prepare(homography, observations, assigned, weights, centres)
            key = (model, initial.tobytes(), constraints.fragment_ids.tobytes(), constraints.sample_ids.tobytes(),
                   constraints.intervals.tobytes(), constraints.positions.tobytes(), constraints.weights.tobytes())
            if key not in fit_cache:
                fitted = fitting.refine(corners / scale, constraints, size, True, initial, centres)
                if fitted["corners_px"] is not None:
                    fitted["corners_px"] = (np.asarray(fitted["corners_px"]) * scale).tolist()
                fit_cache[key] = fitted
            fitted = fit_cache[key]
            variants = [("start", corners, starting, start_score)]
            if fitted["successful"]:
                changed = np.asarray(fitted["corners_px"])
                renewed = legacy.evidence(changed, case, prepared, construction)
                changed_score = score(changed, observations, weights, size, scale, centres, assigned)
                variants.append(("refit", changed, renewed, changed_score))
            for stage, coordinates, evidence, stripe in variants:
                entry = {"id": f"{parent['id']}/{model}/{stage}", "parent_id": parent["id"],
                         "model": model, "stage": stage, "corners_px": coordinates.tolist(),
                         "eligible": eligible(evidence), "gate_evidence": evidence, "stripe": stripe,
                         "score": None, "fit": fitted if stage == "refit" else None}
                if entry["eligible"]:
                    entry["score"] = (3 * stripe["exclusive"]["score"] + evidence["net_score"]) / 4
                entries.append(entry)
            # Preserve failed attempts and fitting identities alongside their starting court.
            entries[-len(variants)]["attempt"] = fitted
            entries[-len(variants)]["fit_samples"] = {
                name: getattr(constraints, name).tolist()
                for name in ("fragment_ids", "sample_ids", "intervals", "positions", "weights")
            }
    orders = {}
    for model in MODELS:
        active = [entry for entry in entries if entry["model"] == model and entry["eligible"]]
        orders[model] = [entry["id"] for entry in sorted(active, key=lambda entry: (-entry["score"], entry["id"]))]
    return {"id": case["id"], "dimensions": case["dimensions"], "working_size": size,
            "entries": entries, "orders": orders, "unique_fits": len(fit_cache),
            "elapsed_seconds": perf_counter() - started}


def verify_control(records: list[dict], frozen: dict) -> None:
    """Check archived fits/gates and report score changes from boundary roundoff."""
    old_records = {record["id"]: record for record in frozen["records"]}
    for record in records:
        old = {entry["id"]: entry for entry in old_records[record["id"]]["entries"]}
        expected = {
            (entry["parent_id"], "start" if entry["model"] == "start" else "refit")
            for entry in old.values() if entry["model"] in ("start", "fixed_position")
        }
        actual = {(entry["parent_id"], entry["stage"]) for entry in record["entries"] if entry["model"] == "legacy"}
        if actual != expected:
            raise ValueError(f"Control candidate population changed: {record['id']}")
        for entry in record["entries"]:
            if entry["model"] != "legacy":
                continue
            suffix = "start" if entry["stage"] == "start" else "fixed_position"
            previous = old[f"{entry['parent_id']}/{suffix}"]
            if entry["eligible"] != previous["eligible"]:
                raise ValueError(f"Control eligibility changed: {record['id']}/{entry['id']}")
            # Native OpenCV/SciPy versions can change final solver rounding.
            delta = float(np.max(np.abs(np.asarray(entry["corners_px"]) - previous["corners_px"])))
            entry["control_coordinate_max_abs_delta_px"] = delta
            entry["control_score_delta"] = (
                abs(entry["score"] - previous["stripe_score"]) if entry["eligible"] else None
            )
            if delta > 0.01:
                raise ValueError(f"Control numerical drift: {record['id']}/{entry['id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorded", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", nargs="*")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    legacy_dir = args.output / "legacy"
    legacy_dir.mkdir(exist_ok=True)
    replay = args.recorded / "marking_refit_replay.zip"
    with ZipFile(replay) as archive:
        for name in archive.namelist():
            if Path(name).name == name and name.endswith(".py"):
                (legacy_dir / name).write_bytes(archive.read(name))
    sys.path.insert(0, str(legacy_dir.resolve()))
    legacy = importlib.import_module("run_alignment")
    cv2.setNumThreads(1)
    packed, saved = read_replay(replay)
    parents = {record["id"]: frozen_entries(record) for record in saved["records"]}
    cases = [case for case in packed["cases"] if not args.ids or case["id"] in args.ids]
    if args.ids and {case["id"] for case in cases} != set(args.ids):
        raise ValueError("Requested case IDs must exist in the replay")
    records = []
    for case in cases:
        record = run_case(case, parents[case["id"]], legacy)
        records.append(record)
        checkpoint = args.output / f"{case['id']}.json.gz"
        checkpoint.write_bytes(gzip.compress(json.dumps(record, allow_nan=False).encode(), mtime=0))
        print(f"{case['id']}: {record['unique_fits']} unique fits, {record['elapsed_seconds']:.2f}s", flush=True)
    with ZipFile(args.recorded / "stripe_diagnostics.zip") as archive:
        control = json.loads(gzip.decompress(archive.read("refit_selection_v2.json.gz")))
    verify_control(records, control)
    attach_metrics(records, packed["references"])
    from .boundary_metrics import load_corner_metadata, measure

    metadata = load_corner_metadata(args.annotations)
    picks = []
    for record in records:
        identifier = record["id"]
        annotation = metadata[identifier]
        reference = packed["references"][identifier]
        if not np.allclose(annotation["corners_px"], reference["corners_px"], rtol=0, atol=1e-6):
            raise ValueError(f"Annotation coordinates differ from frozen reference: {identifier}")
        size = (record["dimensions"]["width"], record["dimensions"]["height"])
        entries = {entry["id"]: entry for entry in record["entries"]}
        for entry in entries.values():
            entry["boundary_metrics"] = measure(np.asarray(entry["corners_px"]), reference, size,
                                                 annotation["indices"], annotation["click_inset_m"])
        row = {"id": identifier, "click_convention": annotation["convention"],
               "click_inset_m": annotation["click_inset_m"], "picks": {}}
        for model, order in record["orders"].items():
            winner = entries[order[0]] if order else None
            row["picks"][model] = None if winner is None else {
                key: winner[key] for key in ("id", "score", "corners_px", "metrics", "boundary_metrics")
            }
        picks.append(row)
    result = {"schema": "paired-paint-refit/1", "development_data": True,
              "acceptance_evaluated": False, "generation_and_gates": "unchanged legacy geometry",
              "control_coordinate_tolerance_px": 0.01, "projection_roundoff_px": PROJECTION_ROUNDOFF_PX,
              "control_scores": "remeasured with roundoff allowance in both arms; archived deltas retained",
              "models": {name: centres.tolist() for name, centres in MODELS.items()},
              "frames": len(records), "picks": picks, "records": records}
    (args.output / "results.json.gz").write_bytes(gzip.compress(json.dumps(result, allow_nan=False).encode(), mtime=0))
    print(f"Wrote {len(records)} paired cases; archived control passed.", flush=True)


if __name__ == "__main__":
    main()
