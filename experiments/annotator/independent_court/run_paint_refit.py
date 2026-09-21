"""Compare paint conventions on the same frozen parents and line observations.

This isolates measurement/refinement geometry. Generation, player/net evidence
and the legacy floor gate remain the recorded controls. It does not evaluate a
complete detector using physical paint geometry throughout.
"""

from __future__ import annotations

import argparse
import gzip
import importlib
import io
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from types import ModuleType
from zipfile import ZipFile

import cv2
import numpy as np

from . import fixed_stripe_refit as fitting
from . import stripe_observations as stripes
from .assignment import Observations, prepare_observations
from .case_provenance import (
    CaseProvenance,
    load_frozen_case_provenance,
    require_same_image_boxes,
)
from .detector import CORNER_COURT_M, SEGMENTS_M
from .paint_geometry import CENTRE_SEGMENTS_M
from .run_assignment import attach_metrics, frozen_entries, read_replay_bytes
from .run_junctions import bytes_md5, provenance_binding, require_replay_pack
from .run_refit_selection import eligible

MODELS = {"legacy": SEGMENTS_M, "physical": CENTRE_SEGMENTS_M}
# Clipping and inverse/project round trips can move endpoints about 1e-12 pixels.
PROJECTION_ROUNDOFF_PX = 1e-7
ANNOTATION_ARTEFACTS = (Path("hand_corners.csv"), Path("2026-09-08/hand_corners.csv.gz"))
OUTPUT_SCHEMA = "paired-paint-refit/2"
PAINT_REFIT_PACK_FILENAME = "marking_refit_inputs.json.gz"
_MD5_RE = re.compile(r"[0-9a-f]{32}")


def _require_md5(value: object, label: str) -> str:
    """Require one canonical lowercase MD5 digest string."""
    if not isinstance(value, str) or _MD5_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a 32-character lowercase MD5")
    return value


def validate_result_provenance(
    results: dict, *, replay_bytes: bytes | None = None, control_bytes: bytes | None = None,
) -> None:
    """Require a current, pinned paint-refit result before consuming it."""
    if results.get("schema") != OUTPUT_SCHEMA:
        raise ValueError("Paint-refit results are old or unsupported; rerun run_paint_refit.py")
    provenance = results.get("provenance")
    if not isinstance(provenance, dict):
        raise TypeError("Paint-refit results must contain an input provenance object")
    pack_filename = provenance.get("pack_filename")
    if not isinstance(pack_filename, str):
        raise TypeError("Paint-refit provenance pack_filename must be a string")
    if pack_filename != PAINT_REFIT_PACK_FILENAME:
        raise ValueError("Paint-refit results must come from the amateur replay pack")
    try:
        expected_binding = provenance_binding(Path(pack_filename))
    except KeyError as error:
        raise ValueError("Paint-refit provenance names an unsupported frozen pack") from error
    expected_keys = {
        *expected_binding,
        "replay_artefact_md5",
        "control_artefact_md5",
        "annotation_artefacts_md5",
    }
    if set(provenance) != expected_keys:
        raise ValueError("Paint-refit provenance does not match the pinned pack/sidecar contract")
    for name in ("pack_md5", "sidecar_md5", "replay_artefact_md5", "control_artefact_md5"):
        _require_md5(provenance.get(name), f"Paint-refit provenance {name}")
    if any(provenance.get(name) != value for name, value in expected_binding.items()):
        raise ValueError("Paint-refit provenance does not match the pinned pack/sidecar contract")
    annotation_hashes = provenance.get("annotation_artefacts_md5")
    expected_annotations = {str(path) for path in ANNOTATION_ARTEFACTS}
    if not isinstance(annotation_hashes, dict) or set(annotation_hashes) != expected_annotations:
        raise ValueError("Paint-refit provenance annotation artefact binding is malformed")
    for name, value in annotation_hashes.items():
        _require_md5(value, f"Paint-refit provenance annotation hash {name!r}")
    if replay_bytes is not None and provenance["replay_artefact_md5"] != bytes_md5(replay_bytes):
        raise ValueError("Paint-refit results were produced from different replay bytes")
    if control_bytes is not None and provenance["control_artefact_md5"] != bytes_md5(control_bytes):
        raise ValueError("Paint-refit results were produced from different control bytes")


def score(
    corners: np.ndarray, observations: Observations, weights: np.ndarray, size: tuple[int, int],
    scale: np.ndarray, centres: np.ndarray, assignment: dict | None = None,
) -> dict:
    homography = cv2.getPerspectiveTransform(CORNER_COURT_M, (corners / scale).astype(np.float32))
    measured = stripes.measure(homography, observations, size, centres, PROJECTION_ROUNDOFF_PX)
    return stripes.score_model(measured, weights, 3, assignment)


def run_case(
    case: dict, parents: list[dict], legacy: ModuleType, provenance: CaseProvenance,
) -> dict:
    """Keep all starts; fix fragment identities separately under each paint model."""
    if provenance.case_id != case["id"]:
        raise ValueError(f"provenance case {provenance.case_id!r} does not match {case['id']!r}")
    require_same_image_boxes(provenance)
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


def preflight_cases(cases: list[dict], provenance: Mapping[str, CaseProvenance]) -> None:
    """Validate every selected case before importing or writing archived replay work."""
    case_ids = [case["id"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Replay contains duplicate selected case IDs")
    for case_id in case_ids:
        try:
            case_provenance = provenance[case_id]
        except KeyError as error:
            raise ValueError(f"Selected case {case_id!r} is absent from the validated provenance pack") from error
        require_same_image_boxes(case_provenance)


def output_provenance(
    binding: dict[str, str],
    replay_bytes: bytes,
    control_bytes: bytes,
    annotation_artefacts: dict[str, str],
) -> dict:
    """Bind paint-refit output to its replay, control and annotation inputs."""
    return {
        **binding,
        "replay_artefact_md5": bytes_md5(replay_bytes),
        "control_artefact_md5": bytes_md5(control_bytes),
        "annotation_artefacts_md5": annotation_artefacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorded", type=Path, required=True)
    parser.add_argument("--provenance-pack", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", nargs="*")
    args = parser.parse_args()
    replay = args.recorded / "marking_refit_replay.zip"
    replay_bytes = replay.read_bytes()
    provenance = load_frozen_case_provenance(args.provenance_pack)
    require_replay_pack(replay, args.provenance_pack)
    packed, saved = read_replay_bytes(replay_bytes)
    parents = {record["id"]: frozen_entries(record) for record in saved["records"]}
    cases = [case for case in packed["cases"] if not args.ids or case["id"] in args.ids]
    if args.ids and {case["id"] for case in cases} != set(args.ids):
        raise ValueError("Requested case IDs must exist in the replay")
    preflight_cases(cases, provenance)

    args.output.mkdir(parents=True, exist_ok=True)
    legacy_dir = args.output / "legacy"
    legacy_dir.mkdir(exist_ok=True)
    with ZipFile(io.BytesIO(replay_bytes)) as archive:
        for name in archive.namelist():
            if Path(name).name == name and name.endswith(".py"):
                (legacy_dir / name).write_bytes(archive.read(name))
    sys.path.insert(0, str(legacy_dir.resolve()))
    legacy = importlib.import_module("run_alignment")
    cv2.setNumThreads(1)
    records = []
    for case in cases:
        record = run_case(case, parents[case["id"]], legacy, provenance[case["id"]])
        records.append(record)
        checkpoint = args.output / f"{case['id']}.json.gz"
        checkpoint.write_bytes(gzip.compress(json.dumps(record, allow_nan=False).encode(), mtime=0))
        print(f"{case['id']}: {record['unique_fits']} unique fits, {record['elapsed_seconds']:.2f}s", flush=True)
    with ZipFile(args.recorded / "stripe_diagnostics.zip") as archive:
        control_bytes = archive.read("refit_selection_v2.json.gz")
    control = json.loads(gzip.decompress(control_bytes))
    verify_control(records, control)
    attach_metrics(records, packed["references"])
    from .boundary_metrics import load_corner_metadata, measure

    annotation_artefacts = {
        str(relative_path): bytes_md5((args.annotations / relative_path).read_bytes())
        for relative_path in ANNOTATION_ARTEFACTS
    }
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
    result = {"schema": OUTPUT_SCHEMA, "development_data": True,
              "acceptance_evaluated": False, "generation_and_gates": "unchanged legacy geometry",
              "provenance": output_provenance(
                  provenance_binding(args.provenance_pack), replay_bytes, control_bytes, annotation_artefacts,
              ),
              "control_coordinate_tolerance_px": 0.01, "projection_roundoff_px": PROJECTION_ROUNDOFF_PX,
              "control_scores": "remeasured with roundoff allowance in both arms; archived deltas retained",
              "models": {name: centres.tolist() for name, centres in MODELS.items()},
              "frames": len(records), "picks": picks, "records": records}
    (args.output / "results.json.gz").write_bytes(gzip.compress(json.dumps(result, allow_nan=False).encode(), mtime=0))
    print(f"Wrote {len(records)} paired cases; archived control passed.", flush=True)


if __name__ == "__main__":
    main()
