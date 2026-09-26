"""Measure centre-line junction evidence without changing frozen candidate rankings."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from zipfile import BadZipFile, ZipFile

import cv2
import numpy as np

from scratch.court_det_fix.court_detector import junctions
from scratch.court_det_fix.court_detector.geometry import CORNER_COURT_M
from scratch.court_det_fix.court_detector.image_sources import (
    PACK_MD5_BY_NAME,
    SIDECAR_MD5,
    SIDECAR_SCHEMA,
    CaseProvenance,
    load_frozen_case_provenance,
    require_same_image_boxes,
)
from scratch.court_det_fix.court_detector.line_observations import prepare_observations

from .run_assignment import read_replay

REPLAY_PACK_MEMBER = "marking_inputs.json.gz"


def require_replay_pack(replay_path: Path, provenance_pack_path: Path) -> None:
    """Require the replay's embedded inputs to be the validated pinned pack bytes."""
    try:
        with ZipFile(replay_path) as archive:
            replay_pack = archive.read(REPLAY_PACK_MEMBER)
    except (FileNotFoundError, BadZipFile, KeyError) as error:
        raise ValueError(
            f"Could not read {REPLAY_PACK_MEMBER!r} from replay {replay_path}"
        ) from error
    if replay_pack != provenance_pack_path.read_bytes():
        raise ValueError(
            "Replay input bytes do not match --provenance-pack; the replay must contain "
            "the validated pinned pack exactly"
        )


def provenance_binding(pack_path: Path) -> dict[str, str]:
    """Describe the exact pinned pack and sidecar contract used by the diagnostic."""
    return {
        "pack_filename": pack_path.name,
        "pack_md5": PACK_MD5_BY_NAME[pack_path.name],
        "sidecar_schema": SIDECAR_SCHEMA,
        "sidecar_md5": SIDECAR_MD5,
    }


def validate_stripe_provenance(stripe_results: dict, expected_binding: dict[str, str]) -> None:
    """Require stripe results produced from the same validated replay input pack."""
    if stripe_results.get("schema") != "frozen-stripe-observations/2":
        raise ValueError("Stripe results are old or unsupported; rerun run_stripes.py with --provenance-pack")
    if stripe_results.get("input_provenance") != expected_binding:
        raise ValueError("Stripe result provenance does not match the validated replay input pack")


def bytes_md5(value: bytes) -> str:
    """Return the MD5 used to bind the exact artefact bytes consumed."""
    return hashlib.md5(value, usedforsecurity=False).hexdigest()


def run_case(case: dict, frozen: dict, provenance: CaseProvenance) -> dict:
    if provenance.case_id != case["id"]:
        raise ValueError(f"provenance case {provenance.case_id!r} does not match {case['id']!r}")
    require_same_image_boxes(provenance)
    started = perf_counter()
    width, height = case["dimensions"]["width"], case["dimensions"]["height"]
    size = tuple(frozen["working_size"])
    scale = np.asarray([width / size[0], height / size[1]])
    segments = np.asarray(case["segments_px"], dtype=float).reshape(-1, 4) / np.tile(scale, 2)
    observations = prepare_observations(segments, size)
    boxes = np.asarray(case["bbox_px"], dtype=float).reshape(-1, 4) / np.tile(scale, 2)
    entries = []
    for source in frozen["entries"]:
        if not source["eligible"]:
            continue
        homography = cv2.getPerspectiveTransform(
            CORNER_COURT_M, (np.asarray(source["corners_px"]) / scale).astype(np.float32),
        )
        entries.append({"id": source["id"], **junctions.measure(homography, observations, boxes, size)})
    return {"id": case["id"], "entries": entries, "elapsed_seconds": perf_counter() - started}


def preflight_cases(
    inputs: dict, saved: dict, provenance: Mapping[str, CaseProvenance],
) -> list[tuple[dict, dict, CaseProvenance]]:
    """Validate all replay, stripe and typed provenance IDs before measuring any case."""
    cases = {case["id"]: case for case in inputs["cases"]}
    replay_ids = tuple(cases)
    if len(replay_ids) != len(inputs["cases"]):
        raise ValueError("Replay contains duplicate case IDs")
    stripe_ids = tuple(source["id"] for source in saved["records"])
    if len(stripe_ids) != len(set(stripe_ids)):
        raise ValueError("Stripe results contain duplicate case IDs")
    if set(replay_ids) != set(stripe_ids):
        raise ValueError(
            f"Replay/stripe case-set mismatch (replay={sorted(replay_ids)!r}, stripes={sorted(stripe_ids)!r})"
        )
    if set(provenance) != set(stripe_ids):
        raise ValueError(
            f"Replay/provenance case-set mismatch (replay={sorted(replay_ids)!r}, "
            f"provenance={sorted(provenance)!r})"
        )
    selected = []
    for source in saved["records"]:
        case_id = source["id"]
        selected.append((cases[case_id], source, require_same_image_boxes(provenance[case_id])))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--stripes", type=Path, required=True)
    parser.add_argument("--provenance-pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    provenance = load_frozen_case_provenance(args.provenance_pack)
    require_replay_pack(args.replay, args.provenance_pack)
    inputs, _ = read_replay(args.replay)
    stripe_bytes = args.stripes.read_bytes()
    saved = json.loads(gzip.decompress(stripe_bytes))
    binding = provenance_binding(args.provenance_pack)
    validate_stripe_provenance(saved, binding)
    selected = preflight_cases(inputs, saved, provenance)
    records = []
    for case, source, case_provenance in selected:
        record = run_case(case, source, case_provenance)
        records.append(record)
        print(f"{record['id']}: {len(record['entries'])} eligible geometries, "
              f"{record['elapsed_seconds']:.2f}s", flush=True)
    output = {
        "schema": "frozen-junction-diagnostic/2",
        "diagnostic_only": True,
        "provenance": {**binding, "stripe_artefact_md5": bytes_md5(stripe_bytes)},
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(gzip.compress(json.dumps(output, allow_nan=False).encode(), mtime=0))


if __name__ == "__main__":
    main()
