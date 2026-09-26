"""Saved W5 observed-paint contradiction trial without manual colour labels."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from scratch.court_det_fix.colour_consistency import observed_colour as observed
from scratch.court_det_fix.colour_consistency.run import (
    CHROMATIC_CHANNEL_TOLERANCE,
    CHROMATIC_PIXEL_FRACTION,
)
from scratch.court_det_fix.w5_holistic import verifier

COURT = Path(__file__).resolve().parents[1]
WIDER = COURT / "wider_evaluation/runs/20260922"
OUTPUT = Path(__file__).with_suffix(".json.gz")
CASES = (
    "am1_window_00_frame_54", "gxBQ_window_00_frame_0", "gxBQ_window_00_frame_5",
    "am2_window_00_frame_150", "am3_window_00_frame_0", "am4_window_00_frame_0",
    "centre_short_frame_36", "letterboxed_short_frame_45", "yellow_short_frame_14",
    "shuttleset_03_scene_0019", "shuttleset_03_scene_0029", "shuttleset_03_scene_0034",
)
FLOORS = (15.0, 20.0, 25.0)


def read(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def write(path: Path, value: dict) -> None:
    path.write_bytes(gzip.compress(json.dumps(value, allow_nan=False, separators=(",", ":")).encode(), mtime=0))


def compare(signatures: dict, fragments: list[dict], floor: float, chromatic: bool) -> dict:
    targets = {}
    for target in sorted(observed.OUTER_MARKINGS):
        row = signatures.get(target)
        others = {name: value for name, value in signatures.items() if name != target}
        if row is None:
            targets[target] = {"decision": "no_decision", "reason": "target_absent",
                               "ambiguous_reference": False}
            continue
        if not chromatic:
            targets[target] = {"decision": "no_decision", "reason": "greyscale", "target": row,
                               "ambiguous_reference": False}
            continue
        names = sorted(others)
        if len(names) < 2:
            targets[target] = {"decision": "no_decision", "reason": "insufficient_distinct_markings", "target": row,
                               "reference_markings": names, "reference_count": len(names),
                               "ambiguous_reference": False}
            continue
        colours = np.asarray([others[name]["raw_ab"] for name in names])
        distances = np.linalg.norm(colours[:, None] - colours[None], axis=2)
        medoid = int(np.argmin(distances.sum(axis=1)))
        inliers = np.flatnonzero(distances[medoid] <= 20.0)
        coherent = len(inliers) >= 2 and len(inliers) > len(names) / 2
        references = [names[index] for index in inliers]
        if not coherent:
            targets[target] = {"decision": "no_decision", "reason": "ambiguous_reference", "target": row,
                               "reference_markings": names, "coherent_markings": references,
                               "reference_count": len(references), "ambiguous_reference": True}
            continue
        centre = np.median(colours[inliers], axis=0)
        reference_ids = {fragment_id for name in references for fragment_id in others[name]["fragment_ids"]}
        reference_fragments = [fragment for fragment in fragments if fragment["raw_fragment_id"] in reference_ids]
        scale = float(np.percentile([np.linalg.norm(np.asarray(fragment["median_raw_ab"]) - centre)
                                     for fragment in reference_fragments], 90))
        distance = float(np.linalg.norm(np.asarray(row["raw_ab"]) - centre))
        threshold = max(floor, 3 * scale)
        targets[target] = {"decision": "abstain" if distance > threshold else "keep",
                           "reason": "observed_paint_contradiction" if distance > threshold else "no_colour_contradiction",
                           "target": row, "reference_markings": references,
                           "other_supported_markings": names, "reference_count": len(references),
                           "ambiguous_reference": False,
                           "reference_raw_ab": centre.tolist(), "reference_scale_ab": scale,
                           "distance_ab": distance, "threshold_ab": threshold,
                           "reference_floor_relative_ab": np.median(
                               [others[name]["floor_relative_ab"] for name in references], axis=0).tolist()}
    decisions = [row["decision"] for row in targets.values()]
    after = "abstain" if "abstain" in decisions else "keep" if "keep" in decisions else "no_decision"
    return {"after": after, "targets": targets}


def run() -> dict:
    manifest = {row["case_id"]: row for row in read(WIDER / "manifest.json.gz")["cases"]}
    numeric = {row["case_id"]: row for row in read(WIDER / "numeric_fit.json.gz")["cases"]}
    if len(manifest) != len(read(WIDER / "manifest.json.gz")["cases"]):
        raise ValueError("duplicate manifest case ID")
    if len(numeric) != len(read(WIDER / "numeric_fit.json.gz")["cases"]):
        raise ValueError("duplicate numeric case ID")
    cases = []
    for case_id in CASES:
        item, record = manifest[case_id], numeric[case_id]
        context = verifier.prepare_view(COURT, case_id)
        if item["same_image_boxes"] != context.same_image_mask_available:
            raise ValueError(f"{case_id}: person box provenance differs")
        if list(context.native_size) != record["native_size"] or list(context.size) != record["working_size"]:
            raise ValueError(f"{case_id}: image size differs from saved record")
        image, lab, grey, boxes = observed.native_image(context, COURT)
        channel_range = image.max(axis=2).astype(np.int16) - image.min(axis=2).astype(np.int16)
        chromatic_fraction = float(np.mean(channel_range > CHROMATIC_CHANNEL_TOLERANCE))
        chromatic = chromatic_fraction >= CHROMATIC_PIXEL_FRACTION
        key = record["selections"]["full"]["gated"]
        candidate = record["candidates"].get(key) if key is not None else None
        base = {"case_id": case_id, "selected_key": key, "image": item["image"],
                "before": "keep" if candidate is not None else "no_saved_candidate",
                "native_size": record["native_size"], "working_size": record["working_size"],
                "box_mask": {"same_image_available": context.same_image_mask_available, "count": len(boxes),
                             "unavailable_reason": context.person_mask_unavailable_reason},
                "chromatic_fraction": chromatic_fraction, "chromatic": chromatic}
        if candidate is None:
            cases.append({**base, "status": "no_saved_candidate", "corners_native_px": None,
                          "primary": {"after": "no_decision"}})
            continue
        corners = candidate["corners_px"]
        evidence = observed.observed_fragments(context, corners, lab, grey, boxes)
        signatures = observed.marking_signatures(evidence["fragments"])
        cases.append({**base, "status": "measured", "corners_native_px": corners,
                      "observed": evidence, "markings": signatures,
                      "primary": compare(signatures, evidence["fragments"], 20, chromatic),
                      "sensitivity": {str(int(floor)): compare(signatures, evidence["fragments"], floor, chromatic)
                                      for floor in FLOORS}})
        print(case_id, key, cases[-1]["primary"]["after"])
    return {"schema": "observed-fragment-colour-trial/1", "cases": cases,
            "settings": {"raw_ab_floor": 20, "sensitivity_floors": FLOORS,
                         "minimum_distinct_reference_markings": 2,
                         "sampling": "native observed fragment ridge or valley"}}


if __name__ == "__main__":
    cv2.setNumThreads(1)
    write(OUTPUT, run())
