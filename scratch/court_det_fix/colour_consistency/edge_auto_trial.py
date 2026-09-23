"""Test automatic stripe polarity on frozen centre-to-edge refits."""

from __future__ import annotations

import gzip
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from experiments.annotator.independent_court import fixed_stripe_refit as fitting
from scratch.court_det_fix.colour_consistency import observed_colour as observed
from scratch.court_det_fix.edge_polarity import run_probe as probe
from scratch.court_det_fix.edge_polarity.local_audit import diagnose, run_amateur

OUTPUT = Path(__file__).with_suffix(".json.gz")
AUDIT = probe.ROOT / "edge_polarity/local_audit"
MIN_POLARITY_SAMPLES = 8
POLARITY_FRACTION = 0.8
MIN_SIDE_PAIRS = 7


def infer_polarity(sampled: dict) -> dict:
    signs = [row["grey_polarity"] for row in sampled["samples"] if row["valid"]]
    positive = signs.count(1)
    negative = signs.count(-1)
    usable = len(signs)
    winning = max(positive, negative)
    polarity = 0
    if usable >= MIN_POLARITY_SAMPLES and winning / usable >= POLARITY_FRACTION:
        polarity = 1 if positive > negative else -1
    return {"polarity": polarity, "usable_samples": usable, "positive_samples": positive,
            "negative_samples": negative, "confidence": winning / usable if usable else 0.0}


def automatic_position(old: int, contrast: float, pairs: int, expected_position_one: float,
                       polarity: int) -> int:
    if polarity == 0 or pairs < MIN_SIDE_PAIRS or abs(contrast) < probe.CONTRAST_LEVELS:
        return old
    adjusted = contrast * polarity
    if old == 0:
        return 1 if adjusted * expected_position_one > 0 else 2
    expected_original = expected_position_one * (1 if old == 1 else -1)
    return 3 - old if adjusted * expected_original < 0 else old


def saved_case(case_id: str) -> tuple[dict, dict | None]:
    if case_id in diagnose.CASES:
        original = next(row for row in probe.read(probe.ROOT / "edge_polarity/results.json.gz")["cases"]
                        if row["case_id"] == case_id)
        diagnostic = next(row for row in probe.read(AUDIT / "diagnostics.json.gz")["cases"]
                          if row["case_id"] == case_id)
        return original, diagnostic
    original = next(row for row in probe.read(AUDIT / "amateur_results.json.gz")["cases"]
                    if row["case_id"] == case_id)
    return original, None


def run_case(case_id: str, verifier: object, runtime: dict) -> dict:
    context, parent, starting_corners, constraints = diagnose.load_case(case_id, verifier)
    saved, diagnostic = saved_case(case_id)
    if saved["parent_key"] != parent["origin_key"]:
        raise ValueError(f"{case_id}: saved parent changed")
    _, lab, grey, boxes = observed.native_image(context, probe.ROOT)
    scale = np.asarray(context.native_size, dtype=float) / np.asarray(context.size, dtype=float)
    _, old_rows, _ = probe.relabel(context, parent, constraints, verifier)
    old_by_id = {row["raw_fragment_id"]: row for row in old_rows}
    positions = constraints.positions.copy()
    lookup = {int(fragment_id): index for index, fragment_id in enumerate(context.observations.fragment_ids)}
    homography = np.asarray(parent["homography_working"])
    rows = []
    for raw_id in sorted(old_by_id):
        old = old_by_id[raw_id]
        index = lookup[raw_id]
        segment = context.observations.segments[index]
        sampled = observed.sample_fragment(segment.copy(), scale, lab, grey, boxes)
        inference = infer_polarity(sampled)
        direction = context.observations.directions[index]
        normal = np.array([[-direction[1], direction[0]]])
        expected = probe.expected_bright_side(
            homography, segment.mean(axis=0)[None], normal, np.array([old["marking"]]),
            np.array([1]), verifier.detector, verifier.paint_geometry.STRIPE_WIDTH_M,
        )[0]
        if expected == 0:
            raise ValueError(f"{case_id}: zero expected side for fragment {raw_id}")
        new = automatic_position(old["old_position"], old["signed_contrast_by_distance"][0],
                                 old["valid_pairs_by_distance"][0], expected, inference["polarity"])
        positions[constraints.fragment_ids == raw_id] = new
        rows.append({**old, **inference, "expected_position_one_side": float(expected),
                     "automatic_position": new, "automatic_changed": new != old["old_position"]})
    revised = replace(constraints, positions=positions)
    for field in ("points", "intervals", "weights", "fragment_ids", "sample_ids"):
        if getattr(revised, field) is not getattr(constraints, field):
            raise AssertionError(f"{case_id}: {field} was replaced")
    arm_positions = {"original": constraints.positions.copy()}
    bright = constraints.positions.copy()
    for row in rows:
        bright[constraints.fragment_ids == row["raw_fragment_id"]] = row["new_position"]
    arm_positions["bright_edges"] = bright
    strong = bright.copy()
    choices = diagnostic["centre_choices"] if diagnostic else saved["centre_choices"]
    for raw_id, position in choices.items():
        strong[constraints.fragment_ids == int(raw_id)] = position
    arm_positions["bright_centres"] = strong
    matched = next((name for name, labels in arm_positions.items() if np.array_equal(labels, positions)), None)
    saved_fit_name = {"original": "original" if diagnostic is None else "baseline",
                      "bright_edges": "polarity", "bright_centres": "strong_centres"}
    if matched:
        source = diagnostic if diagnostic and matched == "bright_centres" else saved
        fit = source["fits"][saved_fit_name[matched]]
    else:
        fit = fitting.refine(starting_corners, revised, context.size, True,
                             centres=verifier.paint_geometry.CENTRE_SEGMENTS_M)
        fit["measurement"] = None
        if fit["successful"]:
            maps = verifier.detector._distance_maps(verifier.detector._wide_line_families(context.segments),
                                                    context.size)
            fit["measurement"] = probe.describe(np.asarray(fit["corners_px"]), context, verifier, runtime, maps)
    measurement = fit.get("measurement") if fit["successful"] else None
    return {"case_id": case_id, "parent_key": parent["origin_key"], "fragments": rows,
            "fits_saved": {name: saved["fits"][key] for name, key in
                           (("original", saved_fit_name["original"]), ("bright_edges", "polarity"))},
            "bright_centres_fit": (diagnostic or saved)["fits"]["strong_centres"],
            "automatic_fit": fit, "reused_saved_arm": matched,
            "automatic_valid": bool(fit["successful"] and measurement is not None
                                    and measurement["camera_eligible"]),
            "frozen_arrays_preserved": True}


def run() -> dict:
    cv2.setNumThreads(1)
    _, verifier, runtime = probe.load_runtime(probe.ROOT)
    case_ids = list(diagnose.CASES)
    case_ids.extend(row["case_id"] for row in run_amateur.sample_cases())
    if len(case_ids) != 12 or len(set(case_ids)) != 12:
        raise ValueError("expected four diagnostic and eight distinct amateur cases")
    cases = []
    for case_id in case_ids:
        row = run_case(case_id, verifier, runtime)
        cases.append(row)
        print(case_id, row["reused_saved_arm"], row["automatic_fit"]["status"], row["automatic_valid"])
    return {"schema": "automatic-stripe-polarity-trial/1", "settings": {
        "minimum_usable_samples": MIN_POLARITY_SAMPLES, "minimum_agreement_fraction": POLARITY_FRACTION,
        "minimum_side_pairs": MIN_SIDE_PAIRS, "minimum_side_contrast": probe.CONTRAST_LEVELS}, "cases": cases}


if __name__ == "__main__":
    OUTPUT.write_bytes(gzip.compress(json.dumps(run(), allow_nan=False, separators=(",", ":")).encode(), mtime=0))
