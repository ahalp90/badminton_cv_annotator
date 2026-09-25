"""Refit the chosen court to its painted stripes, after re-deciding which part of
each stripe every fit fragment sits on.

A line fragment on a painted stripe sits on its centre (position 0) or on one of its
two edges (1, 2). W5's refit took the position from geometry alone. Here each fit
fragment's position is re-decided from two image cues: the grey contrast across it
on the working frame (brightness_profiles) and the colour polarity sampled along it
on the native frame (sample_fragment, infer_polarity). automatic_position keeps the
old position unless the polarity is known, at least MIN_SIDE_PAIRS side pairs exist
and the contrast reaches CONTRAST_LEVELS grey levels. Only the positions change;
then the court is refitted (fixed_stripe_refit.refine) and validated (fit_geometry).

Research scripts import these back, so this module stays a leaf: it imports numpy,
OpenCV, the experiments package and, for observable_points, the package-name copy
of w5_holistic/verifier.py. It never edits sys.path.
"""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from experiments.annotator.independent_court import fixed_stripe_refit as fitting
from scratch.court_det_fix.w5_holistic.verifier import observable_points

FRACTIONS = np.linspace(0.1, 0.9, 24)
SHIFTS_WORKING = np.asarray((-4, -2, 0, 2, 4), dtype=float)
SIDE_OFFSET_WORKING = 6.0
MIN_RIDGE_LAB = 10.0
MIN_VALID_SAMPLES = 8
MIN_POLARITY_SAMPLES = 8
POLARITY_FRACTION = 0.8
MIN_SIDE_PAIRS = 7
CONTRAST_LEVELS = 10.0
PROFILE_DISTANCES = np.array([1.0, 2.0])
REPLAY_ATOL_NATIVE_PX = 1e-4


def colour_planes(image: np.ndarray, context: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lab and grey planes of the native frame, and the photometry mask boxes in native pixels."""
    if image.shape[1::-1] != context.native_size:
        raise ValueError(f"{context.case_id}: native image dimensions changed")
    scale = np.asarray(context.native_size, dtype=float) / np.asarray(context.size, dtype=float)
    boxes = context.mask_boxes * np.tile(scale, 2)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return lab, grey, boxes


def sample_image(values: np.ndarray, points: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = points.reshape(-1, 2)
    valid = np.isfinite(flat).all(axis=1)
    valid &= observable_points(flat, (values.shape[1], values.shape[0]), boxes)
    valid &= ((flat >= 0) & (flat <= np.asarray([values.shape[1] - 1, values.shape[0] - 1]))).all(axis=1)
    output = np.full((len(flat),) + values.shape[2:], np.nan, dtype=float)
    if valid.any():
        xy = flat[valid].astype(np.float32)
        sampled = cv2.remap(values, xy[:, 0], xy[:, 1], cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        output[valid] = sampled.reshape(output[valid].shape)
    return output.reshape(points.shape[:-1] + values.shape[2:]), valid.reshape(points.shape[:-1])


def sample_fragment(segment: np.ndarray, scale: np.ndarray, lab: np.ndarray, grey: np.ndarray,
                    boxes: np.ndarray) -> dict:
    direction = segment[1] - segment[0]
    direction /= np.linalg.norm(direction)
    normal = np.array((-direction[1], direction[0]))
    bases = segment[0] + FRACTIONS[:, None] * (segment[1] - segment[0])
    centres_working = bases[:, None, :] + SHIFTS_WORKING[None, :, None] * normal
    points_working = centres_working[:, :, None, :] + np.asarray((-SIDE_OFFSET_WORKING, 0, SIDE_OFFSET_WORKING))[
        None, None, :, None] * normal
    points = points_working * scale
    colours, valid = sample_image(lab, points, boxes)
    greys, grey_valid = sample_image(grey, points, boxes)
    valid &= grey_valid
    negative = colours[:, :, 1] - colours[:, :, 0]
    positive = colours[:, :, 1] - colours[:, :, 2]
    coherent = np.einsum("ijk,ijk->ij", negative, positive) > 0
    score = np.minimum(np.linalg.norm(negative, axis=2), np.linalg.norm(positive, axis=2))
    score[~(valid.all(axis=2) & coherent)] = -np.inf
    chosen = np.argmax(score, axis=1)
    rows = []
    for index, shift_index in enumerate(chosen):
        accepted = bool(score[index, shift_index] >= MIN_RIDGE_LAB)
        if not accepted:
            rows.append({"valid": False})
            continue
        trio = colours[index, shift_index]
        grey_trio = greys[index, shift_index]
        grey_deltas = grey_trio[1] - grey_trio[[0, 2]]
        polarity = int(np.sign(grey_deltas[0])) if grey_deltas[0] * grey_deltas[1] > 0 else 0
        floor = trio[[0, 2]].mean(axis=0)
        rows.append({"valid": True, "shift_working_px": float(SHIFTS_WORKING[shift_index]),
                     "score_lab": float(score[index, shift_index]),
                     "xy_native": points[index, shift_index].tolist(),
                     "side_minus_lab": trio[0].tolist(), "centre_lab": trio[1].tolist(),
                     "side_plus_lab": trio[2].tolist(), "mean_side_lab": floor.tolist(),
                     "delta_lab": (trio[1] - floor).tolist(), "grey_polarity": polarity})
    count = sum(row["valid"] for row in rows)
    return {"usable": count >= MIN_VALID_SAMPLES, "valid_count": count, "samples": rows}


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
    if polarity == 0 or pairs < MIN_SIDE_PAIRS or abs(contrast) < CONTRAST_LEVELS:
        return old
    adjusted = contrast * polarity
    if old == 0:
        return 1 if adjusted * expected_position_one > 0 else 2
    expected_original = expected_position_one * (1 if old == 1 else -1)
    return 3 - old if adjusted * expected_original < 0 else old


def brightness_profiles(context: Any, verifier: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measure raw-fragment polarity once, independently of court assignments."""
    directions = context.observations.directions
    normals = np.column_stack((-directions[:, 1], directions[:, 0]))
    samples = context.observations.samples[:, 1:-1]
    offsets = normals[:, None, None, :] * PROFILE_DISTANCES[None, None, :, None]
    positive, negative = samples[:, :, None, :] + offsets, samples[:, :, None, :] - offsets
    positive_valid = verifier.observable_points(positive, context.size, context.mask_boxes)
    negative_valid = verifier.observable_points(negative, context.size, context.mask_boxes)
    valid = positive_valid & negative_valid
    differences = (
        verifier.grayscale_sample(context.frame, positive) - verifier.grayscale_sample(context.frame, negative)
    )
    medians = np.zeros((len(samples), len(PROFILE_DISTANCES)))
    counts = valid.sum(axis=1)
    for fragment in range(len(samples)):
        for distance in range(len(PROFILE_DISTANCES)):
            if counts[fragment, distance]:
                medians[fragment, distance] = np.median(differences[fragment, valid[fragment, :, distance], distance])
    return normals, medians, counts


def expected_bright_side(
    homography: np.ndarray, midpoints: np.ndarray, normals: np.ndarray,
    markings: np.ndarray, positions: np.ndarray, detector: Any, stripe_width: float,
) -> np.ndarray:
    """Express the expected paint side in each raw fragment's normal convention."""
    world, _ = detector.project(np.linalg.inv(homography)[None], midpoints)
    world = world[0]
    shifted = world.copy()
    axis = np.where(markings < 5, 0, 1)
    shifted[np.arange(len(world)), axis] += stripe_width
    projected, _ = detector.project(homography[None], shifted)
    positive_axis = projected[0] - midpoints
    image_sign = np.sign(np.einsum("ij,ij->i", positive_axis, normals))
    return image_sign * np.where(positions == 1, 1, -1)


def relabel(
    context: Any, parent: dict, constraints: Any, verifier: Any,
) -> tuple[Any, list[dict], dict]:
    started = perf_counter()
    normals, contrasts, counts = brightness_profiles(context, verifier)
    assignments = parent["evidence"]["stripe_assignments"]
    positions = np.asarray(assignments["position"], dtype=int)
    markings = np.asarray(assignments["marking"], dtype=int)
    expected = expected_bright_side(
        np.asarray(parent["homography_working"]), context.observations.segments.mean(axis=1),
        normals, markings, positions, verifier.detector, verifier.paint_geometry.STRIPE_WIDTH_M,
    )
    minimum_samples = (context.observations.samples.shape[1] - 2 + 1) // 2
    edge = (positions == 1) | (positions == 2)
    sufficient = (counts[:, 0] >= minimum_samples) & (np.abs(contrasts[:, 0]) >= CONTRAST_LEVELS)
    contradicted = edge & sufficient & (contrasts[:, 0] * expected < 0)
    revised = np.where(contradicted, 3 - positions, positions)
    lookup = {int(raw_id): index for index, raw_id in enumerate(context.observations.fragment_ids)}
    new_positions = np.array([revised[lookup[int(raw_id)]] for raw_id in constraints.fragment_ids])
    changed = replace(constraints, positions=new_positions)
    retained = set(constraints.fragment_ids.tolist())
    rows = []
    for index, raw_id in enumerate(context.observations.fragment_ids):
        if int(raw_id) not in retained:
            continue
        rows.append({
            "raw_fragment_id": int(raw_id), "marking": int(markings[index]),
            "old_position": int(positions[index]), "new_position": int(revised[index]),
            "expected_bright_side": float(expected[index]) if edge[index] else None,
            "signed_contrast_by_distance": contrasts[index].tolist(),
            "valid_pairs_by_distance": counts[index].tolist(),
            "strong_polarity": bool(sufficient[index]), "changed": bool(contradicted[index]),
        })
    metadata = {"seconds": perf_counter() - started, "all_fragment_count": len(positions),
                "retained_fragment_count": len(retained),
                "changed_point_count": int(np.sum(new_positions != constraints.positions)),
                "changed_fragment_count": sum(row["changed"] for row in rows)}
    return changed, rows, metadata


def describe(corners_working: np.ndarray, context: Any, verifier: Any, runtime: dict, maps: np.ndarray) -> dict:
    detector = verifier.detector
    scale = np.asarray(context.native_size) / context.size
    corners_native = corners_working * scale
    homography = cv2.getPerspectiveTransform(detector.CORNER_COURT_M, corners_working.astype(np.float32))
    evidence, _ = verifier.measure_candidate(context, {"homography_working": homography}, {})
    gates = runtime["gate_evidence"](
        corners_native, context.source, scale, context.size, context.families, maps, runtime["zone"],
    )
    return {"corners_native_px": corners_native.tolist(), "corners_working_px": corners_working.tolist(),
            "paint_score": evidence["q_paint10_span_weighted"], "geometry_score": evidence["q_geom_span_weighted"],
            "camera_eligible": verifier.camera_eligible({"gates": gates}),
            "historical": verifier.historical_predicates(gates), "gates": gates}


def selected_parent(record: dict, origin_key: str) -> tuple[dict, dict, dict | None]:
    candidates = record["parents"] + record["valid_children"]
    by_key = {item["origin_key"]: item for item in candidates}
    if len(by_key) != len(candidates):
        raise ValueError("duplicate origin keys in source record")
    selected = by_key[origin_key]
    parent_key = selected.get("parent_origin_key") or origin_key
    parent = by_key[parent_key]
    if parent["kind"] == "child" or parent["origin_key"] != parent_key:
        raise ValueError(f"{origin_key}: invalid parent resolution")
    attempts = [row for row in record.get("fit_attempts", []) if row["origin_key"] == parent_key]
    if len(attempts) > 1:
        raise ValueError(f"{origin_key}: multiple saved fit attempts")
    attempt = attempts[0] if attempts else None
    if selected["kind"] == "child":
        if selected["parent_origin_key"] != parent_key or selected["origin_key"] != f"{parent_key}/child":
            raise ValueError(f"{origin_key}: child parent link differs")
        if attempt is not None:
            if attempt["child_origin_key"] != origin_key:
                raise ValueError(f"{origin_key}: saved attempt names another child")
            np.testing.assert_allclose(attempt["attempted_corners_native"], selected["corners_px"],
                                       rtol=0, atol=REPLAY_ATOL_NATIVE_PX)
    return selected, parent, attempt


def fit_geometry(fit: dict, context, verifier: object, runtime: dict, maps: np.ndarray) -> dict:
    result = {"fit": verifier.jsonable(fit), "status": fit.get("status"), "valid": False,
              "validity_reason": None, "measurement": None, "homography_working": None}
    corners = fit.get("corners_px")
    if corners is None:
        result["validity_reason"] = "no_fit_corners"
        return result
    working = np.asarray(corners, dtype=float)
    scale = np.asarray(context.native_size, dtype=float) / np.asarray(context.size, dtype=float)
    result["corners_working_px"] = working.tolist()
    result["corners_native_px"] = (working * scale).tolist()
    if not fit.get("successful", False):
        result["validity_reason"] = fit.get("status", "solver_failed")
        return result
    if fit.get("jacobian_rank") != 8 or fit.get("minimum_corner_denominator", 0) <= 1e-6:
        result["validity_reason"] = "fit_rank_or_depth_invalid"
        return result
    if not np.isfinite(working).all() or not verifier.convex_corners(working):
        result["validity_reason"] = "non_finite_or_non_convex_corners"
        return result
    homography = cv2.getPerspectiveTransform(verifier.detector.CORNER_COURT_M.astype(np.float32),
                                             working.astype(np.float32)).astype(float)
    result["homography_working"] = homography.tolist()
    measurement = describe(working, context, verifier, runtime, maps)
    result["measurement"] = measurement
    valid, reason = verifier.hard_validity({"homography_working": homography, "gates": measurement["gates"]})
    result["valid"] = valid
    result["validity_reason"] = reason
    return result



def refit_chosen(record: dict, origin_key: str, context: Any, native_frame: np.ndarray, verifier: Any,
                 runtime: dict, line_maps: np.ndarray, *, replay_check: bool = True) -> dict:
    """Refit the chosen court with automatic stripe positions.

    :param record: W5 case record as read back from its JSON file: parents,
        valid_children and fit_attempts.
    :param origin_key: The chosen court.
    :param native_frame: The view's native BGR frame, for colour sampling.
    :param line_maps: The view's two wide-family distance maps (run_w5.view_line_maps).
    :param replay_check: First replay the parent's saved W5 fit and require it to
        match within REPLAY_ATOL_NATIVE_PX native pixels.
    :return: The chosen and parent geometry, the replay (with replay_check), the
        per-fragment position changes, and "corrected": fit_geometry's result for
        the refitted court.
    """
    started = perf_counter()
    selected, parent, attempt = selected_parent(record, origin_key)
    homography = np.asarray(parent["homography_working"], dtype=float)
    constraints = fitting.prepare(homography, context.observations, parent["evidence"]["stripe_assignments"],
                                  context.weights, centres=verifier.paint_geometry.CENTRE_SEGMENTS_M)
    starting = verifier.detector.project(homography[None], verifier.detector.CORNER_COURT_M)[0][0]
    scale = np.asarray(context.native_size, dtype=float) / np.asarray(context.size, dtype=float)
    result = {"selected_origin_key": origin_key,
              "selected_geometry": {"corners_native_px": selected["corners_px"],
                                    "homography_working": selected["homography_working"],
                                    "gates": selected["gates"]},
              "parent_origin_key": parent["origin_key"], "parent_homography_working": parent["homography_working"]}
    if replay_check:
        original = fitting.refine(starting, constraints, context.size, use_positions=True,
                                  centres=verifier.paint_geometry.CENTRE_SEGMENTS_M)
        reference = attempt["attempted_corners_native"] if attempt is not None else selected["corners_px"]
        reference_name = "saved_fit_attempt" if attempt is not None else "saved_child_corners"
        if attempt is None and selected["kind"] != "child":
            raise ValueError(f"{origin_key}: parent has no saved attempt or child replay reference")
        if original.get("corners_px") is None:
            raise ValueError(f"{origin_key}: original fit returned no corners")
        replay_native = np.asarray(original["corners_px"], dtype=float) * scale
        difference = replay_native - np.asarray(reference, dtype=float)
        np.testing.assert_allclose(replay_native, reference, rtol=0, atol=REPLAY_ATOL_NATIVE_PX)
        result["replay_reference"] = reference_name
        result["original_replay"] = {"status": "matched",
                                     "maximum_absolute_difference_native_px": float(np.abs(difference).max()),
                                     "corners_native_px": replay_native.tolist(), "fit_status": original.get("status")}
    replay_seconds = perf_counter() - started

    lab, grey, boxes = colour_planes(native_frame, context)
    _, old_rows, _ = relabel(context, parent, constraints, verifier)
    lookup = {int(raw_id): index for index, raw_id in enumerate(context.observations.fragment_ids)}
    positions = constraints.positions.copy()
    rows = []
    for old in old_rows:
        fragment_id = old["raw_fragment_id"]
        index = lookup[fragment_id]
        segment = context.observations.segments[index]
        sampled = sample_fragment(segment.copy(), scale, lab, grey, boxes)
        inferred = infer_polarity(sampled)
        direction = context.observations.directions[index]
        normal = np.array([[-direction[1], direction[0]]])
        expected = expected_bright_side(
            homography, segment.mean(axis=0)[None], normal, np.array([old["marking"]]),
            np.array([1]), verifier.detector, verifier.paint_geometry.STRIPE_WIDTH_M,
        )[0]
        if expected == 0:
            raise ValueError(f"{context.case_id}: zero expected side for fragment {fragment_id}")
        new = automatic_position(
            old["old_position"], old["signed_contrast_by_distance"][0],
            old["valid_pairs_by_distance"][0], expected, inferred["polarity"],
        )
        positions[constraints.fragment_ids == fragment_id] = new
        rows.append({"raw_fragment_id": fragment_id, "old_position": old["old_position"],
                     "automatic_position": new, "inferred_polarity": inferred["polarity"],
                     "usable_samples": inferred["usable_samples"]})
    revised = replace(constraints, positions=positions)
    for field in ("points", "intervals", "weights", "fragment_ids", "sample_ids"):
        if getattr(revised, field) is not getattr(constraints, field):
            raise AssertionError(f"{field} array was replaced")
        np.testing.assert_array_equal(getattr(revised, field), getattr(constraints, field))
    corrected = fitting.refine(starting, revised, context.size, use_positions=True,
                               centres=verifier.paint_geometry.CENTRE_SEGMENTS_M)
    geometry = fit_geometry(corrected, context, verifier, runtime, line_maps)
    result["automatic"] = {
        "changed_fragment_count": sum(row["automatic_position"] != row["old_position"] for row in rows),
        "unresolved_fragment_count": sum(row["inferred_polarity"] == 0 for row in rows),
        "fragment_count": len(rows), "changed_point_count": int(np.sum(positions != constraints.positions)),
        "fragments": rows, "unchanged_constraint_arrays": True,
    }
    result["corrected"] = geometry
    result["timings_seconds"] = {"original_replay": replay_seconds,
                                 "automatic_and_corrected": perf_counter() - started - replay_seconds}
    return result
