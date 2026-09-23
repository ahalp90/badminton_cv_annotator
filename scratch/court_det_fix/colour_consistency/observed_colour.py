"""Sample observed stripe fragments at fixed candidate geometry in native pixels."""

from __future__ import annotations

import cv2
import numpy as np

from experiments.annotator.independent_court import (
    assignment,
    detector,
    paint_geometry,
    stripe_observations,
)
from experiments.annotator.independent_court import fixed_stripe_refit as fitting
from scratch.court_det_fix.w5_holistic import verifier

FRACTIONS = np.linspace(0.1, 0.9, 24)
SHIFTS_WORKING = np.asarray((-4, -2, 0, 2, 4), dtype=float)
SIDE_OFFSET_WORKING = 6.0
MIN_RIDGE_LAB = 10.0
MIN_VALID_SAMPLES = 8
OUTER_MARKINGS = frozenset(("far_baseline", "near_baseline", "left_doubles", "right_doubles"))


def native_image(context: verifier.ViewContext, root) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = verifier.frame_path(root, context.source, context.provenance)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    if image.shape[1::-1] != context.native_size:
        raise ValueError(f"{context.case_id}: native image dimensions changed")
    scale = np.asarray(context.native_size, dtype=float) / np.asarray(context.size, dtype=float)
    boxes = context.mask_boxes * np.tile(scale, 2)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return image, lab, grey, boxes


def sample_image(values: np.ndarray, points: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = points.reshape(-1, 2)
    valid = np.isfinite(flat).all(axis=1)
    valid &= verifier.observable_points(flat, (values.shape[1], values.shape[0]), boxes)
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


def observed_fragments(context: verifier.ViewContext, corners_native: list, lab: np.ndarray,
                       grey: np.ndarray, boxes: np.ndarray) -> dict:
    scale = np.asarray(context.native_size, dtype=float) / np.asarray(context.size, dtype=float)
    corners_working = np.asarray(corners_native, dtype=float) / scale
    homography = cv2.getPerspectiveTransform(detector.CORNER_COURT_M.astype(np.float32),
                                             corners_working.astype(np.float32))
    evidence = stripe_observations.measure(
        homography, context.observations, context.size, centres=paint_geometry.CENTRE_SEGMENTS_M,
    )
    score = stripe_observations.score_model(evidence, context.weights, 3)
    assigned = score["assignments"]
    constraints = fitting.prepare(homography, context.observations, assigned, context.weights,
                                  centres=paint_geometry.CENTRE_SEGMENTS_M)
    lookup = {int(fragment_id): index for index, fragment_id in enumerate(context.observations.fragment_ids)}
    rows = []
    for fragment_id in sorted(set(constraints.fragment_ids.tolist())):
        indices = np.flatnonzero(constraints.fragment_ids == fragment_id)
        observation = lookup[fragment_id]
        marking_index = int(assigned["marking"][observation])
        if marking_index < 0 or marking_index >= len(assignment.MARKINGS):
            raise ValueError(f"{context.case_id}: invalid retained marking {marking_index}")
        intervals = sorted({int(value) for value in constraints.intervals[indices]})
        if not set(intervals).issubset(assignment.MARKING_INTERVALS[marking_index]):
            raise ValueError(f"{context.case_id}: retained interval and marking disagree")
        projected, depth = detector.project(homography[None], paint_geometry.CENTRE_SEGMENTS_M[intervals])
        if not np.isfinite(projected).all() or not (depth > 0).all():
            raise ValueError(f"{context.case_id}: retained interval has invalid projection")
        # The fixed candidate supplies membership; observed endpoints supply the colour profile.
        sampled = sample_fragment(context.observations.segments[observation].copy(), scale, lab, grey, boxes)
        valid = [row for row in sampled["samples"] if row["valid"]]
        sampled.update({"raw_fragment_id": fragment_id, "marking": assignment.MARKINGS[marking_index],
                        "intervals": intervals, "sample_ids": constraints.sample_ids[indices].tolist(),
                        "segment_working_px": context.observations.segments[observation].tolist()})
        if sampled["usable"]:
            sampled["median_raw_ab"] = np.median([row["centre_lab"][1:] for row in valid], axis=0).tolist()
            sampled["median_side_ab"] = np.median([row["mean_side_lab"][1:] for row in valid], axis=0).tolist()
        rows.append(sampled)
    return {"stripe_assignments": assigned, "retained_fragment_ids": sorted(set(constraints.fragment_ids.tolist())),
            "fragments": rows}


def marking_signatures(fragments: list[dict]) -> dict:
    groups: dict[str, list[dict]] = {}
    for fragment in fragments:
        if fragment["usable"]:
            groups.setdefault(fragment["marking"], []).append(fragment)
    result = {}
    for name, members in groups.items():
        raw = np.median([row["median_raw_ab"] for row in members], axis=0)
        side = np.median([row["median_side_ab"] for row in members], axis=0)
        result[name] = {"fragment_ids": [row["raw_fragment_id"] for row in members],
                        "raw_ab": raw.tolist(), "side_ab": side.tolist(),
                        "floor_relative_ab": (raw - side).tolist()}
    return result
