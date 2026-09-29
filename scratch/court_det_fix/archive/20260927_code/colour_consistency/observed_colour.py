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
from scratch.court_det_fix.court_detector.stripe_refit import (
    colour_planes,
    sample_fragment,
)
from scratch.court_det_fix.w5_holistic import verifier

OUTER_MARKINGS = frozenset(("far_baseline", "near_baseline", "left_doubles", "right_doubles"))


def native_image(context: verifier.ViewContext, root) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = root / context.frame_relative_path
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    lab, grey, boxes = colour_planes(image, context)
    return image, lab, grey, boxes


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
