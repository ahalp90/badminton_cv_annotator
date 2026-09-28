"""Build court candidates from crossing lines and extra starting points."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

from . import template_arrays
from .template_arrays import SAMPLES_PER_LINE, SUPPORT_DISTANCE

RECTANGLE_CAP = 4096
PROPOSAL_CAP = 256
TEMPLATE_COUNT = 150
# Rectangles per scoring batch, each with 150 templates. The CPU keeps about 1024
# hypotheses per batch; the L40 trial measured 128 rectangles on the GPU.
BATCH_RECTANGLES = {"cpu": max(1, 1024 // TEMPLATE_COUNT), "cuda": 128}
DIVERSITY_RADIUS = 12.0
CAMERA_LIMIT = 0.1
CAMERA_RECHECK_MARGIN = 1e-3
FULL_RECTANGLE_ORDER = 10**9
VISIBILITY_COLUMN_LABELS = {
    "lengthwise": "first six projected court-template pieces (x-family): sidelines plus split centre",
    "cross_court": "second six projected court-template pieces (y-family): baselines and service lines",
}


@dataclass(frozen=True)
class Generation:
    """A deterministic source population and its label-free accounting."""

    entries: tuple[dict, ...]
    metadata: dict


@dataclass(frozen=True)
class AdmissionSelection:
    """Selection results for one visibility floor and its floor-zero baseline."""

    selected: np.ndarray
    scanned: int
    visibility_admitted: np.ndarray
    floor_zero_selected: np.ndarray
    floor_zero_scanned: int
    newly_admitted: np.ndarray
    removed_from_floor_zero: np.ndarray


def geometry_and_support(
    homographies: np.ndarray,
    distance_map: np.ndarray,
    size: tuple[int, int],
    detector,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score two directions and count projected pieces retained after clipping, with NumPy.

    template_arrays.geometry_and_support holds the rules. The first six projected pieces
    are the lengthwise x-family. The second six are the cross-court y-family.

    :param homographies: (hypotheses, 3, 3) float32 court metres to working pixels.
    :return: valid corners, direction means, the valid mask and visible pieces per direction.
    """
    view = template_arrays.place_view(np, distance_map, size, detector)
    valid, _, valid_corners, means, visibility = template_arrays.geometry_and_support(np, homographies, view)
    return valid_corners, means, valid, visibility


def vector_camera_errors(homographies: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """The frozen camera diagnostic over its 200 focal lengths, with NumPy (see template_arrays.camera_errors)."""
    image_width, _ = size
    return template_arrays.camera_errors(np, homographies, template_arrays.camera_focals(image_width), size)


def recheck_camera_frontier(
    errors: np.ndarray,
    corners: np.ndarray,
    native_scale: np.ndarray,
    native_size: tuple[int, int],
    zone,
) -> tuple[np.ndarray, int, float]:
    """Replace vector camera errors near the hard-gate limit with the candidate gates' scalar values.

    :param errors: (valid,) float32 vector camera errors.
    :return: a corrected float64 copy of the errors, the number rechecked and the largest change.
        The copy is float64 so each rechecked value meets CAMERA_LIMIT exactly as the scalar
        gate computed it.
    """
    errors = errors.astype(np.float64)
    frontier = np.flatnonzero(np.abs(errors - CAMERA_LIMIT) <= CAMERA_RECHECK_MARGIN)
    differences = []
    for index in frontier:
        native_corners = (corners[index] * native_scale).astype(np.float32)
        scalar_error = float(zone.net_segments(native_corners, native_size)[1])
        differences.append(abs(float(errors[index]) - scalar_error))
        errors[index] = scalar_error
    return errors, len(frontier), max(differences, default=0.0)


def camera_errors_with_frontier_recheck(
    homographies: np.ndarray,
    corners: np.ndarray,
    working_size: tuple[int, int],
    native_scale: np.ndarray,
    native_size: tuple[int, int],
    zone,
) -> tuple[np.ndarray, int, float]:
    """Use the candidate gates' scalar camera values at the hard-gate frontier."""
    errors = vector_camera_errors(homographies, working_size)
    return recheck_camera_frontier(errors, corners, native_scale, native_size, zone)


def greedy_diverse(
    order: np.ndarray,
    corners: np.ndarray,
    cap: int = PROPOSAL_CAP,
    radius: float = DIVERSITY_RADIUS,
) -> tuple[np.ndarray, int]:
    """Retain an ordered prefix under the inherited corner-distance rule."""
    retained: list[int] = []
    scanned = 0
    for candidate_index in order:
        scanned += 1
        candidate_index = int(candidate_index)
        if retained:
            separation = np.linalg.norm(corners[candidate_index] - corners[retained], axis=2).max(axis=1)
            if np.any(separation <= radius):
                continue
        retained.append(candidate_index)
        if len(retained) == cap:
            break
    return np.asarray(retained, dtype=np.int64), scanned


def _validate_visibility_floor(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(value)


def _validate_visibility_floors(
    min_visible_lengthwise: int,
    min_visible_cross_court: int,
) -> tuple[int, int]:
    return (
        _validate_visibility_floor(min_visible_lengthwise, "min_visible_lengthwise"),
        _validate_visibility_floor(min_visible_cross_court, "min_visible_cross_court"),
    )


def visibility_eligible(
    visibility: np.ndarray,
    min_visible_lengthwise: int,
    min_visible_cross_court: int,
) -> np.ndarray:
    """Return hypotheses whose two projected court-template floors are met."""
    min_visible_lengthwise, min_visible_cross_court = _validate_visibility_floors(
        min_visible_lengthwise,
        min_visible_cross_court,
    )
    if visibility.ndim != 2 or visibility.shape[1] != 2:
        raise ValueError("visibility must have shape (hypotheses, 2)")
    return (
        (visibility[:, 0] >= min_visible_lengthwise)
        & (visibility[:, 1] >= min_visible_cross_court)
    )


def select_with_visibility_floor(
    scores: np.ndarray,
    camera_eligible: np.ndarray,
    rectangle_ids: np.ndarray,
    templates: np.ndarray,
    corners: np.ndarray,
    visibility: np.ndarray,
    min_visible_lengthwise: int,
    min_visible_cross_court: int,
    cap: int = PROPOSAL_CAP,
    radius: float = DIVERSITY_RADIUS,
) -> AdmissionSelection:
    """Admit by visibility before score ordering, then compare with the camera-only pool."""
    visibility_admitted = visibility_eligible(
        visibility,
        min_visible_lengthwise,
        min_visible_cross_court,
    )
    eligible_indices = np.flatnonzero(camera_eligible & visibility_admitted)
    score_order = np.lexsort(
        (templates[eligible_indices], rectangle_ids[eligible_indices], -scores[eligible_indices])
    )
    admitted_order = eligible_indices[score_order]
    camera_indices = np.flatnonzero(camera_eligible)
    camera_score_order = np.lexsort(
        (templates[camera_indices], rectangle_ids[camera_indices], -scores[camera_indices])
    )
    camera_order = camera_indices[camera_score_order]
    floor_zero_selected, floor_zero_scanned = greedy_diverse(
        camera_order,
        corners,
        cap=cap,
        radius=radius,
    )
    selected, scanned = greedy_diverse(admitted_order, corners, cap=cap, radius=radius)
    floor_zero_set = {int(index) for index in floor_zero_selected}
    selected_set = {int(index) for index in selected}
    newly_admitted = np.asarray(
        [index for index in selected if int(index) not in floor_zero_set],
        dtype=np.int64,
    )
    removed_from_floor_zero = np.asarray(
        [index for index in floor_zero_selected if int(index) not in selected_set],
        dtype=np.int64,
    )
    return AdmissionSelection(
        selected=selected,
        scanned=scanned,
        visibility_admitted=visibility_admitted,
        floor_zero_selected=floor_zero_selected,
        floor_zero_scanned=floor_zero_scanned,
        newly_admitted=newly_admitted,
        removed_from_floor_zero=removed_from_floor_zero,
    )


def _ranked_records(
    selected: np.ndarray,
    scores: np.ndarray,
    means: np.ndarray,
    corners: np.ndarray,
    visibility: np.ndarray,
    camera_errors: np.ndarray,
    rectangle_ids: np.ndarray,
    rectangle_orders: np.ndarray,
    templates: np.ndarray,
    native_scale: np.ndarray,
    detector,
    cap: int,
) -> list[dict]:
    records = []
    for index in selected[:cap]:
        index = int(index)
        rectangle_id = int(rectangle_ids[index])
        template_index = int(templates[index])
        proposal_id = f"rectangle_{rectangle_id}:template_{template_index}"
        homography_working = cv2.getPerspectiveTransform(
            detector.CORNER_COURT_M.astype(np.float32), corners[index],
        ).astype(float)
        records.append(
            {
                "candidate_id": proposal_id,
                "proposal_id": proposal_id,
                "rectangle_id": rectangle_id,
                "rectangle_order": int(rectangle_orders[index]),
                "template_index": template_index,
                "corners_px": (corners[index] * native_scale).round(7).tolist(),
                "homography_working": homography_working.tolist(),
                "gates": {
                    "geometry_valid": True,
                    "camera_error": float(camera_errors[index]),
                },
                "profile": {},
                "stripe": {},
                "line_template": {
                    "admission_score": float(scores[index]),
                    "direction_means": [float(value) for value in means[index]],
                    "visible_lengthwise_pieces": int(visibility[index, 0]),
                    "visible_cross_court_pieces": int(visibility[index, 1]),
                    "camera_error_before_candidate_gates": float(camera_errors[index]),
                    "geometry_valid_before_candidate_gates": True,
                },
            }
        )
    return records


def _empty_metadata(settings: dict, started: float, reason: str) -> dict:
    return {
        "name": "line_template",
        "status": "empty",
        "reason": reason,
        "settings": settings,
        "contamination_check": {
            "references_loaded": False,
            "prior_controls_loaded": False,
            "automatic_rule_uses": [
                "image dimensions",
                "cached working fragments",
                "frozen coverage VP ordering",
                "template geometry",
            ],
        },
        "generation": {
            "visibility_admission": {
                "min_visible_lengthwise": int(settings["min_visible_lengthwise"]),
                "min_visible_cross_court": int(settings["min_visible_cross_court"]),
                "visibility_columns": dict(VISIBILITY_COLUMN_LABELS),
                "hypotheses_before": 0,
                "hypotheses_after": 0,
                "hypotheses_rejected": 0,
                "floor_zero_scanned_for_proposal_cap": 0,
                "floor_zero_selected_count": 0,
                "floor_zero_proposal_ids": [],
                "scanned_for_proposal_cap": 0,
                "removed_from_floor_zero_count": 0,
                "refilled_proposal_count": 0,
                "newly_admitted_indices": [],
                "newly_admitted_proposal_ids": [],
                "removed_from_floor_zero_indices": [],
                "removed_from_floor_zero_proposal_ids": [],
            },
            "elapsed_seconds": perf_counter() - started,
        },
        "proposal_count": 0,
    }


def attach_candidate_gates(entries: list[dict], context, runtime: dict, detector, native_scale: np.ndarray) -> None:
    """Attach each entry's candidate gates (court_checks.gate_evidence) after label-free admission."""
    raw_families = detector._wide_line_families(context.segments)
    gate_maps = detector._distance_maps(raw_families, context.size)
    for entry in entries:
        entry["gates"] = runtime["gate_evidence"](
            np.asarray(entry["corners_px"], dtype=float),
            context.source,
            native_scale,
            context.size,
            context.families,
            gate_maps,
            runtime["zone"],
        )


def generate(
    context,
    runtime: dict,
    detector,
    *,
    min_visible_lengthwise: int = 0,
    min_visible_cross_court: int = 0,
    seed_points: np.ndarray | None = None,
    device: str = "cpu",
) -> Generation:
    """Generate the audited line/template source for one prepared view.

    :param seed_points: Extra homogeneous vanishing points, (points, 3), appended to the
        estimator's own before rectangle selection. The all_lines and painted_lines direction
        pairs never see them.
    :param device: "cpu" scores the hypotheses with NumPy; "cuda" scores them with CuPy.
        Rectangle setup, admission and the candidate gates run on the CPU either way.
    """
    min_visible_lengthwise, min_visible_cross_court = _validate_visibility_floors(
        min_visible_lengthwise,
        min_visible_cross_court,
    )
    array_module = template_arrays.array_module(device)
    started = perf_counter()
    print(f"[{context.case_id}] line-template: preparing rectangles", flush=True)
    settings = {
        "wide_families": True,
        "family_line_cap": 32,
        "min_supported_lines_preparation_only": 3,
        "vp_angle_deg": 1.5,
        "vp_direction_lines": 128,
        "vp_pencils": 16,
        "vp_overlap": 0.8,
        "vp_candidate_batch": 256,
        "vp_rectangle_order_bound": FULL_RECTANGLE_ORDER,
        "support_distance": SUPPORT_DISTANCE,
        "samples_per_line": SAMPLES_PER_LINE,
        "template_count": TEMPLATE_COUNT,
        "rectangle_order": "frozen coverage VP round-robin order",
        "global_rectangle_cap": RECTANGLE_CAP,
        "proposal_cap": PROPOSAL_CAP,
        "min_visible_lengthwise": min_visible_lengthwise,
        "min_visible_cross_court": min_visible_cross_court,
        "visibility_columns": dict(VISIBILITY_COLUMN_LABELS),
        "corner_diversity_radius": DIVERSITY_RADIUS,
        "camera_limit": CAMERA_LIMIT,
        "camera_gate": "frozen candidate-gate camera error <= 0.1 before admission ordering",
        "geometry_validity": "finite positive-depth convex visible-span >= 0.15; rectangle area >= 100",
        "admission_score": "minimum of the two union-map direction supports",
        "full_candidate_gates": "run_diagnosis.gate_evidence after admission, with raw wide-fragment maps",
    }
    if len(detector.TEMPLATE_TRANSFORMS) != TEMPLATE_COUNT:
        raise ValueError("frozen detector template count changed")

    from .directions import Settings as VPSettings
    from .directions import estimate, rectangle_population, select

    vp_settings = VPSettings(
        angle_deg=1.5,
        direction_lines=128,
        pencils=16,
        overlap=0.8,
        rectangles=FULL_RECTANGLE_ORDER,
        candidate_batch=256,
        pencil_selection="coverage",
    )
    points, estimator = estimate(context.segments, context.size, vp_settings)
    if seed_points is not None:
        points = np.concatenate((points, seed_points))
    detector_settings = detector.Settings(wide_families=True, min_supported_lines=3)
    _, selection = select(context.families, points, context.size, detector_settings, vp_settings)
    selected_ids = np.asarray(selection.get("selected_pair_product_ids", []), dtype=np.int64)
    selected_ids = selected_ids[:RECTANGLE_CAP]
    if not len(selected_ids):
        return Generation((), _empty_metadata(settings, started, "no_coverage_rectangles"))

    quads, _, all_ids = rectangle_population(context.families, context.size)
    quad_by_id = {int(rectangle_id): quad for rectangle_id, quad in zip(all_ids, quads, strict=True)}
    rectangle_ids = []
    rectangle_orders = []
    rectangles = []
    for rectangle_order, rectangle_id in enumerate(selected_ids):
        quad = quad_by_id[int(rectangle_id)].astype(np.float32)
        if cv2.contourArea(quad) < 100:
            continue
        rectangle_ids.append(int(rectangle_id))
        rectangle_orders.append(rectangle_order)
        rectangles.append(cv2.getPerspectiveTransform(detector.UNIT_CORNERS, quad))
    rectangles_array = np.asarray(rectangles, dtype=np.float32).reshape(-1, 3, 3)
    if not len(rectangles_array):
        return Generation((), _empty_metadata(settings, started, "no_area_valid_rectangles"))

    union_map = detector.distance_map(context.segments, context.size)
    native_scale = np.asarray(context.native_size, dtype=np.float64) / np.asarray(context.size, dtype=np.float64)
    view = template_arrays.place_view(array_module, union_map, context.size, detector)
    device_rectangles = array_module.asarray(rectangles_array)
    batch_rectangles = BATCH_RECTANGLES[device]
    batches = []
    last_progress = perf_counter()
    for offset in range(0, len(rectangles_array), batch_rectangles):
        if perf_counter() - last_progress >= 30:
            print(
                f"[{context.case_id}] line-template: rectangles {offset}/{len(rectangles_array)} "
                f"(+{perf_counter() - started:.0f}s)",
                flush=True,
            )
            last_progress = perf_counter()
        rectangle_batch = device_rectangles[offset:offset + batch_rectangles]
        batches.append(template_arrays.score_rectangles(array_module, rectangle_batch, view))
    scored = template_arrays.ScoredHypotheses(*(np.concatenate(parts) for parts in zip(*batches, strict=True)))
    if not len(scored.corners):
        return Generation((), _empty_metadata(settings, started, "no_geometry_valid_hypotheses"))

    corners, means, visibility = scored.corners, scored.means, scored.visibility
    camera_errors, scalar_recheck_count, vector_scalar_max_abs_diff = recheck_camera_frontier(
        scored.camera_errors,
        corners,
        native_scale,
        context.native_size,
        runtime["zone"],
    )
    # Hypothesis numbers run rectangle by rectangle, with the 150 templates varying fastest.
    valid_indices = np.flatnonzero(scored.valid)
    rectangle_indices = valid_indices // TEMPLATE_COUNT
    rectangle_ids = np.asarray(rectangle_ids, dtype=np.int64)[rectangle_indices]
    rectangle_orders = np.asarray(rectangle_orders, dtype=np.int64)[rectangle_indices]
    templates = (valid_indices % TEMPLATE_COUNT).astype(np.int16)
    scores = means.min(axis=1)
    camera_eligible = camera_errors <= CAMERA_LIMIT
    admission = select_with_visibility_floor(
        scores,
        camera_eligible,
        rectangle_ids,
        templates,
        corners,
        visibility,
        min_visible_lengthwise,
        min_visible_cross_court,
    )
    selected = admission.selected
    scanned = admission.scanned
    entries = _ranked_records(
        selected,
        scores,
        means,
        corners,
        visibility,
        camera_errors,
        rectangle_ids,
        rectangle_orders,
        templates,
        native_scale,
        detector,
        PROPOSAL_CAP,
    )
    attach_candidate_gates(entries, context, runtime, detector, native_scale)
    metadata = {
        "name": "line_template",
        "status": "generated",
        "settings": settings,
        "contamination_check": {
            "references_loaded": False,
            "prior_controls_loaded": False,
            "automatic_rule_uses": [
                "image dimensions",
                "cached working fragments",
                "frozen coverage VP ordering",
                "template geometry",
            ],
        },
        "ordering": {
            "all_union_rectangle_count": len(selection.get("union_pair_product_ids", [])),
            "full_selected_rectangle_count": len(selection.get("selected_pair_product_ids", [])),
            "global_cap_count": len(selected_ids),
            "area_retained_rectangle_count": len(rectangles_array),
            "selected_rectangle_ids_first_cap": selected_ids.tolist(),
        },
        "generation": {
            "rectangle_template_hypotheses": int(len(rectangles_array) * TEMPLATE_COUNT),
            "valid_geometry_hypotheses": len(corners),
            "camera_eligible_hypotheses": int(camera_eligible.sum()),
            "visibility_admission": {
                "min_visible_lengthwise": min_visible_lengthwise,
                "min_visible_cross_court": min_visible_cross_court,
                "visibility_columns": dict(VISIBILITY_COLUMN_LABELS),
                "hypotheses_before": len(visibility),
                "hypotheses_after": int(admission.visibility_admitted.sum()),
                "hypotheses_rejected": int((~admission.visibility_admitted).sum()),
                "floor_zero_scanned_for_proposal_cap": admission.floor_zero_scanned,
                "floor_zero_selected_count": len(admission.floor_zero_selected),
                "floor_zero_proposal_ids": [
                    f"rectangle_{int(rectangle_ids[index])}:template_{int(templates[index])}"
                    for index in admission.floor_zero_selected
                ],
                "scanned_for_proposal_cap": admission.scanned,
                "removed_from_floor_zero_count": len(admission.removed_from_floor_zero),
                "refilled_proposal_count": len(admission.newly_admitted),
                "newly_admitted_indices": admission.newly_admitted.tolist(),
                "newly_admitted_proposal_ids": [
                    f"rectangle_{int(rectangle_ids[index])}:template_{int(templates[index])}"
                    for index in admission.newly_admitted
                ],
                "removed_from_floor_zero_indices": admission.removed_from_floor_zero.tolist(),
                "removed_from_floor_zero_proposal_ids": [
                    f"rectangle_{int(rectangle_ids[index])}:template_{int(templates[index])}"
                    for index in admission.removed_from_floor_zero
                ],
            },
            "combined_admission_hypotheses": int(
                np.count_nonzero(camera_eligible & admission.visibility_admitted)
            ),
            "camera_scalar_recheck_count": int(scalar_recheck_count),
            "camera_vector_scalar_max_abs_diff": float(vector_scalar_max_abs_diff),
            "scanned_for_proposal_cap": int(scanned),
            "full_candidate_gate_count": len(entries),
            "elapsed_seconds": perf_counter() - started,
        },
        "caps": {
            str(cap): {
                "count": min(cap, len(selected)),
                "proposal_ids": [
                    entries[index]["proposal_id"] for index in range(min(cap, len(entries)))
                ],
            }
            for cap in (32, 64, 128, 256)
        },
        "estimator": {
            "retained_point_count": len(points),
            "raw_fragment_count": len(context.segments),
            "details": estimator,
        },
        "proposal_count": len(entries),
    }
    return Generation(tuple(entries), metadata)
