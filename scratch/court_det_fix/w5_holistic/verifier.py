"""Shared W5 measurement, validity and ranking helpers.

The automatic path in this module only receives frozen observations, candidate geometry and
player boxes. Reference records are loaded by the driver after automatic rankings are complete.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from experiments.annotator.independent_court import (
    assignment,
    detector,
    junction_observations,
    paint_geometry,
    stripe_observations,
)

WORKING_SIZE = (960, 540)
CAMERA_LIMIT = 0.1
PHOTO_CENTRE_OFFSETS_PX = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
PHOTO_SIDE_DISTANCE_PX = 6.0
Q_DIRECTION_LENGTHWISE = (0, 1, 2, 3, 4)
Q_DIRECTION_TRANSVERSE = (5, 6, 7, 8, 9, 10)

CASE_PACKS = {
    "gx": "frozen_views/packs/gx_extension_inputs.json.gz",
    "amateur": "frozen_views/packs/marking_refit_inputs.json.gz",
    "broadcast": "frozen_views/packs/broadcast_extension_inputs.json.gz",
}
CASE_ORDER = (
    ("gxBQ_window_00_frame_0", "gx", "GX0"),
    ("gxBQ_window_00_frame_5", "gx", "GX5"),
    ("am2_window_00_frame_150", "amateur", "Am2-150"),
    ("am2_window_01_frame_28019", "amateur", "Am2-28019"),
    ("am3_window_00_frame_0", "amateur", "Am3-0"),
    ("shuttleset_03_scene_0017", "broadcast", "SS03-17"),
    ("shuttleset_03_scene_0019", "broadcast", "SS03-19"),
    ("shuttleset_03_scene_0016", "broadcast", "SS03-16"),
    ("shuttleset_21_scene_0020", "broadcast", "SS21-20"),
)
CASE_IDS = tuple(case_id for case_id, _, _ in CASE_ORDER)
CASE_LABELS = {case_id: label for case_id, _, label in CASE_ORDER}
PACK_OF = {case_id: pack for case_id, pack, _ in CASE_ORDER}


@dataclass(frozen=True)
class ViewContext:
    """Raw, label-free inputs prepared once for one frozen view."""

    case_id: str
    source: dict
    native_size: tuple[int, int]
    size: tuple[int, int]
    segments: np.ndarray
    families: tuple[np.ndarray, np.ndarray]
    observations: assignment.Observations
    weights: np.ndarray
    frame: np.ndarray
    mask_boxes: np.ndarray
    same_image_mask_available: bool
    image_kind: str
    frame_relative_path: str


def jsonable(value: Any) -> Any:
    """Convert NumPy and pathlib values into strict JSON values."""
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def read_json_gz(path: Path) -> dict:
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def write_json_gz(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(jsonable(value), allow_nan=False, sort_keys=True).encode()
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(gzip.compress(payload, mtime=0))
    temporary.replace(path)


def relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def load_source(root: Path, case_id: str) -> dict:
    source_pack = read_json_gz(root / CASE_PACKS[PACK_OF[case_id]])
    return next(source for source in source_pack["cases"] if source["id"] == case_id)


def frame_path(root: Path, source: dict) -> Path:
    case_id = source["id"]
    if case_id.startswith("gxBQ"):
        return root / "frozen_views/frames/gx" / source["image"]
    if case_id.startswith("shuttleset"):
        return root / "frozen_views/frames/original" / source["image"]
    video = case_id.split("_", 1)[0]
    frame = int(case_id.rsplit("_", 1)[1])
    return root / "frozen_views/frames/amateur" / video / f"frame_{frame:08d}.png"


def image_kind(source: dict) -> str:
    provenance = source.get("provenance") or {}
    if source.get("image_kind"):
        return str(source["image_kind"])
    if provenance.get("image_kind"):
        return str(provenance["image_kind"])
    return "source_frame"


def has_same_image_boxes(source: dict) -> bool:
    """Return whether the selected boxes describe the exact image being measured."""
    case_id = source["id"]
    provenance = source.get("provenance") or {}
    if image_kind(source) != "source_frame":
        return False
    if case_id.startswith("gxBQ"):
        return provenance.get("anchor_frame_index") == provenance.get("chosen_bbox_frame")
    if case_id.startswith(("am2_", "am3_")):
        return source.get("anchor_frame_index") == source.get("bbox_frame_index")
    if case_id.startswith("shuttleset"):
        return provenance.get("anchor_frame_index") == provenance.get("chosen_bbox_frame")
    raise ValueError(f"unsupported W5 case prefix: {case_id}")


def mask_boxes_working(source: dict, size: tuple[int, int]) -> np.ndarray:
    if not has_same_image_boxes(source):
        return np.empty((0, 4), dtype=float)
    boxes = np.asarray(source.get("bbox_px", []), dtype=float).reshape(-1, 4)
    native_size = np.asarray([source["dimensions"]["width"], source["dimensions"]["height"]], dtype=float)
    scale = native_size / np.asarray(size, dtype=float)
    return boxes / np.tile(scale, 2)


def prepare_segments(source: dict) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], tuple[int, int]]:
    """Reproduce the frozen wide-family working-image preparation."""
    width, height = source["dimensions"]["width"], source["dimensions"]["height"]
    settings = detector.Settings(wide_families=True, min_supported_lines=3)
    resize = min(1.0, settings.max_dimension / max(width, height))
    size = (round(width * resize), round(height * resize))
    native_scale = np.asarray([width, height], dtype=float) / size
    segments = np.asarray(source["segments_px"], dtype=float) / np.tile(native_scale, 2)
    raw_families = detector._wide_line_families(segments)
    families = (detector._merge_lines(raw_families[0], settings), detector._merge_lines(raw_families[1], settings))
    return segments, families, size


def prepare_view(root: Path, case_id: str) -> ViewContext:
    source = load_source(root, case_id)
    segments, families, size = prepare_segments(source)
    observations = assignment.prepare_observations(segments, size)
    frame_file = frame_path(root, source)
    frame = cv2.imread(str(frame_file))
    if frame is None:
        raise FileNotFoundError(frame_file)
    expected_shape = (source["dimensions"]["height"], source["dimensions"]["width"])
    if frame.shape[:2] != expected_shape:
        raise ValueError(f"{case_id}: frame shape {frame.shape[:2]} does not match {expected_shape}")
    if (frame.shape[1], frame.shape[0]) != size:
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    native_size = (source["dimensions"]["width"], source["dimensions"]["height"])
    return ViewContext(
        case_id=case_id,
        source=source,
        native_size=native_size,
        size=size,
        segments=segments,
        families=families,
        observations=observations,
        weights=stripe_observations.fragment_weights(observations),
        frame=frame,
        mask_boxes=mask_boxes_working(source, size),
        same_image_mask_available=has_same_image_boxes(source),
        image_kind=image_kind(source),
        frame_relative_path=relative_path(frame_file, root),
    )


def source_provenance(context: ViewContext, g0_source: str) -> dict:
    source = context.source
    provenance = source.get("provenance", {})
    return {
        "case_id": context.case_id,
        "working_dimensions": list(context.size),
        "native_dimensions": list(context.native_size),
        "frame_path": context.frame_relative_path,
        "image_kind": context.image_kind,
        "g0_source": g0_source,
        "same_image_mask_available": context.same_image_mask_available,
        "photometry_occlusion_aware": context.same_image_mask_available,
        "player_foot_source": provenance.get("people_source", "frozen_pack_all_feet_px"),
        "chosen_bbox_frame": provenance.get("chosen_bbox_frame", source.get("bbox_frame_index")),
        "anchor_frame": provenance.get("anchor_frame_index", source.get("anchor_frame_index")),
    }


def project_corners(homography: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    corners, denominators = detector.project(homography[None], detector.CORNER_COURT_M)
    return corners[0], denominators[0]


def convex_corners(corners: np.ndarray) -> bool:
    edges = np.roll(corners, -1, axis=0) - corners
    turns = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    return bool(np.isfinite(turns).all() and np.all(turns > 1e-10))


def hard_validity(entry: dict) -> tuple[bool, str | None]:
    gates = entry.get("gates", {})
    if not bool(gates.get("geometry_valid", False)):
        return False, "historical_geometry_invalid"
    homography = np.asarray(entry.get("homography_working"), dtype=float)
    if homography.shape != (3, 3) or not np.isfinite(homography).all():
        return False, "non_finite_homography"
    try:
        corners, denominators = project_corners(homography)
    except (ValueError, np.linalg.LinAlgError):
        return False, "projection_failed"
    if not np.isfinite(corners).all() or not np.isfinite(denominators).all():
        return False, "non_finite_projection"
    if np.any(denominators <= 1e-6):
        return False, "non_positive_corner_denominator"
    if not convex_corners(corners):
        return False, "non_convex_corners"
    return True, None


def historical_predicates(gates: dict) -> dict[str, bool]:
    camera_error = gates.get("camera_error")
    camera_valid = camera_error is not None and camera_error <= CAMERA_LIMIT
    geometry_valid = bool(gates.get("geometry_valid", False))
    fractions = gates.get("player_fractions", [None, None])
    one_player = fractions[0] if len(fractions) > 0 else None
    two_player = fractions[1] if len(fractions) > 1 else None
    return {
        "historical_fullcourt": bool(geometry_valid and one_player == 1.0 and two_player is not None
                                      and two_player >= 0.5 and camera_valid),
        "historical_camera": bool(geometry_valid and camera_valid),
    }


def grayscale_sample(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    maps = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    values = cv2.remap(grey, maps[:, 0], maps[:, 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return values.reshape(np.asarray(points).shape[:-1])


def observable_points(points: np.ndarray, size: tuple[int, int], boxes: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    in_frame = ((points >= 0) & (points < np.asarray(size, dtype=float))).all(axis=-1)
    if not len(boxes):
        return in_frame
    inside_box = ((points[..., None, :] >= boxes[None, ..., :2])
                  & (points[..., None, :] <= boxes[None, ..., 2:])).all(axis=-1).any(axis=-1)
    return in_frame & ~inside_box


def photometric_samples(
    image: np.ndarray, samples: np.ndarray, direction: np.ndarray, boxes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return maximum ridge contrast and the historical contrast-10 flags per sample."""
    samples = np.asarray(samples, dtype=float).reshape(-1, 2)
    if not len(samples):
        return np.empty(0), np.empty(0, dtype=bool)
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]])
    centres = samples[:, None] + PHOTO_CENTRE_OFFSETS_PX[None, :, None] * normal
    sides = PHOTO_SIDE_DISTANCE_PX * normal
    minus = centres - sides
    plus = centres + sides
    available = (observable_points(centres, (image.shape[1], image.shape[0]), boxes)
                 & observable_points(minus, (image.shape[1], image.shape[0]), boxes)
                 & observable_points(plus, (image.shape[1], image.shape[0]), boxes))
    centre_values = grayscale_sample(image, centres)
    minus_values = grayscale_sample(image, minus)
    plus_values = grayscale_sample(image, plus)
    contrast = np.minimum(centre_values - minus_values, centre_values - plus_values)
    contrast = np.where(available, contrast, -np.inf)
    best = contrast.max(axis=1, initial=-np.inf)
    known = np.isfinite(best)
    best = np.where(known, best, np.nan)
    return best, known & (best >= 10.0)


def quantile_summary(values: Sequence[float | None]) -> dict:
    numeric = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    if not len(numeric):
        return {"count": 0}
    quantiles = np.quantile(numeric, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {
        "count": len(numeric),
        "minimum": float(numeric.min()),
        "maximum": float(numeric.max()),
        "p10": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p90": float(quantiles[4]),
    }


def exclusive_sample_support(
    stripe_evidence: stripe_observations.StripeEvidence,
    assignments: dict,
    marking: int,
    sample_start: int,
    sample_count: int,
) -> np.ndarray:
    """Calculate C(s) from one marking's exclusive fragment assignments."""
    assignment_marking = np.asarray(assignments["marking"], dtype=int)
    assignment_position = np.asarray(assignments["position"], dtype=int)
    full_forward = np.asarray(stripe_evidence.forward[marking])[:, sample_start:sample_start + sample_count, :]
    if not full_forward.shape[1]:
        return np.empty(0)
    support = np.zeros(full_forward.shape[1], dtype=float)
    for position in range(3):
        allowed = (assignment_marking == marking) & (assignment_position == position)
        if allowed.any():
            support = np.maximum(support, np.max(np.where(allowed[None, :], full_forward[position], 0.0), axis=1))
    return support


def physical_marking_evidence(
    context: ViewContext, homography: np.ndarray, stripe_evidence: stripe_observations.StripeEvidence,
    stripe_score: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    projected, _ = detector.project(homography[None], paint_geometry.CENTRE_SEGMENTS_M)
    endpoints = projected.reshape(12, 2, 2)
    samples, visible = detector._visible_samples(endpoints[None], context.size, assignment.MARKING_SAMPLES)
    samples = samples[0]
    visible = visible[0]
    assignments = stripe_score["assignments"]
    marking_records = []
    arrays: dict[str, np.ndarray] = {}
    q_geom_values: list[float | None] = []
    q_paint_values: list[float | None] = []
    for marking, intervals in enumerate(assignment.MARKING_INTERVALS):
        c_parts, ridge_parts, p10_parts = [], [], []
        spans = []
        cursor = 0
        for interval in intervals:
            if not visible[interval]:
                continue
            interval_samples = samples[interval]
            interval_c = exclusive_sample_support(
                stripe_evidence, assignments, marking, cursor, len(interval_samples)
            )
            cursor += len(interval_samples)
            vector = endpoints[interval, 1] - endpoints[interval, 0]
            ridge, p10 = photometric_samples(context.frame, interval_samples, vector, context.mask_boxes)
            c_parts.append(interval_c)
            ridge_parts.append(ridge)
            p10_parts.append(p10)
            spans.append(float(np.linalg.norm(interval_samples[-1] - interval_samples[0])))
        c_support = np.concatenate(c_parts) if c_parts else np.empty(0)
        ridge = np.concatenate(ridge_parts) if ridge_parts else np.empty(0)
        p10 = np.concatenate(p10_parts) if p10_parts else np.empty(0, dtype=bool)
        known = np.isfinite(ridge)
        q_geom = float(c_support.mean()) if len(c_support) else None
        q_paint = float(np.mean(c_support[known] * p10[known])) if known.any() else None
        q_geom_values.append(q_geom)
        q_paint_values.append(q_paint)
        assignment_count = int(np.sum(np.asarray(assignments["marking"]) == marking))
        marking_records.append({
            "marking": assignment.MARKINGS[marking],
            "marking_index": marking,
            "visible_samples": len(c_support),
            "known_photometry_samples": int(known.sum()),
            "known_photometry_fraction": float(known.mean()) if len(known) else None,
            "projected_visible_span_px": float(sum(spans)) if spans else 0.0,
            "q_geom": q_geom,
            "q_paint10": q_paint,
            "exclusive_fragment_count": assignment_count,
            "ridge_contrast": quantile_summary(ridge.tolist()),
        })
        arrays[f"marking_{marking}_c_support"] = c_support
        arrays[f"marking_{marking}_ridge_contrast"] = ridge
    lengthwise = [value for value in q_geom_values[:5] if value is not None]
    transverse = [value for value in q_geom_values[5:] if value is not None]
    lengthwise_paint = [value for value in q_paint_values[:5] if value is not None]
    transverse_paint = [value for value in q_paint_values[5:] if value is not None]

    def span_weighted(values: list[float | None], marking_indices: Sequence[int]) -> tuple[float | None, int]:
        weighted_values = []
        total_span = 0.0
        for value, marking_index in zip(values, marking_indices, strict=True):
            span = float(marking_records[marking_index]["projected_visible_span_px"])
            if value is None or span <= 0.0:
                continue
            weighted_values.append(float(value) * span)
            total_span += span
        if not total_span:
            return None, 0
        return sum(weighted_values) / total_span, len(weighted_values)

    lengthwise_geom_span, lengthwise_geom_count = span_weighted(q_geom_values[:5], range(5))
    transverse_geom_span, transverse_geom_count = span_weighted(q_geom_values[5:], range(5, 11))
    lengthwise_paint_span, lengthwise_paint_count = span_weighted(q_paint_values[:5], range(5))
    transverse_paint_span, transverse_paint_count = span_weighted(q_paint_values[5:], range(5, 11))
    q_geom = min(float(np.mean(lengthwise)), float(np.mean(transverse))) if lengthwise and transverse else None
    q_paint = (min(float(np.mean(lengthwise_paint)), float(np.mean(transverse_paint)))
               if lengthwise_paint and transverse_paint else None)
    q_geom_span = (min(lengthwise_geom_span, transverse_geom_span)
                   if lengthwise_geom_span is not None and transverse_geom_span is not None else None)
    q_paint_span = (min(lengthwise_paint_span, transverse_paint_span)
                    if lengthwise_paint_span is not None and transverse_paint_span is not None else None)
    evidence = {
        "markings": marking_records,
        "directional": {
            "lengthwise_q_geom": float(np.mean(lengthwise)) if lengthwise else None,
            "transverse_q_geom": float(np.mean(transverse)) if transverse else None,
            "lengthwise_q_paint10": float(np.mean(lengthwise_paint)) if lengthwise_paint else None,
            "transverse_q_paint10": float(np.mean(transverse_paint)) if transverse_paint else None,
            "lengthwise_markings_geom": len(lengthwise),
            "transverse_markings_geom": len(transverse),
            "lengthwise_markings_paint10": len(lengthwise_paint),
            "transverse_markings_paint10": len(transverse_paint),
            "lengthwise_q_geom_span_weighted": lengthwise_geom_span,
            "transverse_q_geom_span_weighted": transverse_geom_span,
            "lengthwise_q_paint10_span_weighted": lengthwise_paint_span,
            "transverse_q_paint10_span_weighted": transverse_paint_span,
            "lengthwise_markings_geom_span_weighted": lengthwise_geom_count,
            "transverse_markings_geom_span_weighted": transverse_geom_count,
            "lengthwise_markings_paint10_span_weighted": lengthwise_paint_count,
            "transverse_markings_paint10_span_weighted": transverse_paint_count,
        },
        "q_geom": q_geom,
        "q_paint10": q_paint,
        "q_geom_span_weighted": q_geom_span,
        "q_paint10_span_weighted": q_paint_span,
        "photometry_occlusion_aware": context.same_image_mask_available,
        "exclusive_reverse": float(stripe_score["exclusive"]["reverse"]),
        "exclusive_score": float(stripe_score["exclusive"]["score"]),
        "assignment_strength": quantile_summary(stripe_score["assignments"]["strength"]),
        "assigned_fragment_count": int(np.sum(np.asarray(assignments["marking"]) >= 0)),
    }
    return evidence, arrays


def support_samples(
    samples: np.ndarray, direction: np.ndarray, observations: assignment.Observations,
) -> np.ndarray:
    if not len(samples) or not len(observations.segments):
        return np.zeros(len(samples), dtype=float)
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    compatible = np.abs(observations.directions @ direction) >= np.cos(np.deg2rad(assignment.MATCH_ANGLE_DEG))
    distances = assignment.distances_to_segments(samples, observations.segments)
    response = np.exp(-0.5 * np.square(distances / assignment.DISTANCE_SIGMA_PX))
    response = np.where(compatible[None], response, 0.0)
    return response.max(axis=1, initial=0.0)


def raw_junctions(context: ViewContext, homography: np.ndarray, stripe_score: dict) -> dict:
    physical = paint_geometry.CENTRE_SEGMENTS_M
    historical = junction_observations.measure(
        homography, context.observations, context.mask_boxes, context.size, centres=physical,
    )
    sites = []
    for name, transverse in zip(junction_observations.SITE_NAMES, physical[6:]):
        junction_m = np.array([physical[2, 0, 0], transverse[0, 1]])
        expected = junction_observations.expected_vertical_arms(float(junction_m[1]), physical)
        arms = {}
        for arm, direction_values in junction_observations.ARM_DIRECTIONS.items():
            direction = np.asarray(direction_values, dtype=float)
            normal = np.array([-direction[1], direction[0]])
            along = np.linspace(junction_observations.ARM_START_M, junction_observations.ARM_END_M,
                                junction_observations.ARM_SAMPLES)
            centre_m = junction_m + along[:, None] * direction
            queries_m = centre_m[None] + paint_geometry.POSITION_OFFSETS_M[:, None, None] * normal
            queries, _ = detector.project(homography[None], queries_m)
            queries = queries.reshape(3, junction_observations.ARM_SAMPLES, 2)
            position_records = []
            for position_samples in queries:
                usable = observable_points(position_samples, context.size, context.mask_boxes)
                visible_samples = position_samples[usable]
                projected_direction = position_samples[-1] - position_samples[0]
                projected_direction /= np.linalg.norm(projected_direction)
                support = support_samples(visible_samples, projected_direction, context.observations)
                ridge, p10 = photometric_samples(
                    context.frame, visible_samples, projected_direction, context.mask_boxes,
                )
                position_records.append({
                    "usable_samples": int(usable.sum()),
                    "projected_span_px": (float(np.linalg.norm(visible_samples[-1] - visible_samples[0]))
                                           if len(visible_samples) >= 2 else 0.0),
                    "fragment_support": quantile_summary(support.tolist()),
                    "fragment_support_mean": float(support.mean()) if len(support) else None,
                    "ridge_contrast": quantile_summary(ridge.tolist()),
                    "known_photometry_samples": int(np.isfinite(ridge).sum()),
                    "historical_p10_fraction": float(p10[np.isfinite(ridge)].mean()) if np.isfinite(ridge).any()
                    else None,
                    "ridge_contrast_samples": [None if not np.isfinite(value) else float(value) for value in ridge],
                })
            supported = [row for row in position_records if row["fragment_support_mean"] is not None]
            arms[arm] = {
                "support": max((row["fragment_support_mean"] for row in supported), default=None),
                "usable_samples": max((row["usable_samples"] for row in supported), default=0),
                "projected_span_px": max((row["projected_span_px"] for row in supported), default=0.0),
                "positions": position_records,
                "mask_available": context.same_image_mask_available,
            }
        contradictions = {}
        for arm in ("far", "near"):
            if not expected[arm]:
                contradictions[arm] = {
                    "expected_painted": False,
                    "fragment_support": arms[arm]["support"],
                    "photometric_continuation": [
                        {
                            "known_samples": row["known_photometry_samples"],
                            "p10_fraction": row["historical_p10_fraction"],
                            "ridge_contrast": row["ridge_contrast"],
                        }
                        for row in arms[arm]["positions"]
                    ],
                }
        sites.append({"marking": name, "expected": expected, "arms": arms, "contradictions": contradictions})
    return {
        "historical": historical,
        "sites": sites,
        "mask_available": context.same_image_mask_available,
        "exclusive_reverse": float(stripe_score["exclusive"]["reverse"]),
    }


def measure_candidate(
    context: ViewContext, entry: dict, cache: dict[bytes, tuple[dict, dict[str, np.ndarray]]],
) -> tuple[dict, dict[str, np.ndarray]]:
    homography = np.asarray(entry["homography_working"], dtype=float)
    key = homography.tobytes()
    if key in cache:
        return cache[key]
    stripe_evidence = stripe_observations.measure(
        homography, context.observations, context.size, centres=paint_geometry.CENTRE_SEGMENTS_M,
    )
    stripe_score = stripe_observations.score_model(stripe_evidence, context.weights, 3)
    physical, arrays = physical_marking_evidence(context, homography, stripe_evidence, stripe_score)
    physical["junctions"] = raw_junctions(context, homography, stripe_score)
    physical["stripe_assignments"] = {
        "marking": stripe_score["assignments"]["marking"],
        "position": stripe_score["assignments"]["position"],
        "strength": stripe_score["assignments"]["strength"],
        "alternative_marking": stripe_score["assignments"]["alternative_marking"],
        "alternative_strength": stripe_score["assignments"]["alternative_strength"],
    }
    result = (physical, arrays)
    cache[key] = result
    return result


def origin_sort_key(candidate: dict) -> tuple[int, int, int, str]:
    return (int(candidate["source_order"]), int(candidate["origin_index"]), int(candidate["kind_order"]),
            str(candidate["origin_key"]))


def camera_eligible(candidate: dict) -> bool:
    camera_error = candidate.get("gates", {}).get("camera_error")
    return camera_error is not None and camera_error <= CAMERA_LIMIT


def order_candidates(candidates: Sequence[dict], score_name: str) -> list[dict]:
    eligible = [candidate for candidate in candidates if candidate.get("hard_valid") and
                candidate.get("evidence", {}).get(score_name) is not None]
    return sorted(
        eligible,
        key=lambda candidate: (-float(candidate["evidence"][score_name]),
                              -float(candidate["evidence"].get("exclusive_reverse", 0.0)),
                              *origin_sort_key(candidate)),
    )


def provisional_order(candidates: Sequence[dict]) -> tuple[str, list[dict]]:
    paint_order = order_candidates(candidates, "q_paint10")
    if paint_order:
        return "q_paint10", paint_order
    return "q_geom", order_candidates(candidates, "q_geom")


def legacy_winners(entries: Sequence[dict]) -> dict:
    eligible = [entry for entry in entries
                if entry.get("gates", {}).get("camera_error") is not None
                and entry["gates"]["camera_error"] <= CAMERA_LIMIT
                and entry.get("profile", {}).get("score") is not None]
    line = max(eligible, key=lambda entry: entry["stripe"]["exclusive"]["score"], default=None)
    paint = max(eligible, key=lambda entry: (entry["profile"]["score"], entry["stripe"]["exclusive"]["score"]),
                default=None)
    return {
        "line": None if line is None else line["candidate_id"],
        "paint": None if paint is None else paint["candidate_id"],
        "eligible_count": len(eligible),
    }


def sparse_fallbacks(candidates: Sequence[dict]) -> dict:
    eligible = [candidate for candidate in candidates if candidate.get("hard_valid") and camera_eligible(candidate)]

    def best(score_name: str) -> dict | None:
        ranked = [candidate for candidate in eligible if candidate.get("evidence", {}).get(score_name) is not None]
        if not ranked:
            return None
        winner = min(
            ranked,
            key=lambda candidate: (-float(candidate["evidence"][score_name]),
                                  -float(candidate["evidence"].get("exclusive_reverse", 0.0)),
                                  *origin_sort_key(candidate)),
        )
        return {"origin_key": winner["origin_key"], "score": winner["evidence"][score_name]}

    def best_direction(direction: str) -> dict | None:
        score_names = (
            f"{direction}_q_paint10_span_weighted",
            f"{direction}_q_paint10",
            f"{direction}_q_geom_span_weighted",
            f"{direction}_q_geom",
        )
        for score_name in score_names:
            ranked = [
                candidate for candidate in eligible
                if candidate.get("evidence", {}).get("directional", {}).get(score_name) is not None
            ]
            if ranked:
                winner = min(
                    ranked,
                    key=lambda candidate: (
                        -float(candidate["evidence"]["directional"][score_name]),
                        -float(candidate["evidence"].get("exclusive_reverse", 0.0)),
                        *origin_sort_key(candidate),
                    ),
                )
                return {
                    "origin_key": winner["origin_key"],
                    "score": winner["evidence"]["directional"][score_name],
                    "score_name": score_name,
                }
        return None

    return {
        "lengthwise": best_direction("lengthwise"),
        "transverse": best_direction("transverse"),
        "q_geom": best("q_geom"),
        "q_paint10": best("q_paint10"),
        "exclusive_reverse": best("exclusive_reverse"),
    }


def rank_candidates(candidates: Sequence[dict]) -> dict:
    hard_valid = [candidate for candidate in candidates if candidate.get("hard_valid")]
    q_geom = order_candidates(hard_valid, "q_geom")
    q_paint = order_candidates(hard_valid, "q_paint10")
    ungated_criterion, ungated_provisional = provisional_order(hard_valid)
    exclusive = sorted(hard_valid, key=lambda candidate: (-float(candidate["evidence"].get("exclusive_reverse", 0.0)),
                                                          *origin_sort_key(candidate)))
    fullcourt = [candidate for candidate in ungated_provisional if candidate["historical"]["historical_fullcourt"]]
    camera = [candidate for candidate in ungated_provisional if candidate["historical"]["historical_camera"]]
    camera_pool = [candidate for candidate in hard_valid if camera_eligible(candidate)]
    r1_paint = order_candidates(camera_pool, "q_paint10")
    r2_paint = order_candidates(camera_pool, "q_paint10_span_weighted")
    r2_geom = order_candidates(camera_pool, "q_geom_span_weighted")
    if r2_paint:
        r2_criterion, provisional = "q_paint10_span_weighted", r2_paint
    elif r2_geom:
        r2_criterion, provisional = "q_geom_span_weighted", r2_geom
    else:
        r2_criterion, provisional = None, []
    if not hard_valid:
        status = "no_valid_candidate"
    elif not camera_pool:
        status = "no_plausible_camera"
    elif not provisional:
        status = "evidence_sparse"
    else:
        status = "provisional_for_review"
    return {
        "q_geom_rank": [candidate["origin_key"] for candidate in q_geom],
        "q_paint10_rank": [candidate["origin_key"] for candidate in q_paint],
        "provisional_rank": [candidate["origin_key"] for candidate in provisional],
        "exclusive_reverse_rank": [candidate["origin_key"] for candidate in exclusive],
        "historical_fullcourt_subset_rank": [candidate["origin_key"] for candidate in fullcourt],
        "historical_camera_subset_rank": [candidate["origin_key"] for candidate in camera],
        "r1_paint10_rank": [candidate["origin_key"] for candidate in r1_paint],
        "r2_spanw_paint10_rank": [candidate["origin_key"] for candidate in r2_paint],
        "ungated_provisional_rank": [candidate["origin_key"] for candidate in ungated_provisional],
        "criterion": ungated_criterion,
        "r1_criterion": "q_paint10" if r1_paint else None,
        "r2_criterion": r2_criterion,
        "status": status,
        "selected_origin_key": provisional[0]["origin_key"] if provisional else None,
        "r1_selected_origin_key": r1_paint[0]["origin_key"] if r1_paint else None,
        "r2_selected_origin_key": provisional[0]["origin_key"] if provisional else None,
        "sparse_fallbacks": sparse_fallbacks(hard_valid) if status == "evidence_sparse" else {},
    }


def permutation_determinism(candidates: Sequence[dict]) -> dict:
    original = rank_candidates(list(candidates))["provisional_rank"]
    permutation = np.random.default_rng(20260920).permutation(len(candidates))
    permuted = rank_candidates([candidates[index] for index in permutation])["provisional_rank"]
    return {"match": original == permuted, "original": original, "permuted": permuted}


def candidate_review(candidate: dict) -> dict:
    evidence = candidate.get("evidence", {})
    return {
        "origin_key": candidate["origin_key"],
        "candidate_id": candidate["candidate_id"],
        "kind": candidate["kind"],
        "parent_origin_key": candidate.get("parent_origin_key"),
        "source": candidate.get("source"),
        "source_order": candidate.get("source_order"),
        "origin_index": candidate.get("origin_index"),
        "kind_order": candidate.get("kind_order"),
        "expected_ruling": candidate.get("expected_ruling"),
        "in_automatic_pool": candidate.get("in_automatic_pool"),
        "corners_px": candidate.get("corners_px"),
        "homography_working": candidate.get("homography_working"),
        "hard_valid": candidate.get("hard_valid", False),
        "hard_validity_reason": candidate.get("hard_validity_reason"),
        "camera_eligible": camera_eligible(candidate),
        "gates": candidate.get("gates"),
        "historical": candidate.get("historical"),
        "q_geom": evidence.get("q_geom"),
        "q_paint10": evidence.get("q_paint10"),
        "q_geom_span_weighted": evidence.get("q_geom_span_weighted"),
        "q_paint10_span_weighted": evidence.get("q_paint10_span_weighted"),
        "exclusive_reverse": evidence.get("exclusive_reverse"),
        "exclusive_score": evidence.get("exclusive_score"),
        "directional": evidence.get("directional"),
        "markings": evidence.get("markings"),
        "junctions": {
            "historical": evidence.get("junctions", {}).get("historical"),
            "contradictions": [
                {"marking": site["marking"], "contradictions": site["contradictions"]}
                for site in evidence.get("junctions", {}).get("sites", []) if site["contradictions"]
            ],
        },
        "refit": candidate.get("refit"),
    }


def reference_corner_error(corners: np.ndarray, reference: np.ndarray) -> dict:
    corners = np.asarray(corners, dtype=float)
    reference = np.asarray(reference, dtype=float)
    direct = np.linalg.norm(corners - reference, axis=1)
    rotated = np.linalg.norm(corners - reference[[2, 3, 0, 1]], axis=1)
    if direct.max() <= rotated.max():
        return {"relabelled_180": False, "per_corner": direct.tolist(), "maximum": float(direct.max())}
    return {"relabelled_180": True, "per_corner": rotated.tolist(), "maximum": float(rotated.max())}


def projected_landmark_diagnostics(
    homography: np.ndarray, landmarks: Sequence[dict], context: ViewContext,
) -> list[dict]:
    if not landmarks:
        return []
    court = np.asarray([landmark["court_m"] for landmark in landmarks], dtype=float)
    projected, _ = detector.project(homography[None], court)
    scale = np.asarray(context.native_size, dtype=float) / np.asarray(context.size, dtype=float)
    projected_native = projected[0] * scale
    court_step = np.array([1e-4, 0.0])
    tangent_plus, _ = detector.project(homography[None], court + court_step)
    tangent_minus, _ = detector.project(homography[None], court - court_step)
    tangent_native = (tangent_plus[0] - tangent_minus[0]) * scale
    diagnostics = []
    for index, (landmark, prediction, tangent) in enumerate(
        zip(landmarks, projected_native, tangent_native, strict=True)
    ):
        observed = np.asarray(landmark["image_px"], dtype=float)
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        normal /= np.linalg.norm(normal)
        diagnostics.append({
            "index": index,
            "name": landmark.get("name", f"landmark_{index}"),
            "court_m": landmark["court_m"],
            "observed_px": observed.tolist(),
            "predicted_px": prediction.tolist(),
            "error_px": float(np.linalg.norm(prediction - observed)),
            "normal_convention": "left of increasing court-x direction",
            "signed_normal_offset_px": float(np.dot(prediction - observed, normal)),
        })
    return diagnostics
