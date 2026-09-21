"""Audit bounded line/template proposal admission without W5 integration."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np


CASE_PACKS = {
    "gx": "frozen_views/packs/gx_extension_inputs.json.gz",
    "amateur": "frozen_views/packs/marking_refit_inputs.json.gz",
    "broadcast": "frozen_views/packs/broadcast_extension_inputs.json.gz",
}
CASE_TO_PACK = {
    "gxBQ_window_00_frame_0": "gx",
    "gxBQ_window_00_frame_5": "gx",
    "am2_window_00_frame_150": "amateur",
    "am2_window_01_frame_28019": "amateur",
    "am3_window_00_frame_0": "amateur",
    "shuttleset_03_scene_0017": "broadcast",
    "shuttleset_03_scene_0019": "broadcast",
    "shuttleset_03_scene_0016": "broadcast",
    "shuttleset_21_scene_0020": "broadcast",
}
CASE_IDS = tuple(CASE_TO_PACK)
CAPS = (32, 64, 128, 256)
TEMPLATE_COUNT = 150
SUPPORT_DISTANCE = 4.0
SAMPLES_PER_LINE = 24
DIVERSITY_RADIUS = 12.0
CAMERA_LIMIT = 0.1
CAMERA_RECHECK_MARGIN = 1e-3
FULL_RECTANGLE_ORDER = 10**9
RECTANGLE_CAP = 4096


def load_helpers(root: Path):
    """Import the frozen detector and VP selector from the experiment snapshot."""
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))
    vp_root = root / "vp_pruning_20260914"
    if not vp_root.exists():
        vp_root = root / "scratch/court_det_fix/frozen_helpers_20260914/vp_pruning"
    sys.path.insert(0, str(vp_root))
    from experiments.annotator.independent_court import detector  # noqa: PLC0415
    from vp_pruning import Settings as VPSettings  # noqa: PLC0415
    from vp_pruning import estimate, rectangle_population, select  # noqa: PLC0415

    return detector, VPSettings, estimate, rectangle_population, select


def load_camera_helper(root: Path):
    camera_root = root / "smoke/legacy"
    if not camera_root.exists():
        camera_root = root / "scratch/court_det_fix/worklog/checks/independent/player_guided/20260908"
    sys.path.insert(0, str(camera_root))
    import camera_diagnostic  # noqa: PLC0415

    return camera_diagnostic


def read_json_gz(path: Path) -> dict:
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def write_json_gz(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    path.write_bytes(gzip.compress(payload, mtime=0))


def input_digest(source: dict) -> str:
    payload = json.dumps(
        {"id": source["id"], "dimensions": source["dimensions"], "segments_px": source["segments_px"]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def load_case(root: Path, case_id: str) -> tuple[dict, str]:
    pack_name = CASE_TO_PACK[case_id]
    pack_path = root / CASE_PACKS[pack_name]
    packed = read_json_gz(pack_path)
    source = next(case for case in packed["cases"] if case["id"] == case_id)
    return source, CASE_PACKS[pack_name]


def prepare_source(source: dict, detector) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], tuple[int, int]]:
    """Reproduce W5's cached-fragment preparation in working coordinates."""
    width = source["dimensions"]["width"]
    height = source["dimensions"]["height"]
    settings = detector.Settings(wide_families=True, min_supported_lines=3)
    resize = min(1.0, settings.max_dimension / max(width, height))
    size = (round(width * resize), round(height * resize))
    native_scale = np.asarray([width, height], dtype=np.float64) / size
    segments = np.asarray(source["segments_px"], dtype=np.float64) / np.tile(native_scale, 2)
    raw_families = detector._wide_line_families(segments)
    families = (
        detector._merge_lines(raw_families[0], settings),
        detector._merge_lines(raw_families[1], settings),
    )
    return segments, families, size


def union_distance_map(segments: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    mask = np.full((height, width), 255, dtype=np.uint8)
    for x1, y1, x2, y2 in np.rint(segments).astype(int):
        cv2.line(mask, (x1, y1), (x2, y2), 0, 1)
    return cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def geometry_and_support(
    homographies: np.ndarray,
    maps: np.ndarray,
    map_kind: str,
    size: tuple[int, int],
    detector,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply inherited geometry checks and return two directional support means."""
    corners, denominator = detector.project(homographies, detector.CORNER_COURT_M)
    valid = np.isfinite(corners).all(axis=(1, 2)) & np.all(denominator > 1e-6, axis=1)
    edges = np.roll(corners, -1, axis=1) - corners
    turns = edges[..., 0] * np.roll(edges[..., 1], -1, axis=1) - edges[..., 1] * np.roll(
        edges[..., 0], -1, axis=1
    )
    valid &= np.all(turns > 0, axis=1)
    visible_lower = np.maximum(corners.min(axis=1), 0)
    visible_upper = np.minimum(corners.max(axis=1), np.asarray(size) - 1)
    visible_span = (visible_upper - visible_lower) / np.asarray(size)
    valid &= np.all(visible_span >= 0.15, axis=1)
    if not valid.any():
        return corners[:0], np.empty((0, 2), dtype=np.float32), valid, np.empty((0, 2), dtype=np.int16)

    valid_corners = corners[valid]
    endpoints, _ = detector.project(homographies[valid], detector.SEGMENTS_M)
    samples, visible = detector._visible_samples(
        endpoints.reshape(-1, 12, 2, 2), size, SAMPLES_PER_LINE
    )
    pixel_x = np.clip(samples[..., 0], 0, size[0] - 1).astype(int)
    pixel_y = np.clip(samples[..., 1], 0, size[1] - 1).astype(int)
    if map_kind == "family":
        family_ids = np.repeat([0, 1], 6)[None, :, None]
        support = (maps[family_ids, pixel_y, pixel_x] <= SUPPORT_DISTANCE).mean(axis=-1)
    elif map_kind == "union":
        support = (maps[pixel_y, pixel_x] <= SUPPORT_DISTANCE).mean(axis=-1)
    else:
        raise ValueError(f"unknown map kind: {map_kind}")
    support *= visible
    means = np.stack(
        [
            support[:, :6].sum(axis=1) / np.maximum(visible[:, :6].sum(axis=1), 1),
            support[:, 6:].sum(axis=1) / np.maximum(visible[:, 6:].sum(axis=1), 1),
        ],
        axis=1,
    ).astype(np.float32)
    visibility = np.stack([visible[:, :6].sum(axis=1), visible[:, 6:].sum(axis=1)], axis=1).astype(np.int16)
    return valid_corners, means, valid, visibility


def vector_camera_errors(homographies: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Vectorise the frozen camera_diagnostic.camera error calculation."""
    width, height = size
    focals = np.geomspace(0.4 * width, 4.0 * width, 200)
    axes = np.broadcast_to(homographies[:, None, :, :2], (len(homographies), len(focals), 3, 2)).copy()
    principal = np.asarray([width / 2.0, height / 2.0])
    axes[:, :, :2] -= principal[None, None, :, None] * homographies[:, None, 2:3, :2]
    axes[:, :, :2] /= focals[None, :, None, None]
    norms = np.linalg.norm(axes, axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = (axes[:, :, :, 0] * axes[:, :, :, 1]).sum(axis=2) / np.prod(norms, axis=2)
        ratio = np.log(norms[:, :, 0] / norms[:, :, 1])
        errors = np.hypot(cosine, ratio)
    errors = np.where(
        np.isfinite(errors) & np.all(np.isfinite(norms), axis=2) & np.all(norms > 0, axis=2),
        errors,
        np.inf,
    )
    return errors.min(axis=1)


def camera_errors_with_frontier_recheck(
    homographies: np.ndarray,
    corners_working: np.ndarray,
    working_size: tuple[int, int],
    native_scale: np.ndarray,
    native_size: tuple[int, int],
    camera_helper,
) -> tuple[np.ndarray, int, float]:
    """Use vectorised frozen-camera math and scalar checks near the hard gate."""
    errors = vector_camera_errors(homographies, working_size)
    frontier = np.flatnonzero(np.abs(errors - CAMERA_LIMIT) <= CAMERA_RECHECK_MARGIN)
    differences = []
    for index in frontier:
        # W5 rebuilds H from float32 native corners before calling camera().
        native_corners = (corners_working[index] * native_scale).astype(np.float32)
        scalar_error = float(camera_helper.camera(native_corners, native_size)[0])
        differences.append(abs(float(errors[index]) - scalar_error))
        errors[index] = scalar_error
    return errors, len(frontier), max(differences, default=0.0)


def greedy_diverse(
    order: np.ndarray,
    corners: np.ndarray,
    cap: int,
    radius: float = DIVERSITY_RADIUS,
) -> tuple[np.ndarray, int]:
    """Retain a deterministic prefix under the inherited corner-distance rule."""
    retained: list[int] = []
    scanned = 0
    for index in order:
        scanned += 1
        index = int(index)
        if retained:
            separation = np.linalg.norm(corners[index] - corners[retained], axis=2).max(axis=1)
            if np.any(separation <= radius):
                continue
        retained.append(index)
        if len(retained) == cap:
            break
    return np.asarray(retained, dtype=np.int64), scanned


def candidate_records(
    selected: np.ndarray,
    scores: np.ndarray,
    means: np.ndarray,
    corners: np.ndarray,
    visibility: np.ndarray,
    camera_errors: np.ndarray,
    rectangle_ids: np.ndarray,
    rectangle_ranks: np.ndarray,
    templates: np.ndarray,
    native_scale: np.ndarray,
    cap: int,
    scanned: int,
) -> dict:
    records = []
    for index in selected[:cap]:
        index = int(index)
        rectangle_id = int(rectangle_ids[index])
        template_index = int(templates[index])
        records.append(
            {
                "proposal_id": f"rectangle_{rectangle_id}:template_{template_index}",
                "rectangle_id": rectangle_id,
                "rectangle_order": int(rectangle_ranks[index]),
                "template_index": template_index,
                "score": float(scores[index]),
                "direction_means": [float(value) for value in means[index]],
                "visible_direction_samples": [int(value) for value in visibility[index]],
                "camera_error": float(camera_errors[index]),
                "corners_working": corners[index].round(7).tolist(),
                "corners_native": (corners[index] * native_scale).round(7).tolist(),
            }
        )
    return {"scanned_for_256": int(scanned), "count": len(records), "proposals": records}


def rank_variant(
    scores: np.ndarray,
    means: np.ndarray,
    corners: np.ndarray,
    visibility: np.ndarray,
    rectangle_ids: np.ndarray,
    rectangle_ranks: np.ndarray,
    templates: np.ndarray,
    native_scale: np.ndarray,
    camera_errors: np.ndarray,
) -> dict:
    camera_eligible = camera_errors <= CAMERA_LIMIT
    eligible_indices = np.flatnonzero(camera_eligible)
    eligible_order = np.lexsort(
        (templates[eligible_indices], rectangle_ids[eligible_indices], -scores[eligible_indices])
    )
    order = eligible_indices[eligible_order]
    selected, scanned = greedy_diverse(order, corners, max(CAPS))
    caps = {
        str(cap): candidate_records(
            selected, scores, means, corners, visibility, camera_errors, rectangle_ids,
            rectangle_ranks, templates, native_scale, cap, scanned,
        )
        for cap in CAPS
    }
    return {
        "geometry_valid_count": int(len(scores)),
        "camera_eligible_count": int(len(order)),
        "full_order_count": int(len(order)),
        "top_score": float(scores[order[0]]) if len(order) else None,
        "selected_256_count": int(len(selected)),
        "caps": caps,
    }


def save_arrays(
    path: Path,
    rectangle_ids: np.ndarray,
    rectangle_ranks: np.ndarray,
    templates: np.ndarray,
    corners: np.ndarray,
    family_means: np.ndarray,
    union_means: np.ndarray,
    visibility: np.ndarray,
    camera_errors: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        rectangle_ids=rectangle_ids,
        rectangle_ranks=rectangle_ranks,
        templates=templates,
        corners=corners.astype(np.float32),
        family_means=family_means.astype(np.float32),
        union_means=union_means.astype(np.float32),
        visibility=visibility,
        camera_errors=camera_errors.astype(np.float32),
    )


def generate_case(root: Path, output: Path, case_id: str, detector, VPSettings, estimate, rectangle_population, select) -> None:
    source, pack_name = load_case(root, case_id)
    started = perf_counter()
    segments, families, size = prepare_source(source, detector)
    width, height = source["dimensions"]["width"], source["dimensions"]["height"]
    native_scale = np.asarray([width, height], dtype=np.float64) / np.asarray(size, dtype=np.float64)
    camera_helper = load_camera_helper(root)

    vp_settings = VPSettings(
        angle_deg=1.5,
        direction_lines=128,
        pencils=16,
        overlap=0.8,
        rectangles=FULL_RECTANGLE_ORDER,
        candidate_batch=256,
        pencil_selection="coverage",
    )
    points, estimator = estimate(segments, size, vp_settings)
    detector_settings = detector.Settings(wide_families=True, min_supported_lines=3)
    _, selection = select(families, points, size, detector_settings, vp_settings)
    selected_ids = np.asarray(selection.get("selected_pair_product_ids", []), dtype=np.int64)
    selected_ids = selected_ids[:RECTANGLE_CAP]
    quads, _, all_ids = rectangle_population(families, size)
    quad_by_id = {int(rectangle_id): quad for rectangle_id, quad in zip(all_ids, quads, strict=True)}
    valid_rectangle_ids = []
    valid_rectangle_ranks = []
    rectangles = []
    for rectangle_rank, rectangle_id in enumerate(selected_ids):
        quad = quad_by_id[int(rectangle_id)].astype(np.float32)
        if cv2.contourArea(quad) < 100:
            continue
        valid_rectangle_ids.append(int(rectangle_id))
        valid_rectangle_ranks.append(rectangle_rank)
        rectangles.append(cv2.getPerspectiveTransform(detector.UNIT_CORNERS, quad))
    rectangles_array = np.asarray(rectangles, dtype=np.float64).reshape(-1, 3, 3)
    rectangle_ids = np.repeat(np.asarray(valid_rectangle_ids, dtype=np.int64), TEMPLATE_COUNT)
    rectangle_ranks = np.repeat(np.asarray(valid_rectangle_ranks, dtype=np.int64), TEMPLATE_COUNT)
    templates = np.tile(np.arange(TEMPLATE_COUNT, dtype=np.int16), len(rectangles_array))
    total = len(rectangles_array) * TEMPLATE_COUNT

    raw_families = detector._wide_line_families(segments)
    family_maps = detector._distance_maps(raw_families, size)
    union_map = union_distance_map(segments, size)
    all_corners = []
    all_family_means = []
    all_union_means = []
    all_visibility = []
    all_camera_errors = []
    camera_scalar_recheck_count = 0
    camera_vector_scalar_max_abs_diff = 0.0
    valid_mask_parts = []
    batch_size = 1024
    for offset in range(0, len(rectangles_array), max(1, batch_size // TEMPLATE_COUNT)):
        batch_rectangles = rectangles_array[offset:offset + max(1, batch_size // TEMPLATE_COUNT)]
        homographies = (batch_rectangles[:, None] @ detector.TEMPLATE_TRANSFORMS).reshape(-1, 3, 3)
        family_corners, family_means, valid, family_visibility = geometry_and_support(
            homographies, family_maps, "family", size, detector,
        )
        union_corners, union_means, union_valid, union_visibility = geometry_and_support(
            homographies, union_map, "union", size, detector,
        )
        if not np.array_equal(valid, union_valid):
            raise AssertionError("family and union maps changed inherited geometry validity")
        if not np.allclose(family_corners, union_corners, atol=0, rtol=0):
            raise AssertionError("family and union maps changed candidate geometry")
        all_corners.append(family_corners.astype(np.float32))
        all_family_means.append(family_means)
        all_union_means.append(union_means)
        all_visibility.append(family_visibility)
        camera_errors, recheck_count, max_difference = camera_errors_with_frontier_recheck(
            homographies[valid], family_corners, size, native_scale, (width, height), camera_helper,
        )
        all_camera_errors.append(camera_errors.astype(np.float32))
        camera_scalar_recheck_count += recheck_count
        camera_vector_scalar_max_abs_diff = max(camera_vector_scalar_max_abs_diff, max_difference)
        valid_mask_parts.append(valid)
    # Geometry validity is identical for both maps and the records are packed in
    # rectangle/template order before invalid rows are removed per batch.
    valid_mask = np.concatenate(valid_mask_parts) if valid_mask_parts else np.empty(0, dtype=bool)
    # The support batches above contain only valid rows, so align provenance by
    # applying the same mask to the original candidate sequence batch by batch.
    packed_rectangle_ids = rectangle_ids[valid_mask]
    packed_rectangle_ranks = rectangle_ranks[valid_mask]
    packed_templates = templates[valid_mask]
    corners = np.concatenate(all_corners) if all_corners else np.empty((0, 4, 2), dtype=np.float32)
    family_means = np.concatenate(all_family_means) if all_family_means else np.empty((0, 2), dtype=np.float32)
    union_means = np.concatenate(all_union_means) if all_union_means else np.empty((0, 2), dtype=np.float32)
    visibility = np.concatenate(all_visibility) if all_visibility else np.empty((0, 2), dtype=np.int16)
    camera_errors = np.concatenate(all_camera_errors) if all_camera_errors else np.empty(0, dtype=np.float32)
    if len(corners) != int(valid_mask.sum()):
        raise AssertionError("valid geometry and support rows lost alignment")
    if len(camera_errors) != len(corners):
        raise AssertionError("camera diagnostic rows lost alignment")

    arrays_path = output / "arrays" / f"{case_id}.npz"
    save_arrays(
        arrays_path, packed_rectangle_ids, packed_rectangle_ranks, packed_templates,
        corners, family_means, union_means, visibility, camera_errors,
    )
    variants = {}
    for map_name, means in (("family_map", family_means), ("union_map", union_means)):
        variants[f"{map_name}_mean_directions"] = rank_variant(
            means.mean(axis=1), means, corners, visibility, packed_rectangle_ids,
            packed_rectangle_ranks, packed_templates, native_scale, camera_errors,
        )
        variants[f"{map_name}_minimum_directions"] = rank_variant(
            means.min(axis=1), means, corners, visibility, packed_rectangle_ids,
            packed_rectangle_ranks, packed_templates, native_scale, camera_errors,
        )
    finished = perf_counter()
    record = {
        "schema": "line-template-admission-audit/1",
        "case_id": case_id,
        "source_pack": pack_name,
        "input_digest": input_digest(source),
        "coordinate_semantics": {
            "input": "native cached segments",
            "working_size": list(size),
            "proposal_corners": "working pixels",
            "native_scale": native_scale.tolist(),
        },
        "settings": {
            "family_line_cap": 32,
            "wide_families": True,
            "min_supported_lines_preparation_only": 3,
            "support_distance": SUPPORT_DISTANCE,
            "samples_per_line": SAMPLES_PER_LINE,
            "template_count": TEMPLATE_COUNT,
            "rectangle_order": "frozen coverage VP round-robin order",
            "rectangle_order_settings": asdict(vp_settings),
            "global_rectangle_cap": RECTANGLE_CAP,
            "corner_diversity_radius": DIVERSITY_RADIUS,
            "camera_limit": CAMERA_LIMIT,
            "camera_gate": "frozen camera_diagnostic.camera error <= 0.1, applied before each support order",
            "geometry_validity": "finite positive-depth convex visible-span >= 0.15; rectangle area >= 100",
        },
        "contamination_check": {
            "references_loaded": False,
            "prior_controls_loaded": False,
            "automatic_rule_uses": ["dimensions", "cached fragments", "frozen VP ordering", "template geometry"],
        },
        "ordering": {
            "all_union_rectangle_count": int(len(selection.get("union_pair_product_ids", []))),
            "full_selected_rectangle_count": int(len(selection.get("selected_pair_product_ids", []))),
            "global_cap_count": int(len(selected_ids)),
            "area_retained_rectangle_count": int(len(rectangles_array)),
            "selected_rectangle_ids_first_cap": selected_ids.tolist(),
        },
        "generation": {
            "rectangle_template_hypotheses": int(total),
            "valid_geometry_hypotheses": int(len(corners)),
            "camera_scalar_recheck_count": int(camera_scalar_recheck_count),
            "camera_vector_scalar_max_abs_diff": float(camera_vector_scalar_max_abs_diff),
            "arrays": str(arrays_path.relative_to(output)),
            "elapsed_seconds": finished - started,
        },
        "variants": variants,
    }
    write_json_gz(output / "automatic" / f"{case_id}.json.gz", record)
    print(json.dumps({"case_id": case_id, "valid": len(corners), "seconds": round(finished - started, 2)}), flush=True)


def load_arrays(output: Path, case_id: str) -> dict[str, np.ndarray]:
    with np.load(output / "arrays" / f"{case_id}.npz") as data:
        return {name: data[name] for name in data.files}


def run_smoke(root: Path) -> None:
    """Exercise the raw-fragment map shape and one inherited geometry pass."""
    detector, *_ = load_helpers(root)
    segments = np.asarray(
        [[10, 10, 90, 10], [10, 90, 90, 90], [10, 10, 10, 90], [90, 10, 90, 90]],
        dtype=np.float64,
    )
    maps = detector._distance_maps((segments, segments), (100, 100))
    assert maps.shape == (2, 100, 100)
    image_corners = np.asarray([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(detector.CORNER_COURT_M.astype(np.float32), image_corners)
    corners, means, valid, visibility = geometry_and_support(
        homography[None], maps, "family", (100, 100), detector,
    )
    assert valid.shape == (1,) and len(corners) == len(means) == len(visibility)
    camera_helper = load_camera_helper(root)
    scalar_error = camera_helper.camera(image_corners, (100, 100))[0]
    vector_error = vector_camera_errors(homography[None], (100, 100))[0]
    assert abs(scalar_error - vector_error) < 1e-10
    print("synthetic_smoke_exit=0")


def diagnose_case(output: Path, target_path: Path, case_id: str, detector) -> None:
    record = read_json_gz(output / "automatic" / f"{case_id}.json.gz")
    target = json.loads(target_path.read_text())
    if target["case_id"] != case_id:
        raise ValueError(f"target is for {target['case_id']}, not {case_id}")
    arrays = load_arrays(output, case_id)
    target_working = np.asarray(target["corners_px"], dtype=float) / np.asarray(record["coordinate_semantics"]["native_scale"])
    exact_id = f"rectangle_{target['pair_product_id']}:template_{target['template_index']}"
    posthoc = {
        "schema": "line-template-admission-posthoc/1",
        "case_id": case_id,
        "automatic_output": f"automatic/{case_id}.json.gz",
        "target_source": target.get("source", "approved geometry diagnostic"),
        "target_proposal_id": exact_id,
        "target_corners_working": target_working.tolist(),
        "variants": {},
    }
    for variant_name, mode in (
        ("family_map_mean_directions", "family_means"),
        ("family_map_minimum_directions", "family_means"),
        ("union_map_mean_directions", "union_means"),
        ("union_map_minimum_directions", "union_means"),
    ):
        means = arrays[mode]
        scores = means.mean(axis=1) if "mean" in variant_name else means.min(axis=1)
        camera_eligible = arrays["camera_errors"] <= CAMERA_LIMIT
        eligible_indices = np.flatnonzero(camera_eligible)
        eligible_order = np.lexsort(
            (
                arrays["templates"][eligible_indices],
                arrays["rectangle_ids"][eligible_indices],
                -scores[eligible_indices],
            )
        )
        order = eligible_indices[eligible_order]
        selected, scanned = greedy_diverse(order, arrays["corners"], max(CAPS))
        target_matches = np.flatnonzero(
            (arrays["rectangle_ids"] == target["pair_product_id"])
            & (arrays["templates"] == target["template_index"])
        )
        target_rank = None
        target_diverse_position = None
        target_error = None
        target_camera_error = None
        if len(target_matches):
            target_index = int(target_matches[0])
            target_error = float(np.linalg.norm(arrays["corners"][target_index] - target_working, axis=1).max())
            target_camera_error = float(arrays["camera_errors"][target_index])
            target_order_position = np.flatnonzero(order == target_index)
            target_rank = int(target_order_position[0]) if len(target_order_position) else None
            selected_target = np.flatnonzero(selected == target_index)
            target_diverse_position = int(selected_target[0]) if len(selected_target) else None
        min_errors = np.linalg.norm(arrays["corners"][order] - target_working, axis=2).max(axis=1)
        closest_order_position = int(np.argmin(min_errors)) if len(min_errors) else None
        closest = int(order[closest_order_position]) if closest_order_position is not None else None
        caps = {}
        for cap in CAPS:
            prefix = selected[:cap]
            errors = np.linalg.norm(arrays["corners"][prefix] - target_working, axis=2).max(axis=1) if len(prefix) else np.empty(0)
            caps[str(cap)] = {
                "diverse_count": int(len(prefix)),
                "contains_exact_target": bool(target_index in prefix) if len(target_matches) else False,
                "closest_max_corner_error_working": float(errors.min()) if len(errors) else None,
            }
        posthoc["variants"][variant_name] = {
            "geometry_valid_count": int(len(scores)),
            "camera_eligible_count": int(len(order)),
            "scanned_for_256": int(scanned),
            "exact_target_present_in_geometry_pool": bool(len(target_matches)),
            "exact_target_camera_eligible": bool(
                len(target_matches) and target_camera_error <= CAMERA_LIMIT
            ),
            "exact_target_score_order_rank": target_rank,
            "exact_target_diverse_position": target_diverse_position,
            "exact_target_max_corner_error_working": target_error,
            "exact_target_camera_error": target_camera_error,
            "closest_pool_proposal_id": (
                f"rectangle_{int(arrays['rectangle_ids'][closest])}:template_{int(arrays['templates'][closest])}"
                if closest is not None else None
            ),
            "closest_pool_max_corner_error_working": (
                float(min_errors[closest_order_position])
                if closest_order_position is not None else None
            ),
            "caps": caps,
        }
    write_json_gz(output / "posthoc" / f"{case_id}.json.gz", posthoc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--root", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--case-id", choices=CASE_IDS, nargs="+", required=True)
    diagnose = subparsers.add_parser("diagnose")
    diagnose.add_argument("--output", type=Path, required=True)
    diagnose.add_argument("--target", type=Path, required=True)
    diagnose.add_argument("--case-id", required=True)
    diagnose.add_argument("--root", type=Path, required=True)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cv2.setNumThreads(1)
    if args.command == "smoke":
        run_smoke(args.root)
    elif args.command == "generate":
        detector, VPSettings, estimate, rectangle_population, select = load_helpers(args.root)
        for case_id in args.case_id:
            generate_case(args.root, args.output, case_id, detector, VPSettings, estimate, rectangle_population, select)
    else:
        detector, *_ = load_helpers(args.root)
        diagnose_case(args.output, args.target, args.case_id, detector)


if __name__ == "__main__":
    main()
