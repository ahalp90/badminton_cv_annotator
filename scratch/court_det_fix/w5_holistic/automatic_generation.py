"""Live automatic direction generation with a bounded SVD family screen."""

from __future__ import annotations

from dataclasses import asdict, replace
from itertools import permutations
from pathlib import Path
from time import perf_counter, process_time
from types import ModuleType

import numpy as np

from experiments.annotator.independent_court import assignment

SCREEN_METHOD = "svd-family-residual/1"


def screen_groups(estimator: dict, budget: int) -> dict:
    """Rank original support groups while retaining their original point identities."""
    if budget not in (12, 16):
        raise ValueError(f"direction budget must be 12 or 16, got {budget}")
    points = np.asarray(estimator["points_working"], dtype=np.float64).reshape(-1, 3)
    count = len(points)
    if count > 16:
        raise ValueError(f"SVD screen supports at most 16 families, got {count}")
    masks = np.asarray(estimator["retained_support_masks"], dtype=bool)
    lines = np.asarray(estimator["direction_lines"], dtype=np.float64)
    if count == 0 and masks.size == 0:
        masks = np.empty((0, len(lines)), dtype=bool)
    if count == 0 and lines.size == 0:
        lines = np.empty((0, 3), dtype=np.float64)
    transform = np.asarray(estimator["normalised_to_working"], dtype=np.float64)
    if points.shape != (count, 3) or lines.ndim != 2 or lines.shape[1] != 3:
        raise ValueError("Invalid estimator point or line shape")
    if transform.shape != (3, 3) or masks.shape != (count, len(lines)):
        raise ValueError("Invalid estimator transform or support-mask shape")
    if not np.isfinite(points).all() or not np.isfinite(lines).all() or not np.isfinite(transform).all():
        raise ValueError("Non-finite estimator geometry")

    if budget == 16:
        return {"method": SCREEN_METHOD, "budget": budget, "ranked_original_ids": list(range(count)),
                "selected_original_ids": list(range(count)), "diagnostics": []}

    normalised_lines = lines @ transform
    diagnostics = []
    for group_index, mask in enumerate(masks):
        support = normalised_lines[mask]
        if len(support) < 2:
            raise ValueError(f"Group {group_index} has fewer than two support lines")
        normals = np.linalg.norm(support[:, :2], axis=1)
        if not np.isfinite(support).all() or (normals <= 0).any():
            raise ValueError(f"Group {group_index} has invalid normalised lines")
        support = support / normals[:, None]
        _, _, right = np.linalg.svd(support, full_matrices=True)
        point = right[-1]
        diagnostics.append({"group_index": group_index, "line_count": len(support),
                            "algebraic_rms": float(np.sqrt(np.mean(np.square(support @ point))))})
    ranked = sorted(diagnostics, key=lambda row: (row["algebraic_rms"], -row["line_count"], row["group_index"]))
    ranked_ids = [row["group_index"] for row in ranked]
    return {"method": SCREEN_METHOD, "budget": budget, "ranked_original_ids": ranked_ids,
            "selected_original_ids": ranked_ids[:min(budget, count)], "diagnostics": diagnostics}


def generate(source: dict, saved: dict, zone: object, root: Path, helpers: ModuleType,
             direction_budget: int = 12, pool_path: Path | None = None, *,
             keep_axes: int = 512, keep_per_pair: int = 256, keep_global: int = 256,
             max_matched_pairs: int | None = None) -> dict:
    """Generate courts from original directions, screening pairs before matcher work."""
    started = perf_counter()
    cpu_started = process_time()
    if min(keep_axes, keep_per_pair, keep_global) <= 0:
        raise ValueError("candidate caps must be positive")
    if max_matched_pairs is not None and max_matched_pairs <= 0:
        raise ValueError("max_matched_pairs must be positive")
    segments, families, size = helpers.prepare(source)
    assert list(size) == saved["working_size"]
    assert saved["settings"]["pencil_selection"] == "coverage"
    estimator = saved["estimator"]
    screen = screen_groups(estimator, direction_budget)
    points = np.asarray(estimator["points_working"])
    selected = set(screen["selected_original_ids"])
    native_size = (source["dimensions"]["width"], source["dimensions"]["height"])
    scale = np.asarray(native_size) / size
    point_scale = np.append(scale, 1.)
    feet = np.asarray([[[np.nan, np.nan] if foot is None else foot for foot in frame]
                       for frame in source["all_feet_px"]], dtype=float) / scale
    observations = assignment.prepare_observations(segments, size)
    settings = helpers.Settings(keep_axes=keep_axes)
    def select(candidates: list, limit: int) -> list:
        if limit == helpers.KEEP_COURTS:
            return helpers.select_pool(candidates)
        selection = replace(helpers.detector.DEFAULT_SETTINGS, keep_candidates=limit, distinct_corner_distance=2.)
        return helpers.retain(candidates, selection)

    pooled, provenance, pair_records = [], {}, []
    pool_records = []
    matched_pairs = 0
    for pair_id, pencil_ids in enumerate(permutations(range(len(points)), 2)):
        record = {"pair_id": pair_id, "pencils": list(pencil_ids)}
        if not set(pencil_ids) <= selected:
            pair_records.append({**record, "status": "skipped_svd_mask"})
            continue
        pair_points = points[list(pencil_ids)]
        bound = helpers.camera_direction_bound(pair_points * point_scale, native_size)
        record["camera_direction_bound"] = bound
        if bound > helpers.CAMERA_ERROR_LIMIT + helpers.CAMERA_ROUNDING_MARGIN:
            pair_records.append({**record, "status": "camera_direction_bound"})
            continue
        if max_matched_pairs is not None and matched_pairs >= max_matched_pairs:
            pair_records.append({**record, "status": "smoke_pair_limit"})
            continue
        matched_pairs += 1
        pair_start = perf_counter()
        proposed = helpers.propose_role(pair_points, observations, feet, size, settings)
        retained = select(proposed.candidates, keep_per_pair)
        proposed_positions = {id(candidate): position for position, candidate in enumerate(proposed.candidates)}
        if pool_path is not None:
            proposed_corners = np.asarray([candidate.corners_px for candidate in proposed.candidates],
                                          dtype=np.float32).reshape(-1, 4, 2)
            pool_records.append((pair_id, len(proposed.candidates), proposed_corners,
                                 np.asarray([proposed_positions[id(candidate)] for candidate in retained], dtype=np.int32),
                                 (proposed.combined_corners, proposed.valid, proposed.usable,
                                  proposed.player_any, proposed.player_both_halves)))
        shortlist = []
        for candidate in retained:
            position = proposed_positions[id(candidate)]
            details = {"candidate_id": f"{pair_id}:{position}", "pair_id": pair_id, **proposed.detail(position)}
            provenance[id(candidate)] = details
            shortlist.append({**details, "corners_px": (candidate.corners_px * scale).tolist(),
                              "shortlist_score": candidate.score})
        pooled.extend(retained)
        record.update({"status": "matched", "role": proposed.record, "raw_parent_count": len(proposed.candidates),
                       "per_pair_cap_reached": len(retained) == keep_per_pair, "shortlist": shortlist,
                       "elapsed_s": perf_counter() - pair_start})
        pair_records.append(record)
        print(source["id"], "pair", pair_id, list(pencil_ids), "combined", proposed.record.get("combined", 0),
              "players", len(proposed.candidates), "retained", len(retained), "seconds", record["elapsed_s"], flush=True)
    retained = select(pooled, keep_global)
    shortlist = []
    for candidate in retained:
        shortlist.append({**provenance[id(candidate)], "corners_px": (candidate.corners_px * scale).tolist(),
                          "shortlist_score": candidate.score})
    entries = helpers.evaluate_pool(source, shortlist, observations, size, segments, families, zone, root)
    line_id, paint_id = helpers.winner_ids(entries)
    if pool_path is not None:
        helpers.write_pool(pool_path, pool_records)
    return {"schema": "automatic-directions-axis-matching/1", "case_id": source["id"],
            "automatic_directions": True, "label_guided_generation": False, "emission_decision": None,
            "working_size": size, "settings": asdict(settings), "keep_per_pair": keep_per_pair,
            "keep_global": keep_global, "estimator_settings": saved["settings"], "estimator": estimator,
            "direction_screen": screen,
            "max_matched_pairs": max_matched_pairs,
            "camera_error_limit": helpers.CAMERA_ERROR_LIMIT, "camera_rounding_margin": helpers.CAMERA_ROUNDING_MARGIN,
            "camera_bound_coordinate_space": "native",
            "pairs": pair_records, "pooled_candidates": len(pooled), "entries": entries,
            "raw_groups": [observations.fragment_ids[group].tolist() for group in observations.groups],
            "line_winner_id": line_id, "paint_winner_id": paint_id,
            "elapsed_s": perf_counter() - started, "cpu_s": process_time() - cpu_started,
            "global_cap_reached": len(retained) == keep_global}
