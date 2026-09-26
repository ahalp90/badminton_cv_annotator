"""Search pairs of line directions and collect the courts they propose."""

from __future__ import annotations

import os
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from functools import partial
from importlib import import_module
from itertools import permutations
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter, process_time
from types import ModuleType
from typing import NamedTuple

import cv2
import numpy as np

from . import line_observations as assignment

SCREEN_METHOD = "svd-family-residual/1"


class SearchInputs(NamedTuple):
    """What every pair's search in one view shares. Sent to each worker with its pair."""

    observations: assignment.Observations
    feet: np.ndarray  # (sampled frames, player slots, xy) working px; NaN where missing
    size: tuple[int, int]
    settings: object  # helpers.Settings
    upright_only: bool
    keep_per_pair: int
    capture_pool: bool


class PairSearch(NamedTuple):
    """One pair's kept courts, small enough to send back from a worker process."""

    role: dict  # propose_role's record
    raw_parent_count: int  # courts the pair proposed before retention
    retained: list  # the kept courts, best first
    details: list[dict]  # one per kept court: its candidate ID, pair ID and propose_role provenance
    elapsed_s: float
    pool_record: tuple | None  # this pair's write_pool record; None unless the pool is captured


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


def select(helpers: ModuleType, candidates: list, limit: int) -> list:
    """Keep the best distinct courts, up to limit, in the helpers' retention order."""
    if limit == helpers.KEEP_COURTS:
        return helpers.select_pool(candidates)
    selection = replace(helpers.detector.DEFAULT_SETTINGS, keep_candidates=limit, distinct_corner_distance=2.)
    return helpers.retain(candidates, selection)


def search_pair(helpers: ModuleType, inputs: SearchInputs, pair_id: int, pair_points: np.ndarray) -> PairSearch:
    """Propose one pair's courts and keep its best distinct ones.

    :param pair_points: (2 directions, homogeneous xyw) the pair's vanishing points, working px.
    """
    started = perf_counter()
    proposed = helpers.propose_role(pair_points, inputs.observations, inputs.feet, inputs.size, inputs.settings,
                                    upright_only=inputs.upright_only)
    retained = select(helpers, proposed.candidates, inputs.keep_per_pair)
    # id() keys only hold inside this process, so positions go back to the parent instead.
    proposed_positions = {id(candidate): position for position, candidate in enumerate(proposed.candidates)}
    positions = [proposed_positions[id(candidate)] for candidate in retained]
    details = []
    for position in positions:
        details.append({"candidate_id": f"{pair_id}:{position}", "pair_id": pair_id, **proposed.detail(position)})
    pool_record = None
    if inputs.capture_pool:
        proposed_corners = np.asarray([candidate.corners_px for candidate in proposed.candidates],
                                      dtype=np.float32).reshape(-1, 4, 2)
        pool_record = (pair_id, len(proposed.candidates), proposed_corners, np.asarray(positions, dtype=np.int32),
                       (proposed.combined_corners, proposed.valid, proposed.usable,
                        proposed.player_any, proposed.player_both_halves))
    return PairSearch(proposed.record, len(proposed.candidates), retained, details, perf_counter() - started,
                      pool_record)


def search_pair_in_worker(helpers_name: str, inputs: SearchInputs, pair_id: int,
                          pair_points: np.ndarray) -> PairSearch:
    """search_pair in a worker process, which imports its own copy of the helpers module."""
    return search_pair(import_module(helpers_name), inputs, pair_id, pair_points)


def search_pairs(helpers: ModuleType, inputs: SearchInputs, pairs: list[tuple[int, np.ndarray]],
                 workers: int) -> Iterator[PairSearch]:
    """Search each (pair ID, pair points) and yield the results in the order of pairs.

    Several workers search in fresh spawned processes. Each imports helpers by module name,
    so changes made to the helpers object in this process never reach them. A worker's
    exception is raised here, and pairs still waiting are cancelled.
    """
    if workers == 1 or not pairs:
        for pair_id, pair_points in pairs:
            yield search_pair(helpers, inputs, pair_id, pair_points)
        return
    pair_ids = [pair_id for pair_id, _ in pairs]
    pair_points = [points for _, points in pairs]
    # A spawned process starts with OpenCV's default thread count, so pass this process's on.
    with ProcessPoolExecutor(min(workers, len(pairs)), mp_context=get_context("spawn"),
                             initializer=cv2.setNumThreads, initargs=(cv2.getNumThreads(),)) as executor:
        yield from executor.map(partial(search_pair_in_worker, helpers.__name__, inputs), pair_ids, pair_points)


def cpu_seconds() -> float:
    """CPU time of this process plus its finished child processes, such as search workers."""
    times = os.times()
    return process_time() + times.children_user + times.children_system


def generate(source: dict, saved: dict, zone: object, root: Path, helpers: ModuleType,
             direction_budget: int = 12, pool_path: Path | None = None, *,
             keep_axes: int = 512, keep_per_pair: int = 256, keep_global: int = 256,
             max_matched_pairs: int | None = None, legacy_evidence: bool = True,
             max_horizon_tilt_deg: float | None = None, workers: int = 1) -> dict:
    """Generate courts from original directions, screening pairs before matcher work.

    :param legacy_evidence: Also score each entry's stripes and paint profile and pick the
        two research winners from them. Off leaves those keys out and both winner IDs None.
    :param max_horizon_tilt_deg: Skip pairs whose horizon tilts more than this, and drop
        courts above their pair's horizon. Both need a camera turned on its side or upside
        down. None keeps every pair and court.
    :param workers: Processes that search the eligible pairs. Above 1, each worker imports
        helpers by module name (see search_pairs). The record is the same apart from its
        timings; cpu_s then includes the workers' CPU time.
    """
    started = perf_counter()
    cpu_started = cpu_seconds()
    if min(keep_axes, keep_per_pair, keep_global) <= 0:
        raise ValueError("candidate caps must be positive")
    if max_matched_pairs is not None and max_matched_pairs <= 0:
        raise ValueError("max_matched_pairs must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
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

    # Every skip decision is made here, in pair order, before any search starts.
    pair_records = []
    eligible = []  # (pair ID, pencil IDs, pair points, pair record) per pair to search, in pair order
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
        if max_horizon_tilt_deg is not None:
            tilt = helpers.horizon_tilt_deg(pair_points, size)
            record["horizon_tilt_deg"] = tilt
            if tilt is not None and tilt > max_horizon_tilt_deg:
                pair_records.append({**record, "status": "horizon_tilt"})
                continue
        if max_matched_pairs is not None and matched_pairs >= max_matched_pairs:
            pair_records.append({**record, "status": "smoke_pair_limit"})
            continue
        matched_pairs += 1
        pair_records.append(record)  # completed below, once the pair is searched
        eligible.append((pair_id, pencil_ids, pair_points, record))

    inputs = SearchInputs(observations, feet, size, settings, max_horizon_tilt_deg is not None, keep_per_pair,
                          pool_path is not None)
    searches = search_pairs(helpers, inputs, [(pair_id, pair_points) for pair_id, _, pair_points, _ in eligible],
                            workers)
    # Global retention breaks score ties by pool order, so the pool grows in pair order.
    pooled, provenance, pool_records = [], {}, []
    for (pair_id, pencil_ids, _, record), searched in zip(eligible, searches, strict=True):
        shortlist = []
        for candidate, details in zip(searched.retained, searched.details, strict=True):
            provenance[id(candidate)] = details
            shortlist.append({**details, "corners_px": (candidate.corners_px * scale).tolist(),
                              "shortlist_score": candidate.score})
        pooled.extend(searched.retained)
        if searched.pool_record is not None:
            pool_records.append(searched.pool_record)
        record.update({"status": "matched", "role": searched.role, "raw_parent_count": searched.raw_parent_count,
                       "per_pair_cap_reached": len(searched.retained) == keep_per_pair, "shortlist": shortlist,
                       "elapsed_s": searched.elapsed_s})
        print(source["id"], "pair", pair_id, list(pencil_ids), "combined", searched.role.get("combined", 0),
              "players", searched.raw_parent_count, "retained", len(searched.retained), "seconds", searched.elapsed_s,
              flush=True)
    retained = select(helpers, pooled, keep_global)
    shortlist = []
    for candidate in retained:
        shortlist.append({**provenance[id(candidate)], "corners_px": (candidate.corners_px * scale).tolist(),
                          "shortlist_score": candidate.score})
    if legacy_evidence:
        entries = helpers.evaluate_pool(source, shortlist, observations, size, segments, families, zone, root)
        line_id, paint_id = helpers.winner_ids(entries)
    else:
        entries = helpers.evaluate_pool(source, shortlist, observations, size, segments, families, zone, root,
                                        legacy_evidence=False)
        line_id = paint_id = None
    if pool_path is not None:
        helpers.write_pool(pool_path, pool_records)
    # Off, the record matches the unfiltered runs' records key for key.
    upright = {} if max_horizon_tilt_deg is None else {"max_horizon_tilt_deg": max_horizon_tilt_deg}
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
            "elapsed_s": perf_counter() - started, "cpu_s": cpu_seconds() - cpu_started,
            "global_cap_reached": len(retained) == keep_global, **upright}
