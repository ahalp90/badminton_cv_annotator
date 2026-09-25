"""Shortlist the frozen VP population by continuous marking support without references."""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
from run_diagnosis import attach_metrics, read, run_case, write
from run_population import prepare
from vp_pruning import rectangle_population

from experiments.annotator.independent_court import assignment, detector

KEEP = 128
DIVERSITY_PX = 2.0


def continuous_support(homographies: np.ndarray, maps: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Score finite visible intervals smoothly while counting the split centre once."""
    projected, _ = detector.project(homographies, detector.SEGMENTS_M)
    endpoints = projected.reshape(-1, 12, 2, 2)
    lower, upper, visible = detector._visible_fractions(endpoints, size)
    fractions = lower[..., None] + (upper - lower)[..., None] * np.linspace(0, 1, 64)
    # Same arithmetic as detector._visible_samples, with x and y as separate arrays: numpy is
    # much slower on a trailing axis of length 2. The scores are bit-identical.
    start_x, start_y = endpoints[:, :, 0, 0, None], endpoints[:, :, 0, 1, None]
    end_x, end_y = endpoints[:, :, 1, 0, None], endpoints[:, :, 1, 1, None]
    pixel_x = np.clip(start_x + fractions * (end_x - start_x), 0, size[0] - 1).astype(int)
    pixel_y = np.clip(start_y + fractions * (end_y - start_y), 0, size[1] - 1).astype(int)
    family = np.repeat([0, 1], 6)[None, :, None]
    distance = maps[family, pixel_y, pixel_x]
    response = np.exp(-.5 * np.square(distance / assignment.DISTANCE_SIGMA_PX)).mean(axis=2)
    response *= visible
    per_marking, marking_visible = [], []
    for intervals in assignment.MARKING_INTERVALS:
        count = visible[:, intervals].sum(axis=1)
        per_marking.append(response[:, intervals].sum(axis=1) / np.maximum(count, 1))
        marking_visible.append(count > 0)
    return np.sum(per_marking, axis=0) / np.maximum(np.sum(marking_visible, axis=0), 1)


def geometry(homographies: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Use the detector's original positive-depth, convexity and visible-span conditions."""
    corners, denominator = detector.project(homographies, detector.CORNER_COURT_M)
    edges = np.roll(corners, -1, axis=1) - corners
    turns = edges[..., 0] * np.roll(edges[..., 1], -1, axis=1) - edges[..., 1] * np.roll(edges[..., 0], -1, axis=1)
    span = np.minimum(corners.max(axis=1), np.asarray(size) - 1) - np.maximum(corners.min(axis=1), 0)
    valid = (np.isfinite(corners).all(axis=(1, 2)) & np.all(denominator > 1e-6, axis=1)
             & np.all(turns > 0, axis=1)
             & np.all(span / size >= detector.DEFAULT_SETTINGS.min_visible_span_fraction, axis=1))
    return valid, corners


def retain(candidates: list[detector.Candidate], settings: detector.Settings) -> list[detector.Candidate]:
    """Vectorise distances while preserving the existing greedy retention order."""
    retained = []
    corners = np.empty((settings.keep_candidates, 4, 2))
    for candidate in sorted(candidates, key=lambda item: -item.score):
        separation = np.linalg.norm(corners[:len(retained)] - candidate.corners_px, axis=2).max(axis=1)
        if np.any(separation <= settings.distinct_corner_distance):
            continue
        corners[len(retained)] = candidate.corners_px
        retained.append(candidate)
        if len(retained) == settings.keep_candidates:
            break
    return retained


def scan(source: dict, saved: dict, zone: Any) -> tuple[list[dict], dict]:
    segments, families, size = prepare(source)
    scale = np.array([source['dimensions']['width'], source['dimensions']['height']]) / size
    quads, _, pair_ids = rectangle_population(families, size)
    selected_ids = np.asarray(saved['seed_selection']['retained_pair_product_ids'])
    indices = np.searchsorted(pair_ids, selected_ids)
    np.testing.assert_array_equal(pair_ids[indices], selected_ids)
    rectangles = np.asarray([cv2.getPerspectiveTransform(detector.UNIT_CORNERS, quad.astype(np.float32))
                             for quad in quads[indices]])
    maps = detector._distance_maps(detector._wide_line_families(segments), size)
    feet = np.asarray([[[np.nan, np.nan] if foot is None else foot for foot in frame]
                       for frame in source['all_feet_px']], dtype=float) / scale
    counts = {'generated': 0, 'geometry': 0, 'geometry_and_players': 0}
    retained: list[detector.Candidate] = []
    provenance = {}
    retention = replace(detector.DEFAULT_SETTINGS, keep_candidates=KEEP, distinct_corner_distance=DIVERSITY_PX)
    started = perf_counter()
    for offset in range(0, len(rectangles), 8):
        homographies = (rectangles[offset:offset + 8, None] @ detector.TEMPLATE_TRANSFORMS).reshape(-1, 3, 3)
        valid, corners = geometry(homographies, size)
        one, two = zone.player_fractions(homographies, feet)
        usable = valid & (one == 1) & (two >= .5)
        counts['generated'] += len(homographies)
        counts['geometry'] += int(valid.sum())
        counts['geometry_and_players'] += int(usable.sum())
        if not usable.any():
            continue
        generations = offset * len(detector.TEMPLATE_TRANSFORMS) + np.flatnonzero(usable)
        values = continuous_support(homographies[usable], maps, size)
        coordinates = corners[usable]
        if len(retained) == KEEP and values.max() < retained[-1].score:
            continue
        proposed = []
        for index in np.argsort(-values, kind='stable'):
            candidate = detector.Candidate(coordinates[index], float(values[index]), (0., 0.), (0, 0))
            proposed.append(candidate)
            provenance[id(candidate)] = int(generations[index])
        retained = retain(retained + proposed, retention)
        provenance = {id(candidate): provenance[id(candidate)] for candidate in retained}
    assert counts['generated'] == saved['final']['generated']
    assert counts['geometry'] == saved['counts']['geometry_before_players']
    assert counts['geometry_and_players'] == saved['counts']['players_and_geometry']
    candidates = [{'id': f'generation_{provenance[id(candidate)]}', 'pool': 'automatic_continuous_shortlist',
                   'generation_id': provenance[id(candidate)], 'coarse_score': candidate.score,
                   'corners_px': (candidate.corners_px * scale).tolist()} for candidate in retained]
    return candidates, {'counts': counts, 'keep': KEEP, 'diversity_working_px': DIVERSITY_PX,
                        'elapsed_s': perf_counter() - started, 'labels_used': False,
                        'score': 'equal finite-marking mean Gaussian family-map support, sigma=2 working px'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--saved', type=Path, required=True)
    parser.add_argument('--legacy', type=Path, required=True)
    parser.add_argument('--ids', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.legacy.resolve()))
    zone = importlib.import_module('zone_net')
    cv2.setNumThreads(1)
    packed = read(args.inputs)
    sources = {source['id']: source for source in packed['cases']}
    for case_id in args.ids:
        source = sources[case_id]
        saved = read(args.saved / f'{case_id}.json.gz')
        candidates, accounting = scan(source, saved, zone)
        write(args.output / f'{case_id}.shortlist.json.gz', {'case_id': case_id, **accounting, 'entries': candidates})
        print(case_id, accounting, flush=True)
        result = run_case(source, saved, zone, candidates)
        result['shortlist'] = accounting
        attach_metrics(result, packed['references'][case_id], source)
        write(args.output / f'{case_id}.json.gz', result)
        print(case_id, 'refits complete', result['elapsed_s'], flush=True)


if __name__ == '__main__':
    main()
