"""Trace frozen court markings and compare fixed-identity refits on cached observations."""

from __future__ import annotations

import argparse
import gzip
import importlib
import json
import sys
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'vp_pruning'))
from diagnose_targets import read
from run_population import prepare
from vp_pruning import rectangle_population

from experiments.annotator.independent_court import assignment, detector
from experiments.annotator.independent_court import fixed_stripe_refit as fitting
from experiments.annotator.independent_court import stripe_observations as stripes


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(value, allow_nan=False).encode(), mtime=0))


def finite_profiles(homography: np.ndarray, observations: assignment.Observations, size: tuple[int, int]) -> list[dict]:
    """Retain visible-interval distances, including unsupported parts of each marking."""
    projected, _ = detector.project(homography[None], detector.SEGMENTS_M)
    intervals = projected.reshape(12, 2, 2)
    samples, visible = detector._visible_samples(intervals[None], size, assignment.MARKING_SAMPLES)
    profiles = []
    for interval, segment in enumerate(intervals):
        if not visible[0, interval]:
            profiles.append({'interval': interval, 'visible': False})
            continue
        direction = segment[1] - segment[0]
        direction /= np.linalg.norm(direction)
        compatible = np.abs(observations.directions @ direction) >= np.cos(np.deg2rad(assignment.MATCH_ANGLE_DEG))
        observed_ids = np.flatnonzero(compatible)
        if not len(observed_ids):
            profiles.append({'interval': interval, 'visible': True, 'compatible_fragments': 0})
            continue
        points = samples[0, interval]
        distances = assignment.distances_to_segments(points, observations.segments[observed_ids])
        nearest = distances.argmin(axis=1)
        matched = observations.segments[observed_ids[nearest]]
        vectors = matched[:, 1] - matched[:, 0]
        fraction = np.einsum('pd,pd->p', points - matched[:, 0], vectors) / np.square(vectors).sum(axis=1)
        closest = matched[:, 0] + np.clip(fraction, 0, 1)[:, None] * vectors
        normal = np.array([-direction[1], direction[0]])
        profiles.append({'interval': interval, 'visible': True, 'compatible_fragments': len(observed_ids),
                         'sample_xy_working': points.tolist(),
                         'distance_working_px': distances[np.arange(len(points)), nearest].tolist(),
                         'signed_normal_working_px': ((closest - points) @ normal).tolist(),
                         'nearest_raw_ids': observations.fragment_ids[observed_ids[nearest]].tolist()})
    return profiles


def gate_evidence(
    corners: np.ndarray, source: dict, scale: np.ndarray, size: tuple[int, int], families: tuple,
    maps: np.ndarray, zone: Any,
) -> dict:
    """Measure original gates independently; no retained-pool acceptance is inferred."""
    homography = cv2.getPerspectiveTransform(detector.CORNER_COURT_M, (corners / scale).astype(np.float32))
    feet = np.asarray([[[np.nan, np.nan] if foot is None else foot for foot in frame]
                       for frame in source['all_feet_px']], dtype=float) / scale
    one, two = zone.player_fractions(homography[None], feet)
    projected, scores, means, counts = detector._score(
        homography[None], maps, detector.Settings(wide_families=True, min_supported_lines=3), families,
    )
    if not len(projected):
        return {'geometry_valid': False, 'player_fractions': [float(one[0]), float(two[0])],
                'floor_score': None, 'camera_error': None}
    _, camera_error, _ = zone.net_segments(corners, (source['dimensions']['width'], source['dimensions']['height']))
    return {'geometry_valid': True, 'player_fractions': [float(one[0]), float(two[0])],
            'floor_score': float(scores[0]), 'family_support': means[0].tolist(), 'line_counts': counts[0].tolist(),
            'camera_error': float(camera_error) if np.isfinite(camera_error) else None}


def seed_provenance(generation: int, saved: dict, population: tuple, families: tuple, scale: np.ndarray) -> dict:
    """Decode the exact four merged lines and four template coordinates of a saved seed."""
    rectangle_index, template_index = divmod(generation, len(detector.TEMPLATE_TRANSFORMS))
    pair_id = saved['seed_selection']['retained_pair_product_ids'][rectangle_index]
    quads, ranks, pair_ids = population
    index = int(np.searchsorted(pair_ids, pair_id))
    assert pair_ids[index] == pair_id
    rectangle = cv2.getPerspectiveTransform(detector.UNIT_CORNERS, quads[index].astype(np.float32))
    homography = rectangle @ detector.TEMPLATE_TRANSFORMS[template_index]
    corners, _ = detector.project(homography[None], detector.CORNER_COURT_M)
    x_pairs, y_pairs = list(combinations(detector.X_COORDS, 2)), list(combinations(detector.Y_COORDS, 2))
    x_index, y_index = divmod(template_index, len(y_pairs))
    line_ranks = ranks[index]
    return {'generation_id': generation, 'pair_product_id': pair_id, 'template_index': template_index,
            'axis_coordinates_m': [list(x_pairs[x_index]), list(y_pairs[y_index])],
            'line_ranks': line_ranks.tolist(),
            'lines_working': np.concatenate([families[0][line_ranks[:2]], families[1][line_ranks[2:]]]).tolist(),
            'reconstructed_corners_px': (corners[0] * scale).tolist()}


def entries(saved: dict) -> list[dict]:
    """Preserve automatic retained courts separately from reference-selected diagnostics."""
    result = [{'id': f'retained_{index}', 'pool': 'automatic_retained', 'retained_index': index,
               'corners_px': row['corners_px']} for index, row in enumerate(saved['final']['candidates'])]
    nearest = saved['diagnostic_rankings']['geometry_before_players']['max_corner_native_px'][0]['generation_id']
    result.extend({'id': f"generation_{row['generation_id']}", 'pool': 'saved_diagnostic',
                   'generation_id': row['generation_id'], 'inspected_closest': row['generation_id'] == nearest,
                   'corners_px': row['corners_px']} for row in saved['examples'])
    return result


def run_case(source: dict, saved: dict, zone: Any, candidates: list[dict] | None = None) -> dict:
    started = perf_counter()
    segments, families, size = prepare(source)
    scale = np.array([source['dimensions']['width'], source['dimensions']['height']]) / size
    observations = assignment.prepare_observations(segments, size)
    weights = stripes.fragment_weights(observations)
    maps = detector._distance_maps(detector._wide_line_families(segments), size)
    population = rectangle_population(families, size)
    results = []
    for entry in entries(saved) if candidates is None else candidates:
        corners = np.asarray(entry['corners_px'])
        if 'generation_id' in entry:
            provenance = seed_provenance(entry['generation_id'], saved, population, families, scale)
            np.testing.assert_allclose(provenance.pop('reconstructed_corners_px'), corners, rtol=0, atol=1e-5)
            compatible_raw = []
            for line in np.asarray(provenance['lines_working']):
                distance = np.abs(observations.segments @ line[:2] + line[2]).max(axis=1)
                angle = np.abs(observations.directions @ line[:2])
                mask = (distance <= detector.DEFAULT_SETTINGS.merge_distance) & (
                    angle <= np.sin(np.deg2rad(detector.DEFAULT_SETTINGS.merge_angle_deg)))
                compatible_raw.append(observations.fragment_ids[mask].tolist())
            provenance['compatible_raw_ids'] = compatible_raw
            entry['seed'] = provenance
        homography = cv2.getPerspectiveTransform(detector.CORNER_COURT_M, (corners / scale).astype(np.float32))
        evidence = stripes.measure(homography, observations, size)
        stripe = stripes.score_model(evidence, weights, 3)
        constraints = fitting.prepare(homography, observations, stripe['assignments'], weights)
        entry.update({'stripe': stripe, 'profiles': finite_profiles(homography, observations, size),
                      'gates': gate_evidence(corners, source, scale, size, families, maps, zone),
                      'constraints': {name: getattr(constraints, name).tolist() for name in
                                      ('points', 'intervals', 'positions', 'weights', 'fragment_ids', 'sample_ids')},
                      'fits': {}})
        for model, use_positions in [('nominal_centre', False), ('fixed_position', True)]:
            fit = fitting.refine(corners / scale, constraints, size, use_positions)
            if fit['corners_px'] is not None:
                fit['corners_px'] = (np.asarray(fit['corners_px']) * scale).tolist()
            if fit['successful']:
                fitted = np.asarray(fit['corners_px'])
                changed = cv2.getPerspectiveTransform(detector.CORNER_COURT_M, (fitted / scale).astype(np.float32))
                renewed = stripes.measure(changed, observations, size)
                fit['stripe_fixed'] = stripes.score_model(renewed, weights, 3, stripe['assignments'])
                fit['stripe_reassigned'] = stripes.score_model(renewed, weights, 3)
                fit['profiles'] = finite_profiles(changed, observations, size)
                fit['gates'] = gate_evidence(fitted, source, scale, size, families, maps, zone)
            entry['fits'][model] = fit
        results.append(entry)
    return {'schema': 'fixed-marking-diagnosis/1', 'case_id': source['id'], 'working_size': size,
            'native_scale': scale.tolist(), 'markings': assignment.MARKINGS,
            'observation_raw_ids': observations.fragment_ids.tolist(), 'entries': results,
            'elapsed_s': perf_counter() - started}


def attach_metrics(record: dict, reference: dict, source: dict) -> None:
    """Attach reference-derived diagnostics only after all fitting and evidence measurement."""
    scale = np.array([1280 / source['dimensions']['width'], 720 / source['dimensions']['height']])
    for entry in record['entries']:
        for variant in [entry, *entry['fits'].values()]:
            if variant['corners_px'] is not None:
                errors = np.linalg.norm((np.asarray(variant['corners_px']) - reference['corners_px']) * scale, axis=1)
                variant['max_corner_1280_px'] = float(errors.max())
    record['reference_metrics_are_diagnostic'] = True


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
        record = run_case(source, saved, zone)
        attach_metrics(record, packed['references'][case_id], source)
        write(args.output / f'{case_id}.json.gz', record)
        closest = next(row for row in record['entries'] if row.get('inspected_closest'))
        print(case_id, 'entries', len(record['entries']), 'closest before', closest['max_corner_1280_px'],
              'after', {model: (fit['status'], fit.get('max_corner_1280_px'))
                        for model, fit in closest['fits'].items()}, 'seconds', record['elapsed_s'], flush=True)


if __name__ == '__main__':
    main()
