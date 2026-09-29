"""Locate fixed known-seed losses after the automatic population runs."""

from __future__ import annotations

import argparse
import gzip
import json
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from run_population import prepare
from vp_pruning import angular_residuals, rectangle_population

from experiments.annotator.independent_court import detector


def read(path: Path) -> dict:
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def diagnose(source: dict, target: dict, record: dict) -> dict:
    """Assess fixed target geometry without changing any automatic hypothesis."""
    _, families, size = prepare(source)
    estimation = record['estimator']
    transform = np.asarray(estimation['normalised_to_working'])
    lines = np.asarray(estimation['direction_lines']) @ transform
    pairs = np.asarray(list(combinations(range(len(lines)), 2)))
    intersections = np.cross(lines[pairs[:, 0]], lines[pairs[:, 1]])
    infinity = np.column_stack((lines[:, 1], -lines[:, 0], np.zeros(len(lines))))
    candidates = np.concatenate((intersections, infinity))
    ids = np.asarray(estimation['candidate_ids'])
    candidates = candidates[ids]
    candidates /= np.linalg.norm(candidates, axis=1)[:, None]
    direction_records = []
    for family, ranks in zip(families, (target['x_line_ranks'], target['y_line_ranks']), strict=True):
        residuals = angular_residuals(family[ranks] @ transform, candidates).max(axis=1)
        compatible = residuals <= record['settings']['angle_deg']
        best = int(np.argmin(residuals))
        retained = np.isin(ids, estimation['retained_candidate_ids'])
        states, counts = np.unique(np.asarray(estimation['candidate_status'])[compatible], return_counts=True)
        direction_records.append({
            'target_ranks': ranks, 'compatible_candidates': int(compatible.sum()),
            'compatible_retained': int((compatible & retained).sum()),
            'compatible_candidate_status': dict(zip(states.tolist(), counts.tolist(), strict=True)),
            'minimum_max_angle_deg': float(residuals[best]), 'closest_candidate_id': int(ids[best]),
            'closest_candidate_support_count': estimation['support_counts'][best],
        })
    selection = record['seed_selection']
    target_id = target['pair_product_id']
    quads, _, seed_ids = rectangle_population(families, size)
    quad = quads[np.flatnonzero(seed_ids == target_id)[0]]
    rectangle = cv2.getPerspectiveTransform(detector.UNIT_CORNERS, quad.astype(np.float32))
    homography = rectangle @ detector.TEMPLATE_TRANSFORMS[target['template_index']]
    corners, _ = detector.project(homography[None], detector.CORNER_COURT_M)
    native_scale = np.array([source['dimensions']['width'], source['dimensions']['height']]) / size
    native_corners = corners[0] * native_scale
    np.testing.assert_allclose(native_corners, target['corners_px'], atol=1e-5, rtol=0)
    random_controls = []
    for budget in (4096, 16_384):
        generator = np.random.default_rng(detector.DEFAULT_SETTINGS.seed)
        chosen = generator.choice(len(quads), budget, replace=False) if len(quads) > budget else np.arange(len(quads))
        random_controls.append({'budget': budget, 'selected': len(chosen),
                                'known_seed_selected': bool(target_id in seed_ids[chosen])})
    pool_count = 0
    for pair in selection['vp_pairs']:
        x_ranks, y_ranks = pair['family_ranks']
        if set(target['x_line_ranks']) <= set(x_ranks) and set(target['y_line_ranks']) <= set(y_ranks):
            pool_count += 1
    if not all(direction['compatible_candidates'] for direction in direction_records):
        loss = 'intersection_pool'
    elif not all(direction['compatible_retained'] for direction in direction_records):
        loss = 'pencil_retention'
    elif target_id not in selection.get('union_pair_product_ids', []):
        loss = 'ordered_pair_or_geometry'
    elif target_id not in selection['selected_pair_product_ids']:
        loss = 'rectangle_budget'
    elif target_id not in selection['retained_pair_product_ids']:
        loss = 'area_gate'
    else:
        loss = 'recovered'
    return {'case_id': source['id'], 'pencil_selection': record['settings'].get('pencil_selection', 'ranked'),
            'pair_product_id': target_id, 'directions': direction_records, 'first_loss': loss,
            'target_pool_count': pool_count, 'random_budget_controls': random_controls,
            'reconstructed_max_abs_difference_px': float(np.max(np.abs(native_corners - target['corners_px'])))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, nargs='+', required=True)
    parser.add_argument('--targets', type=Path, required=True)
    parser.add_argument('--results', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sources = {case['id']: case for path in args.inputs for case in read(path)['cases']}
    results = []
    for directory in args.results:
        for target in read(args.targets)['targets']:
            record = read(directory / f"{target['case_id']}.json.gz")
            result = diagnose(sources[target['case_id']], target, record)
            results.append(result)
            print(json.dumps(result), flush=True)
    with gzip.open(args.output, 'wt', compresslevel=9) as stream:
        json.dump({'schema': 'vp-pruning-fixed-target-diagnosis/1', 'records': results}, stream)


if __name__ == '__main__':
    main()
