"""Score saved automatic VP proposals through the unchanged cached-input tracer."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
from diagnose_targets import read
from run_population import prepare
from vp_pruning import Settings, select

from experiments.annotator.independent_court import detector


def optional_float(value: float) -> float | None:
    """Represent unmeasured downstream values as JSON null."""
    return float(value) if np.isfinite(value) else None


def optional_counts(values: np.ndarray) -> list[int] | None:
    """Replace the tracer's unmeasured-count sentinel with JSON null."""
    return None if np.any(values < 0) else values.tolist()


def corner_summary(record: dict, rows: np.ndarray, reference: dict) -> dict:
    """Summarise pre-gate and surviving geometry without boundary landmarks."""
    players = (rows['one_fraction'] == 1) & (rows['two_fraction'] >= .5)
    floor = rows['floor_score'] >= 0
    phases = {'geometry_before_players': np.ones(len(rows), dtype=bool), 'players_and_geometry': players,
              'floor_survivors': floor, 'camera_survivors': floor & (rows['camera_error'] <= .1)}
    record['counts'] = {name: int(mask.sum()) for name, mask in phases.items()}
    record['diagnostic_rankings'] = {}
    chosen: set[int] = set()
    errors = np.linalg.norm(rows['corners_px'] - np.asarray(reference['corners_px']), axis=2).max(axis=1)
    for name, mask in phases.items():
        indices = np.flatnonzero(mask)
        order = indices[np.argsort(errors[indices], kind='stable')[:5]]
        record['diagnostic_rankings'][name] = {
            'max_corner_native_px': [{'generation_id': int(rows['generation_id'][index]), 'value': float(errors[index])}
                                     for index in order],
        }
        chosen.update(order.tolist())
    record['examples'] = []
    for index in sorted(chosen):
        record['examples'].append({'generation_id': int(rows['generation_id'][index]),
                                  'corners_px': rows['corners_px'][index].tolist(),
                                  'max_corner_native_px': float(errors[index]),
                                  'player_pass': bool(players[index]),
                                  'original_floor_score': optional_float(rows['floor_score'][index]),
                                  'camera_error': optional_float(rows['camera_error'][index]),
                                  'family_support': [optional_float(value) for value in rows['family_support'][index]],
                                  'line_counts': optional_counts(rows['line_counts'][index])})
    record['boundary_metrics'] = 'unavailable: reference has no boundary landmarks'
    return record


def score(source: dict, population: dict, zone: Any, trace: Any) -> tuple[dict, np.ndarray]:
    """Replay the saved population and verify direct versus traced final outputs."""
    _, families, size = prepare(source)
    settings = detector.Settings(wide_families=True, min_supported_lines=3)
    started = perf_counter()
    rectangles, selection = select(families, np.asarray(population['estimator']['points_working']), size,
                                   settings, Settings(**population['settings']))
    if selection['retained_pair_product_ids'] != population['seed_selection']['retained_pair_product_ids']:
        raise ValueError('Saved automatic rectangle population did not reproduce')
    selected = perf_counter()
    original = detector._image_rectangles

    def fixed_rectangles(
        x_lines: np.ndarray, y_lines: np.ndarray, given_size: tuple, given_settings: Any,
    ) -> np.ndarray:
        np.testing.assert_array_equal(x_lines, families[0])
        np.testing.assert_array_equal(y_lines, families[1])
        assert given_size == size and given_settings == settings
        return rectangles

    detector._image_rectangles = fixed_rectangles
    try:
        record, rows = trace.capture(source, zone, settings)
    finally:
        detector._image_rectangles = original
    if record['final']['generated'] != len(rectangles) * len(detector.TEMPLATE_TRANSFORMS):
        raise ValueError('The loaded detector did not generate the saved VP rectangle population')
    record['timing'] = {'selection_replay_s': selected - started, 'direct_and_traced_s': perf_counter() - selected,
                        'original_estimate_s': population['timing']['estimate_s']}
    record['seed_selection'] = selection
    return record, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--population', type=Path, required=True)
    parser.add_argument('--legacy', type=Path, required=True)
    parser.add_argument('--trace-directory', type=Path, required=True)
    parser.add_argument('--id', required=True)
    parser.add_argument('--broadcast', action='store_true')
    parser.add_argument('--click-inset-m', type=float, default=.005)
    parser.add_argument('--targets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path[:0] = [str(args.legacy.resolve()), str(args.trace_directory.resolve())]
    zone = importlib.import_module('zone_net')
    trace = importlib.import_module('run_trace')
    cv2.setNumThreads(1)
    packed = read(args.inputs)
    source = next(case for case in packed['cases'] if case['id'] == args.id)
    population = read(args.population / f'{args.id}.json.gz')
    record, rows = score(source, population, zone, trace)
    reference = packed['references'][args.id]
    if args.broadcast or not len(rows):
        record = corner_summary(record, rows, reference)
    else:
        record = trace.summarise(record, rows, source, reference, zone, args.click_inset_m)
    record['known_target'] = None
    for target in read(args.targets)['targets']:
        if target['case_id'] != args.id:
            continue
        selected_ids = record['seed_selection']['retained_pair_product_ids']
        if target['pair_product_id'] not in selected_ids:
            record['known_target'] = {'selected': False}
            continue
        rectangle_index = selected_ids.index(target['pair_product_id'])
        generation = rectangle_index * len(detector.TEMPLATE_TRANSFORMS) + target['template_index']
        matches = np.flatnonzero(rows['generation_id'] == generation)
        target_record = {'selected': True, 'generation_id': generation, 'geometry_valid': bool(len(matches))}
        if len(matches):
            row = rows[matches[0]]
            np.testing.assert_allclose(row['corners_px'], target['corners_px'], atol=1e-5, rtol=0)
            target_record.update({
                'corners_px': row['corners_px'].tolist(),
                'one_fraction': float(row['one_fraction']), 'two_fraction': float(row['two_fraction']),
                'floor_score': optional_float(row['floor_score']),
                'family_support': [optional_float(value) for value in row['family_support']],
                'line_counts': optional_counts(row['line_counts']), 'camera_error': optional_float(row['camera_error']),
            })
        record['known_target'] = target_record
    record['selected_max_corner_1280_px'] = None
    if record['final']['candidates']:
        corners = np.asarray(record['final']['candidates'][0]['corners_px'])
        dimensions = source['dimensions']
        display_scale = np.array([1280 / dimensions['width'], 720 / dimensions['height']])
        errors = np.linalg.norm((corners - reference['corners_px']) * display_scale, axis=1)
        record['selected_max_corner_1280_px'] = float(errors.max())
    record['schema'] = 'vp-pruning-scored/1'
    record['case_id'] = args.id
    record['labels_used_only_after_detection'] = True
    record['click_inset_m'] = args.click_inset_m
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trace.write(args.output, record)
    print(args.id, record['final']['accepted'], record['final']['reason'], record['counts'],
          'selected error', record['selected_max_corner_1280_px'], 'seconds', record['timing'], flush=True)


if __name__ == '__main__':
    main()
