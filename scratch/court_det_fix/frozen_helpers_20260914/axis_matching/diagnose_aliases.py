"""Measure control support and competing court identities after the frozen runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from projective_seed import (
    Settings,
    basis_for,
    corner_errors,
    match_axis,
    offsets,
    score_axes,
)
from run_diagnosis import read, write
from run_given import axis_diagnostic, control_homography
from run_population import prepare

from experiments.annotator.independent_court import assignment, detector
from experiments.annotator.independent_court import stripe_observations as stripes


def diagnose(source: dict, reference: dict, saved: dict, marking: dict, given: dict) -> dict:
    control, origin = control_homography(source, reference, saved, marking)
    segments, _, size = prepare(source)
    scale = np.array([source['dimensions']['width'], source['dimensions']['height']]) / size
    observations = assignment.prepare_observations(segments, size)
    basis, _ = basis_for(control[:, :2].T, size, Settings())
    assert basis is not None
    rectified = np.linalg.solve(basis, control)
    rectified /= rectified[2, 2]
    axis_scores = []
    feet = np.asarray([[[np.nan, np.nan] if foot is None else foot for foot in frame]
                       for frame in source['all_feet_px']], dtype=float) / scale
    for axis, coordinates in enumerate((detector.X_COORDS, detector.Y_COORDS)):
        ids, _, endpoints, ledger = offsets(basis, axis, observations, size, Settings())
        values, matches, count = score_axes(rectified[axis, [axis, 2]][None], coordinates, endpoints, basis, axis)
        group_ids = np.where(matches[0] >= 0, ids[np.maximum(matches[0], 0)], -1)
        population = match_axis(basis, axis, coordinates, observations, size, Settings(), feet)
        closest = axis_diagnostic(population, basis, axis, control)['distinct']
        rank = int(np.flatnonzero(population.distinct == closest['axis_id'])[0])
        identities = [tuple(population.matches[index]) for index in population.distinct[:rank + 1]]
        mirror_distinct = {min(identity, identity[::-1]) for identity in identities}
        axis_scores.append({'axis': axis, 'score': float(values[0]), 'count': int(count[0]),
                            'group_ids': group_ids.tolist(), 'ledger': ledger,
                            'closest_distinct': closest, 'closest_distinct_rank_1based': rank + 1,
                            'distinct_mirror_classes_through_closest': len(mirror_distinct),
                            'axis_boundary_score': float(population.scores[population.retained[-1]])})
    weights = stripes.fragment_weights(observations)
    control_stripe = stripes.score_model(stripes.measure(control, observations, size), weights, 3)
    previous = next(record for record in marking['records'] if record['case_id'] == source['id'])['fit']
    old_transform = cv2.getPerspectiveTransform(detector.CORNER_COURT_M,
                                               (np.asarray(previous['corners_px']) / scale).astype(np.float32))
    relative = np.linalg.solve(control, old_transform)
    relative /= relative[2, 2]
    candidates = [entry for entry in given['entries'] if entry['gates']['camera_error'] is not None
                  and entry['gates']['camera_error'] <= .1]
    winner = max(candidates, key=lambda entry: entry['stripe']['exclusive']['score'], default=None)
    selected = None
    if winner is not None:
        changed = np.linalg.solve(control, np.asarray(winner['homography_working']))
        changed /= changed[2, 2]
        selected = {'candidate_id': winner['candidate_id'], 'role': winner['role'],
                    'reference_max_corner_1280_px': winner['reference_max_corner_1280_px'],
                    'stripe': winner['stripe'], 'relative_to_control': changed.tolist(),
                    'corners_px': winner['corners_px'], 'gates': winner['gates']}
    corners = np.asarray([entry['corners_px'] for entry in given['entries']])
    display_scale = np.array([1280, 720]) / np.array([source['dimensions']['width'], source['dimensions']['height']])
    corrected = corner_errors(corners * display_scale, np.asarray(reference['corners_px']) * display_scale)
    previous_errors = np.asarray([entry['reference_max_corner_1280_px'] for entry in given['entries']])
    control_corners = np.asarray(given['control_corners_px']) / scale
    direct = np.linalg.norm(corners / scale - control_corners, axis=2).max(axis=1)
    physical = corner_errors(corners / scale, control_corners)
    return {'case_id': source['id'], 'control_source': origin, 'control_axes': axis_scores,
            'control_stripe': control_stripe, 'previous_wrong_or_good_fit_relative_to_control': relative.tolist(),
            'given_direction_winner': selected,
            'corner_metric_audit': {'reference_max_correction_px': float(np.abs(corrected - previous_errors).max()),
                                    'control_max_correction_px': float(np.abs(physical - direct).max()),
                                    'corrected_closest_control_working_px': float(physical.min())}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', type=Path, required=True)
    parser.add_argument('--vp-saved', type=Path, required=True)
    parser.add_argument('--marking-summary', type=Path, required=True)
    parser.add_argument('--given', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    marking = read(args.marking_summary)
    records = []
    for pack_path in args.inputs:
        packed = read(pack_path)
        for source in packed['cases']:
            path = args.given / f"{source['id']}.json.gz"
            if path.exists():
                result = diagnose(source, packed['references'][source['id']],
                                  read(args.vp_saved / path.name), marking, read(path))
                records.append(result)
                print(source['id'], 'control axes', [(row['score'], row['count']) for row in result['control_axes']],
                      'previous relative', result['previous_wrong_or_good_fit_relative_to_control'], flush=True)
    write(args.output, {'schema': 'axis-identity-diagnosis/1', 'label_guided': True, 'records': records})


if __name__ == '__main__':
    main()
