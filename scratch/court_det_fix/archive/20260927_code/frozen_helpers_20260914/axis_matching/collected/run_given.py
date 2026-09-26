"""Diagnose the axis matcher with supplied directions before automatic VP search."""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from projective_seed import AxisMatches, Settings, basis_for, combine, match_axis
from run_diagnosis import gate_evidence, read, write
from run_population import prepare
from scan_population import geometry, retain

from experiments.annotator.independent_court import assignment, detector
from experiments.annotator.independent_court import stripe_observations as stripes

KEEP_COMBINED = 256


def control_homography(source: dict, reference: dict, vp_saved: dict, marking_summary: dict) -> tuple[np.ndarray, str]:
    case_id = source['id']
    if case_id == 'gxBQ_window_00_frame_5':
        corners = vp_saved['known_target']['corners_px']
        origin = 'visually_approved_GX5_generated_court'
    elif case_id in ('shuttleset_03_scene_0019', 'shuttleset_03_scene_0016', 'shuttleset_21_scene_0020'):
        record = next(record for record in marking_summary['records'] if record['case_id'] == case_id)
        corners = record['fit']['corners_px']
        origin = 'visually_approved_automatic_marking_refit'
    elif case_id == 'am2_window_00_frame_150':
        selected = vp_saved['diagnostic_rankings']['geometry_before_players']['max_corner_native_px'][0]['generation_id']
        corners = next(row['corners_px'] for row in vp_saved['examples'] if row['generation_id'] == selected)
        origin = 'visually_approved_VP_diagnostic_court'
    else:
        corners = reference['corners_px']
        origin = 'manual_reference_directions_only'
    _, _, size = prepare(source)
    scale = np.array([source['dimensions']['width'], source['dimensions']['height']]) / size
    transform = cv2.getPerspectiveTransform(detector.CORNER_COURT_M, (np.asarray(corners) / scale).astype(np.float32))
    return transform, origin


def canonicalise(homographies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    corners, _ = detector.project(homographies, detector.CORNER_COURT_M)
    rotate = corners[:, :2, 1].mean(axis=1) > corners[:, 2:, 1].mean(axis=1)
    symmetry = np.array([[-1., 0., detector.CORNER_COURT_M[:, 0].max()],
                         [0., -1., detector.CORNER_COURT_M[:, 1].max()], [0., 0., 1.]])
    homographies = homographies.copy()
    homographies[rotate] = homographies[rotate] @ symmetry
    return homographies, rotate


def axis_diagnostic(matches: AxisMatches, basis: np.ndarray, axis: int, control: np.ndarray) -> dict:
    """Assess axis loss with the other control axis held fixed; never select proposals."""
    rectified = np.linalg.solve(basis, control)
    rectified /= rectified[2, 2]
    maps = np.tile(rectified, (len(matches.parameters), 1, 1))
    maps[:, axis, axis], maps[:, axis, 2] = matches.parameters.T
    projected, _ = detector.project(basis @ maps, detector.CORNER_COURT_M)
    truth, _ = detector.project(control[None], detector.CORNER_COURT_M)
    errors = np.linalg.norm(projected - truth, axis=2).max(axis=1)
    result = {}
    for name, ids in [('enumerated', np.arange(len(errors))),
                      ('pattern_supported', np.flatnonzero(matches.supported >= Settings().minimum_matches)),
                      ('pattern_and_players', np.flatnonzero((matches.supported >= Settings().minimum_matches)
                                                             & matches.player_compatible)),
                      ('distinct', matches.distinct), ('retained', matches.retained)]:
        if not len(ids):
            result[name] = None
            continue
        index = int(ids[np.argmin(errors[ids])])
        result[name] = {'axis_id': index, 'max_corner_working_px': float(errors[index]),
                        'score': float(matches.scores[index]), 'support_count': int(matches.supported[index]),
                        'matched_groups': matches.matches[index].tolist(), 'anchors': matches.anchors[index].tolist()}
    return result


def pack_axis(matches: AxisMatches) -> dict:
    entries = []
    for index in matches.retained:
        entries.append({'axis_id': int(index), 'parameters': matches.parameters[index].tolist(),
                        'score': float(matches.scores[index]), 'matched_groups': matches.matches[index].tolist(),
                        'anchors': matches.anchors[index].tolist(), 'support_count': int(matches.supported[index])})
    return {'diagnostics': matches.diagnostics, 'entries': entries}


def run_case(
    source: dict, control: np.ndarray, origin: str, zone: object, reference: dict, player_pruning: bool = False,
) -> dict:
    started = perf_counter()
    settings = Settings()
    segments, families, size = prepare(source)
    scale = np.array([source['dimensions']['width'], source['dimensions']['height']]) / size
    observations = assignment.prepare_observations(segments, size)
    weights = stripes.fragment_weights(observations)
    feet = np.asarray([[[np.nan, np.nan] if foot is None else foot for foot in frame]
                       for frame in source['all_feet_px']], dtype=float) / scale
    role_records, candidates, provenance = [], [], {}
    counter = 0
    true_corners, _ = detector.project(control[None], detector.CORNER_COURT_M)
    for role, points in enumerate((control[:, :2].T, control[:, [1, 0]].T)):
        basis, details = basis_for(points, size, settings)
        record = {'role': role, 'basis_status': details}
        if basis is None:
            role_records.append(record)
            continue
        axis_feet = feet if player_pruning else None
        horizontal = match_axis(basis, 0, detector.X_COORDS, observations, size, settings, axis_feet)
        vertical = match_axis(basis, 1, detector.Y_COORDS, observations, size, settings, axis_feet)
        transforms, axis_pairs = combine(basis, horizontal, vertical)
        transforms, rotated = canonicalise(transforms)
        valid, corners = geometry(transforms, size)
        one, two = zone.player_fractions(transforms, feet) if len(transforms) else (np.array([]), np.array([]))
        usable = valid & (one == 1) & (two >= .5)
        record.update({'basis_working': basis.tolist(), 'axes': [pack_axis(horizontal), pack_axis(vertical)],
                       'combined': len(transforms), 'geometry_valid': int(valid.sum()),
                       'geometry_players': int(usable.sum())})
        for index in np.flatnonzero(usable):
            first, second = axis_pairs[index]
            score = float((horizontal.scores[first] + vertical.scores[second]) / 2)
            candidate = detector.Candidate(corners[index], score, (0., 0.), (0, 0))
            candidates.append(candidate)
            provenance[id(candidate)] = {'candidate_id': counter, 'role': role,
                                         'axis_ids': [int(first), int(second)], 'rotated_180': bool(rotated[index]),
                                         'homography_working': transforms[index].tolist()}
            counter += 1
        # Given directions are diagnostic; cap-loss measurements happen after generation.
        if role == 0:
            record['axis_control_diagnostics'] = [axis_diagnostic(horizontal, basis, 0, control),
                                                  axis_diagnostic(vertical, basis, 1, control)]
        role_records.append(record)
    retention = replace(detector.DEFAULT_SETTINGS, keep_candidates=KEEP_COMBINED, distinct_corner_distance=2.)
    retained = retain(candidates, retention)
    maps = detector._distance_maps(detector._wide_line_families(segments), size)
    entries = []
    for candidate in retained:
        details = provenance[id(candidate)]
        homography = np.asarray(details['homography_working'])
        stripe = stripes.score_model(stripes.measure(homography, observations, size), weights, 3)
        corners = candidate.corners_px * scale
        gates = gate_evidence(corners, source, scale, size, families, maps, zone)
        entries.append({**details, 'corners_px': corners.tolist(), 'axis_score': candidate.score,
                        'stripe': stripe, 'gates': gates})
    rankings = {}
    for name, pool in [('geometry_players_before_cap', candidates), ('after_combined_cap', retained)]:
        if not pool:
            rankings[name] = None
            continue
        errors = np.asarray([np.linalg.norm(candidate.corners_px - true_corners[0], axis=1).max() for candidate in pool])
        best = int(np.argmin(errors))
        rankings[name] = {'candidate_id': provenance[id(pool[best])]['candidate_id'],
                          'control_max_corner_working_px': float(errors[best])}
    for entry in entries:
        differences = (np.asarray(entry['corners_px']) - reference['corners_px']) * 1280 / source['dimensions']['width']
        entry['reference_max_corner_1280_px'] = float(np.linalg.norm(differences, axis=1).max())
    return {'schema': 'given-directions-axis-matching/1', 'case_id': source['id'], 'given_direction_source': origin,
            'automatic_detection': False, 'settings': asdict(settings), 'working_size': size,
            'necessary_player_pruning': player_pruning,
            'control_corners_px': (true_corners[0] * scale).tolist(), 'roles': role_records, 'entries': entries,
            'combined_shortlist': len(retained), 'combined_cap': KEEP_COMBINED, 'control_diagnostics': rankings,
            'raw_groups': [observations.fragment_ids[group].tolist() for group in observations.groups],
            'elapsed_s': perf_counter() - started}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--vp-saved', type=Path, required=True)
    parser.add_argument('--marking-summary', type=Path, required=True)
    parser.add_argument('--legacy', type=Path, required=True)
    parser.add_argument('--ids', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--axis-player-pruning', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(args.legacy.resolve()))
    zone = importlib.import_module('zone_net')
    cv2.setNumThreads(1)
    packed = read(args.inputs)
    cases = {source['id']: source for source in packed['cases']}
    marking = read(args.marking_summary)
    for case_id in args.ids:
        source, reference = cases[case_id], packed['references'][case_id]
        control, origin = control_homography(source, reference, read(args.vp_saved / f'{case_id}.json.gz'), marking)
        result = run_case(source, control, origin, zone, reference, args.axis_player_pruning)
        write(args.output / f'{case_id}.json.gz', result)
        print(case_id, 'roles', [{key: role.get(key) for key in ('combined', 'geometry_valid', 'geometry_players')}
                                for role in result['roles']], 'control', result['control_diagnostics'],
              'seconds', result['elapsed_s'], flush=True)


if __name__ == '__main__':
    main()
