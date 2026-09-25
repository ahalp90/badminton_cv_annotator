"""Feed frozen automatically estimated direction pairs into the inspected matcher.

cap_loss copy: generate also records, per direction pair, every court that passed the
player checks and the indices the per-pair cap retained. The recording only reads the
candidate objects; candidate order and selection are untouched.

pregate copy: propose_role comes from this folder's run_given.py, which also hands back every
combined court before the geometry and player masks. The side file gains, per pair, the
combined corners, both masks and the two player fractions.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from dataclasses import asdict, replace
from itertools import permutations
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from inspect_appearance import frame_path, profiles
from projective_seed import Settings
from run_diagnosis import gate_evidence, read, write
import run_given
from run_given import propose_role
from run_population import prepare
from scan_population import retain

from experiments.annotator.independent_court import assignment, detector
from experiments.annotator.independent_court import stripe_observations as stripes

# The pre-gate courts must come from this folder's run_given.py, never the axis_matching sibling.
assert Path(run_given.__file__).resolve().parent == Path(__file__).resolve().parent, run_given.__file__

KEEP_COURTS = 256
# Per pair, the pre-gate arrays in transforms order: combined corners (working px, float32), the
# geometry mask, the player mask, and the two player fractions the player mask reads.
PreGate = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
CAMERA_ERROR_LIMIT = .1
CAMERA_ROUNDING_MARGIN = 1e-6


def camera_direction_bound(points: np.ndarray, size: tuple[int, int]) -> float:
    """Lower-bound the archived camera error without choosing court scale or position."""
    width, height = size
    focals = np.geomspace(.4 * width, 4 * width, 200)
    axes = np.broadcast_to(points, (len(focals), 2, 3)).copy()
    axes[:, :, :2] -= np.array([width / 2, height / 2]) * axes[:, :, 2:]
    axes[:, :, :2] /= focals[:, None, None]
    norms = np.linalg.norm(axes, axis=2)
    cosine = np.sum(axes[:, 0] * axes[:, 1], axis=1) / np.prod(norms, axis=1)
    return float(np.abs(cosine).min())


def select_pool(candidates: list[detector.Candidate]) -> list[detector.Candidate]:
    settings = replace(detector.DEFAULT_SETTINGS, keep_candidates=KEEP_COURTS, distinct_corner_distance=2.)
    return retain(candidates, settings)


def evaluate_pool(
    source: dict, shortlist: list[dict], observations: assignment.Observations,
    size: tuple[int, int], segments: np.ndarray, families: tuple, zone: object, root: Path,
    legacy_evidence: bool = True,
) -> list[dict]:
    """Measure the unchanged stripe, camera, floor and paint evidence for saved courts.

    :param legacy_evidence: Also score each court's stripes and paint profile. Only the
        research rankings read these two; the court detector turns them off.
    """
    scale = np.array([source['dimensions']['width'], source['dimensions']['height']]) / size
    weights = stripes.fragment_weights(observations)
    maps = detector._distance_maps(detector._wide_line_families(segments), size)
    entries = []
    started = perf_counter()
    for position, details in enumerate(shortlist):
        if len(shortlist) > 256 and position % 512 == 0:
            print(source['id'], 'full evidence', position, 'of', len(shortlist),
                  'seconds', perf_counter() - started, flush=True)
        corners = np.asarray(details['corners_px'])
        entry = {**details, 'corners_px': corners.tolist(), 'shortlist_score': details['shortlist_score']}
        if legacy_evidence:
            homography = np.asarray(details['homography_working'])
            entry['stripe'] = stripes.score_model(stripes.measure(homography, observations, size), weights, 3)
        entry['gates'] = gate_evidence(corners, source, scale, size, families, maps, zone)
        entries.append(entry)
    if entries and legacy_evidence:
        path = frame_path(source, root)
        frame = cv2.imread(str(path))
        if frame is None:
            raise FileNotFoundError(path)
        assert frame.shape[:2] == (source['dimensions']['height'], source['dimensions']['width'])
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        paint = profiles(frame, np.asarray([entry['homography_working'] for entry in entries]))
        for entry, profile in zip(entries, paint, strict=True):
            entry['profile'] = profile
    return entries


def winner_ids(entries: list[dict]) -> tuple[str | None, str | None]:
    """Preserve the two existing rankings within the same camera-eligible pool."""
    eligible = [entry for entry in entries if entry['gates']['camera_error'] is not None
                and entry['gates']['camera_error'] <= CAMERA_ERROR_LIMIT and entry['profile']['score'] is not None]
    line = max(eligible, key=lambda entry: entry['stripe']['exclusive']['score'], default=None)
    paint = max(eligible, key=lambda entry: (entry['profile']['score'], entry['stripe']['exclusive']['score']), default=None)
    return (None if line is None else line['candidate_id'], None if paint is None else paint['candidate_id'])


def write_pool(path: Path, records: list[tuple[int, int, np.ndarray, np.ndarray, PreGate]]) -> None:
    """Store the proposed courts and retained indices of every matched pair as one compressed .npz.

    pregate copy: also the combined courts before the masks, both masks and the player fractions.
    """
    arrays = {}
    for pair_id, count, corners, retained_index, pregate in records:
        assert corners.shape == (count, 4, 2), (pair_id, corners.shape, count)
        arrays[f'pair_{pair_id}_corners'] = corners
        arrays[f'pair_{pair_id}_retained_index'] = retained_index
        combined_corners, valid, usable, player_any, player_both_halves = pregate
        combined = len(combined_corners)
        assert combined_corners.shape == (combined, 4, 2), (pair_id, combined_corners.shape)
        assert valid.shape == usable.shape == (combined,), (pair_id, valid.shape, usable.shape)
        assert player_any.shape == player_both_halves.shape == (combined,), (pair_id, player_any.shape)
        arrays[f'pair_{pair_id}_combined_corners'] = combined_corners
        arrays[f'pair_{pair_id}_valid'] = valid
        arrays[f'pair_{pair_id}_usable'] = usable
        arrays[f'pair_{pair_id}_player_any'] = player_any
        arrays[f'pair_{pair_id}_player_both_halves'] = player_both_halves
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with open(temporary, 'wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def generate(source: dict, saved: dict, zone: object, root: Path, pool_path: Path | None = None) -> dict:
    """Generate and rank courts using observations and automatic directions only."""
    started = perf_counter()
    segments, families, size = prepare(source)
    assert list(size) == saved['working_size']
    assert saved['settings']['pencil_selection'] == 'coverage'
    points = np.asarray(saved['estimator']['points_working'])
    native_size = (source['dimensions']['width'], source['dimensions']['height'])
    scale = np.asarray(native_size) / size
    point_scale = np.append(scale, 1.)
    feet = np.asarray([[[np.nan, np.nan] if foot is None else foot for foot in frame]
                       for frame in source['all_feet_px']], dtype=float) / scale
    observations = assignment.prepare_observations(segments, size)
    settings = Settings(keep_axes=512)
    pooled, provenance, pair_records = [], {}, []
    # One entry per matched pair: pair ID, proposed count, corners (working px), retained indices and
    # the pre-gate arrays. Only arrays are kept so the millions of candidate objects can be freed per pair.
    pool_records: list[tuple[int, int, np.ndarray, np.ndarray, PreGate]] = []
    for pair_id, pencil_ids in enumerate(permutations(range(len(points)), 2)):
        pair_points = points[list(pencil_ids)]
        bound = camera_direction_bound(pair_points * point_scale, native_size)
        record = {'pair_id': pair_id, 'pencils': list(pencil_ids), 'camera_direction_bound': bound}
        if bound > CAMERA_ERROR_LIMIT + CAMERA_ROUNDING_MARGIN:
            pair_records.append({**record, 'status': 'camera_direction_bound'})
            continue
        pair_start = perf_counter()
        proposed = propose_role(pair_points, observations, feet, size, settings)
        proposed_corners = np.asarray([candidate.corners_px for candidate in proposed.candidates],
                                      dtype=np.float32).reshape(-1, 4, 2)
        proposed_positions = {id(candidate): position for position, candidate in enumerate(proposed.candidates)}
        retained = select_pool(proposed.candidates)
        pool_records.append((pair_id, len(proposed.candidates), proposed_corners,
                             np.asarray([proposed_positions[id(candidate)] for candidate in retained], dtype=np.int32),
                             (proposed.combined_corners, proposed.valid, proposed.usable,
                              proposed.player_any, proposed.player_both_halves)))
        shortlist = []
        for candidate in retained:
            position = proposed_positions[id(candidate)]
            details = {'candidate_id': f'{pair_id}:{position}', 'pair_id': pair_id, **proposed.detail(position)}
            provenance[id(candidate)] = details
            shortlist.append({**details, 'corners_px': (candidate.corners_px * scale).tolist(),
                              'shortlist_score': candidate.score})
        pooled.extend(retained)
        record.update({'status': 'matched', 'role': proposed.record, 'shortlist': shortlist,
                       'elapsed_s': perf_counter() - pair_start})
        pair_records.append(record)
        print(source['id'], 'pair', pair_id, list(pencil_ids), 'combined', proposed.record.get('combined', 0),
              'players', len(proposed.candidates), 'retained', len(retained), 'seconds', record['elapsed_s'], flush=True)
    retained = select_pool(pooled)
    shortlist = []
    for candidate in retained:
        shortlist.append({**provenance[id(candidate)], 'corners_px': (candidate.corners_px * scale).tolist(),
                          'shortlist_score': candidate.score})
    entries = evaluate_pool(source, shortlist, observations, size, segments, families, zone, root)
    line_id, paint_id = winner_ids(entries)
    if pool_path is not None:
        write_pool(pool_path, pool_records)
    return {'schema': 'automatic-directions-axis-matching/1', 'case_id': source['id'],
            'automatic_directions': True, 'label_guided_generation': False, 'emission_decision': None,
            'working_size': size, 'settings': asdict(settings), 'keep_per_pair': KEEP_COURTS,
            'keep_global': KEEP_COURTS, 'estimator_settings': saved['settings'], 'estimator': saved['estimator'],
            'camera_error_limit': CAMERA_ERROR_LIMIT, 'camera_rounding_margin': CAMERA_ROUNDING_MARGIN,
            'camera_bound_coordinate_space': 'native',
            'pairs': pair_records, 'pooled_candidates': len(pooled), 'entries': entries,
            'raw_groups': [observations.fragment_ids[group].tolist() for group in observations.groups],
            'line_winner_id': line_id, 'paint_winner_id': paint_id,
            'elapsed_s': perf_counter() - started}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--saved', type=Path, required=True)
    parser.add_argument('--legacy', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--ids', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.legacy.resolve()))
    zone = importlib.import_module('zone_net')
    cv2.setNumThreads(1)
    sources = {source['id']: source for source in read(args.inputs)['cases']}
    for case_id in args.ids:
        result = generate(sources[case_id], read(args.saved / f'{case_id}.json.gz'), zone, args.root)
        write(args.output / f'{case_id}.json.gz', result)
        print(case_id, 'complete', result['line_winner_id'], result['paint_winner_id'],
              'seconds', result['elapsed_s'], flush=True)


if __name__ == '__main__':
    main()
