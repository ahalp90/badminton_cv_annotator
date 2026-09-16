"""Test the existing camera check before global retention of saved automatic courts."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from run_automatic import CAMERA_ERROR_LIMIT, evaluate_pool, select_pool, winner_ids
from run_diagnosis import read, write
from run_population import prepare

from experiments.annotator.independent_court import assignment, detector


def rescore(
    source: dict, baseline: dict, zone: object, root: Path, replay: bool, keep_all_camera: bool = False,
) -> dict:
    started = perf_counter()
    segments, families, size = prepare(source)
    assert list(size) == list(baseline['working_size'])
    observations = assignment.prepare_observations(segments, size)
    if replay:
        measured = evaluate_pool(source, baseline['entries'], observations, size, segments, families, zone, root)
        assert measured == baseline['entries'], 'Extracted final evaluation differs from saved baseline'
        assert winner_ids(measured) == (baseline['line_winner_id'], baseline['paint_winner_id'])
        print(source['id'], 'exact final-evaluation replay passed', flush=True)
    native_size = (source['dimensions']['width'], source['dimensions']['height'])
    scale = np.asarray(native_size) / size
    cached = {entry['candidate_id']: entry for entry in baseline['entries']}
    candidates, all_candidates, provenance, camera_records = [], [], {}, []
    saved_camera = {entry['candidate_id']: entry['camera_error'] for entry in baseline.get('camera_prefilter', [])}
    for pair in baseline['pairs']:
        for entry in pair.get('shortlist', []):
            corners = np.asarray(entry['corners_px'])
            if entry['candidate_id'] in saved_camera:
                error = saved_camera[entry['candidate_id']]
            else:
                _, error, _ = zone.net_segments(corners, native_size)
                error = float(error) if np.isfinite(error) else None
            candidate = detector.Candidate(corners / scale, entry['shortlist_score'], (0., 0.), (0, 0))
            all_candidates.append(candidate)
            provenance[id(candidate)] = entry
            camera_records.append({'candidate_id': entry['candidate_id'],
                                   'camera_error': error})
            if error is None or error > CAMERA_ERROR_LIMIT:
                continue
            candidates.append(candidate)
    if replay:
        stage = baseline.get('selection_stage')
        if stage == 'camera_pool_without_global_cap':
            replay_selected = candidates
        else:
            replay_pool = candidates if stage == 'camera_before_global_cap' else all_candidates
            replay_selected = select_pool(replay_pool)
        reconstructed = [provenance[id(candidate)]['candidate_id'] for candidate in replay_selected]
        assert reconstructed == [entry['candidate_id'] for entry in baseline['entries']]
        print(source['id'], 'exact selection replay passed', flush=True)
    retained = candidates if keep_all_camera else select_pool(candidates)
    selected = [provenance[id(candidate)] for candidate in retained]
    missing = [entry for entry in selected if entry['candidate_id'] not in cached]
    for measured in evaluate_pool(source, missing, observations, size, segments, families, zone, root):
        cached[measured['candidate_id']] = measured
    entries = [cached[entry['candidate_id']] for entry in selected]
    line_id, paint_id = winner_ids(entries)
    return {**baseline, 'selection_stage': ('camera_pool_without_global_cap' if keep_all_camera
                                            else 'camera_before_global_cap'),
            'keep_global': None if keep_all_camera else baseline['keep_global'],
            'generation_elapsed_s': baseline.get('generation_elapsed_s', baseline['elapsed_s']), 'elapsed_s': perf_counter() - started,
            'camera_prefilter': camera_records, 'camera_eligible_before_global': len(candidates),
            'entries': entries, 'line_winner_id': line_id, 'paint_winner_id': paint_id}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--legacy', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--ids', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--keep-all-camera', action='store_true')
    args = parser.parse_args()
    assert args.output.resolve() != args.baseline.resolve(), 'Preserve the baseline directory'
    sources = {}
    for path in args.inputs:
        sources.update({source['id']: source for source in read(path)['cases']})
    sys.path.insert(0, str(args.legacy.resolve()))
    zone = importlib.import_module('zone_net')
    cv2.setNumThreads(1)
    for case_id in args.ids:
        baseline = read(args.baseline / f'{case_id}.json.gz')
        result = rescore(sources[case_id], baseline, zone, args.root, args.replay, args.keep_all_camera)
        write(args.output / f'{case_id}.json.gz', result)
        print(case_id, 'camera first', result['camera_eligible_before_global'],
              'retained', len(result['entries']), 'winners', result['line_winner_id'], result['paint_winner_id'],
              'seconds', result['elapsed_s'], flush=True)


if __name__ == '__main__':
    main()
