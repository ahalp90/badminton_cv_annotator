"""E4: feed one arm's directions through the unchanged matcher and both rescoring stages.

Each case-arm writes three records: generation (results), camera-first rescoring and
all-camera rescoring. An arm whose sixteen directions equal the baseline's reuses the
saved baseline records with an identity proof instead of regenerating.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from common import (
    BASELINE_STAGES,
    CASE_IDS,
    LEGACY,
    MATCHER_ARMS,
    SAVED_ESTIMATORS,
    STAGES,
    code_md5,
    load_sources,
    md5,
    read,
    run_dir,
    write,
)
from rescore_camera_pool import rescore
from run_automatic import generate

ADAPTER_SCHEMA = 'direction-agreement-adapter/1'


def adapter(saved: dict, arm_record: dict, run: str, saved_md5: str, e2_md5: str) -> dict:
    """Saved-estimator stand-in carrying the arm's actual directions and their provenance.

    The matcher reads working_size, settings.pencil_selection and estimator.points_working.
    Everything else here documents where the replaced points came from; no field of the
    original estimator is copied unless it still describes these points.
    """
    estimator = saved['estimator']
    return {
        'working_size': saved['working_size'],
        'settings': saved['settings'],
        'estimator': {
            'schema': ADAPTER_SCHEMA, 'run': run, 'arm': arm_record['arm'],
            'anchor_rule': arm_record['anchor_rule'], 'representative_rule': arm_record['representative_rule'],
            'points_working': arm_record['points_working'],
            'retained_candidate_ids': arm_record['representative_candidate_ids'],
            'leader_candidate_ids': arm_record['leader_candidate_ids'],
            'retained_support_masks': arm_record['support_masks'],
            'retained_support_counts': arm_record['support_counts'],
            'direction_lines': estimator['direction_lines'],
            'normalised_to_working': estimator['normalised_to_working'],
            'candidate_ids': estimator['candidate_ids'],
            'allocation_candidate_status': arm_record['allocation_candidate_status'],
            'source_saved_estimator_md5': saved_md5, 'source_e2_record_md5': e2_md5,
            'note': 'Directions come from the E2 arm; the bank identity and merged lines are the saved estimator\'s.',
        },
    }


def stage_paths(output: Path, arm: str, case_id: str) -> dict[str, Path]:
    return {stage: output / 'e4' / arm / stage / f'{case_id}.json.gz' for stage in STAGES}


def reuse_baseline(
    root: Path, output: Path, arm: str, case_id: str, points: np.ndarray, run: str, code: dict, saved_md5: str, e2_md5: str,
) -> None:
    """Copy the saved baseline stages with a proof that the arm's directions equal the baseline's."""
    paths = stage_paths(output, arm, case_id)
    reused = {}
    for stage, directory in BASELINE_STAGES.items():
        source = root / directory / f'{case_id}.json.gz'
        record = read(source)
        record.update({'run': run, 'arm': arm, 'stage': stage, 'identity_reused_from': str(directory / f'{case_id}.json.gz'),
                       'identity_reused_md5': md5(source), 'experiment_code_md5': code,
                       'source_saved_estimator_md5': saved_md5, 'source_e2_record_md5': e2_md5})
        write(paths[stage], record)
        reused[stage] = {'from': str(directory / f'{case_id}.json.gz'), 'md5': md5(source)}
    write(output / 'e4' / arm / f'{case_id}_identity.json.gz', {
        'case_id': case_id, 'arm': arm, 'identical_to_baseline_directions': True,
        'points_working': points.tolist(), 'reused': reused, 'experiment_code_md5': code,
        'source_saved_estimator_md5': saved_md5, 'source_e2_record_md5': e2_md5})


def input_identity(record: dict) -> tuple[str | None, str | None]:
    """The saved-estimator and E2 hashes a stage record was produced from, wherever it stores them."""
    if 'identity_reused_from' in record:
        return record.get('source_saved_estimator_md5'), record.get('source_e2_record_md5')
    estimator = record.get('estimator', {})
    return estimator.get('source_saved_estimator_md5'), estimator.get('source_e2_record_md5')


def run_case(
    case_id: str, source: dict, saved: dict, arms_record: dict, arm: str, zone: object, root: Path, output: Path,
    run: str, code: dict, saved_md5: str, e2_md5: str,
) -> dict:
    arm_record = arms_record['arms'][arm]
    paths = stage_paths(output, arm, case_id)
    if not arm_record['matcher_eligible']:
        for stage, path in paths.items():
            write(path, {'case_id': case_id, 'run': run, 'arm': arm, 'stage': stage, 'entries': [], 'pairs': [],
                         'court_result': arm_record['court_result'], 'experiment_code_md5': code})
        return {'case_id': case_id, 'arm': arm, 'status': 'empty', 'reason': arm_record['court_result']['reason']}
    points = np.asarray(arm_record['points_working'], dtype=float)
    baseline_points = np.asarray(saved['estimator']['points_working'], dtype=float)
    if points.shape == baseline_points.shape and np.array_equal(points, baseline_points):
        reuse_baseline(root, output, arm, case_id, points, run, code, saved_md5, e2_md5)
        return {'case_id': case_id, 'arm': arm, 'status': 'identity_reused'}
    started = perf_counter()
    result = generate(source, adapter(saved, arm_record, run, saved_md5, e2_md5), zone, root)
    result.update({'run': run, 'arm': arm, 'stage': 'results', 'experiment_code_md5': code})
    write(paths['results'], result)
    generated = perf_counter()
    print(case_id, arm, 'generated', 'pooled', result['pooled_candidates'], 'entries', len(result['entries']),
          'winners', result['line_winner_id'], result['paint_winner_id'], 'seconds', round(generated - started, 1), flush=True)
    camera_first = rescore(source, result, zone, root, replay=True, keep_all_camera=False)
    camera_first['stage'] = 'camera_first'
    write(paths['camera_first'], camera_first)
    all_camera = rescore(source, camera_first, zone, root, replay=True, keep_all_camera=True)
    all_camera['stage'] = 'all_camera'
    write(paths['all_camera'], all_camera)
    return {'case_id': case_id, 'arm': arm, 'status': 'generated', 'generation_s': generated - started,
            'camera_first_s': camera_first['elapsed_s'], 'all_camera_s': all_camera['elapsed_s'],
            'entries': {stage: len(record['entries']) for stage, record in
                        (('results', result), ('camera_first', camera_first), ('all_camera', all_camera))},
            'winners': {stage: (record['line_winner_id'], record['paint_winner_id']) for stage, record in
                        (('results', result), ('camera_first', camera_first), ('all_camera', all_camera))}}


def complete(paths: dict[str, Path], code: dict, saved_md5: str, e2_md5: str) -> bool:
    """A case-arm is complete only under identical experiment code and identical E2 and estimator inputs."""
    for path in paths.values():
        if not path.exists():
            return False
        record = read(path)
        if record.get('experiment_code_md5') != code or input_identity(record) != (saved_md5, e2_md5):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--run', required=True)
    parser.add_argument('--arm', choices=MATCHER_ARMS, required=True)
    parser.add_argument('--ids', nargs='+', default=list(CASE_IDS))
    parser.add_argument('--resume', action='store_true', help='skip case-arms whose three stages carry the current code MD5')
    args = parser.parse_args()
    sys.path.insert(0, str((args.root / LEGACY).resolve()))
    zone = importlib.import_module('zone_net')
    cv2.setNumThreads(1)
    sources, _ = load_sources(args.root)
    output = run_dir(args.root, args.run)
    code = code_md5(args.root)
    for case_id in args.ids:
        saved_path = args.root / SAVED_ESTIMATORS / f'{case_id}.json.gz'
        e2_path = output / 'e2' / f'{case_id}.json.gz'
        saved_md5, e2_md5 = md5(saved_path), md5(e2_path)
        if args.resume and complete(stage_paths(output, args.arm, case_id), code, saved_md5, e2_md5):
            print(case_id, args.arm, 'complete under identical code and inputs; skipped', flush=True)
            continue
        saved = read(saved_path)
        arms_record = read(e2_path)
        assert saved['case_id'] == case_id and arms_record['case_id'] == case_id
        summary = run_case(case_id, sources[case_id], saved, arms_record, args.arm, zone, args.root, output, args.run,
                           code, saved_md5, e2_md5)
        print(case_id, args.arm, 'complete', summary, flush=True)


if __name__ == '__main__':
    main()
