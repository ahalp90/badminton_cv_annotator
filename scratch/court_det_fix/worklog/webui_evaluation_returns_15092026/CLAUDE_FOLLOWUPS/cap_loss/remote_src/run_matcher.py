"""Cap-loss E4: regenerate one case-arm through the instrumented matcher and record its per-pair pool.

cap_loss copy of direction_agreement/run_matcher.py. Differences from the original: it imports
generate from the run_automatic.py in this folder, always regenerates (no identity reuse, because
arm B's directions equal the saved estimator's and the recording still has to run), skips both
rescoring stages, reads the E2 records from the direction-agreement run and writes under
cap_loss/runs/<run>/. The adapter is unchanged so the generation record can be compared with the
saved one field by field.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

import run_automatic
from common import (
    ARMS,
    CASE_IDS,
    EXPERIMENT_DIR,
    LEGACY,
    SAVED_ESTIMATORS,
    code_md5,
    load_sources,
    md5,
    read,
    run_dir,
    write,
)
from run_automatic import generate

ADAPTER_SCHEMA = 'direction-agreement-adapter/1'
E2_RUN_ROOT = Path('direction_agreement/runs')


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


def record_paths(output: Path, arm: str, case_id: str) -> dict[str, Path]:
    """Generation record and its pool recording, in the same e4/<arm>/results layout as the original."""
    directory = output / 'e4' / arm / 'results'
    return {'results': directory / f'{case_id}.json.gz', 'pool': directory / f'{case_id}_pool.npz'}


def run_case(
    case_id: str, source: dict, saved: dict, arms_record: dict, arm: str, zone: object, root: Path, output: Path,
    run: str, code: dict, saved_md5: str, e2_md5: str,
) -> dict:
    arm_record = arms_record['arms'][arm]
    assert arm_record['matcher_eligible'], (case_id, arm, arm_record['court_result'])
    paths = record_paths(output, arm, case_id)
    points = np.asarray(arm_record['points_working'], dtype=float)
    baseline_points = np.asarray(saved['estimator']['points_working'], dtype=float)
    identical_directions = points.shape == baseline_points.shape and np.array_equal(points, baseline_points)
    started = perf_counter()
    result = generate(source, adapter(saved, arm_record, run, saved_md5, e2_md5), zone, root, pool_path=paths['pool'])
    result.update({'run': run, 'arm': arm, 'stage': 'results', 'experiment_code_md5': code,
                   'directions_identical_to_saved_estimator': identical_directions,
                   'pool_recording': str(paths['pool'].relative_to(output))})
    write(paths['results'], result)
    generated = perf_counter()
    print(case_id, arm, 'generated', 'pooled', result['pooled_candidates'], 'entries', len(result['entries']),
          'winners', result['line_winner_id'], result['paint_winner_id'], 'seconds', round(generated - started, 1), flush=True)
    return {'case_id': case_id, 'arm': arm, 'status': 'generated', 'generation_s': generated - started,
            'directions_identical_to_saved_estimator': identical_directions,
            'entries': len(result['entries']), 'winners': (result['line_winner_id'], result['paint_winner_id'])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--run', required=True, help='cap_loss_<UTC timestamp>')
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--ids', nargs='+', default=list(CASE_IDS))
    parser.add_argument('--e2-run', default='direction_agreement_20260915_144900',
                        help='direction-agreement run whose E2 records supply the arm directions')
    args = parser.parse_args()
    assert args.run.startswith('cap_loss_'), args.run
    # The instrumented matcher must be this folder's copy, never the sibling on PYTHONPATH.
    assert Path(run_automatic.__file__).resolve().parent == Path(__file__).resolve().parent, run_automatic.__file__
    assert EXPERIMENT_DIR == Path('cap_loss'), EXPERIMENT_DIR
    print('run_automatic from', run_automatic.__file__, flush=True)
    sys.path.insert(0, str((args.root / LEGACY).resolve()))
    zone = importlib.import_module('zone_net')
    cv2.setNumThreads(1)
    sources, _ = load_sources(args.root)
    output = run_dir(args.root, args.run)
    code = code_md5(args.root)
    for case_id in args.ids:
        saved_path = args.root / SAVED_ESTIMATORS / f'{case_id}.json.gz'
        e2_path = args.root / E2_RUN_ROOT / args.e2_run / 'e2' / f'{case_id}.json.gz'
        saved_md5, e2_md5 = md5(saved_path), md5(e2_path)
        saved = read(saved_path)
        arms_record = read(e2_path)
        assert saved['case_id'] == case_id and arms_record['case_id'] == case_id
        summary = run_case(case_id, sources[case_id], saved, arms_record, args.arm, zone, args.root, output, args.run,
                           code, saved_md5, e2_md5)
        print(case_id, args.arm, 'complete', summary, flush=True)


if __name__ == '__main__':
    main()
