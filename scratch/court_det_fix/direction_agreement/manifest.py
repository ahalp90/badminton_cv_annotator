"""Write the run manifest: population, code and input identity, versions, sizes and settings."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import scipy
from common import (
    BANK_CONTROLS,
    BASELINE_STAGES,
    CASES,
    GIVEN_FINITE,
    INPUT_PACKS,
    LEGACY,
    SAVED_ESTIMATORS,
    SVD_RECORDS,
    code_md5,
    load_sources,
    md5,
    read,
    run_dir,
    write,
)
from inspect_appearance import frame_path
from run_population import prepare

SHARED_MODULES = (
    'experiments/annotator/independent_court/assignment.py',
    'experiments/annotator/independent_court/detector.py',
    'experiments/annotator/independent_court/stripe_observations.py',
    'experiments/annotator/independent_court/fixed_stripe_refit.py',
    'src/courtkeynet/court_corners.py',
    'vp_pruning_20260914/vp_pruning.py',
    'vp_pruning_20260914/run_population.py',
    'vp_pruning_20260914/diagnose_targets.py',
    'marking_diagnosis_20260914/run_diagnosis.py',
    'marking_diagnosis_20260914/scan_population.py',
    'axis_matching_20260914/projective_seed.py',
    'axis_matching_20260914/run_given.py',
    'axis_matching_20260914/inspect_appearance.py',
    'automatic_axes_20260914/run_automatic.py',
    'automatic_axes_20260914/rescore_camera_pool.py',
    'automatic_axes_20260914/diagnose_directions.py',
    'automatic_axes_20260914/diagnose_direction_bank.py',
    'automatic_axes_20260914/diagnose_automatic.py',
    'automatic_axes_20260914/check_pool_replays.py',
    'automatic_axes_20260914/svd_fixed/run_svd_fixed.py',
    'smoke/legacy/zone_net.py',
)
EXPECTED_SETTINGS = {'angle_deg': 1.5, 'direction_lines': 128, 'pencils': 16, 'overlap': 0.8, 'pencil_selection': 'coverage'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--run', required=True)
    parser.add_argument('--git-head', required=True)
    parser.add_argument('--git-branch', required=True)
    parser.add_argument('--git-dirty-files', type=int, required=True)
    parser.add_argument('--git-checkpoint', required=True)
    parser.add_argument('--session-model', required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    sources, _ = load_sources(args.root)
    cases = []
    for case_id, pack in CASES:
        source = sources[case_id]
        saved_path = args.root / SAVED_ESTIMATORS / f'{case_id}.json.gz'
        saved = read(saved_path)
        _, _, working = prepare(source)
        image = frame_path(source, args.root)
        frame = cv2.imread(str(image))
        assert frame is not None, image
        inputs = {'saved_estimator': saved_path, 'given_finite': args.root / GIVEN_FINITE / f'{case_id}.json.gz'}
        for stage, directory in BASELINE_STAGES.items():
            inputs[f'baseline_{stage}'] = args.root / directory / f'{case_id}.json.gz'
        if case_id in BANK_CONTROLS:
            inputs['bank_control'] = args.root / BANK_CONTROLS[case_id]
        svd_record = args.root / SVD_RECORDS / f'{case_id}.json.gz'
        if svd_record.exists():
            inputs['svd_fixed_record'] = svd_record
        cases.append({
            'case_id': case_id, 'input_pack': pack, 'image': str(image.relative_to(args.root)), 'image_md5': md5(image),
            'image_size': [frame.shape[1], frame.shape[0]],
            'native_size': [source['dimensions']['width'], source['dimensions']['height']],
            'working_size': list(working), 'saved_working_size': saved['working_size'],
            'raw_fragments': len(source['segments_px']), 'feet_frames': len(source['all_feet_px']),
            'settings': saved['settings'],
            'settings_differ_from_expected': {name: saved['settings'][name] != value for name, value in EXPECTED_SETTINGS.items()},
            'inputs': {name: {'path': str(path.relative_to(args.root)), 'md5': md5(path), 'bytes': path.stat().st_size}
                       for name, path in inputs.items()},
        })
    manifest = {
        'schema': 'direction-agreement-manifest/1', 'run': args.run,
        'written_utc': datetime.now(UTC).isoformat(timespec='seconds'),
        'session_model': args.session_model,
        'git': {'head': args.git_head, 'branch': args.git_branch, 'dirty_tracked_files': args.git_dirty_files,
                'scientific_checkpoint': args.git_checkpoint},
        'versions': {'python': sys.version.split()[0], 'numpy': np.__version__, 'opencv': cv2.__version__,
                     'scipy': scipy.__version__},
        'threads': {'opencv': cv2.getNumThreads()},
        'shared_modules': {path: md5(args.root / path) for path in SHARED_MODULES},
        'experiment_code': code_md5(args.root),
        'input_packs': {pack: md5(args.root / pack) for pack in INPUT_PACKS},
        'legacy_zone_module': str(LEGACY / 'zone_net.py'),
        'cases': cases,
    }
    write(run_dir(args.root, args.run) / 'manifest.json.gz', manifest)
    print('manifest written for', len(cases), 'cases', flush=True)


if __name__ == '__main__':
    main()
