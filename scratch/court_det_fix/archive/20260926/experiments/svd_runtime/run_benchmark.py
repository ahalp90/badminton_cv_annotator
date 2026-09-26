"""Time the frozen pair matcher on all 16 and SVD-ranked top 12 B directions.

This measures matcher-stage work only. It excludes image loading, full court
scoring, scene processing and detector timing. --max-pairs creates a smoke run.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import socket
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import permutations
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter, process_time

for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[name] = '1'

import cv2
import numpy as np

PREFIX = Path('scratch/court_det_fix')
HELPERS = PREFIX / 'frozen_helpers_20260914'
SCREEN = PREFIX / 'evidence/webui_followup3_20260922/received/followup3_svd_screen/run_screen.py'
REVIEW = PREFIX / 'evidence/webui_followup3_20260922/review_20260923/replay/results.json.gz'
E2 = PREFIX / 'direction_agreement/runs/direction_agreement_20260915_144900/e2'
PACKS = PREFIX / 'frozen_views/packs'
BASELINE = PREFIX / 'frozen_views/baseline_directions'
PACK_FOR_CASE = {
    'gxBQ_window_00_frame_0': 'gx_extension_inputs.json.gz',
    'gxBQ_window_00_frame_5': 'gx_extension_inputs.json.gz',
    'am2_window_00_frame_150': 'marking_refit_inputs.json.gz',
    'am2_window_01_frame_28019': 'marking_refit_inputs.json.gz',
    'am3_window_00_frame_0': 'marking_refit_inputs.json.gz',
    'shuttleset_03_scene_0017': 'broadcast_extension_inputs.json.gz',
    'shuttleset_03_scene_0019': 'broadcast_extension_inputs.json.gz',
    'shuttleset_03_scene_0016': 'broadcast_extension_inputs.json.gz',
    'shuttleset_21_scene_0020': 'broadcast_extension_inputs.json.gz',
}
MODULES = (
    'automatic_axes/run_automatic.py', 'axis_matching/run_given.py',
    'axis_matching/projective_seed.py', 'marking_diagnosis/scan_population.py',
    'vp_pruning/run_population.py', 'vp_pruning/vp_pruning.py',
    'legacy/zone_net.py', 'legacy/camera_diagnostic.py',
)


def read(path: Path) -> dict:
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def digest(path: Path) -> dict:
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_bytes(gzip.compress(json.dumps(value, allow_nan=False, separators=(',', ':')).encode(), mtime=0))
    temporary.replace(path)


def setup(root: Path) -> tuple[object, object, object, object, object, object]:
    for directory in ('legacy', 'axis_matching', 'marking_diagnosis', 'vp_pruning', 'automatic_axes'):
        sys.path.insert(0, str((root / HELPERS / directory).resolve()))
    sys.path.insert(0, str((root / 'src').resolve()))
    sys.path.insert(0, str(root.resolve()))
    sys.path.insert(0, str((root / SCREEN.parent).resolve()))
    import run_automatic
    import run_screen
    import zone_net
    from projective_seed import Settings
    from run_given import propose_role
    from run_population import prepare

    from experiments.annotator.independent_court import assignment
    cv2.setNumThreads(1)
    return run_screen, run_automatic, zone_net, Settings, propose_role, (prepare, assignment)


def equal(expected: object, actual: object, label: str) -> None:
    if expected == actual:
        return
    if (isinstance(expected, (int, float)) and not isinstance(expected, bool)
            and isinstance(actual, (int, float)) and np.isfinite(expected) and np.isfinite(actual)
            and abs(expected - actual) <= 1e-8):
        return
    if isinstance(expected, dict) and isinstance(actual, dict) and expected.keys() == actual.keys():
        for key in expected:
            equal(expected[key], actual[key], f'{label}.{key}')
        return
    if isinstance(expected, list) and isinstance(actual, list) and len(expected) == len(actual):
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            equal(left, right, f'{label}[{index}]')
        return
    raise AssertionError(f'Output changed at {label}: {str(expected)[:100]} != {str(actual)[:100]}')


def comparable(pair: dict) -> dict:
    return {key: value for key, value in pair.items() if key not in ('elapsed_s', 'process_s')}


def run_arm(case_id: str, arm: str, points: np.ndarray, chosen: set[int], observations: object,
            feet: np.ndarray, size: tuple[int, int], native_size: tuple[int, int], scale: np.ndarray,
            zone: object, Settings: object, propose_role: object, automatic: object,
            max_pairs: int | None) -> dict:
    settings = Settings(keep_axes=512)
    point_scale = np.append(scale, 1.)
    records = []
    started, cpu_started = perf_counter(), process_time()
    for pair_id, pencil_ids in enumerate(permutations(range(16), 2)):
        if max_pairs is not None and pair_id >= max_pairs:
            break
        if not set(pencil_ids) <= chosen:
            records.append({'pair_id': pair_id, 'pencils': list(pencil_ids), 'status': 'skipped_svd_mask'})
            continue
        pair_start, pair_cpu = perf_counter(), process_time()
        pair_points = points[list(pencil_ids)]
        bound = automatic.camera_direction_bound(pair_points * point_scale, native_size)
        record = {'pair_id': pair_id, 'pencils': list(pencil_ids), 'camera_direction_bound': bound}
        if bound > automatic.CAMERA_ERROR_LIMIT + automatic.CAMERA_ROUNDING_MARGIN:
            record['status'] = 'camera_direction_bound'
        else:
            proposed = propose_role(pair_points, observations, feet, size, settings, zone)
            details = {}
            for index, (candidate, provenance) in enumerate(zip(proposed.candidates, proposed.details, strict=True)):
                details[id(candidate)] = {'candidate_id': f'{pair_id}:{index}', 'pair_id': pair_id, **provenance}
            retained = automatic.select_pool(proposed.candidates)
            shortlist = []
            for candidate in retained:
                shortlist.append({**details[id(candidate)], 'corners_px': (candidate.corners_px * scale).tolist(),
                                  'shortlist_score': candidate.score})
            record.update({'status': 'matched', 'role': proposed.record, 'shortlist': shortlist,
                           'proposed_count': len(proposed.candidates), 'retained_count': len(retained)})
        record['elapsed_s'] = perf_counter() - pair_start
        record['process_s'] = process_time() - pair_cpu
        records.append(record)
        print(case_id, arm, 'pair', pair_id, record['status'], flush=True)
    return {'arm': arm, 'chosen_direction_ids': sorted(chosen), 'pairs': records,
            'elapsed_s': perf_counter() - started, 'process_s': process_time() - cpu_started}


def run_task(root_text: str, output_text: str, case_id: str, case_index: int, repeat: int,
             max_pairs: int | None, cache_text: str | None) -> dict:
    root, output = Path(root_text), Path(output_text)
    screen, automatic, zone, Settings, propose_role, (prepare, assignment) = setup(root)
    pack_path = root / PACKS / PACK_FOR_CASE[case_id]
    baseline_path = root / BASELINE / f'{case_id}.json.gz'
    e2_path = root / E2 / f'{case_id}.json.gz'
    review_path = root / REVIEW
    task_started = perf_counter()
    prepare_started, prepare_cpu = perf_counter(), process_time()
    source = next(item for item in read(pack_path)['cases'] if item['id'] == case_id)
    baseline, e2 = read(baseline_path), read(e2_path)
    review = next(item for item in read(review_path)['cases'] if item['case_id'] == case_id)
    b = e2['arms']['B']
    assert baseline['working_size'] == e2['working_size']
    assert baseline['settings']['pencil_selection'] == 'coverage'
    assert baseline['estimator']['points_working'] == b['points_working']
    assert len(b['points_working']) == len(b['support_masks']) == 16
    segments, _, size = prepare(source)
    assert list(size) == baseline['working_size']
    observations = assignment.prepare_observations(segments, size)
    native_size = (source['dimensions']['width'], source['dimensions']['height'])
    scale = np.asarray(native_size) / size
    feet = np.asarray([[[np.nan, np.nan] if foot is None else foot for foot in frame]
                       for frame in source['all_feet_px']], dtype=float) / scale
    points = np.asarray(b['points_working'], dtype=float)
    preparation = {'elapsed_s': perf_counter() - prepare_started, 'process_s': process_time() - prepare_cpu}
    svd_started, svd_cpu = perf_counter(), process_time()
    fit_groups = screen.load_fit_groups(root)
    estimator = baseline['estimator']
    lines = np.asarray(estimator['direction_lines'], dtype=float)
    transform = np.asarray(estimator['normalised_to_working'], dtype=float)
    masks = np.asarray(b['support_masks'], dtype=bool)
    groups = screen.fit_diagnostics(lines @ transform, masks, fit_groups)
    rank = screen.rank_groups(groups)
    svd_time = {'elapsed_s': perf_counter() - svd_started, 'process_s': process_time() - svd_cpu}
    assert rank == review['rank_order'], case_id
    chosen = set(rank[:12])
    order = ('full16', 'svd12') if (case_index + repeat) % 2 == 0 else ('svd12', 'full16')
    arms = {}
    for arm in order:
        ids = set(range(16)) if arm == 'full16' else chosen
        arms[arm] = run_arm(case_id, arm, points, ids, observations, feet, size, native_size, scale,
                            zone, Settings, propose_role, automatic, max_pairs)
    for full, retained in zip(arms['full16']['pairs'], arms['svd12']['pairs'], strict=True):
        if retained['status'] != 'skipped_svd_mask':
            equal(comparable(full), comparable(retained), f'{case_id}.pair.{full["pair_id"]}')
    cache_result = None
    if cache_text:
        cache_path = Path(cache_text) / f'{case_id}.json.gz'
        cached = read(cache_path)
        for arm in ('full16', 'svd12'):
            for pair in arms[arm]['pairs']:
                if pair['status'] == 'skipped_svd_mask':
                    continue
                old = cached['pairs'][pair['pair_id']]
                comparable_now = comparable(pair)
                comparable_now.pop('proposed_count', None)
                comparable_now.pop('retained_count', None)
                equal(comparable(old), comparable_now, f'{case_id}.cache.{arm}.pair.{pair["pair_id"]}')
        cache_result = {'path': str(cache_path.resolve()), 'sha256': digest(cache_path)['sha256'],
                        'compared_pairs': {arm: sum(pair['status'] != 'skipped_svd_mask'
                                                    for pair in arms[arm]['pairs']) for arm in arms}}
    module_paths = [root / HELPERS / relative for relative in MODULES]
    module_paths += [root / SCREEN, root / screen.SVD_SOURCE,
                     root / 'experiments/annotator/independent_court/assignment.py',
                     root / 'experiments/annotator/independent_court/detector.py', Path(__file__)]
    result = {'schema': 'svd-matcher-runtime/1', 'scope': 'camera bound, propose_role and per-pair select_pool only',
              'case_id': case_id, 'repeat': repeat, 'smoke': max_pairs is not None, 'max_pairs': max_pairs,
              'provenance': {'modules': [digest(path) for path in module_paths],
                             'inputs': [digest(path) for path in (pack_path, baseline_path, e2_path, review_path)],
                             'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                                             'opencv': cv2.__version__, 'host': socket.gethostname(),
                                             'cv2_threads': cv2.getNumThreads(),
                                             'native_thread_env': {name: os.environ[name] for name in (
                                                 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
                                                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')}}},
              'rank_order': rank, 'selected_original_direction_ids': sorted(chosen), 'arm_order': order,
              'preparation': preparation, 'svd_compute_and_rank': svd_time, 'arms': arms,
              'historical_cache_check': cache_result, 'task_elapsed_s': perf_counter() - task_started}
    path = output / f'{case_id}_repeat_{repeat:02d}.json.gz'
    write(path, result)
    return {'case_id': case_id, 'repeat': repeat, 'file': str(path), 'smoke': max_pairs is not None,
            'preparation_s': preparation['elapsed_s'], 'svd_s': svd_time['elapsed_s'],
            'full16_s': arms['full16']['elapsed_s'], 'svd12_s': arms['svd12']['elapsed_s'],
            'task_elapsed_s': result['task_elapsed_s']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Repository checkout')
    parser.add_argument('--output', type=Path, required=True, help='New or empty result directory')
    parser.add_argument('--ids', nargs='+', choices=tuple(PACK_FOR_CASE), default=list(PACK_FOR_CASE))
    parser.add_argument('--workers', type=int, choices=range(1, 7), default=1)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--max-pairs', type=int, help='Explicit smoke limit on original pair IDs')
    parser.add_argument('--baseline-cache-dir', type=Path, help='Optional historical automatic result directory')
    args = parser.parse_args()
    if args.repeats < 1 or args.max_pairs is not None and not 1 <= args.max_pairs <= 240:
        parser.error('--repeats must be positive and --max-pairs must be 1..240')
    if len(args.ids) != len(set(args.ids)):
        parser.error('--ids must be unique')
    root, output = args.root.resolve(), args.output.resolve()
    if not root.is_dir() or output.exists() and any(output.iterdir()):
        parser.error('Root must exist and output must be new or empty')
    output.mkdir(parents=True, exist_ok=True)
    batch_started = perf_counter()
    tasks = [(str(root), str(output), case_id, list(PACK_FOR_CASE).index(case_id), repeat,
              args.max_pairs, str(args.baseline_cache_dir.resolve()) if args.baseline_cache_dir else None)
             for case_id in args.ids for repeat in range(args.repeats)]
    rows = []
    if args.workers == 1:
        for task in tasks:
            rows.append(run_task(*task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as executor:
            futures = [executor.submit(run_task, *task) for task in tasks]
            for future in as_completed(futures):
                rows.append(future.result())
    rows.sort(key=lambda row: (list(PACK_FOR_CASE).index(row['case_id']), row['repeat']))
    summary = {'schema': 'svd-matcher-runtime-summary/1', 'smoke': args.max_pairs is not None,
               'workers': args.workers, 'repeats': args.repeats, 'max_pairs': args.max_pairs,
               'batch_wall_s': perf_counter() - batch_started,
               'sum_task_elapsed_s': sum(row['task_elapsed_s'] for row in rows), 'tasks': rows}
    write(output / 'summary.json.gz', summary)
    print('SMOKE' if args.max_pairs is not None else 'FULL', 'batch wall seconds', summary['batch_wall_s'], flush=True)


if __name__ == '__main__':
    main()
