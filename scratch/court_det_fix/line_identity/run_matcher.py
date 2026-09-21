"""Feed one filter arm's fragments and directions through the unchanged matcher on the compute host.

For each view the arm supplies a filtered pack entry (`inputs/<arm>/cases/<case>.json.gz`, the
frozen pack entry with the dropped fragments removed and nothing else changed) and the direction
record the unchanged coverage selection produced from those fragments
(`inputs/<arm>/estimators/<case>.json.gz`, saved-estimator form). `run_automatic.generate` then
runs exactly as it does for the baseline, followed by the two rescoring stages, and the three
records land under `runs/<run>/matcher/<arm>/{results,camera_first,all_camera}/`.

Run through run_remote.sh so the single-thread environment, PYTHONPATH and receipts are the same
as the direction experiment's.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import sys
from pathlib import Path
from time import perf_counter

import cv2
from rescore_camera_pool import rescore
from run_automatic import generate

from shared import case_provenance, read, require_same_image_boxes, write

STAGES = ('results', 'camera_first', 'all_camera')
LEGACY = Path('smoke/legacy')
HERE = Path(__file__).resolve().parent
PERSON_BOX_ARMS = frozenset({'person', 'person_observations', 'paint_person'})


def preflight_person_input(case_id: str, arm: str) -> None:
    """Reject person-filtered inputs when boxes do not describe the measured image."""
    if arm in PERSON_BOX_ARMS:
        require_same_image_boxes(case_provenance(case_id))


def preflight_person_inputs(case_ids: list[str], arm: str) -> None:
    """Check every selected case before a multi-case matcher run starts."""
    for case_id in case_ids:
        preflight_person_input(case_id, arm)


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def code_md5() -> dict[str, str]:
    return {path.name: md5(path) for path in sorted(HERE.glob('*.py'))}


def run_case(case_id: str, arm: str, zone: object, root: Path, output: Path, run: str, code: dict) -> dict:
    preflight_person_input(case_id, arm)
    case_path = HERE / 'inputs' / arm / 'cases' / f'{case_id}.json.gz'
    estimator_path = HERE / 'inputs' / arm / 'estimators' / f'{case_id}.json.gz'
    source, saved = read(case_path), read(estimator_path)
    assert source['id'] == case_id and saved['case_id'] == case_id and saved['arm'] == arm, (case_id, arm)
    provenance = {'run': run, 'arm': arm, 'experiment_code_md5': code,
                  'input_case_md5': md5(case_path), 'input_estimator_md5': md5(estimator_path),
                  'fragments_kept': saved['fragments_kept'], 'fragments_total': saved['fragments_total']}
    paths = {stage: output / 'matcher' / arm / stage / f'{case_id}.json.gz' for stage in STAGES}
    started = perf_counter()
    result = generate(source, saved, zone, root)
    result.update({**provenance, 'stage': 'results'})
    write(paths['results'], result)
    generated = perf_counter()
    print(case_id, arm, 'generated', 'pooled', result['pooled_candidates'], 'entries', len(result['entries']),
          'winners', result['line_winner_id'], result['paint_winner_id'], 'seconds', round(generated - started, 1), flush=True)
    camera_first = rescore(source, result, zone, root, replay=True, keep_all_camera=False)
    camera_first.update({**provenance, 'stage': 'camera_first'})
    write(paths['camera_first'], camera_first)
    all_camera = rescore(source, camera_first, zone, root, replay=True, keep_all_camera=True)
    all_camera.update({**provenance, 'stage': 'all_camera'})
    write(paths['all_camera'], all_camera)
    return {'case_id': case_id, 'arm': arm, 'status': 'generated', 'generation_s': generated - started,
            'camera_first_s': camera_first['elapsed_s'], 'all_camera_s': all_camera['elapsed_s'],
            'winners': {stage: (record['line_winner_id'], record['paint_winner_id']) for stage, record in
                        (('results', result), ('camera_first', camera_first), ('all_camera', all_camera))}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--run', required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--ids', nargs='+', required=True)
    args = parser.parse_args()
    preflight_person_inputs(args.ids, args.arm)
    sys.path.insert(0, str((args.root / LEGACY).resolve()))
    zone = importlib.import_module('zone_net')
    cv2.setNumThreads(1)
    output = HERE / 'runs' / args.run
    code = code_md5()
    print('run_automatic from', Path(generate.__code__.co_filename).resolve(), flush=True)
    for case_id in args.ids:
        summary = run_case(case_id, args.arm, zone, args.root, output, args.run, code)
        print(case_id, args.arm, 'complete', summary, flush=True)


if __name__ == '__main__':
    main()
