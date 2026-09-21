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
import repair_inputs
from rescore_camera_pool import rescore
from run_automatic import generate

from shared import case_provenance, read, require_same_image_boxes, write

STAGES = ('results', 'camera_first', 'all_camera')
LEGACY = Path('smoke/legacy')
HERE = Path(__file__).resolve().parent
PERSON_BOX_ARMS = frozenset({'person', 'person_observations', 'paint_person'})


def preflight_person_input(
    case_id: str,
    arm: str,
    inputs_dir: Path | None = None,
    repair_manifest: dict | None = None,
) -> None:
    """Validate one person input before importing the matcher zone or generating anything."""
    if repair_manifest is not None:
        if arm != repair_inputs.ARM:
            raise ValueError(f'--repair-manifest is only valid for {repair_inputs.ARM}')
        if inputs_dir is None:
            raise ValueError('--repair-manifest requires --inputs-dir')
        repair_inputs.validate_repair_case(case_id, inputs_dir, repair_manifest)
    elif arm in PERSON_BOX_ARMS:
        require_same_image_boxes(case_provenance(case_id))


def preflight_person_inputs(
    case_ids: list[str],
    arm: str,
    inputs_dir: Path | None = None,
    repair_manifest: dict | None = None,
) -> None:
    """Check every selected case before a multi-case matcher run starts."""
    if repair_manifest is not None:
        if arm != repair_inputs.ARM:
            raise ValueError(f'--repair-manifest is only valid for {repair_inputs.ARM}')
        if inputs_dir is None:
            raise ValueError('--repair-manifest requires --inputs-dir')
        repair_inputs.validate_repair_inputs(case_ids, inputs_dir, repair_manifest)
        return
    for case_id in case_ids:
        preflight_person_input(case_id, arm, inputs_dir, repair_manifest)


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def code_md5() -> dict[str, str]:
    return {path.name: md5(path) for path in sorted(HERE.glob('*.py'))}


def preflight_repair_outputs(
    case_ids: list[str],
    arm: str,
    output: Path,
    repair_manifest: dict | None,
) -> None:
    """Reject existing repair targets before importing the matcher or generating any result."""
    if repair_manifest is None:
        return
    if arm != repair_inputs.ARM:
        raise ValueError(f'--repair-manifest is only valid for {repair_inputs.ARM}')
    collisions = []
    for case_id in case_ids:
        for stage in STAGES:
            path = output / 'matcher' / arm / stage / f'{case_id}.json.gz'
            if path.exists():
                collisions.append(path)
    if collisions:
        paths = ', '.join(str(path) for path in collisions)
        raise FileExistsError(f'repair output paths already exist: {paths}')


def run_case(
    case_id: str,
    arm: str,
    zone: object,
    root: Path,
    output: Path,
    run: str,
    code: dict,
    inputs_dir: Path | None = None,
    repair_manifest: dict | None = None,
) -> dict:
    inputs_dir = HERE / 'inputs' if inputs_dir is None else inputs_dir
    preflight_person_input(case_id, arm, inputs_dir, repair_manifest)
    case_path = inputs_dir / arm / 'cases' / f'{case_id}.json.gz'
    estimator_path = inputs_dir / arm / 'estimators' / f'{case_id}.json.gz'
    source, saved = read(case_path), read(estimator_path)
    if (source.get('id') != case_id or saved.get('case_id') != case_id or saved.get('arm') != arm):
        raise ValueError(f'{case_id} {arm}: input identity does not match the requested case')
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
    parser.add_argument('--inputs-dir', type=Path, default=HERE / 'inputs')
    parser.add_argument('--repair-manifest', type=Path)
    args = parser.parse_args()
    if args.repair_manifest is not None and args.arm != repair_inputs.ARM:
        parser.error(f'--repair-manifest is only valid for {repair_inputs.ARM}')
    repair_manifest = None if args.repair_manifest is None else repair_inputs.load_manifest(args.repair_manifest)
    preflight_person_inputs(args.ids, args.arm, args.inputs_dir, repair_manifest)
    output = HERE / 'runs' / args.run
    preflight_repair_outputs(args.ids, args.arm, output, repair_manifest)
    sys.path.insert(0, str((args.root / LEGACY).resolve()))
    zone = importlib.import_module('zone_net')
    cv2.setNumThreads(1)
    code = code_md5()
    print('run_automatic from', Path(generate.__code__.co_filename).resolve(), flush=True)
    for case_id in args.ids:
        summary = run_case(case_id, args.arm, zone, args.root, output, args.run, code,
                           args.inputs_dir, repair_manifest)
        print(case_id, args.arm, 'complete', summary, flush=True)


if __name__ == '__main__':
    main()
