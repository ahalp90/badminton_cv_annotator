"""Rebuild a working test-run folder from the tracked records, then load all ten test bundles.

``materialise --output DIR`` copies ``results/test-run/`` into DIR, writes DIR in place of the
``SWEEP/`` placeholder, and copies each test bundle from ``results/models/`` into its export
folder. ``--use-current-checkout`` also writes this checkout's ``source_record()`` into the
selection record, which ``input_test`` needs before it runs inference from a later commit.

``check --output DIR`` loads all ten bundles through ``input_test``'s own checks: the selection
record, export settings and results, contact-stage records, seeds and bundle md5s.

Run from the repository root with ``PYTHONPATH=src:.``.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from experiments.annotator.heuristic_only_tune.run import source_record
from experiments.annotator.tree_simplification import input_test
from experiments.annotator.tree_simplification.analysis.common import (
    RESULTS,
    read_json,
    write_json_gz,
)

PLACEHOLDER = 'SWEEP/'
SELECTION = 'selection.json.gz'


def replace_placeholder(value: Any, root: str) -> Any:
    """Swap the run-folder placeholder at the start of every string for a real folder."""
    if isinstance(value, dict):
        return {key: replace_placeholder(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_placeholder(item, root) for item in value]
    if isinstance(value, str) and value.startswith(PLACEHOLDER):
        return root + value.removeprefix(PLACEHOLDER)
    return value


def materialise(results: Path, output: Path, *, use_current_checkout: bool) -> None:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'{output}: choose an empty or new folder')
    records = results / 'test-run'
    root = f'{output}/'
    for source in sorted(records.rglob('*.json.gz')):
        write_json_gz(output / source.relative_to(records), replace_placeholder(read_json(source), root))
    selection_path = output / SELECTION
    selection = read_json(selection_path)
    if use_current_checkout:
        current = source_record()
        if current['dirty'] is not False:
            raise ValueError('This checkout has uncommitted changes; input_test needs a clean checkout')
        selection['evaluation_source'] = current
        write_json_gz(selection_path, selection)
    for finalist in selection['finalists']:
        for seed in finalist['seeds']:
            bundle = output / 'candidates' / finalist['candidate'] / f'seed-{seed["offset"]}' / 'export' / 'bundle'
            if Path(seed['export']) != bundle.parent:
                raise ValueError(f'{seed["export"]}: export folder does not follow the tracked layout')
            source = results / 'models' / f'{finalist["candidate"]}-seed-{seed["offset"]}'
            shutil.copytree(source, bundle)
    print(f'Materialised the test run in {output}', flush=True)


def check(output: Path) -> None:
    selection_path = output.resolve() / SELECTION
    loaded = 0
    for finalist in read_json(selection_path)['finalists']:
        for seed in finalist['seeds']:
            _document, _finalist, selected, _models = input_test._load_test_bundle(
                selection_path, finalist['candidate'], seed['offset'],
            )
            print(f'{finalist["candidate"]} seed {seed["offset"]}: loaded, md5 {selected["bundle_md5"]["annotator_models"]}',
                  flush=True)
            loaded += 1
    if loaded != 10:
        raise ValueError(f'Expected ten test bundles, loaded {loaded}')
    print('All ten test bundles loaded and passed the test runner checks', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    materialise_parser = commands.add_parser('materialise', help='rebuild a working test-run folder')
    materialise_parser.add_argument('--output', type=Path, required=True)
    materialise_parser.add_argument('--results', type=Path, default=RESULTS, help='tracked results folder')
    materialise_parser.add_argument(
        '--use-current-checkout', action='store_true',
        help="write this checkout's source record into the selection, for inference at a later commit",
    )
    check_parser = commands.add_parser('check', help='load all ten bundles through the test runner checks')
    check_parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'materialise':
        materialise(args.results, args.output, use_current_checkout=args.use_current_checkout)
    else:
        check(args.output)


if __name__ == '__main__':
    main()
