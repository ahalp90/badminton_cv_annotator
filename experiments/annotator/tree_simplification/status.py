"""Print available pilot and simplified-stack results without loading model files."""

import argparse
import gzip
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def configuration_records(root: Path) -> dict[str, dict[str, Any]]:
    """Return the latest visible state for each search configuration."""
    aggregate_path = root / 'status.json.gz'
    aggregate = read_json(aggregate_path) if aggregate_path.exists() else {}
    completed = {record['name']: record for record in aggregate.get('completed', ())}
    records = {}
    for path in sorted(root.rglob('status.json.gz')):
        if path == aggregate_path:
            continue
        name = str(path.parent.relative_to(root))
        current = read_json(path)
        # A running per-configuration record is newer than the prior search's
        # aggregate; once it exits, the aggregate carries the latest result.
        records[name] = current if current.get('state') == 'running' else completed.get(path.parent.name, current)
    for record in completed.values():
        records.setdefault(record['name'], record)
    return records


def configuration_state(record: dict[str, Any], result: dict[str, Any] | None) -> str:
    if record.get('state') == 'running':
        return 'running'
    if 'exit_code' not in record:
        return str(record.get('state', 'unknown'))
    exit_code = int(record['exit_code'])
    if exit_code != 0:
        return f'failed (exit {exit_code})'
    if record.get('resumed_complete') or (result is not None and result.get('complete', True)):
        return 'complete (exit 0)'
    return 'exit 0; result incomplete'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--tolerance', choices=('5', '10'), default='10')
    args = parser.parse_args()
    print(f'Validation, tolerance {args.tolerance} frames at 30 fps')
    print('| Configuration | Models | Correct rallies | Recovery | Player contact F1 | Result |')
    print('|---|---:|---:|---:|---:|---|')
    results = {}
    for path in sorted(args.root.rglob('result.json.gz')):
        result = read_json(path)
        name = str(path.parent.relative_to(args.root))
        results[path.parent.name] = result
        if 'layers' not in result and len(result.get('per_video', {})) != 8:
            continue
        rows = (
            [(int(depth) + 1, totals) for depth, totals in result['layers'].items()]
            if 'layers' in result else [(8, result['totals'])]
        )
        for count, totals in rows:
            scores = totals[args.tolerance]
            result_state = 'complete' if result.get('complete', True) else 'partial'
            print(
                f'| {name} | {count} | {scores["fully_correct_rallies"]}/{scores["labelled_rallies"]} '
                f'| {scores["fully_correct_rallies_percent"]:.2f}% | {100 * scores["attributed_f1"]:.2f}% '
                f'| {result_state} |'
            )
    print('\nConfiguration status:')
    for name, record in sorted(configuration_records(args.root).items()):
        print(f'{name}: {configuration_state(record, results.get(Path(name).name))}')
    print('\nLatest progress:')
    for path in sorted(args.root.rglob('progress.json.gz')):
        result_path = path.parent / 'result.json.gz'
        if result_path.exists() and read_json(result_path).get('complete', True):
            continue
        progress = read_json(path)
        name = path.parent.relative_to(args.root)
        details = {key: value for key, value in progress.items() if key not in ('totals', 'stage')}
        print(f'{name}: {progress["stage"]} {details}')
    for path in sorted(args.root.glob('*.exit')):
        print(f'{path.stem}: exit {path.read_text().strip()}')


if __name__ == '__main__':
    main()
