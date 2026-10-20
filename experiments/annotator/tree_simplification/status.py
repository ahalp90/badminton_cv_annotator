"""Print available pilot and simplified-stack results without loading model files."""

import argparse
import gzip
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--tolerance', choices=('5', '10'), default='10')
    args = parser.parse_args()
    print(f'Validation, tolerance {args.tolerance} frames at 30 fps')
    print('| Configuration | Models | Correct rallies | Recovery | Player contact F1 |')
    print('|---|---:|---:|---:|---:|')
    for path in sorted(args.root.rglob('result.json.gz')):
        result = read_json(path)
        name = str(path.parent.relative_to(args.root))
        if 'layers' not in result and len(result.get('per_video', {})) != 8:
            continue
        rows = (
            [(int(depth) + 1, totals) for depth, totals in result['layers'].items()]
            if 'layers' in result else [(8, result['totals'])]
        )
        for count, totals in rows:
            scores = totals[args.tolerance]
            print(
                f'| {name} | {count} | {scores["fully_correct_rallies"]}/{scores["labelled_rallies"]} '
                f'| {scores["fully_correct_rallies_percent"]:.2f}% | {100 * scores["attributed_f1"]:.2f}% |'
            )
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
