"""Print the filter replay's two result tables as Markdown, for pasting into evidence.md and results.md.

Reads runs/<run>/table.csv (direction stage) and axis_table.csv (axis stage). Values are copied,
not recomputed, so the prose tables match the CSVs by construction.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

DIRECTION_ARMS = ('baseline', 'person', 'paint', 'paint_person', 'paint15', 'paint25')
AXIS_ARMS = ('baseline', 'paint_observations', 'person_observations', 'person', 'paint', 'paint_person')
VIEW_ORDER = ('GX0', 'GX5', 'Am2-150', 'Am2-28019', 'Am3-0', 'SS03-17', 'SS03-19', 'SS03-16', 'SS21-20')


def value(text: str, digits: int = 1) -> str:
    if text in ('', 'None'):
        return 'none'
    return f'{float(text):.{digits}f}'


def direction_table(rows: list[dict]) -> str:
    by_key = {(row['label'], row['arm']): row for row in rows}
    lines = ['| View | ' + ' | '.join(f'{arm}: bound / fit' for arm in DIRECTION_ARMS) + ' |',
             '|' + ' --- |' * (len(DIRECTION_ARMS) + 1)]
    for view in VIEW_ORDER:
        cells = [view]
        for arm in DIRECTION_ARMS:
            row = by_key[(view, arm)]
            cells.append(f"{value(row['bound_px'])} / {value(row['best_fit_px'])}")
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.append('')
    lines.append('| View | ' + ' | '.join(f'{arm}: dropped (on markings)' for arm in DIRECTION_ARMS[1:]) + ' | fragments (on markings) |')
    lines.append('|' + ' --- |' * (len(DIRECTION_ARMS) + 1))
    for view in VIEW_ORDER:
        cells = [view]
        for arm in DIRECTION_ARMS[1:]:
            row = by_key[(view, arm)]
            cells.append(f"{row['dropped']} ({row['marking_dropped']})")
        base = by_key[(view, 'baseline')]
        cells.append(f"{base['fragments']} ({base['marking_fragments']})")
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def ladder(side: dict | None) -> tuple[str, str]:
    """The best distinct and best kept courts of one direction with their ranks, or none."""
    if side is None or side['best_distinct_px'] in ('', 'None'):
        return 'none', 'none'
    return (f"{value(side['best_distinct_px'])} ({side['best_distinct_rank']} of {side['distinct']})",
            f"{value(side['best_kept_px'])} ({side['best_kept_rank']})")


def axis_table(rows: list[dict]) -> str:
    by_key = {(row['label'], row['arm'], row['axis']): row for row in rows}
    header = ('| View | Arm | Pair | Direction fit | Horizontal best distinct (rank) | Horizontal best kept (rank) | '
              'Vertical best distinct (rank) | Vertical best kept (rank) | Kept by kept | No cap |')
    lines = [header, '|' + ' --- |' * 10]
    for view in VIEW_ORDER:
        for arm in AXIS_ARMS:
            horizontal = by_key.get((view, arm, 'horizontal'))
            vertical = by_key.get((view, arm, 'vertical'))
            if horizontal is None and vertical is None:
                continue
            row = horizontal or vertical
            h_distinct, h_kept = ladder(horizontal)
            v_distinct, v_kept = ladder(vertical)
            lines.append(f"| {view} | {arm} | {row['pair_id']} | {value(row['direction_fit_px'])} | {h_distinct} | {h_kept} | "
                         f"{v_distinct} | {v_kept} | {value(row['nearest_kept_by_kept_px'])} | {value(row['nearest_distinct_by_distinct_px'])} |")
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path(__file__).resolve().parent / 'runs' / 'filter_replay')
    args = parser.parse_args()
    with open(args.run_dir / 'table.csv', newline='') as stream:
        direction_rows = list(csv.DictReader(stream))
    with open(args.run_dir / 'axis_table.csv', newline='') as stream:
        axis_rows = list(csv.DictReader(stream))
    print('## Direction stage\n')
    print(direction_table(direction_rows))
    print('\n## Axis stage\n')
    print(axis_table(axis_rows))


if __name__ == '__main__':
    main()
