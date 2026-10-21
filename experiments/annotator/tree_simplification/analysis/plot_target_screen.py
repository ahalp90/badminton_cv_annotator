"""Plot fully correct rallies against correct-player F1 for the first grid and the ±5-frame label runs.

Reads ``screening-evidence.json.gz`` and draws ``target-screen.svg``, the figure in appendix A3 of
the report. Dots are the first grid with ±10-frame labels, the dashed lines join one and two
selectors trained with ±5-frame labels on each option list, and the star is the old eight-model
stack refitted with XGBoost. All points are single seed-0 fits on the eight validation videos. The
check ignores Matplotlib's generated element ids and trailing spaces, which do not change the
drawing.

Run from the repository root with ``PYTHONPATH=src:.``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.annotator.tree_simplification.analysis.common import (
    RESULTS,
    check_svg,
    read_json,
    save_svg,
)

# Label in the evidence's labels_and_option_lists section, marker and legend text.
FIVE_FRAME_RUNS = (
    ('original-options-15-leaves', 's', '±5-frame labels, original option list'),
    ('larger-options-15-leaves', '^', '±5-frame labels, larger option list'),
)


def point(totals: dict[str, Any], tolerance: str) -> tuple[float, int]:
    return 100 * totals[tolerance]['attributed_f1'], totals[tolerance]['fully_correct_rallies']


def plot(evidence: dict[str, Any], output: Path) -> None:
    rallies = evidence['validation_rallies']
    first_grid = [run['totals'] for arm in evidence['first_grid_xgboost'].values() for run in arm['selectors'].values()]
    old_stack = evidence['backend_pilot']['old-eight-model-stack-xgboost']['totals']
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for axis, tolerance in zip(axes, ('10', '5'), strict=True):
        f1, correct = zip(*(point(totals, tolerance) for totals in first_grid), strict=True)
        axis.scatter(f1, correct, color='#0072B2', marker='o', s=32, alpha=0.65, label='First grid, ±10-frame labels')
        for label, marker, legend in FIVE_FRAME_RUNS:
            selectors = evidence['labels_and_option_lists'][label]['selectors']
            points = {count: point(selectors[count]['totals'], tolerance) for count in ('1', '2')}
            f1, correct = zip(*points.values(), strict=True)
            axis.plot(f1, correct, color='#D55E00', marker=marker, markersize=7, linestyle='--', linewidth=1,
                      label=legend)
            for count, position in points.items():
                axis.annotate(count, position, xytext=(5, 5), textcoords='offset points', fontsize=9)
        axis.scatter(*point(old_stack, tolerance), color='#111111', marker='*', s=150,
                     label='Old eight-model stack refitted with XGBoost', zorder=4)
        axis.set(title=f'±{tolerance} frames at 30 fps', xlabel='Correct-player F1 (%)',
                 ylabel=f'Fully correct rallies (of {rallies})')
        axis.grid(alpha=0.2)
        axis.margins(x=0.13, y=0.18)
        axis.spines[['top', 'right']].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', ncol=2, frameon=False, fontsize=9)
    figure.suptitle('±5-frame training labels bring back fully correct rallies', fontsize=13)
    figure.text(0.5, 0.90, 'Eight validation videos · one seed · 1 and 2 give the number of rally selectors',
                ha='center', fontsize=9)
    figure.tight_layout(rect=(0, 0.15, 1, 0.90))
    save_svg(figure, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', type=Path, default=RESULTS, help='tracked results folder')
    parser.add_argument('--output', type=Path, required=True, help='SVG file to write')
    args = parser.parse_args()
    plot(read_json(args.results / 'screening-evidence.json.gz'), args.output)
    tracked = args.results / 'target-screen.svg'
    if tracked.exists():
        check_svg(args.output, tracked)


if __name__ == '__main__':
    main()
