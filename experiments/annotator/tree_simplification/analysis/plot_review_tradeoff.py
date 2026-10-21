"""Plot the share of test rally sections kept against the error among judged kept sections.

Reads the review-threshold evaluation in ``input-preprocessing-evidence.json.gz`` and draws the
baseline-preprocessing panel pair in ``input-review-tradeoff.svg``. The check ignores Matplotlib's
generated element ids and trailing spaces, which do not change the drawing.

Run from the repository root with ``PYTHONPATH=src:.``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.annotator.tree_simplification.analysis.common import (
    RESULTS,
    check_svg,
    read_json,
    save_svg,
)

# Ranking name in the evaluation, line colour, marker and legend text.
RANKINGS = (
    ('fitted', '#0072B2', 'o', 'Fitted reviewer'),
    ('selector', '#D55E00', 's', 'Selector fallback'),
)


def plot(reports: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    for axis, tolerance in zip(axes, ('10', '5'), strict=True):
        for ranking, colour, marker, label in RANKINGS:
            points = []
            for row in reports:
                if row['candidate'] != 'raw_baseline' or row['ranking'] != ranking:
                    continue
                pooled = row['by_tolerance'][tolerance]['pooled']
                points.append((100 * pooled['retained'] / pooled['total'], 100 * pooled['retained_judgement']['error']))
                if ranking == 'fitted':
                    target = row['target_coverage']
                    label_text = 'Off' if target is None else f'{100 * target:g}%'
                    # The 95% label sits below its point so it does not cover the unfiltered one.
                    label_offset = (0, -17) if target == .95 else (0, 10)
                    axis.annotate(label_text, points[-1], xytext=label_offset, textcoords='offset points',
                                  ha='center', fontsize=8, color=colour)
            points.sort()
            coverage, error = zip(*points, strict=True)
            axis.plot(coverage, error, color=colour, marker=marker, markersize=5, linewidth=1.5, label=label,
                      linestyle='-' if ranking == 'fitted' else '--')
        axis.set_title(f'±{tolerance} frames at 30 fps')
        axis.set_xlabel('Test spans retained (%)')
        axis.grid(alpha=.22)
        axis.set_xlim(55, 102)
        axis.set_ylim(25, 54)
        axis.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Error among judged retained spans (%)')
    axes[0].legend(frameon=False, loc='upper left')
    figure.suptitle('Baseline: filtering trades coverage for a modest error reduction', fontsize=12)
    figure.tight_layout(rect=(0, .05, 1, .95))
    figure.text(.5, .02, 'Point labels: validation coverage target for the fitted reviewer; Off = unfiltered',
                ha='center', fontsize=9)
    save_svg(figure, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', type=Path, default=RESULTS, help='tracked results folder')
    parser.add_argument('--output', type=Path, required=True, help='SVG file to write')
    args = parser.parse_args()
    evaluation = read_json(args.results / 'input-preprocessing-evidence.json.gz')['test_review_evaluation']
    plot(evaluation['reports'], args.output)
    check_svg(args.output, args.results / 'input-review-tradeoff.svg')


if __name__ == '__main__':
    main()
