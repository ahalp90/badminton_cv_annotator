"""Draw the MonoTrack comparison charts from ``metrics.py`` output.

Each figure carries its own title, axis labels and notes so it reads without the report.

Usage: python make_figures.py --data DIR --out DIR
"""
from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend import Legend
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

MONOTRACK_COLOUR = '#eb6834'
ANNOTATOR_COLOUR = '#2a78d6'
TREES_OFF_COLOUR = '#1baf7a'
TEXT = '#0b0b0b'
MUTED = '#52514e'
GRID = '#e4e3df'
SURFACE = '#fcfcfb'
SEEDS = (0, 100, 200, 300, 400)
MIN_KEPT = 50
POPULATION = '46 ShuttleSet22 videos, 3,327 labelled rallies, 37,184 labelled hits'
SCORING_NOTE = 'A hit counts as found within ±10 frames (1/3 s). Trained models use the mean of five seeds.'
ERROR_TYPES = (
    ('missed', 'Labelled hit missed', '#6250d6'),
    ('wrong_player', 'Hit found, wrong player', '#eda100'),
    ('extra_inside_windows', 'Extra hit inside a scored rally', '#e87ba4'),
    ('extra_outside_windows', 'Extra hit outside every labelled rally', '#6f6e69'),
)


@dataclass(frozen=True)
class Condition:
    key: str
    label: str
    colour: str
    marker: str
    found: bool


CONDITIONS = (
    Condition('monotrack_own_courts', 'MonoTrack, its own courts', MONOTRACK_COLOUR, 'o', False),
    Condition('monotrack_our_courts', 'MonoTrack, our courts', MONOTRACK_COLOUR, 's', False),
    Condition('annotator_our_courts', 'Our annotator, our courts', ANNOTATOR_COLOUR, 'o', False),
    Condition('annotator_monotrack_courts', "Our annotator, MonoTrack's courts", ANNOTATOR_COLOUR, 's', False),
    Condition('heuristic_our_courts', 'Our annotator, trees off, our courts', TREES_OFF_COLOUR, 'o', False),
    Condition('heuristic_monotrack_courts', "Our annotator, trees off, MonoTrack's courts", TREES_OFF_COLOUR, 's',
              False),
    Condition('annotator_discovery', 'Our annotator, finds its own rallies', ANNOTATOR_COLOUR, 'D', True),
    Condition('heuristic_discovery', 'Our annotator, trees off, finds its own rallies', TREES_OFF_COLOUR, 'D', True),
)
ROW_GAP = 0.8


def style_axes(axes: plt.Axes, grid_axis: str = 'x') -> None:
    axes.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        axes.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        axes.spines[side].set_color(MUTED)
    axes.tick_params(colors=MUTED, labelcolor=TEXT)
    axes.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)


def condition_rows() -> np.ndarray:
    """Row positions top to bottom, with a gap before the found-rally block."""
    rows = np.arange(len(CONDITIONS))[::-1].astype(float)
    found = np.array([condition.found for condition in CONDITIONS])
    rows[found] -= ROW_GAP
    return rows


def block_labels(axes: plt.Axes, rows: np.ndarray, x: float) -> None:
    supplied_top = rows[0] + 0.55
    found_top = rows[[condition.found for condition in CONDITIONS]].max() + 0.55
    axes.text(x, supplied_top, 'Given the rally bounds', fontsize=9.5, color=MUTED, style='italic', va='bottom')
    axes.text(x, found_top, 'Finds the rallies itself', fontsize=9.5, color=MUTED, style='italic', va='bottom')


def value(summary: dict, key: str, metric: str, statistic: str = 'mean') -> float:
    return summary[key]['10'][metric][statistic]


DISCOVERY_FACE = '#9cc3ef'
# key, label, colour, marker; the last row finds its own rallies and sits below a divider.
HEADLINE_ROWS = (
    ('annotator_our_courts', 'Our annotator, our courts', ANNOTATOR_COLOUR, 'o'),
    ('annotator_monotrack_courts', "Our annotator, MonoTrack's courts", ANNOTATOR_COLOUR, 'o'),
    ('monotrack_own_courts', 'MonoTrack, its own courts', MONOTRACK_COLOUR, 's'),
    ('monotrack_our_courts', 'MonoTrack, our courts', MONOTRACK_COLOUR, 's'),
    ('annotator_discovery', 'Our annotator,\nfinds its own rallies', DISCOVERY_FACE, 'D'),
)
HEADLINE_METRICS = (
    ('f1', 'Contact F1 (%)\nhit found within ±10 frames', (84, 100)),
    ('player_f1', 'Correct-player F1 (%)\nhit found and hitter right', (84, 100)),
    ('fully_correct', 'Fully correct rallies (%)\nevery hit and hitter right, no extras', (40, 65)),
)


def headline_figure(summary: dict, out: Path) -> None:
    for key, _label, _colour, _marker in HEADLINE_ROWS:
        if any(len(summary[key]['10'][metric]['runs']) != len(SEEDS) for metric, _title, _limits in HEADLINE_METRICS):
            raise ValueError(f'{key}: five training seeds are required for the headline figure')
    figure, axes_row = plt.subplots(1, 3, figsize=(13, 4.9), sharey=True, facecolor=SURFACE)
    rows = np.arange(len(HEADLINE_ROWS))[::-1].astype(float)
    rows[-1] -= 0.5  # gap before the row that finds its own rallies: a harder task
    for axes, (metric, title, (low, high)) in zip(axes_row, HEADLINE_METRICS, strict=True):
        low = min(low, min(value(summary, key, metric, 'min') for key, *_rest in HEADLINE_ROWS) - 2)
        high = max(high, max(value(summary, key, metric, 'max') for key, *_rest in HEADLINE_ROWS) + 2)
        style_axes(axes)
        for row, (key, _label, colour, marker) in zip(rows, HEADLINE_ROWS, strict=True):
            mean = value(summary, key, metric)
            axes.plot([low, mean], [row, row], color=GRID, linewidth=1.5, zorder=1)
            label_at = mean
            if len(summary[key]['10'][metric]['runs']) > 1:
                axes.plot([value(summary, key, metric, 'min'), value(summary, key, metric, 'max')], [row, row],
                          color=TEXT, linewidth=1.6, zorder=2)
                label_at = max(mean, value(summary, key, metric, 'max'))
            axes.scatter(mean, row, s=90, color=colour, marker=marker, zorder=3, linewidth=1.5,
                         edgecolor=ANNOTATOR_COLOUR if colour == DISCOVERY_FACE else SURFACE)
            axes.annotate(f'{mean:.1f}', (label_at, row), xytext=(9, 0), textcoords='offset points', va='center',
                          fontsize=10, color=TEXT)
        axes.set_xlim(low, high + 2.5)
        axes.set_title(title, fontsize=11, color=TEXT, loc='left')
    axes_row[0].set_yticks(rows, [label for _key, label, _colour, _marker in HEADLINE_ROWS], fontsize=10)
    figure.suptitle(f'Our annotator and MonoTrack on {POPULATION}', fontsize=13, color=TEXT, x=0.01, ha='left')
    figure.text(0.01, 0.047, 'Rows 1–4 score contacts within supplied rally boundaries, as required by MonoTrack. '
                'We allow a buffer of up to 1 s before and after each rally.', fontsize=9, color=MUTED)
    figure.text(0.01, 0.01, 'Both trained systems: mean of five seeds; black lines span the lowest to highest. '
                'Settings chosen on validation.', fontsize=9, color=MUTED)
    figure.tight_layout(rect=(0, 0.08, 1, 0.95))
    # One dotted line across all three panels sets the self-finding row apart. It starts under the
    # row label above it. Drawn after layout, because it needs the final label and panel positions.
    figure.draw_without_rendering()
    to_figure = figure.transFigure.inverted()
    _x, divider_y = to_figure.transform(axes_row[0].transData.transform((0, rows[-1] + 0.75)))
    label_above = axes_row[0].get_yticklabels()[len(HEADLINE_ROWS) - 2]
    left = to_figure.transform(label_above.get_window_extent().p0)[0]
    right = axes_row[-1].get_position().x1
    figure.add_artist(Line2D([left, right], [divider_y, divider_y], transform=figure.transFigure,
                             color=MUTED, linewidth=0.8, linestyle=':'))
    figure.savefig(out / 'fig1_headline.png', dpi=150, facecolor=SURFACE)
    plt.close(figure)


def error_figure(summary: dict, out: Path) -> None:
    figure, axes = plt.subplots(figsize=(13, 5.6), facecolor=SURFACE)
    style_axes(axes)
    rows = condition_rows()
    largest = 0.0
    for row, condition in zip(rows, CONDITIONS, strict=True):
        start = 0.0
        for key, _label, colour in ERROR_TYPES:
            width = value(summary, condition.key, key)
            if width == 0:
                continue
            axes.barh(row, width, left=start, height=0.62, color=colour, edgecolor=SURFACE, linewidth=1.5)
            if width >= 700:
                axes.text(start + width / 2, row, f'{width:,.0f}', ha='center', va='center', fontsize=8.5,
                          color='white' if key in ('missed', 'extra_outside_windows') else TEXT)
            start += width
        rallies = value(summary, condition.key, 'rallies_not_fully_correct')
        axes.text(start + 150, row, f'{start:,.0f} hit errors\n{rallies:,.0f} rallies not fully correct', va='center', fontsize=9,
                  color=TEXT)
        largest = max(largest, start)
    axes.set_yticks(rows, [condition.label for condition in CONDITIONS], fontsize=10)
    axes.set_xlim(0, largest * 1.42)
    axes.set_ylim(rows.min() - 0.6, rows.max() + 1.0)
    block_labels(axes, rows, 0)
    axes.set_xlabel('Hit-level errors: each one is a hit a person would add, delete or reassign', color=TEXT)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colour) for _key, _label, colour in ERROR_TYPES]
    axes.legend(handles, [label for _key, label, _colour in ERROR_TYPES], loc='upper center', frameon=False,
                fontsize=9.5, ncol=3, bbox_to_anchor=(0.4, -0.12))
    figure.suptitle(f'Hit errors by type, ±10 frames ({POPULATION})', fontsize=13, color=TEXT, x=0.01, ha='left')
    figure.text(0.01, 0.04, 'With rally bounds given, hits outside the bounds are not scored. Trained-model values are '
                'means of five training seeds.', fontsize=9, color=MUTED)
    figure.text(0.01, 0.012, 'Predictions inside the 524 ShuttleSet22 rallies that the test set leaves out because of '
                'flawed labels are ignored.', fontsize=9, color=MUTED)
    figure.tight_layout(rect=(0, 0.07, 1, 0.95))
    figure.savefig(out / 'fig3_hit_errors.png', dpi=150, facecolor=SURFACE)
    plt.close(figure)


def per_video_figure(data: Path, out: Path) -> None:
    ours = pd.read_csv(data / 'per_video_annotator_our_courts.csv', index_col='video')
    theirs = pd.read_csv(data / 'per_video_monotrack_own_courts.csv', index_col='video')
    metrics = (('f1', 'Contact F1 (%)'), ('player_f1', 'Correct-player F1 (%)'),
               ('fully_correct', 'Fully correct rallies (%)'))
    figure, axes_row = plt.subplots(1, 3, figsize=(14, 4.9), facecolor=SURFACE)
    for axes, (metric, title) in zip(axes_row, metrics, strict=True):
        style_axes(axes, 'both')
        x, y = theirs[metric], ours[metric]
        low = float(min(x.min(), y.min())) - 2
        high = float(max(x.max(), y.max())) + 2
        axes.plot([low, high], [low, high], color=MUTED, linewidth=1, linestyle='--', zorder=1)
        axes.scatter(x, y, s=34, color=ANNOTATOR_COLOUR, edgecolor=SURFACE, linewidth=0.8, zorder=3)
        above, below = int((y > x).sum()), int((y < x).sum())
        # The empty lower-right corner keeps the counts clear of the dots.
        axes.text(0.97, 0.03, f'Above the line (ours higher): {above} videos\nBelow (MonoTrack higher): {below}',
                  transform=axes.transAxes, ha='right', va='bottom', fontsize=9.5, color=TEXT)
        axes.set_xlim(low, high)
        axes.set_ylim(low, high)
        axes.set_aspect('equal')
        axes.set_title(title, fontsize=11, color=TEXT, loc='left')
        axes.set_xlabel('MonoTrack, its own courts', color=TEXT)
        axes.set_ylabel('Our annotator, our courts', color=TEXT)
    figure.suptitle('Each dot is one of the 46 test videos; rally bounds given to both systems',
                    fontsize=13, color=TEXT, x=0.01, ha='left')
    figure.text(0.01, 0.01, 'Both trained systems use per-video means of five seeds. ±10 frames.',
                fontsize=9, color=MUTED)
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    figure.savefig(out / 'fig4_per_video.png', dpi=150, facecolor=SURFACE)
    plt.close(figure)


LADDER_RUNGS = (
    ('boundaries_right', 'Boundaries right', 0.35),
    ('hits_right', 'Hits right (hitters not checked)', 0.65),
    ('fully_correct', 'Fully correct', 1.0),
)
# Section title, conditions and the steps that apply; given bounds make the boundary step automatic.
LADDER_SECTIONS = (
    ('Rally bounds provided', ('monotrack_own_courts', 'annotator_our_courts', 'heuristic_our_courts'),
     LADDER_RUNGS[1:]),
    ('Rally bounds discovered', ('annotator_discovery', 'heuristic_discovery'), LADDER_RUNGS),
)
LADDER_LABELS = {
    'monotrack_own_courts': 'MonoTrack, its own courts',
    'annotator_our_courts': 'Our annotator, our courts',
    'heuristic_our_courts': 'Our annotator, trees off,\nour courts',
    'annotator_discovery': 'Our annotator',
    'heuristic_discovery': 'Our annotator, trees off',
}


# The legends end at the axis edge, beside each section's short lower bars, so the bars keep the full width.
LADDER_AXIS_END = 100


def ladder_figure(summary: dict, out: Path, boundary_policy: dict[str, float | None]) -> None:
    """Rally-level outcomes: boundaries, then hits, then hitters."""
    by_key = {condition.key: condition for condition in CONDITIONS}
    figure, axes = plt.subplots(figsize=(13, 6.6), facecolor=SURFACE)
    style_axes(axes)
    ticks, labels = [], []
    row, divider = 0.0, None
    for section_index, (title, keys, rungs) in enumerate(LADDER_SECTIONS):
        if section_index:
            divider = row + 0.1
            row -= 0.9
        axes.text(0.5, row, title, va='center', fontsize=11, color=TEXT, fontweight='bold')
        row -= 1.1
        for key in keys:
            colour = by_key[key].colour
            first = row
            for metric, _label, shade in rungs:
                mean = value(summary, key, metric)
                axes.barh(row, mean, height=0.8, color=colour, alpha=shade, edgecolor=SURFACE)
                axes.text(mean + 0.8, row, f'{mean:.1f}%', va='center', fontsize=9.5, color=TEXT)
                row -= 1
            ticks.append((first + row + 1) / 2)
            labels.append(LADDER_LABELS[key])
            row -= 0.6
        # Each legend swatch shows that step's shade in every colour used in this section.
        handles = [tuple(plt.Rectangle((0, 0), 1, 1, color=by_key[key].colour, alpha=shade) for key in keys)
                   for _metric, _label, shade in rungs]
        legend = Legend(axes, handles, [label for _metric, label, _shade in rungs], loc='lower right',
                             bbox_to_anchor=(LADDER_AXIS_END, row + 0.7), bbox_transform=axes.transData, frameon=False,
                             borderaxespad=0, borderpad=0,
                             fontsize=9.5, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
                             handlelength=1.2 * len(keys))
        axes.add_artist(legend)
    axes.axhline(divider, color=MUTED, linestyle='--', linewidth=1)
    axes.set_yticks(ticks, labels, fontsize=10)
    axes.set_ylim(row + 0.3, 0.8)
    axes.set_xlim(0, LADDER_AXIS_END)
    axes.set_xlabel('Share of the 3,327 labelled rallies (%)', color=TEXT)
    figure.suptitle('How many labelled rallies come out right, step by step (±10 frames)', fontsize=13, color=TEXT,
                    x=0.01, ha='left')
    figure.text(0.01, 0.045, 'Hits right: every hit found, no extras. Fully correct: every hitter right too. '
                'Trained-model bars show five-seed means.', fontsize=9, color=MUTED)
    figure.text(0.01, 0.015, boundary_note(boundary_policy), fontsize=9, color=MUTED)
    figure.tight_layout(rect=(0, 0.07, 1, 0.95))
    figure.savefig(out / 'fig5_rally_ladder.png', dpi=150, facecolor=SURFACE)
    plt.close(figure)


CUTOFF_LABELS = {
    'balanced': 'best balance',
    'keep_half': 'keeps half',
    'boundaries_99': '99% validation target',
}
# Label offsets in points, placed to keep each label clear of both curves.
CUTOFF_LABEL_OFFSETS = {'balanced': (6, -23), 'keep_half': (8, 10), 'boundaries_99': (20, 8)}
# Alignment of each cut-off's number above the left panel, so close cut-offs' numbers sit apart.
CUTOFF_TOP_LABEL_ALIGN = {'balanced': 'right', 'keep_half': 'left', 'boundaries_99': 'center'}


def seed_band(curves: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cut-offs and the seed mean, lowest and highest, blanked where too few rallies are kept."""
    table = curves.pivot(index='threshold', columns='seed', values=column)
    kept = curves.pivot(index='threshold', columns='seed', values='kept')
    # Above a high cut-off only a handful of rallies remain, so shares there are noise.
    table = table.where(kept.min(axis=1) >= MIN_KEPT, axis=0)
    return table.index.to_numpy(), table.mean(axis=1).to_numpy(), table.min(axis=1).to_numpy(), table.max(axis=1).to_numpy()


def cutoff_text(value: float) -> str:
    """Keep labels short and mark any displayed rounding explicitly."""
    label = f'{value:.3f}'
    return label if float(label) == value else f'≈{label}'


def boundary_note(policy: dict[str, float | None]) -> str:
    before, after = policy['before_seconds'], policy['after_seconds']
    if before is None and after is None:
        return ('One complete labelled rally per found span, with no maximum context limit. '
                'Edges may fall 10 frames inside.')
    return (f'Bounds: up to {before:g} s before the first labelled hit and {after:g} s after the last; '
            'edges may fall 10 frames inside. One labelled rally per found span.')


def cutoff_row(curves: pd.DataFrame, threshold: float) -> pd.Series:
    """Read the actual selected score, keeping neighbouring attainable scores separate."""
    selected = curves[curves['threshold'] == threshold]
    if selected.empty:
        raise ValueError(f'selected validation cut-off {threshold!r} is absent from the curve')
    return selected.mean(numeric_only=True)


def confidence_figure(data: Path, out: Path) -> None:
    # Round-trip parsing preserves distinct nearby confidence scores from the full scan.
    test = pd.read_csv(data / 'test_confidence_curve.csv.gz', float_precision='round_trip')
    validation = pd.read_csv(data / 'validation_confidence_curve.csv.gz', float_precision='round_trip')
    with gzip.open(data / 'confidence_cutoffs.json.gz', 'rt') as stream:
        selection = json.load(stream)
    cutoffs = selection['cutoffs']
    available = {name: value for name, value in cutoffs.items() if value is not None}
    figure, (left, right) = plt.subplots(1, 2, figsize=(14, 6.2), facecolor=SURFACE,
                                         gridspec_kw={'width_ratios': (1.25, 1)})
    style_axes(left, 'both')
    lines = (
        ('boundaries_right_share', 'Kept rallies with the right boundaries (% of kept)', TEXT, '-'),
        ('fully_correct_share', 'Kept rallies fully correct (% of kept)', ANNOTATOR_COLOUR, '-'),
        ('coverage', 'Fully correct rallies handed over (% of all 3,327 labelled rallies)', ANNOTATOR_COLOUR, ':'),
    )
    for column, label, colour, style in lines:
        thresholds, mean, low, high = seed_band(test, column)
        left.fill_between(thresholds, low, high, color=colour, alpha=0.15, linewidth=0)
        left.plot(thresholds, mean, color=colour, linestyle=style, linewidth=2, label=label)
    left.axhline(99, color=MUTED, linewidth=0.8, linestyle=(0, (1, 2)))
    left.text(0.01, 99.6, '99%', fontsize=8.5, color=MUTED)
    for name, value in available.items():
        left.axvline(value, color=MUTED, linewidth=1, linestyle='-', alpha=0.6)
        align = 'right' if value > 0.9 else 'left' if value < 0.1 else CUTOFF_TOP_LABEL_ALIGN[name]
        offset = 3 if align == 'left' else -3 if align == 'right' else 0
        left.annotate(cutoff_text(value), (value, 100), xytext=(offset, 5), textcoords='offset points',
                      ha=align, va='bottom', fontsize=8.5, color=TEXT)
    left.set_xlim(0, 1)
    left.set_ylim(0, 100)
    left.set_xlabel('Keep found rallies with confidence at or above this threshold', color=TEXT)
    left.set_ylabel('%', color=TEXT)
    left.set_title('Test videos: what each threshold keeps', fontsize=11, color=TEXT, loc='left', pad=16)
    left.legend(loc='lower left', frameon=False, fontsize=9)

    style_axes(right, 'both')
    coverage_max = max(test['coverage'].max(), validation['coverage'].max())
    coverage_limit = min(100, max(62, coverage_max + 3))
    for curves, label, colour, style in ((test, 'Test (46 videos)', ANNOTATOR_COLOUR, '-'),
                                         (validation, 'Validation (8 other videos)', MUTED, '--')):
        _thresholds, coverage, _low, _high = seed_band(curves, 'coverage')
        _thresholds, share, _low, _high = seed_band(curves, 'fully_correct_share')
        right.plot(coverage, share, color=colour, linestyle=style, linewidth=2, label=label)
        for name, value in available.items():
            row = cutoff_row(curves, value)
            # Validation points are hollow; cut-off labels belong to filled test points.
            if pd.isna(row['fully_correct_share']):
                continue
            right.scatter(row['coverage'], row['fully_correct_share'], s=36, zorder=4, linewidth=1.5,
                          color=colour if curves is test else SURFACE, edgecolor=colour)
            if curves is test:
                offset_x, offset_y = CUTOFF_LABEL_OFFSETS[name]
                align = 'right' if row['coverage'] > 0.7 * coverage_limit else 'left'
                offset_x = -abs(offset_x) if align == 'right' else offset_x
                right.annotate(f'{cutoff_text(value)} {CUTOFF_LABELS[name]}', (row['coverage'], row['fully_correct_share']),
                               xytext=(offset_x, offset_y), ha=align, textcoords='offset points', fontsize=9, color=TEXT,
                               arrowprops={'arrowstyle': '-', 'color': MUTED, 'lw': 0.8}
                               if name == 'boundaries_99' else None)
    precision_min = min(test['fully_correct_share'].min(), validation['fully_correct_share'].min())
    right.set_xlim(0, coverage_limit)
    right.set_ylim(max(0, min(30, precision_min - 3)), 100)
    right.set_xlabel('Fully correct rallies handed over (% of all labelled rallies)', color=TEXT)
    right.set_ylabel('Share of kept rallies fully correct (%)', color=TEXT)
    right.set_title('Trade-off: each point is one validation threshold', fontsize=11, color=TEXT, loc='left', pad=16)
    right.legend(loc='upper right', frameon=False, fontsize=9)
    figure.suptitle('Our annotator finding its own rallies: keeping only the confident ones', fontsize=13, color=TEXT,
                    x=0.01, ha='left')
    boundary_cutoff = cutoffs['boundaries_99']
    if boundary_cutoff is None:
        target_note = '99% mean boundary accuracy is unattainable on validation with rallies kept in all five seeds.'
    else:
        boundary_validation = selection['validation']['boundaries_99']
        target_note = (f'{cutoff_text(boundary_cutoff)} is the lowest cut-off reaching 99% mean boundary accuracy; '
                       f'validation retains {boundary_validation["kept_min"]:.0f}–'
                       f'{boundary_validation["kept_max"]:.0f} rallies per seed.')
    figure.text(0.01, 0.115, f'Chosen on validation: {cutoff_text(cutoffs["balanced"])} maximises mean F1; '
                f'{cutoff_text(cutoffs["keep_half"])} keeps closest to half of the pooled found rallies. '
                'Every distinct validation score is searched.', fontsize=9, color=MUTED)
    figure.text(0.01, 0.09, target_note, fontsize=9, color=MUTED)
    figure.text(0.01, 0.065, f'Mean of five seeds; shading spans lowest to highest. Lines require at least '
                f'{MIN_KEPT} kept rallies in every seed; selected cut-offs remain marked.', fontsize=9, color=MUTED)
    example = cutoff_row(test, cutoffs['keep_half'])
    share = example['fully_correct_share']
    precision_text = f'{share:.1f}%' if pd.notna(share) else 'undefined (no rallies kept)'
    figure.text(0.01, 0.04, f'Test at {cutoff_text(cutoffs["keep_half"])}: {example["kept"]:,.0f} found rallies kept, '
                f'{precision_text} fully correct ({example["fully_correct_kept"]:,.0f}); '
                f'{example["coverage"]:.1f}% of all 3,327 labelled rallies handed over.', fontsize=9, color=MUTED)
    figure.text(0.01, 0.015, boundary_note(selection['boundary_policy']), fontsize=9, color=MUTED)
    figure.tight_layout(rect=(0, 0.145, 1, 0.95))
    figure.savefig(out / 'fig6_confidence.png', dpi=150, facecolor=SURFACE)
    plt.close(figure)


# Band colour and the darker mean-line colour for each contact-model training set.
THIRTY_TWO_VIDEO_COLOURS = (ANNOTATOR_COLOUR, '#163f75')
FORTY_VIDEO_COLOURS = ('#7b52c9', '#3f2380')
# heading, 32-video key, 40-video key; the last group finds its own rallies and sits below a divider.
STABILITY_GROUPS = (
    ('Given rally bounds,\nour courts', 'annotator_our_courts', 'annotator40_our_courts'),
    ("Given rally bounds,\nMonoTrack's courts", 'annotator_monotrack_courts', 'annotator40_monotrack_courts'),
    ('Finds its own rallies', 'annotator_discovery', 'annotator40_discovery'),
)
# Both F1 panels share one range so their bands compare directly.
STABILITY_METRICS = (
    ('f1', 'Contact F1 (%)', (89.4, 97.2)),
    ('player_f1', 'Correct-player F1 (%)', (89.4, 97.2)),
    ('fully_correct', 'Fully correct rallies (%)', (50.3, 62.7)),
)


def stability_figure(summary: dict, out: Path) -> None:
    """Fig 1's layout, one row per contact-model training set, with each row's seed range as a band."""
    figure, axes_row = plt.subplots(1, 3, figsize=(13, 6.6), sharey=True, facecolor=SURFACE)
    rows, headings = [], []
    top = 0.0
    for group_index, (heading, key32, key40) in enumerate(STABILITY_GROUPS):
        if group_index == len(STABILITY_GROUPS) - 1:
            top -= 0.5  # extra gap before the group that finds its own rallies: a harder task
        headings.append((top, heading))
        rows += [(top - 0.6, key32, '32 videos (headline)', THIRTY_TWO_VIDEO_COLOURS),
                 (top - 1.5, key40, '40 videos', FORTY_VIDEO_COLOURS)]
        top -= 2.6
    for axes, (metric, title, (low, high)) in zip(axes_row, STABILITY_METRICS, strict=True):
        compared_metrics = ('f1', 'player_f1') if metric in ('f1', 'player_f1') else (metric,)
        lowest = min(value(summary, key, compared, 'min')
                     for _row, key, _label, _colours in rows for compared in compared_metrics)
        highest = max(value(summary, key, compared, 'max')
                      for _row, key, _label, _colours in rows for compared in compared_metrics)
        low, high = min(low, lowest - 1), max(high, highest + 1)
        style_axes(axes)
        for row, key, _label, (band_colour, mean_colour) in rows:
            lowest, mean, highest = (value(summary, key, metric, statistic) for statistic in ('min', 'mean', 'max'))
            axes.plot([low, lowest], [row, row], color=GRID, linewidth=1.5, zorder=1)
            # The edge keeps sub-point seed ranges visible on the shared F1 scale.
            axes.barh(row, highest - lowest, left=lowest, height=0.4, color=band_colour, alpha=0.4,
                      edgecolor=band_colour, linewidth=1, zorder=2)
            axes.plot([mean, mean], [row - 0.2, row + 0.2], color=mean_colour, linewidth=1.5, zorder=3)
            # Labels sit under the band so the x range only has to hold the data. Near a panel edge,
            # the label starts at the band's inner end so it stays inside the panel.
            position = (mean - low) / (high - low)
            anchor, align = (lowest, 'left') if position < 0.2 else (highest, 'right') if position > 0.8 else (
                mean, 'center')
            axes.annotate(f'{mean:.1f} ({lowest:.1f}–{highest:.1f})', (anchor, row), xytext=(0, -12),
                          textcoords='offset points', ha=align, va='top', fontsize=8.5, color=TEXT)
        axes.set_xlim(low, high)
        axes.xaxis.set_major_locator(plt.MultipleLocator(2))
        axes.set_title(title, fontsize=11, color=TEXT, loc='left')
    first = axes_row[0]
    first.set_yticks([row for row, *_rest in rows], [label for _row, _key, label, _colours in rows], fontsize=10)
    first.set_ylim(rows[-1][0] - 0.6, headings[0][0] + 0.2)
    for row, heading in headings:
        first.text(-0.04, row + 0.05, heading, transform=first.get_yaxis_transform(), ha='right', va='center',
                   fontsize=10, fontweight='bold', color=TEXT)
    figure.suptitle("Our annotator's performance on ShuttleSet22, contact model trained on 32 videos (train only) vs "
                    '40 videos (train + validation)', fontsize=13, color=TEXT, x=0.01, y=0.985, ha='left', va='top')
    # Why only the contact model changes: the later trees are cross-fitted on the 32 training videos.
    figure.text(0.01, 0.938, 'The rally selector and review ranker keep their 32-video fits in both, so only the contact '
                'model differs. They learn from contact-model predictions on training videos held out from that '
                'contact model.', fontsize=9, color=MUTED, va='top')
    figure.text(0.01, 0.06, 'Shaded band: lowest to highest of five training seeds. Dark line and first number: '
                'the mean. ±10 frames.', fontsize=9, color=MUTED)
    reference = [value(summary, 'monotrack_own_courts', metric) for metric, _title, _limits in STABILITY_METRICS]
    figure.text(0.01, 0.025, 'MonoTrack, given rally bounds on its own courts: '
                f'{reference[0]:.1f} contact F1, {reference[1]:.1f} correct-player F1, '
                f'{reference[2]:.1f}% fully correct rallies. Axes do not start at zero.', fontsize=9, color=MUTED)
    figure.tight_layout(rect=(0, 0.09, 1, 0.91))
    # One unbroken divider across the figure, placed after layout so it lands on the same gap in every panel.
    _x, divider_y = figure.transFigure.inverted().transform(first.transData.transform((0, headings[-1][0] + 0.5)))
    figure.add_artist(plt.Line2D([0.01, 0.99], [divider_y, divider_y], transform=figure.transFigure, color=MUTED,
                                 linewidth=0.8, linestyle=':'))
    figure.savefig(out / 'fig10_tree_stability.png', dpi=150, facecolor=SURFACE)
    plt.close(figure)


PER_SCENE_COLOUR = '#9cc3ef'
COURT_SOURCES = (
    ('ours_rally', 'Our court, per rally', ANNOTATOR_COLOUR),
    ('monotrack', "MonoTrack's court, per rally", MONOTRACK_COLOUR),
    ('ours_discovery', 'Our court, per scene, without rally bounds', PER_SCENE_COLOUR),
)
MAX_COURT_ERROR_SHOWN = 15


def court_error_figure(courts: pd.DataFrame, out: Path) -> None:
    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5.2), facecolor=SURFACE,
                                         gridspec_kw={'width_ratios': (1.5, 1)})
    style_axes(left, grid_axis='both')
    for key, label, colour in COURT_SOURCES:
        errors = courts[f'{key}_error_px']
        found = np.sort(errors.dropna().to_numpy())
        # Rallies without a court count against the curve, so it ends below 100%.
        share = 100 * np.arange(1, len(found) + 1) / len(courts)
        left.step(found, share, where='post', color=colour, linewidth=2.2,
                  label=f'{label}\nmedian {errors.median():.1f} px; no court on {errors.isna().sum()} rallies')
    left.set_xlim(0, MAX_COURT_ERROR_SHOWN)
    left.set_xticks(np.arange(0, MAX_COURT_ERROR_SHOWN + 1))
    left.set_ylim(0, 101)
    left.set_xlabel('Mean distance of the four court corners from the reference corners (px, 1920×1080 frame)',
                    color=TEXT)
    left.set_ylabel('Rallies with a court this close or closer (%)', color=TEXT)
    left.legend(loc='lower right', frameon=False, fontsize=9)
    left.set_title('Per rally: share of the 3,327 rallies within each error', fontsize=11, color=TEXT, loc='left')

    style_axes(right)
    medians = courts.groupby('video')[[f'{key}_error_px' for key, _label, _colour in COURT_SOURCES]].median()
    generator = np.random.default_rng(0)
    for row, (key, _label, colour) in enumerate(COURT_SOURCES):
        values = medians[f'{key}_error_px'].to_numpy()
        jitter = generator.uniform(-0.18, 0.18, size=len(values))
        right.scatter(values, row + jitter, s=26, color=colour, edgecolor=ANNOTATOR_COLOUR if key == 'ours_discovery'
                      else SURFACE, linewidth=0.6)
        right.plot([np.median(values)] * 2, [row - 0.3, row + 0.3], color=TEXT, linewidth=2)
        right.text(np.median(values), row + 0.42, f'median {np.median(values):.1f} px', ha='center', fontsize=9,
                   color=TEXT)
    ours_lower = (medians['ours_rally_error_px'] < medians['monotrack_error_px']).sum()
    right.set_yticks(range(len(COURT_SOURCES)), ['Our court,\nper rally', "MonoTrack's\ncourt", 'Our court,\nper scene'])
    right.set_ylim(len(COURT_SOURCES) - 0.25, -0.5)
    right.set_xlim(0, 12)
    right.set_xlabel('Median error in each video (px); one dot per video', color=TEXT)
    right.set_title(f'Per video: our per-rally court is closer in {ours_lower} of {len(medians)} videos',
                    fontsize=11, color=TEXT, loc='left')
    figure.suptitle("How close each court is to ShuttleSet22's reference corners", fontsize=13, color=TEXT,
                    x=0.01, ha='left')
    figure.text(0.01, 0.04, 'The reference is one set of corners per video, drawn for the main camera view.',
                fontsize=9, color=MUTED)
    figure.text(0.01, 0.01, 'In video 17 the broadcast zoomed in after rally 46, so all three courts read about 163 px '
                'there; those 27 rallies lie beyond the right edge of the left panel.', fontsize=9, color=MUTED)
    figure.tight_layout(rect=(0, 0.07, 1, 0.95))
    figure.savefig(out / 'fig2_court_error.png', dpi=150, facecolor=SURFACE)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--confidence-data', type=Path, help='Separate finite-boundary confidence survey')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.data / 'contact_summary.json').read_text())['summary']
    with gzip.open(args.data / 'evaluation-policy.json.gz', 'rt') as stream:
        boundary_policy = json.load(stream)['boundary_policy']
    headline_figure(summary, args.out)
    error_figure(summary, args.out)
    per_video_figure(args.data, args.out)
    ladder_figure(summary, args.out, boundary_policy)
    confidence_figure(args.confidence_data or args.data, args.out)
    stability_figure(summary, args.out)
    court_error_figure(pd.read_csv(args.data / 'rally_courts.csv.gz'), args.out)


if __name__ == '__main__':
    main()
