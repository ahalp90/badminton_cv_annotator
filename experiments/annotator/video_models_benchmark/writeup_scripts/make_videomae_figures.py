"""Draw the VideoMAE-B pilot figures and print the report tables.

Reads the compact run records and stride-three scores that ``rescore_videomae.py``
copies into ``results/``. Each run is scored at its earliest minimum-validation-CE
checkpoint. A run still inside its continuation is marked as in progress, and the
learning-rate ranking stays provisional until the queue writes ``selection.json.gz``.
Figures reuse the X3D and MonoTrack report styles.

Usage, from the repository root:
    python experiments/annotator/video_models_benchmark/writeup_scripts/make_videomae_figures.py

Once ``results/videomae_test/`` holds both selected test seeds, the same command
also draws the held-out test comparison and the test stride-three figure.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.annotator.monotrack_benchmark.writeup_scripts.make_figures import (
    ANNOTATOR_COLOUR,
    GRID,
    MONOTRACK_COLOUR,
    MUTED,
    SURFACE,
    TEXT,
    style_axes,
)
from experiments.annotator.video_models_benchmark.schedule import (
    PATIENCE,
    checks_without_improvement,
)
from experiments.annotator.video_models_benchmark.writeup_scripts.make_x3d_figures import (
    METRICS,
    MONOTRACK_METRICS,
    ORIGINAL_UPDATES,
    SEED_MARKERS,
    TEST_CONTACTS,
    TEST_RALLIES,
    TOLERANCE,
    X3D_COLOUR,
    DotRow,
    dot_plot,
    rank_configurations,
    read_json,
)
from experiments.annotator.video_models_benchmark.writeup_scripts.make_x3d_figures import (
    load_runs as load_x3d_runs,
)

BENCHMARK = Path(__file__).resolve().parents[1]
RESULTS = BENCHMARK / 'results'
FIGURES = BENCHMARK / 'writeup_figures'
MONOTRACK_SUMMARY = (BENCHMARK.parent / 'monotrack_benchmark/benchmark_results/discovery_context/primary'
                     / 'contact_summary.json.gz')
VIDEOMAE_COLOUR = '#138a8a'
SEED_LINES = {0: '-', 1: (0, (5, 2))}
LOSS_BIN_UPDATES = 1_000
# Every scored validation centre, at the natural rate: about 10% are within a frame of a contact.
VALIDATION_CONTACT_PERCENT = 10
# Panel rows for the stride figure: (metric, hitter rule, title). Contact scores are
# the same under both hitter rules, because alternation runs after suppression.
STRIDE_PANELS = (
    (('contact_recall', 'alternating', 'Contact recall (%)'),
     ('contact_precision', 'alternating', 'Contact precision (%)')),
    (('correct_player_f1', 'raw', "Correct-player F1 (%)\nraw: each hit's own most likely hitter"),
     ('correct_player_f1', 'alternating', 'Correct-player F1 (%)\nalternating hitters, as reported')),
    (('fully_correct_percent', 'raw', 'Fully correct rallies (%)\nraw hitters'),
     ('fully_correct_percent', 'alternating', 'Fully correct rallies (%)\nalternating hitters, as reported')),
)
SAMPLING_LABELS = {(1, 0): 'Dense', (3, 0): 'Stride 3, offset 0', (3, 1): 'Stride 3, offset 1',
                   (3, 2): 'Stride 3, offset 2'}
VALIDATION_STRIDE_NOTES = (
    ('Each run is scored at its lowest-validation-loss checkpoint; a hit counts within ±10 frames at 30 fps. '
     'Axes do not start at zero.'),
    ('Stride 3 with offset k keeps only centres whose video frame number leaves remainder k when divided by 3. '
     'Suppression and matching are unchanged.'),
    ("Offsets are three samplings of one model, not extra seeds. Raw hitters keep each hit's own guess; "
     'alternating hitters are what the benchmark reports.'),
)
TEST_COMPARISON_NOTES = (
    'All four systems are given the rally bounds; a hit counts within ±10 frames at 30 fps.',
    ('Each selected checkpoint is tested once. Dots: seed mean; black lines span the lowest to highest seed. '
     'Axes do not start at zero.'),
)
TEST_STRIDE_NOTES = (
    ('Both seeds of the validation-selected learning rate, each tested once at its lowest-validation-loss '
     'checkpoint; a hit counts within ±10 frames at 30 fps.'),
    ('Test scores did not choose any setting. Axes do not start at zero.'),
)
# Fixed SVG dates keep regenerated figures byte-stable.
os.environ.setdefault('SOURCE_DATE_EPOCH', '0')
plt.rcParams['svg.hashsalt'] = 'videomae'


class Window(NamedTuple):
    first: int  # update count before the window
    last: int  # update count at the reading
    loss: float  # mean sampled-batch CE over the window's updates
    rate: float  # head learning rate at the reading


@dataclass(frozen=True)
class PilotRun:
    name: str
    learning_rate: float
    seed: int
    max_steps: int
    history: list[dict]
    progress: list[dict]

    @property
    def selected(self) -> dict:
        # History is in update order and min keeps the first of equal values, matching fit's strict saves.
        return min(self.history, key=lambda check: check['validation']['loss'])

    @property
    def finished(self) -> bool:
        """Continuation stops after ``PATIENCE`` checks in a row without a lower validation CE."""
        return self.history[-1]['steps'] >= self.max_steps and checks_without_improvement(self.history) >= PATIENCE

    @property
    def label(self) -> str:
        return f'LR {rate_label(self.learning_rate)}, seed {self.seed}'

    def score(self, metric: str) -> float:
        return self.selected['events'][TOLERANCE][metric]


def rate_label(rate: float) -> str:
    mantissa, exponent = f'{rate:e}'.split('e')
    return f'{float(mantissa):g}e{int(exponent)}'


def thousands(step: float, _position: object = None) -> str:
    return f'{step / 1000:g}k'


def save(figure: plt.Figure, out: Path, stem: str) -> None:
    for suffix in ('.png', '.svg'):
        figure.savefig(out / f'{stem}{suffix}', dpi=150, facecolor=SURFACE)
    plt.close(figure)


def load_runs(directory: Path) -> list[PilotRun]:
    runs = []
    for path in sorted(path for path in directory.iterdir() if path.is_dir()):
        settings = read_json(path / 'settings.json.gz')
        runs.append(PilotRun(path.name, settings['learning_rate'], settings['seed'], settings['max_steps'],
                             read_json(path / 'history.json.gz'), read_json(path / 'training_progress.json.gz')))
    return sorted(runs, key=lambda run: (run.learning_rate, run.seed))


def mean_score(members: list[PilotRun], metric: str) -> float:
    return float(np.mean([run.score(metric) for run in members]))


def rank_learning_rates(runs: list[PilotRun]) -> list[list[PilotRun]]:
    """Order learning rates as the queue does: mean fully correct rallies, mean correct-player F1, lower LR."""
    groups: dict[float, list[PilotRun]] = {}
    for run in runs:
        groups.setdefault(run.learning_rate, []).append(run)
    for members in groups.values():
        if [run.seed for run in members] != [0, 1]:
            raise ValueError(f'{members[0].name}: expected seeds 0 and 1')
    return sorted(groups.values(), key=lambda members: (-mean_score(members, 'fully_correct_percent'),
                                                        -mean_score(members, 'correct_player_f1'),
                                                        members[0].learning_rate))


def check_selection(ranking: list[list[PilotRun]], selection: dict) -> None:
    """The queue's saved selection must agree with these finished histories."""
    if not all(run.finished for members in ranking for run in members):
        raise ValueError('selection.json.gz exists, but a copied history is unfinished; copy the runs again')
    saved = [(row['learning_rate'], [run['best_step'] for run in row['runs']]) for row in selection['ranking']]
    recomputed = [(members[0].learning_rate, [run.selected['steps'] for run in members]) for members in ranking]
    if saved != recomputed:
        raise ValueError(f'saved selection {saved} differs from the copied histories {recomputed}')


def validation_population(runs: list[PilotRun]) -> tuple[int, int]:
    counts = {(run.score('rallies'), run.score('labelled')) for run in runs}
    if len(counts) != 1:
        raise ValueError(f'runs were scored on different validation populations: {counts}')
    return counts.pop()


def training_windows(progress: list[dict]) -> list[Window]:
    """Mean sampled-batch loss between consecutive progress readings, latest attempt only.

    Each reading stores the mean since its validation interval began. Differencing
    the cumulative losses within one session and interval recovers each window's own
    mean. A resumed session replays updates from its first interval start, so it
    supersedes earlier readings beyond that point.
    """
    first_start = {}
    for reading in progress:
        first_start.setdefault(reading['session_started'], reading['interval_start_step'])
    previous: dict[tuple[float, int], tuple[int, float]] = {}
    windows = []
    for reading in progress:
        key = (reading['session_started'], reading['interval_start_step'])
        first, cumulative_before = previous.get(key, (reading['interval_start_step'], 0.0))
        cumulative = reading['train_loss'] * (reading['step'] - reading['interval_start_step'])
        previous[key] = (reading['step'], cumulative)
        later_starts = [start for session, start in first_start.items() if session > reading['session_started']]
        if later_starts and reading['step'] > min(later_starts):
            continue
        windows.append(Window(first, reading['step'], (cumulative - cumulative_before) / (reading['step'] - first),
                              reading['lr']))
    return sorted(windows, key=lambda window: window.last)


def binned_training_loss(windows: list[Window], width: int) -> tuple[np.ndarray, np.ndarray]:
    """Update-weighted mean loss per ``width`` updates; each window counts in the bin of its last update."""
    last = np.array([window.last for window in windows])
    updates = np.array([window.last - window.first for window in windows])
    losses = np.array([window.loss for window in windows])
    bins = (last - 1) // width
    loss_totals = np.bincount(bins, weights=losses * updates)
    update_totals = np.bincount(bins, weights=updates)
    present = update_totals > 0
    centres = (np.arange(len(update_totals)) + 0.5) * width
    return centres[present], loss_totals[present] / update_totals[present]


def lr_colours(runs: list[PilotRun]) -> dict[float, str]:
    rates = sorted({run.learning_rate for run in runs})
    return dict(zip(rates, (ANNOTATOR_COLOUR, MONOTRACK_COLOUR), strict=True))


def run_handles(runs: list[PilotRun], colours: dict[float, str]) -> list:
    handles: list = [Line2D([], [], color=colour, linewidth=2.2, label=f'Head LR {rate_label(rate)}')
                     for rate, colour in colours.items()]
    for seed in sorted({run.seed for run in runs}):
        handles.append(Line2D([], [], color=MUTED, linestyle=SEED_LINES[seed], marker=SEED_MARKERS[seed],
                              label=f'Seed {seed}'))
    handles.append(Line2D([], [], color=TEXT, marker='o', markersize=11, markerfacecolor='none', linestyle='none',
                          label='Selected checkpoint: lowest validation loss'))
    handles.append(Patch(color=GRID, alpha=0.6, label=f'Continuation after {ORIGINAL_UPDATES // 1000}k updates'))
    return handles


def mark_in_progress(axes: plt.Axes, run: PilotRun, value: float) -> None:
    if not run.finished:
        axes.annotate('still training', (run.history[-1]['steps'], value), xytext=(6, 0),
                      textcoords='offset points', va='center', fontsize=8.5, color=MUTED, style='italic')


def selection_figure(ranking: list[list[PilotRun]], selected: bool, x3d_rows: DotRow, populations: tuple[int, int],
                     out: Path) -> None:
    colours = lr_colours([run for members in ranking for run in members])
    rows = []
    for rank, members in enumerate(ranking):
        rate = members[0].learning_rate
        label = f'VideoMAE-B, LR {rate_label(rate)}'
        if selected and rank == 0:
            label += ' (selected)'
        unfinished = [run for run in members if not run.finished]
        if unfinished:
            label += '\n' + ', '.join(f'seed {run.seed}' for run in unfinished) + ' still training'
        rows.append(DotRow(label, colours[rate], 'o', {metric: [run.score(metric) for run in members]
                                                       for metric, _title in METRICS}))
    rows.append(x3d_rows)
    rallies, contacts = populations
    notes = [
        ('Each run is scored at its lowest-validation-loss checkpoint. Dots: mean of seeds 0 and 1; black lines '
         'join the two seeds. Axes do not start at zero.'),
        ('Rally bounds are given; a hit counts within ±10 frames at 30 fps. X3D-M: its selected configuration on '
         'the same validation rallies.'),
    ]
    if not selected:
        notes.append('The learning-rate order is provisional until every run finishes and the queue selects one.')
    for suffix in ('.png', '.svg'):
        dot_plot(rows, f'VideoMAE-B validation pilot: 8 videos, {rallies:,} rallies, {contacts:,} labelled hits',
                 tuple(notes), out / f'videomae_fig1_validation_selection{suffix}')


def training_figure(runs: list[PilotRun], out: Path) -> None:
    colours = lr_colours(runs)
    figure, (rate_axes, train_axes, validation_axes) = plt.subplots(3, 1, figsize=(12, 11), sharex=True,
                                                                    facecolor=SURFACE)
    last_step = max(run.history[-1]['steps'] for run in runs)
    for axes in (rate_axes, train_axes, validation_axes):
        style_axes(axes, 'both')
        axes.axvspan(ORIGINAL_UPDATES, last_step + 2500, color=GRID, alpha=0.6, linewidth=0, zorder=0)
    for run in runs:
        colour, linestyle, marker = colours[run.learning_rate], SEED_LINES[run.seed], SEED_MARKERS[run.seed]
        windows = training_windows(run.progress)
        rate_axes.plot([window.last for window in windows], [window.rate for window in windows], color=colour,
                       linestyle=linestyle, linewidth=1.6)
        centres, losses = binned_training_loss(windows, LOSS_BIN_UPDATES)
        train_axes.plot(centres, losses, color=colour, linestyle=linestyle, linewidth=1.4)
        steps = [check['steps'] for check in run.history]
        validation_losses = [check['validation']['loss'] for check in run.history]
        validation_axes.plot(steps, validation_losses, color=colour, linestyle=linestyle, linewidth=1.8,
                             marker=marker, markersize=4.5)
        validation_axes.scatter(run.selected['steps'], run.selected['validation']['loss'], s=150, facecolor='none',
                                edgecolor=TEXT, linewidth=1.5, zorder=4)
        mark_in_progress(validation_axes, run, validation_losses[-1])
    rate_axes.yaxis.set_major_formatter(FuncFormatter(lambda rate, _position: f'{rate * 1e5:g}'))
    rate_axes.set_ylabel('Head learning rate (×10⁻⁵)', color=TEXT)
    rate_axes.set_title('Learning rate; both seeds share each schedule. Each earlier transformer block uses 0.75 '
                        'times the rate of the block above it.',
                        fontsize=10, color=TEXT, loc='left')
    # Log scale keeps the warm-up losses in view without flattening the later differences.
    train_axes.set_yscale('log')
    train_axes.yaxis.set_major_locator(FixedLocator([0.15, 0.2, 0.3, 0.5, 0.7, 1.0]))
    train_axes.yaxis.set_minor_locator(NullLocator())
    train_axes.yaxis.set_major_formatter(FuncFormatter(lambda loss, _position: f'{loss:g}'))
    train_axes.set_ylabel('Cross-entropy (log scale)', color=TEXT)
    train_axes.set_title(f'Training loss on sampled batches, half centred on contacts; mean per '
                         f'{LOSS_BIN_UPDATES:,} updates', fontsize=10, color=TEXT, loc='left')
    validation_axes.set_ylabel('Cross-entropy', color=TEXT)
    validation_axes.set_title(f'Validation loss on every validation centre, about {VALIDATION_CONTACT_PERCENT}% '
                              'near contacts; checked at 2,500, 5,000, then every 5,000 updates',
                              fontsize=10, color=TEXT, loc='left')
    validation_axes.set_xlabel('Updates (batches of 16 clips)', color=TEXT)
    validation_axes.set_xlim(0, last_step + 2500)
    validation_axes.xaxis.set_major_formatter(FuncFormatter(thousands))
    figure.suptitle('VideoMAE-B pilot: learning rate, training loss and validation loss for all four runs',
                    fontsize=13, color=TEXT, x=0.01, ha='left')
    figure.legend(handles=run_handles(runs, colours), loc='upper left', bbox_to_anchor=(0.01, 0.965), ncol=3,
                  frameon=False, fontsize=9.5)
    figure.text(0.01, 0.012, 'Training and validation losses use different contact rates, so their levels are not '
                'directly comparable. Compare each with itself over time.', fontsize=9, color=MUTED)
    figure.tight_layout(rect=(0, 0.03, 1, 0.925), h_pad=2.0)
    save(figure, out, 'videomae_fig2_training')


def validation_events_figure(runs: list[PilotRun], out: Path) -> None:
    colours = lr_colours(runs)
    figure, axes_row = plt.subplots(1, 3, figsize=(14, 5.4), sharex=True, facecolor=SURFACE)
    last_step = max(run.history[-1]['steps'] for run in runs)
    for axes, (metric, title) in zip(axes_row, METRICS, strict=True):
        style_axes(axes, 'both')
        axes.axvspan(ORIGINAL_UPDATES, last_step + 2500, color=GRID, alpha=0.6, linewidth=0, zorder=0)
        for run in runs:
            steps = [check['steps'] for check in run.history]
            scores = [check['events'][TOLERANCE][metric] for check in run.history]
            axes.plot(steps, scores, color=colours[run.learning_rate], linestyle=SEED_LINES[run.seed],
                      linewidth=1.6, marker=SEED_MARKERS[run.seed], markersize=4)
            axes.scatter(run.selected['steps'], run.score(metric), s=150, facecolor='none', edgecolor=TEXT,
                         linewidth=1.5, zorder=4)
            mark_in_progress(axes, run, scores[-1])
        axes.set_title(title, fontsize=11, color=TEXT, loc='left')
        axes.set_xlabel('Updates', color=TEXT)
        axes.xaxis.set_major_formatter(FuncFormatter(thousands))
    axes_row[0].set_xlim(0, last_step + 2500)
    figure.suptitle('VideoMAE-B validation event scores at every check, all four runs', fontsize=13, color=TEXT,
                    x=0.01, ha='left')
    figure.legend(handles=run_handles(runs, colours), loc='upper left', bbox_to_anchor=(0.01, 0.94), ncol=6,
                  frameon=False, fontsize=9)
    figure.text(0.01, 0.015, 'Rally bounds are given; a hit counts within ±10 frames at 30 fps. Circles mark the '
                'checkpoint chosen by lowest validation loss, not by these scores. Axes do not start at zero.',
                fontsize=9, color=MUTED)
    figure.tight_layout(rect=(0, 0.04, 1, 0.9))
    save(figure, out, 'videomae_fig3_validation_events')


def read_scores(path: Path, runs: list[PilotRun]) -> dict[tuple[str, int, int, str, str], dict]:
    """Rows keyed by (run, stride, offset, hitter rule, tolerance), checked against the histories' checkpoints."""
    selected_steps = {run.name: run.selected['steps'] for run in runs}
    rows = {}
    with gzip.open(path, 'rt', newline='') as source:
        for row in csv.DictReader(source):
            if int(row['best_step']) != selected_steps[row['run']]:
                raise ValueError(f'{path}: {row["run"]} was rescored at {row["best_step"]} updates, but its history '
                                 f'now selects {selected_steps[row["run"]]}; rerun rescore_videomae.py')
            key = (row['run'], int(row['stride']), int(row['offset']), row['hitter_rule'], row['tolerance'])
            rows[key] = {name: float(value) for name, value in row.items()
                         if name not in ('split', 'run', 'hitter_rule')}
    return rows


def available_samplings(scores: dict, members: list[PilotRun], tolerance: str) -> list[tuple[int, int]]:
    """Return sampling conditions present for every run and hitter rule at this tolerance."""
    expected: set[tuple[int, int]] | None = None
    for run in members:
        conditions_by_rule = {
            rule: {(stride, offset) for name, stride, offset, row_rule, row_tolerance in scores
                   if name == run.name and row_rule == rule and row_tolerance == tolerance}
            for rule in ('raw', 'alternating')
        }
        if conditions_by_rule['raw'] != conditions_by_rule['alternating']:
            raise ValueError(f'{run.name}: raw and alternating scores have different sampling conditions at '
                             f'±{tolerance} frames')
        conditions = conditions_by_rule['raw']
        if not conditions:
            raise ValueError(f'{run.name}: no scores at ±{tolerance} frames')
        if expected is not None and conditions != expected:
            raise ValueError(f'{run.name}: sampling conditions {sorted(conditions)} differ from '
                             f'{sorted(expected)} for the other runs')
        expected = conditions
    return sorted(expected or set())


def test_sampling(paths: list[Path]) -> tuple[int, int]:
    """Read test sampling metadata, treating historical events without it as dense."""
    metadata = [read_json(path).get('sampling') for path in paths]
    if any(value is None for value in metadata) and not all(value is None for value in metadata):
        raise ValueError('VideoMAE test seeds disagree: some events have sampling metadata and others do not')
    if all(value is None for value in metadata):
        return 1, 0
    if any(value != metadata[0] for value in metadata[1:]):
        raise ValueError(f'VideoMAE test seeds have inconsistent sampling metadata: {metadata}')
    sampling = metadata[0]
    return int(sampling['stride']), int(sampling['offset'])


def sampling_label(sampling: tuple[int, int]) -> str:
    stride, offset = sampling
    if stride == 1:
        return 'dense (every frame)'
    return f'stride {stride}, offset {offset}'


def video_system_label(rate: float, seed_count: int, sampling: tuple[int, int]) -> str:
    return f'VideoMAE-B test, {sampling_label(sampling)}\nLR {rate_label(rate)} ({seed_count} seeds)'


def stride_figure(scores: dict, members: list[PilotRun], colours: dict[float, str], group_labels: list[str],
                  title: str, notes: tuple[str, ...], out: Path, stem: str) -> None:
    """Compare only the dense and stride conditions present in the score file."""
    figure, axes_grid = plt.subplots(3, 2, figsize=(12, 11.5), facecolor=SURFACE)
    groups = np.arange(len(members))
    samplings = available_samplings(scores, members, TOLERANCE)
    stride_samplings = [sampling for sampling in samplings if sampling != (1, 0)]
    if len(stride_samplings) == 3:
        stride_shifts = (0.02, 0.13, 0.24)
    elif len(stride_samplings) == 1 and (1, 0) not in samplings:
        stride_shifts = (0.0,)
    else:
        first_shift, last_shift = ((0.02, 0.24) if (1, 0) in samplings else (-0.12, 0.12))
        stride_shifts = np.linspace(first_shift, last_shift, len(stride_samplings))
    for panel_row, panels in zip(axes_grid, STRIDE_PANELS, strict=True):
        for axes, (metric, rule, panel_title) in zip(panel_row, panels, strict=True):
            style_axes(axes, 'y')
            for group, run in zip(groups, members, strict=True):
                colour = colours[run.learning_rate]
                if (1, 0) in samplings:
                    dense = scores[run.name, 1, 0, rule, TOLERANCE][metric]
                    axes.plot([group - 0.32, group + 0.32], [dense, dense], color=MUTED, linewidth=0.9,
                              linestyle=':')
                    axes.scatter(group - 0.16, dense, s=80, color=colour, marker='o', edgecolor=SURFACE, zorder=3)
                for (stride, offset), shift in zip(stride_samplings, stride_shifts, strict=True):
                    value = scores[run.name, stride, offset, rule, TOLERANCE][metric]
                    axes.scatter(group + shift, value, s=48, facecolor=SURFACE, edgecolor=colour, marker='^',
                                 linewidth=1.6, zorder=3)
            axes.set_xticks(groups, group_labels, fontsize=9.5)
            axes.set_xlim(-0.6, len(members) - 0.4)
            axes.set_title(panel_title, fontsize=10.5, color=TEXT, loc='left')
        low = min(axes.dataLim.y0 for axes in panel_row)
        high = max(axes.dataLim.y1 for axes in panel_row)
        margin = 0.08 * (high - low)
        for axes in panel_row:
            axes.set_ylim(low - margin, high + margin)
    handles = []
    if (1, 0) in samplings:
        handles.append(Line2D([], [], color=MUTED, marker='o', markersize=8, linestyle='none',
                              label='Dense: every frame'))
    if stride_samplings == [(3, 0), (3, 1), (3, 2)]:
        handles.append(Line2D([], [], color=MUTED, marker='^', markersize=7, markerfacecolor=SURFACE,
                              linestyle='none', label='Stride 3: offsets 0, 1 and 2, left to right'))
    else:
        handles.extend(Line2D([], [], color=MUTED, marker='^', markersize=7, markerfacecolor=SURFACE,
                               linestyle='none', label=f'Stride {stride}, offset {offset}')
                       for stride, offset in stride_samplings)
    if (1, 0) in samplings:
        handles.append(Line2D([], [], color=MUTED, linestyle=':', label='Dense level, for reference'))
    rates = sorted({run.learning_rate for run in members})
    handles += [Line2D([], [], color=colours[rate], marker='s', linestyle='none', label=f'Head LR {rate_label(rate)}')
                for rate in rates]
    figure.suptitle(title, fontsize=13, color=TEXT, x=0.01, ha='left')
    figure.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.01, 0.955), ncol=3, frameon=False, fontsize=9.5)
    note_height = 0.018
    for index, note in enumerate(reversed(notes)):
        figure.text(0.01, 0.01 + index * note_height, note, fontsize=9, color=MUTED)
    figure.tight_layout(rect=(0, 0.015 + len(notes) * note_height, 1, 0.925), h_pad=2.2)
    save(figure, out, stem)


def seed_scores(paths: list[Path], tolerance: str) -> dict[str, list[float]]:
    scores: dict[str, list[float]] = {metric: [] for metric, _title in METRICS}
    for path in paths:
        metrics = read_json(path)['metrics'][tolerance]
        if (metrics['rallies'], metrics['labelled']) != (TEST_RALLIES, TEST_CONTACTS):
            raise ValueError(f'{path}: scored {metrics["rallies"]} rallies and {metrics["labelled"]} hits, '
                             f'not the shared test set of {TEST_RALLIES} and {TEST_CONTACTS}')
        for metric, _title in METRICS:
            scores[metric].append(metrics[metric])
    return scores


def test_systems(videomae_paths: list[Path], results: Path, monotrack_summary: Path, rate: float,
                 sampling: tuple[int, int], tolerance: str) -> list[DotRow]:
    """VideoMAE's selected seeds beside X3D-M, our annotator and MonoTrack, all given the rally bounds."""
    summary = read_json(monotrack_summary)['summary']
    systems = [
        DotRow(video_system_label(rate, len(videomae_paths), sampling), VIDEOMAE_COLOUR, '^',
               seed_scores(videomae_paths, tolerance)),
        DotRow('X3D-M, selected configuration (2 seeds)', X3D_COLOUR, 'D',
               seed_scores(sorted((results / 'x3d_test').glob('seed*/events.json.gz')), tolerance)),
    ]
    for key, label, colour, marker in (('annotator_our_courts', 'Our annotator, our courts', ANNOTATOR_COLOUR, 'o'),
                                       ('monotrack_own_courts', 'MonoTrack, its own courts', MONOTRACK_COLOUR, 's')):
        scores = {metric: summary[key][tolerance][MONOTRACK_METRICS[metric]]['runs'] for metric, _title in METRICS}
        systems.append(DotRow(f'{label} ({len(scores["fully_correct_percent"])} seeds)', colour, marker, scores))
    return systems


def print_validation_tables(runs: list[PilotRun], ranking: list[list[PilotRun]], selected: bool,
                            scores: dict) -> None:
    print('| Run | Status | Selected (updates) | Last check (updates) | Validation loss | Fully correct rallies (%) '
          '| Correct-player F1 (%) | Contact F1 (%) |')
    print('|---|---|---:|---:|---:|---:|---:|---:|')
    for run in runs:
        status = 'complete' if run.finished else 'still training'
        print(f"| {run.label} | {status} | {run.selected['steps']:,} | {run.history[-1]['steps']:,} "
              f"| {run.selected['validation']['loss']:.4f} | {run.score('fully_correct_percent'):.2f} "
              f"| {run.score('correct_player_f1'):.2f} | {run.score('contact_f1'):.2f} |")
    print(f"\nLearning-rate ranking ({'selected by the queue' if selected else 'provisional'}):\n")
    print('| Rank | Head LR | Fully correct rallies (%), seed 0 / 1 / mean | Correct-player F1 (%), seed 0 / 1 / mean '
          '| Contact F1 (%), mean | Selected checkpoints |')
    print('|---:|---|---:|---:|---:|---|')
    for rank, members in enumerate(ranking, start=1):
        cells = []
        for metric in ('fully_correct_percent', 'correct_player_f1'):
            seed0, seed1 = (run.score(metric) for run in members)
            cells.append(f'{seed0:.2f} / {seed1:.2f} / **{mean_score(members, metric):.2f}**')
        steps = ' / '.join(f"{run.selected['steps'] // 1000}k" for run in members)
        print(f"| {rank} | {rate_label(members[0].learning_rate)} | {cells[0]} | {cells[1]} "
              f"| {mean_score(members, 'contact_f1'):.2f} | {steps} |")
    for tolerance in (TOLERANCE, '5'):
        print_stride_table(scores, runs, tolerance)


def print_stride_table(scores: dict, members: list[PilotRun], tolerance: str) -> None:
    samplings = available_samplings(scores, members, tolerance)
    labels = ', '.join(SAMPLING_LABELS.get(sampling, f'Stride {sampling[0]}, offset {sampling[1]}')
                       for sampling in samplings)
    print(f'\nAvailable sampling scores (%) at ±{tolerance} frames: {labels}\n')
    print('| Run | Sampling | Contact recall | Contact precision | Contact F1 | Raw player F1 | Alternating player F1 '
          '| Raw fully correct | Alternating fully correct |')
    print('|---|---|---:|---:|---:|---:|---:|---:|---:|')
    for run in members:
        for stride, offset in samplings:
            sampling = SAMPLING_LABELS.get((stride, offset), f'Stride {stride}, offset {offset}')
            raw = scores[run.name, stride, offset, 'raw', tolerance]
            alternating = scores[run.name, stride, offset, 'alternating', tolerance]
            print(f"| {run.label} | {sampling} | {raw['contact_recall']:.2f} | {raw['contact_precision']:.2f} "
                  f"| {raw['contact_f1']:.2f} | {raw['correct_player_f1']:.2f} "
                  f"| {alternating['correct_player_f1']:.2f} | {raw['fully_correct_percent']:.2f} "
                  f"| {alternating['fully_correct_percent']:.2f} |")


def print_test_table(systems_by_tolerance: dict[str, list[DotRow]]) -> None:
    print('\n| Tolerance | System | Contact F1 (%) | Correct-player F1 (%) | Fully correct rallies (%) |')
    print('|---|---|---:|---:|---:|')
    for tolerance, systems in systems_by_tolerance.items():
        for system in systems:
            cells = [f'{np.mean(system.scores[metric]):.2f} ({min(system.scores[metric]):.2f}–'
                     f'{max(system.scores[metric]):.2f})' for metric, _title in METRICS]
            print(f"| ±{tolerance} frames | {system.label} | {' | '.join(cells)} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', type=Path, default=RESULTS)
    parser.add_argument('--out', type=Path, default=FIGURES)
    parser.add_argument('--monotrack-summary', type=Path, default=MONOTRACK_SUMMARY,
                        help="MonoTrack benchmark's contact_summary.json.gz, for the test comparison")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.results / 'videomae_pilot')
    ranking = rank_learning_rates(runs)
    selection_path = args.results / 'videomae_pilot' / 'selection.json.gz'
    selected = selection_path.exists()
    if selected:
        check_selection(ranking, read_json(selection_path))
    populations = validation_population(runs)
    x3d_best = rank_configurations(load_x3d_runs(args.results / 'x3d_pilot'))[0]
    if validation_population(runs) != (x3d_best.runs[0].selected['events'][TOLERANCE]['rallies'],
                                       x3d_best.runs[0].selected['events'][TOLERANCE]['labelled']):
        raise ValueError('X3D-M and VideoMAE-B were validated on different populations')
    x3d_row = DotRow('X3D-M, selected configuration', X3D_COLOUR, 'D',
                     {metric: x3d_best.scores(metric) for metric, _title in METRICS})
    validation_scores = read_scores(args.results / 'videomae_stride3' / 'validation_scores.csv.gz', runs)
    print_validation_tables(runs, ranking, selected, validation_scores)
    selection_figure(ranking, selected, x3d_row, populations, args.out)
    training_figure(runs, args.out)
    validation_events_figure(runs, args.out)
    rallies, contacts = populations
    stride_figure(validation_scores, runs, lr_colours(runs), [run.label.replace(', ', '\n') for run in runs],
                  f'VideoMAE-B validation: dense versus every third frame, {rallies:,} rallies, {contacts:,} hits',
                  VALIDATION_STRIDE_NOTES, args.out, 'videomae_fig4_stride_validation')
    test_directory = args.results / 'videomae_test'
    if not test_directory.exists():
        print('\nTest results: pending. Run rescore_videomae.py with --test-index once both test seeds finish.')
        return
    if not selected:
        raise ValueError('Test results exist without the queue selection that chose their checkpoints')
    chosen = ranking[0]
    paths = [test_directory / f'seed{run.seed}' / 'events.json.gz' for run in chosen]
    test_sampling_condition = test_sampling(paths)
    systems = {tolerance: test_systems(paths, args.results, args.monotrack_summary, chosen[0].learning_rate,
                                       test_sampling_condition, tolerance)
               for tolerance in (TOLERANCE, '5')}
    print_test_table(systems)
    for suffix in ('.png', '.svg'):
        dot_plot(systems[TOLERANCE], f'Held-out test: 46 ShuttleSet22 videos, {TEST_RALLIES:,} rallies, '
                 f'{TEST_CONTACTS:,} labelled hits',
                 TEST_COMPARISON_NOTES, args.out / f'videomae_fig5_test_comparison{suffix}')
    test_scores = read_scores(args.results / 'videomae_stride3' / 'test_scores.csv.gz', chosen)
    for tolerance in (TOLERANCE, '5'):
        print_stride_table(test_scores, chosen, tolerance)
    stride_figure(test_scores, chosen, lr_colours(runs),
                  [f'Seed {run.seed}\nselected at {run.selected["steps"] // 1000}k' for run in chosen],
                  f'VideoMAE-B held-out test: {sampling_label(test_sampling_condition)}, '
                  f'{TEST_RALLIES:,} rallies',
                  TEST_STRIDE_NOTES, args.out, 'videomae_fig6_stride_test')


if __name__ == '__main__':
    main()
