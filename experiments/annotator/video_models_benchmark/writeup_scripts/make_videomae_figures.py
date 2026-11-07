"""Draw the VideoMAE (base) pilot figures and print the report tables.

Reads the compact run records and stride-three scores that ``rescore_videomae.py``
copies into ``results/``. Each run is scored at its earliest minimum-validation-CE
checkpoint. A run still inside its continuation is marked as in progress, and the
learning-rate ranking stays provisional until the queue writes ``selection.json.gz``.
Figures reuse the X3D and MonoTrack report styles.

Usage, from the repository root:
    python experiments/annotator/video_models_benchmark/writeup_scripts/make_videomae_figures.py

Once ``results/videomae_test/`` holds both selected test seeds, the same command
also draws the shared test comparison. Once
``compare_videomae_lr5e4.py`` has filled ``results/videomae_lr5e4/``, it also draws
all three learning rates at stride-three validation. That trial is one
validation-only seed and does not enter the learning-rate ranking.
"""
from __future__ import annotations

import argparse
import csv
import gzip
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
    read_json,
    sampling_scores,
)

BENCHMARK = Path(__file__).resolve().parents[1]
RESULTS = BENCHMARK / 'results'
FIGURES = BENCHMARK / 'writeup_figures'
MONOTRACK_SUMMARY = (BENCHMARK.parent / 'monotrack_benchmark/benchmark_results/hp_search/primary'
                     / 'contact_summary.json.gz')
VIDEOMAE_COLOUR = '#138a8a'
# Indigo stays distinct from LR 1e-4's orange under protanopia; the two share one figure.
TRIAL_COLOUR = '#6250d6'
SEED_LINES = {0: '-', 1: (0, (5, 2))}
LOSS_BIN_UPDATES = 1_000
SAMPLING_LABELS = {(1, 0): 'Dense', (3, 0): 'Stride 3, offset 0', (3, 1): 'Stride 3, offset 1',
                   (3, 2): 'Stride 3, offset 2'}


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
    figure.savefig(out / f'{stem}.png', dpi=150, facecolor=SURFACE)
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
    handles: list = [Line2D([], [], color=colour, linewidth=2.2, label=f'LR {rate_label(rate)}')
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
    rate_axes.set_ylabel('Learning rate (×10⁻⁵)', color=TEXT)
    rate_axes.set_title('Learning-rate schedule', fontsize=10, color=TEXT, loc='left')
    # Log scale keeps the warm-up losses in view without flattening the later differences.
    train_axes.set_yscale('log')
    train_axes.yaxis.set_major_locator(FixedLocator([0.15, 0.2, 0.3, 0.5, 0.7, 1.0]))
    train_axes.yaxis.set_minor_locator(NullLocator())
    train_axes.yaxis.set_major_formatter(FuncFormatter(lambda loss, _position: f'{loss:g}'))
    train_axes.set_ylabel('Cross-entropy (log scale)', color=TEXT)
    train_axes.set_title(f'Training loss on sampled batches, half centred on contacts; mean per '
                         f'{LOSS_BIN_UPDATES:,} updates', fontsize=10, color=TEXT, loc='left')
    validation_axes.set_ylabel('Cross-entropy', color=TEXT)
    validation_axes.set_title('Validation loss (stride 1)', fontsize=10, color=TEXT, loc='left')
    validation_axes.set_xlabel('Updates (batches of 16 clips)', color=TEXT)
    validation_axes.set_xlim(0, last_step + 2500)
    validation_axes.xaxis.set_major_formatter(FuncFormatter(thousands))
    figure.suptitle('VideoMAE (base) pilot: learning rate, training loss and validation loss for all four runs',
                    fontsize=13, color=TEXT, x=0.01, ha='left')
    figure.legend(handles=run_handles(runs, colours), loc='upper left', bbox_to_anchor=(0.01, 0.965), ncol=3,
                  frameon=False, fontsize=9.5)
    figure.text(0.01, 0.012, 'LR is shown for the classifier; each earlier transformer block uses 0.75× '
                'the rate of the block above it.',
                fontsize=9, color=MUTED)
    figure.tight_layout(rect=(0, 0.03, 1, 0.925), h_pad=2.0)
    save(figure, out, 'videomae_training')


def checked_rows(path: Path, runs: list[PilotRun]) -> list[dict[str, str]]:
    """Read a score CSV whose every row must come from its run's current lowest-loss checkpoint."""
    selected_steps = {run.name: run.selected['steps'] for run in runs}
    with gzip.open(path, 'rt', newline='') as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        if int(row['best_step']) != selected_steps[row['run']]:
            raise ValueError(f'{path}: {row["run"]} was rescored at {row["best_step"]} updates, but its history '
                             f'now selects {selected_steps[row["run"]]}; rerun the script that wrote this file')
    return rows


def read_scores(path: Path, runs: list[PilotRun]) -> dict[tuple[str, int, int, str, str], dict]:
    """Rows keyed by (run, stride, offset, hitter rule, tolerance), checked against the histories' checkpoints."""
    rows = {}
    for row in checked_rows(path, runs):
        key = (row['run'], int(row['stride']), int(row['offset']), row['hitter_rule'], row['tolerance'])
        rows[key] = {name: float(value) for name, value in row.items() if name not in ('split', 'run', 'hitter_rule')}
    return rows


def read_frame_scores(path: Path, runs: list[PilotRun]) -> dict[tuple[str, int, int], dict]:
    """Validation cross-entropy and macro F1 keyed by (run, stride, offset)."""
    rows = {}
    for row in checked_rows(path, runs):
        rows[row['run'], int(row['stride']), int(row['offset'])] = {
            'cross_entropy': float(row['cross_entropy']), 'macro_f1': float(row['macro_f1'])}
    return rows


def validation_sampling(run: PilotRun) -> tuple[int, int]:
    """The run's validation sampling; histories that predate sampling metadata were dense."""
    sampling = run.history[-1]['validation'].get('sampling', {'stride': 1, 'offset': 0})
    return sampling['stride'], sampling['offset']


def training_rate(progress: list[dict]) -> float:
    """Training updates per second, between the first and last progress reading of each interval."""
    intervals: dict[tuple[float, int], list[dict]] = {}
    for reading in progress:
        intervals.setdefault((reading['session_started'], reading['interval_start_step']), []).append(reading)
    updates = seconds = 0.0
    for readings in intervals.values():
        updates += readings[-1]['step'] - readings[0]['step']
        seconds += readings[-1]['updated_at'] - readings[0]['updated_at']
    return updates / seconds


def recorded_hours(run: PilotRun, last_step: int) -> float:
    """Training, validation and event-scoring time recorded up to a check; start-up time is not recorded."""
    return sum(check['seconds'] for check in run.history if check['steps'] <= last_step) / 3600


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


def test_systems(videomae_paths: list[Path], results: Path, monotrack_summary: Path, tolerance: str) -> list[DotRow]:
    """VideoMAE's selected seeds beside X3D-M, our annotator and MonoTrack, all given the rally bounds."""
    summary = read_json(monotrack_summary)['summary']
    systems = [
        DotRow('VideoMAE (base)', VIDEOMAE_COLOUR, '^',
               seed_scores(videomae_paths, tolerance)),
        DotRow('X3D-M', X3D_COLOUR, 'D',
               sampling_scores(results / 'x3d_stride3' / 'test', tolerance)),
    ]
    for key, label, colour, marker in (('annotator_our_courts', 'Our annotator', ANNOTATOR_COLOUR, 'o'),
                                       ('monotrack_own_courts', 'MonoTrack', MONOTRACK_COLOUR, 's')):
        scores = {metric: summary[key][tolerance][MONOTRACK_METRICS[metric]]['runs'] for metric, _title in METRICS}
        systems.append(DotRow(label, colour, marker, scores))
    return systems


def validation_figure(runs: list[PilotRun], scores: dict, selected_rate: float, out: Path) -> None:
    """Compare every learning rate on the same stride-three validation centres."""
    groups: dict[float, list[PilotRun]] = {}
    for run in runs:
        groups.setdefault(run.learning_rate, []).append(run)
    rows = []
    colours = (ANNOTATOR_COLOUR, MONOTRACK_COLOUR, TRIAL_COLOUR)
    for index, (rate, members) in enumerate(sorted(groups.items())):
        label = f'LR {rate_label(rate)}' + (' (selected)' if rate == selected_rate else '')
        values = {metric: [scores[run.name, 3, 0, 'alternating', TOLERANCE][metric] for run in members]
                  for metric, _title in METRICS}
        rows.append(DotRow(label, colours[index], 'D' if len(members) == 1 else 'o', values))
    notes = (
        'Stride 3, offset 0; enforced alternation; given rally bounds; contact tolerance ±10 frames at 30 fps.',
        'Dots: seed mean; lines: seed range. LR 3.125e-5 and 1e-4: two seeds each. LR 5e-4: seed 0 only.',
        'All configurations use positive centres within ±1 frame and 50% contact clips per batch.',
    )
    rallies, contacts = validation_population(runs)
    dot_plot(rows, f'VideoMAE (base) validation: 8 videos, {rallies:,} rallies, {contacts:,} labelled hits',
             notes, out / 'videomae_validation.png')


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
    print('| Rank | LR | Fully correct rallies (%), seed 0 / 1 / mean | Correct-player F1 (%), seed 0 / 1 / mean '
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
            # Figure labels wrap onto two lines; a Markdown table cell must stay on one.
            label = system.label.replace('\n', ', ')
            print(f"| ±{tolerance} frames | {label} | {' | '.join(cells)} |")


def print_trial_tables(runs: list[PilotRun], trial: PilotRun, reference: list[PilotRun], scores: dict,
                       frame: dict) -> None:
    """Every run at the trial's sampling, then training time for the trial and the selected learning rate."""
    stride, offset = validation_sampling(trial)
    print(f'\nValidation at stride {stride}, offset {offset}, each run at its selected checkpoint, '
          f'±{TOLERANCE} frames:\n')
    print('| Run | Selected (updates) | Validation loss | Macro F1 | Contact recall (%) | Contact precision (%) '
          '| Contact F1 (%) | Raw player F1 (%) | Alternating player F1 (%) | Fully correct rallies (%) |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for run in [*runs, trial]:
        raw = scores[run.name, stride, offset, 'raw', TOLERANCE]
        alternating = scores[run.name, stride, offset, 'alternating', TOLERANCE]
        frame_row = frame[run.name, stride, offset]
        print(f"| {run.label} | {run.selected['steps']:,} | {frame_row['cross_entropy']:.4f} "
              f"| {frame_row['macro_f1']:.4f} | {raw['contact_recall']:.2f} | {raw['contact_precision']:.2f} "
              f"| {raw['contact_f1']:.2f} | {raw['correct_player_f1']:.2f} | {alternating['correct_player_f1']:.2f} "
              f"| {alternating['fully_correct_percent']:.2f} |")
    print('\nTraining time. Recorded hours cover training, validation and event scoring at each check:\n')
    print('| Run | Validation sampling | Selected (updates) | Last check (updates) | Recorded hours to selected '
          '| Recorded hours to last check | Training updates per second | Estimated validation minutes per check |')
    print('|---|---|---:|---:|---:|---:|---:|---:|')
    for run in [*reference, trial]:
        rate = training_rate(run.progress)
        last_step = run.history[-1]['steps']
        total_seconds = recorded_hours(run, last_step) * 3600
        validation_minutes = (total_seconds - last_step / rate) / len(run.history) / 60
        print(f"| {run.label} | {sampling_label(validation_sampling(run))} | {run.selected['steps']:,} "
              f"| {last_step:,} | {recorded_hours(run, run.selected['steps']):.2f} | {total_seconds / 3600:.2f} "
              f"| {rate:.2f} | {validation_minutes:.1f} |")


def draw_test_results(args: argparse.Namespace, chosen: list[PilotRun]) -> None:
    """Draw the shared test comparison and retain detailed scores in tables."""
    paths = [args.results / 'videomae_test' / f'seed{run.seed}' / 'events.json.gz' for run in chosen]
    sampling = test_sampling(paths)
    systems = {tolerance: test_systems(paths, args.results, args.monotrack_summary, tolerance)
               for tolerance in (TOLERANCE, '5')}
    print_test_table(systems)
    notes = (
        (f'VideoMAE: LR {rate_label(chosen[0].learning_rate)}, stride {sampling[0]}, offset {sampling[1]}. '
         'X3D-M: LR 0.02, stride 3, offset 0. Both: ±1-frame positives, 50% contact clips.'),
        'All systems: given rally bounds, enforced alternation, contact tolerance ±10 frames at 30 fps.',
        ('Dots: seed mean; lines: seed range. Video models: 2 seeds each; '
         'annotator and MonoTrack: 5 each, using their own courts.'),
    )
    dot_plot(systems[TOLERANCE], f'Test: 46 videos, {TEST_RALLIES:,} rallies, {TEST_CONTACTS:,} labelled hits',
             notes, args.out / 'test_comparison.png')
    test_scores = read_scores(args.results / 'videomae_stride3' / 'test_scores.csv.gz', chosen)
    for tolerance in (TOLERANCE, '5'):
        print_stride_table(test_scores, chosen, tolerance)


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
    validation_scores = read_scores(args.results / 'videomae_stride3' / 'validation_scores.csv.gz', runs)
    print_validation_tables(runs, ranking, selected, validation_scores)
    training_figure(runs, args.out)
    chosen = ranking[0]
    test_directory = args.results / 'videomae_test'
    if test_directory.exists():
        if not selected:
            raise ValueError('Test results exist without the queue selection that chose their checkpoints')
        draw_test_results(args, chosen)
    else:
        print('\nTest results: pending. Run rescore_videomae.py with --test-index once both test seeds finish.')
    trial_directory = args.results / 'videomae_lr5e4'
    if trial_directory.exists():
        (trial,) = load_runs(trial_directory)
        trial_scores = read_scores(trial_directory / 'validation_scores.csv.gz', [trial])
        frame = read_frame_scores(trial_directory / 'frame_scores.csv.gz', [*runs, trial])
        print_trial_tables(runs, trial, chosen, {**validation_scores, **trial_scores}, frame)
        runs = [*runs, trial]
        validation_scores = {**validation_scores, **trial_scores}
    validation_figure(runs, validation_scores, chosen[0].learning_rate, args.out)


if __name__ == '__main__':
    main()
