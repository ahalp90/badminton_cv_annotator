"""Sweep the stage-8 thresholds over one cached whole-video shuttle track.

Stage B of the stage-8 sweep plan
(``local_scratch/autograder_architecture/stage8_sweep_plan.md`` section 5). Two
sequential phases over a single cached track, scored against ShuttleSet GT with
the committed scorer core in ``scripts.stage8_score``:

  1. boundary phase: a 3,000-config grid over the five rally-boundary thresholds
     (REST_SPEED, REST_WINDOW, END_REST_FRAMES, START_SPEED, START_MIN_FRAMES),
     contact thresholds pinned at shipped defaults; the frozen winner is the
     merge-penalised pick (``select_boundary_winner``), not the as-built sort.
  2. contact phase: a 45-config grid over the three contact thresholds
     (SMOOTH_WINDOW, MIN_DIR_CHANGE_DEG, MIN_CONTACT_SPEED) at the frozen
     boundary winner.

Each phase also emits a data-package CSV for a pending key ruling by Ariel:
``boundary_crowns.csv`` (the winner under each candidate boundary key plus the
coverage-vs-merges frontier) and ``contact_frontier.csv`` (the recall_5 /
precision_5 Pareto frontier). Both feed a ruling; neither picks the winner.

Both phases also run one extra labelled ``shipped_defaults`` config (all eight
thresholds at their config.py values) so the summary can always contrast against
what stage 8 ships with, even where the grid does not contain that exact config.

How the thresholds are set: ``scraper.stage8_rally_segmentation`` binds its eight
thresholds as module globals via ``from .config import ...``, and the functions
read those bare globals. So a config is applied by setting attributes on the
imported module (``stage8_module.REST_SPEED = ...``) before calling
``segment_video``. EVERY config sets ALL eight explicitly, so there is no
save/restore dance and each task leaves the module in a fully-known state.
Patching (rather than threading params through the stage-8 signatures) keeps
stage 8's committed code untouched; a params refactor is Ariel's call later.

Parallelism: a multiprocessing Pool fans the configs out. Module state is
per-process and one task runs at a time per process, so patching all eight
globals at the top of each task cannot leak across tasks. The track and GT load
once per worker via the pool initializer (same pattern as
``scripts.wrist_contact_separation.WorkerCtx``). Only the flat per-config row
crosses the process boundary; the full nested metrics dicts stay in the worker.

Contact-winner selection (the plan pins only the boundary rule) is ruled
recall-first at +/-5 (Ariel 2026-07-08): maximise +/-5 recall, then +/-5 F1,
then the unmerged count-gate pass rate, then closeness to shipped defaults.
Recall is unrecoverable downstream while precision is recovered there (wrist
filter, then the featurised verifier), so a miss costs more than a false hit.

Usage:
    python -m scripts.stage8_sweep \\
        --track-npy /scratch/.../1.npy --vid 1 \\
        --out-dir /scratch/.../stage8_sweep [--phase both] [--workers N] \\
        [--mask-npy /scratch/.../1_replay.npy] [--nan-smoothing]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from itertools import product
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

import scraper.stage8_rally_segmentation as stage8_module  # noqa: E402  — needs the src path above

from scripts.stage8_score import (  # noqa: E402  — sibling script, imported after the src path insert
    DEFAULT_SHOTS_MASTER,
    load_gt_rallies,
    score_stage8,
)

# Tolerances scored per config: +/-2 is the canonical credit, +/-5 the noisy-GT
# read (plan section 4). All four GT-offset bands are recorded so configs can be
# re-ranked per band straight from the CSV without a re-run; winner selection is
# unchanged (contact_sort_key).
SWEEP_TOLERANCES = (1, 2, 5, 10)


# ---------------------------------------------------------------------------
# Params and the module-global names they patch
# ---------------------------------------------------------------------------
class Stage8Params(NamedTuple):
    """One full stage-8 threshold config: all eight values, always set together."""

    rest_speed: float
    rest_window: int
    end_rest_frames: int
    start_speed: float
    start_min_frames: int
    smooth_window: int
    min_dir_change_deg: float
    min_contact_speed: float


# Field -> the UPPER_CASE global the stage-8 module reads. Single source for both
# the patch loop and the CSV param columns, so the two never drift.
PARAM_TO_MODULE_ATTR = {
    'rest_speed': 'REST_SPEED',
    'rest_window': 'REST_WINDOW',
    'end_rest_frames': 'END_REST_FRAMES',
    'start_speed': 'START_SPEED',
    'start_min_frames': 'START_MIN_FRAMES',
    'smooth_window': 'SMOOTH_WINDOW',
    'min_dir_change_deg': 'MIN_DIR_CHANGE_DEG',
    'min_contact_speed': 'MIN_CONTACT_SPEED',
}
PARAM_COLUMNS = list(PARAM_TO_MODULE_ATTR)

# The shipped defaults, read from config so this file never re-states the values.
SHIPPED_DEFAULTS = Stage8Params(
    rest_speed=stage8_module.REST_SPEED,
    rest_window=stage8_module.REST_WINDOW,
    end_rest_frames=stage8_module.END_REST_FRAMES,
    start_speed=stage8_module.START_SPEED,
    start_min_frames=stage8_module.START_MIN_FRAMES,
    smooth_window=stage8_module.SMOOTH_WINDOW,
    min_dir_change_deg=stage8_module.MIN_DIR_CHANGE_DEG,
    min_contact_speed=stage8_module.MIN_CONTACT_SPEED,
)

LABEL_GRID = 'grid'
LABEL_SHIPPED = 'shipped_defaults'


def _patch_stage8(params: Stage8Params) -> None:
    """Set all eight stage-8 module globals from ``params``.

    Every field is overwritten each call and one task runs at a time per process,
    so this cannot leak state across tasks. See the module docstring for why this
    patches rather than refactors the stage-8 signatures.
    """
    for field, attr in PARAM_TO_MODULE_ATTR.items():
        setattr(stage8_module, attr, getattr(params, field))


# ---------------------------------------------------------------------------
# Grids (plan section 5; contact grid extended down per the A0 measurement)
# ---------------------------------------------------------------------------
# Boundary grid widened one notch past each fence-post edge (Ariel 2026-07-08):
# the original 324-config grid clipped the plausible range, so each param gains
# the value(s) just outside its old span. 5 x 5 x 6 x 5 x 4 = 3,000 configs.
BOUNDARY_REST_SPEED = (0.002, 0.003, 0.005, 0.01, 0.02)
BOUNDARY_REST_WINDOW = (5, 7, 9, 15, 21)
BOUNDARY_END_REST_FRAMES = (20, 30, 45, 60, 75, 90)
BOUNDARY_START_SPEED = (0.01, 0.015, 0.02, 0.03, 0.05)
BOUNDARY_START_MIN_FRAMES = (1, 2, 3, 5)

CONTACT_SMOOTH_WINDOW = (3, 5, 7)
CONTACT_MIN_DIR_CHANGE_DEG = (30, 45, 60, 75, 90)
CONTACT_MIN_CONTACT_SPEED = (0.005, 0.01, 0.02)


def build_boundary_grid() -> list[Stage8Params]:
    """The 3,000 boundary configs; contact thresholds pinned at shipped defaults."""
    grid: list[Stage8Params] = []
    for rest_speed, rest_window, end_rest_frames, start_speed, start_min_frames in product(
        BOUNDARY_REST_SPEED, BOUNDARY_REST_WINDOW, BOUNDARY_END_REST_FRAMES,
        BOUNDARY_START_SPEED, BOUNDARY_START_MIN_FRAMES,
    ):
        grid.append(Stage8Params(
            rest_speed=rest_speed,
            rest_window=rest_window,
            end_rest_frames=end_rest_frames,
            start_speed=start_speed,
            start_min_frames=start_min_frames,
            smooth_window=SHIPPED_DEFAULTS.smooth_window,
            min_dir_change_deg=SHIPPED_DEFAULTS.min_dir_change_deg,
            min_contact_speed=SHIPPED_DEFAULTS.min_contact_speed,
        ))
    return grid


def build_contact_grid(boundary: Stage8Params) -> list[Stage8Params]:
    """The 45 contact configs; the five boundary thresholds frozen at ``boundary``."""
    grid: list[Stage8Params] = []
    for smooth_window, min_dir_change_deg, min_contact_speed in product(
        CONTACT_SMOOTH_WINDOW, CONTACT_MIN_DIR_CHANGE_DEG, CONTACT_MIN_CONTACT_SPEED,
    ):
        grid.append(boundary._replace(
            smooth_window=smooth_window,
            min_dir_change_deg=min_dir_change_deg,
            min_contact_speed=min_contact_speed,
        ))
    return grid


class SweepTask(NamedTuple):
    """One config to score, plus its provenance label ('grid' or shipped ref)."""

    label: str
    params: Stage8Params


def build_boundary_tasks() -> list[SweepTask]:
    """3,000 grid tasks plus the labelled shipped-defaults reference (3,001 total)."""
    tasks = [SweepTask(LABEL_GRID, params) for params in build_boundary_grid()]
    tasks.append(SweepTask(LABEL_SHIPPED, SHIPPED_DEFAULTS))
    return tasks


def build_contact_tasks(boundary_winner: Stage8Params) -> list[SweepTask]:
    """45 grid tasks plus the labelled shipped-defaults reference (46 total)."""
    tasks = [SweepTask(LABEL_GRID, params) for params in build_contact_grid(boundary_winner)]
    tasks.append(SweepTask(LABEL_SHIPPED, SHIPPED_DEFAULTS))
    return tasks


# ---------------------------------------------------------------------------
# Row flattening (the only place the nested metrics dict is projected to columns)
# ---------------------------------------------------------------------------
# One triple per scored band, in SWEEP_TOLERANCES order (ascending). Header and
# row builder both generate from these two tuples, so neither can drift.
SCORED_METRICS = ('recall', 'precision', 'f1')
TOLERANCE_COLUMNS = [
    f'{metric}_{tolerance}'
    for tolerance in SWEEP_TOLERANCES
    for metric in SCORED_METRICS
]

METRIC_COLUMNS = [
    'n_spans',
    'covered', 'covered_fraction', 'split', 'missed', 'merged_spans', 'spurious_spans',
    'start_alignment_mean', 'start_alignment_median',
    'count_gate_covered_fraction', 'count_gate_unmerged_fraction',
    *TOLERANCE_COLUMNS,
    'total_candidates',
]
ROW_COLUMNS = ['label'] + PARAM_COLUMNS + METRIC_COLUMNS


def flatten_row(label: str, params: Stage8Params, n_spans: int, metrics: dict) -> dict:
    """Project one scored config to a flat row: eight params plus scalar metrics.

    The full nested ``score_stage8`` dict is dropped here so only rows (not
    per-config metrics dicts) accumulate. None survives for any undefined metric
    (no covered rally leaves start_alignment None; a zero-count denominator
    leaves a fraction/precision/f1 None); ``_serialise_value`` blanks it on write.

    :param label: 'grid' or the shipped-defaults reference label.
    :param params: the config that produced ``metrics``.
    :param n_spans: number of detected rally spans (``len(spans)``).
    :param metrics: a ``score_stage8`` result scored at ``SWEEP_TOLERANCES``.
    :return: one dict keyed by ROW_COLUMNS.
    """
    boundaries = metrics['boundaries']
    contacts = metrics['contacts']
    start_alignment = boundaries['start_alignment']  # dict or None (no covered rally)
    count_gate = contacts['count_gate']
    tolerance_curves = contacts['tolerances']  # {str(tol): {recall, precision, f1, candidates, ...}}

    row: dict = {'label': label}
    for field in PARAM_COLUMNS:
        row[field] = getattr(params, field)
    row['n_spans'] = n_spans
    row['covered'] = boundaries['covered']
    row['covered_fraction'] = boundaries['covered_fraction']
    row['split'] = boundaries['split']
    row['missed'] = boundaries['missed']
    row['merged_spans'] = boundaries['merged_spans']
    row['spurious_spans'] = boundaries['spurious_spans']
    row['start_alignment_mean'] = start_alignment['mean'] if start_alignment else None
    row['start_alignment_median'] = start_alignment['median'] if start_alignment else None
    row['count_gate_covered_fraction'] = count_gate['covered']['fraction']
    row['count_gate_unmerged_fraction'] = count_gate['unmerged']['fraction']
    for tolerance in SWEEP_TOLERANCES:
        curve = tolerance_curves[str(tolerance)]
        for metric in SCORED_METRICS:
            row[f'{metric}_{tolerance}'] = curve[metric]
    # Precision denominator (candidates pooled per rally over overlapping spans);
    # equal across tolerances, so any band's curve carries it.
    row['total_candidates'] = tolerance_curves[str(SWEEP_TOLERANCES[0])]['candidates']
    return row


def _serialise_value(value: object) -> object:
    """CSV cell: None -> blank, float -> 6 dp, everything else unchanged."""
    if value is None:
        return ''
    if isinstance(value, float):
        return round(value, 6)
    return value


def _serialise_row(row: dict) -> dict:
    """Row ready for csv.DictWriter, in ROW_COLUMNS order."""
    return {column: _serialise_value(row[column]) for column in ROW_COLUMNS}


# ---------------------------------------------------------------------------
# Winner selection (deterministic)
# ---------------------------------------------------------------------------
def _changed_from_defaults(row: dict) -> int:
    """How many of the eight params differ from the shipped defaults."""
    changed = 0
    for field in PARAM_COLUMNS:
        if row[field] != getattr(SHIPPED_DEFAULTS, field):
            changed += 1
    return changed


def _param_tuple(row: dict) -> tuple:
    """The eight params as a tuple; the final, order-independent tie-break."""
    return tuple(row[field] for field in PARAM_COLUMNS)


def boundary_sort_key_as_built(row: dict) -> tuple:
    """The original boundary sort key (coverage-first); orders ``boundary_sweep.csv`` only.

    Report-only since the 2026-07-08 ruling (selection moved to
    ``select_boundary_winner``): maximise covered_fraction, then fewer split,
    spurious_spans, merged_spans, then closest to shipped defaults, param tuple
    keeping the order total. Kept so ``build_boundary_crowns`` can show what
    coverage-first would have picked.
    """
    covered_fraction = row['covered_fraction']
    covered_fraction = covered_fraction if covered_fraction is not None else -1.0
    return (
        -covered_fraction,
        row['split'],
        row['spurious_spans'],
        row['merged_spans'],
        _changed_from_defaults(row),
        _param_tuple(row),
    )


def _merge_penalised_key(row: dict) -> tuple:
    """Boundary key that penalises merges first (Ariel's 2026-07-08 ruling).

    Don't buy coverage with glue: rank fewest merged_spans first, then split,
    spurious_spans, closeness to the shipped defaults, then the param tuple for a
    total order. Coverage is not in the key: it gates eligibility instead (see
    ``select_boundary_winner``), so a small coverage loss can't be out-shouted by
    a large one, but merges decide among the configs that clear the gate.
    """
    return (
        row['merged_spans'],
        row['split'],
        row['spurious_spans'],
        _changed_from_defaults(row),
        _param_tuple(row),
    )


def _start_alignment_key(row: dict) -> tuple:
    """Boundary key that penalises start-offset magnitude first, then merges.

    Rank smallest abs(start_alignment_median) first (a span that opens tight to
    the first stroke), then the merge-penalised tail. A None median means no
    covered rally gave a measurable offset, so it can't be judged on alignment:
    it sorts worst rather than crashing.
    """
    median = row['start_alignment_median']
    abs_median = abs(median) if median is not None else float('inf')
    return (
        abs_median,
        row['merged_spans'],
        row['split'],
        row['spurious_spans'],
        _changed_from_defaults(row),
        _param_tuple(row),
    )


def _coverage_eligible(rows: list[dict]) -> list[dict]:
    """Grid rows whose covered-rally count is within 2 of the best (ref excluded).

    Eligibility is an allowance, not a strict tie: a config a rally or two shy of
    the coverage maximum still competes, so the merge/alignment keys can prefer it
    over a higher-coverage config that only got there by gluing rallies. Exactly
    equal coverage would let those keys decide between far fewer configs.
    """
    grid_rows = [row for row in rows if row['label'] == LABEL_GRID]
    if not grid_rows:
        raise ValueError('no grid rows to select a boundary winner from')
    best_covered = max(row['covered'] for row in grid_rows)
    return [row for row in grid_rows if best_covered - row['covered'] <= 2]


def select_boundary_winner(rows: list[dict]) -> dict:
    """The frozen boundary winner: merge-penalised within the coverage allowance.

    Ariel's 2026-07-08 ruling. Eligible rows are within 2 covered rallies of the
    best; the winner is the eligible config with fewest merges (``_merge_penalised_key``
    tail settles the rest). Needs the whole row set for the eligibility window, so
    it can't reuse the generic keyed ``select_winner`` (the contact phase still does).
    """
    return min(_coverage_eligible(rows), key=_merge_penalised_key)


def select_start_alignment_winner(rows: list[dict]) -> dict:
    """Start-alignment-penalised winner, same coverage allowance as the merge pick.

    Candidate boundary key for the ``boundary_crowns.csv`` data package, not a
    live winner: eligible rows are within 2 covered rallies of the best, then the
    tightest abs(start_alignment_median) wins (``_start_alignment_key``).
    """
    return min(_coverage_eligible(rows), key=_start_alignment_key)


def contact_sort_key(row: dict) -> tuple:
    """Contact sort key; ascending puts the best config first.

    Ruled by Ariel 2026-07-08: recall-first at +/-5, because recall is
    unrecoverable downstream while precision is recovered downstream per context
    (wrist filter first, then the featurised verifier if needed). So maximise
    +/-5 recall, then +/-5 F1, then the unmerged count-gate pass rate, then
    closeness to shipped defaults, then the param tuple for a total order. These
    thresholds should live in the config module/dataclass when the stage-8 params
    land.

    The crown this key picks is PROVISIONAL (Ariel 2026-07-08, second ruling):
    pure recall-first always favours the loosest turn gate on the grid, so the
    final contact key is ruled on data, off ``contact_frontier.csv``.
    """
    recall_5 = row['recall_5'] if row['recall_5'] is not None else -1.0
    f1_5 = row['f1_5'] if row['f1_5'] is not None else -1.0
    gate = row['count_gate_unmerged_fraction']
    gate = gate if gate is not None else -1.0
    return (
        -recall_5,
        -f1_5,
        -gate,
        _changed_from_defaults(row),
        _param_tuple(row),
    )


def select_winner(rows: list[dict], sort_key: Callable[[dict], tuple]) -> dict:
    """Best grid config by ``sort_key`` (the shipped-defaults reference excluded).

    Only 'grid' rows compete: the shipped-defaults row is a labelled reference,
    and in the contact phase it carries the shipped boundary params rather than
    the frozen winner's, so letting it win would break the freeze.
    """
    grid_rows = [row for row in rows if row['label'] == LABEL_GRID]
    if not grid_rows:
        raise ValueError('no grid rows to select a winner from')
    return min(grid_rows, key=sort_key)


# ---------------------------------------------------------------------------
# Data packages for pending key rulings (crowns and frontiers; not winners)
# ---------------------------------------------------------------------------
CROWN_KEY_COLUMN = 'crown_key'
CROWN_COLUMNS = [CROWN_KEY_COLUMN] + ROW_COLUMNS


def build_boundary_crowns(rows: list[dict]) -> list[dict]:
    """One row per candidate boundary key, plus the coverage-vs-merges frontier.

    The data package for Ariel's pending boundary-key ruling: it lays the winner
    under each key side by side so the choice reads off one file. Each returned
    dict is a full sweep row with a leading ``crown_key`` label.

    Crowns: ``as_built`` (coverage-first, the retired key), ``merge_penalised``
    (the live pick), ``start_alignment_penalised``. Frontier: for each distinct
    covered level (descending), the config with fewest merges at that level
    (merge-penalised tail breaks ties), labelled ``frontier_cov_<covered>``, so a
    reader sees the merge cost of holding out for each extra covered rally.
    """
    crowns: list[dict] = []
    for crown_key, winner in (
        ('as_built', select_winner(rows, boundary_sort_key_as_built)),
        ('merge_penalised', select_boundary_winner(rows)),
        ('start_alignment_penalised', select_start_alignment_winner(rows)),
    ):
        crowns.append({CROWN_KEY_COLUMN: crown_key, **winner})

    grid_rows = [row for row in rows if row['label'] == LABEL_GRID]
    for covered in sorted({row['covered'] for row in grid_rows}, reverse=True):
        at_level = [row for row in grid_rows if row['covered'] == covered]
        winner = min(at_level, key=_merge_penalised_key)
        crowns.append({CROWN_KEY_COLUMN: f'frontier_cov_{covered}', **winner})
    return crowns


def contact_frontier(rows: list[dict]) -> list[dict]:
    """Grid configs on the (recall_5, precision_5) Pareto frontier, best recall first.

    The data package for Ariel's pending contact-key ruling (``contact_sort_key``
    is provisional): a config is dominated when some other grid config is at least
    as good on both recall_5 and precision_5 AND strictly better on at least one;
    the non-dominated configs are the frontier. Exact duplicates on both axes
    don't dominate each other, so both stay. Rows with no +/-5 candidates
    (recall_5 or precision_5 None) can't sit on the frontier, so they drop out
    before the test. Shipped-defaults reference excluded, same as winner selection.
    """
    scored = [
        row for row in rows
        if row['label'] == LABEL_GRID
        and row['recall_5'] is not None and row['precision_5'] is not None
    ]
    frontier = []
    for row in scored:
        dominated = any(
            other['recall_5'] >= row['recall_5'] and other['precision_5'] >= row['precision_5']
            and (other['recall_5'] > row['recall_5'] or other['precision_5'] > row['precision_5'])
            for other in scored
        )
        if not dominated:
            frontier.append(row)
    frontier.sort(key=lambda row: (-row['recall_5'], -row['precision_5'], _param_tuple(row)))
    return frontier


# ---------------------------------------------------------------------------
# Worker: patch, segment, score, flatten
# ---------------------------------------------------------------------------
class SweepCtx(NamedTuple):
    """Read-only inputs every task shares, shipped once per worker via the pool
    initializer (same pattern as wrist_contact_separation.WorkerCtx)."""

    track: np.ndarray  # (t, 3) [x_norm, y_norm, visibility] whole-video track
    gt_rallies: list  # list[GtRally] for the scored video


_CTX: SweepCtx | None = None


def _init_worker(ctx: SweepCtx, nan_smoothing: bool) -> None:
    """Pool initializer: stash the shared context and, when asked, install the
    NaN-ignoring contact smoothing in THIS process.

    Runs once per pool worker (via ``initializer``) and once directly on the serial
    path, so it is the single seam that reaches every process that will call
    ``segment_video``. Installing the smoothing patch here rather than in the parent
    means it doesn't lean on the fork start method copying a parent-side patch: it
    works the same if the pool ever ran under spawn (as ``bric.preprocessing`` forces),
    and it also covers the serial path, which never forks.
    """
    global _CTX
    _CTX = ctx
    if nan_smoothing:
        install_nan_smoothing()


def _score_config(task: SweepTask) -> dict:
    """Patch the thresholds, segment the cached track, score, and flatten a row."""
    ctx = _CTX
    assert ctx is not None, 'worker context not initialised'
    _patch_stage8(task.params)
    spans, contacts = stage8_module.segment_video(ctx.track)  # positions None
    metrics = score_stage8(spans, contacts, ctx.gt_rallies, tolerances=SWEEP_TOLERANCES)
    return flatten_row(task.label, task.params, len(spans), metrics)


def run_phase(tasks: list[SweepTask], ctx: SweepCtx, workers: int, nan_smoothing: bool) -> list[dict]:
    """Score every task, serial when workers<=1 else across a pool.

    Rows return in arbitrary order under the pool, which is fine: CSV writing
    sorts by the phase key and winner selection is a keyed min, both
    order-independent. ``nan_smoothing`` is threaded to every worker (and the serial
    path) through the initializer so the smoothing patch is installed in whichever
    process actually segments the track.
    """
    rows: list[dict] = []
    if workers <= 1:
        _init_worker(ctx, nan_smoothing)
        for task in tasks:
            rows.append(_score_config(task))
        return rows
    with Pool(processes=workers, initializer=_init_worker, initargs=(ctx, nan_smoothing)) as pool:
        for row in pool.imap_unordered(_score_config, tasks, chunksize=8):
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_sweep_csv(path: Path, rows: list[dict], sort_key: Callable[[dict], tuple]) -> None:
    """Write the phase rows to ``path``, sorted best-first by ``sort_key``."""
    ordered = sorted(rows, key=sort_key)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_COLUMNS)
        writer.writeheader()
        for row in ordered:
            writer.writerow(_serialise_row(row))


def write_crowns_csv(path: Path, rows: list[dict]) -> None:
    """Write ``boundary_crowns.csv``: the crown rows with a leading crown_key."""
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=CROWN_COLUMNS)
        writer.writeheader()
        for crown in build_boundary_crowns(rows):
            writer.writerow({CROWN_KEY_COLUMN: crown[CROWN_KEY_COLUMN], **_serialise_row(crown)})


def write_contact_frontier_csv(path: Path, rows: list[dict]) -> None:
    """Write ``contact_frontier.csv``: the recall_5/precision_5 Pareto frontier."""
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_COLUMNS)
        writer.writeheader()
        for row in contact_frontier(rows):
            writer.writerow(_serialise_row(row))


def _params_of(row: dict) -> dict:
    """The eight param columns of a row (winner.json params half)."""
    return {field: row[field] for field in PARAM_COLUMNS}


def _metrics_of(row: dict) -> dict:
    """The metric columns of a row (winner.json metrics half)."""
    return {column: row[column] for column in METRIC_COLUMNS}


def load_boundary_winner_json(path: Path) -> tuple[Stage8Params, dict]:
    """Reconstruct a boundary winner from a prior winner.json (to skip phase 1).

    :return: ``(params, loaded_json)`` so phase 2 can freeze the boundary params
        and the final winner.json can echo the loaded boundary half.
    """
    loaded = json.loads(path.read_text(encoding='utf-8'))
    params_dict = loaded['boundary']
    params = Stage8Params(**{field: params_dict[field] for field in PARAM_COLUMNS})
    return params, loaded


def _find_defaults_row(rows: list[dict]) -> dict | None:
    """The labelled shipped-defaults reference row, if present."""
    for row in rows:
        if row['label'] == LABEL_SHIPPED:
            return row
    return None


def _summarise_row(row: dict) -> str:
    """One compact metric line: coverage, count-gate, and +/-2 / +/-5 credit."""
    def fmt(value: object) -> str:
        return 'n/a' if value is None else f'{value:.3f}'

    return (
        f"covered {fmt(row['covered_fraction'])}  "
        f"count-gate(unmerged) {fmt(row['count_gate_unmerged_fraction'])}  "
        f"@2 P/R/F1 {fmt(row['precision_2'])}/{fmt(row['recall_2'])}/{fmt(row['f1_2'])}  "
        f"@5 P/R/F1 {fmt(row['precision_5'])}/{fmt(row['recall_5'])}/{fmt(row['f1_5'])}"
    )


def print_summary(
    boundary_winner: dict | None, boundary_defaults: dict | None,
    contact_winner: dict | None, contact_defaults: dict | None,
) -> None:
    """Terse end-of-run digest: each phase winner, with the shipped row alongside."""
    print('\n' + '=' * 70)
    print('SWEEP SUMMARY')
    print('=' * 70)
    for phase_name, winner, defaults in (
        ('Boundary', boundary_winner, boundary_defaults),
        ('Contact', contact_winner, contact_defaults),
    ):
        if winner is None:
            continue
        print(f'\n{phase_name} winner:')
        print(f'  params:  {_params_of(winner)}')
        print(f'  metrics: {_summarise_row(winner)}')
        if defaults is not None:
            print(f'  shipped: {_summarise_row(defaults)}')


# ---------------------------------------------------------------------------
# Optional replay-mask track transform (applied once at load, before the sweep)
# ---------------------------------------------------------------------------
def apply_replay_mask(track: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Freeze replay/off-rally frames to the last live position so they read as rest.

    Returns a new ``(t, 3)`` array; ``track`` is not mutated. For each contiguous
    True run in ``mask``, the run's xy (columns 0-1) is set to the xy of the last
    frame BEFORE the run, and its visibility (column 2) forced to 1. A run that
    starts at frame 0 has no earlier frame, so it takes the xy of the first frame
    AFTER it instead.

    Why: stage 8 reads invisible or NaN-speed frames as not-rest, so replay
    closeups hold rally regions open. Freezing the position makes masked footage
    read as sustained sub-REST_SPEED rest, per the mask-before-segmentation design
    (masked frames count as rest). Forcing visibility avoids the NaN-speed path
    reopening the region.

    Fail loud on a length mismatch, and on an all-True mask (nothing live to anchor
    a frozen position to, and a fully-masked video is senseless). An all-False mask
    has no True runs, so the untouched copy returns bit-identical by construction.

    :param track: ``(t, 3)`` ``[x_norm, y_norm, visibility]`` whole-video track.
    :param mask: ``(t,)`` bool, True on replay/off-rally frames (stage-9
        ``1_replay.npy`` convention).
    :return: a new ``(t, 3)`` track with masked frames frozen to rest.
    """
    if len(mask) != len(track):
        raise ValueError(f'mask length {len(mask)} != track length {len(track)}')
    if mask.all():
        raise ValueError('mask is all True: no live frame to anchor a frozen position to')

    frozen = track.copy()
    for start, end in stage8_module.true_runs(mask):
        # start-1 is the last live frame before the run; a run at frame 0 has none,
        # so anchor to end (the first live frame after it). The not-all-True guard
        # above guarantees that frame exists.
        anchor = start - 1 if start > 0 else end
        frozen[start:end, :2] = track[anchor, :2]
        frozen[start:end, 2] = 1
    return frozen


# ---------------------------------------------------------------------------
# Optional NaN-smoothing patch for contact detection (installed once per worker)
# ---------------------------------------------------------------------------
# Stage 8 smooths (x, y) with a centred rolling mean before reading contacts off
# velocity reversals. Invisible frames carry zero-filled (0, 0) xy, so a visibility
# gap inside a rally drags the smoothed track toward the image corner on both edges
# of the gap and can seed a false direction-change contact there. This arm swaps the
# smoothing for a NaN-ignoring variant: invisible frames become NaN and drop out of
# each window rather than averaging in as (0, 0). Same philosophy as the threshold
# global-patching above: committed stage-8 code stays untouched; the swap lives here
# and only runs in a process that installs it (gated by --nan-smoothing).
#
# _rest_mask also reads _rolling_mean, but on a clean 0/1 visibility array with no
# NaN, where the NaN-ignoring variant reproduces the stock mean exactly, so patching
# the global is a no-op for rally boundaries and only changes contact smoothing.

# The stock detect_contacts, captured at import before any install so the wrapper
# below can defer to it (and a double install can't stack the wrapper on itself).
_STOCK_DETECT_CONTACTS = stage8_module.detect_contacts


def nan_rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling mean that IGNORES NaN; drop-in for stage-8 ``_rolling_mean``.

    Same centred, shrinking-at-the-edges arithmetic as the stock ``_rolling_mean``
    (a mode='same' convolution, sums over counts), but NaN entries drop out of both
    the running sum and the sample count rather than poisoning the whole window. A
    window whose samples are all NaN stays NaN (0/0), matching the stock's all-NaN
    edge behaviour and what the downstream visibility gate wants.

    On a NaN-free input this reproduces stock ``_rolling_mean`` bit-for-bit: the
    valid mask is all True, so ``filled`` equals ``values`` and the counts
    convolution equals the stock ``np.ones_like`` one, leaving identical sums and
    counts. That exact match is why an invisible-free track scores the same with the
    patch installed.

    :param values: ``(t,)`` values, may contain NaN.
    :param window: window width in frames.
    :return: ``(t,)`` centred mean; NaN only where a whole window is NaN.
    """
    kernel = np.ones(window)
    valid = ~np.isnan(values)
    filled = np.where(valid, values, 0.0)  # NaN -> 0 so it adds nothing to the sum
    counts = np.convolve(valid.astype(float), kernel, mode='same')  # non-NaN samples per position
    sums = np.convolve(filled, kernel, mode='same')
    with np.errstate(invalid='ignore', divide='ignore'):
        return sums / counts  # all-NaN window -> 0/0 -> NaN, by design


def nan_smoothing_detect_contacts(track: np.ndarray, start: int, end: int) -> list[int]:
    """Drop-in for stage-8 ``detect_contacts`` that hides invisible frames from the smooth.

    Copies the rally span, sets invisible frames' xy to NaN, and defers to the stock
    contact logic with ``_rolling_mean`` patched to ``nan_rolling_mean`` (the two are
    installed together, see ``install_nan_smoothing``), so the gap frames drop out of
    every smoothing window instead of pulling it toward (0, 0). The visibility column
    is left untouched, so the stock ``around_visible`` gate still drops any junction
    that touches an invisible frame; masking only changes what the surviving visible
    frames smooth to.

    Only the span (not the whole track) is copied, so the hot sweep path stays
    O(span) per call: the stock helper is handed a 0-based span and the whole-video
    offset is re-added to its result here.

    :param track: ``(t, 3)`` whole-video track (same contract as stock).
    :param start: rally span start frame (inclusive).
    :param end: rally span end frame (exclusive).
    :return: contact frames in whole-video indices, ascending (stock's contract).
    """
    span = track[start:end].astype(float, copy=True)
    invisible = span[:, 2] != 1
    span[invisible, :2] = np.nan
    local_contacts = _STOCK_DETECT_CONTACTS(span, 0, len(span))
    return [start + local for local in local_contacts]


def install_nan_smoothing() -> None:
    """Swap stage-8's contact smoothing for the NaN-ignoring variant, in this process.

    Patches two ``stage8_module`` globals: ``_rolling_mean`` (NaN-ignoring drop-in)
    and ``detect_contacts`` (masks invisible xy to NaN, then defers to the stock
    logic). ``segment_video`` and the stock ``detect_contacts`` both resolve those
    names through the module dict at call time, so the swap takes hold without
    touching committed code. Idempotent: the wrapper always defers to the import-time
    stock reference, so re-installing can't stack it on itself.
    """
    stage8_module._rolling_mean = nan_rolling_mean
    stage8_module.detect_contacts = nan_smoothing_detect_contacts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Sweep stage-8 thresholds over one cached shuttle track against ShuttleSet GT.',
    )
    parser.add_argument('--track-npy', type=Path, required=True,
                        help='(t, 3) whole-video shuttle track npy; its index is the source frame')
    parser.add_argument('--vid', type=int, required=True,
                        help='ShuttleSet video id to score against (filters shots_master)')
    parser.add_argument('--mask-npy', type=Path, default=None,
                        help='optional (frames,) bool npy, True on replay/off-rally frames '
                             '(stage-9 1_replay.npy convention); freezes those frames to the last '
                             'live position with visibility 1 so they read as rest. Applied once at '
                             'track load, so it affects BOTH sweep phases')
    parser.add_argument('--nan-smoothing', action='store_true',
                        help='exclude invisible frames from the contact-detection smoothing: set '
                             'their xy to NaN and take a NaN-ignoring rolling mean rather than '
                             'averaging the zero-filled (0, 0) in, which otherwise drags the smoothed '
                             'track toward the image corner near a visibility gap. Composes with '
                             '--mask-npy; installed per worker, committed stage-8 code untouched')
    parser.add_argument('--shots-master', type=Path, default=DEFAULT_SHOTS_MASTER,
                        help='ShuttleSet shots_master.csv (default: the in-repo training annotations)')
    parser.add_argument('--out-dir', type=Path, required=True,
                        help='writes boundary_sweep.csv, boundary_crowns.csv, contact_sweep.csv, '
                             'contact_frontier.csv, winner.json')
    parser.add_argument('--workers', type=int, default=os.cpu_count() or 1,
                        help='multiprocessing pool size (default: all cores)')
    parser.add_argument('--phase', choices=('both', 'boundary', 'contact'), default='both',
                        help='which phase(s) to run (default both)')
    parser.add_argument('--boundary-winner-json', type=Path, default=None,
                        help='a prior winner.json; supply to skip phase 1 and freeze its boundary params')
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.boundary_winner_json is not None and args.phase == 'boundary':
        parser.error('--boundary-winner-json skips phase 1, but --phase boundary runs only phase 1; nothing to do')

    run_boundary = args.phase in ('both', 'boundary') and args.boundary_winner_json is None
    run_contact = args.phase in ('both', 'contact')
    if run_contact and not run_boundary and args.boundary_winner_json is None:
        parser.error('--phase contact needs a boundary winner; pass --boundary-winner-json')

    args.out_dir.mkdir(parents=True, exist_ok=True)

    track = np.load(args.track_npy)
    if track.ndim != 2 or track.shape[1] != 3:
        raise ValueError(f'track must be (t, 3) [x_norm, y_norm, visibility]; got shape {track.shape}')
    # Optional: freeze replay/off-rally frames to rest once here, before the worker
    # context is built, so both sweep phases score the same masked track. Absent the
    # flag the track flows through untouched (no transform call at all).
    if args.mask_npy is not None:
        mask = np.load(args.mask_npy)
        track = apply_replay_mask(track, mask)
        print(f'Applied replay mask {args.mask_npy}: {int(mask.sum())} of {len(mask)} frames frozen to rest')
    if args.nan_smoothing:
        print('NaN-smoothing on: invisible frames excluded from the contact smoothing (installed per worker)')
    shots_master = pd.read_csv(args.shots_master)
    gt_rallies = load_gt_rallies(shots_master, args.vid)
    ctx = SweepCtx(track=track, gt_rallies=gt_rallies)

    print(f'Track {args.track_npy} ({len(track)} frames), vid {args.vid}, {len(gt_rallies)} GT rallies')
    print(f'Workers: {args.workers}')

    winner_json: dict = {}
    boundary_winner_row: dict | None = None
    boundary_defaults_row: dict | None = None
    boundary_winner_params: Stage8Params | None = None

    if run_boundary:
        tasks = build_boundary_tasks()
        print(f'\nBoundary phase: {len(tasks)} configs')
        boundary_rows = run_phase(tasks, ctx, args.workers, args.nan_smoothing)
        write_sweep_csv(args.out_dir / 'boundary_sweep.csv', boundary_rows, boundary_sort_key_as_built)
        write_crowns_csv(args.out_dir / 'boundary_crowns.csv', boundary_rows)
        boundary_winner_row = select_boundary_winner(boundary_rows)
        boundary_defaults_row = _find_defaults_row(boundary_rows)
        boundary_winner_params = Stage8Params(**_params_of(boundary_winner_row))
        winner_json['boundary'] = _params_of(boundary_winner_row)
        winner_json['boundary_metrics'] = _metrics_of(boundary_winner_row)
    elif args.boundary_winner_json is not None:
        boundary_winner_params, loaded = load_boundary_winner_json(args.boundary_winner_json)
        winner_json['boundary'] = loaded['boundary']
        if 'boundary_metrics' in loaded:
            winner_json['boundary_metrics'] = loaded['boundary_metrics']
        print(f'\nLoaded boundary winner from {args.boundary_winner_json}: {winner_json["boundary"]}')

    contact_winner_row: dict | None = None
    contact_defaults_row: dict | None = None
    if run_contact:
        assert boundary_winner_params is not None  # guaranteed by the phase guards above
        tasks = build_contact_tasks(boundary_winner_params)
        print(f'\nContact phase: {len(tasks)} configs at the frozen boundary winner')
        contact_rows = run_phase(tasks, ctx, args.workers, args.nan_smoothing)
        write_sweep_csv(args.out_dir / 'contact_sweep.csv', contact_rows, contact_sort_key)
        write_contact_frontier_csv(args.out_dir / 'contact_frontier.csv', contact_rows)
        contact_winner_row = select_winner(contact_rows, contact_sort_key)
        contact_defaults_row = _find_defaults_row(contact_rows)
        winner_json['contact'] = _params_of(contact_winner_row)
        winner_json['contact_metrics'] = _metrics_of(contact_winner_row)

    winner_path = args.out_dir / 'winner.json'
    winner_path.write_text(json.dumps(winner_json, indent=2), encoding='utf-8')
    print(f'\nWrote winner.json to {winner_path}')

    print_summary(boundary_winner_row, boundary_defaults_row, contact_winner_row, contact_defaults_row)


if __name__ == '__main__':
    main()
