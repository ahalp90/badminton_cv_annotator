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
        [--mask-npy /scratch/.../1_replay.npy] [--nan-smoothing] \\
        [--gap-state DEMOTION_BOUND] [--quiet-start W] \\
        [--reentry-guard {two-sided,reentry-only} --reentry-buffer Y]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from enum import StrEnum
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

# Per-tolerance precision over the PHYSICAL candidate set (score_contacts'
# 'precision_raw' curve), one column per band. Merge-structure-independent, unlike
# the pooled precision_* above; recall is already raw so it needs no twin. Appended
# after the stock columns so every pre-existing column keeps its value byte-for-byte
# (the reproduction gate compares the shared columns against the committed sweep).
RAW_PRECISION_COLUMNS = [f'precision_raw_{tolerance}' for tolerance in SWEEP_TOLERANCES]

METRIC_COLUMNS = [
    'n_spans',
    'covered', 'covered_fraction', 'split', 'missed', 'merged_spans', 'spurious_spans',
    'start_alignment_mean', 'start_alignment_median',
    'count_gate_covered_fraction', 'count_gate_unmerged_fraction',
    *TOLERANCE_COLUMNS,
    'total_candidates',
    *RAW_PRECISION_COLUMNS,
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
    raw_precision_curves = contacts['precision_raw']  # {str(tol): {precision_raw, matched, candidates}}

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
    # Raw per-candidate precision, one column per band (denominator is the physical
    # candidate set, so it does not swell with merge structure the way total_candidates
    # does; only the matched numerator moves with tolerance).
    for tolerance in SWEEP_TOLERANCES:
        row[f'precision_raw_{tolerance}'] = raw_precision_curves[str(tolerance)]['precision_raw']
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


def _init_worker(
    ctx: SweepCtx,
    nan_smoothing: bool,
    gap_state_demotion_bound: int | None = None,
    quiet_start_window: int | None = None,
    reentry_guard_variant: ReentryGuardVariant | None = None,
    reentry_guard_buffer: float | None = None,
    serve_start_mode: ServeStartMode | None = None,
    serve_start_threshold: float | None = None,
    serve_start_dist: np.ndarray | None = None,
    serve_start_wideshot: WideshotInputs | None = None,
    serve_start_close: ServeStartClose | None = None,
) -> None:
    """Pool initializer: stash the shared context and install any requested arms in
    THIS process.

    Runs once per pool worker (via ``initializer``) and once directly on the serial
    path, so it is the single seam that reaches every process that will call
    ``segment_video``. Installing the patches here rather than in the parent means they
    don't lean on the fork start method copying a parent-side patch: it works the same
    if the pool ever ran under spawn (as ``bric.preprocessing`` forces), and it also
    covers the serial path, which never forks. Each arm is off when its argument is the
    off value (False / None), leaving the stock function bound.

    :param ctx: shared read-only track + GT for every task.
    :param nan_smoothing: install the NaN-ignoring contact smoothing when True.
    :param gap_state_demotion_bound: install the gap-state _rest_mask at this
        demotion_bound when not None.
    :param quiet_start_window: install the quiet-start _find_rally_spans at this
        window W when not None.
    :param reentry_guard_variant: gap-state re-entry guard variant, or None for the
        committed entry-only classification. Needs gap_state_demotion_bound set too.
    :param reentry_guard_buffer: the guard's near-top y buffer (paired with the variant).
    :param serve_start_mode: install the serve-start _find_rally_spans in this mode when
        not None. Owns _find_rally_spans, so it is mutually exclusive with quiet-start
        (quiet-start wins if both are set; the CLI forbids the pair).
    :param serve_start_threshold: the serve-start gate distance (paired with the mode).
    :param serve_start_dist: the precomputed (t,) nearest-court-scale-centre distance array
        the gate reads (paired with the mode).
    :param serve_start_wideshot: the precomputed wide-shot gate inputs, or None to leave
        the wide-shot refinement off (the default; serve-start behaviour unchanged).
    :param serve_start_close: the serve-start split placement, or None for the single-span
        arm (the default; each qualifying region opens one span at its first qualifying burst).
    """
    global _CTX
    _CTX = ctx
    # Off arms restore the stock bindings explicitly, so this initialiser means
    # "exactly this arm state". The serial path runs in the caller's process, where
    # an earlier install would otherwise linger into a stock run.
    if nan_smoothing:
        install_nan_smoothing()
    else:
        stage8_module._rolling_mean = _STOCK_ROLLING_MEAN
        stage8_module.detect_contacts = _STOCK_DETECT_CONTACTS
    if gap_state_demotion_bound is not None:
        install_gap_state(gap_state_demotion_bound, reentry_guard_variant, reentry_guard_buffer)
    else:
        stage8_module._rest_mask = _STOCK_REST_MASK
    # quiet-start and serve-start both own _find_rally_spans, so at most one installs;
    # off on both restores stock. The non-serve-start paths also drop the serve-start
    # arrays: restoring the binding alone would leave a re-inited worker holding a stale
    # distance/wideshot copy (a few MB) nothing reads any more.
    if quiet_start_window is not None:
        _clear_serve_start_state()
        install_quiet_start(quiet_start_window)
    elif serve_start_mode is not None:
        assert serve_start_dist is not None and serve_start_threshold is not None, \
            'serve-start mode needs both a threshold and a distance array'
        install_serve_start(serve_start_dist, serve_start_threshold, serve_start_mode,
                            serve_start_wideshot, serve_start_close)
    else:
        _clear_serve_start_state()
        stage8_module._find_rally_spans = _STOCK_FIND_RALLY_SPANS


def _score_config(task: SweepTask) -> dict:
    """Patch the thresholds, segment the cached track, score, and flatten a row."""
    ctx = _CTX
    assert ctx is not None, 'worker context not initialised'
    _patch_stage8(task.params)
    spans, contacts = stage8_module.segment_video(ctx.track)  # positions None
    metrics = score_stage8(spans, contacts, ctx.gt_rallies, tolerances=SWEEP_TOLERANCES)
    return flatten_row(task.label, task.params, len(spans), metrics)


def run_phase(
    tasks: list[SweepTask],
    ctx: SweepCtx,
    workers: int,
    nan_smoothing: bool,
    gap_state_demotion_bound: int | None = None,
    quiet_start_window: int | None = None,
    reentry_guard_variant: ReentryGuardVariant | None = None,
    reentry_guard_buffer: float | None = None,
    serve_start_mode: ServeStartMode | None = None,
    serve_start_threshold: float | None = None,
    serve_start_dist: np.ndarray | None = None,
    serve_start_wideshot: WideshotInputs | None = None,
    serve_start_close: ServeStartClose | None = None,
) -> list[dict]:
    """Score every task, serial when workers<=1 else across a pool.

    Rows return in arbitrary order under the pool, which is fine: CSV writing sorts by
    the phase key and winner selection is a keyed min, both order-independent. The arm
    settings are threaded to every worker (and the serial path) through the initializer
    so the patches install in whichever process actually segments the track. A whole
    sweep runs at one fixed arm setting; sweeping the arm parameter itself is the
    measurement runner's job, which installs per config instead.
    """
    initargs = (ctx, nan_smoothing, gap_state_demotion_bound, quiet_start_window,
                reentry_guard_variant, reentry_guard_buffer,
                serve_start_mode, serve_start_threshold, serve_start_dist,
                serve_start_wideshot, serve_start_close)
    rows: list[dict] = []
    if workers <= 1:
        _init_worker(*initargs)
        for task in tasks:
            rows.append(_score_config(task))
        return rows
    with Pool(processes=workers, initializer=_init_worker, initargs=initargs) as pool:
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

# The stock detect_contacts and _rolling_mean, captured at import before any install
# so the wrapper below can defer to the former (a double install can't stack the
# wrapper on itself) and _init_worker can restore both when the arm is off.
_STOCK_DETECT_CONTACTS = stage8_module.detect_contacts
_STOCK_ROLLING_MEAN = stage8_module._rolling_mean


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
# Optional gap-state _rest_mask (installed once per worker; --gap-state)
# ---------------------------------------------------------------------------
# Stock rest reads an invisibility gap as rest via ``mostly_untracked``, which cuts
# a rally whenever the shuttle drops off the tracker mid-point (a lob/clear leaving
# the top of frame, a between-rally dead patch). This arm classifies each gap by how
# the shuttle entered it and only lets DEAD gaps read as rest: a high_shot_oob gap (last seen
# climbing off the top) stays not-rest so the rally holds through the flight, and a
# short BLIP gap gets no say at all and rides the plain slow test. Same philosophy as
# the smoothing patch above: committed stage-8 code stays untouched; the swap lives
# here and only runs in a process that installs it (gated by --gap-state).
#
# The classification numbers are the arm's fixed reading of the spec, not swept:
BLIP_MAX_FRAMES = 10  # a non-high_shot_oob gap this short is a tracker blip, not a real rest
HIGH_SHOT_OOB_LOOKBACK_FRAMES = 5  # visible frames before a gap that fix the entry velocity
HIGH_SHOT_OOB_MIN_VISIBLE_FRAMES = 2  # fewer than this before a gap gives no velocity -> not high_shot_oob
HIGH_SHOT_OOB_EXTRAP_FRAMES = 10  # frames to extrapolate the entry velocity past the last sighting

# Re-entry guard (--reentry-guard): a retrospective second condition on a high_shot_oob
# gap, read off where the shuttle REAPPEARS rather than where it left. The entry
# extrapolation already selects top exits, but 204 between-rally false fires also
# arc off the top; the audit found they reappear scattered down the frame while
# real in-rally high shots reappear near the top and descend. So the guard demands
# the reappearance be near the top AND descending, which keeps ~75% of real high
# shots and drops ~85% of the false fires (scoped 2026-07-08). Retrospective: the
# whole gap must end before it can classify, which is fine for offline stage 8.
REENTRY_LOOKAHEAD_FRAMES = 5  # visible frames after reappearance that fix the descent
REENTRY_MIN_VISIBLE_FRAMES = 2  # fewer than this after a gap gives no velocity -> guard fails


class ReentryGuardVariant(StrEnum):
    """Which sides of a high_shot_oob gap the buffer test applies to.

    REENTRY_ONLY tests the reappearance side alone (position near the top plus a
    descent). TWO_SIDED adds the entry side (last-visible y before the gap also
    within the buffer). The audit found the entry-side half adds little at a loose
    buffer (the extrapolation already selects top exits) but bites at a tight one;
    Ariel ruled the cutoff applies to both sides, so the sweep runs both and the
    rally-level metrics decide.
    """

    TWO_SIDED = 'two-sided'
    REENTRY_ONLY = 'reentry-only'


# The stock _rest_mask / _find_rally_spans, captured at import before any install so
# the "flag off leaves stock" tests can assert identity and a restore has a target.
_STOCK_REST_MASK = stage8_module._rest_mask
_STOCK_FIND_RALLY_SPANS = stage8_module._find_rally_spans

# demotion_bound (--gap-state) and W (--quiet-start) are swept, so they can't be baked
# into the installed function. They live as per-process state the install sets and the
# patched function reads at call time, the same seam _patch_stage8 uses for the eight
# thresholds; one task runs at a time per process, so this cannot leak across tasks.
_GAP_STATE_DEMOTION_BOUND: int | None = None
_QUIET_START_WINDOW: int | None = None

# Re-entry guard state, set by install_gap_state (None on both = guard off, and the
# gap classification then runs the exact committed high_shot_oob-entry branch). Kept
# beside the demotion bound because the guard is part of the gap-state mechanism:
# one install call fully specifies the whole arm's per-process state, so a worker
# reprocessing an unguarded cell after a guarded one can't inherit a stale guard.
_REENTRY_GUARD_VARIANT: ReentryGuardVariant | None = None
_REENTRY_GUARD_BUFFER: float | None = None


def _gap_is_high_shot_oob(track: np.ndarray, gap_start: int) -> bool:
    """Does the shuttle's entry into an invisibility gap arc off the TOP edge?

    Reads the contiguous run of visible frames immediately before ``gap_start``, capped
    at the last HIGH_SHOT_OOB_LOOKBACK_FRAMES. Given at least HIGH_SHOT_OOB_MIN_VISIBLE_FRAMES of them it
    takes the mean per-frame velocity (last minus first over the step count, which is
    the mean of the consecutive steps), extrapolates HIGH_SHOT_OOB_EXTRAP_FRAMES on from the last
    visible position, and calls it high_shot_oob when that lands above the top of the frame
    (y < 0, the spec's parenthetical reading of "leaves the unit square through the
    top"; the x-bound is not tested). Fewer than two visible frames give no velocity,
    so the gap is not high_shot_oob.

    :param track: ``(t, 3)`` whole-video track.
    :param gap_start: first frame of the invisibility run.
    :return: True when the pre-gap motion extrapolates off the top edge.
    """
    run_start = gap_start
    while run_start > 0 and track[run_start - 1, 2] == 1 and gap_start - run_start < HIGH_SHOT_OOB_LOOKBACK_FRAMES:
        run_start -= 1
    n_visible = gap_start - run_start
    if n_visible < HIGH_SHOT_OOB_MIN_VISIBLE_FRAMES:
        return False
    first_xy = track[run_start, :2]  # (2,) oldest visible in the run
    last_xy = track[gap_start - 1, :2]  # (2,) last sighting before the gap
    mean_velocity = (last_xy - first_xy) / (n_visible - 1)  # per-frame, telescoped over the run
    extrapolated_y = last_xy[1] + HIGH_SHOT_OOB_EXTRAP_FRAMES * mean_velocity[1]
    return bool(extrapolated_y < 0.0)


def _reentry_velocity_y(track: np.ndarray, gap_end: int) -> float:
    """Mean per-frame y-velocity over the first visible frames after reappearance.

    Reads the contiguous run of visible frames from ``gap_end`` (the first frame
    after the gap), capped at REENTRY_LOOKAHEAD_FRAMES, and returns the telescoped
    mean step (last minus first over the step count), mirroring the entry-side
    velocity. y grows downward, so a positive value means the shuttle is
    descending on re-entry.

    Returns NaN when the gap never reappears (``gap_end`` past the track end) or
    fewer than REENTRY_MIN_VISIBLE_FRAMES follow before the next gap: no velocity
    to fit.

    :param track: ``(t, 3)`` whole-video track.
    :param gap_end: first frame after the invisibility run (``true_runs`` exclusive end).
    :return: mean re-entry y-velocity, or NaN when undefined.
    """
    n_frames = len(track)
    if gap_end >= n_frames:
        return float('nan')  # open-ended gap, never reappears
    stop = gap_end
    limit = min(gap_end + REENTRY_LOOKAHEAD_FRAMES, n_frames)
    while stop < limit and track[stop, 2] == 1:
        stop += 1
    n_visible = stop - gap_end
    if n_visible < REENTRY_MIN_VISIBLE_FRAMES:
        return float('nan')
    return float((track[stop - 1, 1] - track[gap_end, 1]) / (n_visible - 1))


def _gap_passes_reentry_guard(track: np.ndarray, gap_start: int, gap_end: int) -> bool:
    """Does a high_shot_oob-entry gap also clear the re-entry guard?

    Only called once the entry test (``_gap_is_high_shot_oob``) has fired, so the
    pre-gap visible run exists and ``gap_start - 1`` is a valid last sighting. The
    guard demands the shuttle reappear near the top (reappearance y <= buffer) and
    descend (re-entry y-velocity > 0). TWO_SIDED additionally requires the entry y
    (last visible before the gap) within the same buffer. A gap whose re-entry
    velocity is undefined (too few visible frames after reappearance) fails: the NaN
    compares False. An open-ended gap gives no re-entry evidence at all, so the
    guard abstains (both variants) and the demotion bound owns it, as in the
    unguarded arm.

    :param track: ``(t, 3)`` whole-video track.
    :param gap_start: first frame of the invisibility run.
    :param gap_end: first frame after the run (``true_runs`` exclusive end).
    :return: True when the gap may classify HIGH_SHOT_OOB.
    """
    variant = _REENTRY_GUARD_VARIANT
    buffer = _REENTRY_GUARD_BUFFER
    assert variant is not None and buffer is not None, 'reentry guard checked without install'
    if gap_end >= len(track):
        return True  # never reappears: nothing to judge, defer to the demotion bound
    reappearance_y = track[gap_end, 1]
    near_top = reappearance_y <= buffer
    descending = _reentry_velocity_y(track, gap_end) > 0.0
    if variant is ReentryGuardVariant.TWO_SIDED:
        entry_y = track[gap_start - 1, 1]  # last sighting before the gap
        return bool(entry_y <= buffer and near_top and descending)
    return bool(near_top and descending)


def _gap_holds_open(track: np.ndarray, gap_start: int, gap_end: int) -> bool:
    """Does this gap classify HIGH_SHOT_OOB (hold the rally open) rather than rest?

    The committed top-arc entry test is necessary. With the guard off (the default,
    and every call absent --reentry-guard) it is also sufficient, so this returns
    the entry test verbatim and the classification arithmetic is bit-for-bit the
    committed gap-state arm. With the guard installed the gap must also clear the
    re-entry guard.
    """
    if not _gap_is_high_shot_oob(track, gap_start):
        return False
    if _REENTRY_GUARD_VARIANT is None:
        return True  # guard off: exact committed behaviour
    return _gap_passes_reentry_guard(track, gap_start, gap_end)


def _classify_gap_states(track: np.ndarray, demotion_bound: int) -> tuple[np.ndarray, np.ndarray]:
    """Split every invisibility gap into HIGH_SHOT_OOB / DEAD frame masks (BLIP is neither).

    Per maximal invisible run:
      - HIGH_SHOT_OOB gap (``_gap_holds_open``: top-arc entry, and the re-entry guard too
        when installed): HIGH_SHOT_OOB from the gap start; frames from
        ``gap_start + demotion_bound`` onward demote to DEAD, since a gap that long is
        no longer a shuttle mid-flight.
      - BLIP gap (not HIGH_SHOT_OOB, length <= BLIP_MAX_FRAMES): neither HIGH_SHOT_OOB nor DEAD, so its
        frames fall through to the plain slow test in the caller.
      - DEAD gap (everything else invisible): DEAD across its whole length.

    A gap the re-entry guard rejects takes the same BLIP-or-DEAD path it would have
    had the entry test never fired, so the guard only ever removes a hold, never
    adds one.

    :param track: ``(t, 3)`` whole-video track.
    :param demotion_bound: frames of high_shot_oob before an ongoing gap demotes to DEAD.
    :return: ``(high_shot_oob, dead)``, each ``(t,)`` bool and disjoint by construction.
    """
    n_frames = len(track)
    high_shot_oob = np.zeros(n_frames, dtype=bool)
    dead = np.zeros(n_frames, dtype=bool)
    invisible = track[:, 2] != 1
    for gap_start, gap_end in stage8_module.true_runs(invisible):
        if _gap_holds_open(track, gap_start, gap_end):
            demotion_frame = min(gap_start + demotion_bound, gap_end)
            high_shot_oob[gap_start:demotion_frame] = True
            dead[demotion_frame:gap_end] = True  # empty slice when the gap never reaches the bound
        elif gap_end - gap_start > BLIP_MAX_FRAMES:
            dead[gap_start:gap_end] = True
        # else: a short non-high_shot_oob gap is a BLIP, left False in both masks
    return high_shot_oob, dead


def gap_state_rest_mask(speed: np.ndarray, track: np.ndarray) -> np.ndarray:
    """Gap-state rest flag: ``DEAD OR (slow AND NOT HIGH_SHOT_OOB)``; the --gap-state _rest_mask.

    Replaces stock's ``slow | mostly_untracked``. The slow test is unchanged (rolling
    nanmedian speed below REST_SPEED). What changes is how invisibility reads: a high_shot_oob
    gap is never rest so it can't cut a rally mid-clear, a DEAD gap is always rest, and
    a BLIP gap gets no special treatment and rides the slow test like any frame.
    demotion_bound comes from the per-process ``_GAP_STATE_DEMOTION_BOUND`` set at install.

    On an all-visible (gap-free) track there are no gaps, so HIGH_SHOT_OOB and DEAD are empty and
    this returns ``slow`` exactly, which is what stock returns there too (its
    mostly_untracked is all False when every window is fully tracked): the bit-identical
    pin the tests assert.

    :param speed: ``(t,)`` per-frame speed (NaN on non-visible steps).
    :param track: ``(t, 3)`` whole-video track.
    :return: ``(t,)`` bool rest flag.
    """
    demotion_bound = _GAP_STATE_DEMOTION_BOUND
    assert demotion_bound is not None, 'gap-state arm ran without a demotion_bound; install_gap_state first'
    speed_median = stage8_module.rolling_nanmedian(speed, stage8_module.REST_WINDOW)  # (t,)
    slow = speed_median < stage8_module.REST_SPEED  # NaN windows read not-slow, as in stock
    high_shot_oob, dead = _classify_gap_states(track, demotion_bound)
    return dead | (slow & ~high_shot_oob)


def install_gap_state(
    demotion_bound: int,
    reentry_guard_variant: ReentryGuardVariant | None = None,
    reentry_guard_buffer: float | None = None,
) -> None:
    """Swap stage-8's ``_rest_mask`` for the gap-state variant, in this process.

    Sets the per-process demotion_bound and re-entry guard state the variant reads,
    then rebinds ``stage8_module._rest_mask``. ``segment_video`` resolves ``_rest_mask``
    through the module dict at call time, so the swap takes hold without touching
    committed code. Re-callable, and it always writes ALL of the arm's per-process
    state (guard included, even to None), so a worker reusing this seam across cells
    can't inherit a stale guard from an earlier one: the CLI installs once per
    worker, the measurement runner once per config.

    :param demotion_bound: frames of high_shot_oob before an ongoing gap demotes to DEAD.
    :param reentry_guard_variant: which guard sides to test, or None to leave the arm
        at its committed entry-only behaviour.
    :param reentry_guard_buffer: the near-top y buffer; required when a variant is set.
    """
    global _GAP_STATE_DEMOTION_BOUND, _REENTRY_GUARD_VARIANT, _REENTRY_GUARD_BUFFER
    if (reentry_guard_variant is None) != (reentry_guard_buffer is None):
        raise ValueError('reentry guard needs both a variant and a buffer, or neither')
    _GAP_STATE_DEMOTION_BOUND = demotion_bound
    _REENTRY_GUARD_VARIANT = reentry_guard_variant
    _REENTRY_GUARD_BUFFER = reentry_guard_buffer
    stage8_module._rest_mask = gap_state_rest_mask


# ---------------------------------------------------------------------------
# Optional quiet-start _find_rally_spans (installed once per worker; --quiet-start)
# ---------------------------------------------------------------------------
# Stock opens a rally at the FIRST sustained fast burst in each active region. On this
# footage that first burst is often a warm-up swing or replay motion well before the
# real serve, so spans open early and merge across rallies. This arm demands a quiet
# run-up: a burst only starts a rally when the W frames before it are mostly at rest.
# A region with no quiet-preceded burst falls back to its first burst (the stock pick),
# so a span forms wherever stock formed one. That preserves span COUNT, not coverage:
# a later start shrinks the span, and a rally whose first strokes fall in the cut
# prefix drops out of coverage (measured on the pilot: W25 uncovers 11 rallies).
QUIET_START_REST_FRACTION = 0.8  # min at_rest fraction over the pre-burst window to qualify


def _is_quiet_before(at_rest: np.ndarray, burst_start: int, window: int) -> bool:
    """Is the ``window``-frame run immediately before a burst mostly at rest?

    True when the at_rest fraction over ``[burst_start - window, burst_start)`` is at
    least QUIET_START_REST_FRACTION. Near the video start the window truncates to the
    frames that exist; an empty window (a burst at frame 0) can't be confirmed quiet,
    so it reads False and the burst falls back to the stock choice downstream.

    :param at_rest: ``(t,)`` per-frame rest flag the caller already computed.
    :param burst_start: first frame of the candidate burst.
    :param window: frames to look back (the swept W).
    :return: True when the pre-burst window is quiet enough to open a rally.
    """
    before = at_rest[max(0, burst_start - window):burst_start]
    if len(before) == 0:
        return False
    return bool(before.mean() >= QUIET_START_REST_FRACTION)


def quiet_start_find_rally_spans(speed: np.ndarray, at_rest: np.ndarray) -> list[tuple[int, int]]:
    """Span-start rule that demands a quiet run-up; the --quiet-start _find_rally_spans.

    Same region and long-rest structure as stock (extended rest separates rallies). The
    one change: a rally opens at the first fast burst in its active region whose
    preceding W frames are quiet (``_is_quiet_before``) rather than the first burst
    outright. A region with no quiet-preceded burst falls back to its first burst, the
    stock choice, so a span forms wherever stock formed one. That preserves span COUNT,
    not coverage: moving a start later shrinks the span, and a rally whose first
    strokes fall in the cut prefix drops out of coverage (the pilot measured 11 such
    rallies at W25). W comes from the per-process ``_QUIET_START_WINDOW`` set at
    install.

    :param speed: ``(t,)`` per-frame speed (NaN on non-visible steps).
    :param at_rest: ``(t,)`` per-frame rest flag.
    :return: list of ``(start_frame, end_frame)`` half-open rally spans.
    """
    window = _QUIET_START_WINDOW
    assert window is not None, 'quiet-start arm ran without a window; install_quiet_start first'
    fast = np.nan_to_num(speed, nan=0.0) > stage8_module.START_SPEED  # (t,) NaN steps are not fast

    long_rest = np.zeros(len(speed), dtype=bool)  # (t,) frames inside an extended rest
    for start, end in stage8_module.true_runs(at_rest):
        if end - start >= stage8_module.END_REST_FRAMES:
            long_rest[start:end] = True

    fast_runs = [
        (start, end) for start, end in stage8_module.true_runs(fast)
        if end - start >= stage8_module.START_MIN_FRAMES
    ]

    spans: list[tuple[int, int]] = []
    for region_start, region_end in stage8_module.true_runs(~long_rest):
        bursts = [start for start, _ in fast_runs if region_start <= start < region_end]
        if not bursts:
            continue
        quiet_burst = next((start for start in bursts if _is_quiet_before(at_rest, start, window)), None)
        burst_start = quiet_burst if quiet_burst is not None else bursts[0]  # fallback keeps coverage
        spans.append((int(burst_start), int(region_end)))
    return spans


def install_quiet_start(window: int) -> None:
    """Swap stage-8's ``_find_rally_spans`` for the quiet-start variant, in this process.

    Sets the per-process window the variant reads, then rebinds
    ``stage8_module._find_rally_spans``. ``segment_video`` resolves it through the module
    dict at call time, so the swap takes hold without touching committed code.
    Re-callable, same as ``install_gap_state``.
    """
    global _QUIET_START_WINDOW
    _QUIET_START_WINDOW = window
    stage8_module._find_rally_spans = quiet_start_find_rally_spans


# ---------------------------------------------------------------------------
# Optional serve-start _find_rally_spans (installed once per worker; --serve-start)
# ---------------------------------------------------------------------------
# Stock (and quiet-start) open a rally at the first sustained fast burst in each active
# region. On this footage the dead time between rallies is saturated with bursts (tracker
# jitter, carry, replays), so the first burst lands a median ~109 s before the real serve.
# This arm opens a span only at a burst whose lookback looks like SERVE SETUP: over the
# last second before a serve the shuttle sits close to a court-scale player (held for the
# toss). Scoped in start_side_scoping.py: the gate "median shuttle-to-nearest-court-scale-
# person-bbox-centre distance <= 0.10 over the visible frames in the last 25 frames" keeps
# 103/113 GT serves and excludes 600/619 dead-time controls. Same seam as quiet-start:
# committed stage-8 code stays untouched; the swap lives here and only runs in a process
# that installs it (gated by --serve-start).
#
# Two fallback modes for a region with NO qualifying burst, both measured (Ariel's ruling):
#   TRIM   - fall back to the region's first burst (the stock pick), so the span survives
#            at the stock start; only the qualifying regions get trimmed to a later start.
#   REJECT - produce no span at all (drop the region): the stronger anti-weld / anti-spurious
#            lever, kills replay and crowd stretches, at the risk of dropping a GT rally that
#            sits in a no-qualify region.
#
# LOUD CAVEAT ON THE STAND-IN COURT REGION. These constants are PILOT-VIDEO-1-ONLY, data-derived
# stand-ins for the court quad (production gets CourtKeyNet). They are the axis-quantile
# foot-point box and padded pixel-height band fitted in start_side_scoping.py from the
# top-2-score in-rally detections. NOT a measured court quad; do not reuse for another
# video (one broadcast camera, one court framing).
PILOT_STANDIN_COURT_X = (635.0, 1316.0)  # foot-point x bounds, pixels (frame is 1920 wide)
PILOT_STANDIN_COURT_Y = (254.0, 1030.0)  # foot-point y bounds, pixels (frame is 1080 tall)

# The homography alternative the --court-box flag selects. Bounding box of ShuttleSet's recorded
# court quad for pilot vid 1, mapped into the 1920x1080 working space by homography_check.py
# (verified 2026-07-10 against ShuttleSet's static, per-video homography; the four corners live in
# local_scratch/autograder_architecture/pilot_results/homography_view_rule/homography_check.csv).
# Being the ACTUAL court-outline quad's bounding box, it sits TIGHTER than the occupancy stand-ins
# above: players legitimately stand outside it (lunges, clears past the baseline), and courtside
# officials sit outside it but inside the stand-in box.
PILOT_HOMOG_COURT_X = (460.8, 1459.5)  # quad x-range, pixels (downleft .. downright corners)
PILOT_HOMOG_COURT_Y = (461.1, 1006.8)  # quad y-range, pixels (upright .. downleft corners)
# The net line's image y under the same homography (H^-1 of the court-space mid-line, spread
# under 2.2 px across the court width; homography_check.py, 2026-07-10). Perspective
# compresses the far half, so no arithmetic midpoint lands here: the stand-in half-split
# (642) sits 42 px above the real net, the quad-y midpoint (~734) 50 px below it.
PILOT_HOMOG_COURT_MID_Y = 683.9
# The homography half-split is a BAND, not a knife edge: the recorded homography binds a flat
# template to a monocular image of a 3D net (own height, minor x/y/z sag), so the floor-plane
# net line carries model error beyond its numeric spread (Ariel's ruling, 2026-07-10). The
# band is +/-0.5 m along court length at the net, through H^-1 at centre-court x; feet inside
# it count to NEITHER court half. Vid 15's band is (583.9, 626.6) for the coming
# generalisation window.
PILOT_HOMOG_COURT_MID_BAND = (664.6, 703.7)

PILOT_PLAYER_HEIGHT = (84.0, 336.0)  # court-player bbox pixel-height band
PILOT_RESOLUTION = (1920.0, 1080.0)  # (W, H) the shuttle xy and the bbox centres normalise by

# The active court geometry the serve-start machinery reads: the foot-point x/y filter in
# _court_scale_boxes plus the top/bottom half-split band in the wide-shot gate. --court-box
# swaps all three together (install_court_box rebinds them); stand-in is the default,
# bit-identical to before the flag existed (its band is zero-width: the stand-in split never
# had a buffer and the audited 70/113 + 10/619 anchors pin it). The height band stays the
# stand-in under both choices (the homography fixes the court outline, it says nothing about
# bbox pixel heights).
PILOT_STANDIN_COURT_MID_Y = (PILOT_STANDIN_COURT_Y[0] + PILOT_STANDIN_COURT_Y[1]) / 2.0  # 642 px
PILOT_STANDIN_COURT_MID_BAND = (PILOT_STANDIN_COURT_MID_Y, PILOT_STANDIN_COURT_MID_Y)
PILOT_COURT_X = PILOT_STANDIN_COURT_X
PILOT_COURT_Y = PILOT_STANDIN_COURT_Y
PILOT_COURT_MID_BAND = PILOT_STANDIN_COURT_MID_BAND
COURT_BOXES = {
    'standin': (PILOT_STANDIN_COURT_X, PILOT_STANDIN_COURT_Y, PILOT_STANDIN_COURT_MID_BAND),
    'homography': (PILOT_HOMOG_COURT_X, PILOT_HOMOG_COURT_Y, PILOT_HOMOG_COURT_MID_BAND),
}

SERVE_START_LOOKBACK_FRAMES = 25  # the last second before a burst (25 fps): the serve-setup window

# Wide-shot refinement (--serve-start-wideshot, OFF by default): over the same lookback the
# burst must also look like the serve WIDE SHOT (both players standing in position). Pinned
# by the windows.csv audit: median court-scale detection count >= 2, BOTH court halves
# occupied (a court-scale box present in >= half the lookback frames per half), and per-half
# total foot drift <= 0.05 image-fraction. With THESE PILOT_* constants the gate alone keeps
# 70/113 GT starts and 10/619 scoping controls; windows.csv itself reads 9/619 because the
# scoping scored with its unrounded derived court model, and the constant rounding flips one
# knife-edge dead-time control at the drift bound (0.0481 rounded vs 0.0526 derived).
WIDESHOT_COUNT_MED_MIN = 2.0      # median court-scale detections over the lookback
WIDESHOT_SLOT_PRESENT_FRAC = 0.5  # a half counts as occupied when present >= this fraction
WIDESHOT_DRIFT_MAX = 0.05         # max per-half total foot drift, image-fraction
WIDESHOT_DRIFT_END_FRAMES = 10    # drift = gap between head/tail means over up-to-10 present feet
# The active top/bottom half-split lives in PILOT_COURT_MID_BAND (above, beside the court
# boxes); --court-box homography rebinds it to the buffered net-line band.


class ServeStartMode(StrEnum):
    """What a region whose bursts are none of them serve-setup-preceded does.

    TRIM keeps the span at the region's first burst (the stock pick), so coverage is only
    ever traded by a later start on the QUALIFYING regions. REJECT drops the region outright,
    the stronger anti-weld / anti-spurious lever, at the risk of dropping a GT rally that
    happens to sit in a no-qualify region. Ariel ruled both get measured.
    """

    TRIM = 'trim'
    REJECT = 'reject'


class ServeStartClose(StrEnum):
    """Where a split span closes; the --serve-start-split axis (None = single span, off).

    In split mode every serve-setup-qualifying burst opens a span. BURST closes the previous
    span exactly at the next qualifying burst, so the split spans union back to the single
    span and coverage is unchanged. LAST_REST closes it at the start of the last rest run
    before that burst, leaving the between-rally dead tail (where the junk contacts live)
    outside the span, at the risk of uncovering a rally whose own serve failed the gate.
    """

    BURST = 'burst'
    LAST_REST = 'last_rest'


class WideshotInputs(NamedTuple):
    """Per-frame wide-shot gate inputs, precomputed once per process.

    Built by ``build_serve_start_wideshot_inputs`` from the same raw pose boxes the
    serve-start distance array uses, so the gate never recomputes per burst.
    """

    count: np.ndarray  # (t,) court-scale detection count per frame
    top_foot: np.ndarray  # (t, 2) best top-half court-scale foot, image-fraction; NaN when absent
    bot_foot: np.ndarray  # (t, 2) same for the bottom half


# demotion_bound / W live on the module for the other arms; serve-start's swept knobs
# (threshold, mode) and its precomputed distance array live here as per-process state the
# install sets and the patched function reads at call time. One task runs at a time per
# process, so this cannot leak across tasks.
_SERVE_START_DIST: np.ndarray | None = None
_SERVE_START_THRESHOLD: float | None = None
_SERVE_START_MODE: ServeStartMode | None = None
_SERVE_START_WIDESHOT: WideshotInputs | None = None  # None = refinement off (the default)
_SERVE_START_CLOSE: ServeStartClose | None = None    # None = single span (split off, the default)

# Diagnostics from the LAST serve_start_find_rally_spans call: how many regions had a burst
# but no qualifying one (fell back under TRIM / dropped under REJECT) and their extents, so
# the measurement runner can count the GT rallies at risk. Single writer (the patched
# function rewrites it every call), valid to read immediately after segment_video in the same
# process. None until the first call.
_SERVE_START_LAST_DIAGNOSTICS: dict | None = None


def install_court_box(choice: str) -> None:
    """Point the serve-start court geometry at the chosen box, in this process.

    'standin' keeps the data-fitted occupancy box and its zero-width 642 px half-split (the
    module default, bit-identical to before the flag existed); 'homography' swaps in the
    court quad's bounding box AND the buffered net-line half-split band (the geometry moves
    together: a 642 split against the tighter quad would sit 42 px above the real net, and
    the band absorbs the net-line's 3D model error). The height band stays the stand-in
    under both (the homography fixes the court outline, not bbox heights). Main-only:
    _court_scale_boxes and the wide-shot half-split run when the serve-start
    distance/wide-shot inputs are precomputed, both in the main process, so there is no
    per-worker install to thread. Re-callable, same seam as install_serve_start.
    """
    global PILOT_COURT_X, PILOT_COURT_Y, PILOT_COURT_MID_BAND
    PILOT_COURT_X, PILOT_COURT_Y, PILOT_COURT_MID_BAND = COURT_BOXES[choice]


def _court_scale_boxes(
    frame_bboxes: np.ndarray, frame_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One frame's court-scale person boxes; the serve-start builders' shared filter.

    Keeps the detections whose foot point (bottom-centre) sits inside the pilot court
    region AND whose pixel height is court-player scale; both builders filter through
    here so the rule lives once. Detection validity is ``np.isfinite(scores)``, NOT a
    >= 0.5 cutoff: the pose extraction already floored scores at 0.30, so ``isfinite``
    equals the scoping's ndet (verified bit-for-bit), and a 0.5 cutoff would drop real
    detections the scoping kept and reconcile the distance gate to 100/113, not 103/113.
    Padding slots carry NaN scores and drop out.

    :param frame_bboxes: (16, 4) xyxy person boxes in pixels, NaN-padded past the detections.
    :param frame_scores: (16,) detection scores, NaN on padding slots.
    :return: (x1, y1, x2, y2, scores) filtered to the court-scale detections, each (k,).
    """
    valid = np.isfinite(frame_scores)
    x1, y1, x2, y2 = frame_bboxes[valid].T  # each (m,) pixels
    foot_x = (x1 + x2) / 2.0  # bottom-centre; foot y is y2
    box_height = y2 - y1
    x_lo, x_hi = PILOT_COURT_X
    y_lo, y_hi = PILOT_COURT_Y
    height_lo, height_hi = PILOT_PLAYER_HEIGHT
    in_court = (x_lo <= foot_x) & (foot_x <= x_hi) & (y_lo <= y2) & (y2 <= y_hi)
    in_scale = (height_lo <= box_height) & (box_height <= height_hi)
    court_scale = in_court & in_scale
    return (x1[court_scale], y1[court_scale], x2[court_scale], y2[court_scale],
            frame_scores[valid][court_scale])


def build_serve_start_dist(track: np.ndarray, bboxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Per-frame nearest court-scale player bbox-centre distance; the serve-start gate input.

    For every frame where the shuttle is visible, take the frame's court-scale detections
    (the rule lives in ``_court_scale_boxes``) and return the smallest normalised
    (image-fraction) distance from the shuttle to any of their bbox centres. NaN where the
    shuttle is invisible or no court-scale detection is present. Mirrors
    start_side_scoping.py's b1_centre_dist exactly, so it reproduces the audited
    late-window count (103/113 GT serves under the <= 0.10 gate).

    :param track: (t, 3) [x_norm, y_norm, visibility] whole-video track.
    :param bboxes: (t, 16, 4) xyxy person boxes in pixels, NaN-padded past the detections.
    :param scores: (t, 16) detection scores, NaN on padding slots.
    :return: (t,) float; NaN where the shuttle is invisible or no court-scale player is present.
    """
    n_frames = len(track)
    width, height = PILOT_RESOLUTION

    dist = np.full(n_frames, np.nan)
    for frame in np.flatnonzero(track[:, 2] == 1):
        x1, y1, x2, y2, _ = _court_scale_boxes(bboxes[frame], scores[frame])
        if len(x1) == 0:
            continue
        centre_x = (x1 + x2) / 2.0 / width
        centre_y = (y1 + y2) / 2.0 / height
        gaps = np.hypot(centre_x - track[frame, 0], centre_y - track[frame, 1])  # image-fraction
        dist[frame] = gaps.min()
    return dist


def _serve_setup_before(dist: np.ndarray, burst_start: int, threshold: float) -> bool:
    """Does the lookback before a burst look like serve setup?

    True when the median of the finite nearest-court-scale-centre distances over the
    SERVE_START_LOOKBACK_FRAMES frames immediately before burst_start is <= threshold. A
    lookback with no finite frame (the shuttle never visible near a court-scale player in that
    second) can't evidence setup, so it reads False (NaN fails), mirroring the scoping's gate.
    Near the video start the lookback truncates to the frames that exist.

    :param dist: (t,) per-frame nearest-court-scale-centre distance, NaN where undefined.
    :param burst_start: first frame of the candidate burst.
    :param threshold: the swept gate distance (image-fraction).
    :return: True when the pre-burst second reads as serve setup.
    """
    lookback = dist[max(0, burst_start - SERVE_START_LOOKBACK_FRAMES):burst_start]
    finite = lookback[np.isfinite(lookback)]
    if len(finite) == 0:
        return False
    return bool(np.median(finite) <= threshold)


def build_serve_start_wideshot_inputs(bboxes: np.ndarray, scores: np.ndarray) -> WideshotInputs:
    """Per-frame court-scale count and per-half best feet; the wide-shot gate input.

    For EVERY frame (shuttle visibility is irrelevant to the wide shot), count the frame's
    court-scale detections (the rule lives in ``_court_scale_boxes``, shared with
    build_serve_start_dist) and keep the highest-score court-scale foot per court half
    (top: foot y < the court mid-line; bottom: the rest), normalised to image-fraction.
    Mirrors the scoping's per-frame extraction (start_side_scoping.py extract_window)
    under the rounded PILOT_* constants.

    :param bboxes: (t, 16, 4) xyxy person boxes in pixels, NaN-padded past the detections.
    :param scores: (t, 16) detection scores, NaN on padding slots.
    :return: WideshotInputs of (t,) counts and (t, 2) per-half feet (NaN when a half is empty).
    """
    n_frames = len(bboxes)
    width, height = PILOT_RESOLUTION

    count = np.zeros(n_frames, dtype=int)
    top_foot = np.full((n_frames, 2), np.nan)
    bot_foot = np.full((n_frames, 2), np.nan)
    for frame in range(n_frames):
        x1, _, x2, y2, cs_scores = _court_scale_boxes(bboxes[frame], scores[frame])
        count[frame] = len(x1)
        if len(x1) == 0:
            continue
        foot_x = (x1 + x2) / 2.0  # bottom-centre; foot y is y2
        # Feet inside the mid band claim NEITHER half (net-line 3D model error); the standin
        # band is zero-width, so there y2 >= band_hi is exactly the audited ~(y2 < mid).
        band_lo, band_hi = PILOT_COURT_MID_BAND
        in_top_half = y2 < band_lo
        in_bot_half = y2 >= band_hi
        for half_idx, foot_out in ((np.flatnonzero(in_top_half), top_foot),
                                   (np.flatnonzero(in_bot_half), bot_foot)):
            if len(half_idx):
                best = half_idx[np.argmax(cs_scores[half_idx])]
                foot_out[frame] = (foot_x[best] / width, y2[best] / height)
    return WideshotInputs(count=count, top_foot=top_foot, bot_foot=bot_foot)


def _slot_total_drift(foot_series: np.ndarray) -> tuple[float, int]:
    """Total foot drift and present-count for one court half over a lookback.

    Drift is the gap between the means of the first and last WIDESHOT_DRIFT_END_FRAMES
    present feet (net relocation, robust to end jitter), NaN when the series has no more
    feet than one window. Between 11 and 19 present feet the two windows partially
    overlap and damp the reading; that is the scoping's exact arithmetic and stays,
    since the audited 70/113 + 10/619 keep rates pin it. Mirrors the scoping's
    slot_drift otherwise.

    :param foot_series: (n, 2) image-fraction feet, NaN rows where the half is empty.
    :return: (total_drift, present_count).
    """
    present = np.flatnonzero(np.isfinite(foot_series[:, 0]))
    if len(present) <= WIDESHOT_DRIFT_END_FRAMES:
        # One window's worth of feet or fewer: head and tail would fully overlap and
        # read exactly 0.0, silently passing the drift bound even for a sprinting
        # player. Abstain to NaN, which the gate fails closed. Only a truncated
        # lookback near the video start can get here past the presence gate (a full
        # 25-frame lookback guarantees >= 13 present per half), so the audited keep
        # rates, all full-lookback windows, cannot move.
        return float('nan'), len(present)
    head = foot_series[present[:WIDESHOT_DRIFT_END_FRAMES]].mean(axis=0)
    tail = foot_series[present[-WIDESHOT_DRIFT_END_FRAMES:]].mean(axis=0)
    return float(np.linalg.norm(tail - head)), len(present)


def _wide_shot_before(inputs: WideshotInputs, burst_start: int) -> bool:
    """Does the lookback before a burst look like the serve wide shot?

    Three conditions over the SERVE_START_LOOKBACK_FRAMES frames immediately before
    burst_start (windows.csv's late-window definitions, audited at 70/113 GT starts and
    10/619 controls under the PILOT_* constants): median court-scale detection count
    >= WIDESHOT_COUNT_MED_MIN, both court halves occupied (each half's best box present
    in >= WIDESHOT_SLOT_PRESENT_FRAC of the lookback), and every occupied half's total
    foot drift <= WIDESHOT_DRIFT_MAX. A NaN drift (under two present feet) fails, same
    as the scoping's gate. Near the video start the lookback truncates.

    :param inputs: the precomputed per-frame gate inputs.
    :param burst_start: first frame of the candidate burst.
    :return: True when the pre-burst second reads as the serve wide shot.
    """
    lookback = slice(max(0, burst_start - SERVE_START_LOOKBACK_FRAMES), burst_start)
    count = inputs.count[lookback]
    if len(count) == 0 or np.median(count) < WIDESHOT_COUNT_MED_MIN:
        return False

    top_drift, top_present = _slot_total_drift(inputs.top_foot[lookback])
    bot_drift, bot_present = _slot_total_drift(inputs.bot_foot[lookback])
    present_min = WIDESHOT_SLOT_PRESENT_FRAC * len(count)
    if top_present < present_min or bot_present < present_min:
        return False

    # Max over the two halves: both players must be still, so the noisier half binds.
    # NaN halves drop out and no finite drift at all fails, the scoping's exact arithmetic
    # (a full 25-frame lookback past the present gate always has both finite; only a
    # truncated lookback near the video start can reach the no-finite branch).
    finite_drifts = [drift for drift in (top_drift, bot_drift) if np.isfinite(drift)]
    if not finite_drifts:
        return False
    return bool(max(finite_drifts) <= WIDESHOT_DRIFT_MAX)


def _last_rest_close(rest_runs: list[tuple[int, int]], open_frame: int, next_burst: int) -> int:
    """Where a split span closes under close='last_rest'.

    The START of the last at_rest run (any length) that ends at or before the next qualifying
    burst AND opens after this span's own open, so the dead tail between the previous rally's
    last action and the next serve falls outside the span (that tail is where the between-rally
    junk contacts sit). Falls back to next_burst when no rest run sits between the two opens
    (the span then swallows the tail, same as close='burst').

    :param rest_runs: every ``(start, end)`` at_rest run, ascending (stage8 true_runs order).
    :param open_frame: this span's opening (qualifying) burst frame.
    :param next_burst: the next qualifying burst frame (where close='burst' would cut).
    :return: the close frame (half-open span end).
    """
    rest_starts = [rest_start for rest_start, rest_end in rest_runs
                   if rest_end <= next_burst and rest_start > open_frame]
    return rest_starts[-1] if rest_starts else next_burst  # ascending, so [-1] is the last run


def serve_start_find_rally_spans(speed: np.ndarray, at_rest: np.ndarray) -> list[tuple[int, int]]:
    """Span-start rule that opens only at a serve-setup-preceded burst; the --serve-start _find_rally_spans.

    Same region / long-rest / fast-run structure as stock. The change: a rally opens at a
    fast burst in its active region whose lookback passes the serve-setup gate
    (_serve_setup_before) rather than the first burst outright. With the optional wide-shot
    refinement installed, the same burst's lookback must ALSO pass the wide-shot gate
    (_wide_shot_before). A region with no qualifying burst is handled by the mode: TRIM
    falls back to the first burst (span survives at the stock start), REJECT produces no
    span (the region is dropped).

    The close-placement state controls what a QUALIFYING region does. Off (None): open one
    span at the FIRST qualifying burst, running to region end (the pre-split behaviour, kept
    bit-for-bit). Split (BURST / LAST_REST): open a span at EVERY qualifying burst, each
    closing where the next one opens (_last_rest_close places the cut), the last running to
    region end, so a weld of two rallies gets cut at the second serve. threshold, mode, the
    distance array, the optional wide-shot inputs and the close placement all come from the
    per-process state set at install.

    Records per-call diagnostics (regions that fell back / were dropped, plus the per-region
    qualifying-burst counts and consecutive-burst spacings for the double-fire read) on
    _SERVE_START_LAST_DIAGNOSTICS for the measurement runner; see its note above.

    :param speed: (t,) per-frame speed (NaN on non-visible steps).
    :param at_rest: (t,) per-frame rest flag.
    :return: list of (start_frame, end_frame) half-open rally spans.
    """
    dist = _SERVE_START_DIST
    threshold = _SERVE_START_THRESHOLD
    mode = _SERVE_START_MODE
    wideshot = _SERVE_START_WIDESHOT
    close = _SERVE_START_CLOSE
    assert dist is not None and threshold is not None and mode is not None, \
        'serve-start arm ran without state; install_serve_start first'

    def qualifies(burst: int) -> bool:
        """Serve-setup distance gate, AND the wide-shot gate when the refinement is on."""
        if not _serve_setup_before(dist, burst, threshold):
            return False
        return wideshot is None or _wide_shot_before(wideshot, burst)

    fast = np.nan_to_num(speed, nan=0.0) > stage8_module.START_SPEED  # (t,) NaN steps are not fast

    rest_runs = stage8_module.true_runs(at_rest)  # every rest run, any length; close='last_rest' reads these
    long_rest = np.zeros(len(speed), dtype=bool)  # (t,) frames inside an extended rest
    for start, end in rest_runs:
        if end - start >= stage8_module.END_REST_FRAMES:
            long_rest[start:end] = True

    fast_runs = [
        (start, end) for start, end in stage8_module.true_runs(fast)
        if end - start >= stage8_module.START_MIN_FRAMES
    ]

    spans: list[tuple[int, int]] = []
    no_qualify_regions: list[tuple[int, int]] = []
    n_regions_with_burst = 0
    # Per-region qualifying-burst counts (spans per region under split) and the frames between
    # consecutive qualifying bursts (the double-fire read: a small spacing means a second serve
    # signature fired just after a span opened, cutting one rally in two). Diagnosed, not suppressed.
    qualifying_counts: list[int] = []
    qualifying_spacings: list[int] = []
    for region_start, region_end in stage8_module.true_runs(~long_rest):
        bursts = [start for start, _ in fast_runs if region_start <= start < region_end]
        if not bursts:
            continue  # no burst at all: stock forms no span here either, not a serve-start drop
        n_regions_with_burst += 1
        # Gate every burst, not just up to the first: the split modes open a span at each, and
        # the double-fire diagnostics need the whole per-region qualifying picture regardless.
        qualifying = [start for start in bursts if qualifies(start)]
        qualifying_counts.append(len(qualifying))
        qualifying_spacings.extend(int(later - earlier)
                                   for earlier, later in zip(qualifying, qualifying[1:]))

        if not qualifying:
            # No qualifying burst: the mode owns the region, split-agnostic. TRIM keeps the span
            # at the stock first burst; REJECT drops it. Both record the region as no-qualify.
            if mode is ServeStartMode.TRIM:
                spans.append((int(bursts[0]), int(region_end)))
            no_qualify_regions.append((int(region_start), int(region_end)))
        elif close is None:
            # Split off: one span at the first qualifying burst, running to region end. This is
            # bit-for-bit the pre-split behaviour (the reproduction pin rides on it).
            spans.append((int(qualifying[0]), int(region_end)))
        else:
            # Split on: every qualifying burst opens a span, closing where the next one opens
            # (close='burst') or at the last rest run before it (close='last_rest'); the last
            # span runs to region end. Under close='burst' the spans union to the single span.
            for idx, open_frame in enumerate(qualifying):
                if idx + 1 < len(qualifying):
                    next_burst = qualifying[idx + 1]
                    close_frame = (next_burst if close is ServeStartClose.BURST
                                   else _last_rest_close(rest_runs, open_frame, next_burst))
                else:
                    close_frame = region_end
                spans.append((int(open_frame), int(close_frame)))

    global _SERVE_START_LAST_DIAGNOSTICS
    _SERVE_START_LAST_DIAGNOSTICS = {
        'n_regions_with_burst': n_regions_with_burst,
        'n_qualified': n_regions_with_burst - len(no_qualify_regions),
        'n_no_qualify': len(no_qualify_regions),
        'no_qualify_regions': no_qualify_regions,
        'qualifying_counts': qualifying_counts,
        'qualifying_spacings': qualifying_spacings,
    }
    return spans


def install_serve_start(
    dist: np.ndarray, threshold: float, mode: ServeStartMode,
    wideshot: WideshotInputs | None = None,
    close: ServeStartClose | None = None,
) -> None:
    """Swap stage-8's _find_rally_spans for the serve-start variant, in this process.

    Sets the per-process distance array, threshold and mode the variant reads, then rebinds
    stage8_module._find_rally_spans. ``wideshot`` is the optional wide-shot refinement:
    None (the default) leaves the arm's behaviour untouched; a WideshotInputs makes every
    qualifying burst also pass the wide-shot gate. ``close`` is the split placement: None
    (the default) opens one span per region at the first qualifying burst; BURST / LAST_REST
    cut the region at every qualifying burst. segment_video resolves the binding through the
    module dict at call time, so the swap takes hold without touching committed code.
    Re-callable, same as install_quiet_start; serve-start and quiet-start both own
    _find_rally_spans, so only one may be installed at a time (the CLI and _init_worker
    enforce this).
    """
    global _SERVE_START_DIST, _SERVE_START_THRESHOLD, _SERVE_START_MODE, _SERVE_START_WIDESHOT
    global _SERVE_START_CLOSE
    _SERVE_START_DIST = dist
    _SERVE_START_THRESHOLD = threshold
    _SERVE_START_MODE = mode
    _SERVE_START_WIDESHOT = wideshot
    _SERVE_START_CLOSE = close
    stage8_module._find_rally_spans = serve_start_find_rally_spans


def _clear_serve_start_state() -> None:
    """Drop the serve-start per-process arrays; the non-serve-start init paths call this.

    Restoring the _find_rally_spans binding alone leaves the distance and wideshot
    arrays referenced even though nothing reads them once stock or quiet-start owns
    the binding.
    """
    global _SERVE_START_DIST, _SERVE_START_THRESHOLD, _SERVE_START_MODE, _SERVE_START_WIDESHOT
    global _SERVE_START_CLOSE
    _SERVE_START_DIST = None
    _SERVE_START_THRESHOLD = None
    _SERVE_START_MODE = None
    _SERVE_START_WIDESHOT = None
    _SERVE_START_CLOSE = None


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
    parser.add_argument('--gap-state', type=int, default=None, metavar='DEMOTION_BOUND',
                        help='swap _rest_mask for the gap-state variant: an invisibility gap only '
                             'reads as rest when it is DEAD (a long non-high_shot_oob gap), a high_shot_oob gap (shuttle '
                             'last seen climbing off the top) stays not-rest so it cannot cut a rally '
                             'mid-clear, and a short BLIP gap rides the plain slow test. The arg is '
                             'demotion_bound: frames of high_shot_oob before an ongoing gap demotes to DEAD. '
                             'Installed per worker; the whole sweep runs at this one value')
    parser.add_argument('--quiet-start', type=int, default=None, metavar='W',
                        help='swap _find_rally_spans for the quiet-start variant: a rally opens at '
                             'the first fast burst whose preceding W frames are at least 80%% at rest, '
                             'else it falls back to the stock first-burst pick, so coverage never '
                             'drops. The arg is W, the pre-burst window in frames. Installed per '
                             'worker; composes with --gap-state')
    parser.add_argument('--reentry-guard', choices=tuple(ReentryGuardVariant), default=None,
                        metavar='VARIANT',
                        help='add the re-entry guard to --gap-state (needs it, and --reentry-buffer): a '
                             'high_shot_oob gap only holds a rally open if the shuttle also reappears near the '
                             'top and descends. VARIANT is "reentry-only" (reappearance side only) or '
                             '"two-sided" (also the entry side within the buffer)')
    parser.add_argument('--reentry-buffer', type=float, default=None, metavar='Y',
                        help='the re-entry guard\'s near-top y buffer (paired with --reentry-guard)')
    parser.add_argument('--serve-start', choices=tuple(ServeStartMode), default=None, metavar='MODE',
                        help='swap _find_rally_spans for the serve-start variant: a rally opens only at '
                             'a burst whose last second reads as serve setup (shuttle near a court-scale '
                             'player). MODE is "trim" (a no-qualify region falls back to its first burst, '
                             'coverage held) or "reject" (a no-qualify region is dropped, the stronger '
                             'anti-spurious lever). Needs --serve-start-pose-dir; mutually exclusive with '
                             '--quiet-start; composes with --gap-state')
    parser.add_argument('--serve-start-threshold', type=float, default=0.10, metavar='DIST',
                        help='serve-start gate distance in image-fraction (default 0.10, the scoped '
                             'operating point that keeps 103/113 GT serves)')
    parser.add_argument('--serve-start-pose-dir', type=Path, default=None,
                        help='directory of the raw pose npys (<prefix>_bboxes.npy, <prefix>_scores.npy) '
                             'the serve-start gate reads to precompute the shuttle-to-court-scale-centre '
                             'distance. Required with --serve-start')
    parser.add_argument('--serve-start-pose-prefix', type=str, default='pilot_1080p_raw',
                        help='filename prefix of the pose npys in --serve-start-pose-dir (default pilot_1080p_raw)')
    parser.add_argument('--serve-start-wideshot', action='store_true',
                        help='optional serve-start refinement (off by default): a burst qualifies only '
                             'if its lookback ALSO reads as the serve wide shot (median court-scale '
                             'detection count >= 2, both court halves occupied, per-half foot drift '
                             '<= 0.05). Needs --serve-start')
    parser.add_argument('--serve-start-split', choices=tuple(ServeStartClose), default=None, metavar='CLOSE',
                        help='optional serve-start refinement (off by default): cut each active region '
                             'at EVERY serve-setup-qualifying burst instead of opening one span at the '
                             'first. CLOSE is "burst" (the previous span closes at the next qualifying '
                             'burst, so coverage matches the single-span arm) or "last_rest" (it closes '
                             'at the start of the last rest run before that burst, dropping the '
                             'between-rally dead tail). Needs --serve-start')
    parser.add_argument('--court-box', choices=('standin', 'homography'), default='standin',
                        help='which court geometry the serve-start machinery uses: "standin" (default, '
                             'the data-fitted occupancy box with its zero-width 642 px half-split) or '
                             '"homography" (the homography quad bounding box for pilot vid 1, tighter '
                             'than the stand-in, with a buffered net-line half-split band 664.6-703.7 '
                             'px; feet inside the band claim neither half). The player-height band '
                             'stays the stand-in either way (the homography says nothing about bbox '
                             'pixel heights). No effect unless --serve-start is on')
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

    if (args.reentry_guard is None) != (args.reentry_buffer is None):
        parser.error('--reentry-guard and --reentry-buffer must be given together')
    if args.reentry_guard is not None and args.gap_state is None:
        parser.error('--reentry-guard modifies the gap-state arm; pass --gap-state too')
    reentry_variant = ReentryGuardVariant(args.reentry_guard) if args.reentry_guard is not None else None

    if args.serve_start is not None and args.quiet_start is not None:
        parser.error('--serve-start and --quiet-start both own _find_rally_spans; pass at most one')
    if args.serve_start is not None and args.serve_start_pose_dir is None:
        parser.error('--serve-start needs --serve-start-pose-dir to precompute the gate distance array')
    if args.serve_start_wideshot and args.serve_start is None:
        parser.error('--serve-start-wideshot refines the serve-start arm; pass --serve-start too')
    if args.serve_start_split is not None and args.serve_start is None:
        parser.error('--serve-start-split refines the serve-start arm; pass --serve-start too')
    serve_start_mode = ServeStartMode(args.serve_start) if args.serve_start is not None else None
    serve_start_close = ServeStartClose(args.serve_start_split) if args.serve_start_split is not None else None

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
    if args.gap_state is not None:
        print(f'Gap-state on: gap-state _rest_mask at demotion_bound {args.gap_state} (installed per worker)')
    if args.quiet_start is not None:
        print(f'Quiet-start on: quiet-start _find_rally_spans at W {args.quiet_start} (installed per worker)')
    if reentry_variant is not None:
        print(f'Re-entry guard on: {reentry_variant.value} variant at buffer {args.reentry_buffer}')

    # Point the serve-start court geometry at the chosen box before the gate arrays build
    # below (both build in this process). Standin is the default no-op; homography swaps the
    # x/y outline and the half-split together, leaving the height band as the stand-in.
    install_court_box(args.court_box)
    if args.court_box != 'standin':
        print(f'Court box: {args.court_box} (foot-point filter uses the homography quad, half-split '
              f'the buffered net-line band {PILOT_HOMOG_COURT_MID_BAND}; height band stays the stand-in)')

    # Serve-start needs the per-frame gate distance array (and, with the wide-shot
    # refinement on, the per-frame count + per-half feet). Build them once here from the raw
    # pose boxes and pass them into every worker; None (and no build) when the arm is off.
    serve_start_dist: np.ndarray | None = None
    serve_start_wideshot: WideshotInputs | None = None
    if serve_start_mode is not None:
        pose_dir = args.serve_start_pose_dir
        bboxes = np.load(pose_dir / f'{args.serve_start_pose_prefix}_bboxes.npy')
        scores = np.load(pose_dir / f'{args.serve_start_pose_prefix}_scores.npy')
        serve_start_dist = build_serve_start_dist(track, bboxes, scores)
        print(f'Serve-start on: {serve_start_mode.value} mode, gate <= {args.serve_start_threshold} '
              f'over the last {SERVE_START_LOOKBACK_FRAMES}f; distance array built from {pose_dir}')
        if args.serve_start_wideshot:
            serve_start_wideshot = build_serve_start_wideshot_inputs(bboxes, scores)
            print('Wide-shot refinement on: qualifying bursts also need count_med >= '
                  f'{WIDESHOT_COUNT_MED_MIN}, both halves occupied, drift <= {WIDESHOT_DRIFT_MAX}')
        if serve_start_close is not None:
            print(f'Serve-start split on: close at {serve_start_close.value} '
                  f'(each qualifying burst opens a span)')

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
        boundary_rows = run_phase(tasks, ctx, args.workers, args.nan_smoothing,
                                  args.gap_state, args.quiet_start,
                                  reentry_variant, args.reentry_buffer,
                                  serve_start_mode, args.serve_start_threshold, serve_start_dist,
                                  serve_start_wideshot, serve_start_close)
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
        contact_rows = run_phase(tasks, ctx, args.workers, args.nan_smoothing,
                                 args.gap_state, args.quiet_start,
                                 reentry_variant, args.reentry_buffer,
                                 serve_start_mode, args.serve_start_threshold, serve_start_dist,
                                 serve_start_wideshot, serve_start_close)
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
