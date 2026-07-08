"""Tests for the stage-8 threshold-sweep runner.

Synthetic only, no real track or GT files. Covers the four load-bearing pieces
plus a micro end-to-end that drives the real ``segment_video`` and scorer:

  1. grid construction counts (324 / 45, and the extra shipped-defaults task)
  2. deterministic winner selection at every tie-break level
  3. module patching actually changes ``segment_video`` behaviour
  4. row flattening survives None metrics (an empty covered set)
  5. plumbing: a tiny track + two configs through patch -> segment -> score -> row
"""
import numpy as np

from scripts.stage8_score import GtRally, score_stage8
from scripts.stage8_sweep import (
    LABEL_GRID,
    LABEL_SHIPPED,
    PARAM_COLUMNS,
    ROW_COLUMNS,
    SHIPPED_DEFAULTS,
    SWEEP_TOLERANCES,
    Stage8Params,
    SweepCtx,
    SweepTask,
    _init_worker,
    _patch_stage8,
    _score_config,
    _serialise_row,
    boundary_sort_key,
    build_boundary_grid,
    build_boundary_tasks,
    build_contact_grid,
    build_contact_tasks,
    contact_frontier,
    contact_sort_key,
    flatten_row,
    select_winner,
    stage8_module,
)


# ---------------------------------------------------------------------------
# Row helper: a baseline row with all-neutral metrics, overridable per test
# ---------------------------------------------------------------------------
def _mk_row(label: str = LABEL_GRID, params: Stage8Params = SHIPPED_DEFAULTS, **metrics) -> dict:
    """A well-formed row: given params and label, neutral metrics, then overrides."""
    row: dict = {'label': label}
    for field in PARAM_COLUMNS:
        row[field] = getattr(params, field)
    row.update({
        'n_spans': 0, 'covered': 0, 'covered_fraction': 0.0, 'split': 0,
        'missed': 0, 'merged_spans': 0, 'spurious_spans': 0,
        'start_alignment_mean': None, 'start_alignment_median': None,
        'count_gate_covered_fraction': None, 'count_gate_unmerged_fraction': None,
        'total_candidates': 0,
    })
    for tolerance in SWEEP_TOLERANCES:
        row[f'recall_{tolerance}'] = None
        row[f'precision_{tolerance}'] = None
        row[f'f1_{tolerance}'] = None
    row.update(metrics)
    return row


# ---------------------------------------------------------------------------
# 1. Grid construction
# ---------------------------------------------------------------------------
def test_boundary_grid_count_and_pinned_contacts():
    grid = build_boundary_grid()
    assert len(grid) == 324
    assert len(set(grid)) == 324  # every config distinct
    # The three contact thresholds are held at shipped defaults across the grid.
    for params in grid:
        assert params.smooth_window == SHIPPED_DEFAULTS.smooth_window
        assert params.min_dir_change_deg == SHIPPED_DEFAULTS.min_dir_change_deg
        assert params.min_contact_speed == SHIPPED_DEFAULTS.min_contact_speed


def test_contact_grid_count_and_frozen_boundary():
    winner = SHIPPED_DEFAULTS._replace(rest_window=21, start_speed=0.05, end_rest_frames=45)
    grid = build_contact_grid(winner)
    assert len(grid) == 45
    assert len(set(grid)) == 45
    # The five boundary thresholds are frozen at the supplied winner.
    for params in grid:
        assert params.rest_speed == winner.rest_speed
        assert params.rest_window == winner.rest_window
        assert params.end_rest_frames == winner.end_rest_frames
        assert params.start_speed == winner.start_speed
        assert params.start_min_frames == winner.start_min_frames


def test_task_lists_add_one_shipped_defaults_row():
    boundary_tasks = build_boundary_tasks()
    assert len(boundary_tasks) == 325  # 324 grid + 1 defaults
    assert sum(1 for task in boundary_tasks if task.label == LABEL_SHIPPED) == 1

    contact_tasks = build_contact_tasks(SHIPPED_DEFAULTS)
    assert len(contact_tasks) == 46  # 45 grid + 1 defaults
    assert sum(1 for task in contact_tasks if task.label == LABEL_SHIPPED) == 1

    defaults_task = next(task for task in boundary_tasks if task.label == LABEL_SHIPPED)
    assert defaults_task.params == SHIPPED_DEFAULTS


# ---------------------------------------------------------------------------
# 2. Winner selection: every tie-break level, deterministic
# ---------------------------------------------------------------------------
def test_boundary_winner_maximises_covered_fraction():
    low = _mk_row(covered_fraction=0.5)
    high = _mk_row(covered_fraction=0.8, params=SHIPPED_DEFAULTS._replace(rest_window=9))
    assert select_winner([low, high], boundary_sort_key) is high


def test_boundary_winner_tiebreak_split():
    more = _mk_row(covered_fraction=0.8, split=3, params=SHIPPED_DEFAULTS._replace(rest_window=9))
    less = _mk_row(covered_fraction=0.8, split=1, params=SHIPPED_DEFAULTS._replace(rest_window=21))
    assert select_winner([more, less], boundary_sort_key) is less


def test_boundary_winner_tiebreak_spurious_then_merged():
    # Covered and split tie; fewer spurious spans wins.
    more_spurious = _mk_row(covered_fraction=0.8, spurious_spans=4,
                            params=SHIPPED_DEFAULTS._replace(rest_window=9))
    less_spurious = _mk_row(covered_fraction=0.8, spurious_spans=1,
                            params=SHIPPED_DEFAULTS._replace(rest_window=21))
    assert select_winner([more_spurious, less_spurious], boundary_sort_key) is less_spurious

    # Covered, split, spurious all tie; fewer merged spans wins.
    more_merged = _mk_row(covered_fraction=0.8, merged_spans=2,
                          params=SHIPPED_DEFAULTS._replace(rest_window=9))
    less_merged = _mk_row(covered_fraction=0.8, merged_spans=0,
                          params=SHIPPED_DEFAULTS._replace(rest_window=21))
    assert select_winner([more_merged, less_merged], boundary_sort_key) is less_merged


def test_boundary_winner_tiebreak_closest_to_defaults():
    # All metrics tie; the config that changed fewer params from shipped wins.
    far = _mk_row(covered_fraction=0.8, params=SHIPPED_DEFAULTS._replace(rest_window=9, start_speed=0.05))
    near = _mk_row(covered_fraction=0.8, params=SHIPPED_DEFAULTS._replace(rest_window=9))
    assert select_winner([far, near], boundary_sort_key) is near


def test_boundary_winner_param_tuple_is_final_deterministic_tiebreak():
    # Same metrics, same changed-count (both change one param): the param tuple
    # settles it, and the choice does not depend on input order.
    a = _mk_row(covered_fraction=0.8, params=SHIPPED_DEFAULTS._replace(rest_speed=0.005))
    b = _mk_row(covered_fraction=0.8, params=SHIPPED_DEFAULTS._replace(rest_window=9))
    assert select_winner([a, b], boundary_sort_key) is select_winner([b, a], boundary_sort_key)


def test_select_winner_excludes_shipped_defaults_reference():
    # The reference has the best metrics but must never be picked as the winner.
    reference = _mk_row(label=LABEL_SHIPPED, covered_fraction=0.99)
    grid = _mk_row(label=LABEL_GRID, covered_fraction=0.70,
                   params=SHIPPED_DEFAULTS._replace(rest_window=9))
    assert select_winner([reference, grid], boundary_sort_key) is grid


def test_contact_winner_maximises_recall_then_falls_through_tiebreaks():
    # Recall-first at +/-5 (Ariel 2026-07-08): higher +/-5 recall wins outright.
    low = _mk_row(recall_5=0.40)
    high = _mk_row(recall_5=0.70, params=SHIPPED_DEFAULTS._replace(smooth_window=3))
    assert select_winner([low, high], contact_sort_key) is high

    # The flip: higher +/-5 recall beats higher +/-2 F1. Under the old F1-first
    # rule the low-recall row would have won.
    high_f1 = _mk_row(recall_5=0.40, f1_2=0.90, params=SHIPPED_DEFAULTS._replace(smooth_window=3))
    high_recall = _mk_row(recall_5=0.70, f1_2=0.50, params=SHIPPED_DEFAULTS._replace(smooth_window=7))
    assert select_winner([high_f1, high_recall], contact_sort_key) is high_recall

    # +/-5 recall ties; higher +/-5 F1 wins.
    worse_5 = _mk_row(recall_5=0.70, f1_5=0.60, params=SHIPPED_DEFAULTS._replace(smooth_window=3))
    better_5 = _mk_row(recall_5=0.70, f1_5=0.85, params=SHIPPED_DEFAULTS._replace(smooth_window=7))
    assert select_winner([worse_5, better_5], contact_sort_key) is better_5

    # +/-5 recall and F1 tie; higher unmerged count-gate pass rate wins.
    worse_gate = _mk_row(recall_5=0.70, f1_5=0.85, count_gate_unmerged_fraction=0.3,
                         params=SHIPPED_DEFAULTS._replace(smooth_window=3))
    better_gate = _mk_row(recall_5=0.70, f1_5=0.85, count_gate_unmerged_fraction=0.9,
                          params=SHIPPED_DEFAULTS._replace(smooth_window=7))
    assert select_winner([worse_gate, better_gate], contact_sort_key) is better_gate


# ---------------------------------------------------------------------------
# 2b. Contact Pareto frontier data package
# ---------------------------------------------------------------------------
def test_contact_frontier_pareto_extraction():
    # High recall / low precision and low recall / high precision: neither
    # dominates the other, so both sit on the frontier.
    high_recall = _mk_row(recall_5=0.9, precision_5=0.3, params=SHIPPED_DEFAULTS._replace(smooth_window=3))
    high_precision = _mk_row(recall_5=0.4, precision_5=0.9, params=SHIPPED_DEFAULTS._replace(smooth_window=7))
    # Beaten by high_recall on both axes (0.5<0.9 recall, 0.2<0.3 precision).
    dominated = _mk_row(recall_5=0.5, precision_5=0.2,
                        params=SHIPPED_DEFAULTS._replace(min_dir_change_deg=30))
    # No +/-5 candidates: recall_5 None, off the frontier by definition (skipped).
    no_candidates = _mk_row(recall_5=None, precision_5=0.99,
                            params=SHIPPED_DEFAULTS._replace(min_dir_change_deg=90))
    # Shipped-defaults reference must never appear, even at a perfect (1.0, 1.0).
    reference = _mk_row(label=LABEL_SHIPPED, recall_5=1.0, precision_5=1.0)

    frontier = contact_frontier([high_recall, high_precision, dominated, no_candidates, reference])
    # Only the two non-dominated grid configs survive, recall_5 descending.
    assert [row['recall_5'] for row in frontier] == [0.9, 0.4]
    assert all(row['label'] == LABEL_GRID for row in frontier)


# ---------------------------------------------------------------------------
# 3. Module patching actually changes segment_video behaviour
# ---------------------------------------------------------------------------
def _burst_track() -> np.ndarray:
    """Rest, then a visible burst at ~0.025/frame, then rest.

    The burst speed sits between a lowered START_SPEED (0.02) and the shipped one
    (0.03), so a rally span forms only under the lower threshold. Long rests on
    both sides (>= END_REST_FRAMES) isolate the burst as its own active region.
    """
    rest_pre, burst, rest_post = 40, 20, 40
    burst_step = 0.025
    xs = [0.5] * rest_pre
    position = 0.5
    for _ in range(burst):
        position += burst_step
        xs.append(position)
    xs += [position] * rest_post
    xs_arr = np.array(xs)
    ys = np.full_like(xs_arr, 0.5)
    vis = np.ones_like(xs_arr)
    return np.column_stack([xs_arr, ys, vis])


def test_patch_changes_segment_video_behaviour():
    track = _burst_track()

    # Shipped START_SPEED (0.03): the 0.025 burst never reads as fast -> no span.
    _patch_stage8(SHIPPED_DEFAULTS)
    spans_shipped, _ = stage8_module.segment_video(track)
    assert spans_shipped == []

    # Lower START_SPEED to 0.02: the same burst now qualifies -> a span forms.
    lowered = SHIPPED_DEFAULTS._replace(start_speed=0.02)
    _patch_stage8(lowered)
    spans_lowered, _ = stage8_module.segment_video(track)
    assert len(spans_lowered) >= 1

    # Restore shipped defaults; the module global is shared within this process.
    _patch_stage8(SHIPPED_DEFAULTS)


# ---------------------------------------------------------------------------
# 4. Row flattening survives None metrics (empty covered set)
# ---------------------------------------------------------------------------
def test_flatten_row_handles_none_metrics():
    # A rally with no detected spans: nothing covered, no candidates. The scorer
    # returns None for start_alignment, the count-gate fraction, precision and f1.
    gt_rallies = [GtRally(set_id='set1', rally=1, stroke_frames=(100, 102))]
    metrics = score_stage8(spans=[], contacts=[], gt_rallies=gt_rallies, tolerances=SWEEP_TOLERANCES)

    row = flatten_row(LABEL_GRID, SHIPPED_DEFAULTS, n_spans=0, metrics=metrics)
    assert row['covered'] == 0
    assert row['covered_fraction'] == 0.0
    assert row['missed'] == 1
    assert row['start_alignment_mean'] is None
    assert row['start_alignment_median'] is None
    assert row['count_gate_covered_fraction'] is None
    # Every scored band is None here (no candidates), the outer +/-1 / +/-10 included.
    for tolerance in SWEEP_TOLERANCES:
        assert row[f'precision_{tolerance}'] is None
        assert row[f'f1_{tolerance}'] is None
    assert row['total_candidates'] == 0

    # Every None serialises to a blank cell, and the row keys the full column set.
    serial = _serialise_row(row)
    assert set(serial) == set(ROW_COLUMNS)
    assert serial['start_alignment_mean'] == ''
    assert serial['f1_1'] == ''
    assert serial['f1_10'] == ''
    assert serial['count_gate_covered_fraction'] == ''


# ---------------------------------------------------------------------------
# 5. Plumbing: tiny track + two configs through the real segment_video + scorer
# ---------------------------------------------------------------------------
def _rally_track_and_gt() -> tuple[np.ndarray, list[GtRally]]:
    """Rest + a three-reversal rally + long rest, with GT over the rally strokes.

    Mirrors tests/test_scraper_stage8.py's construction: shipped defaults find
    one span and its reversals, so the plumbing exercises boundary and contact
    scoring both.
    """
    rest_y, step, rest_pre, rest_post = 0.01, 0.14, 45, 60
    up = np.round(np.arange(rest_y, 0.99 + step / 2, step), 4)  # 0.01 .. ~0.99
    down = up[::-1]
    path = np.concatenate([up, down[1:], up[1:], down[1:]])  # three reversals at the apexes
    apex_local = [
        len(up) - 1,
        len(up) - 1 + len(down[1:]),
        len(up) - 1 + len(down[1:]) + len(up[1:]),
    ]
    rally_y = path[1:]  # drop the leading REST_Y so it seams with the rest
    ys = np.concatenate([np.full(rest_pre, rest_y), rally_y, np.full(rest_post, rest_y)])
    xs = np.full_like(ys, 0.5)
    vis = np.ones_like(ys)
    track = np.column_stack([xs, ys, vis])

    contact_frames = tuple(rest_pre + (local - 1) for local in apex_local)  # -1: rally_y dropped path[0]
    gt_rallies = [GtRally(set_id='set1', rally=1, stroke_frames=contact_frames)]
    return track, gt_rallies


def test_plumbing_two_configs_end_to_end():
    track, gt_rallies = _rally_track_and_gt()
    _init_worker(SweepCtx(track=track, gt_rallies=gt_rallies))

    shipped_row = _score_config(SweepTask(LABEL_SHIPPED, SHIPPED_DEFAULTS))
    assert shipped_row['n_spans'] >= 1
    assert shipped_row['covered'] == 1
    assert shipped_row['recall_5'] is not None and shipped_row['recall_5'] > 0

    # A very high MIN_CONTACT_SPEED gate suppresses every reversal, so the row
    # still forms but emits no candidates; the boundary span is unaffected.
    strict = SHIPPED_DEFAULTS._replace(min_contact_speed=0.5)
    strict_row = _score_config(SweepTask(LABEL_GRID, strict))
    assert strict_row['n_spans'] == shipped_row['n_spans']
    assert strict_row['total_candidates'] == 0
    assert strict_row['total_candidates'] <= shipped_row['total_candidates']

    # Both rows carry the full column set and serialise cleanly.
    assert set(_serialise_row(shipped_row)) == set(ROW_COLUMNS)
    assert set(_serialise_row(strict_row)) == set(ROW_COLUMNS)

    _patch_stage8(SHIPPED_DEFAULTS)  # leave the shared module at defaults
