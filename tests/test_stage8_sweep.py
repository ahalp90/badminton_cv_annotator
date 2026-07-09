"""Tests for the stage-8 threshold-sweep runner.

Synthetic only, no real track or GT files. Covers the four load-bearing pieces
plus a micro end-to-end that drives the real ``segment_video`` and scorer:

  1. grid construction counts (3,000 / 45, and the extra shipped-defaults task)
  2. deterministic winner selection at every tie-break level
  3. module patching actually changes ``segment_video`` behaviour
  4. row flattening survives None metrics (an empty covered set)
  5. plumbing: a tiny track + two configs through patch -> segment -> score -> row
"""
import csv

import numpy as np
import pytest

import scripts.stage8_sweep as stage8_sweep
from scripts.stage8_score import GtRally, score_stage8
from scripts.stage8_sweep import (
    BOUNDARY_END_REST_FRAMES,
    BOUNDARY_REST_SPEED,
    BOUNDARY_REST_WINDOW,
    BOUNDARY_START_MIN_FRAMES,
    BOUNDARY_START_SPEED,
    CROWN_KEY_COLUMN,
    LABEL_GRID,
    LABEL_SHIPPED,
    PARAM_COLUMNS,
    RAW_PRECISION_COLUMNS,
    ROW_COLUMNS,
    SERVE_START_LOOKBACK_FRAMES,
    SHIPPED_DEFAULTS,
    SWEEP_TOLERANCES,
    ReentryGuardVariant,
    ServeStartMode,
    Stage8Params,
    SweepCtx,
    SweepTask,
    WideshotInputs,
    _STOCK_FIND_RALLY_SPANS,
    _STOCK_REST_MASK,
    _gap_holds_open,
    _gap_is_high_shot_oob,
    _gap_passes_reentry_guard,
    _init_worker,
    _is_quiet_before,
    _patch_stage8,
    _reentry_velocity_y,
    _score_config,
    _serialise_row,
    _serve_setup_before,
    _wide_shot_before,
    apply_replay_mask,
    boundary_sort_key_as_built,
    build_boundary_crowns,
    build_boundary_grid,
    build_boundary_tasks,
    build_contact_grid,
    build_contact_tasks,
    build_serve_start_wideshot_inputs,
    contact_frontier,
    contact_sort_key,
    flatten_row,
    gap_state_rest_mask,
    install_gap_state,
    install_nan_smoothing,
    install_quiet_start,
    install_serve_start,
    nan_rolling_mean,
    nan_smoothing_detect_contacts,
    quiet_start_find_rally_spans,
    select_boundary_winner,
    select_start_alignment_winner,
    select_winner,
    serve_start_find_rally_spans,
    stage8_module,
    write_crowns_csv,
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
        row[f'precision_raw_{tolerance}'] = None
    row.update(metrics)
    return row


# ---------------------------------------------------------------------------
# 1. Grid construction
# ---------------------------------------------------------------------------
def test_boundary_grid_count_and_pinned_contacts():
    grid = build_boundary_grid()
    # Widened grid (Ariel 2026-07-08): 5 x 5 x 6 x 5 x 4 per-param sizes.
    assert len(grid) == 3000
    assert len(set(grid)) == 3000  # every config distinct
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
    assert len(boundary_tasks) == 3001  # 3,000 grid + 1 defaults
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
    assert select_winner([low, high], boundary_sort_key_as_built) is high


def test_boundary_winner_tiebreak_split():
    more = _mk_row(covered_fraction=0.8, split=3, params=SHIPPED_DEFAULTS._replace(rest_window=9))
    less = _mk_row(covered_fraction=0.8, split=1, params=SHIPPED_DEFAULTS._replace(rest_window=21))
    assert select_winner([more, less], boundary_sort_key_as_built) is less


def test_boundary_winner_tiebreak_spurious_then_merged():
    # Covered and split tie; fewer spurious spans wins.
    more_spurious = _mk_row(covered_fraction=0.8, spurious_spans=4,
                            params=SHIPPED_DEFAULTS._replace(rest_window=9))
    less_spurious = _mk_row(covered_fraction=0.8, spurious_spans=1,
                            params=SHIPPED_DEFAULTS._replace(rest_window=21))
    assert select_winner([more_spurious, less_spurious], boundary_sort_key_as_built) is less_spurious

    # Covered, split, spurious all tie; fewer merged spans wins.
    more_merged = _mk_row(covered_fraction=0.8, merged_spans=2,
                          params=SHIPPED_DEFAULTS._replace(rest_window=9))
    less_merged = _mk_row(covered_fraction=0.8, merged_spans=0,
                          params=SHIPPED_DEFAULTS._replace(rest_window=21))
    assert select_winner([more_merged, less_merged], boundary_sort_key_as_built) is less_merged


def test_boundary_winner_tiebreak_closest_to_defaults():
    # All metrics tie; the config that changed fewer params from shipped wins.
    far = _mk_row(covered_fraction=0.8, params=SHIPPED_DEFAULTS._replace(rest_window=9, start_speed=0.05))
    near = _mk_row(covered_fraction=0.8, params=SHIPPED_DEFAULTS._replace(rest_window=9))
    assert select_winner([far, near], boundary_sort_key_as_built) is near


def test_boundary_winner_param_tuple_is_final_deterministic_tiebreak():
    # Same metrics, same changed-count (both change one param): the param tuple
    # settles it, and the choice does not depend on input order.
    a = _mk_row(covered_fraction=0.8, params=SHIPPED_DEFAULTS._replace(rest_speed=0.005))
    b = _mk_row(covered_fraction=0.8, params=SHIPPED_DEFAULTS._replace(rest_window=9))
    assert select_winner([a, b], boundary_sort_key_as_built) is select_winner([b, a], boundary_sort_key_as_built)


def test_select_winner_excludes_shipped_defaults_reference():
    # The reference has the best metrics but must never be picked as the winner.
    reference = _mk_row(label=LABEL_SHIPPED, covered_fraction=0.99)
    grid = _mk_row(label=LABEL_GRID, covered_fraction=0.70,
                   params=SHIPPED_DEFAULTS._replace(rest_window=9))
    assert select_winner([reference, grid], boundary_sort_key_as_built) is grid


# ---------------------------------------------------------------------------
# 2b. Merge-penalised and start-alignment boundary selection (Ariel 2026-07-08)
# ---------------------------------------------------------------------------
def test_select_boundary_winner_prefers_fewer_merges_within_allowance():
    # Coverage maximum, but bought with glue: 3 merged spans.
    glued = _mk_row(covered=20, merged_spans=3, params=SHIPPED_DEFAULTS._replace(rest_window=9))
    # Two covered rallies short of the max (inside the +/-2 allowance), no merges: wins.
    clean = _mk_row(covered=18, merged_spans=0, params=SHIPPED_DEFAULTS._replace(rest_window=21))
    # Three short (outside the allowance) and the shipped defaults themselves, so
    # it would win every tie-break if eligible; the coverage gate is the only
    # reason it loses.
    too_low = _mk_row(covered=17, merged_spans=0)
    assert select_boundary_winner([glued, clean, too_low]) is clean


def test_select_start_alignment_winner_uses_abs_median_none_worst():
    # abs() ranking: a -3 median beats a +5 (tighter magnitude) ...
    neg3 = _mk_row(covered=20, start_alignment_median=-3.0, params=SHIPPED_DEFAULTS._replace(rest_window=9))
    pos5 = _mk_row(covered=20, start_alignment_median=5.0, params=SHIPPED_DEFAULTS._replace(rest_window=21))
    assert select_start_alignment_winner([neg3, pos5]) is neg3

    # ... but loses to a +1 (smaller magnitude still).
    pos1 = _mk_row(covered=20, start_alignment_median=1.0, params=SHIPPED_DEFAULTS._replace(rest_window=7))
    assert select_start_alignment_winner([neg3, pos1]) is pos1

    # None median (no covered rally to measure) sorts worst, even against a big
    # magnitude carrying more merges.
    none_median = _mk_row(covered=20, start_alignment_median=None, merged_spans=0,
                          params=SHIPPED_DEFAULTS._replace(rest_window=9))
    measured = _mk_row(covered=20, start_alignment_median=8.0, merged_spans=5,
                       params=SHIPPED_DEFAULTS._replace(rest_window=21))
    assert select_start_alignment_winner([none_median, measured]) is measured


# ---------------------------------------------------------------------------
# 2c. Boundary crowns data package (build + CSV) and contact Pareto frontier
# ---------------------------------------------------------------------------
def _crowns_rows() -> list[dict]:
    """Five grid rows over covered levels 10/9/8/6, each a distinct rest_window.

    Chosen so the three crowns diverge: as_built chases coverage (the cov-10 row
    with fewest merges), merge_penalised takes the clean cov-9, start_alignment
    the tight cov-8. rest_window is the per-row fingerprint the assertions read.
    """
    glued_max = _mk_row(covered=10, covered_fraction=1.0, merged_spans=3,
                        start_alignment_median=6.0, params=SHIPPED_DEFAULTS._replace(rest_window=9))
    clean_max = _mk_row(covered=10, covered_fraction=1.0, merged_spans=1,
                        start_alignment_median=3.0, params=SHIPPED_DEFAULTS._replace(rest_window=7))
    clean_high = _mk_row(covered=9, covered_fraction=0.9, merged_spans=0,
                         start_alignment_median=2.0, params=SHIPPED_DEFAULTS._replace(rest_window=15))
    aligned_mid = _mk_row(covered=8, covered_fraction=0.8, merged_spans=0, split=1,
                          start_alignment_median=1.0, params=SHIPPED_DEFAULTS._replace(rest_window=21))
    low_cov = _mk_row(covered=6, covered_fraction=0.6, merged_spans=0,
                      start_alignment_median=4.0, params=SHIPPED_DEFAULTS._replace(rest_window=5))
    return [glued_max, clean_max, clean_high, aligned_mid, low_cov]


# (crown_key, winning row's rest_window fingerprint), in the order build emits.
_EXPECTED_CROWNS = [
    ('as_built', 7),
    ('merge_penalised', 15),
    ('start_alignment_penalised', 21),
    ('frontier_cov_10', 7),
    ('frontier_cov_9', 15),
    ('frontier_cov_8', 21),
    ('frontier_cov_6', 5),
]


def test_build_boundary_crowns_labels_and_winners():
    crowns = build_boundary_crowns(_crowns_rows())
    assert [(crown[CROWN_KEY_COLUMN], crown['rest_window']) for crown in crowns] == _EXPECTED_CROWNS


def test_write_crowns_csv_lands_with_crown_key_column(tmp_path):
    path = tmp_path / 'boundary_crowns.csv'
    write_crowns_csv(path, _crowns_rows())
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None and reader.fieldnames[0] == CROWN_KEY_COLUMN
        assert set(reader.fieldnames) == {CROWN_KEY_COLUMN, *ROW_COLUMNS}
        read_rows = list(reader)
    landed = [(row[CROWN_KEY_COLUMN], int(row['rest_window'])) for row in read_rows]
    assert landed == _EXPECTED_CROWNS


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


def test_contact_frontier_equal_axis_domination():
    """Pins the adversarial-review fix: dominance is >= on both axes and > on at
    least one, so equal-on-one-axis-worse-on-the-other is still dominated, while
    exact duplicates don't dominate each other and both stay."""
    best = _mk_row(recall_5=0.9, precision_5=0.9, params=SHIPPED_DEFAULTS._replace(smooth_window=3))
    # Equal recall, strictly worse precision: dominated (the strict-both bug kept it).
    equal_recall_worse_precision = _mk_row(recall_5=0.9, precision_5=0.5,
                                           params=SHIPPED_DEFAULTS._replace(smooth_window=5))
    # Exact duplicate of best on both axes: neither dominates, both stay.
    duplicate = _mk_row(recall_5=0.9, precision_5=0.9,
                        params=SHIPPED_DEFAULTS._replace(smooth_window=7))

    frontier = contact_frontier([best, equal_recall_worse_precision, duplicate])
    assert [row['precision_5'] for row in frontier] == [0.9, 0.9]
    assert {row['smooth_window'] for row in frontier} == {3, 7}


# ---------------------------------------------------------------------------
# 1b. Widened boundary grid: per-param sizes and the 3,000 product
# ---------------------------------------------------------------------------
def test_boundary_grid_param_sizes_and_product():
    sizes = (
        len(BOUNDARY_REST_SPEED),
        len(BOUNDARY_REST_WINDOW),
        len(BOUNDARY_END_REST_FRAMES),
        len(BOUNDARY_START_SPEED),
        len(BOUNDARY_START_MIN_FRAMES),
    )
    assert sizes == (5, 5, 6, 5, 4)
    product = 1
    for size in sizes:
        product *= size
    assert product == 3000
    # Every param stays ascending after the widening.
    for values in (BOUNDARY_REST_SPEED, BOUNDARY_REST_WINDOW, BOUNDARY_END_REST_FRAMES,
                   BOUNDARY_START_SPEED, BOUNDARY_START_MIN_FRAMES):
        assert list(values) == sorted(values)


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
    _init_worker(SweepCtx(track=track, gt_rallies=gt_rallies), nan_smoothing=False)

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


# ---------------------------------------------------------------------------
# 6. Optional replay-mask track transform (freeze masked frames to rest)
# ---------------------------------------------------------------------------
def _distinct_track(n_frames: int) -> np.ndarray:
    """A (n, 3) track with a unique xy per frame and vis 0 everywhere.

    Distinct xy lets a test read exactly which frame a masked run froze to; vis 0
    everywhere means a forced vis 1 is unambiguous evidence of masking.
    """
    xs = np.arange(n_frames, dtype=float) * 0.1
    ys = np.arange(n_frames, dtype=float) * 0.01 + 0.5
    vis = np.zeros(n_frames, dtype=float)
    return np.column_stack([xs, ys, vis])


def test_apply_replay_mask_mid_run_freezes_to_preceding_frame():
    track = _distinct_track(6)
    original = track.copy()
    mask = np.array([False, False, True, True, False, False])

    frozen = apply_replay_mask(track, mask)
    # Frames 2, 3 take frame 1's xy (the last live frame before the run); vis -> 1.
    assert np.array_equal(frozen[2, :2], track[1, :2])
    assert np.array_equal(frozen[3, :2], track[1, :2])
    assert frozen[2, 2] == 1.0 and frozen[3, 2] == 1.0
    # Pure: the source track is untouched.
    assert np.array_equal(track, original)


def test_apply_replay_mask_run_at_frame_zero_freezes_to_first_post_run_frame():
    track = _distinct_track(6)
    mask = np.array([True, True, False, False, False, False])

    frozen = apply_replay_mask(track, mask)
    # No frame before the run, so frames 0, 1 take frame 2's xy (first live after).
    assert np.array_equal(frozen[0, :2], track[2, :2])
    assert np.array_equal(frozen[1, :2], track[2, :2])
    assert frozen[0, 2] == 1.0 and frozen[1, 2] == 1.0


def test_apply_replay_mask_two_runs_each_freeze_to_own_predecessor():
    track = _distinct_track(8)
    mask = np.array([False, True, False, False, True, True, False, False])

    frozen = apply_replay_mask(track, mask)
    # First run (frame 1) anchors to frame 0.
    assert np.array_equal(frozen[1, :2], track[0, :2])
    # Second run (frames 4, 5) anchors to frame 3, its own predecessor.
    assert np.array_equal(frozen[4, :2], track[3, :2])
    assert np.array_equal(frozen[5, :2], track[3, :2])
    assert frozen[1, 2] == 1.0 and frozen[4, 2] == 1.0 and frozen[5, 2] == 1.0


def test_apply_replay_mask_all_false_returns_bit_identical():
    track = _distinct_track(5)
    mask = np.zeros(5, dtype=bool)
    assert np.array_equal(apply_replay_mask(track, mask), track)


def test_apply_replay_mask_length_mismatch_raises():
    track = _distinct_track(5)
    with pytest.raises(ValueError):
        apply_replay_mask(track, np.zeros(4, dtype=bool))


def test_apply_replay_mask_all_true_raises():
    track = _distinct_track(5)
    with pytest.raises(ValueError):
        apply_replay_mask(track, np.ones(5, dtype=bool))


def test_apply_replay_mask_leaves_unmasked_frames_untouched():
    track = _distinct_track(6)
    mask = np.array([False, False, True, True, False, False])

    frozen = apply_replay_mask(track, mask)
    # Unmasked frames keep their original xy and their original vis (0).
    for frame in (0, 1, 4, 5):
        assert np.array_equal(frozen[frame], track[frame])
        assert frozen[frame, 2] == 0.0


# ---------------------------------------------------------------------------
# 7. Optional NaN-smoothing patch (exclude invisible frames from contact smooth)
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _restore_stage8_smoothing():
    """Snapshot and restore the four patchable stage-8 globals around every test.

    Every arm (``install_nan_smoothing``, ``install_gap_state``,
    ``install_quiet_start``) mutates ``stage8_module`` in place, the same
    module-global patching the sweep uses, so without this a test that installs a
    patch would leak it into later tests and other files. Restores on pass and on
    failure (the teardown runs after the yield regardless).
    """
    saved_rolling = stage8_module._rolling_mean
    saved_detect = stage8_module.detect_contacts
    saved_rest_mask = stage8_module._rest_mask
    saved_find_spans = stage8_module._find_rally_spans
    # The gap-state / guard and serve-start per-process state live on the sweep module,
    # not stage8_module, so snapshot them too or a test leaks its buffer/variant/threshold.
    saved_demotion = stage8_sweep._GAP_STATE_DEMOTION_BOUND
    saved_guard_variant = stage8_sweep._REENTRY_GUARD_VARIANT
    saved_guard_buffer = stage8_sweep._REENTRY_GUARD_BUFFER
    saved_serve_dist = stage8_sweep._SERVE_START_DIST
    saved_serve_threshold = stage8_sweep._SERVE_START_THRESHOLD
    saved_serve_mode = stage8_sweep._SERVE_START_MODE
    saved_serve_wideshot = stage8_sweep._SERVE_START_WIDESHOT
    yield
    stage8_module._rolling_mean = saved_rolling
    stage8_module.detect_contacts = saved_detect
    stage8_module._rest_mask = saved_rest_mask
    stage8_module._find_rally_spans = saved_find_spans
    stage8_sweep._GAP_STATE_DEMOTION_BOUND = saved_demotion
    stage8_sweep._REENTRY_GUARD_VARIANT = saved_guard_variant
    stage8_sweep._REENTRY_GUARD_BUFFER = saved_guard_buffer
    stage8_sweep._SERVE_START_DIST = saved_serve_dist
    stage8_sweep._SERVE_START_THRESHOLD = saved_serve_threshold
    stage8_sweep._SERVE_START_MODE = saved_serve_mode
    stage8_sweep._SERVE_START_WIDESHOT = saved_serve_wideshot


def _gap_track() -> tuple[np.ndarray, np.ndarray]:
    """A straight-line shuttle track with a mid-rally invisibility gap.

    The shuttle really flies a monotone diagonal well away from (0, 0); frames
    18-22 are invisible and zero-filled (0, 0), as stage 8 stores an untracked
    frame. Returns the track and the true (ungapped) x so a test can measure how
    far each smoothing pulls the gap-adjacent frames off the real trajectory.
    """
    n = 40
    true_x = 0.2 + np.arange(n) * 0.015  # 0.2 .. ~0.785, all positive
    true_y = 0.3 + np.arange(n) * 0.01
    vis = np.ones(n)
    gap = slice(18, 23)  # five invisible frames mid-track
    xs, ys = true_x.copy(), true_y.copy()
    xs[gap] = 0.0
    ys[gap] = 0.0
    vis[gap] = 0.0
    return np.column_stack([xs, ys, vis]), true_x


def test_nan_rolling_mean_matches_stock_on_visible_track():
    # No invisible frames -> no NaN in the smoothing input, so the NaN-ignoring
    # mean must fall back to the stock convolution bit-for-bit (same arithmetic).
    rng = np.random.default_rng(0)
    values = rng.random(200)  # a fully-visible track's positions, no NaN
    for window in (3, 5, 7):
        stock = stage8_module._rolling_mean(values, window)
        patched = nan_rolling_mean(values, window)
        assert np.array_equal(stock, patched)


def test_nan_smoothing_survives_midrally_gap():
    # Stock smoothing averages the gap's zero-fill in and drags the smoothed x
    # toward the corner near the gap; NaN smoothing drops the gap and stays on the
    # real trajectory. window=5 (shipped SMOOTH_WINDOW); frames 17 and 23 straddle
    # the gap so their windows overlap the zero-fill.
    track, true_x = _gap_track()
    x_raw = track[:, 0]  # zero-filled at the gap, as stage 8 sees it
    stock_smooth = stage8_module._rolling_mean(x_raw, 5)

    x_nan = x_raw.copy()
    x_nan[track[:, 2] != 1] = np.nan  # what the patch does before smoothing
    nan_smooth = nan_rolling_mean(x_nan, 5)

    for frame in (17, 23):
        assert stock_smooth[frame] < true_x[frame] - 0.1  # dragged toward the corner
        assert abs(nan_smooth[frame] - true_x[frame]) < 0.03  # stays on the trajectory


def test_nan_smoothing_flag_off_leaves_stock_smoothing():
    # --nan-smoothing off: the initializer must not touch the module globals, so
    # the stock function objects stay bound.
    stock_rolling = stage8_module._rolling_mean
    stock_detect = stage8_module.detect_contacts
    _init_worker(SweepCtx(track=np.zeros((3, 3)), gt_rallies=[]), nan_smoothing=False)
    assert stage8_module._rolling_mean is stock_rolling
    assert stage8_module.detect_contacts is stock_detect


def test_nan_smoothing_flag_on_installs_patch():
    # --nan-smoothing on: the initializer installs both drop-ins in this process.
    _init_worker(SweepCtx(track=np.zeros((3, 3)), gt_rallies=[]), nan_smoothing=True)
    assert stage8_module._rolling_mean is nan_rolling_mean
    assert stage8_module.detect_contacts is nan_smoothing_detect_contacts


def test_install_nan_smoothing_is_idempotent():
    # Re-installing keeps the wrapper deferring to the import-time stock, never to
    # itself: two installs leave detect_contacts as the wrapper, not stacked.
    install_nan_smoothing()
    install_nan_smoothing()
    assert stage8_module.detect_contacts is nan_smoothing_detect_contacts
    # A clean straight all-visible track has no reversals, so the wrapper returns
    # no contacts and, crucially, the call completes: proof it didn't wrap itself
    # into a recursive double-smooth.
    n = 30
    straight = np.column_stack([
        0.2 + np.arange(n) * 0.015,
        0.3 + np.arange(n) * 0.01,
        np.ones(n),
    ])
    assert nan_smoothing_detect_contacts(straight, 0, n) == []


# ---------------------------------------------------------------------------
# 8. Raw per-candidate precision columns (Change 1: additive scorer + CSV columns)
# ---------------------------------------------------------------------------
def test_flatten_row_carries_raw_precision_columns():
    # A candidate pooled into two rallies must count ONCE in the raw precision. One
    # merged span holds two GT rallies and a lone candidate at frame 11 within +/-2 of
    # a stroke in each; a spurious span adds a never-matched candidate at 105. So the
    # raw denominator is the two physical candidates {11, 105} and the numerator the
    # one matched frame {11}: precision_raw = 0.5, not the pooled 1.0.
    rallies = [GtRally(set_id='set1', rally=1, stroke_frames=(10,)),
               GtRally(set_id='set1', rally=2, stroke_frames=(12,))]
    spans = [(5, 20), (100, 110)]
    contacts = [(0, 11, None), (1, 105, None)]
    metrics = score_stage8(spans, contacts, rallies, tolerances=SWEEP_TOLERANCES)

    row = flatten_row(LABEL_GRID, SHIPPED_DEFAULTS, n_spans=2, metrics=metrics)
    assert set(RAW_PRECISION_COLUMNS) <= set(ROW_COLUMNS)  # columns are declared
    assert set(RAW_PRECISION_COLUMNS) <= set(row)          # and populated on the row
    assert row['precision_raw_2'] == pytest.approx(0.5)
    # Pooled precision over the same input double-counts the shared candidate: the two
    # metrics genuinely differ, which is the point of the raw column.
    assert row['precision_2'] == pytest.approx(1.0)
    # The row still keys exactly the full column set after the additive columns.
    assert set(_serialise_row(row)) == set(ROW_COLUMNS)


# ---------------------------------------------------------------------------
# 9. Gap-state arm (Change 2: --gap-state _rest_mask patch)
# ---------------------------------------------------------------------------
def _high_shot_oob_gap_track() -> np.ndarray:
    """Six visible frames climbing off the top, a 12-frame invisible high_shot_oob gap, six visible.

    The pre-gap climb (y 0.5 -> 0.05) extrapolates well above the top edge, so the gap
    is high_shot_oob. Twelve invisible frames are long enough that stock's REST_WINDOW=15
    reads the gap centre as mostly-untracked rest; the arm must not.
    """
    pre_y = np.linspace(0.5, 0.05, 6)
    gap = 12
    ys = np.concatenate([pre_y, np.zeros(gap), np.full(6, 0.05)])
    xs = np.concatenate([np.full(6, 0.5), np.zeros(gap), np.full(6, 0.5)])
    vis = np.concatenate([np.ones(6), np.zeros(gap), np.ones(6)])
    return np.column_stack([xs, ys, vis])


def test_gap_is_high_shot_oob_reads_entry_kinematics():
    invisible = np.array([[0.0, 0.0, 0.0]])
    # Five visible frames climbing toward the top (y 0.5 -> 0.1); the 10-frame
    # extrapolation clears y < 0, so the following gap is high_shot_oob.
    climbing = np.column_stack([np.full(5, 0.5), np.linspace(0.5, 0.1, 5), np.ones(5)])
    assert _gap_is_high_shot_oob(np.vstack([climbing, invisible]), gap_start=5)
    # Same frames descending (y 0.1 -> 0.5): the extrapolation heads down, not high_shot_oob.
    descending = np.column_stack([np.full(5, 0.5), np.linspace(0.1, 0.5, 5), np.ones(5)])
    assert not _gap_is_high_shot_oob(np.vstack([descending, invisible]), gap_start=5)
    # Only one visible frame before the gap: no velocity to fit, so not high_shot_oob.
    one_visible = np.vstack([np.zeros((4, 3)), np.array([[0.5, 0.1, 1.0]]), invisible])
    assert not _gap_is_high_shot_oob(one_visible, gap_start=5)


def test_gap_state_flag_off_leaves_stock_rest_mask():
    # --gap-state off: the initializer must not touch _rest_mask.
    _init_worker(SweepCtx(track=np.zeros((3, 3)), gt_rallies=[]), nan_smoothing=False)
    assert stage8_module._rest_mask is _STOCK_REST_MASK


def test_gap_state_flag_on_installs_patch():
    _init_worker(SweepCtx(track=np.zeros((3, 3)), gt_rallies=[]),
                 nan_smoothing=False, gap_state_demotion_bound=75)
    assert stage8_module._rest_mask is gap_state_rest_mask


def test_gap_state_matches_stock_on_gap_free_track():
    # No gaps -> HIGH_SHOT_OOB and DEAD are empty and mostly_untracked is all False, so the
    # gap-state mask must equal stock bit-for-bit (the gap-free half of the pin).
    _patch_stage8(SHIPPED_DEFAULTS)
    track, _gt = _rally_track_and_gt()  # all frames visible, forms one span under shipped
    speed = stage8_module.compute_speed(track)
    stock_mask = _STOCK_REST_MASK(speed, track)
    stock_spans = _STOCK_FIND_RALLY_SPANS(speed, stock_mask)
    assert stock_spans  # the track really does form a span, so the equality below bites
    install_gap_state(75)
    assert np.array_equal(gap_state_rest_mask(speed, track), stock_mask)
    # segment_video agrees too, since _rest_mask is the only thing the arm changed.
    assert stage8_module.segment_video(track)[0] == stock_spans
    _patch_stage8(SHIPPED_DEFAULTS)


def test_gap_state_differs_on_gapped_track():
    # The high_shot_oob gap frames read as rest under stock (mostly_untracked) but never under
    # gap-state (high_shot_oob holds), so the two masks disagree across the gap: the deliberate
    # difference the arm exists to make.
    _patch_stage8(SHIPPED_DEFAULTS)
    track = _high_shot_oob_gap_track()
    speed = stage8_module.compute_speed(track)
    stock_mask = _STOCK_REST_MASK(speed, track)
    install_gap_state(75)
    gap_mask = gap_state_rest_mask(speed, track)
    mid_gap = 12
    assert stock_mask[mid_gap] and not gap_mask[mid_gap]
    assert not np.array_equal(stock_mask, gap_mask)


def test_gap_state_demotes_long_high_shot_oob_gap_to_rest():
    # A demotion_bound shorter than the high_shot_oob gap flips the tail of the gap to DEAD
    # (rest), even though its entry arced off the top.
    _patch_stage8(SHIPPED_DEFAULTS)
    pre_y = np.linspace(0.5, 0.05, 6)
    gap = 40
    ys = np.concatenate([pre_y, np.zeros(gap), np.full(6, 0.05)])
    xs = np.concatenate([np.full(6, 0.5), np.zeros(gap), np.full(6, 0.5)])
    vis = np.concatenate([np.ones(6), np.zeros(gap), np.ones(6)])
    track = np.column_stack([xs, ys, vis])
    speed = stage8_module.compute_speed(track)
    install_gap_state(10)  # gap starts at 6; frames from 16 onward demote to DEAD
    mask = gap_state_rest_mask(speed, track)
    assert not mask[10]  # still inside the high_shot_oob window [6, 16): not rest
    assert mask[30]      # past the demotion bound: DEAD, so rest


# ---------------------------------------------------------------------------
# 9b. Re-entry guard on the gap-state arm (Change 4: --reentry-guard)
# ---------------------------------------------------------------------------
def _reentry_track(
    reappearance_ys: list[float], entry_end: float = 0.05, gap: int = 12,
    trailing_invisible: int = 0,
) -> np.ndarray:
    """Top-arc entry, a ``gap``-frame invisible run, then given reappearance frames.

    Six pre-gap frames climb from y 0.5 to ``entry_end`` so the entry extrapolates off
    the top (high_shot_oob). ``reappearance_ys`` are the y of the visible frames right after
    the gap; y grows downward, so an increasing sequence descends. ``trailing_invisible``
    appends invisible frames after the reappearance run, to cap how many consecutive
    visible frames the guard sees (for the undefined-velocity case). The gap spans
    ``[6, 6 + gap)``, so gap_start=6 and gap_end=18 at the default gap.
    """
    pre_y = np.linspace(0.5, entry_end, 6)
    reappear = np.asarray(reappearance_ys, dtype=float)
    n_re = len(reappear)
    ys = np.concatenate([pre_y, np.zeros(gap), reappear, np.zeros(trailing_invisible)])
    xs = np.concatenate([np.full(6, 0.5), np.zeros(gap), np.full(n_re, 0.5), np.zeros(trailing_invisible)])
    vis = np.concatenate([np.ones(6), np.zeros(gap), np.ones(n_re), np.zeros(trailing_invisible)])
    return np.column_stack([xs, ys, vis])


def test_install_gap_state_without_guard_leaves_guard_off():
    # The exact-stock pin: install_gap_state with no guard args must leave the guard
    # state None, so _gap_holds_open returns the committed entry test verbatim.
    install_gap_state(75)
    assert stage8_sweep._REENTRY_GUARD_VARIANT is None
    assert stage8_sweep._REENTRY_GUARD_BUFFER is None
    # A top-arc gap still holds open with the guard off (committed behaviour).
    assert _gap_holds_open(_reentry_track([0.02, 0.04, 0.06]), gap_start=6, gap_end=18)


def test_install_gap_state_rejects_half_specified_guard():
    with pytest.raises(ValueError):
        install_gap_state(75, ReentryGuardVariant.TWO_SIDED, None)
    with pytest.raises(ValueError):
        install_gap_state(75, None, 0.05)


def test_reentry_guard_installs_through_init_worker():
    _init_worker(SweepCtx(track=np.zeros((3, 3)), gt_rallies=[]), nan_smoothing=False,
                 gap_state_demotion_bound=75,
                 reentry_guard_variant=ReentryGuardVariant.TWO_SIDED, reentry_guard_buffer=0.05)
    assert stage8_module._rest_mask is gap_state_rest_mask
    assert stage8_sweep._REENTRY_GUARD_VARIANT is ReentryGuardVariant.TWO_SIDED
    assert stage8_sweep._REENTRY_GUARD_BUFFER == 0.05


def test_reentry_guard_pass_near_top_descending():
    # Reappears at y 0.02 (<= buffer) and descends (0.02 -> 0.10): passes both variants,
    # since the entry y (0.05) is also within the buffer.
    track = _reentry_track([0.02, 0.04, 0.06, 0.08, 0.10])
    install_gap_state(75, ReentryGuardVariant.REENTRY_ONLY, 0.10)
    assert _gap_passes_reentry_guard(track, gap_start=6, gap_end=18)
    install_gap_state(75, ReentryGuardVariant.TWO_SIDED, 0.10)
    assert _gap_passes_reentry_guard(track, gap_start=6, gap_end=18)


def test_reentry_guard_fails_when_reappears_low():
    # Descends, but reappears at y 0.50, well below the buffer: the position half fails.
    track = _reentry_track([0.50, 0.52, 0.54, 0.56, 0.58])
    install_gap_state(75, ReentryGuardVariant.REENTRY_ONLY, 0.10)
    assert not _gap_passes_reentry_guard(track, gap_start=6, gap_end=18)


def test_reentry_guard_fails_when_ascending():
    # Reappears near the top (y 0.10) but climbs (0.10 -> 0.02), so vel_y < 0: the
    # descent half fails even though the position half passes.
    track = _reentry_track([0.10, 0.08, 0.06, 0.04, 0.02])
    install_gap_state(75, ReentryGuardVariant.REENTRY_ONLY, 0.15)
    assert not _gap_passes_reentry_guard(track, gap_start=6, gap_end=18)


def test_reentry_guard_fails_on_undefined_velocity():
    # One visible frame after the gap, then invisible: fewer than two, so the re-entry
    # velocity is NaN and the guard fails closed (NaN > 0 is False).
    track = _reentry_track([0.02], trailing_invisible=3)
    install_gap_state(75, ReentryGuardVariant.REENTRY_ONLY, 0.10)
    assert np.isnan(_reentry_velocity_y(track, gap_end=18))
    assert not _gap_passes_reentry_guard(track, gap_start=6, gap_end=18)


def test_reentry_guard_abstains_on_open_ended_gap():
    # The gap runs to the end of the track (never reappears): no re-entry evidence to
    # judge, so the guard abstains for both variants and the demotion bound owns the
    # gap, exactly as in the unguarded arm.
    track = _reentry_track([])  # length 18, gap [6, 18), gap_end == len(track)
    assert len(track) == 18
    install_gap_state(75, ReentryGuardVariant.REENTRY_ONLY, 0.10)
    assert _gap_passes_reentry_guard(track, gap_start=6, gap_end=18)
    install_gap_state(75, ReentryGuardVariant.TWO_SIDED, 0.10)
    assert _gap_passes_reentry_guard(track, gap_start=6, gap_end=18)


def test_init_worker_with_arms_off_restores_stock_bindings():
    # An initialiser that only half-initialises is a trap for a process reused across
    # settings: install everything, then re-init with every arm off and expect the
    # stock bindings back.
    ctx = SweepCtx(track=np.zeros((3, 3)), gt_rallies=[])
    _init_worker(ctx, nan_smoothing=True, gap_state_demotion_bound=75,
                 quiet_start_window=25,
                 reentry_guard_variant=ReentryGuardVariant.TWO_SIDED, reentry_guard_buffer=0.05)
    assert stage8_module._rest_mask is not _STOCK_REST_MASK
    _init_worker(ctx, nan_smoothing=False)
    assert stage8_module._rest_mask is _STOCK_REST_MASK
    assert stage8_module._find_rally_spans is _STOCK_FIND_RALLY_SPANS
    assert stage8_module.detect_contacts is stage8_sweep._STOCK_DETECT_CONTACTS
    assert stage8_module._rolling_mean is stage8_sweep._STOCK_ROLLING_MEAN


def test_reentry_guard_two_sided_vs_reentry_only_differ_on_entry():
    # Reappears near the top and descends, but the entry y (0.15) sits above the buffer:
    # reentry-only passes (it never looks at entry), two-sided fails on the entry half.
    track = _reentry_track([0.02, 0.04, 0.06, 0.08, 0.10], entry_end=0.15)
    assert _gap_is_high_shot_oob(track, gap_start=6)  # entry still extrapolates off the top
    install_gap_state(75, ReentryGuardVariant.REENTRY_ONLY, 0.10)
    assert _gap_passes_reentry_guard(track, gap_start=6, gap_end=18)
    install_gap_state(75, ReentryGuardVariant.TWO_SIDED, 0.10)
    assert not _gap_passes_reentry_guard(track, gap_start=6, gap_end=18)


def test_reentry_guard_removes_a_hold_at_the_mask_level():
    # End to end through gap_state_rest_mask: a top-arc gap that reappears low holds open
    # with the guard off (mid-gap not rest) but falls to DEAD with the guard on (rest),
    # exactly as if the entry test had never fired.
    _patch_stage8(SHIPPED_DEFAULTS)
    track = _reentry_track([0.50, 0.52, 0.54, 0.56, 0.58])  # top-arc entry, reappears low
    speed = stage8_module.compute_speed(track)
    mid_gap = 12  # window [10, 15) is all invisible, so only DEAD can make it rest
    install_gap_state(75)  # guard off: the top-arc holds the gap open
    assert not gap_state_rest_mask(speed, track)[mid_gap]
    install_gap_state(75, ReentryGuardVariant.REENTRY_ONLY, 0.10)  # guard on: low re-entry fails
    assert gap_state_rest_mask(speed, track)[mid_gap]
    _patch_stage8(SHIPPED_DEFAULTS)


# ---------------------------------------------------------------------------
# 10. Quiet-start arm (Change 3: --quiet-start _find_rally_spans patch)
# ---------------------------------------------------------------------------
def _quiet_start_speed_and_rest(rest_slice: slice) -> tuple[np.ndarray, np.ndarray]:
    """Length-120 speed with two 5-frame fast bursts at 10 and 60, plus a rest mask.

    ``rest_slice`` marks the frames to flag at_rest; the caller shapes it so the first
    or second burst is (or isn't) quiet-preceded. END_REST_FRAMES is patched to 40 by
    the tests, and no rest run reaches that, so the whole track is one active region.
    """
    speed = np.zeros(120)
    speed[10:15] = 0.05  # > shipped START_SPEED 0.03, >= START_MIN_FRAMES 3
    speed[60:65] = 0.05
    at_rest = np.zeros(120, dtype=bool)
    at_rest[rest_slice] = True
    return speed, at_rest


def test_quiet_start_flag_off_leaves_stock_find_spans():
    _init_worker(SweepCtx(track=np.zeros((3, 3)), gt_rallies=[]), nan_smoothing=False)
    assert stage8_module._find_rally_spans is _STOCK_FIND_RALLY_SPANS


def test_quiet_start_flag_on_installs_patch():
    _init_worker(SweepCtx(track=np.zeros((3, 3)), gt_rallies=[]),
                 nan_smoothing=False, quiet_start_window=50)
    assert stage8_module._find_rally_spans is quiet_start_find_rally_spans


def test_quiet_start_moves_start_to_first_quiet_burst():
    # Burst 10 is preceded by active frames (not quiet); burst 60 by a 30-frame rest
    # (quiet). Stock opens at 10; quiet-start skips to 60. Same region end, so the
    # rally still forms (coverage holds), it just opens later.
    _patch_stage8(SHIPPED_DEFAULTS._replace(end_rest_frames=40))
    speed, at_rest = _quiet_start_speed_and_rest(slice(30, 60))
    stock_spans = _STOCK_FIND_RALLY_SPANS(speed, at_rest)
    install_quiet_start(25)
    quiet_spans = quiet_start_find_rally_spans(speed, at_rest)
    assert stock_spans == [(10, 120)]
    assert quiet_spans == [(60, 120)]
    _patch_stage8(SHIPPED_DEFAULTS)


def test_quiet_start_matches_stock_when_first_burst_quiet():
    # Burst 10 already sits behind a rest, so quiet-start picks the same first burst
    # and the spans are identical: the bit-identical case.
    _patch_stage8(SHIPPED_DEFAULTS._replace(end_rest_frames=40))
    speed, at_rest = _quiet_start_speed_and_rest(slice(0, 10))
    stock_spans = _STOCK_FIND_RALLY_SPANS(speed, at_rest)
    install_quiet_start(25)
    assert quiet_start_find_rally_spans(speed, at_rest) == stock_spans
    _patch_stage8(SHIPPED_DEFAULTS)


def test_quiet_start_falls_back_when_no_burst_is_quiet():
    # No burst has a quiet run-up, so quiet-start falls back to the first burst: the
    # coverage-preserving guarantee, identical to stock.
    _patch_stage8(SHIPPED_DEFAULTS._replace(end_rest_frames=40))
    speed, at_rest = _quiet_start_speed_and_rest(slice(0, 0))  # no rest anywhere
    stock_spans = _STOCK_FIND_RALLY_SPANS(speed, at_rest)
    install_quiet_start(25)
    assert quiet_start_find_rally_spans(speed, at_rest) == stock_spans
    assert stock_spans == [(10, 120)]
    _patch_stage8(SHIPPED_DEFAULTS)


def test_is_quiet_before_fraction_and_truncation():
    at_rest = np.array([True] * 8 + [False] * 2 + [True] * 10)
    # Window [max(0,10-5), 10) = frames 5..9: three rest, two active -> 0.6 < 0.8.
    assert not _is_quiet_before(at_rest, burst_start=10, window=5)
    # Widen to 10: frames 0..9 -> eight rest of ten = 0.8 >= 0.8.
    assert _is_quiet_before(at_rest, burst_start=10, window=10)
    # A burst at frame 0 has an empty window: cannot be confirmed quiet.
    assert not _is_quiet_before(at_rest, burst_start=0, window=5)


# ---------------------------------------------------------------------------
# 11. Serve-start arm (Change 5: --serve-start _find_rally_spans patch)
# ---------------------------------------------------------------------------
def _serve_start_speed_rest_dist(qualifying_bursts: set[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Length-120 speed with two 5-frame bursts at 10 and 60, no rest, and a gate dist array.

    ``qualifying_bursts`` is the subset of burst starts (of {10, 60}) whose SERVE_START_LOOKBACK
    lookback is made to pass the serve-setup gate: those lookback frames get a small (<= 0.10)
    distance; every other frame stays NaN so its lookback fails. END_REST_FRAMES is patched to 40
    by the caller and no rest run reaches it, so the whole track is one active region [0, 120).
    """
    speed = np.zeros(120)
    speed[10:15] = 0.05  # > shipped START_SPEED 0.03, >= START_MIN_FRAMES 3
    speed[60:65] = 0.05
    at_rest = np.zeros(120, dtype=bool)
    dist = np.full(120, np.nan)
    for burst in qualifying_bursts:
        dist[max(0, burst - SERVE_START_LOOKBACK_FRAMES):burst] = 0.03  # small -> median passes <= 0.10
    return speed, at_rest, dist


def test_serve_setup_before_gate_pass_fail_nan():
    dist = np.full(80, np.nan)
    dist[35:60] = 0.03  # small distances over the 25-frame lookback before burst 60
    # Median 0.03 <= 0.10: passes.
    assert _serve_setup_before(dist, burst_start=60, threshold=0.10)
    # Same lookback, tighter threshold 0.02: 0.03 > 0.02, fails.
    assert not _serve_setup_before(dist, burst_start=60, threshold=0.02)
    # A finite but far lookback (median above threshold) fails.
    far = np.full(80, np.nan)
    far[35:60] = 0.5
    assert not _serve_setup_before(far, burst_start=60, threshold=0.10)
    # An all-NaN lookback (no visible-shuttle court-scale frame in that second) fails.
    assert not _serve_setup_before(dist, burst_start=20, threshold=0.10)


def test_serve_start_opens_at_first_qualifying_burst():
    # Burst 10's lookback is NaN (fails); burst 60's is small (passes). Both modes open the
    # span at 60, same region end, so coverage of the late strokes holds.
    _patch_stage8(SHIPPED_DEFAULTS._replace(end_rest_frames=40))
    speed, at_rest, dist = _serve_start_speed_rest_dist({60})
    assert _STOCK_FIND_RALLY_SPANS(speed, at_rest) == [(10, 120)]  # stock opens at the first burst
    for mode in (ServeStartMode.TRIM, ServeStartMode.REJECT):
        install_serve_start(dist, 0.10, mode)
        assert serve_start_find_rally_spans(speed, at_rest) == [(60, 120)]
    _patch_stage8(SHIPPED_DEFAULTS)


def test_serve_start_trim_falls_back_when_no_qualifying_burst():
    # No burst's lookback qualifies. TRIM falls back to the first burst (the stock start), so
    # the span survives; the region is recorded as fell-back.
    _patch_stage8(SHIPPED_DEFAULTS._replace(end_rest_frames=40))
    speed, at_rest, dist = _serve_start_speed_rest_dist(set())  # all-NaN dist: nothing qualifies
    install_serve_start(dist, 0.10, ServeStartMode.TRIM)
    assert serve_start_find_rally_spans(speed, at_rest) == [(10, 120)]  # stock first burst
    diag = stage8_sweep._SERVE_START_LAST_DIAGNOSTICS
    assert diag['n_no_qualify'] == 1 and diag['n_qualified'] == 0
    assert diag['no_qualify_regions'] == [(0, 120)]
    _patch_stage8(SHIPPED_DEFAULTS)


def test_serve_start_reject_drops_region_when_no_qualifying_burst():
    # Same no-qualify region under REJECT: no span at all (the region is dropped).
    _patch_stage8(SHIPPED_DEFAULTS._replace(end_rest_frames=40))
    speed, at_rest, dist = _serve_start_speed_rest_dist(set())
    install_serve_start(dist, 0.10, ServeStartMode.REJECT)
    assert serve_start_find_rally_spans(speed, at_rest) == []
    diag = stage8_sweep._SERVE_START_LAST_DIAGNOSTICS
    assert diag['n_no_qualify'] == 1 and diag['no_qualify_regions'] == [(0, 120)]
    _patch_stage8(SHIPPED_DEFAULTS)


def test_serve_start_flag_off_leaves_stock_find_spans():
    _init_worker(SweepCtx(track=np.zeros((3, 3)), gt_rallies=[]), nan_smoothing=False)
    assert stage8_module._find_rally_spans is _STOCK_FIND_RALLY_SPANS


def test_serve_start_installs_and_uninstalls_through_init_worker():
    ctx = SweepCtx(track=np.zeros((3, 3)), gt_rallies=[])
    dist = np.full(3, np.nan)
    _init_worker(ctx, nan_smoothing=False, serve_start_mode=ServeStartMode.REJECT,
                 serve_start_threshold=0.10, serve_start_dist=dist)
    assert stage8_module._find_rally_spans is serve_start_find_rally_spans
    assert stage8_sweep._SERVE_START_MODE is ServeStartMode.REJECT
    assert stage8_sweep._SERVE_START_THRESHOLD == 0.10
    assert stage8_sweep._SERVE_START_DIST is dist
    # Re-init with the arm off: the stock binding must return.
    _init_worker(ctx, nan_smoothing=False)
    assert stage8_module._find_rally_spans is _STOCK_FIND_RALLY_SPANS


def test_serve_start_absent_is_exact_stock():
    # Bit-for-bit pin: install the arm, then turn it off, and segment_video is the stock
    # segmentation again. The arm changes _find_rally_spans and nothing else.
    _patch_stage8(SHIPPED_DEFAULTS)
    track, _gt = _rally_track_and_gt()  # all visible, forms one span under shipped
    speed = stage8_module.compute_speed(track)
    stock_mask = _STOCK_REST_MASK(speed, track)
    stock_spans = _STOCK_FIND_RALLY_SPANS(speed, stock_mask)
    assert stock_spans  # the track really does form a span, so the equality below bites
    install_serve_start(np.full(len(track), np.nan), 0.10, ServeStartMode.TRIM)
    _init_worker(SweepCtx(track=track, gt_rallies=[]), nan_smoothing=False)  # arm off
    assert stage8_module._find_rally_spans is _STOCK_FIND_RALLY_SPANS
    assert stage8_module.segment_video(track)[0] == stock_spans
    _patch_stage8(SHIPPED_DEFAULTS)


# ---------------------------------------------------------------------------
# 12. Serve-start wide-shot refinement (--serve-start-wideshot)
# ---------------------------------------------------------------------------
# Synthetic court-scale boxes under the PILOT_* constants (court x [635, 1316],
# foot y [254, 1030], height [84, 336], mid-line 642): one player per half, static.
TOP_BOX = (900.0, 500.0, 150.0)  # (foot_x, foot_y, height) px; foot y 500 < 642 -> top half
BOT_BOX = (1000.0, 900.0, 250.0)  # foot y 900 >= 642 -> bottom half


def _mk_wideshot_inputs(frame_boxes: list[list[tuple[float, float, float]]]) -> WideshotInputs:
    """WideshotInputs from per-frame (foot_x, foot_y, height) pixel boxes.

    Boxes become xyxy with a fixed 60 px width; scores descend with list order so the
    first box of a half is its highest-score pick. Empty frames stay all-NaN, the raw
    pose padding convention build_serve_start_wideshot_inputs expects.
    """
    n_frames = len(frame_boxes)
    bboxes = np.full((n_frames, 16, 4), np.nan)
    scores = np.full((n_frames, 16), np.nan)
    for frame, boxes in enumerate(frame_boxes):
        for slot, (foot_x, foot_y, height) in enumerate(boxes):
            bboxes[frame, slot] = (foot_x - 30.0, foot_y - height, foot_x + 30.0, foot_y)
            scores[frame, slot] = 0.9 - 0.1 * slot
    return build_serve_start_wideshot_inputs(bboxes, scores)


def test_wide_shot_gate_passes_on_static_two_player_lookback():
    # 25 frames, one static court-scale player per half: count_med 2, both halves
    # occupied, drift 0. The canonical serve wide shot.
    inputs = _mk_wideshot_inputs([[TOP_BOX, BOT_BOX]] * 25)
    assert _wide_shot_before(inputs, burst_start=25)


def test_wide_shot_gate_count_fail():
    # Each half occupied for 13 of 25 frames (>= the 12.5 present floor) but only frame 12
    # has both at once: count median 1 < 2. Isolates the count condition; presence and
    # drift both pass.
    frames = [[TOP_BOX] for _ in range(25)]
    for frame in range(12, 25):
        frames[frame] = [BOT_BOX]
    frames[12] = [TOP_BOX, BOT_BOX]
    inputs = _mk_wideshot_inputs(frames)
    assert not _wide_shot_before(inputs, burst_start=25)


def test_wide_shot_gate_slot_fail():
    # Two static court-scale players, both in the TOP half: count_med 2 passes but the
    # bottom half is never occupied.
    second_top = (800.0, 550.0, 160.0)
    inputs = _mk_wideshot_inputs([[TOP_BOX, second_top]] * 25)
    assert not _wide_shot_before(inputs, burst_start=25)


def test_wide_shot_gate_drift_fail():
    # Both halves occupied every frame, but the bottom player walks 10 px/frame: the
    # head/tail foot means sit 150 px apart (0.078 image-fraction > 0.05). Count and
    # presence pass; the drift condition binds.
    frames = [
        [TOP_BOX, (1000.0 + 10.0 * frame, 900.0, 250.0)] for frame in range(25)
    ]
    inputs = _mk_wideshot_inputs(frames)
    assert not _wide_shot_before(inputs, burst_start=25)


def test_wide_shot_gate_short_series_drift_abstains():
    # Ten present feet fill exactly one drift window, so head and tail would fully
    # overlap and read 0.0 even though the bottom player sprints 15 px/frame. The
    # short series must abstain to NaN and the gate fail closed. Presence still passes
    # (10 of 10 frames >= the 5-frame floor of this truncated lookback); drift decides.
    frames = [
        [TOP_BOX, (1000.0 + 15.0 * frame, 900.0, 250.0)] for frame in range(10)
    ]
    inputs = _mk_wideshot_inputs(frames)
    assert not _wide_shot_before(inputs, burst_start=10)


def test_wide_shot_gate_empty_or_truncated_lookback_fails():
    # No detections at all in the lookback: count 0, no halves. And burst_start 0 has no
    # lookback frames at all. Both read as not-wide-shot rather than crash.
    inputs = _mk_wideshot_inputs([[] for _ in range(25)])
    assert not _wide_shot_before(inputs, burst_start=25)
    assert not _wide_shot_before(inputs, burst_start=0)


def test_serve_start_wideshot_requires_both_gates():
    # Bursts at 10 and 60 BOTH pass the distance gate; only burst 60's lookback holds the
    # wide shot. With the refinement on, the span opens at 60; off, at 10 (the prior
    # serve-start pick). The refinement is a strict AND on top of the distance gate.
    _patch_stage8(SHIPPED_DEFAULTS._replace(end_rest_frames=40))
    speed, at_rest, dist = _serve_start_speed_rest_dist({10, 60})
    frames: list[list[tuple[float, float, float]]] = [[] for _ in range(120)]
    for frame in range(35, 60):  # burst 60's 25-frame lookback
        frames[frame] = [TOP_BOX, BOT_BOX]
    inputs = _mk_wideshot_inputs(frames)
    install_serve_start(dist, 0.10, ServeStartMode.TRIM, wideshot=inputs)
    assert serve_start_find_rally_spans(speed, at_rest) == [(60, 120)]
    install_serve_start(dist, 0.10, ServeStartMode.TRIM)  # refinement off
    assert serve_start_find_rally_spans(speed, at_rest) == [(10, 120)]
    _patch_stage8(SHIPPED_DEFAULTS)


def test_serve_start_wideshot_off_is_prior_behaviour_bit_for_bit():
    # With wideshot left at the default None the arm must reproduce the pre-refinement
    # spans exactly, even straight after a wideshot install (install resets the state).
    _patch_stage8(SHIPPED_DEFAULTS._replace(end_rest_frames=40))
    speed, at_rest, dist = _serve_start_speed_rest_dist({60})
    failing_everywhere = _mk_wideshot_inputs([[] for _ in range(120)])
    install_serve_start(dist, 0.10, ServeStartMode.TRIM, wideshot=failing_everywhere)
    assert serve_start_find_rally_spans(speed, at_rest) == [(10, 120)]  # gate vetoes 60
    for mode, expected in ((ServeStartMode.TRIM, [(60, 120)]),
                           (ServeStartMode.REJECT, [(60, 120)])):
        install_serve_start(dist, 0.10, mode)
        assert stage8_sweep._SERVE_START_WIDESHOT is None
        assert serve_start_find_rally_spans(speed, at_rest) == expected  # the pinned prior picks
    _patch_stage8(SHIPPED_DEFAULTS)


def test_serve_start_wideshot_installs_and_uninstalls_through_init_worker():
    ctx = SweepCtx(track=np.zeros((3, 3)), gt_rallies=[])
    dist = np.full(3, np.nan)
    inputs = _mk_wideshot_inputs([[] for _ in range(3)])
    _init_worker(ctx, nan_smoothing=False, serve_start_mode=ServeStartMode.REJECT,
                 serve_start_threshold=0.10, serve_start_dist=dist,
                 serve_start_wideshot=inputs)
    assert stage8_module._find_rally_spans is serve_start_find_rally_spans
    assert stage8_sweep._SERVE_START_WIDESHOT is inputs
    # Re-init with serve-start on but the refinement off: the wideshot state must clear.
    _init_worker(ctx, nan_smoothing=False, serve_start_mode=ServeStartMode.REJECT,
                 serve_start_threshold=0.10, serve_start_dist=dist)
    assert stage8_module._find_rally_spans is serve_start_find_rally_spans
    assert stage8_sweep._SERVE_START_WIDESHOT is None
    # Re-init with the arm off entirely: stock binding returns AND the per-process
    # arrays drop, so an off-arm worker holds no stale distance/wideshot copy.
    _init_worker(ctx, nan_smoothing=False)
    assert stage8_module._find_rally_spans is _STOCK_FIND_RALLY_SPANS
    assert stage8_sweep._SERVE_START_DIST is None
    assert stage8_sweep._SERVE_START_WIDESHOT is None
