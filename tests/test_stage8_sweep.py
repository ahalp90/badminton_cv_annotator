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
    apply_replay_mask,
    boundary_sort_key_as_built,
    build_boundary_crowns,
    build_boundary_grid,
    build_boundary_tasks,
    build_contact_grid,
    build_contact_tasks,
    contact_frontier,
    contact_sort_key,
    flatten_row,
    install_nan_smoothing,
    nan_rolling_mean,
    nan_smoothing_detect_contacts,
    select_boundary_winner,
    select_start_alignment_winner,
    select_winner,
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
    """Snapshot and restore the two stage-8 smoothing globals around every test.

    ``install_nan_smoothing`` mutates ``stage8_module`` in place (the same
    module-global patching the sweep uses), so without this a test that installs
    the patch would leak it into later tests and other files. Restores on pass and
    on failure (the teardown runs after the yield regardless).
    """
    saved_rolling = stage8_module._rolling_mean
    saved_detect = stage8_module.detect_contacts
    yield
    stage8_module._rolling_mean = saved_rolling
    stage8_module.detect_contacts = saved_detect


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
