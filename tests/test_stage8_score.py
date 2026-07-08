"""Stage-8 scoring tests: synthetic spans/contacts/GT with hand-checkable truth.

Every fixture is small enough that the expected coverage taxonomy, alignment
arithmetic, greedy-match assignment and count-gate fractions are obvious by
inspection. No file I/O beyond a couple of in-memory DataFrames for the CLI
helper checks.
"""
import numpy as np
import pandas as pd
import pytest

from scripts.stage8_score import (
    GtRally,
    RallyBoundary,
    _parse_proximity,
    _parse_tolerances,
    _spans_from_df,
    classify_rally_boundary,
    greedy_match,
    load_gt_rallies,
    merged_span_indices,
    score_boundaries,
    score_contacts,
    score_stage8,
)


def _rally(set_id: str, rally: int, frames: tuple[int, ...]) -> GtRally:
    return GtRally(set_id=set_id, rally=rally, stroke_frames=frames)


# ---------------------------------------------------------------------------
# Boundary taxonomy
# ---------------------------------------------------------------------------
def test_covered_all_strokes_in_one_span():
    rally = _rally('set1', 1, (10, 12, 14))
    spans = [(8, 20)]
    category, mapped = classify_rally_boundary(rally.stroke_frames, spans)
    assert category is RallyBoundary.COVERED
    assert mapped == 0

    result = score_boundaries(spans, [rally])
    assert result['covered'] == 1
    assert result['split'] == 0
    assert result['missed'] == 0
    assert result['covered_fraction'] == 1.0


def test_split_across_two_spans():
    rally = _rally('set1', 1, (10, 40))  # 10 in span0, 40 in span1
    spans = [(8, 20), (30, 50)]
    category, mapped = classify_rally_boundary(rally.stroke_frames, spans)
    assert category is RallyBoundary.SPLIT
    assert mapped is None
    assert score_boundaries(spans, [rally])['split'] == 1


def test_split_partly_outside_any_span():
    rally = _rally('set1', 1, (10, 25))  # 25 falls outside the only span
    spans = [(8, 20)]
    category, _ = classify_rally_boundary(rally.stroke_frames, spans)
    assert category is RallyBoundary.SPLIT


def test_missed_no_stroke_in_any_span():
    rally = _rally('set1', 1, (100, 102))
    spans = [(8, 20)]
    category, mapped = classify_rally_boundary(rally.stroke_frames, spans)
    assert category is RallyBoundary.MISSED
    assert mapped is None
    assert score_boundaries(spans, [rally])['missed'] == 1


def test_merged_span_contains_two_rallies():
    rallies = [_rally('set1', 1, (10, 14)), _rally('set1', 2, (20, 24))]
    spans = [(5, 30)]  # swallows both extents
    assert merged_span_indices(spans, rallies) == {0}
    result = score_boundaries(spans, rallies)
    assert result['merged_spans'] == 1
    assert result['covered'] == 2  # each rally still maps cleanly to the one span


def test_spurious_span_holds_no_strokes():
    rally = _rally('set1', 1, (10, 14))
    spans = [(8, 20), (50, 60)]  # span1 covers no GT stroke
    result = score_boundaries(spans, [rally])
    assert result['spurious_spans'] == 1
    assert 'replay' in result['spurious_spans_note']


# ---------------------------------------------------------------------------
# Start / end alignment arithmetic
# ---------------------------------------------------------------------------
def test_start_and_end_alignment_stats():
    rallies = [_rally('set1', 1, (10, 14)), _rally('set1', 2, (30, 34))]
    spans = [(8, 20), (25, 40)]
    # start offsets = span_start - first_stroke = 8-10, 25-30 = -2, -5
    # end offsets   = span_end   - last_stroke  = 20-14, 40-34 =  6,  6
    result = score_boundaries(spans, rallies)

    start = result['start_alignment']
    assert start['n'] == 2
    assert start['mean'] == pytest.approx(-3.5)
    assert start['median'] == pytest.approx(-3.5)
    assert start['p10'] == pytest.approx(-4.7)  # np.percentile([-2,-5], 10)
    assert start['p90'] == pytest.approx(-2.3)  # np.percentile([-2,-5], 90)

    end = result['end_alignment']
    assert end['mean'] == pytest.approx(6.0)
    assert end['median'] == pytest.approx(6.0)


def test_alignment_none_when_no_covered_rally():
    rally = _rally('set1', 1, (100, 102))  # missed -> no covered rally
    result = score_boundaries([(8, 20)], [rally])
    assert result['start_alignment'] is None
    assert result['end_alignment'] is None


# ---------------------------------------------------------------------------
# Greedy matching truth table
# ---------------------------------------------------------------------------
def test_greedy_exact_and_within_tolerance():
    assert greedy_match([10], [10], tolerance=2) == [(0, 0)]
    assert greedy_match([10], [12], tolerance=2) == [(0, 0)]  # dist 2 == tol, inclusive


def test_greedy_tolerance_edge_excludes_beyond():
    assert greedy_match([10], [12], tolerance=1) == []  # dist 2 > tol 1


def test_greedy_closest_candidate_wins():
    # Two candidates for one stroke: the closer one (frame 11, dist 1) binds.
    assert greedy_match([10], [8, 11], tolerance=5) == [(0, 1)]


def test_greedy_tie_breaks_to_lower_candidate_index():
    # Equal distance (both 2): tie resolves to the lower candidate index.
    assert greedy_match([10], [8, 12], tolerance=5) == [(0, 0)]


def test_greedy_candidate_claims_at_most_one_stroke():
    # One candidate equidistant from two strokes: lower GT index claims it,
    # the other stroke goes uncredited (one-to-one).
    assert greedy_match([10, 12], [11], tolerance=2) == [(0, 0)]


def test_greedy_multi_pair_assignment():
    # gt0 near c0/c1 (dist 1 each), gt1 near c2 (dist 1): gt0 takes c0, gt1 c2.
    assert greedy_match([10, 20], [9, 11, 19], tolerance=3) == [(0, 0), (1, 2)]


# ---------------------------------------------------------------------------
# Contact metrics: count gate, tolerance curve, per-set
# ---------------------------------------------------------------------------
def test_count_gate_covered_fraction():
    rallies = [_rally('set1', 1, (10, 12, 14)), _rally('set1', 2, (20, 22))]
    spans = [(8, 16), (18, 26)]
    # span0 gets 3 contacts (== 3 strokes -> pass), span1 gets 3 (!= 2 -> fail)
    contacts = [
        (0, 10, None), (0, 12, None), (0, 14, None),
        (1, 20, None), (1, 22, None), (1, 25, None),
    ]
    result = score_contacts(spans, contacts, rallies, tolerances=(2,))
    gate = result['count_gate']['covered']
    assert gate['pass'] == 1
    assert gate['total'] == 2
    assert gate['fraction'] == pytest.approx(0.5)


def test_count_gate_unmerged_excludes_merged_span_rallies():
    rallies = [
        _rally('set1', 1, (10, 14)),   # covered by merged span0
        _rally('set1', 2, (20, 24)),   # covered by merged span0
        _rally('set1', 3, (40, 42)),   # covered by clean span1
    ]
    spans = [(5, 30), (38, 46)]  # span0 merges rallies 1 and 2
    contacts = [
        (0, 11, None), (0, 13, None), (0, 21, None), (0, 23, None),  # 4 in the merged span
        (1, 41, None), (1, 42, None),                                # 2 in the clean span
    ]
    result = score_contacts(spans, contacts, rallies, tolerances=(2,))

    covered = result['count_gate']['covered']
    assert covered['total'] == 3
    assert covered['pass'] == 1  # only rally 3 (2 candidates == 2 strokes)

    unmerged = result['count_gate']['unmerged']
    assert unmerged['total'] == 1  # merged-span rallies dropped
    assert unmerged['pass'] == 1
    assert unmerged['fraction'] == pytest.approx(1.0)


def test_count_gate_none_when_no_covered_rally():
    rally = _rally('set1', 1, (100, 102))  # missed
    result = score_contacts([(8, 20)], [], [rally], tolerances=(2,))
    gate = result['count_gate']['covered']
    assert gate['total'] == 0
    assert gate['fraction'] is None


def test_tolerance_curve_precision_recall_f1():
    rally = _rally('set1', 1, (10, 20))
    spans = [(5, 30)]
    contacts = [(0, 10, None), (0, 21, None), (0, 25, None)]  # 25 is a false candidate
    result = score_contacts(spans, contacts, [rally], tolerances=(2,))

    metric = result['tolerances']['2']
    assert metric['gt'] == 2
    assert metric['candidates'] == 3
    assert metric['matched'] == 2  # 10<->10, 20<->21; 25 uncredited
    assert metric['recall'] == pytest.approx(1.0)
    assert metric['precision'] == pytest.approx(2 / 3)
    assert metric['f1'] == pytest.approx(0.8)


def test_candidates_pool_from_all_overlapping_spans():
    # A split rally straddles two spans; its candidates come from both.
    rally = _rally('set1', 1, (10, 40))
    spans = [(8, 20), (30, 50)]
    contacts = [(0, 10, None), (1, 40, None)]
    result = score_contacts(spans, contacts, [rally], tolerances=(2,))
    metric = result['tolerances']['2']
    assert metric['candidates'] == 2  # both spans overlap the extent
    assert metric['matched'] == 2


def test_raw_precision_dedupes_double_matched_and_counts_all_candidates():
    # Two adjacent GT rallies under ONE merged span; a lone candidate at frame 11
    # sits within +/-2 of a stroke in BOTH, so the pooled curve pools it into each
    # rally and matches it twice. Raw precision must count that physical candidate
    # once. A second candidate in a spurious span (no GT) proves the raw
    # denominator counts every detected candidate, not just the pooled ones.
    rallies = [_rally('set1', 1, (10,)), _rally('set1', 2, (12,))]
    spans = [(5, 20), (100, 110)]          # span 0 merges both rallies; span 1 spurious
    contacts = [(0, 11, None), (1, 105, None)]
    result = score_contacts(spans, contacts, rallies, tolerances=(2,))

    pooled = result['tolerances']['2']
    assert pooled['matched'] == 2          # frame 11 credited in each rally's matching
    assert pooled['candidates'] == 2       # frame 11 pooled once per overlapping rally

    raw = result['precision_raw']['2']
    assert raw['matched'] == 1             # ... but it is one physical candidate
    assert raw['candidates'] == 2          # frames {11, 105}: the spurious one counts too
    assert raw['precision_raw'] == pytest.approx(0.5)


def test_per_set_breakdown_splits_by_set_id():
    rallies = [_rally('set1', 1, (10, 12)), _rally('set2', 1, (110, 112))]
    spans = [(8, 20), (108, 120)]
    contacts = [
        (0, 10, None), (0, 12, None),   # set1 rally: 2 candidates, 2 strokes -> pass
        (1, 110, None),                 # set2 rally: 1 candidate, 2 strokes -> fail
    ]
    result = score_contacts(spans, contacts, rallies, tolerances=(2,))
    per_set = result['per_set']
    assert set(per_set) == {'set1', 'set2'}
    assert per_set['set1']['count_gate_covered']['pass'] == 1
    assert per_set['set2']['count_gate_covered']['pass'] == 0
    assert per_set['set1']['tolerances']['2']['matched'] == 2
    assert per_set['set2']['tolerances']['2']['matched'] == 1


# ---------------------------------------------------------------------------
# Top-level score and GT loading
# ---------------------------------------------------------------------------
def test_score_stage8_shape():
    rally = _rally('set1', 1, (10, 12, 14))
    spans = [(8, 20)]
    contacts = [(0, 10, None), (0, 12, None), (0, 14, None)]
    result = score_stage8(spans, contacts, [rally], tolerances=(1, 2))
    assert result['n_gt_rallies'] == 1
    assert result['tolerances'] == [1, 2]
    assert set(result['boundaries']) >= {'covered', 'merged_spans', 'spurious_spans'}
    assert set(result['contacts']) == {'count_gate', 'tolerances', 'precision_raw', 'per_set'}


def test_load_gt_rallies_groups_and_filters_vid():
    shots = pd.DataFrame({
        'vid': [1, 1, 1, 2],
        'set_id': ['set1', 'set1', 'set2', 'set1'],
        'rally': [1, 1, 1, 1],
        'frame_num': [14, 10, 50, 999],  # first rally out of order to test the sort
    })
    rallies = load_gt_rallies(shots, vid=1)
    assert len(rallies) == 2  # vid 2 excluded
    assert rallies[0].set_id == 'set1'
    assert rallies[0].stroke_frames == (10, 14)  # sorted ascending
    assert rallies[0].extent == (10, 14)
    assert rallies[0].n_strokes == 2
    assert rallies[1].set_id == 'set2'


def test_load_gt_rallies_empty_vid_raises():
    shots = pd.DataFrame({'vid': [1], 'set_id': ['set1'], 'rally': [1], 'frame_num': [10]})
    with pytest.raises(ValueError, match='no strokes'):
        load_gt_rallies(shots, vid=99)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
def test_spans_from_df_requires_contiguous_rally_ids():
    good = pd.DataFrame({
        'video_id': ['v', 'v'],
        'rally_id': [1, 0],  # out of order, still 0..1 once sorted
        'start_frame': [30, 8],
        'end_frame': [50, 20],
    })
    assert _spans_from_df(good) == [(8, 20), (30, 50)]

    gapped = pd.DataFrame({
        'video_id': ['v', 'v'],
        'rally_id': [0, 2],  # gap -> not 0..n-1
        'start_frame': [8, 30],
        'end_frame': [20, 50],
    })
    with pytest.raises(ValueError, match='contiguous'):
        _spans_from_df(gapped)


def test_parse_proximity_true_false_blank():
    assert _parse_proximity('True') is True
    assert _parse_proximity('False') is False
    assert _parse_proximity(np.nan) is None
    assert _parse_proximity(float('nan')) is None


def test_parse_tolerances_list():
    assert _parse_tolerances('1,2,5,10') == [1, 2, 5, 10]
    assert _parse_tolerances('3') == [3]
    assert _parse_tolerances('1, 2') == [1, 2]  # tolerates whitespace
