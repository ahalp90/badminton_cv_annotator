"""Calibration scoring tests: synthetic spans, contacts, and GT."""
import pandas as pd
import pytest

from annotator.calibration.scoring import (
    CANONICAL_CONTACT_TOLERANCE_BASE30,
    CONTACT_TOLERANCES_BASE30,
    GtRally,
    RallyBoundary,
    classify_rally_boundary,
    greedy_match,
    load_gt_rallies,
    merged_span_indices,
    safe_f1,
    score_boundaries,
    score_contacts,
)


def _rally(set_id: str, rally: int, frames: tuple[int, ...]) -> GtRally:
    return GtRally(set_id=set_id, rally=rally, stroke_frames=frames)


def test_contact_tolerances_keep_the_persisted_base30_order() -> None:
    assert CANONICAL_CONTACT_TOLERANCE_BASE30 == 5
    assert CONTACT_TOLERANCES_BASE30 == (1, 2, 5, 10)


@pytest.mark.parametrize(
    ('precision', 'recall'),
    ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (0.25, 0.5), (2 / 3, 1.0)),
)
def test_safe_f1_matches_every_former_formula(precision: float, recall: float) -> None:
    denominator = precision + recall
    precision_first = 0.0 if denominator == 0 else 2 * precision * recall / denominator
    recall_first = 0.0 if denominator == 0 else 2 * recall * precision / denominator
    assert safe_f1(precision, recall) == precision_first == recall_first


def test_covered_all_strokes_in_one_span():
    rally = _rally('set1', 1, (10, 12, 14))
    category, mapped = classify_rally_boundary(rally.stroke_frames, [(8, 20)])
    assert category is RallyBoundary.COVERED and mapped == 0
    result = score_boundaries([(8, 20)], [rally])
    assert result['covered'] == 1 and result['split'] == 0 and result['missed'] == 0
    assert result['covered_fraction'] == 1.0


def test_split_across_two_spans():
    rally = _rally('set1', 1, (10, 40))
    category, mapped = classify_rally_boundary(rally.stroke_frames, [(8, 20), (30, 50)])
    assert category is RallyBoundary.SPLIT and mapped is None
    assert score_boundaries([(8, 20), (30, 50)], [rally])['split'] == 1


def test_split_partly_outside_any_span():
    assert classify_rally_boundary((10, 25), [(8, 20)])[0] is RallyBoundary.SPLIT


def test_missed_no_stroke_in_any_span():
    rally = _rally('set1', 1, (100, 102))
    category, mapped = classify_rally_boundary(rally.stroke_frames, [(8, 20)])
    assert category is RallyBoundary.MISSED and mapped is None
    assert score_boundaries([(8, 20)], [rally])['missed'] == 1


def test_merged_span_contains_two_rallies():
    rallies = [_rally('set1', 1, (10, 14)), _rally('set1', 2, (20, 24))]
    assert merged_span_indices([(5, 30)], rallies) == {0}
    result = score_boundaries([(5, 30)], rallies)
    assert result['merged_spans'] == 1 and result['covered'] == 2


def test_spurious_span_holds_no_strokes():
    result = score_boundaries([(8, 20), (50, 60)], [_rally('set1', 1, (10, 14))])
    assert result['spurious_spans'] == 1 and 'replay' in result['spurious_spans_note']


def test_start_and_end_alignment_stats():
    rallies = [_rally('set1', 1, (10, 14)), _rally('set1', 2, (30, 34))]
    result = score_boundaries([(8, 20), (25, 40)], rallies)
    start, end = result['start_alignment'], result['end_alignment']
    assert start['n'] == 2 and start['mean'] == pytest.approx(-3.5)
    assert start['median'] == pytest.approx(-3.5)
    assert start['p10'] == pytest.approx(-4.7) and start['p90'] == pytest.approx(-2.3)
    assert end['mean'] == pytest.approx(6.0) and end['median'] == pytest.approx(6.0)


def test_alignment_none_when_no_covered_rally():
    result = score_boundaries([(8, 20)], [_rally('set1', 1, (100, 102))])
    assert result['start_alignment'] is None and result['end_alignment'] is None


def test_greedy_exact_and_within_tolerance():
    assert greedy_match([10], [10], tolerance=2) == [(0, 0)]
    assert greedy_match([10], [12], tolerance=2) == [(0, 0)]


def test_greedy_tolerance_edge_excludes_beyond():
    assert greedy_match([10], [12], tolerance=1) == []


def test_greedy_closest_candidate_wins():
    assert greedy_match([10], [8, 11], tolerance=5) == [(0, 1)]


def test_greedy_tie_breaks_to_lower_candidate_index():
    assert greedy_match([10], [8, 12], tolerance=5) == [(0, 0)]


def test_greedy_candidate_claims_at_most_one_stroke():
    assert greedy_match([10, 12], [11], tolerance=2) == [(0, 0)]


def test_greedy_multi_pair_assignment():
    assert greedy_match([10, 20], [9, 11, 19], tolerance=3) == [(0, 0), (1, 2)]


def test_count_gate_covered_fraction():
    rallies = [_rally('set1', 1, (10, 12, 14)), _rally('set1', 2, (20, 22))]
    contacts = [(0, 10, None), (0, 12, None), (0, 14, None), (1, 20, None), (1, 22, None), (1, 25, None)]
    result = score_contacts([(8, 16), (18, 26)], contacts, rallies, tolerances=(2,))
    assert result['count_gate']['covered'] == {'pass': 1, 'total': 2, 'fraction': 0.5}


def test_count_gate_unmerged_excludes_merged_span_rallies():
    rallies = [_rally('set1', 1, (10, 14)), _rally('set1', 2, (20, 24)), _rally('set1', 3, (40, 42))]
    contacts = [(0, 11, None), (0, 13, None), (0, 21, None), (0, 23, None), (1, 41, None), (1, 42, None)]
    result = score_contacts([(5, 30), (38, 46)], contacts, rallies, tolerances=(2,))
    assert result['count_gate']['covered']['total'] == 3
    assert result['count_gate']['covered']['pass'] == 1
    assert result['count_gate']['unmerged'] == {'pass': 1, 'total': 1, 'fraction': 1.0}


def test_count_gate_none_when_no_covered_rally():
    gate = score_contacts([(8, 20)], [], [_rally('set1', 1, (100, 102))], tolerances=(2,))['count_gate']['covered']
    assert gate['total'] == 0 and gate['fraction'] is None


def test_tolerance_curve_precision_recall_f1():
    result = score_contacts([(5, 30)], [(0, 10, None), (0, 21, None), (0, 25, None)], [_rally('set1', 1, (10, 20))], tolerances=(2,))
    metric = result['tolerances']['2']
    assert metric['gt'] == 2 and metric['candidates'] == 3 and metric['matched'] == 2
    assert metric['recall'] == pytest.approx(1.0) and metric['precision'] == pytest.approx(2 / 3)
    assert metric['f1'] == pytest.approx(0.8)


def test_candidates_pool_from_all_overlapping_spans():
    rally = _rally('set1', 1, (10, 40))
    result = score_contacts([(8, 20), (30, 50)], [(0, 10, None), (1, 40, None)], [rally], tolerances=(2,))
    assert result['tolerances']['2']['candidates'] == 2 and result['tolerances']['2']['matched'] == 2


def test_raw_precision_dedupes_double_matched_and_counts_all_candidates():
    rallies = [_rally('set1', 1, (10,)), _rally('set1', 2, (12,))]
    result = score_contacts([(5, 20), (100, 110)], [(0, 11, None), (1, 105, None)], rallies, tolerances=(2,))
    pooled, raw = result['tolerances']['2'], result['precision_raw']['2']
    assert pooled['matched'] == 2 and pooled['candidates'] == 2
    assert raw['matched'] == 1 and raw['candidates'] == 2 and raw['precision_raw'] == pytest.approx(0.5)


def test_per_set_breakdown_splits_by_set_id():
    rallies = [_rally('set1', 1, (10, 12)), _rally('set2', 1, (110, 112))]
    contacts = [(0, 10, None), (0, 12, None), (1, 110, None)]
    result = score_contacts([(8, 20), (108, 120)], contacts, rallies, tolerances=(2,))
    per_set = result['per_set']
    assert set(per_set) == {'set1', 'set2'}
    assert per_set['set1']['count_gate_covered']['pass'] == 1
    assert per_set['set2']['count_gate_covered']['pass'] == 0
    assert per_set['set1']['tolerances']['2']['matched'] == 2
    assert per_set['set2']['tolerances']['2']['matched'] == 1


def test_load_gt_rallies_groups_and_filters_vid():
    shots = pd.DataFrame({'vid': [1, 1, 1, 2], 'set_id': ['set1', 'set1', 'set2', 'set1'], 'rally': [1, 1, 1, 1], 'frame_num': [14, 10, 50, 999]})
    rallies = load_gt_rallies(shots, vid=1)
    assert len(rallies) == 2 and rallies[0].set_id == 'set1'
    assert rallies[0].stroke_frames == (10, 14) and rallies[0].extent == (10, 14)
    assert rallies[0].n_strokes == 2 and rallies[1].set_id == 'set2'


def test_load_gt_rallies_empty_vid_raises():
    shots = pd.DataFrame({'vid': [1], 'set_id': ['set1'], 'rally': [1], 'frame_num': [10]})
    with pytest.raises(ValueError, match='no strokes'):
        load_gt_rallies(shots, vid=99)
