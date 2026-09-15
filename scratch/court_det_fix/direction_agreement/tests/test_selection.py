"""Synthetic checks for bucket instrumentation and precision representatives (E2)."""

import numpy as np
import pytest
import vp_pruning
from selection import (
    LEADER_RULE,
    PRECISION_RULE,
    arm_record,
    choose_precise,
    retain_pencils_with_buckets,
    support_arrays,
)

ANGLE_DEG = 1.5


def random_support(seed: int, candidates: int = 240, lines: int = 24) -> tuple[np.ndarray, ...]:
    """Residuals with duplicated near-identical rows so IoU suppression removes candidates."""
    generator = np.random.default_rng(seed)
    residuals = generator.uniform(0., 4., size=(candidates, lines))
    copies = generator.integers(candidates, size=candidates // 3)
    residuals[:len(copies)] = np.clip(residuals[copies] + generator.normal(scale=.05, size=(len(copies), lines)), 0., None)
    candidate_ids = np.arange(candidates) * 3 + 1
    return (*support_arrays(residuals, ANGLE_DEG), candidate_ids, residuals)


@pytest.mark.parametrize('seed', [0, 1, 2])
@pytest.mark.parametrize('rule', ['coverage', 'ranked'])
def test_buckets_preserve_selection_and_partition_the_suppressed(seed: int, rule: str) -> None:
    masks, counts, supports, candidate_ids, _ = random_support(seed)
    settings = vp_pruning.Settings(pencil_selection=rule, pencils=8)
    expected_retained, expected_status = vp_pruning.retain_pencils(masks, counts, supports, candidate_ids, settings)
    retained, status, buckets = retain_pencils_with_buckets(masks, counts, supports, candidate_ids, settings)
    assert retained == expected_retained
    assert np.array_equal(status, expected_status)
    members = [row for bucket in buckets for row in bucket['member_rows']]
    assert len(members) == len(set(members))
    assert [bucket['leader_row'] for bucket in buckets] == retained
    for bucket in buckets:
        assert bucket['leader_row'] in bucket['member_rows']
    assert set(members) == set(np.flatnonzero(np.isin(status, ['retained', 'redundant'])).tolist())


def test_precision_choice_breaks_equal_scores_by_candidate_id() -> None:
    residuals = np.array([[.2, .3, 1.], [.2, .3, 1.]])
    leader_mask = np.array([True, True, True])
    rows = np.array([0, 1])
    chosen, scores = choose_precise(residuals, leader_mask, rows, np.array([10, 4]), ANGLE_DEG)
    assert chosen == 1
    np.testing.assert_allclose(scores, scores[0])
    chosen, _ = choose_precise(residuals, leader_mask, rows, np.array([4, 10]), ANGLE_DEG)
    assert chosen == 0


def test_precision_choice_can_differ_from_the_coverage_leader() -> None:
    # The leader supports all four lines loosely; the member supports three tightly and misses one.
    residuals = np.array([[1.4, 1.4, 1.4, 1.4], [.1, .1, .1, 3.]])
    masks, counts, supports = support_arrays(residuals, ANGLE_DEG)
    candidate_ids = np.array([7, 9])
    settings = vp_pruning.Settings(pencil_selection='coverage', pencils=16, overlap=.5)
    retained, status, buckets = retain_pencils_with_buckets(masks, counts, supports, candidate_ids, settings)
    assert retained == [0] and buckets == [{'leader_row': 0, 'member_rows': [0, 1]}]
    candidates = np.array([[1., 0., 0.], [0., 1., 0.]])
    generators = [[0, 1], [2]]
    baseline = arm_record('B', buckets, residuals, masks, counts, status, candidate_ids, candidates, generators,
                          np.eye(3), ANGLE_DEG)
    precise = arm_record('R', buckets, residuals, masks, counts, status, candidate_ids, candidates, generators,
                         np.eye(3), ANGLE_DEG)
    assert baseline['representative_rule'] == LEADER_RULE and precise['representative_rule'] == PRECISION_RULE
    assert baseline['representative_candidate_ids'] == [7]
    assert precise['representative_candidate_ids'] == [9]
    assert precise['leader_candidate_ids'] == baseline['leader_candidate_ids'] == [7]
    assert precise['groups'][0]['bucket_candidate_ids'] == baseline['groups'][0]['bucket_candidate_ids']
    assert precise['support_masks'] == [[True, True, True, False]]
    assert precise['support_counts'] == [3]
    assert baseline['support_masks'] == [[True] * 4]
    assert precise['representatives_differ_from_leaders'] == 1
    assert not baseline['matcher_eligible'] and baseline['court_result']['status'] == 'empty'


def test_two_buckets_make_an_arm_matcher_eligible() -> None:
    residuals = np.array([[.1, .1, 3., 3.], [3., 3., .1, .1]])
    masks, counts, supports = support_arrays(residuals, ANGLE_DEG)
    settings = vp_pruning.Settings(pencil_selection='coverage', pencils=16)
    _, status, buckets = retain_pencils_with_buckets(masks, counts, supports, np.array([1, 2]), settings)
    record = arm_record('M', buckets, residuals, masks, counts, status, np.array([1, 2]), np.eye(2, 3), [[0], [1]],
                        np.eye(3), ANGLE_DEG)
    assert record['matcher_eligible'] and record['court_result'] is None
    assert record['direction_count'] == 2
