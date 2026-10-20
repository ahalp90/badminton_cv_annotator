from __future__ import annotations

from copy import deepcopy

import pytest

from experiments.annotator.tree_simplification.review_thresholds import (
    SEED_OFFSETS,
    choose_review_thresholds,
    evaluate_thresholds,
)


def review_rows(
    split: str,
    scores: list[float | None],
    *,
    candidate: str = 'candidate_a',
    ranking: str = 'fitted',
) -> list[dict[str, object]]:
    rows = []
    for seed in SEED_OFFSETS:
        for index, score in enumerate(scores):
            correct = (-1, 0, 1)[index % 3]
            for tolerance in (5, 10):
                rows.append({
                    'candidate': candidate,
                    'seed_offset': seed,
                    'split': split,
                    'video': 'video_a' if index < 3 else 'video_b',
                    'span': index,
                    'tolerance': tolerance,
                    'ranking': ranking,
                    'correct': correct if tolerance == 10 else (1 - correct if correct != -1 else -1),
                    'score': score,
                })
    return rows


def threshold_entry(thresholds: dict[str, object], name: str) -> dict[str, object]:
    entry = thresholds['candidate_rankings'][0]
    return next(threshold for threshold in entry['thresholds'] if threshold['name'] == name)


def test_choose_uses_scores_only_and_prefers_greater_coverage_on_ties() -> None:
    scores = [0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5]
    rows = review_rows('validation', scores)
    changed_labels = deepcopy(rows)
    for row in changed_labels:
        row['correct'] = -1

    thresholds = choose_review_thresholds(rows)
    changed_thresholds = choose_review_thresholds(changed_labels)
    selected = threshold_entry(thresholds, 'retain_0.7')
    assert selected['cutoff'] == 0.2
    assert selected['achieved_validation_coverage'] == 0.8
    assert threshold_entry(changed_thresholds, 'retain_0.7')['cutoff'] == selected['cutoff']
    assert len(thresholds['candidate_rankings'][0]['validation_tolerance_10_tradeoff']) == 8
    assert set(selected['validation_by_tolerance']) == {'5', '10'}
    assert selected['validation_by_tolerance']['5']['pooled']['total'] == 50


def test_apply_keeps_missing_for_review_and_does_not_adapt_cutoff() -> None:
    thresholds = choose_review_thresholds(review_rows('validation', [0.1, 0.3, 0.5, 0.7, 0.9]))
    shifted_rows = review_rows('test', [0.01, 0.02, 0.03, 0.04, None])
    for row in shifted_rows:
        if row['span'] == 4:
            row['correct'] = -1
    result = evaluate_thresholds(shifted_rows, thresholds)
    reports = {report['threshold']: report for report in result['reports']}
    threshold = reports['retain_0.5']
    assert threshold['cutoff'] == 0.5
    assert threshold['achieved_validation_coverage'] == 0.6
    for tolerance in ('5', '10'):
        pooled = threshold['by_tolerance'][tolerance]['pooled']
        assert (pooled['total'], pooled['scored'], pooled['missing']) == (25, 20, 5)
        assert (pooled['retained'], pooled['accepted'], pooled['filtered'], pooled['manual_review']) == (5, 0, 20, 5)
        assert pooled['retained_judgement']['unknown'] == 5
        assert pooled['accepted_judgement']['error'] is None
        assert threshold['by_tolerance'][tolerance]['by_seed']['0']['by_video']['video_b']['manual_review'] == 1

    unfiltered = reports['unfiltered']['by_tolerance']['10']['pooled']
    assert (unfiltered['retained'], unfiltered['accepted'], unfiltered['filtered']) == (25, 0, 0)
    assert unfiltered['manual_review'] == 25


def test_choose_requires_validation_five_seeds_and_aligned_tolerances() -> None:
    rows = review_rows('validation', [0.2, 0.8])
    with pytest.raises(ValueError, match='validation'):
        choose_review_thresholds(review_rows('test', [0.2, 0.8]))
    with pytest.raises(ValueError, match='seed offsets'):
        choose_review_thresholds([row for row in rows if row['seed_offset'] != 400])

    changed_score = deepcopy(rows)
    changed_score[0]['score'] = 0.7
    with pytest.raises(ValueError, match='different scores'):
        choose_review_thresholds(changed_score)

    missing_tolerance = [row for row in rows if not (row['seed_offset'] == 0 and row['tolerance'] == 5)]
    with pytest.raises(ValueError, match='both tolerances'):
        choose_review_thresholds(missing_tolerance)


def test_duplicate_full_keys_are_counted_once() -> None:
    rows = review_rows('validation', [0.2, 0.8])
    thresholds = choose_review_thresholds(rows + [deepcopy(rows[0])])
    assert thresholds['validation_row_count'] == len(rows)


def test_apply_rejects_unknown_candidate_or_ranking() -> None:
    thresholds = choose_review_thresholds(review_rows('validation', [0.2, 0.8]))
    with pytest.raises(ValueError, match='candidate/ranking set'):
        evaluate_thresholds(review_rows('test', [0.2, 0.8], candidate='unknown'), thresholds)
    with pytest.raises(ValueError, match='candidate/ranking set'):
        evaluate_thresholds(review_rows('test', [0.2, 0.8], ranking='unknown'), thresholds)


def test_apply_rejects_malformed_threshold_policy() -> None:
    thresholds = choose_review_thresholds(review_rows('validation', [0.2, 0.8]))
    test_rows = review_rows('test', [0.2, 0.8])

    missing_unfiltered = deepcopy(thresholds)
    missing_unfiltered['candidate_rankings'][0]['thresholds'].pop(0)
    with pytest.raises(ValueError, match='threshold list'):
        evaluate_thresholds(test_rows, missing_unfiltered)

    null_target_cutoff = deepcopy(thresholds)
    null_target_cutoff['candidate_rankings'][0]['thresholds'][1]['cutoff'] = None
    with pytest.raises(ValueError, match='only the unfiltered'):
        evaluate_thresholds(test_rows, null_target_cutoff)

    wrong_tolerance = deepcopy(thresholds)
    wrong_tolerance['policy']['threshold_tolerance'] = 5
    with pytest.raises(ValueError, match='tolerance 10'):
        evaluate_thresholds(test_rows, wrong_tolerance)
