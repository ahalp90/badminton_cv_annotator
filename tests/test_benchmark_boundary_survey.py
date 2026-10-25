"""Fresh benchmark inputs and exhaustive validation-only threshold selection."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from annotator.outcomes.point_winner import Half
from annotator.sequence.contacts import ContactEvent, ContactSequence, stream_payload
from annotator.training.sequences import RallyBoundaryPolicy
from experiments.annotator.monotrack_benchmark.writeup_scripts import (
    annotator_runs,
    confidence_cutoffs,
    metrics,
)


def found_rows(scores: list[float], outcomes: list[str]) -> pd.DataFrame:
    rows = []
    for seed in metrics.SEEDS:
        for identity, (score, outcome) in enumerate(zip(scores, outcomes, strict=True)):
            rows.append({'seed': seed, 'video': 'video', 'rally_id': str(identity),
                         'confidence': score, 'outcome': outcome})
    return pd.DataFrame(rows)


def test_full_scan_finds_best_cutoff_between_old_grid_points() -> None:
    found = found_rows([0.501, 0.502, 0.503], ['extra_hits', 'fully_correct', 'fully_correct'])
    thresholds = confidence_cutoffs.attainable_thresholds(found)
    np.testing.assert_array_equal(thresholds, [0.501, 0.502, 0.503])
    summary = confidence_cutoffs.seed_summary(confidence_cutoffs.curve(found, 3, thresholds))
    chosen = confidence_cutoffs.choose_cutoffs(summary)
    assert chosen['balanced'] == 0.502
    assert chosen['keep_half'] == 0.502
    assert summary.loc[0.502, ('f1', 'mean')] == 100


def test_unattainable_99_percent_is_explicitly_absent() -> None:
    found = found_rows([0.1, 0.9], ['fully_correct', 'overlong_rally'])
    summary = confidence_cutoffs.seed_summary(
        confidence_cutoffs.curve(found, 2, confidence_cutoffs.attainable_thresholds(found)),
    )
    assert confidence_cutoffs.choose_cutoffs(summary)['boundaries_99'] is None
    assert confidence_cutoffs.at_cutoff(summary, None) is None


def test_tied_scores_are_indivisible_and_tied_objectives_choose_lower_cutoff() -> None:
    found = found_rows([0.2, 0.2, 0.8], ['extra_hits', 'extra_hits', 'extra_hits'])
    thresholds = confidence_cutoffs.attainable_thresholds(found)
    np.testing.assert_array_equal(thresholds, [0.2, 0.8])
    curves = confidence_cutoffs.curve(found, 3, thresholds)
    assert curves[curves['threshold'] == 0.2]['kept'].tolist() == [3] * 5
    assert confidence_cutoffs.choose_cutoffs(confidence_cutoffs.seed_summary(curves))['balanced'] == 0.2


def test_boundary_target_uses_mean_seed_precision_and_requires_every_seed() -> None:
    found = found_rows([0.4], ['fully_correct'])
    additions = pd.DataFrame([
        {'seed': 0, 'video': 'large', 'rally_id': str(identity), 'confidence': 0.4,
         'outcome': 'fully_correct' if identity < 995 else 'overlong_rally'}
        for identity in range(1000)
    ])
    # At .4 pooled precision exceeds 99%, while one sparse seed is completely wrong.
    found.loc[found['seed'] == 100, 'outcome'] = 'overlong_rally'
    found.loc[found['seed'] == 200, 'confidence'] = 0.8
    found = pd.concat([found, additions], ignore_index=True)
    summary = confidence_cutoffs.seed_summary(confidence_cutoffs.curve(found, 1001, [0.4, 0.8]))
    assert summary.loc[0.4, ('pooled_boundaries_right_share', 'mean')] >= 99
    assert summary.loc[0.4, ('boundaries_right_share', 'mean')] < 99
    assert summary.loc[0.8, ('boundaries_right_share', 'mean')] == 100
    assert confidence_cutoffs.choose_cutoffs(summary)['boundaries_99'] is None


def test_cumulative_curve_counts_unique_labelled_rallies_and_matches_direct_counts() -> None:
    found = found_rows([0.1, 0.3, 0.3, 0.9], ['fully_correct', 'fully_correct', 'extra_hits', 'overlong_rally'])
    found.loc[found['rally_id'] == '1', 'rally_id'] = '0'
    curves = confidence_cutoffs.curve(found, 4, [0, 0.2, 0.3, 0.31, 1])
    for row in curves.itertuples():
        run = found[found['seed'] == row.seed]
        kept = run[run['confidence'] >= row.threshold]
        correct = kept[kept['outcome'] == 'fully_correct']
        assert row.kept == len(kept)
        assert row.fully_correct_kept == len(correct)
        assert row.coverage == 100 * correct[['video', 'rally_id']].drop_duplicates().shape[0] / 4
        denominator = len(kept) + (run['outcome'] == 'fully_correct').sum()
        assert row.f1 == pytest.approx(200 * len(correct) / denominator)


def test_fresh_model_and_supplied_paths_include_every_comparison_condition(tmp_path: Path) -> None:
    models_40, training = tmp_path / '40', tmp_path / 'training'
    run = annotator_runs.Run('test', 'our_courts', 'body_window_36', 100)
    assert annotator_runs.bundle_path(run, models_40, training) == training / 'models/body_window_36-seed-100'
    assert annotator_runs.bundle_path(annotator_runs.Run('test', 'discovery', '40_videos', 0), models_40, training) == (
        models_40 / 'seed-0'
    )
    supplied = metrics.supplied_runs(tmp_path / 'annotator', tmp_path / 'mono')
    assert len(supplied) == 9
    assert supplied['annotator_body_window_36_our_courts'][0] == (
        tmp_path / 'annotator/test/our_courts/body_window_36/seed-0'
    )


def test_boundary_policy_propagates_through_video_scoring_and_task() -> None:
    item = {'item_id': 'video', 'fps': 30.0, 'spans': [{
        'span_id': 0, 'rally_id': 'rally', 'start_frame': 0, 'end_frame': 300,
        'label_frames': [100, 200], 'label_sides': [Half.TOP.value, Half.BOT.value],
    }]}
    events = (ContactEvent(10, .9, Half.TOP), ContactEvent(100, .9, Half.TOP), ContactEvent(200, .9, Half.BOT))
    sequence = ContactSequence(0, 10, 251, events)
    stream = stream_payload((sequence,), events) | {'confidence': {'0': 0.7}}
    strict, judged = metrics.score_task((item, stream, 10, [], False, RallyBoundaryPolicy(2, 3)))
    wider, wider_judged = metrics.score_task((item, stream, 10, [], False, RallyBoundaryPolicy(3, 3)))
    assert strict['fully_correct_strict'] == strict['fully_correct_matched'] == 0
    assert judged[0]['outcome'] == 'overlong_rally'
    assert wider['boundaries_right'] == 1
    assert wider['fully_correct_strict'] == wider['fully_correct_matched'] == 0
    assert wider_judged[0]['outcome'] == 'extra_hits'
    for field in ('predicted_contacts', 'matched_contacts', 'correct_side_contacts'):
        assert strict[field] == wider[field]
    assert strict['predicted_contacts'] == 3
    assert strict['matched_contacts'] == 2


def test_saved_check_ignores_rally_totals_for_alternate_boundary_policy() -> None:
    rows = []
    saved = {'by_tolerance': {}}
    for tolerance in metrics.TOLERANCES:
        rows.append({'tolerance': tolerance, 'labelled_contacts': 2, 'predicted_contacts': 2,
                     'ignored_predictions': 1, 'matched_contacts': 2, 'correct_side_contacts': 2,
                     'fully_correct_strict_unmasked': 0})
        saved['by_tolerance'][str(tolerance)] = {'labelled_contacts': 2, 'predicted_contacts': 3,
                                               'matched_contacts': 2, 'correct_side_contacts': 2,
                                               'fully_correct_rallies': 99}
    metrics.check_against_saved('fresh', pd.DataFrame(rows), saved)
    saved['by_tolerance']['10']['matched_contacts'] = 1
    with pytest.raises(ValueError, match='matched_contacts'):
        metrics.check_against_saved('fresh', pd.DataFrame(rows), saved)


def test_cli_selection_is_unchanged_when_test_outcomes_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    found = found_rows([0.501, 0.502, 0.503], ['overlong_rally', 'fully_correct', 'fully_correct'])
    monkeypatch.setattr(confidence_cutoffs, 'validation_found', lambda *_args: (found, 3))
    metrics_root = tmp_path / 'metrics'
    metrics_root.mkdir()
    confidence_cutoffs.write_json(metrics_root / 'evaluation-policy.json.gz', {
        'boundary_policy': {'before_seconds': 2.0, 'after_seconds': 3.0},
        'boundary_basis': metrics.DISCOVERY_BOUNDARY_BASIS,
    })
    chosen = []
    for run_number, test_outcome in enumerate(('fully_correct', 'overlong_rally')):
        for seed in metrics.SEEDS:
            test = found[found['seed'] == seed].assign(outcome=test_outcome)
            test.to_csv(metrics_root / f'found_rallies_annotator_discovery_seed{seed}.csv.gz', index=False)
        output = tmp_path / f'out-{run_number}'
        monkeypatch.setattr('sys.argv', [
            'confidence_cutoffs.py', '--validation', str(tmp_path / 'fresh-validation'),
            '--metrics', str(metrics_root), '--shots-master', str(tmp_path / 'shots_master.csv'),
            '--test-rallies', '3', '--out', str(output),
        ])
        confidence_cutoffs.main()
        result = confidence_cutoffs.read_json(output / 'confidence_cutoffs.json.gz')
        chosen.append(result['cutoffs'])
        assert result['thresholds_searched'] == 3
        assert result['selection_split'] == 'validation'
        assert result['test']['balanced']['fully_correct_share_mean'] == (100 if run_number == 0 else 0)
    assert chosen[0] == chosen[1]
