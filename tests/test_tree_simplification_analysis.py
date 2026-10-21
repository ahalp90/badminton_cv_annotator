"""Smoke tests for the scripts that rebuild and check the tree simplification evidence."""

import hashlib
import json

import matplotlib
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.annotator.tree_simplification.analysis import (
    common,
    input_evidence,
    input_selection,
    plot_review_tradeoff,
    plot_target_screen,
    pool_coverage,
    profile_selected,
    screening_evidence,
    test_run,
    validation_evidence,
)
from experiments.annotator.tree_simplification.options import expand_pool
from tests.test_tree_simplification_options import example_video


def counts(correct_side: int, predicted: int, labelled: int, rallies: int = 0) -> dict[str, int]:
    return {
        'labelled_rallies': 10, 'labelled_contacts': labelled, 'predicted_contacts': predicted,
        'matched_contacts': correct_side, 'correct_side_contacts': correct_side, 'fully_correct_rallies': rallies,
    }


def test_check_counts_values_and_skips_copied_facts() -> None:
    expected = {'count': 3, 'rate': 0.5, 'nested': [{'flag': True}], 'source': 'run A'}
    actual = {'count': 3, 'rate': 0.5 + 1e-13, 'nested': [{'flag': True}], 'source': 'run B'}
    assert common.check(actual, expected, 'example', skip=('source',)) == 3


@pytest.mark.parametrize('actual', [
    {'count': 4, 'rate': 0.5},
    {'count': 3, 'rate': 0.5 + 1e-9},
    {'count': 3},
])
def test_check_fails_on_any_difference(actual: dict) -> None:
    with pytest.raises(ValueError, match='differences'):
        common.check(actual, {'count': 3, 'rate': 0.5}, 'example')


def test_json_writes_are_byte_identical(tmp_path) -> None:
    first, second = tmp_path / 'first.json.gz', tmp_path / 'nested' / 'second.json.gz'
    common.write_json_gz(first, {'b': 1, 'a': [1.5, None]})
    common.write_json_gz(second, {'a': [1.5, None], 'b': 1})
    assert first.read_bytes() == second.read_bytes()
    assert common.read_json(second) == {'a': [1.5, None], 'b': 1}


def test_svg_has_no_date_and_repeats_exactly(tmp_path) -> None:
    paths = [tmp_path / 'first.svg', tmp_path / 'second.svg']
    for path in paths:
        figure, axis = plt.subplots()
        axis.plot([1, 2], [3, 4], marker='o')
        common.save_svg(figure, path)
        plt.close(figure)
    assert paths[0].read_text() == paths[1].read_text()
    assert '<dc:date>' not in paths[0].read_text()
    common.check_svg(paths[0], paths[1])
    assert common.normalised_svg('<path id="m0123456789"/> <use xlink:href="#m9876543210"/>') == (
        '<path id="m-generated"/> <use xlink:href="#m-generated"/>'
    )


def test_seed_metrics_pool_counts_before_f1() -> None:
    per_video = {
        'first': {'10': counts(8, 10, 10, 2), '5': counts(6, 10, 10, 1)},
        'second': {'10': counts(2, 10, 10, 1), '5': counts(2, 10, 10, 0)},
    }
    assert common.seed_metrics(per_video) == {
        'f1_10': 0.5, 'fully_correct_rallies_10': 3, 'f1_5': 0.4, 'fully_correct_rallies_5': 1,
    }


def test_recorded_hash_uses_the_old_key_names() -> None:
    analysis = {'rankings': {'fully_correct_rallies': ['a']}, 'mean': {'fully_correct_rallies_10': 1}}
    old = input_selection.NEW_TO_OLD
    written = json.dumps({'rankings': {old['fully_correct_rallies']: ['a']}, 'mean': {old['fully_correct_rallies_10']: 1}},
                         indent=2) + '\n'
    assert input_selection.recorded_hash(analysis) == hashlib.sha256(written.encode()).hexdigest()


def test_rank_prefers_the_baseline_then_fewer_changes() -> None:
    mean = {'f1_10': 0.8, 'fully_correct_rallies_10': 300, 'f1_5': 0.7, 'fully_correct_rallies_5': 200}
    rows = {
        'two_changes': {'mean': mean, 'changes': 2},
        'one_change': {'mean': mean, 'changes': 1},
        'raw_baseline': {'mean': mean, 'changes': 0},
        # Only ±5 F1 differs, which the F1 objective uses and the rally objective does not.
        'better': {'mean': {**mean, 'f1_5': 0.71}, 'changes': 3},
    }
    assert input_selection.rank(rows, 'f1') == ['better', 'raw_baseline', 'one_change', 'two_changes']
    assert input_selection.rank(rows, 'fully_correct_rallies') == ['raw_baseline', 'one_change', 'two_changes', 'better']


def test_frame_rate_differences_and_ranks() -> None:
    first = {'10': counts(8, 10, 10, 3), '5': counts(6, 10, 10, 2)}
    second = {'10': counts(6, 10, 10, 1), '5': counts(6, 10, 10, 2)}
    differences = validation_evidence.rate_differences(first, second)
    assert differences['10'] == {'correct_rallies': 2, 'attributed_f1_pp': pytest.approx(20.0)}
    assert differences['5'] == {'correct_rallies': 0, 'attributed_f1_pp': 0.0}
    assert input_evidence.nearest_rank([4.0, 1.0, 3.0, 2.0], 0.5) == 2.0


def test_replace_placeholder_only_swaps_leading_run_folders() -> None:
    record = {'export': 'SWEEP/candidates/a', 'items': ['SWEEP/b', 'not SWEEP/c'], 'count': 2}
    assert test_run.replace_placeholder(record, '/data/run/') == {
        'export': '/data/run/candidates/a', 'items': ['/data/run/b', 'not SWEEP/c'], 'count': 2,
    }


def test_materialise_refuses_a_folder_in_use(tmp_path) -> None:
    (tmp_path / 'existing.txt').write_text('keep')
    with pytest.raises(FileExistsError):
        test_run.materialise(common.RESULTS, tmp_path, use_current_checkout=False)


def test_tracked_test_run_loads_all_ten_bundles(tmp_path) -> None:
    test_run.materialise(common.RESULTS, tmp_path / 'test-run', use_current_checkout=False)
    test_run.check(tmp_path / 'test-run')


def test_screening_totals_must_equal_the_recorded_totals() -> None:
    def video_counts(rallies: int) -> dict:
        return {tolerance: {**counts(9, 10, 10, 5), 'labelled_rallies': rallies} for tolerance in ('10', '5')}

    # Seven videos of 83 rallies and one of 87 give the 668 validation rallies.
    per_video = {f'sset_{index}': video_counts(83) for index in range(7)} | {'sset_7': video_counts(87)}
    recorded = common.pooled_both(per_video)
    totals = screening_evidence.totals_from_videos(per_video, recorded, 'example')
    assert totals['10']['labelled_rallies'] == 668
    recorded['10']['fully_correct_rallies'] += 1
    with pytest.raises(ValueError, match='differences'):
        screening_evidence.totals_from_videos(per_video, recorded, 'example')


def test_screening_settings_must_match_the_label() -> None:
    tree = {'backend': 'xgboost', 'max_leaf_nodes': 15, 'min_child_weight': 5.0, 'max_iter': 200, 'learning_rate': 0.05}
    original_options = screening_evidence.OLD_RECORD_NAMES['original_options']
    settings = {'tree': tree, original_options: True, 'layers': 2, 'span_weight': 371.5}
    arm = screening_evidence.Arm('run', 'xgboost', 'original', screening_evidence.TEN, 15, 5.0, 2)
    assert screening_evidence.arm_settings(settings, arm, 'run')['option_list'] == 'original'
    with pytest.raises(ValueError, match='do not match'):
        screening_evidence.arm_settings({**settings, original_options: False}, arm, 'run')


def test_diagnostics_pool_by_split_and_frame_rate() -> None:
    def video(group: str, fps: float, correct_side: int) -> dict:
        return {'group': group, 'fps': fps, 'scores': {tolerance: {'counts': counts(correct_side, 10, 10)}
                                                         for tolerance in ('10', '5')}}

    pooled = screening_evidence.pooled_by_split_and_rate(
        {'a': video('A', 25.0, 8), 'b': video('B', 25.0, 6), 'c': video('V', 30.0, 5)},
    )
    assert pooled['train']['25']['10']['correct_side_contacts'] == 14
    assert pooled['train']['25']['10']['attributed_f1'] == 0.7
    assert pooled['validation']['30']['5']['attributed_f1'] == 0.5


def test_pool_coverage_marks_every_labelled_rally_reachable() -> None:
    video = example_video()
    coverage = pool_coverage.inspect_pool(expand_pool(video))
    assert coverage['labelled_rallies'] == len(video.rallies)
    assert coverage['with_correct_option'] == len(video.rallies)
    assert coverage['bounds_impossible'] == coverage['frame_union_impossible'] == 0


def test_profile_paths_hide_the_checkout() -> None:
    inside = f'{common.REPOSITORY}/src/annotator/hybrid.py'
    assert profile_selected.placeholder_path(inside) == 'CHECKOUT/src/annotator/hybrid.py'
    assert profile_selected.placeholder_path('<string>') == '<string>'


@pytest.mark.parametrize(('module', 'evidence', 'figure'), [
    (plot_review_tradeoff, 'input-preprocessing-evidence.json.gz', 'input-review-tradeoff.svg'),
])
def test_figures_rebuild_from_tracked_evidence(tmp_path, module, evidence: str, figure: str) -> None:
    document = common.read_json(common.RESULTS / evidence)
    output = tmp_path / figure
    if module is plot_target_screen:
        module.plot(document, output)
    else:
        module.plot(document['test_review_evaluation']['reports'], output)
    common.check_svg(output, common.RESULTS / figure)
