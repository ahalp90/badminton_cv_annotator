"""Rescore saved validation predictions without repeating inference."""
from __future__ import annotations

import itertools
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import five_seed
from common import fixed_stream, write_json
from five_seed import SEEDS, SETTINGS


def make_source(root: Path, original: tuple[str, float, str]) -> None:
    write_json(root / 'selected.json.gz', {
        'arm': original[0], 'l2': original[1], 'interpolation': original[2],
    })
    for arm, l2, interpolation in SETTINGS:
        for seed in SEEDS:
            run = five_seed.validation_directory(root, arm, l2, interpolation, seed)
            write_json(run / 'validation-fixture.json.gz', {'saved': True})
    for court, seed in itertools.product(('monotrack', 'ours'), SEEDS):
        run = root / 'test' / court / f'seed-{seed}'
        write_json(run / 'test-fixture.json.gz', {'saved': True})
        write_json(run / 'report_scores.json.gz', {'saved': True})
        write_json(run / 'scores.json.gz', {'saved': True})


def report_result(fully_correct: float = 50.0) -> dict:
    return {'by_tolerance': {'10': {
        'counts': {'labelled_rallies': 1, 'labelled_contacts': 2, 'predicted_contacts': 2,
                   'matched_contacts': 2, 'correct_side_contacts': 2, 'fully_correct_matched': 1},
        'rates': {'fully_correct': fully_correct, 'player_f1': 90.0},
    }}}


def setup_reselection(monkeypatch, source: Path, *, changed: bool) -> tuple[list, list, list]:
    validation = {'item_id': 'validation-fixture', 'fps': 30, 'spans': []}
    test = {'item_id': 'test-fixture', 'fps': 30, 'spans': []}
    config = {'paths': {'inputs': source}}
    monkeypatch.setattr(five_seed, 'load_config', lambda _: config)
    monkeypatch.setattr(five_seed, 'evaluation_items',
                        lambda _config, split: [validation] if split == 'validation' else [test])
    monkeypatch.setattr(five_seed, 'test_left_out', lambda: {'test-fixture': []})
    monkeypatch.setattr(five_seed, 'ProcessPoolExecutor', ThreadPoolExecutor)
    scored_paths, inference_calls, score_calls = [], [], []

    def fake_report(items, run, windows):
        scored_paths.append(run)
        if changed and run.name == 'seed-0' and run.parent.name == 'zeroed_l20.0001_on':
            return report_result(80.0)
        return report_result()

    def fake_run_grid(_config_file, _config, _trials, output, split, settings, _parallel):
        inference_calls.append((split, settings))
        for *_model, court, seed in settings:
            run = output / split / court / f'seed-{seed}'
            write_json(run / 'test-fixture.json.gz', {'item_id': 'test-fixture', 'status': 'completed'})

    def fake_score_run(_inputs, run):
        score_calls.append(run)
        write_json(run / 'scores.json.gz', {'scored': True})

    monkeypatch.setattr(five_seed, 'report_scores', fake_report)
    monkeypatch.setattr(five_seed, 'run_grid', fake_run_grid)
    monkeypatch.setattr(five_seed, 'score_run', fake_score_run)
    return scored_paths, inference_calls, score_calls


def test_unchanged_selection_reuses_saved_test_runs(tmp_path: Path, monkeypatch) -> None:
    source, output = tmp_path / 'source', tmp_path / 'selection'
    make_source(source, SETTINGS[0])
    scored_paths, inference_calls, score_calls = setup_reselection(monkeypatch, source, changed=False)

    selected, changed = five_seed.reselect_saved_validation(
        tmp_path / 'config.toml', None, source, output, 8, selection_only=False,
    )

    assert not changed
    assert (selected['arm'], selected['l2'], selected['interpolation']) == SETTINGS[0]
    assert len(scored_paths) == len(SETTINGS) * len(SEEDS)
    assert len(list(output.glob('validation/*/seed-*/report_scores.json.gz'))) == 60
    assert not (output / 'test').exists()
    assert inference_calls == []
    assert score_calls == []


def test_changed_selection_resumes_scores_and_runs_only_its_test_grid(
    tmp_path: Path, monkeypatch,
) -> None:
    source, output, trials = tmp_path / 'source', tmp_path / 'selection', tmp_path / 'trials'
    make_source(source, SETTINGS[0])
    trials.mkdir()
    # One completed score survives an interrupted rescore and is not recomputed.
    cached = five_seed._score_path(output, *SETTINGS[0], SEEDS[-1])
    write_json(cached, report_result())
    scored_paths, inference_calls, score_calls = setup_reselection(monkeypatch, source, changed=True)

    selected, changed = five_seed.reselect_saved_validation(
        tmp_path / 'config.toml', None, source, output, 8, selection_only=True,
    )
    expected = ('zeroed', 1e-4, 'on')
    assert changed
    assert (selected['arm'], selected['l2'], selected['interpolation']) == expected
    assert len(scored_paths) == 59
    assert inference_calls == []

    five_seed.reselect_saved_validation(
        tmp_path / 'config.toml', trials, source, output, 8, selection_only=False,
    )

    assert len(scored_paths) == 69
    assert len(inference_calls) == 1
    split, grid = inference_calls[0]
    assert split == 'test'
    assert grid == [(*expected, court, seed) for court in ('monotrack', 'ours') for seed in SEEDS]
    assert len(score_calls) == 10

    five_seed.reselect_saved_validation(
        tmp_path / 'config.toml', trials, source, output, 8, selection_only=False,
    )
    assert len(scored_paths) == 69
    assert len(inference_calls) == 1
    assert len(score_calls) == 10


def test_incomplete_test_condition_reports_the_missing_video(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / 'selection'
    (output / 'test' / 'monotrack' / 'seed-0').mkdir(parents=True)
    monkeypatch.setattr(five_seed, 'evaluation_items', lambda _config, _split: [{'item_id': 'missing-video'}])

    with pytest.raises(ValueError, match='incomplete video output; missing missing-video'):
        five_seed.run_reselected_test(
            tmp_path / 'config.toml', {'paths': {'inputs': tmp_path}}, tmp_path / 'trials', output,
            {'arm': 'kept', 'l2': 0.0, 'interpolation': 'off'}, 1,
        )


def test_report_scoring_preserves_flagged_rally_contact_counts(tmp_path: Path) -> None:
    item = {'item_id': 'sset_1', 'fps': 30, 'spans': [
        {'span_id': 0, 'rally_id': 'set1:30', 'start_frame': 70, 'end_frame': 151,
         'label_frames': [100, 120], 'label_sides': ['Bot', 'Top']},
        {'span_id': 1, 'rally_id': 'clean-rally', 'start_frame': 160, 'end_frame': 241,
         'label_frames': [180, 200], 'label_sides': ['Bot', 'Top']},
    ]}
    events = [[frame, 1.0, side] for frame, side in (
        (100, 'Bot'), (120, 'Top'), (180, 'Bot'), (200, 'Top'),
    )]
    write_json(tmp_path / 'sset_1.json.gz', {
        'item_id': 'sset_1', 'status': 'completed', 'stream': fixed_stream(events, item['spans']),
    })

    result = five_seed.report_scores([item], tmp_path, {'sset_1': []})['by_tolerance']['10']

    assert result['counts']['labelled_rallies'] == 1
    assert result['counts']['fully_correct_matched'] == 1
    assert result['counts']['labelled_contacts'] == 4
    assert result['counts']['predicted_contacts'] == 4
    assert result['counts']['matched_contacts'] == 4
