"""Validation selection uses every seed and the report's rally judgement."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import five_seed
from common import fixed_stream, read_json, write_json
from five_seed import (
    SEEDS,
    SETTINGS,
    TRAINING_SETTINGS,
    report_scores,
    select_configuration,
    training_name,
)
from monotrack_runner import validation_courts


def candidates() -> list[dict]:
    return [{'arm': arm, 'l2': l2, 'interpolation': interpolation,
             'seeds': [{'seed': seed, 'rates': {'fully_correct': 50.0, 'player_f1': 90.0}} for seed in SEEDS]}
            for arm, l2, interpolation in SETTINGS]


def test_selection_uses_mean_not_best_seed() -> None:
    grid = candidates()
    grid[0]['seeds'][0]['rates']['fully_correct'] = 100.0
    for record in grid[1]['seeds']:
        record['rates']['fully_correct'] = 61.0
    selected = select_configuration(grid)
    assert (selected['arm'], selected['interpolation']) == ('kept', 'on')
    assert selected['mean_validation']['fully_correct'] == 61.0


def test_selection_uses_f1_only_after_fully_correct_rallies() -> None:
    grid = candidates()
    grid[6]['seeds'][0]['rates']['player_f1'] = 95.0
    assert select_configuration(grid)['arm'] == 'zeroed'
    grid[1]['seeds'][0]['rates']['fully_correct'] = 50.1
    selected = select_configuration(grid)
    assert (selected['arm'], selected['interpolation']) == ('kept', 'on')


def test_selection_requires_every_configuration_and_seed() -> None:
    grid = candidates()
    with pytest.raises(ValueError, match='complete'):
        select_configuration(grid[:-1])
    grid[0]['seeds'][-1] = copy.deepcopy(grid[0]['seeds'][0])
    with pytest.raises(ValueError, match='distinct'):
        select_configuration(grid)
    grid = candidates()
    grid[0]['seeds'][0]['rates']['fully_correct'] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        select_configuration(grid)


def test_validation_uses_report_overlap_judgement_and_keeps_failed_rallies(tmp_path: Path) -> None:
    item = {'item_id': 'validation-video', 'fps': 30, 'spans': [
        {'span_id': 0, 'rally_id': 'r1', 'start_frame': 70, 'end_frame': 151,
         'label_frames': [100, 120], 'label_sides': ['Bot', 'Top']},
        {'span_id': 1, 'rally_id': 'r2', 'start_frame': 151, 'end_frame': 210,
         'label_frames': [155, 175], 'label_sides': ['Bot', 'Top']},
        {'span_id': 2, 'rally_id': 'r3', 'start_frame': 230, 'end_frame': 300,
         'label_frames': [250, 270], 'label_sides': ['Bot', 'Top']},
    ]}
    events = [[frame, 1.0, side] for frame, side in ((100, 'Bot'), (120, 'Top'), (155, 'Bot'), (175, 'Top'))]
    write_json(tmp_path / 'validation-video.json.gz', {
        'item_id': item['item_id'], 'status': 'completed', 'stream': fixed_stream(events, item['spans']),
        'failed_rallies': [{'span_id': 2, 'reason': 'court_detection'}],
    })
    result = report_scores([item], tmp_path, {item['item_id']: []})['by_tolerance']['10']
    assert result['counts']['labelled_rallies'] == 3
    assert result['counts']['labelled_contacts'] == 6
    assert result['counts']['fully_correct_strict'] == 2
    assert result['counts']['fully_correct_matched'] == 1
    assert result['rates']['fully_correct'] == pytest.approx(100 / 3)
    write_json(tmp_path / 'report_scores.json.gz', result)


def test_validation_courts_keep_failures_and_reject_unfinished(tmp_path: Path) -> None:
    item = {'item_id': 'v', 'spans': [{'span_id': 0}, {'span_id': 1}]}
    write_json(tmp_path / 'v_0000.json.gz', {'status': 'completed', 'corners_native_px': [[1, 2]]})
    write_json(tmp_path / 'v_0001.json.gz', {'status': 'failed'})
    records = validation_courts(tmp_path, item)
    assert [record['scene_valid'] for record in records] == [True, False]
    write_json(tmp_path / 'v_0001.json.gz', {'status': 'running'})
    with pytest.raises(ValueError, match='unfinished'):
        validation_courts(tmp_path, item)


def test_workflow_freezes_validation_selection_before_test(tmp_path: Path, monkeypatch) -> None:
    item = {'item_id': 'fixture', 'fps': 30, 'spans': [
        {'span_id': 0, 'rally_id': 'r1', 'start_frame': 70, 'end_frame': 151,
         'label_frames': [100, 120], 'label_sides': ['Bot', 'Top']},
    ]}
    trials, output = tmp_path / 'trials', tmp_path / 'output'
    for arm, l2 in TRAINING_SETTINGS:
        for seed in SEEDS:
            directory = trials / training_name(arm, l2) / f'seed-{seed}'
            directory.mkdir(parents=True)
            write_json(directory / 'summary.json.gz', {
                'arm': arm, 'seed': seed, 'dropout': 0, 'l2': l2, 'smoke': False, 'best_val_loss': 0.3,
            })
            (directory / 'hitnet.h5').write_bytes(b'checkpoint fixture')
            (directory / 'history.csv.gz').write_bytes(b'history fixture')
    monkeypatch.setattr(five_seed, 'load_config', lambda _: {'paths': {'inputs': tmp_path}})
    monkeypatch.setattr(five_seed, 'evaluation_items', lambda *_: [item])
    monkeypatch.setattr(five_seed, 'test_left_out', lambda: {'fixture': []})
    monkeypatch.setattr(five_seed, 'score_run', lambda *_: None)
    phases = []

    def inference(_config_file, _config, _trials, destination, split, settings, _parallel):
        phases.append(split)
        if split == 'test':
            selection = read_json(destination / 'selected.json.gz')
            assert (selection['arm'], selection['l2'], selection['interpolation']) == ('zeroed', 1e-4, 'on')
            assert len(settings) == 10
        for arm, l2, interpolation, court, seed in settings:
            folder = f'{training_name(arm, l2)}_{interpolation}' if split == 'validation' else court
            directory = destination / split / folder / f'seed-{seed}'
            events = []
            if (arm, l2, interpolation) == ('zeroed', 1e-4, 'on'):
                events = [[100, 1.0, 'Bot'], [120, 1.0, 'Top']]
            write_json(directory / 'fixture.json.gz', {
                'item_id': 'fixture', 'status': 'completed', 'stream': fixed_stream(events, item['spans']),
            })

    monkeypatch.setattr(five_seed, 'run_grid', inference)
    five_seed.evaluate(tmp_path / 'config.toml', trials, output, 2)
    assert phases == ['validation', 'test']
    assert (output / 'complete').exists()
    assert len(list((output / 'weights').glob('*/*/hitnet.h5'))) == 30
    assert len(list((output / 'test').glob('*/*/report_scores.json.gz'))) == 10
