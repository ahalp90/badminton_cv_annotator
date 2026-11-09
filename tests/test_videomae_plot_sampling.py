import gzip
import json
from pathlib import Path

import pytest

from experiments.annotator.video_models_benchmark.writeup_scripts import (
    make_videomae_figures as figures,
)


def runs() -> list[figures.PilotRun]:
    return [figures.PilotRun(f'run-seed{seed}', 0.0001, seed, 10, [], []) for seed in (0, 1)]


def sparse_scores(members: list[figures.PilotRun]) -> dict:
    scores = {}
    for run in members:
        for rule in ('raw', 'alternating'):
            for tolerance in (figures.TOLERANCE, '5'):
                scores[run.name, 3, 0, rule, tolerance] = {
                    'contact_recall': 90.0 + run.seed,
                    'contact_precision': 80.0 + run.seed,
                    'contact_f1': 85.0 + run.seed,
                    'correct_player_f1': 70.0 + run.seed + (rule == 'alternating'),
                    'fully_correct_percent': 20.0 + run.seed + (rule == 'alternating'),
                }
    return scores


def test_validation_figure_uses_stride_three_offset_zero(monkeypatch: pytest.MonkeyPatch,
                                                         tmp_path: Path) -> None:
    members = runs()
    for run in members:
        run.history.append({'steps': 10, 'validation': {'loss': 0.1},
                            'events': {figures.TOLERANCE: {'rallies': 668, 'labelled': 5696}}})
    scores = sparse_scores(members)
    for run in members:
        scores[run.name, 1, 0, 'alternating', figures.TOLERANCE] = {
            metric: 99.0 for metric, _title in figures.METRICS
        }
    captured = []
    monkeypatch.setattr(figures, 'dot_plot', lambda *args: captured.append(args))

    figures.validation_figure(members, scores, 0.0001, tmp_path)

    rows, title, notes, output = captured[0]
    assert len(rows) == 1
    assert rows[0].scores['fully_correct_percent'] == [21.0, 22.0]
    assert rows[0].scores['correct_player_f1'] == [71.0, 72.0]
    assert 'Stride 3, offset 0' in notes[0]
    assert title.startswith('VideoMAE (base) validation:')
    assert output == tmp_path / 'videomae_validation.png'


def test_sparse_stride_table_lists_only_available_conditions(capsys: pytest.CaptureFixture[str]) -> None:
    members = runs()

    figures.print_stride_table(sparse_scores(members), members, figures.TOLERANCE)

    output = capsys.readouterr().out
    assert 'Available sampling scores' in output
    assert 'Stride 3, offset 0' in output
    assert 'Dense' not in output
    assert 'offset 1' not in output
    assert 'offset 2' not in output


def write_events(path: Path, sampling: dict | None) -> None:
    payload = {} if sampling is None else {'sampling': sampling}
    with gzip.open(path, 'wt') as stream:
        json.dump(payload, stream)


def test_test_sampling_supports_historical_dense_and_rejects_seed_mismatch(tmp_path: Path) -> None:
    dense_paths = [tmp_path / 'dense0.json.gz', tmp_path / 'dense1.json.gz']
    for path in dense_paths:
        write_events(path, None)
    assert figures.test_sampling(dense_paths) == (1, 0)

    sparse_paths = [tmp_path / 'sparse0.json.gz', tmp_path / 'sparse1.json.gz']
    write_events(sparse_paths[0], {'stride': 3, 'offset': 0, 'total_centres': 100, 'evaluated_centres': 34,
                                   'unsampled_probabilities': 'background'})
    write_events(sparse_paths[1], {'stride': 3, 'offset': 1, 'total_centres': 100, 'evaluated_centres': 33,
                                   'unsampled_probabilities': 'background'})
    with pytest.raises(ValueError, match='inconsistent sampling metadata'):
        figures.test_sampling(sparse_paths)
