import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
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


def test_sparse_stride_figure_shows_only_stride_three_offset_zero(monkeypatch: pytest.MonkeyPatch,
                                                                   tmp_path: Path) -> None:
    members = runs()
    scores = sparse_scores(members)
    captured = []
    monkeypatch.setattr(figures, 'save', lambda figure, _out, _stem: captured.append(figure))

    figures.stride_figure(scores, members, {0.0001: figures.VIDEOMAE_COLOUR}, ['Seed 0', 'Seed 1'],
                          'VideoMAE-B held-out test: stride 3, offset 0', (), tmp_path, 'sparse-test')

    figure = captured[0]
    legend_labels = [text.get_text() for text in figure.legends[0].get_texts()]
    assert 'Stride 3, offset 0' in legend_labels
    assert not any('Dense' in label or 'offset 1' in label or 'offset 2' in label for label in legend_labels)
    for axes in figure.axes:
        assert len(axes.lines) == 0
        assert [collection.get_offsets()[0, 0] for collection in axes.collections] == [0, 1]
    assert figure._suptitle.get_text() == 'VideoMAE-B held-out test: stride 3, offset 0'
    plt.close(figure)


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


def test_video_system_label_names_sparse_test_sampling() -> None:
    label = figures.video_system_label(0.0001, 2, (3, 0))

    assert label == 'VideoMAE-B test, stride 3, offset 0\nLR 1e-4 (2 seeds)'
    assert figures.sampling_label((1, 0)) == 'dense (every frame)'
