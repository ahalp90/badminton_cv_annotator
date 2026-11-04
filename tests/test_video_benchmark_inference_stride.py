"""Check that sparse inference skips model calls and preserves absolute-frame scoring."""
import lzma
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.annotator.video_models_benchmark import infer, train
from experiments.annotator.video_models_benchmark.data import ClipDataset
from experiments.annotator.video_models_benchmark.evaluate import score_probabilities
from experiments.annotator.video_models_benchmark.sampling import EvaluationSampler
from experiments.annotator.video_models_benchmark.writeup_scripts import (
    rescore_videomae as rescoring,
)


class TinyClips(ClipDataset):
    def __init__(self) -> None:
        frames = np.concatenate((np.arange(7, 18), np.arange(25, 36)))
        self.rows = np.column_stack((np.zeros(len(frames)), np.zeros(len(frames)), frames)).astype(np.int32)
        self.labels = np.ones(len(frames), dtype=np.int64)
        self.labels[2] = -100
        self.videos = [{'item_id': 'video', 'role': 'validation', 'fps': 30, 'spans': [
            {'span_id': index, 'rally_id': f'rally{index}', 'start_frame': start, 'end_frame': start + 11,
             'label_frames': [start + 1], 'label_sides': ['Bot']}
            for index, start in enumerate((7, 25))
        ]}]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.full((3, 16, 2, 2), float(self.rows[index, 2])), int(self.labels[index])


class CountingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.centres: list[int] = []

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        self.centres.extend(int(value) for value in clips[:, 0, 0, 0, 0])
        return torch.tensor([0.0, 4.0, 0.0]).repeat(len(clips), 1)


@pytest.mark.parametrize(('stride', 'offset'), [(1, 0), (3, 0), (3, 2)])
@pytest.mark.parametrize('model_name', ['x3d', 'videomae'])
def test_inference_evaluates_only_selected_centres(
    monkeypatch: pytest.MonkeyPatch, stride: int, offset: int, model_name: str,
) -> None:
    dataset = TinyClips()
    model = CountingModel()
    monkeypatch.setattr(train, 'prepare_clips', lambda clips, **kwargs: clips)
    settings = train.Settings(model_name=model_name, batch_size=4, workers=0, precision='fp32')
    result, probabilities = infer.evaluate(model, dataset, settings, torch.device('cpu'), stride, offset)
    selected = dataset.rows[:, 2] % stride == offset
    assert model.centres == dataset.rows[selected, 2].tolist()
    assert len(dataset) == 22
    assert result['sampling']['evaluated_centres'] == int(selected.sum())
    assert result['sampling']['stride'] == stride
    assert result['sampling']['offset'] == offset
    assert np.asarray(result['confusion']).sum() == np.count_nonzero(dataset.labels[selected] != -100)
    np.testing.assert_array_equal(probabilities[~selected], np.tile([1, 0, 0], ((~selected).sum(), 1)))
    assert (probabilities[selected].argmax(axis=1) == 1).all()


@pytest.mark.parametrize('model_name', ['x3d', 'videomae'])
def test_training_validation_keeps_full_event_denominators(
    monkeypatch: pytest.MonkeyPatch, model_name: str,
) -> None:
    dataset = TinyClips()
    model = CountingModel()
    monkeypatch.setattr(train, 'prepare_clips', lambda clips, **kwargs: clips)
    settings = train.Settings(model_name=model_name, batch_size=4, workers=0, precision='fp32', validation_stride=3)
    loader = train.make_loader(dataset, settings, training=False)
    result = train.validate(model, loader, torch.device('cpu'), settings.precision, model_name=model_name)
    selected = dataset.rows[:, 2] % 3 == 0
    assert model.centres == dataset.rows[selected, 2].tolist()
    assert result['sampling']['evaluated_centres'] == 6
    assert np.asarray(result['confusion']).sum() == 5
    expected_loss = torch.nn.functional.cross_entropy(torch.tensor([[0., 4., 0.]]), torch.tensor([1]))
    assert result['loss'] == pytest.approx(expected_loss.item())
    events = score_probabilities(dataset, result['probabilities'])
    assert events['metrics']['10']['labelled'] == 2
    assert events['metrics']['10']['rallies'] == 2


def test_fit_saves_sparse_validation_settings_history_and_arrays(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = TinyClips()
    dataset.labels[::4] = 0
    monkeypatch.setattr(train, 'prepare_clips', lambda clips, **kwargs: clips)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 16 * 2 * 2, 3))
    with torch.no_grad():
        model[1].weight.zero_()
        model[1].bias.copy_(torch.tensor([0., 4., 0.]))
    settings = train.Settings(batch_size=4, workers=0, precision='fp32', validation_stride=3,
                              max_steps=1, warmup_steps=0, validation_steps=[1], learning_rate=1e-6)
    train.fit(model, train.make_loader(dataset, settings, training=True),
              train.make_loader(dataset, settings, training=False), settings, torch.device('cpu'), tmp_path, 'A')
    saved = train.read_json(tmp_path / 'validation_events.json.gz')
    history = train.read_json(tmp_path / 'history.json.gz')
    assert saved['sampling'] == history[0]['validation']['sampling']
    assert saved['sampling']['evaluated_centres'] == 6
    assert saved['metrics']['10']['labelled'] == 2
    assert train.read_json(tmp_path / 'settings.json.gz')['validation_stride'] == 3
    assert torch.load(tmp_path / 'best.pt', weights_only=True)['settings']['validation_stride'] == 3
    with lzma.open(tmp_path / 'validation_probabilities.npy.xz', 'rb') as stream:
        probabilities = np.load(stream)
    assert probabilities.shape == (22, 3)
    skipped = dataset.rows[:, 2] % 3 != 0
    np.testing.assert_array_equal(probabilities[skipped], np.tile([1, 0, 0], (skipped.sum(), 1)))
    manifest = {'runs': [{'name': tmp_path.name, 'seed': 0, 'learning_rate': settings.learning_rate}]}
    job = rescoring.validation_jobs(tmp_path.parent, manifest, dataset.videos, None)[0]
    assert job['sampling']['stride'] == 3
    rows = rescoring.rescore(job)
    assert {(row['stride'], row['offset']) for row in rows} == {(3, 0)}


@pytest.mark.parametrize('model_name', ['x3d', 'videomae'])
@pytest.mark.parametrize('saved_stride', [None, 1, 3])
def test_new_training_defaults_and_resume_protocol(
    tmp_path: Path, model_name: str, saved_stride: int | None,
) -> None:
    config = tmp_path / 'config.toml'
    config.write_text(f'model_name = "{model_name}"\n')
    assert train.load_settings(config, tmp_path, resume=False).validation_stride == 3
    saved = {} if saved_stride is None else {'validation_stride': saved_stride}
    train.write_json(tmp_path / 'settings.json.gz', saved)
    config.write_text(f'model_name = "{model_name}"\nvalidation_stride = 3\n')
    assert train.load_settings(config, tmp_path, resume=True).validation_stride == (saved_stride or 1)
    config.write_text(f'model_name = "{model_name}"\nvalidation_stride = 1\n')
    assert train.load_settings(config, tmp_path, resume=False).validation_stride == 1


@pytest.mark.parametrize('model_name', ['x3d', 'videomae'])
@pytest.mark.parametrize('role', ['validation', 'test'])
def test_inference_cli_defaults_to_stride_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_name: str, role: str,
) -> None:
    class StopBeforeInference(Exception):
        pass

    class FakeModel:
        def to(self, _device: torch.device) -> 'FakeModel':
            return self

        def load_state_dict(self, _state: dict) -> None:
            pass

    monkeypatch.setattr(sys, 'argv', ['infer', '--checkpoint', 'unused', '--cache', 'unused',
                                    '--output', str(tmp_path / 'test'), '--role', role])
    monkeypatch.setattr(torch, 'load', lambda *_args, **_kwargs: {
        'settings': {'model_name': model_name}, 'model': {}, 'scheme': 'A',
    })
    monkeypatch.setattr(infer, 'model_from_settings', lambda *_args, **_kwargs: FakeModel())
    monkeypatch.setattr(infer, 'ClipDataset', lambda *_args, **_kwargs: TinyClips())

    def capture(_model, _dataset, _settings, _device, stride, offset, _progress) -> None:
        assert (stride, offset) == (3, 0)
        raise StopBeforeInference

    monkeypatch.setattr(infer, 'evaluate', capture)
    with pytest.raises(StopBeforeInference):
        infer.main()


@pytest.mark.parametrize(('stride', 'offset'), [(0, 0), (3, -1), (3, 3)])
def test_invalid_sampling_fails(stride: int, offset: int) -> None:
    with pytest.raises(ValueError, match='Stride'):
        EvaluationSampler(TinyClips().rows[:, 2], stride, offset)


@pytest.mark.parametrize('predict_contacts', [True, False])
def test_sparse_saved_array_is_not_rescored_as_dense_or_other_offsets(tmp_path: Path, predict_contacts: bool) -> None:
    video = {'item_id': 'example', 'fps': 30, 'spans': [
        {'span_id': 0, 'rally_id': 'rally', 'start_frame': 7, 'end_frame': 40,
         'label_frames': [12, 30], 'label_sides': ['Bot', 'Top']},
    ]}
    probabilities = np.tile([1.0, 0.0, 0.0], (33, 1))
    if predict_contacts:
        probabilities[12 - 7] = [0, 1, 0]
        probabilities[30 - 7] = [0, 0, 1]
    path = tmp_path / 'probabilities.npy.xz'
    with lzma.open(path, 'wb') as stream:
        np.save(stream, probabilities)
    sequences = rescoring.decode_video(video, probabilities, 3, 0)
    metrics = {}
    for tolerance in (5, 10):
        counts = rescoring.score_video_rules(video, sequences, tolerance)['alternating']
        metrics[str(tolerance)] = rescoring.add_rates(counts)
    rows = rescoring.rescore({'videos': [video], 'probabilities': path, 'saved_metrics': metrics,
                             'sampling': {'stride': 3, 'offset': 0}, 'streams': None,
                             'label': {'split': 'test', 'run': 'example'}})
    assert len(rows) == 4
    assert {(row['stride'], row['offset']) for row in rows} == {(3, 0)}
    assert all(row['fully_correct'] == int(predict_contacts) for row in rows)
    assert all(row['contact_precision'] == (100 if predict_contacts else 0) for row in rows)
