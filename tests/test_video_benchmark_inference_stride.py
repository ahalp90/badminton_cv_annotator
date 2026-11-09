"""Check stride-three sampling on absolute frames and loading of older saved settings."""
import lzma
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.annotator.video_models_benchmark import infer, train
from experiments.annotator.video_models_benchmark.data import ClipDataset
from experiments.annotator.video_models_benchmark.sampling import ContactBatchSampler
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


def test_inference_evaluates_only_selected_centres(monkeypatch: pytest.MonkeyPatch) -> None:
    # Rallies start at frame 7, so a rally-relative offset would select different centres.
    stride, offset = 3, 2
    dataset = TinyClips()
    model = CountingModel()
    monkeypatch.setattr(train, 'prepare_clips', lambda clips, **kwargs: clips)
    settings = train.Settings(batch_size=4, workers=0, precision='fp32')
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


def test_new_training_defaults_and_resume_protocol(tmp_path: Path) -> None:
    config = tmp_path / 'config.toml'
    config.write_text('model_name = "videomae"\n')
    assert train.load_settings(config, tmp_path, resume=False).validation_stride == 3
    # Records saved before stride selection have no stride and used dense validation.
    train.write_json(tmp_path / 'settings.json.gz', {})
    config.write_text('model_name = "videomae"\nvalidation_stride = 3\n')
    assert train.load_settings(config, tmp_path, resume=True).validation_stride == 1


def test_old_x3d_settings_resume_with_new_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train, 'prepare_clips', lambda clips, **_kwargs: clips)
    monkeypatch.setattr(train, 'validate', lambda *_args: {'loss': 0.5, 'probabilities': np.ones((4, 3)) / 3})
    monkeypatch.setattr(train, 'score_probabilities', lambda *_args, **_kwargs: {'metrics': {}})
    settings = train.Settings(batch_size=4, max_steps=4, warmup_steps=1, validation_steps=[2, 4],
                              workers=0, precision='float32', break_alternation_ties=True)
    labels = np.array([0, 0, 1, 2])
    dataset = TensorDataset(torch.rand(4, 2), torch.from_numpy(labels))
    sampler = ContactBatchSampler(labels, 4, 0.5, 4, 0)
    loader = DataLoader(dataset, batch_sampler=sampler, generator=torch.Generator().manual_seed(11))
    model = torch.nn.Linear(2, 3)
    (tmp_path / 'PAUSE').touch()
    train.fit(model, loader, loader, settings, torch.device('cpu'), tmp_path, 'A')
    saved_settings = train.read_json(tmp_path / 'settings.json.gz')
    # Rewrite the record as an older X3D run saved it: former tie-break name, no stride.
    saved_settings['force_alternation'] = saved_settings.pop('break_alternation_ties')
    del saved_settings['validation_stride']
    for name in ('model_name', 'microbatch_size', 'checkpoint_revision', 'layer_decay', 'stochastic_depth',
                 'continuation_half_cycle_steps', 'continuation_validation_interval'):
        assert name not in saved_settings
    train.write_json(tmp_path / 'settings.json.gz', saved_settings)
    (tmp_path / 'PAUSE').unlink()
    train.fit(model, loader, loader, settings, torch.device('cpu'), tmp_path, 'A', resume=True)
    assert train.read_json(tmp_path / 'status.json.gz')['step'] == 4
