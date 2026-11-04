"""Check that sparse inference skips model calls and preserves absolute-frame scoring."""
import lzma
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.annotator.video_models_benchmark import infer, train
from experiments.annotator.video_models_benchmark.data import ClipDataset
from experiments.annotator.video_models_benchmark.writeup_scripts import (
    rescore_videomae as rescoring,
)


class TinyClips(ClipDataset):
    def __init__(self) -> None:
        frames = np.concatenate((np.arange(7, 18), np.arange(25, 36)))
        self.rows = np.column_stack((np.zeros(len(frames)), np.zeros(len(frames)), frames)).astype(np.int32)
        self.labels = np.ones(len(frames), dtype=np.int64)
        self.labels[2] = -100

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
def test_inference_evaluates_only_selected_centres(monkeypatch: pytest.MonkeyPatch, stride: int, offset: int) -> None:
    dataset = TinyClips()
    model = CountingModel()
    monkeypatch.setattr(train, 'prepare_clips', lambda clips, **kwargs: clips)
    settings = train.Settings(model_name='videomae', batch_size=4, workers=0, precision='fp32')
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


def test_only_videomae_test_defaults_to_stride_three() -> None:
    assert infer.inference_stride('videomae', 'test', None) == 3
    assert infer.inference_stride('videomae', 'validation', None) == 1
    assert infer.inference_stride('x3d', 'test', None) == 1
    assert infer.inference_stride('videomae', 'test', 1) == 1


@pytest.mark.parametrize(('stride', 'offset'), [(0, 0), (3, -1), (3, 3)])
def test_invalid_sampling_fails(stride: int, offset: int) -> None:
    with pytest.raises(ValueError, match='Stride'):
        infer.sampled_dataset(TinyClips(), stride, offset)


def test_sparse_saved_array_is_not_rescored_as_dense_or_other_offsets(tmp_path: Path) -> None:
    video = {'item_id': 'example', 'fps': 30, 'spans': [
        {'span_id': 0, 'rally_id': 'rally', 'start_frame': 7, 'end_frame': 40,
         'label_frames': [12, 30], 'label_sides': ['Bot', 'Top']},
    ]}
    probabilities = np.tile([1.0, 0.0, 0.0], (33, 1))
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
    assert all(row['fully_correct'] == 1 for row in rows)
