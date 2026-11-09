"""CPU checks for VideoMAE inputs, pretrained-weight loading, accumulation and recovery."""
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import VideoMAEConfig, VideoMAEForVideoClassification

from experiments.annotator.video_models_benchmark import train
from experiments.annotator.video_models_benchmark.model import prepare_clips
from experiments.annotator.video_models_benchmark.sampling import ContactBatchSampler
from experiments.annotator.video_models_benchmark.videomae import (
    VideoMAE,
    load_pretrained_weights,
)


@pytest.fixture(autouse=True)
def cpu_threads() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def small_model() -> VideoMAE:
    config = VideoMAEConfig(image_size=16, patch_size=8, num_frames=4, hidden_size=24,
                           intermediate_size=48, num_attention_heads=3, num_hidden_layers=2)
    config.num_labels = 3
    config._attn_implementation = "sdpa"
    return VideoMAE(VideoMAEForVideoClassification(config), stochastic_depth=0.2)


def test_input_preserves_frame_channel_identity_and_letterbox() -> None:
    frame_values = torch.arange(16).view(1, 1, 16, 1, 1)
    channel_values = torch.tensor([30, 80, 140]).view(1, 3, 1, 1, 1)
    clips = (frame_values + channel_values).expand(2, 3, 16, 256, 256).to(torch.uint8).clone()
    clips[..., :32, :] = 0
    prepared = prepare_clips(clips, training=False, model_name="videomae")
    assert prepared.shape == (2, 16, 3, 224, 224)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    expected = ((frame_values + channel_values).float() / 255).squeeze(-1).squeeze(-1).permute(0, 2, 1)
    torch.testing.assert_close(prepared[:, :, :, 100, 100], ((expected - mean) / std).expand(2, -1, -1))
    torch.testing.assert_close(prepared[:, :, :, 0, 0], (-mean / std).expand(2, 16, 3))


def test_legacy_query_value_biases_load_and_missing_weights_fail() -> None:
    original = small_model()
    state = {}
    for name, value in original.state_dict().items():
        name = name.replace(".attention.query.bias", ".attention.q_bias")
        name = name.replace(".attention.value.bias", ".attention.v_bias")
        state[name] = torch.rand_like(value)
    restored = small_model()
    load_pretrained_weights(restored, state)
    for index, block in enumerate(restored.videomae.encoder.layer):
        attention = block.attention.attention
        assert attention.key.bias is None
        prefix = f"videomae.encoder.layer.{index}.attention.attention"
        torch.testing.assert_close(attention.query.bias, state[f"{prefix}.q_bias"])
        torch.testing.assert_close(attention.value.bias, state[f"{prefix}.v_bias"])
    state.pop("fc_norm.weight")
    with pytest.raises(RuntimeError, match="Missing key"):
        load_pretrained_weights(restored, state)


def test_accumulation_averages_unequal_microbatches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train, "prepare_clips", lambda clips, **_kwargs: clips)
    torch.manual_seed(7)
    full = torch.nn.Linear(4, 3)
    split = torch.nn.Linear(4, 3)
    split.load_state_dict(full.state_dict())
    batch = torch.rand(5, 4), torch.tensor([0, 1, 2, 0, 1])
    losses = []
    for model, microbatch in ((full, None), (split, 2)):
        optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
        losses.append(train.train_batch(model, batch, optimiser, torch.device("cpu"), "float32",
                                       model_name="videomae", microbatch_size=microbatch))
        assert all(state["step"] == 1 for state in optimiser.state.values())
    torch.testing.assert_close(losses[0], losses[1])
    for left, right in zip(full.parameters(), split.parameters(), strict=True):
        torch.testing.assert_close(left, right)


def test_transformer_training_recovers_across_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train, "prepare_clips", lambda clips, **_kwargs: clips.float().permute(0, 2, 1, 3, 4) / 255)
    monkeypatch.setattr(train, "score_probabilities", lambda *_args: {"metrics": {}})
    settings = train.Settings(model_name="videomae", batch_size=4, microbatch_size=2, max_steps=4,
                              warmup_steps=1, validation_steps=[2, 4], workers=0, precision="float32",
                              learning_rate=1e-3, weight_decay=0.05)
    labels = np.array([0, 0, 1, 2])
    clips = torch.arange(4 * 3 * 4 * 16 * 16).reshape(4, 3, 4, 16, 16).remainder(255).to(torch.uint8)
    dataset = TensorDataset(clips, torch.from_numpy(labels))

    def loader() -> DataLoader:
        sampler = ContactBatchSampler(labels, 4, 0.5, 4, 0)
        return DataLoader(dataset, batch_sampler=sampler, generator=torch.Generator().manual_seed(11))

    validation = DataLoader(dataset, batch_size=2)
    for name, pause in (("whole", False), ("resumed", True)):
        train.seed_everything(9)
        model = small_model()
        output = tmp_path / name
        output.mkdir()
        if pause:
            (output / "PAUSE").touch()
        train.fit(model, loader(), validation, settings, torch.device("cpu"), output, "A")
        if pause:
            assert train.read_json(output / "status.json.gz")["step"] == 2
            (output / "PAUSE").unlink()
            train.fit(small_model(), loader(), validation, settings, torch.device("cpu"), output, "A", resume=True)
    expected = torch.load(tmp_path / "whole/last.pt", weights_only=True)
    actual = torch.load(tmp_path / "resumed/last.pt", weights_only=True)
    assert actual["step"] == expected["step"] == 4
    assert {group["lr_scale"] for group in actual["optimiser"]["param_groups"]} == {0.75**layer for layer in range(4)}
    assert {group["weight_decay"] for group in actual["optimiser"]["param_groups"]} == {0.0, settings.weight_decay}
    for name, value in expected["model"].items():
        torch.testing.assert_close(actual["model"][name], value)
    for index, state in expected["optimiser"]["state"].items():
        torch.testing.assert_close(actual["optimiser"]["state"][index]["exp_avg"], state["exp_avg"])
    assert [group["lr_scale"] for group in actual["optimiser"]["param_groups"]] == [
        group["lr_scale"] for group in expected["optimiser"]["param_groups"]
    ]
    assert len(train.read_json(tmp_path / "resumed/history.json.gz")) == 2
    with pytest.raises(ValueError, match="Resume settings differ"):
        train.fit(small_model(), loader(), validation, replace(settings, stochastic_depth=0.1),
                  torch.device("cpu"), tmp_path / "resumed", "A", resume=True)
