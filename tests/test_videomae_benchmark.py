"""CPU checks for VideoMAE inputs, optimiser updates and recovery."""
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
from experiments.annotator.video_models_benchmark.schedule import continuation_rate
from experiments.annotator.video_models_benchmark.videomae import (
    DropPath,
    VideoMAE,
    load_pretrained_weights,
    parameter_groups,
)


@pytest.fixture(autouse=True)
def cpu_threads() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def small_model(*, layers: int = 2, stochastic_depth: float = 0.2) -> VideoMAE:
    config = VideoMAEConfig(image_size=16, patch_size=8, num_frames=4, hidden_size=24,
                           intermediate_size=48, num_attention_heads=3, num_hidden_layers=layers)
    config.num_labels = 3
    config._attn_implementation = "sdpa"
    return VideoMAE(VideoMAEForVideoClassification(config), stochastic_depth)


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


def test_every_base_parameter_has_the_expected_layer_rate_and_decay() -> None:
    model = small_model(layers=12)
    assignments = {}
    for group in parameter_groups(model, 0.75, 0.05):
        for parameter in group["params"]:
            assert id(parameter) not in assignments
            assignments[id(parameter)] = group
    expected = {}
    for parameter in model.videomae.embeddings.parameters():
        expected[id(parameter)] = 0.75**13
    for index, block in enumerate(model.videomae.encoder.layer):
        for parameter in block.parameters():
            expected[id(parameter)] = 0.75 ** (12 - index)
    for module in (model.fc_norm, model.classifier):
        for parameter in module.parameters():
            expected[id(parameter)] = 1.0
    assert len(assignments) == len(expected) == 186
    for parameter in model.parameters():
        group = assignments[id(parameter)]
        assert group["lr_scale"] == pytest.approx(expected[id(parameter)])
        assert group["weight_decay"] == (0.05 if parameter.ndim > 1 else 0)


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


def test_stochastic_depth_sits_before_both_residual_additions() -> None:
    model = small_model(stochastic_depth=0.8)
    blocks = model.videomae.encoder.layer
    assert blocks[0].attention.output.dropout[1].probability == 0
    assert blocks[-1].output.dropout[1].probability == pytest.approx(0.8)
    block = blocks[-1]
    block.train()
    values = torch.randn(4, 8, 24)
    # Dropping both branch outputs must leave the incoming residual untouched.
    for module in (block.attention.output.dropout, block.output.dropout):
        module.register_forward_hook(lambda _module, _args, output: torch.zeros_like(output))
    torch.testing.assert_close(block(values), values)
    drop = DropPath(0.5).train()
    torch.manual_seed(17)
    result = drop(torch.ones(100, 8, 24))
    assert set(result.unique().tolist()) == {0, 2}
    assert torch.equal(result[:, :1, :1].expand_as(result), result)
    drop.eval()
    torch.testing.assert_close(drop(values), values)


def test_eval_matches_transformers_before_stochastic_depth() -> None:
    config = VideoMAEConfig(image_size=16, patch_size=8, num_frames=4, hidden_size=24,
                           intermediate_size=48, num_attention_heads=3, num_hidden_layers=2)
    config.num_labels = 3
    network = VideoMAEForVideoClassification(config).eval()
    clips = torch.rand(2, 4, 3, 16, 16)
    with torch.no_grad():
        expected = network(clips).logits
        actual = VideoMAE(network, 0.2).eval()(clips)
    torch.testing.assert_close(actual, expected)


def test_group_rates_survive_schedule_boundaries_and_adam_resume() -> None:
    model = small_model()
    settings = train.Settings(model_name="videomae", learning_rate=1e-4, weight_decay=0.05)
    optimiser = train.make_optimiser(model, settings)
    loss = model(torch.rand(2, 4, 3, 16, 16)).square().mean()
    loss.backward()
    optimiser.step()
    restored = train.make_optimiser(model, settings)
    restored.load_state_dict(optimiser.state_dict())
    peak = train.learning_rate_at_step(settings, 20000)
    rates = [train.learning_rate_at_step(settings, step) for step in (0, 2000, 20000, 30000)]
    rates += [continuation_rate(step, 30000, peak) for step in (30000, 35000, 40000, 50000)]
    for rate in rates:
        train.set_learning_rate(restored, rate)
        for group in restored.param_groups:
            assert group["lr"] == pytest.approx(rate * group["lr_scale"])
    assert all(state["step"] == 1 for state in restored.state.values())
    assert any(torch.count_nonzero(state["exp_avg"]) for state in restored.state.values())


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


def test_timing_reports_feeding_and_complete_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments.annotator.video_models_benchmark import timing

    monkeypatch.setattr(train, "prepare_clips", lambda clips, **_kwargs: clips.float().permute(0, 2, 1, 3, 4) / 255)
    monkeypatch.setattr(timing, "score_probabilities", lambda *_args: {})
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", list)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda *_args: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *_args: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_args: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_args: 1024**3)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_args: "CPU timing test")
    settings = train.Settings(model_name="videomae", batch_size=2, max_steps=4, warmup_steps=1,
                              validation_steps=[2, 4], workers=0, precision="float32", weight_decay=0.05)
    model = small_model()
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    labels = np.array([0, 0, 1, 2])

    class CachedClips(TensorDataset):
        cache = tmp_path

    dataset = CachedClips(torch.randint(0, 255, (4, 3, 4, 16, 16), dtype=torch.uint8), torch.from_numpy(labels))
    train.write_json(tmp_path / "index.json.gz", {"smoke_rallies": None})
    sampler = ContactBatchSampler(labels, 2, 0.5, 4, 0)
    training = DataLoader(dataset, batch_sampler=sampler)
    validation = DataLoader(dataset, batch_size=2)
    timing.time_videomae(model, training, validation, settings, torch.device("cpu"), tmp_path,
                        warmup_batches=1, measured_batches=2)
    result = train.read_json(tmp_path / "timing.json.gz")
    assert result["full_cache"] is True
    assert result["four_run_initial_hours"] == 4 * result["initial_run_hours"]
    assert result["measurements"]["validation"]["centres"] == 4
    for name in ("gpu_only", "with_loading"):
        assert result["measurements"][name]["clips_per_second"] > 0
    for name, value in initial.items():
        torch.testing.assert_close(model.state_dict()[name], value)


def test_old_x3d_settings_resume_with_new_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train, "prepare_clips", lambda clips, **_kwargs: clips)
    monkeypatch.setattr(train, "validate", lambda *_args: {"loss": 0.5, "probabilities": np.ones((4, 3)) / 3})
    monkeypatch.setattr(train, "score_probabilities", lambda *_args: {"metrics": {}})
    settings = train.Settings(batch_size=4, max_steps=4, warmup_steps=1, validation_steps=[2, 4],
                              workers=0, precision="float32")
    labels = np.array([0, 0, 1, 2])
    dataset = TensorDataset(torch.rand(4, 2), torch.from_numpy(labels))
    sampler = ContactBatchSampler(labels, 4, 0.5, 4, 0)
    loader = DataLoader(dataset, batch_sampler=sampler, generator=torch.Generator().manual_seed(11))
    model = torch.nn.Linear(2, 3)
    (tmp_path / "PAUSE").touch()
    train.fit(model, loader, loader, settings, torch.device("cpu"), tmp_path, "A")
    saved_settings = train.read_json(tmp_path / "settings.json.gz")
    for name in ("model_name", "microbatch_size", "checkpoint_revision", "layer_decay", "stochastic_depth",
                 "continuation_half_cycle_steps", "continuation_validation_interval"):
        assert name not in saved_settings
    train.write_json(tmp_path / "settings.json.gz", saved_settings)
    (tmp_path / "PAUSE").unlink()
    train.fit(model, loader, loader, settings, torch.device("cpu"), tmp_path, "A", resume=True)
    assert train.read_json(tmp_path / "status.json.gz")["step"] == 4


def test_larger_batch_can_keep_the_same_clip_exposure_and_continuation_rates() -> None:
    reference = train.Settings(model_name="videomae", learning_rate=1e-4,
                               continuation_half_cycle_steps=10000, continuation_validation_interval=5000)
    larger = replace(reference, batch_size=32, max_steps=15000, warmup_steps=1000,
                     validation_steps=[1250, 2500, 5000, 7500, 10000, 12500, 15000],
                     continuation_half_cycle_steps=5000, continuation_validation_interval=2500)
    assert reference.max_steps * reference.batch_size == larger.max_steps * larger.batch_size
    peaks = []
    rates = []
    for settings in (reference, larger):
        half_cycle = settings.continuation_half_cycle_steps
        assert half_cycle is not None
        peaks.append(train.learning_rate_at_step(settings, settings.max_steps - half_cycle))
        rates.append([continuation_rate(settings.max_steps + offset, settings.max_steps, peaks[-1],
                                        half_cycle_steps=half_cycle)
                      for offset in (0, half_cycle // 2, half_cycle, 2 * half_cycle)])
    # The cosine reaches zero on the last update, leaving a one-update rounding difference.
    assert peaks[0] == pytest.approx(peaks[1], rel=2e-4)
    assert rates[0] == pytest.approx(rates[1], rel=2e-4)


def test_continuation_uses_saved_interval_and_half_cycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = train.Settings(model_name="videomae", batch_size=4, max_steps=6, warmup_steps=2,
                              validation_steps=[2, 4, 6], continuation_half_cycle_steps=2,
                              continuation_validation_interval=1, workers=0, precision="float32")
    losses = iter([2.0, 1.0, 0.8, 0.8, 0.8, 0.8])
    monkeypatch.setattr(train, "validate", lambda *_args, **_kwargs: {
        "loss": next(losses), "probabilities": np.ones((4, 3)) / 3,
    })
    monkeypatch.setattr(train, "score_probabilities", lambda *_args: {"metrics": {}})
    recorded = []

    def update(_model, _batch, optimiser, *_args, **_kwargs) -> torch.Tensor:
        recorded.append([(group["lr"], group["lr_scale"]) for group in optimiser.param_groups])
        return torch.tensor(0.5)

    monkeypatch.setattr(train, "train_batch", update)
    labels = np.array([0, 0, 1, 2])
    dataset = TensorDataset(torch.zeros(4, 2), torch.from_numpy(labels))
    sampler = ContactBatchSampler(labels, 4, 0.5, 6, 0, continuous=True)
    loader = DataLoader(dataset, batch_sampler=sampler, generator=torch.Generator().manual_seed(11))
    validation = DataLoader(dataset, batch_size=4)
    model = small_model()
    train.fit(model, loader, validation, settings, torch.device("cpu"), tmp_path, "A")
    recorded.clear()
    train.fit(small_model(), loader, validation, settings, torch.device("cpu"), tmp_path, "A", resume=True, extend=True)
    history = train.read_json(tmp_path / "history.json.gz")
    assert [row["steps"] for row in history] == [2, 4, 6, 7, 8, 9]
    peak = train.learning_rate_at_step(settings, 4)
    for offset, groups in enumerate(recorded):
        rate = continuation_rate(6 + offset, 6, peak, half_cycle_steps=2)
        for actual, scale in groups:
            assert actual == pytest.approx(rate * scale)
    assert train.read_json(tmp_path / "status.json.gz")["bad_checks"] == 3
    assert train.read_json(tmp_path / "continuation.json.gz")["half_cycle_steps"] == 2


def test_new_head_uses_the_author_initial_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub
    import safetensors.torch
    import transformers

    from experiments.annotator.video_models_benchmark.videomae import build_videomae

    config = VideoMAEConfig(image_size=16, patch_size=8, num_frames=4, hidden_size=24,
                           intermediate_size=48, num_attention_heads=3, num_hidden_layers=2)
    config.num_labels = 400
    original = VideoMAE(VideoMAEForVideoClassification(config), 0.2)
    state = original.state_dict()
    monkeypatch.setattr(transformers, "VideoMAEConfig", lambda **_kwargs: config)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *_args, **_kwargs: "test-checkpoint")
    monkeypatch.setattr(safetensors.torch, "load_file", lambda _path: state)
    torch.manual_seed(11)
    model = build_videomae(pretrained=True, revision="test", stochastic_depth=0.2)
    assert model.classifier.weight.shape == (3, 24)
    assert 1e-5 < float(model.classifier.weight.detach().std()) < 3e-5
    assert torch.count_nonzero(model.classifier.bias) == 0
    torch.testing.assert_close(model.fc_norm.weight, original.fc_norm.weight)
