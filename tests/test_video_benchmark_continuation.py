"""Check continuation rates, historical patience, sampling and exact recovery."""
import shutil
from itertools import islice
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.annotator.video_models_benchmark import schedule, train
from experiments.annotator.video_models_benchmark.sampling import ContactBatchSampler


def small_loader(settings: train.Settings, *, continuous: bool = False) -> DataLoader:
    labels = np.array([0, 0, 0, 0, 1, 2, 1, 2])
    dataset = TensorDataset(torch.arange(16, dtype=torch.float32).reshape(8, 2) / 16, torch.from_numpy(labels))
    sampler = ContactBatchSampler(labels, settings.batch_size, settings.contact_fraction, settings.max_steps,
                                  settings.seed, continuous=continuous)
    return DataLoader(dataset, batch_sampler=sampler, generator=torch.Generator().manual_seed(11))


def short_schedule(monkeypatch: pytest.MonkeyPatch) -> train.Settings:
    monkeypatch.setattr(train, "VALIDATION_INTERVAL", 2)
    monkeypatch.setattr(train, "HALF_CYCLE_STEPS", 4)
    monkeypatch.setattr(schedule, "HALF_CYCLE_STEPS", 4)
    monkeypatch.setattr(train, "score_probabilities", lambda *args: {"metrics": {}})
    return train.Settings(batch_size=4, max_steps=6, warmup_steps=2, validation_steps=[2, 4, 6],
                          workers=0, precision="float32", seed=7)


def validation_losses(monkeypatch: pytest.MonkeyPatch, losses: list[float]) -> None:
    values = iter(losses)
    monkeypatch.setattr(train, "validate", lambda *args: {"loss": next(values), "probabilities": np.ones((8, 3))})


@pytest.mark.parametrize("fraction", [0.25, 0.5])
def test_continuous_sampler_extends_original_stream_and_replays(fraction: float) -> None:
    labels = np.array([0] * 17 + [1, 2] * 5)
    original = ContactBatchSampler(labels, 8, fraction, 13, 1)
    longer = ContactBatchSampler(labels, 8, fraction, 31, 1)
    continuous = ContactBatchSampler(labels, 8, fraction, 13, 1, continuous=True)
    batches = list(islice(continuous, 31))
    assert batches[:13] == list(original)
    assert batches == list(longer)
    continuous.start_step = 19
    assert list(islice(continuous, 12)) == batches[19:]


def test_continuation_rate_uses_the_original_20k_rate_and_repeats() -> None:
    settings = train.Settings()
    peak = train.learning_rate_at_step(settings, 20000)
    assert peak == pytest.approx(0.005660512742448498)
    assert [schedule.continuation_rate(step, 30000, peak) for step in range(30000, 75000, 5000)] == pytest.approx(
        [0, peak / 2, peak, peak / 2, 0, peak / 2, peak, peak / 2, 0], abs=1e-14,
    )


@pytest.mark.parametrize("base_losses,extra_losses,stop_step,best_step", [
    ([3.0, 1.0, 2.0], [1.0, 1.1], 10, 4),
    ([3.0, 2.0, 1.0], [1.0, 1.1, 1.2], 12, 6),
    ([3.0, 1.0, 2.0], [0.9, 1.0, 1.0, 1.0], 14, 8),
])
def test_continuation_carries_patience_and_keeps_global_best(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, base_losses: list[float], extra_losses: list[float],
    stop_step: int, best_step: int,
) -> None:
    settings = short_schedule(monkeypatch)
    monkeypatch.setattr(train, "train_batch", lambda *args: torch.tensor(0.5))
    model = torch.nn.Linear(2, 3)
    loader = small_loader(settings)
    original = tmp_path / "original"
    validation_losses(monkeypatch, base_losses)
    train.fit(model, loader, loader, settings, torch.device("cpu"), original, "A")
    output = tmp_path / "continuation"
    shutil.copytree(original, output)
    validation_losses(monkeypatch, extra_losses)
    train.fit(model, small_loader(settings, continuous=True), loader, settings, torch.device("cpu"),
              output, "A", resume=True, extend=True)
    assert torch.load(output / "last.pt", weights_only=True)["step"] == stop_step
    assert torch.load(output / "best.pt", weights_only=True)["step"] == best_step
    state = train.read_json(output / "status.json.gz")
    assert (state["phase"], state["bad_checks"], state["stage"]) == ("complete", 3, "continuation")
    assert torch.load(original / "last.pt", weights_only=True)["step"] == settings.max_steps
    assert len(train.read_json(original / "history.json.gz")) == 3


def test_continuation_can_already_have_met_patience_at_original_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = short_schedule(monkeypatch)
    settings = train.replace(settings, validation_steps=[1, 2, 4, 6])
    monkeypatch.setattr(train, "train_batch", lambda *args: torch.tensor(0.5))
    model = torch.nn.Linear(2, 3)
    loader = small_loader(settings)
    validation_losses(monkeypatch, [1.0, 2.0, 2.0, 2.0])
    train.fit(model, loader, loader, settings, torch.device("cpu"), tmp_path, "A")
    monkeypatch.setattr(train, "train_batch", lambda *args: pytest.fail("Patience already exhausted"))
    train.fit(model, small_loader(settings, continuous=True), loader, settings, torch.device("cpu"),
              tmp_path, "A", resume=True, extend=True)
    assert train.read_json(tmp_path / "status.json.gz")["step"] == 6


@pytest.mark.parametrize("interrupt_call", [1, 3])
def test_continuation_resume_preserves_weights_momentum_rng_and_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupt_call: int,
) -> None:
    settings = short_schedule(monkeypatch)
    monkeypatch.setattr(train, "prepare_clips", lambda clips, training: clips + torch.rand_like(clips) * 0.01)
    monkeypatch.setattr(train.Progress, "due", lambda self: True)

    def fresh_model() -> torch.nn.Module:
        train.seed_everything(settings.seed)
        return torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                                   torch.nn.Linear(8, 3))

    model = fresh_model()
    loader = small_loader(settings)
    validation_losses(monkeypatch, [3.0, 2.0, 1.0])
    original = tmp_path / "original"
    train.fit(model, loader, loader, settings, torch.device("cpu"), original, "A")
    for name in ("reference", "recovered"):
        shutil.copytree(original, tmp_path / name)
    monkeypatch.setattr(train, "validate", lambda *args: {"loss": 1.0, "probabilities": np.ones((8, 3))})
    train.fit(fresh_model(), small_loader(settings, continuous=True), loader, settings, torch.device("cpu"),
              tmp_path / "reference", "A", resume=True, extend=True)

    original_batch = train.train_batch
    calls = 0

    def interrupted_batch(*args) -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls == interrupt_call:
            raise RuntimeError("simulated interruption")
        return original_batch(*args)

    monkeypatch.setattr(train, "train_batch", interrupted_batch)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train.fit(fresh_model(), small_loader(settings, continuous=True), loader, settings, torch.device("cpu"),
                  tmp_path / "recovered", "A", resume=True, extend=True)
    monkeypatch.setattr(train, "train_batch", original_batch)
    train.fit(fresh_model(), small_loader(settings, continuous=True), loader, settings, torch.device("cpu"),
              tmp_path / "recovered", "A", resume=True, extend=True)
    reference = torch.load(tmp_path / "reference/last.pt", weights_only=True)
    recovered = torch.load(tmp_path / "recovered/last.pt", weights_only=True)
    assert reference["step"] == recovered["step"] == 12
    for name, tensor in reference["model"].items():
        torch.testing.assert_close(recovered["model"][name], tensor, rtol=0, atol=0)
    for index, state in reference["optimiser"]["state"].items():
        torch.testing.assert_close(recovered["optimiser"]["state"][index]["momentum_buffer"],
                                   state["momentum_buffer"], rtol=0, atol=0)
    assert recovered["optimiser"]["param_groups"] == reference["optimiser"]["param_groups"]
    assert torch.equal(recovered["torch_rng"], reference["torch_rng"])
    assert torch.equal(recovered["loader_rng"], reference["loader_rng"])
