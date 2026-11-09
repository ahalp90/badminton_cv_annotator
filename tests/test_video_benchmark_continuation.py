"""Check that continuation extends the sampled stream, carries patience and keeps the global best."""
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


def test_continuous_sampler_extends_original_stream_and_replays() -> None:
    labels = np.array([0] * 17 + [1, 2] * 5)
    original = ContactBatchSampler(labels, 8, 0.25, 13, 1)
    longer = ContactBatchSampler(labels, 8, 0.25, 31, 1)
    continuous = ContactBatchSampler(labels, 8, 0.25, 13, 1, continuous=True)
    batches = list(islice(continuous, 31))
    assert batches[:13] == list(original)
    assert batches == list(longer)
    continuous.start_step = 19
    assert list(islice(continuous, 12)) == batches[19:]


def test_continuation_carries_patience_and_keeps_global_best(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = short_schedule(monkeypatch)
    monkeypatch.setattr(train, "train_batch", lambda *args: torch.tensor(0.5))
    model = torch.nn.Linear(2, 3)
    loader = small_loader(settings)
    original = tmp_path / "original"
    # One bad check carries over; an equal loss is not an improvement, so two more checks stop at step 10.
    validation_losses(monkeypatch, [3.0, 1.0, 2.0])
    train.fit(model, loader, loader, settings, torch.device("cpu"), original, "A")
    output = tmp_path / "continuation"
    shutil.copytree(original, output)
    validation_losses(monkeypatch, [1.0, 1.1])
    train.fit(model, small_loader(settings, continuous=True), loader, settings, torch.device("cpu"),
              output, "A", resume=True, extend=True)
    assert torch.load(output / "last.pt", weights_only=True)["step"] == 10
    assert torch.load(output / "best.pt", weights_only=True)["step"] == 4
    state = train.read_json(output / "status.json.gz")
    assert (state["phase"], state["bad_checks"], state["stage"]) == ("complete", 3, "continuation")
    assert torch.load(original / "last.pt", weights_only=True)["step"] == settings.max_steps
    assert len(train.read_json(original / "history.json.gz")) == 3


def test_integrated_continuation_keeps_each_check_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Pause after the first continuation check, past the initial six-update budget.
    pause_step = 8
    settings = train.replace(short_schedule(monkeypatch), continue_until_patience=True)
    model = torch.nn.Linear(2, 3)
    loader = small_loader(settings, continuous=True)
    losses = iter([3.0, 2.0, 1.0, 0.9, 1.1, 1.2, 1.3])
    rates = []
    checks = 0

    def update(*args) -> torch.Tensor:
        rates.append(args[2].param_groups[0]["lr"])
        return torch.tensor(0.5)

    def validation(*args) -> dict:
        nonlocal checks
        checks += 1
        if checks * 2 == pause_step:
            (tmp_path / "PAUSE").touch()
        return {"loss": next(losses), "probabilities": np.full((8, 3), checks / 10)}

    monkeypatch.setattr(train, "train_batch", update)
    monkeypatch.setattr(train, "validate", validation)
    train.fit(model, loader, small_loader(settings), settings, torch.device("cpu"), tmp_path, "A")
    assert train.read_json(tmp_path / "status.json.gz")["phase"] == "paused"
    assert torch.load(tmp_path / "last.pt", weights_only=True)["step"] == pause_step
    (tmp_path / "PAUSE").unlink()
    train.fit(model, loader, small_loader(settings), settings, torch.device("cpu"), tmp_path, "A", resume=True)
    assert train.read_json(tmp_path / "status.json.gz")["step"] == 14
    assert torch.load(tmp_path / "best.pt", weights_only=True)["step"] == 8
    assert rates == pytest.approx([train.learning_rate_at_step(settings, step) for step in range(14)])
    assert rates[5:7] == [0.0, 0.0]
    for index, step in enumerate(range(2, 15, 2), start=1):
        directory = tmp_path / "validation_checks" / f"step{step:06d}"
        assert torch.load(directory / "checkpoint.pt", weights_only=True)["step"] == step
        with train.lzma.open(directory / "probabilities.npy.xz", "rb") as stream:
            assert np.load(stream).mean() == pytest.approx(index / 10)
        assert train.read_json(directory / "frame_metrics.json.gz")["loss"] > 0
