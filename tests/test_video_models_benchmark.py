"""Check contact labels, cached clips, event scoring and X3D checkpoint recovery."""
import lzma
from pathlib import Path

import numpy as np
import pytest
import torch

from annotator.types import CourtHalf
from experiments.annotator.video_models_benchmark.data import (
    IMAGE_SIZE,
    ClipDataset,
    context_intervals,
    target_labels,
    write_json,
)
from experiments.annotator.video_models_benchmark.evaluate import (
    decode_span,
    score_probabilities,
)
from experiments.annotator.video_models_benchmark.prepare import prepare_video


def span(start: int, end: int, frames: list[int], sides: list[str | None], span_id: int = 0) -> dict:
    return {"start_frame": start, "end_frame": end, "label_frames": frames, "label_sides": sides,
            "span_id": span_id, "rally_id": str(span_id)}


def test_positive_priority_and_ignored_band() -> None:
    rally = span(0, 14, [4, 7], ["Top", "Bot"])
    assert target_labels(rally, "A").tolist() == [0, 0, -100, 2, 2, 2, 1, 1, 1, -100, 0, 0, 0, 0]
    assert target_labels(rally, "B").tolist() == [0, 0, 2, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0]
    reversed_contacts = span(0, 14, [7, 4], ["Bot", "Top"])
    np.testing.assert_array_equal(target_labels(rally, "B"), target_labels(reversed_contacts, "B"))


def write_cache(root: Path, video: dict) -> None:
    directory = root / video["item_id"]
    directory.mkdir()
    intervals = context_intervals(video)
    frame_ids = np.concatenate([np.arange(start, end) for start, end in intervals])
    pixels = np.broadcast_to(frame_ids[:, None, None, None], (len(frame_ids), IMAGE_SIZE, IMAGE_SIZE, 3))
    pixels.astype(np.uint8).tofile(directory / "frames.rgb")
    write_json(directory / "frames.json.gz", {"image_size": IMAGE_SIZE, "intervals": intervals})
    write_json(root / "index.json.gz", {"videos": [video], "smoke_rallies": None})


def test_clips_read_beyond_rallies_and_repeat_only_video_edges(tmp_path: Path) -> None:
    video = {"item_id": "one", "role": "train", "frame_count": 70, "fps": 30,
             "spans": [span(0, 15, [3], ["Bot"]), span(40, 70, [60], ["Top"], 1)]}
    write_cache(tmp_path, video)
    dataset = ClipDataset(tmp_path, "train", "A", training=False)
    first, _ = dataset[0]
    np.testing.assert_array_equal(first[0, :, 0, 0], np.clip(np.arange(-8, 8), 0, 69))
    second_rally, _ = dataset[15]
    np.testing.assert_array_equal(second_rally[0, :, 0, 0], np.arange(32, 48))
    last, _ = dataset[len(dataset) - 1]
    np.testing.assert_array_equal(last[0, :, 0, 0], np.clip(np.arange(61, 77), 0, 69))
    training = ClipDataset(tmp_path, "train", "A", training=True)
    assert np.all(training.labels != -100)
    assert len(training) == len(dataset) - 4


def test_decoder_background_suppression_ties_and_alternation() -> None:
    rally = span(100, 140, [101], ["Bot"])
    probabilities = np.tile([0.9, 0.05, 0.05], (40, 1))
    probabilities[1] = [0.2, 0.1, 0.7]
    probabilities[2] = [0.1, 0.8, 0.1]
    probabilities[8] = [0.1, 0.8, 0.1]  # Equal score, exactly six away: earlier wins.
    probabilities[15] = [0.1, 0.8, 0.1]
    events = decode_span(rally, probabilities, 30).events
    assert [event.frame for event in events] == [102, 115]
    assert [event.side for event in events] == [CourtHalf.BOT, CourtHalf.BOT]  # Alternation tie keeps raw guesses.
    probabilities[30] = [0.1, 0.8, 0.1]
    events = decode_span(rally, probabilities, 30).events
    assert [event.side for event in events] == [CourtHalf.BOT, CourtHalf.TOP, CourtHalf.BOT]
    probabilities[8] = [0.05, 0.9, 0.05]
    assert decode_span(rally, probabilities, 25).events[0].frame == 102  # Six exceeds scaled distance five.


def test_supplied_rallies_reuse_existing_scorer() -> None:
    rally = span(0, 50, [10, 30], ["Bot", "Top"])
    video = {"item_id": "one", "role": "validation", "fps": 30, "spans": [rally]}
    class FakeDataset:
        def __init__(self) -> None:
            self.videos = [video]

        def __len__(self) -> int:
            return 50

    probabilities = np.tile([1.0, 0, 0], (50, 1))
    probabilities[10] = [0, 1, 0]
    probabilities[30] = [0, 0, 1]
    result = score_probabilities(FakeDataset(), probabilities)
    for counts in result["metrics"].values():
        assert counts["fully_correct"] == 1
        assert counts["contact_f1"] == 100
        assert counts["correct_player_f1"] == 100
    probabilities[20] = [0, 1, 0]
    counts = score_probabilities(FakeDataset(), probabilities)["metrics"]["10"]
    assert counts["fully_correct"] == 0
    assert counts["contact_f1"] == 80


@pytest.mark.parametrize("break_ties,expected_correct_player,expected_rallies", [(False, 2, 1), (True, 1, 0)])
def test_scoring_preserves_the_selected_tie_rule(
    break_ties: bool, expected_correct_player: int, expected_rallies: int,
) -> None:
    rally = span(0, 50, [10, 30], ["Top", "Top"])

    class FakeDataset:
        def __init__(self) -> None:
            self.videos = [{"item_id": "one", "role": "validation", "fps": 30, "spans": [rally]}]

        def __len__(self) -> int:
            return 50

    probabilities = np.tile([1.0, 0.0, 0.0], (50, 1))
    probabilities[10] = [0.0, 0.4, 0.6]
    probabilities[30] = [0.0, 0.1, 0.9]
    result = score_probabilities(FakeDataset(), probabilities, break_alternation_ties=break_ties)
    for counts in result["metrics"].values():
        assert counts["matched"] == 2
        assert counts["correct_player"] == expected_correct_player
        assert counts["fully_correct"] == expected_rallies
    events = result["streams"]["one"]["events"]
    assert [event[2] for event in events] == (["Bot", "Top"] if break_ties else ["Top", "Top"])


def test_ffmpeg_cache_selects_correct_source_frames_and_reuses(tmp_path: Path) -> None:
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is unavailable")
    frames = np.zeros((60, 24, 32, 3), dtype=np.uint8)
    frames[:, :, :, 0] = np.arange(60)[:, None, None] * 3
    source = tmp_path / "source.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "32x24",
                    "-r", "30", "-i", "pipe:0", "-c:v", "ffv1", str(source)], input=frames.tobytes(), check=True)
    video = {"item_id": "one", "source_video": str(source), "frame_count": 60,
             "spans": [span(10, 15, [12], ["Bot"]), span(45, 50, [47], ["Top"], 1)]}
    cache = tmp_path / "cache"
    result = prepare_video(video, tmp_path, cache)
    assert not result["reused"]
    pixels = np.memmap(cache / "one/frames.rgb", mode="r", dtype=np.uint8,
                       shape=(result["frames"], IMAGE_SIZE, IMAGE_SIZE, 3))
    expected = np.concatenate([np.arange(start, end) for start, end in context_intervals(video)]) * 3
    np.testing.assert_allclose(pixels[:, IMAGE_SIZE // 2, IMAGE_SIZE // 2, 0], expected, atol=2)
    assert np.all(pixels[:, 0] == 0)
    assert prepare_video(video, tmp_path, cache)["reused"]


def test_contact_batches_balance_cover_and_resume() -> None:
    from experiments.annotator.video_models_benchmark.sampling import (
        ContactBatchSampler,
    )

    labels = np.array([0] * 12 + [1, 1, 2, 2])
    sampler = ContactBatchSampler(labels, batch_size=4, contact_fraction=0.5, steps=8, seed=9)
    batches = np.asarray(list(sampler))
    assert np.all((labels[batches] > 0).sum(axis=1) == 2)
    contacts = batches[labels[batches] > 0]
    backgrounds = batches[labels[batches] == 0]
    for start in range(0, 16, 4):
        assert sorted(contacts[start:start + 4]) == [12, 13, 14, 15]
    assert sorted(backgrounds[:12]) == list(range(12))
    assert not np.array_equal(contacts[:4], contacts[4:8])
    recovered = ContactBatchSampler(labels, 4, 0.5, 8, 9)
    recovered.start_step = 3
    np.testing.assert_array_equal(list(recovered), batches[3:])
    assert len(recovered) == 5
    with pytest.raises(ValueError, match="whole"):
        ContactBatchSampler(labels, 4, 0.3, 8, 9)


def tiny_train_loader(settings):
    from torch.utils.data import DataLoader, TensorDataset

    from experiments.annotator.video_models_benchmark.sampling import (
        ContactBatchSampler,
    )

    labels = np.array([0, 0, 0, 0, 1, 2, 1, 2])
    dataset = TensorDataset(torch.arange(16, dtype=torch.float32).reshape(8, 2) / 16, torch.from_numpy(labels))
    sampler = ContactBatchSampler(labels, settings.batch_size, settings.contact_fraction,
                                  settings.max_steps, settings.seed)
    return DataLoader(dataset, batch_sampler=sampler, generator=torch.Generator().manual_seed(11))


def test_fit_saves_strict_best_at_update_boundaries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments.annotator.video_models_benchmark import train

    model = torch.nn.Linear(2, 3)
    settings = train.Settings(batch_size=4, max_steps=6, warmup_steps=2, validation_steps=[2, 4, 6],
                              workers=0, precision="float32")
    loader = tiny_train_loader(settings)
    rates = []
    losses = iter([1.0, 0.99995, 1.1])

    def batch_step(model, batch, optimiser, device, precision) -> torch.Tensor:
        rates.append(optimiser.param_groups[0]["lr"])
        optimiser.zero_grad()
        loss = model(batch[0]).square().mean()
        loss.backward()
        optimiser.step()
        return loss.detach()

    monkeypatch.setattr(train, "train_batch", batch_step)
    monkeypatch.setattr(train, "validate", lambda *args: {"loss": next(losses), "probabilities": np.ones((8, 3))})
    monkeypatch.setattr(train, "score_probabilities", lambda *args: {"metrics": {}})
    output = tmp_path / "run"
    train.fit(model, loader, loader, settings, torch.device("cpu"), output, "A")
    best = torch.load(output / "best.pt", weights_only=True)
    assert best["step"] == 4
    with lzma.open(output / "validation_probabilities.npy.xz", "rb") as stream:
        np.testing.assert_array_equal(np.load(stream), np.ones((8, 3)))
    assert rates[:2] == pytest.approx([0.002, 0.02])
    assert rates[-1] == 0.0
    assert [row["steps"] for row in train.read_json(output / "history.json.gz")] == [2, 4, 6]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, best["model"][name])


def test_resume_replays_batches_augmentation_momentum_and_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.annotator.video_models_benchmark import train

    settings = train.Settings(batch_size=4, max_steps=8, warmup_steps=2, validation_steps=[2, 4, 6, 8],
                              workers=0, precision="float32", seed=7)
    monkeypatch.setattr(train, "prepare_clips", lambda clips, training: clips + torch.rand_like(clips) * 0.01)

    def validation(model, *args):
        model.eval()
        with torch.inference_mode():
            logits = model(torch.ones(8, 2))
            return {"loss": float(logits.square().mean()), "probabilities": logits.softmax(1).numpy()}

    monkeypatch.setattr(train, "validate", validation)
    monkeypatch.setattr(train, "score_probabilities", lambda *args: {"metrics": {}})

    def fresh_model_and_loader():
        train.seed_everything(settings.seed)
        model = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                                    torch.nn.Linear(8, 3))
        return model, tiny_train_loader(settings)

    model, loader = fresh_model_and_loader()
    train.fit(model, loader, loader, settings, torch.device("cpu"), tmp_path / "reference", "A")
    reference = torch.load(tmp_path / "reference/last.pt", weights_only=True)
    monkeypatch.setattr(train.Progress, "due", lambda self: True)
    original_batch = train.train_batch
    calls = 0

    def interrupt_after_checkpoint(model, batch, optimiser, device, precision) -> torch.Tensor:
        nonlocal calls
        calls += 1
        # The fifth update follows the step-4 validation commit.
        if calls == 5:
            raise RuntimeError("simulated interruption")
        return original_batch(model, batch, optimiser, device, precision)

    monkeypatch.setattr(train, "train_batch", interrupt_after_checkpoint)
    model, loader = fresh_model_and_loader()
    output = tmp_path / "recover"
    with pytest.raises(RuntimeError, match="simulated"):
        train.fit(model, loader, loader, settings, torch.device("cpu"), output, "A")
    saved = torch.load(output / "last.pt", weights_only=True)
    assert saved["step"] == 4
    assert saved["optimiser"]["state"]
    monkeypatch.setattr(train, "train_batch", original_batch)
    model, loader = fresh_model_and_loader()
    train.fit(model, loader, loader, settings, torch.device("cpu"), output, "A", resume=True)
    recovered = torch.load(output / "last.pt", weights_only=True)
    assert recovered["step"] == 8
    for name, value in reference["model"].items():
        torch.testing.assert_close(recovered["model"][name], value, rtol=0, atol=0)
    for index, state in reference["optimiser"]["state"].items():
        torch.testing.assert_close(recovered["optimiser"]["state"][index]["momentum_buffer"],
                                   state["momentum_buffer"], rtol=0, atol=0)
    assert recovered["optimiser"]["param_groups"] == reference["optimiser"]["param_groups"]
    assert torch.equal(recovered["torch_rng"], reference["torch_rng"])
    assert torch.equal(recovered["loader_rng"], reference["loader_rng"])
    assert train.read_json(output / "status.json.gz")["phase"] == "complete"
    assert train.read_json(output / "training_progress.json.gz")[-1]["step"] == 8
