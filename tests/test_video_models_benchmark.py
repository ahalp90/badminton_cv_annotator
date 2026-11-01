"""Check contact alignment, clip transformations and the existing event-score boundary."""
from pathlib import Path

import numpy as np
import pytest
import torch

from annotator.outcomes.point_winner import Half
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
from experiments.annotator.video_models_benchmark.model import prepare_clips
from experiments.annotator.video_models_benchmark.prepare import prepare_video


def span(start: int, end: int, frames: list[int], sides: list[str], span_id: int = 0) -> dict:
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


def test_augmentation_is_temporally_consistent_and_keeps_black_padding() -> None:
    torch.manual_seed(17)
    image = torch.randint(30, 200, (4, 3, 1, 32, 32), dtype=torch.uint8)
    image[:, :, :, :8] = 0
    image[:, :, :, -8:] = 0
    clips = image.expand(-1, -1, 16, -1, -1).clone()
    augmented = prepare_clips(clips, training=True)
    torch.testing.assert_close(augmented[:, :, :1].expand_as(augmented), augmented)
    torch.testing.assert_close(augmented[:, :, :, 0], torch.full_like(augmented[:, :, :, 0], -2.0))
    assert not torch.equal(augmented[0], augmented[1])
    torch.testing.assert_close(prepare_clips(clips, training=False), (clips.float() / 255 - 0.45) / 0.225)


def test_decoder_background_suppression_ties_and_alternation() -> None:
    rally = span(100, 140, [101], ["Bot"])
    probabilities = np.tile([0.9, 0.05, 0.05], (40, 1))
    probabilities[1] = [0.2, 0.1, 0.7]
    probabilities[2] = [0.1, 0.8, 0.1]
    probabilities[8] = [0.1, 0.8, 0.1]  # Equal score, exactly six away: earlier wins.
    probabilities[15] = [0.1, 0.8, 0.1]
    events = decode_span(rally, probabilities, 30).events
    assert [event.frame for event in events] == [102, 115]
    assert [event.side for event in events] == [Half.BOT, Half.BOT]  # Alternation tie keeps raw guesses.
    probabilities[30] = [0.1, 0.8, 0.1]
    events = decode_span(rally, probabilities, 30).events
    assert [event.side for event in events] == [Half.BOT, Half.TOP, Half.BOT]
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
        assert counts["contact_f1"] == counts["correct_player_f1"] == 100
    probabilities[20] = [0, 1, 0]
    counts = score_probabilities(FakeDataset(), probabilities)["metrics"]["10"]
    assert counts["fully_correct"] == 0
    assert counts["contact_f1"] == 80


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


def test_validation_ignores_loss_band_but_predicts_every_centre() -> None:
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset

    from experiments.annotator.video_models_benchmark.train import validate

    class FixedLogits(torch.nn.Module):
        def forward(self, clips: torch.Tensor) -> torch.Tensor:
            return clips.new_tensor([2.0, 1.0, 0.0]).expand(len(clips), -1)

    targets = torch.tensor([0, -100, 2])
    clips = torch.zeros(3, 3, 16, 4, 4, dtype=torch.uint8)
    result = validate(FixedLogits(), DataLoader(TensorDataset(clips, targets), batch_size=2),
                      torch.device("cpu"), "float32")
    expected = F.cross_entropy(torch.tensor([[2.0, 1.0, 0.0], [2.0, 1.0, 0.0]]), torch.tensor([0, 2]))
    assert result["loss"] == pytest.approx(float(expected))
    assert result["probabilities"].shape == (3, 3)
    assert np.asarray(result["confusion"]).sum() == 2


def test_fit_saves_strict_best_and_uses_plateau_patience(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from torch.utils.data import DataLoader, TensorDataset

    from experiments.annotator.video_models_benchmark import train

    model = torch.nn.Linear(1, 3)
    dataset = TensorDataset(torch.ones(4, 1), torch.zeros(4, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=2)
    rates = []
    losses = iter([1.0, 0.99995, 1.1, 1.1, 1.1, 1.1])

    def batch_step(model, batch, optimiser, device, precision) -> torch.Tensor:
        rates.append(optimiser.param_groups[0]["lr"])
        optimiser.zero_grad()
        loss = model(batch[0]).square().mean()
        loss.backward()
        optimiser.step()
        return loss.detach()

    monkeypatch.setattr(train, "train_batch", batch_step)
    monkeypatch.setattr(train, "validate", lambda *args: {"loss": next(losses), "probabilities": np.ones((4, 3))})
    monkeypatch.setattr(train, "score_probabilities", lambda *args: {"metrics": {}})
    settings = train.Settings(max_epochs=10, early_stop_patience=5, workers=0, precision="float32")
    output = tmp_path / "run"
    train.fit(model, loader, loader, settings, torch.device("cpu"), 2, output, "A")
    best = torch.load(output / "best.pt", weights_only=True)
    assert best["epoch"] == 2  # Strict CE improvement still saves below the stopping threshold.
    assert rates[:2] == pytest.approx([0.002, 0.02])
    assert rates[8:10] == pytest.approx([0.02, 0.02])
    assert rates[10:12] == pytest.approx([0.002, 0.002])  # Fourth bad check reduces the LR.
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, best["model"][name])
