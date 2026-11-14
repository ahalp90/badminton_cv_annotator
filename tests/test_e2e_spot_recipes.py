"""Recipe geometry, independent MixUp and recoverable checkpoint selection."""
from __future__ import annotations

import subprocess
from dataclasses import asdict, replace
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
import torch
from PIL import Image, JpegImagePlugin
from torch import nn
from torchvision import transforms

from experiments.annotator.e2e_spot_benchmark import engine
from experiments.annotator.e2e_spot_benchmark.config import Settings, load_settings
from experiments.annotator.e2e_spot_benchmark.data import (
    DenseClips,
    FrameReader,
    dense_targets,
)
from experiments.annotator.e2e_spot_benchmark.model import transform_clips
from experiments.annotator.e2e_spot_benchmark.prepare import prepare_video
from experiments.annotator.video_models_benchmark.data import read_json, write_json
from experiments.annotator.video_models_benchmark.model import prepare_clips


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def recipe_cache(root: Path, *, user: bool = False) -> Path:
    root.mkdir()
    videos = [{"item_id": role, "role": role, "fps": 30, "frame_count": 12,
               "spans": [{"span_id": "rally", "rally_id": "rally", "start_frame": 2, "end_frame": 10,
                          "label_frames": [4, 7], "label_sides": ["Bot", "Top"]}]}
              for role in ("train", "validation")]
    write_json(root / "index.json.gz", {"videos": videos})
    height = 256 if user else 224
    for video in videos:
        directory = root / video["item_id"]
        directory.mkdir()
        with ZipFile(directory / "frames.zip", "w") as archive:
            for frame in range(12):
                pixels = np.full((height, 256, 3), 30 + 10 * frame, dtype=np.uint8)
                if user:
                    pixels[:40] = 0
                    pixels[-40:] = 0
                stream = BytesIO()
                Image.fromarray(pixels).save(stream, format="JPEG", quality=95, subsampling=2)
                archive.writestr(f"{frame:09d}.jpg", stream.getvalue())
        write_json(directory / "frames.json.gz", {
            "storage": "jpeg_zip", "geometry": "letterbox256" if user else "height224-aspect",
            "image_height": height, "image_width": 256, "jpeg_quality": 95, "jpeg_subsampling": 2,
            "intervals": [[0, 12]], "frames": 12,
        })
    return root


@pytest.mark.parametrize("training", [False, True])
def test_user_recipe_matches_shared_transform_and_preserves_temporal_padding(training: bool) -> None:
    raw = torch.randint(30, 200, (2, 3, 2, 256, 256), dtype=torch.uint8)
    raw[..., :40, :] = 0
    raw[..., -40:, :] = 0
    padded = torch.full((2, 3, 4, 256, 256), 255, dtype=torch.uint8)
    padded[:, :, 1:3] = raw
    valid = torch.tensor([[False, True, True, False]]).expand(2, -1)
    torch.manual_seed(17)
    reference = prepare_clips(raw, training=training, model_name="videomae")
    torch.manual_seed(17)
    actual = transform_clips(padded, valid, training=training, augmentation="user")
    torch.testing.assert_close(actual[:, 1:3], reference, rtol=0, atol=0)
    assert torch.count_nonzero(actual[~valid]) == 0
    assert torch.count_nonzero(actual[:, 1:3, :, :10]) > 0


def test_e2e_transform_matches_original_order_and_is_clip_consistent() -> None:
    reference = nn.Sequential(
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(hue=.2)]), p=.25),
        transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(saturation=(.7, 1.2))]), p=.25),
        transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(brightness=(.7, 1.2))]), p=.25),
        transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(contrast=(.7, 1.2))]), p=.25),
        transforms.RandomApply(nn.ModuleList([transforms.GaussianBlur(5)]), p=.25),
        transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    )
    image = torch.randint(0, 256, (1, 3, 1, 224, 224), dtype=torch.uint8)
    clips = image.expand(2, -1, 4, -1, -1).clone()
    valid = torch.tensor([[False, True, True, False], [True, True, False, False]])
    torch.manual_seed(4)
    expected = [reference(clips[index].permute(1, 0, 2, 3)[mask].float() / 255)
                for index, mask in enumerate(valid)]
    torch.manual_seed(4)
    actual = transform_clips(clips, valid, training=True, augmentation="e2e")
    for index, mask in enumerate(valid):
        torch.testing.assert_close(actual[index, mask], expected[index], rtol=0, atol=0)
        torch.testing.assert_close(actual[index, mask][0], actual[index, mask][1], rtol=0, atol=0)
    assert torch.count_nonzero(actual[~valid]) == 0


def test_e2e_partners_are_independent_and_repeatable_in_singleton_batches(tmp_path: Path) -> None:
    dataset = DenseClips(recipe_cache(tmp_path / "cache"), "train", "supplied", 4, training=True,
                         seed=12, frames_per_cycle=43, cycles=2, augmentation="e2e", mixup=True)
    samples = [dataset[index] for index in range(11)]
    assert any(not torch.equal(sample[3], sample[7]) for sample in samples)
    assert len({sample[8] for sample in samples}) == 11
    assert sum(sample[4] for sample in samples) == 43
    assert samples[-1][1][-1] == samples[-1][6][-1] == -100
    for index, sample in enumerate(samples):
        replay = dataset[index]
        for field in (0, 1, 3, 5, 6, 7):
            torch.testing.assert_close(sample[field], replay[field], rtol=0, atol=0)
        assert sample[8] == replay[8]
    loader = engine.make_loader(dataset, 1, 0)
    clips, labels, valid, positions, paired = engine.device_batch(next(iter(loader)), torch.device("cpu"))
    assert clips.shape[0] == labels.shape[0] == valid.shape[0] == 1
    assert positions.item() == 4
    assert paired is not None and paired[3].shape == (1,)


def test_per_clip_mixup_loss_normalises_both_ignore_masks() -> None:
    logits = torch.tensor([[[1., 2., 0.], [3., 1., 0.]], [[0., 1., 2.], [2., 0., 1.]]], requires_grad=True)
    first = torch.tensor([[1, -100], [2, 0]])
    second = torch.tensor([[-100, 2], [0, -100]])
    mixing = torch.tensor([.2, .8])
    weights = torch.tensor([1., 5., 5.])
    numerator, denominator = 0., 0.
    for batch in range(2):
        for labels, fraction in ((first, mixing[batch]), (second, 1 - mixing[batch])):
            for frame in range(2):
                label = labels[batch, frame]
                if label != -100:
                    weight = fraction * weights[label]
                    numerator += weight * -logits[batch, frame].log_softmax(0)[label]
                    denominator += weight
    actual = engine.masked_loss(logits, first, weights, paired_targets=second, mixing=mixing)
    torch.testing.assert_close(actual, numerator / denominator)
    actual.backward()
    assert torch.isfinite(logits.grad).all()


class TinyModel(nn.Module):
    def __init__(self, _backbone: str, _length: int, classes: int, *, pretrained: bool) -> None:
        super().__init__()
        self.head = nn.Linear(3, classes)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.head(pixels.mean((-1, -2)))


def tiny_transform(clips: torch.Tensor, valid: torch.Tensor, *, training: bool,
                   augmentation: str = "hybrid") -> torch.Tensor:
    pixels = clips.float().mean((-1, -2), keepdim=True).permute(0, 2, 1, 3, 4) / 255
    return pixels.masked_fill(~valid[:, :, None, None, None], 0)


def test_legacy_settings_resume_and_new_e2e_midcycle_replay(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(engine, "SpotModel", TinyModel)
    monkeypatch.setattr(engine, "transform_clips", tiny_transform)

    def validate(*_args):
        return {"loss": 1., "seconds": 0., "metrics": {}, "streams": {}}, [np.full((12, 3), 1 / 3)]

    monkeypatch.setattr(engine, "validate", validate)
    cache = recipe_cache(tmp_path / "cache")
    settings = Settings(cycles=5, warmup_cycles=0, clip_length=4, frames_per_cycle=12,
                        batch_size=2, augmentation="e2e", mixup=True, early_stopping_patience=2,
                        ignore_contact_halo=False)
    complete = engine.fit(cache, tmp_path / "whole", settings)
    engine.fit(cache, tmp_path / "resumed", settings, max_steps=2)
    resumed = engine.fit(cache, tmp_path / "resumed", settings, resume=True)
    assert complete["best_cycle"] == resumed["best_cycle"] == 1
    assert complete["completed_cycles"] == resumed["completed_cycles"] == 3
    states = [torch.load(tmp_path / name / "recovery.pt", weights_only=False) for name in ("whole", "resumed")]
    for name, weights in states[0]["model"].items():
        torch.testing.assert_close(weights, states[1]["model"][name], rtol=0, atol=0)
    legacy = replace(settings, augmentation="hybrid", mixup=False, early_stopping_patience=0,
                     ignore_contact_halo=True)
    engine.fit(cache, tmp_path / "legacy", legacy, max_steps=1)
    saved = torch.load(tmp_path / "legacy/recovery.pt", weights_only=False)
    record = asdict(legacy)
    for name in ("augmentation", "early_stopping_patience", "cudnn_benchmark", "ignore_contact_halo",
                 "checkpoint_metric", "selection_tolerance"):
        del record[name]
    saved["settings"] = record
    del saved["best_metric"]
    torch.save(saved, tmp_path / "legacy/recovery.pt")
    write_json(tmp_path / "legacy/settings.json.gz", record)
    assert engine.fit(cache, tmp_path / "legacy", legacy, resume=True, max_steps=1)["steps"] == 2


def test_raw_user_cache_offsets_and_recipe_geometry(tmp_path: Path) -> None:
    video = {"item_id": "raw"}
    directory = tmp_path / "raw"
    directory.mkdir()
    pixels = np.broadcast_to(np.array([2, 3, 8, 9], dtype=np.uint8)[:, None, None, None], (4, 256, 256, 3))
    pixels.tofile(directory / "frames.rgb")
    write_json(directory / "frames.json.gz", {"image_size": 256, "padding": "black",
                                              "intervals": [[2, 4], [8, 10]], "frames": 4})
    reader = FrameReader(tmp_path, video, augmentation="user")
    assert reader.read(8, 10)[:, 0, 0, 0].tolist() == [8, 9]
    with pytest.raises(ValueError, match="gap"):
        reader.read(3, 9)
    with pytest.raises(ValueError, match="user recipe"):
        FrameReader(tmp_path, video)
    cache = recipe_cache(tmp_path / "jpeg", user=True)
    with pytest.raises(ValueError, match="height224-aspect"):
        DenseClips(cache, "train", "supplied", 4)
    dataset = DenseClips(cache, "train", "supplied", 4, augmentation="user")
    assert dataset[0][0].shape == (3, 4, 256, 256)


def test_user_preparation_is_source_letterbox_jpeg420(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=white:s=320x180:r=2",
                    "-frames:v", "4", "-threads", "1", str(source)], check=True)
    video = {"item_id": "clip", "source_video": source.name, "frame_count": 4, "spans": []}
    cache = tmp_path / "cache"
    first = prepare_video(video, tmp_path, cache, "discovery", "user")
    assert first["frames"] == 4 and not first["reused"]
    assert prepare_video(video, tmp_path, cache, "discovery", "user")["reused"]
    metadata = read_json(cache / "clip/frames.json.gz")
    assert metadata["geometry"] == "letterbox256"
    with ZipFile(cache / "clip/frames.zip") as archive, Image.open(BytesIO(archive.read("000000000.jpg"))) as image:
        assert image.size == (256, 256) and JpegImagePlugin.get_sampling(image) == 2
        pixels = np.asarray(image)
        assert pixels[0].max() == 0 and pixels[128].min() > 250
    with pytest.raises(ValueError, match="metadata differs"):
        prepare_video(video, tmp_path, cache, "discovery", "e2e")


def test_fit_check_runs_one_pass_and_configures_autotuning(monkeypatch, tmp_path: Path) -> None:
    from experiments.annotator.e2e_spot_benchmark import __main__ as cli

    monkeypatch.setattr(cli, "SpotModel", TinyModel)
    monkeypatch.setattr(engine, "transform_clips", tiny_transform)
    monkeypatch.setattr(cli, "transform_clips", tiny_transform)
    original = cli.optimise_batch
    updates = []

    def count_update(*args, **kwargs):
        updates.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(cli, "optimise_batch", count_update)
    settings = Settings(augmentation="e2e", mixup=True, clip_length=4, batch_size=2, validation_batch_size=2)
    result = cli.measure(recipe_cache(tmp_path / "cache"), settings, device="cpu", workers=0,
                         steps=2, warmup=1, compare_device_only=False, validation_steps=1)
    assert result["validation_fit"] == {"steps": 1, "peak_allocated_bytes": None}
    assert len(updates) == 3 and "device_only" not in result
    assert result["end_to_end"]["frames"] == 16
    benchmark_before = torch.backends.cudnn.benchmark
    try:
        engine.configure_runtime(replace(settings, cudnn_benchmark=True), torch.device("cuda"))
        assert torch.backends.cudnn.benchmark
        engine.configure_runtime(settings, torch.device("cuda"))
        assert not torch.backends.cudnn.benchmark
    finally:
        torch.backends.cudnn.benchmark = benchmark_before


def test_e2e_mixing_uses_each_clips_weight_and_keeps_partner_at_padding(monkeypatch) -> None:
    monkeypatch.setattr(engine, "transform_clips", tiny_transform)
    model = TinyModel("200mf", 2, 3, pretrained=False)
    captured = []
    model.register_forward_pre_hook(lambda _model, inputs: captured.append(inputs[0].detach().clone()))
    first = torch.full((2, 3, 2, 224, 224), 80, dtype=torch.uint8)
    second = torch.full_like(first, 160)
    valid = torch.tensor([[False, True], [True, False]])
    partner_valid = ~valid
    labels = torch.tensor([[-100, 1], [2, -100]])
    partner_labels = torch.tensor([[2, -100], [-100, 1]])
    mixing = torch.tensor([.2, .8])
    optimiser = torch.optim.AdamW(model.parameters())
    engine.optimise_batch(model, optimiser, first, labels, valid,
                          Settings(augmentation="e2e", mixup=True), torch.device("cpu"), torch.tensor([1., 5., 5.]),
                          paired=(second, partner_labels, partner_valid, mixing))
    first_pixels = tiny_transform(first, valid, training=True)
    second_pixels = tiny_transform(second, partner_valid, training=True)
    fractions = mixing[:, None, None, None, None]
    torch.testing.assert_close(captured[0], fractions * first_pixels + (1 - fractions) * second_pixels)
    assert torch.count_nonzero(captured[0][~valid]) > 0


@pytest.mark.parametrize("mode", ["supplied", "discovery"])
def test_contact_halo_policy_preserves_unknowns_positives_and_overlap(mode: str) -> None:
    video = {"frame_count": 22, "spans": [
        {"start_frame": 2, "end_frame": 20, "label_frames": [5, 10, 14],
         "label_sides": ["Bot", None, "Top"]},
        {"start_frame": 8, "end_frame": 18, "label_frames": [12], "label_sides": ["Top"]},
        {"start_frame": 11, "end_frame": 19, "label_frames": [12], "label_sides": ["Bot"]},
    ]}
    legacy = dense_targets(video, mode)
    new = dense_targets(video, mode, ignore_contact_halo=False)
    assert legacy[[3, 7, 16]].tolist() == [-100, -100, -100]
    assert new[[3, 7, 16]].tolist() == [0, 0, 0]
    assert new[4:7].tolist() == [1, 1, 1]
    assert new[8:11].tolist() == [-100, -100, -100]
    assert new[11:16].tolist() == [2, 2, 2, 2, 2]
    assert new[[0, 1, 20, 21]].tolist() == ([3] * 4 if mode == "discovery" else [-100] * 4)
    reverse = {**video, "spans": list(reversed(video["spans"]))}
    np.testing.assert_array_equal(new, dense_targets(reverse, mode, ignore_contact_halo=False))
    np.testing.assert_array_equal(legacy, dense_targets(reverse, mode))


def test_new_grid_disables_halo_and_legacy_configs_keep_it() -> None:
    root = Path(__file__).resolve().parents[1] / "experiments/annotator/e2e_spot_benchmark/configs"
    assert Settings().ignore_contact_halo
    assert load_settings(root / "pilot.toml").ignore_contact_halo
    for name in ("supplied200-user.toml", "supplied800.toml", "discovery200-user.toml", "discovery800.toml"):
        settings = load_settings(root / name)
        assert not settings.ignore_contact_halo
        assert settings.checkpoint_metric == "rally_f1" and settings.selection_tolerance == 1


def test_halo_policy_reaches_fit_validation_inference_and_smoke(monkeypatch, tmp_path: Path) -> None:
    from experiments.annotator.e2e_spot_benchmark import __main__ as cli

    monkeypatch.setattr(engine, "SpotModel", TinyModel)
    monkeypatch.setattr(cli, "SpotModel", TinyModel)
    monkeypatch.setattr(engine, "transform_clips", tiny_transform)
    cache = recipe_cache(tmp_path / "cache")
    settings = Settings(cycles=1, warmup_cycles=0, clip_length=4, frames_per_cycle=12,
                        batch_size=2, ignore_contact_halo=False)
    result = engine.fit(cache, tmp_path / "run", settings)
    assert sum(result["validation"]["class_counts"]) == 8
    state = torch.load(tmp_path / "run/recovery.pt", weights_only=False)
    assert state["settings"]["ignore_contact_halo"] is False
    with pytest.raises(ValueError, match="protocol"):
        engine.fit(cache, tmp_path / "run", replace(settings, ignore_contact_halo=True), resume=True)
    cli.infer(tmp_path / "run/best.pt", cache, tmp_path / "inference", "validation", "cpu", 0)
    assert sum(read_json(tmp_path / "inference/result.json.gz")["class_counts"]) == 8
    smoke = cli.measure(cache, settings, device="cpu", workers=0, steps=1, warmup=0,
                        compare_device_only=False, full_validation=True)
    assert sum(smoke["validation"]["class_counts"]) == 8
    pairs = DenseClips(cache, "train", "supplied", 4, training=True, augmentation="e2e", mixup=True,
                       ignore_contact_halo=False)
    for index in range(8):
        sample = pairs[index]
        for labels, ids in ((sample[1], sample[3]), (sample[6], sample[7])):
            np.testing.assert_array_equal(labels[ids >= 0].numpy(), pairs.targets[0][ids[ids >= 0].numpy()])
            assert torch.all(labels[ids < 0] == -100)
            assert torch.all(labels[ids >= 0] >= 0)


@pytest.mark.parametrize("metric,expected_best", [("loss", .3), ("rally_f1", 10.)])
def test_checkpoint_objective_ties_retention_and_stop_resume(monkeypatch, tmp_path: Path,
                                                           metric: str, expected_best: float) -> None:
    monkeypatch.setattr(engine, "SpotModel", TinyModel)
    monkeypatch.setattr(engine, "transform_clips", tiny_transform)
    losses = [.4, .3, .3, .35] if metric == "loss" else [.4, .3, .2, .1]
    values = iter(zip(losses, [0., 10., 10., 5.], strict=True))

    def validate(*_args):
        loss, f1 = next(values)
        return {"loss": loss, "seconds": 0., "metrics": {
            "1": {"fully_correct_f1": f1}, "2": {"fully_correct_f1": 100. - f1}},
            "streams": {"validation": {"value": f1}}}, [np.full((12, 3), 1 / 3)]

    monkeypatch.setattr(engine, "validate", validate)
    cache = recipe_cache(tmp_path / "cache")
    output = tmp_path / "run"
    settings = Settings(cycles=8, warmup_cycles=0, clip_length=4, frames_per_cycle=12,
                        batch_size=2, early_stopping_patience=2,
                        checkpoint_metric=metric, selection_tolerance=1)
    result = engine.fit(cache, output, settings)
    assert result["best_cycle"] == 2
    assert result["best_metric"] == expected_best
    assert result["best_loss"] == min(losses)
    assert result["completed_cycles"] == 4
    assert result["stopping_reason"] == "early_stopping"
    assert result["frames"] == 48
    assert result["validation_artifacts"] == "validation-cycle0002"
    best = torch.load(output / "best.pt", weights_only=False)
    assert best["best_metric"] == expected_best
    assert best["validation"]["loss"] == .3
    assert len(list(output.glob("validation-cycle*"))) == 1
    streams = sorted((output / "validation-streams").glob("*.json.gz"))
    assert [read_json(path)["streams"]["validation"]["value"] for path in streams] == [0., 10., 10., 5.]
    for changed in (replace(settings, selection_tolerance=2),
                    replace(settings, checkpoint_metric="loss" if metric == "rally_f1" else "rally_f1")):
        with pytest.raises(ValueError, match="protocol"):
            engine.fit(cache, output, changed, resume=True)
    before = torch.load(output / "recovery.pt", weights_only=False)
    (output / "result.json.gz").unlink()
    (output / "history.json.gz").unlink()
    monkeypatch.setattr(engine, "optimise_batch", lambda *_a, **_kw: pytest.fail("trained beyond stopping boundary"))
    assert engine.fit(cache, output, settings, resume=True) == result
    assert len(read_json(output / "history.json.gz")) == 4
    after = torch.load(output / "recovery.pt", weights_only=False)
    assert after["step"] == before["step"]


def test_zero_rally_f1_saves_first_checkpoint_and_ties_exhaust_patience(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(engine, "SpotModel", TinyModel)
    monkeypatch.setattr(engine, "transform_clips", tiny_transform)
    monkeypatch.setattr(engine, "validate", lambda *_args: (
        {"loss": 1., "seconds": 0., "metrics": {"1": {"fully_correct_f1": 0.}}, "streams": {}},
        [np.full((12, 3), 1 / 3)]))
    settings = Settings(cycles=8, warmup_cycles=0, clip_length=4, frames_per_cycle=12,
                        batch_size=2, early_stopping_patience=2,
                        checkpoint_metric="rally_f1", selection_tolerance=1)
    result = engine.fit(recipe_cache(tmp_path / "cache"), tmp_path / "run", settings)
    assert result["best_cycle"] == 1 and result["best_metric"] == 0.
    assert result["completed_cycles"] == 3 and result["stopping_reason"] == "early_stopping"


@pytest.mark.parametrize("mode", ["supplied", "discovery"])
def test_tight_metrics_include_serve_contact_and_keep_shared_tolerances(mode: str) -> None:
    from experiments.annotator.e2e_spot_benchmark.evaluate import score_predictions
    from experiments.annotator.video_models_benchmark.scoring import TOLERANCES

    video = {"item_id": "validation", "role": "validation", "fps": 30, "frame_count": 30,
             "spans": [{"span_id": "rally", "rally_id": "rally", "start_frame": 1, "end_frame": 25,
                        "label_frames": [5, 15], "label_sides": ["Bot", "Top"]}]}
    rows = np.tile([.9, .03, .03, .04] if mode == "discovery" else [.94, .03, .03], (30, 1))
    rows[7, :3] = [.02, .95, .03]
    rows[15, :3] = [.02, .03, .95]
    metrics = score_predictions([video], [rows], mode)["metrics"]
    assert set(metrics) == {"1", "2", "5", "10"}
    assert metrics["1"]["fully_correct_f1"] == 0.
    assert metrics["2"]["fully_correct_f1"] == 100.
    assert metrics["1"]["contact_f1"] == 50.
    assert metrics["1"]["correct_player_f1"] == 50.
    assert set(TOLERANCES) == {5, 10}
