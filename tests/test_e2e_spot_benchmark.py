"""CPU checks for dense E2E-Spot targets, sampling, losses and recovery."""
from __future__ import annotations

import importlib.util
import lzma
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
from torch.nn import functional as F

from experiments.annotator.e2e_spot_benchmark.config import Settings
from experiments.annotator.e2e_spot_benchmark.data import (
    IGNORE,
    DenseClips,
    FrameReader,
    cycle_batches,
    dense_targets,
    overlap_starts,
)
from experiments.annotator.e2e_spot_benchmark.engine import (
    assemble_probabilities,
    fit,
    learning_rate,
    make_loader,
    masked_loss,
    validation_loss,
)
from experiments.annotator.e2e_spot_benchmark.evaluate import discovery_intervals
from experiments.annotator.e2e_spot_benchmark.model import GSM, GatedShift, SpotModel
from experiments.annotator.e2e_spot_benchmark.search import (
    rank_candidates,
    run_name,
    stage_candidates,
)
from experiments.annotator.video_models_benchmark.data import read_json, write_json


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def video_record(role: str = "train") -> dict:
    return {"item_id": role, "role": role, "fps": 30, "frame_count": 30,
            "spans": [{"span_id": "first", "rally_id": "first", "start_frame": 3, "end_frame": 10,
                       "label_frames": [5], "label_sides": ["Bot"]},
                      {"span_id": "second", "rally_id": "second", "start_frame": 21, "end_frame": 28,
                       "label_frames": [24], "label_sides": ["Top"]}]}


def build_cache(root: Path, *, complete: bool = False) -> Path:
    root.mkdir()
    videos = [video_record(role) for role in ("train", "validation")]
    write_json(root / "index.json.gz", {"videos": videos})
    for video in videos:
        directory = root / video["item_id"]
        directory.mkdir()
        intervals = [[0, 30]] if complete else [[0, 12], [20, 30]]
        ids = np.concatenate([np.arange(start, end) for start, end in intervals])
        pixels = np.broadcast_to(ids[:, None, None, None], (len(ids), 224, 256, 3)).astype(np.uint8)
        with ZipFile(directory / "frames.zip", "w") as archive:
            for frame, pixels_for_frame in zip(ids, pixels, strict=True):
                stream = BytesIO()
                Image.fromarray(pixels_for_frame).save(stream, format="JPEG", quality=95, subsampling=2)
                archive.writestr(f"{frame:09d}.jpg", stream.getvalue())
        write_json(directory / "frames.json.gz", {"image_height": 224, "image_width": 256, "geometry": "height224-aspect",
                                                               "storage": "jpeg_zip", "jpeg_quality": 95, "jpeg_subsampling": 2,
                                                               "intervals": intervals, "frames": len(ids)})
    return root


def test_scheme_a_unknowns_and_positive_precedence() -> None:
    video = {"frame_count": 15, "spans": [{"start_frame": 2, "end_frame": 13,
             "label_frames": [5, 7, 11], "label_sides": ["Bot", "Top", None]}]}
    labels = dense_targets(video, "supplied")
    assert labels.tolist() == [IGNORE, IGNORE, 0, IGNORE, 1, 1, 2, 2, 2, IGNORE, IGNORE, IGNORE, IGNORE, IGNORE, IGNORE]
    discovery = dense_targets(video, "discovery")
    np.testing.assert_array_equal(discovery[2:13], labels[2:13])
    assert discovery[[0, 1, 13, 14]].tolist() == [3, 3, 3, 3]


def test_disjoint_cache_and_context_masks(tmp_path: Path) -> None:
    cache = build_cache(tmp_path / "cache")
    reader = FrameReader(cache, video_record())
    assert reader.read(21, 23)[:, 0, 0, 0].tolist() == [21, 22]
    with pytest.raises(ValueError, match="gap"):
        reader.read(10, 22)
    dataset = DenseClips(cache, "train", "supplied", 16, training=True, frames_per_cycle=19, cycles=2)
    for index in range(len(dataset)):
        clips, labels, _, ids, _ = dataset[index][:5]
        actual = clips[0, :, 0, 0].numpy()
        assert not (np.any(actual[ids.numpy() >= 0] < 12) and np.any(actual[ids.numpy() >= 0] >= 20))
        assert np.all(labels.numpy()[ids.numpy() < 0] == IGNORE)
        assert np.all(actual[ids.numpy() < 0] == 0)
        np.testing.assert_array_equal(clips.numpy(), dataset[index][0].numpy())
    with pytest.raises(ValueError, match="gap"):
        DenseClips(cache, "train", "discovery", 4, training=True)


@pytest.mark.parametrize("count,length", [(1, 4), (4, 4), (5, 4), (11, 4), (23, 10)])
def test_overlap_short_and_tail_coverage(count: int, length: int) -> None:
    coverage = np.zeros(count, dtype=int)
    for start in overlap_starts(0, count, length):
        coverage[max(0, start):min(start + length, count)] += 1
    assert coverage.min() >= 1
    if count < length:
        expected_last = -((length - count) // 2)
    else:
        expected_last = count - length
    assert overlap_starts(0, count, length)[-1] == expected_last


def test_weighted_masked_mixup_and_unweighted_validation() -> None:
    logits = torch.tensor([[[0., 1., 2.], [2., 0., 1.], [1., 2., 0.]]], requires_grad=True)
    first = torch.tensor([[1, IGNORE, 0]])
    second = torch.tensor([[IGNORE, 2, 1]])
    weights = torch.tensor([1., 5., 5.])
    expected = F.cross_entropy(logits[0, [0, 2]], first[0, [0, 2]], weight=weights)
    torch.testing.assert_close(masked_loss(logits, first, weights), expected)
    first_sum = F.cross_entropy(logits[0, [0, 2]], first[0, [0, 2]], weight=weights, reduction="sum")
    second_sum = F.cross_entropy(logits[0, [1, 2]], second[0, [1, 2]], weight=weights, reduction="sum")
    loss = masked_loss(logits, first, weights, paired_targets=second, mixing=0.25)
    torch.testing.assert_close(loss, (0.25 * first_sum + 0.75 * second_sum) / (0.25 * 6 + 0.75 * 10))
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    ignored = torch.full_like(first, IGNORE)
    assert masked_loss(logits, ignored, weights).item() == 0
    rows = logits.detach().softmax(-1)[0].numpy()
    result = validation_loss([rows], [first[0].numpy()], 3)
    expected_validation = F.cross_entropy(logits[0, [0, 2]], first[0, [0, 2]])
    assert result["loss"] == pytest.approx(float(expected_validation.detach()))
    assert result["class_counts"] == [1, 1, 0]


def test_exact_frame_budget_and_warmup(tmp_path: Path) -> None:
    settings = Settings(cycles=50, frames_per_cycle=11, warmup_cycles=3, clip_length=4, batch_size=2)
    assert learning_rate(settings, 0) == 0
    assert learning_rate(settings, 11) == pytest.approx(settings.learning_rate / 3)
    assert learning_rate(settings, 33) == settings.learning_rate
    assert learning_rate(settings, 11 * 50) == 0
    midpoint = int(11 * (3 + 47 / 2))
    assert learning_rate(settings, midpoint) == pytest.approx(settings.learning_rate / 2, abs=2e-6)
    cache = build_cache(tmp_path / "cache")
    dataset = DenseClips(cache, "train", "supplied", 4, training=True, frames_per_cycle=11, cycles=2)
    batches = cycle_batches(dataset.samples_per_cycle, 2, 2)
    assert batches == [[0, 1], [2], [3, 4], [5]]
    assert sum(dataset[index][-1] for batch in batches for index in batch) == 22
    assert cycle_batches(3, 2, 2, start_cycle=0, start_batch=1) == batches[1:]


def test_discovery_sampling_uses_full_timelines(tmp_path: Path) -> None:
    cache = build_cache(tmp_path / "cache", complete=True)
    dataset = DenseClips(cache, "train", "discovery", 4, training=True, frames_per_cycle=400, cycles=1)
    assert dataset.scopes == [(0, 0, 30)]
    assert any((dataset[index][1] == 3).any() for index in range(100))
    rows = np.array([[.2, .2, .2, .4], [.1, .1, .1, .7], [.3, .1, .1, .5]])
    assert discovery_intervals(rows) == [(0, 1), (2, 3)]


def test_overlap_averages_original_frames_once(monkeypatch, tmp_path: Path) -> None:
    from experiments.annotator.e2e_spot_benchmark import engine

    class FrameModel(nn.Module):
        def forward(self, clips: torch.Tensor) -> torch.Tensor:
            values = clips[:, :, 0, 0, 0]
            return torch.stack((values, values * 0, -values), -1)

    monkeypatch.setattr(engine, "transform_clips",
                        lambda clips, valid_frames, training, **_kwargs: clips.float().permute(0, 2, 1, 3, 4))
    dataset = DenseClips(build_cache(tmp_path / "cache"), "validation", "supplied", 16)
    loader = make_loader(dataset, 2, 0)
    predictions = assemble_probabilities(FrameModel(), dataset, loader, torch.device("cpu"), 3)
    for _, start, end in dataset.scopes:
        frames = torch.arange(start, end).float()
        expected = torch.stack((frames, frames * 0, -frames), -1).softmax(-1).numpy()
        np.testing.assert_allclose(predictions[0][start:end], expected, atol=1e-7)
    assert np.all(predictions[0][12:20] == 0)


def test_gsm_numeric_and_gradient_parity_with_upstream() -> None:
    source = Path(__file__).resolve().parents[2] / "spot/model/impl/gsm.py"
    if not source.exists():
        pytest.skip("Optional upstream checkout is unavailable")
    spec = importlib.util.spec_from_file_location("e2e_spot_reference_gsm", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ftens = lambda *shape: torch.empty(shape)
    reference = module._GSM(8, 3).eval()
    candidate = GSM(8, 3).eval()
    with torch.no_grad():
        reference.conv3D.weight.normal_()
        reference.conv3D.bias.normal_()
    candidate.load_state_dict(reference.state_dict())
    inputs = torch.randn(6, 8, 5, 5, requires_grad=True)
    comparison = inputs.detach().clone().requires_grad_()
    actual, expected = candidate(inputs), reference(comparison)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(inputs.grad, comparison.grad, rtol=0, atol=0)
    for ours, upstream in zip(candidate.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(ours.grad, upstream.grad, rtol=0, atol=0)


@pytest.mark.parametrize("backbone,features", [("200mf", 368), ("800mf", 768)])
def test_actual_architecture_cpu_inference(backbone: str, features: int) -> None:
    model = SpotModel(backbone, 2, 4, pretrained=False).eval()
    assert model.gru.input_size == features
    assert model.gru.hidden_size == features
    assert model.gru.bidirectional and model.gru.num_layers == 1
    assert model.dropout.p == .5
    assert all(parameter.requires_grad for parameter in model.parameters())
    for stage in (model.features.s1, model.features.s2, model.features.s3, model.features.s4):
        assert all(isinstance(block.conv1, GatedShift) for block in stage.children())
    with torch.no_grad():
        logits = model(torch.randn(1, 2, 3, 224, 224))
    assert logits.shape == (1, 2, 4)
    assert torch.isfinite(logits).all()
    if backbone == "200mf":
        model.train()
        logits = model(torch.randn(1, 2, 3, 224, 224))
        masked_loss(logits, torch.tensor([[1, 2]])).backward()
        assert model.features.stem.conv.weight.grad.abs().sum() > 0
        assert model.gru.weight_hh_l0.grad.abs().sum() > 0
        # timm initially zeros each residual branch's final BN scale. The first
        # update opens that branch; the next backward pass must reach its GSM.
        optimiser = torch.optim.SGD(model.parameters(), lr=.01)
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)
        masked_loss(model(torch.randn(1, 2, 3, 224, 224)), torch.tensor([[1, 2]])).backward()
        assert model.features.s1.b1.conv1.gsm.conv3D.weight.grad.abs().sum() > 0


def test_two_seed_selection_and_five_seed_completion(tmp_path: Path) -> None:
    candidates = stage_candidates("supplied200-lr", Settings())
    for settings in candidates:
        output = tmp_path / run_name(settings)
        output.mkdir()
        write_json(output / "result.json.gz", {"complete": True, "settings": asdict(settings),
                   "validation": {"metrics": {"10": {"fully_correct_percent": 50,
                                  "correct_player_f1": 75 if settings.learning_rate == .0003 else 70}}}})
    ranking = rank_candidates(candidates, tmp_path)
    winner = Settings(**ranking[0]["settings"])
    assert winner.learning_rate == .0003
    selection = {"stage": "supplied200-recipe", "settings": asdict(winner)}
    assert [value.seed for value in stage_candidates("complete", winner, selection)] == list(range(5))
    assert all(value.learning_rate == .0003 for value in stage_candidates("complete", winner, selection))
    changed = [replace(value, frames_per_cycle=100) for value in candidates]
    with pytest.raises(ValueError, match="protocol"):
        rank_candidates(changed, tmp_path)


def test_training_resume_preserves_clips_rng_optimizer_and_selection(monkeypatch, tmp_path: Path) -> None:
    from experiments.annotator.e2e_spot_benchmark import engine

    class TinyModel(nn.Module):
        def __init__(self, _backbone: str, _length: int, classes: int, *, pretrained: bool) -> None:
            super().__init__()
            self.dropout = nn.Dropout(.3)
            self.classifier = nn.Linear(3, classes)

        def forward(self, clips: torch.Tensor) -> torch.Tensor:
            return self.classifier(self.dropout(clips.mean((-1, -2))))

    def transform(clips: torch.Tensor, valid_frames: torch.Tensor, *, training: bool,
                  augmentation: str = "hybrid") -> torch.Tensor:
        pixels = clips.float().permute(0, 2, 1, 3, 4) / 30
        pixels = pixels * torch.rand(()) if training else pixels
        return pixels.masked_fill(~valid_frames[:, :, None, None, None], 0)

    monkeypatch.setattr(engine, "SpotModel", TinyModel)
    monkeypatch.setattr(engine, "transform_clips", transform)
    cache = build_cache(tmp_path / "cache")
    settings = Settings(cycles=2, warmup_cycles=1, frames_per_cycle=11, clip_length=4, batch_size=2, mixup=True)
    full = fit(cache, tmp_path / "full", settings, workers=1)
    interrupted = fit(cache, tmp_path / "resume", settings, workers=1, max_steps=1)
    assert not interrupted["complete"]
    resumed = fit(cache, tmp_path / "resume", settings, workers=1, resume=True)
    assert full["frames"] == resumed["frames"] == 22
    assert full["steps"] == resumed["steps"] == 4
    for name, result in (("full", full), ("resume", resumed)):
        artifacts = tmp_path / name / result["validation_artifacts"]
        saved = read_json(artifacts / "result.json.gz")
        assert saved["loss"] == result["validation"]["loss"]
        assert saved["streams"]
        with lzma.open(artifacts / "validation.npy.xz", "rb") as stream:
            probabilities = np.load(stream)
        assert probabilities.shape == (30, 3)
        assert np.all(probabilities[12:20] == 0)
        assert len(list((tmp_path / name).glob("validation-cycle*"))) == 1
    assert full["best_cycle"] == resumed["best_cycle"]
    assert full["validation"]["loss"] == resumed["validation"]["loss"]
    final_states = [torch.load(tmp_path / name / "recovery.pt", weights_only=False) for name in ("full", "resume")]
    for name, parameter in final_states[0]["model"].items():
        torch.testing.assert_close(parameter, final_states[1]["model"][name], rtol=0, atol=0)
    assert read_json(tmp_path / "resume/result.json.gz")["complete"]
    with pytest.raises(ValueError, match="protocol"):
        fit(cache, tmp_path / "resume", replace(settings, learning_rate=.0003), resume=True)


def test_author_crop_before_collation_and_no_second_spatial_transform(tmp_path: Path) -> None:
    from experiments.annotator.e2e_spot_benchmark.model import transform_clips

    cache = build_cache(tmp_path / "cache")
    directory = cache / "train"
    with ZipFile(directory / "frames.zip", "w") as archive:
        columns = np.broadcast_to(np.arange(256, dtype=np.uint8)[None, :, None], (224, 256, 3))
        stream = BytesIO()
        Image.fromarray(columns).save(stream, format="JPEG", quality=95, subsampling=2)
        for frame in list(range(12)) + list(range(20, 30)):
            archive.writestr(f"{frame:09d}.jpg", stream.getvalue())
    evaluation = DenseClips(cache, "train", "supplied", 4)
    centred, *_ = evaluation[0]
    reader = FrameReader(cache, video_record())
    full = reader.read(3, 7)
    np.testing.assert_array_equal(centred.permute(1, 2, 3, 0), full[:, :, 16:240])
    training = DenseClips(cache, "train", "supplied", 4, training=True, frames_per_cycle=80)
    left_pixels = set()
    for index in range(20):
        clip, _, _, ids, _ = training[index][:5]
        assert clip.shape == (3, 4, 224, 224)
        torch.testing.assert_close(clip, training[index][0], rtol=0, atol=0)
        valid_positions = torch.nonzero(ids >= 0).flatten()
        assert torch.equal(clip[:, valid_positions[0]], clip[:, valid_positions[-1]])
        left_pixels.add(int(clip[0, valid_positions[0], 0, 0]))
    assert len(left_pixels) > 1
    result = transform_clips(centred.unsqueeze(0), torch.ones((1, 4), dtype=torch.bool), training=False)
    mean = torch.tensor([.485, .456, .406])[None, None, :, None, None]
    std = torch.tensor([.229, .224, .225])[None, None, :, None, None]
    expected = centred.permute(1, 0, 2, 3).unsqueeze(0).float() / 255
    torch.testing.assert_close(result * std + mean, expected)


def test_fresh_source_preparation_jpeg420_geometry_and_reuse(tmp_path: Path) -> None:
    from experiments.annotator.e2e_spot_benchmark.prepare import prepare_video

    source = tmp_path / "source.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30",
                    "-frames:v", "8", "-threads", "1", "-c:v", "ffv1", str(source)], check=True)
    video = {"item_id": "sample", "frame_count": 8, "source_video": source.name,
             "spans": [{"start_frame": 2, "end_frame": 6}]}
    cache = tmp_path / "cache"
    result = prepare_video(video, tmp_path, cache, "supplied")
    assert result["frames"] == 4
    metadata = read_json(cache / "sample/frames.json.gz")
    assert metadata["image_height"] == 224
    assert metadata["image_width"] == 398
    assert metadata["intervals"] == [[2, 6]]
    assert metadata["jpeg_quality"] == 95 and metadata["jpeg_subsampling"] == 2
    reader = FrameReader(cache, video)
    assert reader.read(2, 4).shape == (2, 224, 398, 3)
    with ZipFile(cache / "sample/frames.zip") as archive, Image.open(BytesIO(archive.read("000000002.jpg"))) as image:
        assert JpegImagePlugin.get_sampling(image) == 2
    assert prepare_video(video, tmp_path, cache, "supplied")["reused"]
    with pytest.raises(ValueError, match="metadata differs"):
        prepare_video(video, tmp_path, cache, "discovery")


def test_one_to_one_rallies_duplicate_fp_missing_fn_and_shared_recall() -> None:
    from annotator.types import ContactEvent, ContactSequence, CourtHalf
    from experiments.annotator.e2e_spot_benchmark.evaluate import (
        fully_correct_rallies,
        rally_rates,
    )
    from experiments.annotator.video_models_benchmark.scoring import score_video_rules

    video = video_record("validation")
    first = ContactSequence("first", 3, 10, (ContactEvent(5, .9, CourtHalf.BOT),))
    duplicate = ContactSequence("duplicate", 3, 10, first.events)
    second = ContactSequence("second", 21, 28, (ContactEvent(24, .9, CourtHalf.TOP),))
    wrong = ContactSequence("wrong", 21, 28, (ContactEvent(24, .9, CourtHalf.BOT),))
    empty = ContactSequence("empty", 12, 18, ())
    actual = fully_correct_rallies(video, [first, duplicate, wrong, empty], 10)
    assert actual == {"fully_correct": 1, "predicted_rallies": 3, "rally_fp": 2, "rally_fn": 1}
    assert fully_correct_rallies(video, [empty], 10)["rally_fn"] == 2
    shared = score_video_rules(video, {"decoded": [first, second]}, 10)["decoded"]
    counts = {**shared, **fully_correct_rallies(video, [first, second], 10)}
    assert counts["fully_correct"] == shared["fully_correct"] == 2
    rates = rally_rates(counts)
    assert rates["fully_correct_recall"] == rates["fully_correct_percent"] == 100
    assert rates["fully_correct_precision"] == rates["fully_correct_f1"] == 100


def test_discovery_scores_accepted_contact_endpoints_like_existing_benchmark() -> None:
    from experiments.annotator.contact_records import stream_from_payload
    from experiments.annotator.e2e_spot_benchmark.evaluate import score_predictions
    from experiments.annotator.monotrack_benchmark.benchmark_scripts.common import (
        labels_for_item,
    )
    from experiments.annotator.monotrack_benchmark.writeup_scripts.metrics import (
        judge_found_rallies,
    )

    video = video_record("validation")
    rows = np.tile([.8, .05, .05, .1], (30, 1))
    rows[5] = [.05, .9, .02, .03]
    # Rally probability exceeds .5, but NOT_RALLY wins the four-way argmax.
    rows[14] = [.2, .3, .1, .4]
    result = score_predictions([video], [rows], "discovery")
    counts = result["metrics"]["10"]
    assert counts["fully_correct"] == 1
    assert counts["rally_fn"] == 1 and counts["rally_fp"] == 0
    sequences, _ = stream_from_payload(result["streams"]["validation"])
    assert (sequences[0].start_frame, sequences[0].end_frame) == (0, 30)
    assert len(sequences[0].events) == 1
    judged = judge_found_rallies(sequences, labels_for_item(video), 10, fps=30)
    assert judged[0]["outcome"] == "fully_correct"
    assert judged[0]["rally_id"] == "first"


def test_timing_cpu_device_batch_feeder_and_full_validation(tmp_path: Path) -> None:
    from experiments.annotator.e2e_spot_benchmark.__main__ import measure

    cache = build_cache(tmp_path / "cache")
    settings = Settings(clip_length=2, batch_size=1, validation_batch_size=3)
    result = measure(cache, settings, device="cpu", workers=0, steps=1, warmup=0, full_validation=True)
    assert result["pretrained"] is False
    for name in ("device_only", "end_to_end"):
        assert result[name]["frames"] == 2
        assert result[name]["frames_per_second"] > 0
    assert result["validation"]["seconds"] > 0
    assert sum(result["validation"]["class_counts"]) == 10


def test_checkpoint_tie_keeps_earliest_validation_artifacts(monkeypatch, tmp_path: Path) -> None:
    from experiments.annotator.e2e_spot_benchmark import engine

    class TinyModel(nn.Module):
        def __init__(self, _backbone: str, _length: int, classes: int, *, pretrained: bool) -> None:
            super().__init__()
            self.head = nn.Linear(3, classes)

        def forward(self, clips: torch.Tensor) -> torch.Tensor:
            return self.head(clips.mean((-1, -2)))

    def tied_validation(*_args):
        return {"loss": 1.0, "seconds": 0.0, "metrics": {}, "streams": {"validation": {}}}, [np.full((30, 3), 1 / 3)]

    monkeypatch.setattr(engine, "SpotModel", TinyModel)
    monkeypatch.setattr(engine, "validate", tied_validation)
    cache = build_cache(tmp_path / "cache")
    settings = Settings(clip_length=2, batch_size=2, cycles=2, warmup_cycles=0, frames_per_cycle=4)
    result = fit(cache, tmp_path / "run", settings)
    assert result["best_cycle"] == 1
    assert result["validation_artifacts"] == "validation-cycle0001"
    assert len(list((tmp_path / "run").glob("validation-cycle*"))) == 1


@pytest.mark.parametrize("role,predicted,fully_correct", [("test", 1, 1), ("validation", 3, 0)])
def test_discovery_shared_exclusions_precede_endpoints_and_empty_omission(
    monkeypatch, role: str, predicted: int, fully_correct: int,
) -> None:
    from experiments.annotator.contact_records import stream_from_payload
    from experiments.annotator.e2e_spot_benchmark import evaluate

    video = {**video_record(role), "frame_count": 100}
    monkeypatch.setattr(evaluate, "left_out_windows_for", lambda videos: {videos[0]["item_id"]: [(50, 100)]})
    rows = np.tile([.03, .03, .04, .9], (100, 1))
    rows[:70] = [.8, .05, .05, .1]
    rows[80:] = [.8, .05, .05, .1]
    rows[5] = [.03, .9, .02, .05]
    rows[60] = rows[90] = [.03, .02, .9, .05]
    result = evaluate.score_predictions([video], [rows], "discovery")
    for counts in result["metrics"].values():
        assert counts["predicted"] == predicted
        assert counts["fully_correct"] == fully_correct
        assert counts["predicted_rallies"] == (1 if role == "test" else 2)
    # Saved outputs retain both mask spans and all contacts for later rescoring.
    sequences, _ = stream_from_payload(result["streams"][role])
    assert [(sequence.start_frame, sequence.end_frame) for sequence in sequences] == [(0, 70), (80, 100)]
    assert [event.frame for sequence in sequences for event in sequence.events] == [5, 60, 90]


def test_first_step_failure_allows_unchanged_restart_but_protects_checkpoint(monkeypatch, tmp_path: Path) -> None:
    from experiments.annotator.e2e_spot_benchmark import engine

    class TinyModel(nn.Module):
        def __init__(self, _backbone: str, _length: int, classes: int, *, pretrained: bool) -> None:
            super().__init__()
            self.head = nn.Linear(3, classes)

        def forward(self, clips: torch.Tensor) -> torch.Tensor:
            return self.head(clips.mean((-1, -2)))

    original_step = engine.optimise_batch

    def fail_first_step(*_args):
        raise RuntimeError("simulated first-step failure")

    monkeypatch.setattr(engine, "SpotModel", TinyModel)
    monkeypatch.setattr(engine, "optimise_batch", fail_first_step)
    cache = build_cache(tmp_path / "cache")
    output = tmp_path / "run"
    settings = Settings(clip_length=4, batch_size=2, cycles=2, warmup_cycles=0, frames_per_cycle=11)
    with pytest.raises(RuntimeError, match="first-step"):
        fit(cache, output, settings)
    assert (output / "settings.json.gz").exists()
    assert (output / "inputs.json.gz").exists()
    assert not (output / "recovery.pt").exists()
    with pytest.raises(ValueError, match="different protocol"):
        fit(cache, output, replace(settings, batch_size=1))
    index = read_json(cache / "index.json.gz")
    index["videos"][0]["spans"][0]["label_sides"] = ["Top"]
    write_json(cache / "index.json.gz", index)
    with pytest.raises(ValueError, match="Cache labels"):
        fit(cache, output, settings)
    index["videos"][0]["spans"][0]["label_sides"] = ["Bot"]
    write_json(cache / "index.json.gz", index)
    monkeypatch.setattr(engine, "optimise_batch", original_step)
    restarted = fit(cache, output, settings, max_steps=1)
    assert restarted["steps"] == 1 and (output / "recovery.pt").exists()
    with pytest.raises(ValueError, match="checkpoint already exists"):
        fit(cache, output, settings)
    resumed = fit(cache, output, settings, resume=True, max_steps=1)
    assert resumed["steps"] == 2


@pytest.mark.parametrize("stage,predecessor,mode", [
    ("supplied200-recipe", "supplied200-lr", "supplied"),
    ("supplied800", "supplied200-recipe", "supplied"),
    ("discovery200-recipe", "discovery200-lr", "discovery"),
    ("discovery200-length", "discovery200-recipe", "discovery"),
    ("discovery800-length", "discovery200-length", "discovery"),
])
def test_search_stage_uses_explicit_config_runtime_and_exact_predecessor(stage: str, predecessor: str, mode: str) -> None:
    base = Settings(clip_length=250 if mode == "discovery" else 100, batch_size=2, validation_batch_size=7,
                    cycles=12, warmup_cycles=2, frames_per_cycle=1234)
    selected = replace(base, mode=mode, learning_rate=.0003,
                       augmentation="user" if stage.endswith("-recipe") else "e2e",
                       mixup=not stage.endswith("-recipe"))
    selection = {"stage": predecessor, "settings": asdict(selected)}
    candidates = stage_candidates(stage, base, selection)
    for candidate in candidates:
        assert (candidate.validation_batch_size, candidate.cycles) == (7, 12)
        assert (candidate.warmup_cycles, candidate.frames_per_cycle) == (2, 1234)
        assert candidate.learning_rate == selected.learning_rate
        expected_batch = 1 if candidate.clip_length == 500 else 2
        assert candidate.batch_size == expected_batch
    if stage.endswith("-recipe"):
        assert {(value.augmentation, value.mixup) for value in candidates} == {("user", False), ("e2e", True)}
    else:
        assert all(candidate.mixup for candidate in candidates)
    if stage == "discovery800-length":
        assert {candidate.clip_length for candidate in candidates} == {250, 500}
    wrong_stage = "supplied200-lr" if predecessor != "supplied200-lr" else "supplied200-recipe"
    with pytest.raises(ValueError, match="requires a selection from"):
        stage_candidates(stage, base, {**selection, "stage": wrong_stage})


def test_search_completion_preserves_full_protocol_and_reuses_screening_seeds(monkeypatch, tmp_path: Path) -> None:
    from experiments.annotator.e2e_spot_benchmark import engine
    from experiments.annotator.e2e_spot_benchmark.search import run_queue

    winner = Settings(augmentation="user", learning_rate=.0003, batch_size=2, validation_batch_size=7, cycles=12)
    selection = {"stage": "supplied200-recipe", "settings": asdict(winner)}
    completion = stage_candidates("complete", winner, selection)
    assert completion == [replace(winner, seed=seed) for seed in range(5)]
    for field in ("batch_size", "validation_batch_size", "cycles", "frames_per_cycle"):
        changed = replace(winner, **{field: getattr(winner, field) + 1})
        with pytest.raises(ValueError, match=field):
            stage_candidates("complete", changed, selection)
    with pytest.raises(ValueError, match="supplied200-recipe"):
        stage_candidates("complete", winner, {**selection, "stage": "supplied200-lr"})
    screening = stage_candidates("supplied200-lr", winner)
    for settings in screening:
        if settings.learning_rate != winner.learning_rate:
            continue
        output = tmp_path / run_name(settings)
        output.mkdir()
        write_json(output / "result.json.gz", {"complete": True, "settings": asdict(settings)})
    launched = []

    def record_fit(_cache, _output, settings, **_kwargs):
        launched.append(settings)
        return {"complete": True, "settings": asdict(settings)}

    monkeypatch.setattr(engine, "fit", record_fit)
    run_queue(completion, tmp_path, tmp_path, limit=3, device="cpu", workers=0)
    assert [settings.seed for settings in launched] == [2, 3, 4]


@pytest.mark.parametrize("unknown_side", [False, True])
def test_supplied_centres_use_only_one_rally_and_zero_padding(monkeypatch, tmp_path: Path, unknown_side: bool) -> None:
    from experiments.annotator.e2e_spot_benchmark.model import transform_clips

    cache = build_cache(tmp_path / "cache", complete=True)
    if unknown_side:
        index = read_json(cache / "index.json.gz")
        index["videos"][0]["spans"][1]["label_sides"] = [None]
        write_json(cache / "index.json.gz", index)
    dataset = DenseClips(cache, "train", "supplied", 16, training=True, frames_per_cycle=1600, cycles=1)
    reader = dataset.readers[0]
    original_read = reader.read
    reads = []

    def read_scope(start, end):
        assert (start, end) in ((3, 10), (21, 28))
        reads.append((start, end))
        return original_read(start, end)

    monkeypatch.setattr(reader, "read", read_scope)
    centres = set()
    for sample in range(len(dataset)):
        clips, labels, _, ids, _ = dataset[sample][:5]
        valid = ids >= 0
        centre = int(ids[dataset.length // 2])
        centres.add(centre)
        expected_ids = np.arange(centre - 8, centre + 8)
        scope_start, scope_end = reads[-1]
        expected_ids[(expected_ids < scope_start) | (expected_ids >= scope_end)] = -1
        np.testing.assert_array_equal(ids, expected_ids)
        assert valid.sum() == 7
        assert torch.all(clips[:, ~valid] == 0)
        ignored_real = valid & (labels == IGNORE)
        assert ignored_real.any() and torch.all(clips[:, ignored_real] > 0)
        pixels = transform_clips(clips.unsqueeze(0), valid.unsqueeze(0), training=False)[0]
        assert torch.all(pixels[~valid] == 0)
        assert torch.all(pixels[ignored_real] != 0)
    assert centres == set(range(3, 10)) | set(range(21, 28))
    assert set(reads) == {(3, 10), (21, 28)}
    assert dataset.frames_per_cycle == 1600
    partial = DenseClips(cache, "train", "supplied", 16, training=True, frames_per_cycle=1, cycles=1)
    clips, labels, _, ids, positions = partial[0][:5]
    assert positions == 1 and torch.all(labels == IGNORE)
    assert torch.all(clips[:, ids >= 0] > 0)


def test_supplied_scopes_remain_separate_and_short_clips_centre_padding(tmp_path: Path) -> None:
    cache = build_cache(tmp_path / "cache", complete=True)
    index = read_json(cache / "index.json.gz")
    index["videos"][0]["spans"][1].update(start_frame=9, end_frame=16, label_frames=[12])
    write_json(cache / "index.json.gz", index)
    dataset = DenseClips(cache, "train", "supplied", 16)
    assert dataset.scopes == [(0, 3, 10), (0, 9, 16)]
    for sample, expected_ids in enumerate((list(range(3, 10)), list(range(9, 16)))):
        clips, labels, _, ids, _ = dataset[sample][:5]
        assert ids.tolist() == [-1] * 4 + expected_ids + [-1] * 5
        assert torch.all(clips[:, :4] == 0) and torch.all(clips[:, -5:] == 0)
        assert torch.all(labels[:4] == IGNORE) and torch.all(labels[-5:] == IGNORE)


def test_preparation_keeps_only_supplied_bounds_and_merges_storage() -> None:
    from experiments.annotator.e2e_spot_benchmark.prepare import preparation_intervals

    video = {"frame_count": 2000, "spans": [{"start_frame": 500, "end_frame": 750},
                                            {"start_frame": 700, "end_frame": 800}]}
    assert preparation_intervals(video, "supplied") == [[500, 800]]
    assert preparation_intervals(video, "discovery") == [[0, 2000]]


@pytest.mark.parametrize("training", [False, True])
def test_normalised_padding_is_zero_and_does_not_change_real_colour(training: bool) -> None:
    from experiments.annotator.e2e_spot_benchmark.model import transform_clips

    real = torch.randint(30, 200, (1, 3, 2, 224, 224), dtype=torch.uint8)
    padded = torch.full((1, 3, 5, 224, 224), 255, dtype=torch.uint8)
    padded[:, :, 1:3] = real
    valid = torch.tensor([[False, True, True, False, False]])
    torch.manual_seed(13)
    expected = transform_clips(real, torch.ones((1, 2), dtype=torch.bool), training=training)
    torch.manual_seed(13)
    actual = transform_clips(padded, valid, training=training)
    torch.testing.assert_close(actual[:, 1:3], expected)
    assert torch.count_nonzero(actual[~valid]) == 0


def test_mixup_preserves_partner_frames_at_original_padding(monkeypatch) -> None:
    from experiments.annotator.e2e_spot_benchmark import engine
    from experiments.annotator.e2e_spot_benchmark.model import transform_clips

    class CaptureModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Linear(3, 3)
            self.inputs = None

        def forward(self, pixels: torch.Tensor) -> torch.Tensor:
            self.inputs = pixels.detach().clone()
            return self.head(pixels.mean((-1, -2)))

    clips = torch.full((2, 3, 2, 224, 224), 80, dtype=torch.uint8)
    valid = torch.tensor([[False, True], [True, False]])
    labels = torch.tensor([[IGNORE, 1], [2, IGNORE]])
    monkeypatch.setattr(engine.np.random, "beta", lambda *_args: .5)
    monkeypatch.setattr(engine.torch, "randperm", lambda _count, device: torch.tensor([1, 0], device=device))
    torch.manual_seed(7)
    transformed = transform_clips(clips, valid, training=True)
    expected = .5 * transformed + .5 * transformed.flip(0)
    model = CaptureModel()
    optimiser = torch.optim.AdamW(model.parameters())
    torch.manual_seed(7)
    engine.optimise_batch(model, optimiser, clips, labels, valid, Settings(mixup=True),
                          torch.device("cpu"), torch.tensor([1., 5., 5.]))
    torch.testing.assert_close(model.inputs, expected)
    assert torch.count_nonzero(model.inputs[~valid]) > 0


def test_search_length_batches_plan_run_rank_and_frozen_completion(monkeypatch, tmp_path: Path) -> None:
    from experiments.annotator.e2e_spot_benchmark import engine, search

    config = tmp_path / "discovery.toml"
    config.write_text("clip_length = 250\nbatch_size = 8\nvalidation_batch_size = 3\n")
    selection = tmp_path / "discovery-selection.json.gz"
    selected = Settings(mode="discovery", augmentation="user", clip_length=250, batch_size=8, validation_batch_size=3)
    write_json(selection, {"stage": "discovery200-recipe", "settings": asdict(selected)})
    root = tmp_path / "runs"
    arguments = ["--stage", "discovery200-length", "--config", str(config), "--selection", str(selection),
                 "--runs", str(root), "--batch-size-for-length", "250=8", "--batch-size-for-length", "500=4"]
    search.main(["plan", *arguments])
    plan = read_json(root / "discovery200-length-plan.json.gz")
    assert len(plan["candidates"]) == 4
    assert {(row["clip_length"], row["batch_size"]) for row in plan["candidates"]} == {(250, 8), (500, 4)}
    for row in plan["candidates"]:
        assert row["validation_batch_size"] == 3
        output = root / run_name(Settings(**row))
        output.mkdir()
        recall = 80 if row["clip_length"] == 500 else 50
        write_json(output / "result.json.gz", {"complete": True, "settings": row,
                   "validation": {"metrics": {"10": {"fully_correct_percent": recall, "correct_player_f1": 75}}}})
    launched = []

    def record_fit(_cache, _output, settings, **_kwargs):
        launched.append(settings)
        return {"complete": True, "settings": asdict(settings)}

    monkeypatch.setattr(engine, "fit", record_fit)
    search.main(["run", *arguments, "--user-cache", str(tmp_path / "unused-cache")])
    assert launched == []
    search.main(["rank", *arguments])
    winner_record = read_json(root / "discovery200-length-selection.json.gz")
    winner = Settings(**winner_record["settings"])
    assert (winner.clip_length, winner.batch_size) == (500, 4)
    completion = stage_candidates("complete", winner, winner_record)
    assert completion == [replace(winner, seed=seed) for seed in range(5)]
    search.run_queue(completion, tmp_path, root, limit=3, device="cpu", workers=0)
    assert [settings.seed for settings in launched] == [2, 3, 4]
    assert all(settings.batch_size == 4 for settings in launched)
    changed_batches = stage_candidates("discovery200-length", selected, read_json(selection), batch_by_length={500: 8})
    with pytest.raises(ValueError, match="changed protocol"):
        rank_candidates(changed_batches, root)


def test_search_length_batch_fallback_and_rejected_overrides() -> None:
    base = Settings(mode="discovery", augmentation="user", clip_length=250, batch_size=8)
    selection = {"stage": "discovery200-recipe", "settings": asdict(base)}
    candidates = stage_candidates("discovery200-length", base, selection, batch_by_length={500: 4})
    assert {(value.clip_length, value.batch_size) for value in candidates} == {(250, 8), (500, 4)}
    assert candidates == stage_candidates("discovery200-length", base, selection)
    for invalid in ({100: 4}, {500: 0}, {250: -1}):
        with pytest.raises(ValueError, match="positive batch size"):
            stage_candidates("discovery200-length", base, selection, batch_by_length=invalid)
    for stage in ("supplied200-lr", "supplied800", "discovery200-lr", "complete"):
        with pytest.raises(ValueError, match="only to discovery length stages"):
            stage_candidates(stage, base, selection, batch_by_length={500: 4})


@pytest.mark.parametrize("entries", [["500"], ["500=four"], ["500=4", "500=8"]])
def test_search_length_batch_cli_rejects_malformed_or_repeated(entries: list[str], tmp_path: Path) -> None:
    from experiments.annotator.e2e_spot_benchmark.search import main

    arguments = ["plan", "--stage", "discovery200-length", "--config", str(tmp_path / "unused.toml"),
                 "--runs", str(tmp_path)]
    for entry in entries:
        arguments.extend(("--batch-size-for-length", entry))
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
