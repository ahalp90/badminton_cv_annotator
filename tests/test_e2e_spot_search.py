"""Exercise the agreed staged comparisons without launching training."""
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from experiments.annotator.e2e_spot_benchmark.config import Settings
from experiments.annotator.e2e_spot_benchmark.grid import continue_grid
from experiments.annotator.e2e_spot_benchmark.search import (
    rank_candidates,
    run_name,
    run_queue,
    stage_candidates,
)
from experiments.annotator.video_models_benchmark.data import read_json, write_json


def test_complete_screening_launches_twenty_distinct_runs(monkeypatch, tmp_path: Path) -> None:
    from experiments.annotator.e2e_spot_benchmark import engine

    launched = []
    interrupted = False
    caches = {"user": tmp_path / "user-cache", "e2e": tmp_path / "e2e-cache"}

    def fake_fit(cache: Path, output: Path, settings: Settings, **kwargs: object) -> dict:
        nonlocal interrupted
        if settings.mode == "supplied" and settings.backbone == "800mf" and settings.seed == 1:
            if not interrupted:
                interrupted = True
                output.mkdir()
                (output / "recovery.pt").touch()
                raise RuntimeError("interrupted training")
            assert kwargs["resume"] is True
        assert cache == caches[settings.augmentation]
        launched.append(settings)
        output.mkdir(exist_ok=True)
        preferred_lr = .0003 if settings.mode == "supplied" else .001
        score = 50 + 5 * (settings.learning_rate == preferred_lr) + 10 * (settings.augmentation == "e2e")
        score += 2 * (settings.clip_length == 500)
        result = {"complete": True, "stopping_reason": "early_stopping", "settings": asdict(settings),
                  "validation": {"metrics": {"1": {"fully_correct_percent": score,
                                                     "fully_correct_f1": score, "correct_player_f1": 90}}}}
        write_json(output / "result.json.gz", result)
        return result

    monkeypatch.setattr(engine, "fit", fake_fit)
    supplied = Settings(augmentation="user", batch_size=16, validation_batch_size=16,
                        early_stopping_patience=10, cudnn_benchmark=True,
                        ignore_contact_halo=False, checkpoint_metric="rally_f1", selection_tolerance=1)
    discovery = replace(supplied, mode="discovery", clip_length=250, batch_size=8, validation_batch_size=8)
    prepared = []

    def prepare(stage: str, candidates: list[Settings]) -> list[Settings]:
        prepared.append(stage)
        if stage in ("supplied800", "discovery800-length"):
            return [replace(value, batch_size=value.batch_size // 2) for value in candidates]
        return candidates

    # The original driver owns the first four runs. Continuation only ranks them.
    run_queue(stage_candidates("supplied200-lr", supplied), None, tmp_path,
              limit=4, device="cpu", workers=0, cache_by_augmentation=caches)
    with pytest.raises(RuntimeError, match="interrupted training"):
        continue_grid(tmp_path, supplied, discovery, prepare=prepare, caches=caches, device="cpu", workers=0)
    assert len(launched) == 7
    assert prepared == ["supplied200-recipe", "supplied800"]
    continue_grid(tmp_path, supplied, discovery, prepare=prepare, caches=caches, device="cpu", workers=0)
    assert len(prepared) == 6
    prepared.clear()
    continue_grid(tmp_path, supplied, discovery, prepare=prepare, caches=caches, device="cpu", workers=0)
    assert prepared == []
    selections = {stage: read_json(tmp_path / f"{stage}-selection.json.gz")
                  for stage in ("supplied200-lr", "discovery200-lr")}
    assert len(launched) == len({run_name(settings) for settings in launched}) == 20
    assert selections["supplied200-lr"]["settings"]["learning_rate"] == .0003
    assert selections["discovery200-lr"]["settings"]["learning_rate"] == .001
    assert {settings.clip_length for settings in launched if settings.mode == "discovery"
            and settings.backbone == "800mf"} == {250, 500}
    assert {(settings.batch_size, settings.validation_batch_size) for settings in launched
            if settings.mode == "discovery" and settings.backbone == "200mf"
            and settings.clip_length == 500} == {(4, 4)}
    with pytest.raises(ValueError, match="Changed grid request"):
        continue_grid(tmp_path, supplied, replace(discovery, cycles=51), prepare=prepare,
                      caches=caches, device="cpu", workers=0)
    assert len(launched) == 20
    assert all(settings.early_stopping_patience == 10 and settings.cudnn_benchmark for settings in launched)
    assert all(settings.checkpoint_metric == "rally_f1" and settings.selection_tolerance == 1
               and not settings.ignore_contact_halo for settings in launched)


def test_recipe_comparison_preserves_reference_training_protocol() -> None:
    stage = "discovery200-recipe"
    base = Settings(augmentation="user", mode="discovery", clip_length=250,
                    early_stopping_patience=10, cudnn_benchmark=True)
    selection = {"stage": stage.replace("-recipe", "-lr"), "settings": asdict(base)}
    for field, value in (("batch_size", 4), ("early_stopping_patience", 0), ("cudnn_benchmark", False)):
        with pytest.raises(ValueError, match=field):
            stage_candidates(stage, replace(base, **{field: value}), selection)


def test_discovery_starts_fresh_and_hybrid_cannot_be_a_recipe_reference() -> None:
    hybrid = Settings()
    supplied_selection = {"stage": "supplied200-lr", "settings": asdict(hybrid)}
    with pytest.raises(ValueError, match="hybrid"):
        stage_candidates("supplied200-recipe", hybrid, supplied_selection)
    with pytest.raises(ValueError, match="fresh LR search"):
        stage_candidates("discovery200-lr", hybrid, supplied_selection)
    candidates = stage_candidates("discovery200-lr", hybrid)
    assert {(settings.augmentation, settings.mixup, settings.clip_length) for settings in candidates} == {
        ("user", False, 250)}
    assert run_name(hybrid) != run_name(replace(hybrid, augmentation="user"))


def test_grid_ranks_mean_tight_rally_f1_before_contact_quality(tmp_path: Path) -> None:
    mode = "discovery"
    base = Settings(mode=mode, checkpoint_metric="rally_f1", selection_tolerance=1)
    candidates = [replace(base, learning_rate=rate, seed=seed) for rate in (.001, .0003) for seed in (0, 1)]
    for settings in candidates:
        is_low_lr = settings.learning_rate == .0003
        f1 = 23 if is_low_lr else 20 + 4 * settings.seed
        # Discovery can have higher rally recall but lower F1 through extra predictions.
        recall = 40 - 10 * is_low_lr
        tight = {"fully_correct_percent": recall, "fully_correct_f1": f1,
                 "correct_player_f1": 80 - 10 * is_low_lr}
        loose = {"fully_correct_percent": 90 - 10 * is_low_lr,
                 "fully_correct_f1": 90 - 10 * is_low_lr, "correct_player_f1": 95}
        directory = tmp_path / run_name(settings)
        directory.mkdir()
        write_json(directory / "result.json.gz", {
            "complete": True, "settings": asdict(settings),
            "validation": {"loss": .1 if not is_low_lr else .2, "metrics": {"1": tight, "10": loose}},
        })
    ranking = rank_candidates(candidates, tmp_path)
    assert ranking[0]["settings"]["learning_rate"] == .0003
    assert ranking[0]["rally_metric"] == "fully_correct_f1"
    assert [row["mean_rally_score"] for row in ranking] == [23, 22]


@pytest.mark.parametrize("field,old_value", [
    ("ignore_contact_halo", True), ("checkpoint_metric", "loss"), ("selection_tolerance", 10),
])
def test_larger_backbone_rejects_selection_from_another_protocol(
    field: str, old_value: object,
) -> None:
    stage, predecessor, mode = "discovery800-length", "discovery200-length", "discovery"
    base = Settings(mode=mode, augmentation="user", ignore_contact_halo=False,
                    checkpoint_metric="rally_f1", selection_tolerance=1)
    selected = replace(base, **{field: old_value})
    selection = {"stage": predecessor, "settings": asdict(selected)}
    with pytest.raises(ValueError, match=field):
        stage_candidates(stage, replace(base, backbone="800mf"), selection)
