"""Exercise the agreed staged comparisons without launching training."""
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from experiments.annotator.e2e_spot_benchmark.config import Settings
from experiments.annotator.e2e_spot_benchmark.search import (
    rank_candidates,
    run_name,
    run_queue,
    stage_candidates,
)
from experiments.annotator.video_models_benchmark.data import write_json


def test_complete_screening_launches_twenty_distinct_runs(monkeypatch, tmp_path: Path) -> None:
    from experiments.annotator.e2e_spot_benchmark import engine

    launched = []
    caches = {"user": tmp_path / "user-cache", "e2e": tmp_path / "e2e-cache"}

    def fake_fit(cache: Path, output: Path, settings: Settings, **_kwargs: object) -> dict:
        assert cache == caches[settings.augmentation]
        launched.append(settings)
        output.mkdir()
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
    stages = [
        ("supplied200-lr", supplied, None, 4),
        ("supplied200-recipe", supplied, "supplied200-lr", 2),
        ("supplied800", replace(supplied, batch_size=8), "supplied200-recipe", 2),
        ("discovery200-lr", discovery, None, 4),
        ("discovery200-recipe", discovery, "discovery200-lr", 2),
        ("discovery200-length", discovery, "discovery200-recipe", 2),
        ("discovery800-length", replace(discovery, batch_size=4), "discovery200-length", 4),
    ]
    selections = {}
    for stage, base, predecessor, expected_new in stages:
        selection = selections[predecessor] if predecessor else None
        candidates = stage_candidates(stage, base, selection)
        before = len(launched)
        run_queue(candidates, None, tmp_path, limit=20, device="cpu", workers=0, cache_by_augmentation=caches)
        assert len(launched) - before == expected_new
        ranking = rank_candidates(candidates, tmp_path)
        selections[stage] = {"stage": stage, **ranking[0]}
    assert len(launched) == len({run_name(settings) for settings in launched}) == 20
    assert selections["supplied200-lr"]["settings"]["learning_rate"] == .0003
    assert selections["discovery200-lr"]["settings"]["learning_rate"] == .001
    assert {settings.clip_length for settings in launched if settings.mode == "discovery"
            and settings.backbone == "800mf"} == {250, 500}
    assert all(settings.early_stopping_patience == 10 and settings.cudnn_benchmark for settings in launched)
    assert all(settings.checkpoint_metric == "rally_f1" and settings.selection_tolerance == 1
               and not settings.ignore_contact_halo for settings in launched)


@pytest.mark.parametrize("stage", ["supplied200-recipe", "discovery200-recipe"])
def test_recipe_comparison_preserves_reference_training_protocol(stage: str) -> None:
    base = Settings(augmentation="user", mode="discovery" if stage.startswith("discovery") else "supplied",
                    clip_length=250 if stage.startswith("discovery") else 100,
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


@pytest.mark.parametrize("mode", ["supplied", "discovery"])
def test_grid_ranks_mean_tight_rally_f1_before_contact_quality(mode: str, tmp_path: Path) -> None:
    base = Settings(mode=mode, checkpoint_metric="rally_f1", selection_tolerance=1)
    candidates = [replace(base, learning_rate=rate, seed=seed) for rate in (.001, .0003) for seed in (0, 1)]
    for settings in candidates:
        is_low_lr = settings.learning_rate == .0003
        f1 = 23 if is_low_lr else 20 + 4 * settings.seed
        # Discovery can have higher rally recall but lower F1 through extra predictions.
        recall = f1 if mode == "supplied" else 40 - 10 * is_low_lr
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


@pytest.mark.parametrize("stage,predecessor,mode", [
    ("supplied800", "supplied200-recipe", "supplied"),
    ("discovery800-length", "discovery200-length", "discovery"),
])
@pytest.mark.parametrize("field,old_value", [
    ("ignore_contact_halo", True), ("checkpoint_metric", "loss"), ("selection_tolerance", 10),
])
def test_larger_backbone_rejects_selection_from_another_protocol(
    stage: str, predecessor: str, mode: str, field: str, old_value: object,
) -> None:
    base = Settings(mode=mode, augmentation="user", ignore_contact_halo=False,
                    checkpoint_metric="rally_f1", selection_tolerance=1)
    selected = replace(base, **{field: old_value})
    selection = {"stage": predecessor, "settings": asdict(selected)}
    with pytest.raises(ValueError, match=field):
        stage_candidates(stage, replace(base, backbone="800mf"), selection)
