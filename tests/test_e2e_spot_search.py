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
                  "validation": {"metrics": {"10": {"fully_correct_percent": score, "correct_player_f1": 90}}}}
        write_json(output / "result.json.gz", result)
        return result

    monkeypatch.setattr(engine, "fit", fake_fit)
    supplied = Settings(augmentation="user", batch_size=16, validation_batch_size=16,
                        early_stopping_patience=10, cudnn_benchmark=True)
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
