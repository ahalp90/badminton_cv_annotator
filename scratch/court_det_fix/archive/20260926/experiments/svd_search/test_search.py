"""Focused cap and result-boundary checks for the SVD search runner."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / "w5_holistic"))
import automatic_generation
from run import best_agreement


@dataclass
class Settings:
    keep_axes: int


@dataclass
class PoolSettings:
    keep_candidates: int = 256
    distinct_corner_distance: float = 2.0


def test_default_and_explicit_caps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observations = SimpleNamespace(fragment_ids=np.array([], dtype=int), groups=[])
    monkeypatch.setattr(automatic_generation.assignment, "prepare_observations", lambda *_args: observations)
    helpers = ModuleType("fake_automatic")
    helpers.prepare = lambda _source: (np.empty((0, 4)), (), (100, 100))
    helpers.Settings = Settings
    helpers.CAMERA_ERROR_LIMIT = 0.1
    helpers.CAMERA_ROUNDING_MARGIN = 1e-6
    helpers.KEEP_COURTS = 256
    helpers.camera_direction_bound = lambda *_args: 0.0
    helpers.detector = SimpleNamespace(DEFAULT_SETTINGS=PoolSettings())
    selections = []

    def retain(candidates: list, settings: PoolSettings) -> list:
        selections.append(settings.keep_candidates)
        return candidates[:settings.keep_candidates]

    helpers.retain = retain
    helpers.select_pool = lambda candidates: retain(candidates, helpers.detector.DEFAULT_SETTINGS)

    def propose(*_args: object, **_kwargs: object) -> SimpleNamespace:
        candidates = [SimpleNamespace(corners_px=np.zeros((4, 2)), score=float(index)) for index in range(3)]
        empty = np.empty((0,), dtype=float)
        return SimpleNamespace(candidates=candidates, detail=lambda _position: {},
                               combined_corners=np.empty((0, 4, 2)), valid=empty, usable=empty,
                               player_any=empty, player_both_halves=empty,
                               record={"combined": 3, "axes": [{"diagnostics": {"axis_cap_excluded": 0}}]})

    helpers.propose_role = propose
    helpers.evaluate_pool = lambda _source, shortlist, *_args: shortlist
    helpers.winner_ids = lambda _entries: (None, None)
    source = {"id": "test", "dimensions": {"width": 100, "height": 100}, "all_feet_px": [[[0, 0], [0, 0]]]}
    estimator = {"points_working": [[0, 1, 1], [1, 0, 1]], "direction_lines": [[1, 0, 0], [0, 1, 0]],
                 "retained_support_masks": [[True, True], [True, True]],
                 "normalised_to_working": np.eye(3).tolist()}
    saved = {"working_size": [100, 100], "settings": {"pencil_selection": "coverage"}, "estimator": estimator}
    default = automatic_generation.generate(source, saved, None, tmp_path, helpers, max_matched_pairs=1)
    assert default["settings"]["keep_axes"] == 512
    assert default["keep_per_pair"] == default["keep_global"] == 256
    assert selections == [256, 256]
    selections.clear()
    wider = automatic_generation.generate(source, saved, None, tmp_path, helpers, keep_axes=640,
                                          keep_per_pair=2, keep_global=1, max_matched_pairs=1)
    assert wider["settings"]["keep_axes"] == 640
    assert wider["keep_per_pair"] == 2 and wider["keep_global"] == 1
    assert selections == [2, 1]
    assert wider["pairs"][0]["per_pair_cap_reached"]
    assert wider["global_cap_reached"]
    with pytest.raises(ValueError, match="positive"):
        automatic_generation.generate(source, saved, None, tmp_path, helpers, keep_axes=0)


def test_oracle_excludes_ineligible_candidate() -> None:
    reference = np.zeros((4, 2))
    candidates = [
        {"origin_key": "hard-invalid", "corners_px": np.zeros((4, 2))},
        {"origin_key": "selectable", "corners_px": np.ones((4, 2))},
    ]

    def reference_error(corners: np.ndarray, target: np.ndarray) -> dict:
        return {"maximum": float(np.max(np.abs(corners - target)))}

    verifier = {"reference_corner_error": reference_error}
    result = best_agreement(candidates, {"selectable"}, reference, verifier)
    assert result is not None and result["origin_key"] == "selectable"
    assert best_agreement(candidates, set(), reference, verifier) is None
