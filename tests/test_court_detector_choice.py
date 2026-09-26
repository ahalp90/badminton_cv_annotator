"""The court detector's final choice: the geometry blend."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scratch.court_det_fix.court_detector import net_choice
from scratch.court_det_fix.court_detector.detect import NET_WEIGHT, Switches

# A plausible broadcast court in a 1280x720 frame, in native pixels.
CORNERS_NATIVE = [[430.0, 250.0], [850.0, 250.0], [1060.0, 650.0], [220.0, 650.0]]
# No line fragments, so no net post is supported and the net reward is 0.
CONTEXT = SimpleNamespace(native_size=(1280, 720), size=(960, 540), segments=np.empty((0, 4)))


def row(key: str, paint: float, geometry: float) -> dict:
    return {"origin_key": key, "full_court_rank": 1, "historical_fullcourt": True, "paint_score": paint,
            "geometry_score": geometry, "net_state": "projection_failed", "posts": {}}


def test_geometry_weight_can_change_the_pick() -> None:
    rows = [row("paint_leader", 0.40, 0.40), row("geometry_leader", 0.39, 0.60)]
    assert net_choice.choose(rows, NET_WEIGHT, 4.0)[0] == "paint_leader"
    chosen, scored = net_choice.choose(rows, NET_WEIGHT, 4.0, geometry_weight=0.1)
    assert chosen == "geometry_leader"
    assert scored[1]["combined_score"] == pytest.approx(0.9 * 0.39 + 0.1 * 0.60)


def test_zero_geometry_weight_needs_no_geometry_score() -> None:
    research_row = {"paint_score": 0.4}
    assert net_choice.evidence_score(research_row, 0.0) == 0.4


@pytest.mark.parametrize("weight", [-0.1, 1.5, float("nan")])
def test_switches_refuse_a_geometry_weight_outside_zero_to_one(weight: float) -> None:
    with pytest.raises(ValueError):
        Switches(geometry_weight=weight)
