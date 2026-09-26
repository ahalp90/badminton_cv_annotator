"""The court detector's final choice: the geometry blend and the refitted-court score."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scratch.court_det_fix.court_detector import net_choice
from scratch.court_det_fix.court_detector.detect import NET_WEIGHT, refitted_score

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


def measured(paint: float | None, geometry: float, valid: bool = True) -> dict:
    return {"valid": valid, "corners_native_px": CORNERS_NATIVE,
            "measurement": {"paint_score": paint, "geometry_score": geometry}}


def test_refitted_score_uses_the_ranking_criterion_and_the_blend() -> None:
    paint = "q_paint10_span_weighted"
    assert refitted_score(measured(0.3, 0.5), paint, CONTEXT, 0.0) == pytest.approx(0.3)
    assert refitted_score(measured(0.3, 0.5), paint, CONTEXT, 0.1) == pytest.approx(0.32)
    # A view W5 ranked by geometry alone is rescored by geometry too.
    assert refitted_score(measured(None, 0.5), "q_geom_span_weighted", CONTEXT, 0.0) == pytest.approx(0.5)


def test_refitted_score_drops_invalid_and_unscored_courts() -> None:
    assert refitted_score(measured(0.3, 0.5, valid=False), "q_paint10_span_weighted", CONTEXT, 0.0) is None
    assert refitted_score(measured(None, 0.5), "q_paint10_span_weighted", CONTEXT, 0.0) is None
