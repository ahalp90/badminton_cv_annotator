"""Missing views can hide a flight without changing its contact time or hitter."""

from dataclasses import replace

import numpy as np
import pytest

from annotator.config import BaseAnnotatorConfig
from annotator.courts.scenes import SceneCourt
from annotator.outcomes.point_winner import (
    Half,
    Landing,
    LandingFilterOptions,
    LandingKinematics,
    Verdict,
    VerdictSource,
    geometric_verdict,
    rally_verdict,
)
from annotator.outcomes.video import LandingContext, build_rally_outcome
from annotator.resolve import resolve
from annotator.types import compute_speed


def returning_view_context() -> LandingContext:
    track = np.zeros((25, 3))
    # Only the returning scene's descent may determine the landing.
    track[3:6] = [(0.5, 0.1, 1), (0.5, 0.2, 1), (0.5, 0.3, 1)]
    track[10:14] = [(0.5, 0.45, 1), (0.5, 0.47, 1), (0.5, 0.49, 1), (0.5, 0.5, 1)]
    track[17:20] = [(0.5, 0.7, 1), (0.5, 0.8, 1), (0.5, 0.9, 1)]
    resolution = (1280.0, 720.0)
    court = {'H': np.eye(3), 'border_L': 0.0, 'border_R': resolution[0],
             'border_U': 0.0, 'border_D': resolution[1]}
    exclusions = np.zeros(25, dtype=bool)
    exclusions[:10] = True
    return LandingContext(
        track=track, fps=30.0,
        kinematics=LandingKinematics(np.full(25, np.nan), np.full(25, np.nan), compute_speed(track)),
        landing_options=LandingFilterOptions(3, 0.01, 2, 3, 0.5, use_settle=False, use_carry=False),
        net_band=(0.0, 1.0), resolution=resolution, court_info={},
        resolved=resolve(BaseAnnotatorConfig(), 30.0), definitive_exclusion_mask=exclusions,
        shuttle_hallucination_mask=np.zeros(25, dtype=bool), source_codes=None,
        rejection_diagnostics=[], band_m=0.1, landing_horizons_s=(0.1, 0.5), horizon_rows=[],
        scene_courts=(SceneCourt(10, 16, court, (340.0, 380.0), 0.1),
                      SceneCourt(16, 25, court, (340.0, 380.0), 0.1)),
    )


def test_returning_view_keeps_contact_time_and_bounds_the_landing_search():
    context = returning_view_context()
    outcome = build_rally_outcome(0, Half.TOP, None, 25, [3], context, rally_end=20)
    assert outcome.landing is not None
    assert outcome.landing.frame == 13
    assert outcome.landing.norm == pytest.approx((0.5, 0.5))
    # The observed tail ends at the net, but the unseen flight may have crossed it.
    assert outcome.landing.net_ender is False
    assert outcome.verdict.striker_half is Half.TOP
    short, long = context.horizon_rows
    assert short.final_contact_frame == long.final_contact_frame == 3
    assert short.requested_end_frame == 6
    assert short.capped_landing is None
    assert long.safe_end_frame == 16
    assert long.capped_landing == outcome.landing


@pytest.mark.parametrize('rally_end', [10, 12])
def test_recovery_does_not_borrow_observations_beyond_this_rally(rally_end):
    context = returning_view_context()
    outcome = build_rally_outcome(0, Half.TOP, Half.BOT, 25, [3], context, rally_end=rally_end)
    assert outcome.landing is None
    assert outcome.verdict.verdict is Verdict.LOST
    assert outcome.verdict.verdict_source is VerdictSource.NEXT_SERVER


def test_rejected_final_contact_does_not_borrow_an_earlier_players_flight():
    context = returning_view_context()
    context.shuttle_hallucination_mask[12] = True
    outcome = build_rally_outcome(0, Half.TOP, Half.TOP, 25, [3, 12], context, rally_end=20)
    assert outcome.landing is None
    assert outcome.verdict.striker_half is Half.TOP
    assert outcome.verdict.verdict is Verdict.WON
    assert context.rejection_diagnostics[0]['rule'] == 'final_contact'


def test_recovery_respects_masked_views_and_missing_scenes():
    for masked_view in (False, True):
        context = returning_view_context()
        if masked_view:
            context.definitive_exclusion_mask[10] = True
        else:
            context = replace(context, scene_courts=())
        outcome = build_rally_outcome(0, Half.TOP, None, 25, [3], context, rally_end=20)
        assert outcome.landing is None
        assert outcome.verdict.verdict is None


@pytest.mark.parametrize(('hitter', 'position'), [
    (Half.TOP, (0.5, -0.1)), (Half.BOT, (0.5, 1.1)),
    (Half.TOP, (-0.1, 0.25)), (Half.BOT, (1.1, 0.75)),
])
def test_out_landing_on_hitters_half_withholds_geometry_but_keeps_next_server(hitter, position):
    landing = Landing(20, position, hitter, False, False)
    for best_guess in (False, True):
        verdict, winner, _ = geometric_verdict(hitter, landing, best_guess=best_guess)
        assert verdict is None and winner is None
    assert rally_verdict(0, hitter, None, landing, 0.1).verdict is None
    next_serve = rally_verdict(0, hitter, hitter, landing, 0.1)
    assert next_serve.striker_half is hitter
    assert next_serve.verdict is Verdict.WON
    assert next_serve.verdict_source is VerdictSource.NEXT_SERVER
