"""Missing views can hide a flight without changing its contact time or hitter."""

from types import SimpleNamespace

import numpy as np
import pytest

from annotator.config import BaseAnnotatorConfig, resolve
from annotator.courts.scenes import SceneCourt
from annotator.outcomes import point_winner
from annotator.outcomes.point_winner import (
    Half,
    Landing,
    LandingFilterOptions,
    Verdict,
    VerdictSource,
    geometric_verdict,
    rally_verdict,
)
from annotator.outcomes.video import build_verdict_data
from annotator.sequence.contacts import ContactEvent, ContactSequence, FinishedRally


def returning_view_evidence() -> SimpleNamespace:
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
    return SimpleNamespace(
        track=track, fps=30.0,
        sticky=SimpleNamespace(picks=np.full((25, 2), -1), bbox_height=np.full((25, 2), np.nan)),
        pose_kps=np.empty((25, 0, 17, 2)),
        resolution=resolution,
        resolved=resolve(BaseAnnotatorConfig(), 30.0), exclusion_mask=exclusions,
        shuttle_hallucination_mask=np.zeros(25, dtype=bool), source_codes=None,
        scene_courts=(SceneCourt(10, 16, court, (340.0, 380.0), 0.1),
                      SceneCourt(16, 25, court, (340.0, 380.0), 0.1)),
    )


def resolve_outcome(evidence, frames, *, rally_end=20, next_server=None, horizons=None, rejections=None):
    events = tuple(ContactEvent(frame, np.nan, Half.TOP) for frame in frames)
    rallies = [FinishedRally(ContactSequence(0, 0, rally_end, events), Half.TOP)]
    if next_server is not None:
        rallies.append(FinishedRally(
            ContactSequence(1, 24, 25, (ContactEvent(24, np.nan, next_server),)), next_server,
        ))
    return build_verdict_data(
        evidence, rallies,
        landing_options=LandingFilterOptions(3, 0.01, 2, 3, 0.5, use_settle=False, use_carry=False),
        rejection_diagnostics=rejections,
        landing_horizons_s=(0.1, 0.5) if horizons is not None else (), horizon_rows=horizons,
    )


def test_returning_view_keeps_contact_time_and_bounds_the_landing_search():
    evidence = returning_view_evidence()
    horizons = []
    outcome = resolve_outcome(evidence, [3], horizons=horizons)
    assert outcome.landings[0] is not None
    assert outcome.landings[0].frame == 13
    assert outcome.landings[0].norm == pytest.approx((0.5, 0.5))
    # The observed tail ends at the net, but the unseen flight may have crossed it.
    assert outcome.landings[0].net_ender is False
    assert outcome.verdict_rows[0].striker_half is Half.TOP
    short, long = horizons
    assert short.final_contact_frame == long.final_contact_frame == 3
    assert short.requested_end_frame == 6
    assert short.capped_landing is None
    assert long.safe_end_frame == 16
    assert long.capped_landing == outcome.landings[0]


@pytest.mark.parametrize('rally_end', [10, 12])
def test_recovery_does_not_borrow_observations_beyond_this_rally(rally_end):
    evidence = returning_view_evidence()
    outcome = resolve_outcome(evidence, [3], next_server=Half.BOT, rally_end=rally_end)
    assert outcome.landings[0] is None
    assert outcome.verdict_rows[0].verdict is Verdict.LOST
    assert outcome.verdict_rows[0].verdict_source is VerdictSource.NEXT_SERVER


def test_rejected_final_contact_does_not_borrow_an_earlier_players_flight():
    evidence = returning_view_evidence()
    evidence.shuttle_hallucination_mask[12] = True
    rejections = []
    outcome = resolve_outcome(evidence, [3, 12], next_server=Half.TOP, rejections=rejections)
    assert outcome.landings[0] is None
    assert outcome.verdict_rows[0].striker_half is Half.TOP
    assert outcome.verdict_rows[0].verdict is Verdict.WON
    assert rejections[0]['rule'] == 'final_contact'


def test_recovery_respects_masked_views_and_missing_scenes():
    for masked_view in (False, True):
        evidence = returning_view_evidence()
        if masked_view:
            evidence.exclusion_mask[10] = True
        else:
            evidence.scene_courts = ()
        outcome = resolve_outcome(evidence, [3])
        assert outcome.landings[0] is None
        assert outcome.verdict_rows[0].verdict is None


def test_returning_view_suppresses_cut_speed_without_changing_shared_kinematics(monkeypatch):
    evidence = returning_view_evidence()
    evidence.track[9] = (0.2, 0.2, 1)
    kinematics = point_winner.build_landing_kinematics(
        evidence.track, evidence.sticky, evidence.pose_kps, evidence.resolution,
    )
    cut_speed = kinematics.speed[10]
    assert np.isfinite(cut_speed)
    monkeypatch.setattr(point_winner, 'build_landing_kinematics', lambda *args: kinematics)
    search_speeds = []
    pick_landing = point_winner.pick_landing_to_end

    def observe_search(start, end, track, search_kinematics, *args, **kwargs):
        search_speeds.append(search_kinematics.speed[10])
        return pick_landing(start, end, track, search_kinematics, *args, **kwargs)

    monkeypatch.setattr(point_winner, 'pick_landing_to_end', observe_search)
    resolve_outcome(evidence, [3], horizons=[])

    assert search_speeds and all(np.isnan(speed) for speed in search_speeds)
    assert kinematics.speed[10] == cut_speed


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
