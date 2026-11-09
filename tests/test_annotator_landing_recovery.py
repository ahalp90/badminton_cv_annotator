"""Missing views can hide a flight without changing its contact time or hitter."""

from types import SimpleNamespace

import numpy as np
import pytest

from annotator.config import BaseAnnotatorConfig, scale_config_to_fps
from annotator.courts.frame_geometry import SceneCourt
from annotator.rally_results import landing_and_winner as point_winner
from annotator.rally_results.estimate_results import build_outcomes
from annotator.rally_results.landing_and_winner import (
    LandingFilterOptions,
    WinnerSource,
)
from annotator.types import ContactEvent, ContactSequence, CourtHalf, FinishedRally
from experiments.annotator.landing_search import compare_landing_searches


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
        players=SimpleNamespace(detection_indices=np.full((25, 2), -1), box_heights_px=np.full((25, 2), np.nan)),
        pose_kps=np.empty((25, 0, 17, 2)),
        resolution=resolution,
        video_config=scale_config_to_fps(BaseAnnotatorConfig(), 30.0), exclusion_mask=exclusions,
        rejected_shuttle_frames=np.zeros(25, dtype=bool),
        scene_courts=(SceneCourt(10, 16, court, (340.0, 380.0), 0.1),
                      SceneCourt(16, 25, court, (340.0, 380.0), 0.1)),
    )


def resolve_outcome(evidence, frames, *, rally_end=20, next_server=None, search_trials=None):
    events = tuple(ContactEvent(frame, np.nan, CourtHalf.TOP) for frame in frames)
    rallies = [FinishedRally(ContactSequence(0, 0, rally_end, events), CourtHalf.TOP)]
    if next_server is not None:
        rallies.append(FinishedRally(
            ContactSequence(1, 24, 25, (ContactEvent(24, np.nan, next_server),)), next_server,
        ))
    outcomes = build_outcomes(
        evidence, rallies,
        landing_options=LandingFilterOptions(3, 0.01, 1000, 3, 0.5),
    )
    if search_trials is not None:
        search_trials.extend(compare_landing_searches(
            evidence, rallies, outcomes, (0.1, 0.5), LandingFilterOptions(3, 0.01, 1000, 3, 0.5),
        ))
    return outcomes


def test_returning_view_keeps_contact_time_and_bounds_the_landing_search():
    evidence = returning_view_evidence()
    search_trials = []
    outcome = resolve_outcome(evidence, [3], search_trials=search_trials)
    assert outcome[0].landing is not None
    assert outcome[0].landing.frame == 13
    assert outcome[0].landing.court_xy == pytest.approx((0.5, 0.5))
    # The observed tail ends at the net, but the unseen flight may have crossed it.
    assert outcome[0].landing.net_ender is False
    short, long = search_trials
    assert short.final_contact_frame == long.final_contact_frame == 3
    assert short.time_limit_end_frame == 6
    assert short.trial_outcome.landing is None
    assert long.normal_end_frame == 16
    assert long.trial_outcome.landing == outcome[0].landing


@pytest.mark.parametrize('rally_end', [10, 12])
def test_recovery_does_not_borrow_observations_beyond_this_rally(rally_end):
    evidence = returning_view_evidence()
    outcome = resolve_outcome(evidence, [3], next_server=CourtHalf.BOT, rally_end=rally_end)
    assert outcome[0].landing is None
    assert outcome[0].winner is CourtHalf.BOT
    assert outcome[0].winner_source is WinnerSource.NEXT_SERVER


def test_rejected_final_contact_does_not_borrow_an_earlier_players_flight():
    evidence = returning_view_evidence()
    evidence.rejected_shuttle_frames[12] = True
    outcome = resolve_outcome(evidence, [3, 12], next_server=CourtHalf.TOP)
    assert outcome[0].landing is None
    assert outcome[0].winner is CourtHalf.TOP


def test_recovery_respects_masked_views_and_missing_scenes():
    for masked_view in (False, True):
        evidence = returning_view_evidence()
        if masked_view:
            evidence.exclusion_mask[10] = True
        else:
            evidence.scene_courts = ()
        outcome = resolve_outcome(evidence, [3])
        assert outcome[0].landing is None
        assert outcome[0].winner is None


def test_returning_view_suppresses_cut_speed_without_changing_shared_kinematics(monkeypatch):
    evidence = returning_view_evidence()
    evidence.track[9] = (0.2, 0.2, 1)
    kinematics = point_winner.build_landing_kinematics(
        evidence.track, evidence.players, evidence.pose_kps, evidence.resolution,
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
    resolve_outcome(evidence, [3], search_trials=[])

    assert search_speeds and all(np.isnan(speed) for speed in search_speeds)
    assert kinematics.speed[10] == cut_speed


def test_landing_trials_skip_rallies_without_contacts():
    evidence = returning_view_evidence()
    empty = FinishedRally(ContactSequence(0, 0, 20, ()), None)
    assert compare_landing_searches(
        evidence, (empty,), {}, (1.0,), LandingFilterOptions(3, 0.01, 1000, 3, 0.5),
    ) == []
