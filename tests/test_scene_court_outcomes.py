"""Downstream outcome consumers use the court geometry of each contact scene."""

from types import SimpleNamespace

import numpy as np
import pytest

from annotator.config import BaseAnnotatorConfig, scale_config_to_fps
from annotator.contacts.build_features import contact_feature_dtype
from annotator.contacts.score_candidates import ScoredContacts, contact_feature_matrix
from annotator.courts.frame_geometry import SceneCourt
from annotator.inference.heuristic import select_heuristic_contacts
from annotator.rallies.build_contact_proposals import build_rally_context
from annotator.rally_results import estimate_results as video_outcomes
from annotator.rally_results.landing_and_winner import (
    Landing,
    LandingKinematics,
    RallyOutcome,
)
from annotator.types import (
    ContactCandidate,
    ContactEvent,
    ContactSequence,
    CourtHalf,
    FinishedRally,
    TrackedPlayers,
    WristGaps,
)


def _scene(
    start_frame: int,
    end_frame: int,
    name: str,
    net_band: tuple[float, float],
    error_band: float,
) -> SceneCourt:
    return SceneCourt(start_frame, end_frame, {'scene': name}, net_band, error_band)


@pytest.mark.parametrize('second_band, expected', [
    ((30.0, 31.0), (CourtHalf.BOT, CourtHalf.TOP)),
    ((10.0, 11.0), (CourtHalf.BOT, CourtHalf.BOT)),
])
def test_both_contact_paths_use_each_scene_net_band(second_band, expected):
    scenes = (
        _scene(0, 5, 'first', (10.0, 11.0), 0.1),
        _scene(5, 10, 'second', second_band, 0.2),
    )
    contacts = [ContactCandidate(rally_id, frame, None, None) for rally_id, frame in enumerate((2, 7))]
    track = np.zeros((10, 3))
    track[:, 2] = 1
    bboxes = np.zeros((10, 1, 4))
    bboxes[:, 0, 3] = 20
    players = TrackedPlayers(
        np.zeros((10, 2), dtype=int), np.zeros((10, 2, 2)), np.zeros((10, 2)), np.full(10, 2.0),
    )
    wrist_gaps = WristGaps(np.ones(10), np.ones((10, 2)))
    evidence = SimpleNamespace(
        heuristic_spans=[(0, 5), (5, 10)], heuristic_contacts=contacts, exclusion_mask=np.zeros(10, dtype=bool),
        track=track, players=players, wrist_gaps=wrist_gaps, bboxes=bboxes, scene_courts=scenes, fps=30.0, bounds_supplied=True,
    )
    rallies = select_heuristic_contacts(evidence)
    assert tuple(event.side for rally in rallies for event in rally.sequence.events) == expected
    rows = np.zeros(2, dtype=contact_feature_dtype())
    rows['frame'] = (2, 7)
    scores = ScoredContacts(rows, contact_feature_matrix(rows), np.ones(2), np.arange(2))
    context = build_rally_context(evidence, scores)
    assert tuple(event.side for event in context.events) == expected


def test_hit_height_uses_net_band_at_contact_scene(monkeypatch):
    scenes = (
        _scene(0, 5, 'first', (10.0, 11.0), 0.1),
        _scene(5, 10, 'second', (30.0, 31.0), 0.2),
    )
    received_bands = []

    def fake_hit_height(track, contact_frame, net_band, resolution):
        received_bands.append(net_band)
        return 1

    monkeypatch.setattr(
        video_outcomes.landing_and_winner, 'hit_height', fake_hit_height,
    )
    evidence = SimpleNamespace(track=np.zeros((10, 3)), resolution=(1280.0, 720.0), scene_courts=scenes)
    contacts = (FinishedRally(ContactSequence(0, 0, 10, (ContactEvent(7, np.nan, CourtHalf.TOP),)), CourtHalf.TOP),)
    hit_heights, failures = video_outcomes.build_hit_heights(evidence, contacts)

    assert received_bands == [scenes[1].net_band]
    assert hit_heights == {7: 1}
    assert failures == {}


def test_landing_uses_final_contact_scene_geometry_and_clamps_at_scene_end(monkeypatch):
    scenes = (
        _scene(0, 5, 'first', (10.0, 11.0), 0.1),
        _scene(5, 10, 'second', (30.0, 31.0), 0.37),
    )
    n_frames = 20
    track = np.tile(np.array([0.5, 0.5, 1.0]), (n_frames, 1))
    landing = Landing(8, (0.5, 0.75), CourtHalf.BOT, False, False)
    landing_call = {}
    error_bands = []

    def fake_kinematics(*args):
        return LandingKinematics(
            np.full(n_frames, np.nan), np.full(n_frames, np.nan), np.zeros(n_frames),
        )

    def fake_pick(
        final_contact, end_frame, track_arg, kinematics, options, striker, net_band,
        resolution, court_info, constants, **kwargs,
    ):
        landing_call.update(
            final_contact=final_contact, end_frame=end_frame, net_band=net_band,
            court_info=court_info,
        )
        return landing

    def fake_verdict(striker, next_server, landing_arg, band_m, **kwargs):
        error_bands.append(band_m)
        return RallyOutcome(None, None, landing_arg, None, False, False, None, False)

    monkeypatch.setattr(video_outcomes.landing_and_winner, 'build_landing_kinematics', fake_kinematics)
    monkeypatch.setattr(video_outcomes.landing_and_winner, 'pick_landing_to_end', fake_pick)
    monkeypatch.setattr(video_outcomes.landing_and_winner, 'resolve_rally_outcome', fake_verdict)

    contact_data = (FinishedRally(ContactSequence(0, 0, 12, (ContactEvent(7, np.nan, CourtHalf.TOP),)), CourtHalf.TOP),)
    evidence = SimpleNamespace(
        track=track, fps=25.0, exclusion_mask=np.zeros(n_frames, dtype=bool), players=None,
        video_config=scale_config_to_fps(BaseAnnotatorConfig(), 25.0), pose_kps=None, resolution=(1280.0, 720.0),
        net_band=(1.0, 2.0), rejected_shuttle_frames=np.zeros(n_frames, dtype=bool),
        scene_courts=scenes,
    )
    result = video_outcomes.build_outcomes(
        evidence, contact_data, landing_options=None,
    )

    assert landing_call == {
        'final_contact': 7,
        'end_frame': scenes[1].end_frame,
        'net_band': scenes[1].net_band,
        'court_info': scenes[1].court_info,
    }
    assert error_bands == [scenes[1].landing_error_band_m]
    assert result[0].landing == landing
