"""Downstream outcome consumers use the court geometry of each contact scene."""

import numpy as np

from annotator import video_outcomes
from annotator.config import BaseAnnotatorConfig
from annotator.point_winner import (
    Half,
    HitHeightRow,
    Landing,
    LandingKinematics,
    VerdictRow,
)
from annotator.resolve import resolve
from annotator.scene_courts import SceneCourt
from annotator.types import ContactCandidate


def _scene(
    start_frame: int,
    end_frame: int,
    name: str,
    net_band: tuple[float, float],
    error_band: float,
) -> SceneCourt:
    return SceneCourt(start_frame, end_frame, {'scene': name}, net_band, error_band)


def test_contact_attribution_uses_net_band_at_contact_scene(monkeypatch):
    scenes = (
        _scene(0, 5, 'first', (10.0, 11.0), 0.1),
        _scene(5, 10, 'second', (30.0, 31.0), 0.2),
    )
    received_bands = []

    def fake_attribute_half(frame, track, sticky, bboxes, net_band):
        received_bands.append(net_band)
        return Half.TOP

    monkeypatch.setattr(video_outcomes.point_winner, 'attribute_half', fake_attribute_half)
    contacts = [ContactCandidate(0, 7, None, None, None)]
    result = video_outcomes.build_contact_data(
        [(0, 10)], contacts, np.zeros(10, dtype=bool), np.zeros((10, 3)), None,
        np.zeros((10, 1, 4)), (1.0, 2.0), scene_courts=scenes,
    )

    assert received_bands == [scenes[1].net_band]
    assert result.striker_halves == [Half.TOP]


def test_hit_height_uses_net_band_at_contact_scene(monkeypatch):
    scenes = (
        _scene(0, 5, 'first', (10.0, 11.0), 0.1),
        _scene(5, 10, 'second', (30.0, 31.0), 0.2),
    )
    received_bands = []

    def fake_hit_height_rows(contacts, track, net_band, resolution):
        received_bands.append(net_band)
        return [HitHeightRow(0, 0, 7, 1)]

    monkeypatch.setattr(
        video_outcomes.point_winner, 'build_hit_height_rows', fake_hit_height_rows,
    )
    hit_heights, failures = video_outcomes.build_hit_heights(
        [(0, 10)], {0: [7]}, np.zeros((10, 3)), (1.0, 2.0), (1280.0, 720.0), scenes,
    )

    assert received_bands == [scenes[1].net_band]
    assert hit_heights == {7: 1}
    assert failures == []


def test_landing_uses_final_contact_scene_geometry_and_clamps_at_scene_end(monkeypatch):
    scenes = (
        _scene(0, 5, 'first', (10.0, 11.0), 0.1),
        _scene(5, 10, 'second', (30.0, 31.0), 0.37),
    )
    n_frames = 20
    track = np.tile(np.array([0.5, 0.5, 1.0]), (n_frames, 1))
    landing = Landing(8, (0.5, 0.75), Half.BOT, False, False)
    landing_call = {}
    error_bands = []

    def fake_kinematics(*args):
        return LandingKinematics(
            np.full(n_frames, np.nan), np.full(n_frames, np.nan), np.zeros(n_frames),
        )

    def fake_pick(
        final_contact, end_frame, track_arg, kinematics, options, striker, net_band,
        resolution, court_info, constants, fps, **kwargs,
    ):
        landing_call.update(
            final_contact=final_contact, end_frame=end_frame, net_band=net_band,
            court_info=court_info,
        )
        return landing

    def fake_verdict(rally_id, striker, next_server, landing_arg, band_m):
        error_bands.append(band_m)
        return VerdictRow(rally_id, striker, None, None, None, False, False)

    monkeypatch.setattr(video_outcomes.point_winner, 'build_landing_kinematics', fake_kinematics)
    monkeypatch.setattr(video_outcomes.point_winner, 'pick_landing_to_end', fake_pick)
    monkeypatch.setattr(video_outcomes.point_winner, 'rally_verdict', fake_verdict)

    contact_data = video_outcomes.ContactData(
        filtered_contacts=[], scored_by_rally={}, filtered_by_rally={0: [7]},
        striker_halves=[Half.TOP], n_strokes_list=[1], next_servers=[None],
        fitted_first_all=[Half.TOP],
    )
    result = video_outcomes.build_verdict_data(
        track,
        fps=25.0,
        spans=[(0, 12)],
        definitive_exclusion_mask=np.zeros(n_frames, dtype=bool),
        sticky=None,
        contact_data=contact_data,
        resolved=resolve(BaseAnnotatorConfig(), 25.0),
        kps=None,
        resolution=(1280.0, 720.0),
        video_id=None,
        homo_df=None,
        court_info={'scene': 'global'},
        landing_options=None,
        net_band=(1.0, 2.0),
        ref_err_px=3.5,
        landing_error_band_m=99.0,
        shuttle_hallucination_mask=np.zeros(n_frames, dtype=bool),
        source_codes=None,
        rejection_diagnostics=None,
        landing_horizons_s=(),
        horizon_rows=None,
        scene_courts=scenes,
    )

    assert landing_call == {
        'final_contact': 7,
        'end_frame': scenes[1].end_frame,
        'net_band': scenes[1].net_band,
        'court_info': scenes[1].court_info,
    }
    assert error_bands == [scenes[1].landing_error_band_m]
    assert result.landings == {0: landing}
