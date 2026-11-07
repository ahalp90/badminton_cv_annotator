"""FPS-relativity regression tests for the scraper's base-30 public table."""
from __future__ import annotations

import json
import subprocess
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from annotator.config import BaseAnnotatorConfig, Thresholds, scale_config_to_fps
from annotator.contacts.player_attribution import contact_player_evidence
from annotator.courts.frame_geometry import SceneCourt
from annotator.excluded_frames.replay import court_absence_signal
from annotator.inference.prepare_annotation_inputs import build_player_evidence
from annotator.rallies.find_rallies import segment_video
from annotator.rally_results.landing_and_winner import (
    LandingFilterOptions,
    LandingKinematics,
    convert_landing_options,
    landing_window,
    pick_landing_to_end,
)
from annotator.types import CourtHalf
from shared.camera_cuts import MIN_SCENE_FRAMES_AT_30_FPS
from shared.frame_timing import frames_at_fps, speed_at_fps
from shared.video_metadata import probe_fps
from tests.annotator_config import trajectory_config


def test_scaled_thresholds_has_base_30_identity_for_every_scaled_row() -> None:
    values = scale_config_to_fps(BaseAnnotatorConfig(), 30.0).thresholds
    assert values.rest_speed == 0.002
    # float64 evaluates 0.015 * 30.0 / 30.0 one ulp below the source literal.
    assert values.start_speed == 0.014999999999999998
    assert (
        values.rest_window, values.start_min_frames, values.smooth_window,
        values.end_rest_frames, values.replay_mask_min_frames,
        values.impulse_floor_half_window_frames, values.contact_dedup_radius_frames,
        values.contact_suppression_radius_frames,
        values.sustained_loss_frames,
        values.min_descend_samples, values.body_unit_half_window,
    ) == (5, 3, 3, 90, 15, 12, 3, 9, 10, 3, 12)

    values25 = scale_config_to_fps(BaseAnnotatorConfig(), 25.0).thresholds
    assert values25.rest_speed == 0.0024
    assert values25.start_speed == 0.018
    assert (
        values25.rest_window, values25.start_min_frames, values25.smooth_window,
        values25.end_rest_frames, values25.replay_mask_min_frames,
        values25.impulse_floor_half_window_frames, values25.contact_dedup_radius_frames,
        values25.contact_suppression_radius_frames,
        values25.sustained_loss_frames,
        values25.min_descend_samples, values25.body_unit_half_window,
    ) == (4, 3, 3, 75, 13, 10, 3, 8, 8, 3, 10)


def test_scaled_thresholds_half_up_spots_and_floor_one() -> None:
    values50 = scale_config_to_fps(BaseAnnotatorConfig(), 50.0).thresholds
    values60 = scale_config_to_fps(BaseAnnotatorConfig(), 60.0).thresholds
    assert values50.impulse_floor_half_window_frames == 20
    assert values60.contact_dedup_radius_frames == 6
    assert values60.contact_suppression_radius_frames == 18
    assert values60.replay_mask_min_frames == 30
    assert scale_config_to_fps(BaseAnnotatorConfig(), 25.0).thresholds.contact_suppression_radius_frames == 8
    assert scale_config_to_fps(BaseAnnotatorConfig(), 25.0).thresholds.replay_mask_min_frames == 13
    assert frames_at_fps(MIN_SCENE_FRAMES_AT_30_FPS, 25.0) == 13
    assert scale_config_to_fps(BaseAnnotatorConfig(), 1.0).thresholds.start_min_frames == 1


def test_resolution_scales_body_unit_window() -> None:
    base = BaseAnnotatorConfig()
    assert scale_config_to_fps(base, 25.0).thresholds.body_unit_half_window == 10
    assert scale_config_to_fps(base, 50.0).thresholds.body_unit_half_window == 20


@pytest.mark.parametrize('name', ['court_absent_window', 'composition_min_scene_len'])
def test_fixed_replay_and_camera_settings_are_not_heuristic_overrides(name):
    with pytest.raises(ValueError, match='unknown base-30'):
        scale_config_to_fps(BaseAnnotatorConfig(overrides_base30={name: 20}), 30.0)


def test_replay_mask_min_frames_is_distinct_and_overridable() -> None:
    values = scale_config_to_fps(
        BaseAnnotatorConfig(overrides_base30={'replay_mask_min_frames': 7}), 30.0
    ).thresholds
    assert values.replay_mask_min_frames == 7


def test_scaled_thresholds_visible_sample_rows_floor_at_two_frames() -> None:
    values10 = scale_config_to_fps(BaseAnnotatorConfig(), 10.0).thresholds
    assert values10.high_shot_oob_min_visible_frames == 2
    assert values10.reentry_min_visible_frames == 2
    assert scale_config_to_fps(BaseAnnotatorConfig(), 25.0).thresholds.high_shot_oob_min_visible_frames == 2
    assert scale_config_to_fps(BaseAnnotatorConfig(), 60.0).thresholds.high_shot_oob_min_visible_frames == 5
    assert scale_config_to_fps(BaseAnnotatorConfig(), 25.0).thresholds.reentry_min_visible_frames == 2
    assert scale_config_to_fps(BaseAnnotatorConfig(), 60.0).thresholds.reentry_min_visible_frames == 5


@pytest.mark.parametrize('fps', (23.976, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0))
def test_every_fps_field_override_uses_its_shared_scaling_rule(fps: float) -> None:
    overrides = {
        field.name: float(index) + 0.25
        for index, field in enumerate(fields(Thresholds), start=1)
    }
    constants = scale_config_to_fps(BaseAnnotatorConfig(overrides_base30=overrides), fps).thresholds
    speed_fields = {'rest_speed', 'start_speed'}
    dimensionless_fields = {'contact_impulse_multiple', 'wrist_distance_body_heights'}
    minimum_two_fields = {
        'high_shot_oob_min_visible_frames',
        'reentry_min_visible_frames',
    }
    for field_name, base30_value in overrides.items():
        if field_name in dimensionless_fields:
            assert getattr(constants, field_name) == base30_value
            continue
        scaling = (
            speed_at_fps
            if field_name in speed_fields
            else frames_at_fps
        )
        expected = scaling(base30_value, fps)
        if field_name in minimum_two_fields:
            expected = max(2, expected)
        assert getattr(constants, field_name) == expected


def test_probe_fps_rejects_vfr_and_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "video.mp4"
    video.touch()
    stream = {
        "codec_type": "video",
        "nb_frames": "10",
        "nb_read_frames": "10",
        "width": 64,
        "height": 48,
        "r_frame_rate": "25/1",
        "avg_frame_rate": "30/1",
        "start_time": "0",
    }
    payload = {"streams": [stream], "format": {"start_time": "0"}}

    def fake_run(*args: object, **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, json.dumps(payload), "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(ValueError, match='variable frame rate'):
        probe_fps(video)
    stream["r_frame_rate"] = "0/1"
    stream["avg_frame_rate"] = "0/1"
    with pytest.raises(ValueError, match='must be positive'):
        probe_fps(video)


def test_probe_fps_remains_lightweight_when_header_frame_count_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "valid-cfr.mkv"
    video.touch()
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        payload = {"streams": [{"r_frame_rate": "25/1", "avg_frame_rate": "25/1"}]}
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert probe_fps(video) == 25.0
    assert len(commands) == 1
    assert "-count_frames" not in commands[0]
    assert commands[0][commands[0].index("-show_entries") + 1] == "stream=r_frame_rate,avg_frame_rate"


def test_scaled_speed_threshold_changes_segmentation_at_50fps() -> None:
    track = np.zeros((300, 3), dtype=float)
    track[:, 2] = 1
    track[20:40, 0] = np.arange(20) * 0.01
    track[40:, 0] = track[39, 0]
    unaware, _ = segment_video(track, video_config=replace(trajectory_config(), thresholds=trajectory_config().thresholds))
    aware, _ = segment_video(
        track, video_config=replace(trajectory_config(), thresholds=scale_config_to_fps(BaseAnnotatorConfig(), 50.0).thresholds)
    )
    assert unaware == []
    # The initial rest is too short to split the active region.
    assert aware == [(0, 41)]


def test_replay_court_absence_scales_at_50fps() -> None:
    present = np.ones(40, dtype=bool)
    present[5:25] = False
    assert court_absence_signal(present, 40, 25.0).any()
    assert not court_absence_signal(present, 40, 50.0).any()


def test_landing_options_are_converted_once() -> None:
    opts = LandingFilterOptions(7, 0.004, 5, 7, 0.75)
    scaled = convert_landing_options(opts, 50.0)
    assert (scaled.settle_win, scaled.settle_thr, scaled.settle_min, scaled.carry_win, scaled.carry_thr) == (
        12, 0.0024, 8, 12, 0.75,
    )


@pytest.mark.parametrize('fps', (23.976, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0))
def test_landing_options_use_shared_scaling_rules(fps: float) -> None:
    options = LandingFilterOptions(7, 0.004, 10000, 7, 0.75)
    expected = options._replace(
        settle_win=frames_at_fps(options.settle_win, fps),
        settle_thr=speed_at_fps(options.settle_thr, fps),
        settle_min=frames_at_fps(options.settle_min, fps),
        carry_win=frames_at_fps(options.carry_win, fps),
    )

    assert convert_landing_options(options, fps) == expected


def test_scaled_60fps_config_drives_replay_segmentation_attribution_and_landing(
) -> None:
    """Use one 60-fps config for replay, segmentation, player assignment and landing."""
    base = BaseAnnotatorConfig(overrides_base30={'end_rest_frames': 20})
    video_config = scale_config_to_fps(base, 60.0)
    assert video_config.thresholds.sustained_loss_frames == 20
    assert video_config.thresholds.min_descend_samples == 6
    assert video_config.thresholds.body_unit_half_window == 24
    # 29 is below the correct 30-frame replay threshold; 40 brackets correct 30 and double 60.
    present = np.ones(500, dtype=bool)
    present[10:39] = False
    present[60:100] = False
    replay = court_absence_signal(present, len(present), video_config.fps)
    assert not replay[10:39].any()
    assert replay[60:100].all()

    # A real zig-zag produces impulse contacts.  The 30-frame absent run masks frames that would
    # otherwise open the span, proving the produced replay mask is actually consumed.
    n_frames = 240
    y = np.full(n_frames, 0.1)
    value, direction = 0.1, 1.0
    for offset in range(78):
        value += direction * 0.02
        y[45 + offset] = value
        if (offset + 1) % 13 == 0:
            direction *= -1.0
    y[123:] = y[122]
    y[109:118] = np.linspace(y[108] + 0.01, y[108] + 0.09, 9)
    y[118:] = y[117]
    track = np.column_stack([np.full(n_frames, 0.5), y, np.ones(n_frames)])
    present = np.ones(n_frames, dtype=bool)
    present[45:75] = False
    replay = court_absence_signal(present, n_frames, video_config.fps)
    plain_spans, _ = segment_video(track, video_config=replace(trajectory_config(), thresholds=video_config.thresholds))
    masked_spans, _ = segment_video(
        track, exclusion_mask=replay, video_config=replace(trajectory_config(), thresholds=video_config.thresholds)
    )
    assert plain_spans[0][0] == 45
    assert masked_spans[0][0] == 75

    # This is the smallest real sticky-gate context: one in-court standing pose and identity
    # camera-to-court mapping. The short box-height observations sit outside the base-12
    # window around contact 82, but inside the scaled 24-frame window.
    bboxes = np.zeros((n_frames, 1, 4))
    bboxes[:, 0] = (900.0, 250.0, 1020.0, 350.0)
    bboxes[58:70, 0, 1] = 330.0
    bboxes[95:107, 0, 1] = 330.0
    scores = np.ones((n_frames, 1))
    kps = np.zeros((n_frames, 1, 17, 2))
    kps[:, 0, 9, 0] = 1055.0
    kps[:, 0, 10, 0] = 1055.0
    kps[:, 0, 9, 1] = track[:, 1] * 1080.0
    kps[:, 0, 10, 1] = track[:, 1] * 1080.0
    ndet = np.ones(n_frames, dtype=int)
    court_info = {'H': np.eye(3), 'border_L': 0.0, 'border_R': 1920.0,
                  'border_U': 0.0, 'border_D': 1080.0}
    _, short_gaps = build_player_evidence(
        track.copy(),
        [(0, n_frames)],
        bboxes.copy(),
        scores.copy(),
        kps.copy(),
        ndet.copy(),
        (1920.0, 1080.0),
        12,
        scene_courts=(SceneCourt(0, len(track), court_info.copy(), (0.0, 0.0), 0.0),),
    )
    full_players, full_gaps = build_player_evidence(
        track.copy(),
        [(0, n_frames)],
        bboxes.copy(),
        scores.copy(),
        kps.copy(),
        ndet.copy(),
        (1920.0, 1080.0),
        video_config.thresholds.body_unit_half_window,
        scene_courts=(SceneCourt(0, len(track), court_info.copy(), (0.0, 0.0), 0.0),),
    )
    base_radius_thresholds = replace(video_config.thresholds, contact_suppression_radius_frames=9)
    base_radius_contacts = segment_video(
        track,
        nearest_wrist_distances=short_gaps.nearest,
        video_config=replace(trajectory_config(), thresholds=base_radius_thresholds),
    )[1]
    scaled_contacts = segment_video(
        track, nearest_wrist_distances=short_gaps.nearest, video_config=replace(trajectory_config(), thresholds=video_config.thresholds)
    )[1]
    assert {
        contact.contact_frame for contact in base_radius_contacts
        if contact.wrist_near is not False and contact.temporal_proximity_suppressed is not True
    } >= {73, 82}
    assert sum(
        contact.wrist_near is not False and contact.temporal_proximity_suppressed is not True
        for contact in scaled_contacts if contact.contact_frame in (73, 82)
    ) == 1

    short_window_contacts = base_radius_contacts
    full_contacts = segment_video(
        track,
        nearest_wrist_distances=full_gaps.nearest,
        video_config=replace(trajectory_config(), thresholds=base_radius_thresholds),
    )[1]
    short_contact = next(contact for contact in short_window_contacts if contact.contact_frame == 82)
    full_contact = next(contact for contact in full_contacts if contact.contact_frame == 82)
    assert (short_contact.wrist_near, short_contact.temporal_proximity_suppressed) == (True, False)
    assert (full_contact.wrist_near, full_contact.temporal_proximity_suppressed) == (False, False)

    # The final contact with scaled settings uses the same track, sticky cache, and scaled constants for
    # attribution and landing.
    scaled_full_contacts = segment_video(
        track, nearest_wrist_distances=full_gaps.nearest, video_config=replace(trajectory_config(), thresholds=video_config.thresholds)
    )[1]
    final_contact = [
        contact.contact_frame for contact in scaled_full_contacts
        if contact.wrist_near is not False and contact.temporal_proximity_suppressed is not True
    ][-1]
    attribution_scenes = (SceneCourt(0, len(track), {}, (520.0, 560.0), 0.1),)
    side, _ = contact_player_evidence(final_contact, track, full_players, full_gaps, bboxes, attribution_scenes)
    assert side is CourtHalf.TOP
    # A 15-frame loss is longer than the unscaled base-30 10 but shorter than the scaled 20.
    # It exposes two post-contact descents: five samples (base 3 accepts; scaled 6 rejects),
    # then eight samples (scaled 6 accepts; double-scaled 12 rejects).
    landing_track = track.copy()
    landing_track[final_contact:135, 1] = 0.20
    landing_track[135:140, 1] = np.linspace(0.20, 0.24, 5)
    landing_track[140:143, 1] = (0.20, 0.80, 0.80)
    landing_track[143:158, 2] = 0
    landing_track[158:166, 1] = np.linspace(0.20, 0.27, 8)
    landing_track[166:, 1] = 0.20
    assert landing_window(
        final_contact, n_frames, landing_track, np.zeros(n_frames, dtype=bool), 20,
        np.zeros(len(landing_track), dtype=bool),
    ).end_frame > 158
    assert landing_window(
        final_contact, n_frames, landing_track, np.zeros(n_frames, dtype=bool), 10,
        np.zeros(len(landing_track), dtype=bool),
    ).end_frame == 143

    kin = LandingKinematics(np.full(n_frames, np.nan), np.full(n_frames, np.nan), np.zeros(n_frames))
    opts = LandingFilterOptions(1, 0.0, 10000, 1, 0.0)
    dead = np.zeros(n_frames, dtype=bool)

    def pick_with(constants):
        end_frame = landing_window(
            final_contact, n_frames, landing_track, dead, constants.sustained_loss_frames,
            np.zeros(len(landing_track), dtype=bool),
        ).end_frame
        return pick_landing_to_end(
            final_contact, end_frame, landing_track, kin, opts, CourtHalf.TOP, (520.0, 560.0), (1920.0, 1080.0),
            court_info, constants, np.zeros(len(landing_track), dtype=bool),
        )

    unscaled_landing = pick_with(
        replace(video_config.thresholds, sustained_loss_frames=10, min_descend_samples=3),
    )
    landing = pick_with(video_config.thresholds)
    double_scaled_landing = pick_with(replace(video_config.thresholds, min_descend_samples=12))
    assert unscaled_landing is not None
    assert unscaled_landing.frame == 139
    assert landing is not None
    assert landing.frame == 165
    assert double_scaled_landing is None
