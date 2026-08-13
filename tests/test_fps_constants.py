"""FPS-relativity regression tests for the scraper's base-30 public table."""
from __future__ import annotations

from dataclasses import asdict, fields, replace
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest

from annotator.config import SHIPPED_THRESHOLDS, BaseAnnotatorConfig
from annotator.fps_constants import (
    FPS_CONSTANT_FIELD_NAMES,
    FpsConstants,
    ScalingKind,
    probe_fps,
    scale_for_fps,
)
from annotator.point_winner import (
    Half,
    LandingFilterOptions,
    LandingKinematics,
    attribute_half,
    convert_landing_options,
    landing_window,
    pick_landing_to_end,
    window_end,
)
from annotator.rally_segmentation import (
    build_sticky_result,
    scale_thresholds,
    segment_video,
)
from annotator.replay_mask import combine_mask, court_absence_signal
from annotator.resolve import resolve


def test_scale_for_fps_has_base_30_identity_for_every_scaled_row() -> None:
    values = scale_for_fps(30.0)
    assert values.rest_speed == 0.002
    # float64 evaluates 0.015 * 30.0 / 30.0 one ulp below the source literal.
    assert values.start_speed == 0.014999999999999998
    assert (
        values.rest_window, values.start_min_frames, values.smooth_window,
        values.end_rest_frames, values.court_absent_window, values.replay_mask_min_frames,
        values.impulse_floor_half_window_frames, values.contact_dedup_radius_frames,
        values.contact_suppression_radius_frames, values.serve_start_lookback_frames,
        values.serve_stillness_window_frames,
        values.sustained_loss_frames,
        values.min_descend_samples, values.body_unit_half_window,
        values.composition_min_scene_len,
    ) == (5, 3, 3, 90, 15, 15, 12, 3, 9, 25, 15, 10, 3, 12, 15)

    values25 = scale_for_fps(25.0)
    assert values25.rest_speed == 0.0024
    assert values25.start_speed == 0.018
    assert (
        values25.rest_window, values25.start_min_frames, values25.smooth_window,
        values25.end_rest_frames, values25.court_absent_window, values25.replay_mask_min_frames,
        values25.impulse_floor_half_window_frames, values25.contact_dedup_radius_frames,
        values25.contact_suppression_radius_frames, values25.serve_start_lookback_frames,
        values25.serve_stillness_window_frames,
        values25.sustained_loss_frames,
        values25.min_descend_samples, values25.body_unit_half_window,
        values25.composition_min_scene_len,
    ) == (4, 3, 3, 75, 13, 13, 10, 3, 8, 21, 13, 8, 3, 10, 13)


def test_scale_for_fps_half_up_spots_and_floor_one() -> None:
    values50 = scale_for_fps(50.0)
    values60 = scale_for_fps(60.0)
    assert values50.impulse_floor_half_window_frames == 20
    assert values60.contact_dedup_radius_frames == 6
    assert values60.contact_suppression_radius_frames == 18
    assert values60.composition_min_scene_len == 30
    assert values60.court_absent_window == 30
    assert values60.replay_mask_min_frames == 30
    assert scale_for_fps(25.0).contact_suppression_radius_frames == 8
    assert scale_for_fps(25.0).court_absent_window == 13
    assert scale_for_fps(25.0).replay_mask_min_frames == 13
    assert scale_for_fps(25.0).composition_min_scene_len == 13
    assert scale_for_fps(1.0).start_min_frames == 1


def test_resolution_scales_body_unit_window() -> None:
    base = BaseAnnotatorConfig()
    assert resolve(base, 25.0).constants.body_unit_half_window == 10
    assert resolve(base, 50.0).constants.body_unit_half_window == 20


def test_scale_for_fps_composition_scene_length_is_distinct_but_currently_equal() -> None:
    values60 = scale_for_fps(60.0)
    values25 = scale_for_fps(25.0)
    assert values60.composition_min_scene_len == values60.court_absent_window
    assert values25.composition_min_scene_len == values25.court_absent_window


def test_replay_mask_min_frames_is_distinct_and_overridable() -> None:
    values = scale_for_fps(30.0, {'court_absent_window': 20, 'replay_mask_min_frames': 7})
    assert values.court_absent_window == 20
    assert values.replay_mask_min_frames == 7


def test_scale_for_fps_visible_sample_rows_floor_at_two_frames() -> None:
    values10 = scale_for_fps(10.0)
    assert values10.high_shot_oob_min_visible_frames == 2
    assert values10.reentry_min_visible_frames == 2
    assert scale_for_fps(25.0).high_shot_oob_min_visible_frames == 2
    assert scale_for_fps(60.0).high_shot_oob_min_visible_frames == 5
    assert scale_for_fps(25.0).reentry_min_visible_frames == 2
    assert scale_for_fps(60.0).reentry_min_visible_frames == 5


@pytest.mark.parametrize('fps', (23.976, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0))
def test_every_fps_field_override_uses_its_shared_scaling_rule(fps: float) -> None:
    overrides = {
        field.name: float(index) + 0.25
        for index, field in enumerate(fields(FpsConstants), start=1)
    }
    constants = scale_for_fps(fps, overrides)
    speed_fields = {'rest_speed', 'start_speed'}
    minimum_two_fields = {
        'high_shot_oob_min_visible_frames',
        'reentry_min_visible_frames',
    }
    for field_name, base30_value in overrides.items():
        scaling = (
            ScalingKind.PER_FRAME_SPEED
            if field_name in speed_fields
            else ScalingKind.FRAME_COUNT
        )
        expected = scaling.scale(base30_value, fps)
        if field_name in minimum_two_fields:
            expected = max(2, expected)
        assert getattr(constants, field_name) == expected


def test_resolve_accepts_every_fps_field_and_explicit_contact_threshold() -> None:
    overrides = {
        field.name: float(index) + 0.25
        for index, field in enumerate(fields(FpsConstants), start=1)
    }
    overrides['contact_impulse_multiple'] = 5.5

    resolved = resolve(BaseAnnotatorConfig(overrides_base30=overrides), 30.0)

    assert FPS_CONSTANT_FIELD_NAMES == frozenset(field.name for field in fields(FpsConstants))
    assert asdict(resolved.constants) == asdict(scale_for_fps(30.0, overrides))
    assert resolved.thresholds.contact_impulse_multiple == 5.5


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


def test_stage8_scaled_preset_changes_segmentation_at_50fps() -> None:
    track = np.zeros((300, 3), dtype=float)
    track[:, 2] = 1
    track[20:40, 0] = np.arange(20) * 0.01
    track[40:, 0] = track[39, 0]
    unaware, _ = segment_video(track, thresholds=SHIPPED_THRESHOLDS)
    aware, _ = segment_video(track, thresholds=scale_thresholds(SHIPPED_THRESHOLDS, 50.0))
    assert unaware == []
    assert aware == [(21, 41)]


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
    options = LandingFilterOptions(
        7,
        0.004,
        5,
        7,
        0.75,
        use_settle=False,
        use_carry=True,
        null_if_all_carried=True,
        use_ankle_rule=False,
    )
    expected = options._replace(
        settle_win=int(ScalingKind.FRAME_COUNT.scale(options.settle_win, fps)),
        settle_thr=float(ScalingKind.PER_FRAME_SPEED.scale(options.settle_thr, fps)),
        settle_min=int(ScalingKind.FRAME_COUNT.scale(options.settle_min, fps)),
        carry_win=int(ScalingKind.FRAME_COUNT.scale(options.carry_win, fps)),
    )

    assert convert_landing_options(options, fps) == expected


def test_resolved_60fps_seam_drives_replay_segmentation_attribution_and_landing(
) -> None:
    """One resolved config crosses every promoted FPS-sensitive boundary exactly once."""
    base = BaseAnnotatorConfig()
    resolved = resolve(base, 60.0)
    assert resolved.constants.court_absent_window == 30
    assert resolved.constants.sustained_loss_frames == 20
    assert resolved.constants.min_descend_samples == 6
    assert resolved.constants.body_unit_half_window == 24
    # 29 is below the correct 30-frame replay threshold; 40 brackets correct 30 and double 60.
    present = np.ones(500, dtype=bool)
    present[10:39] = False
    present[60:100] = False
    replay = combine_mask(present, None, None, None, len(present), resolved.fps)
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
    replay = combine_mask(present, None, None, None, n_frames, resolved.fps)
    plain_spans, _ = segment_video(track, thresholds=resolved.thresholds)
    masked_spans, _ = segment_video(track, thresholds=resolved.thresholds, exclusion_mask=replay)
    assert plain_spans[0][0] == 45
    assert masked_spans[0][0] == 75

    # This is the smallest real sticky-gate context: one in-court standing pose and identity
    # camera-to-court mapping. The short box-height observations sit outside the base-12
    # window around contact 82, but inside the resolved-24 window.
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
    resolution_table = pd.DataFrame(
        {'width': [1920.0], 'height': [1080.0]}, index=['v'],
    )
    short_sticky = build_sticky_result(
        track.copy(), [(0, n_frames)], bboxes.copy(), scores.copy(), kps.copy(), ndet.copy(), 'v',
        {'v': court_info.copy()},
        resolution_table, (1920.0, 1080.0), 12,
    )
    full_sticky = build_sticky_result(
        track.copy(), [(0, n_frames)], bboxes.copy(), scores.copy(), kps.copy(), ndet.copy(), 'v',
        {'v': court_info.copy()}, resolution_table, (1920.0, 1080.0),
        resolved.constants.body_unit_half_window,
    )
    base_radius_thresholds = resolved.thresholds._replace(contact_suppression_radius_frames=9)
    base_radius_contacts = segment_video(
        track, thresholds=base_radius_thresholds, sticky_distances=short_sticky.distances,
    )[1]
    resolved_contacts = segment_video(
        track, thresholds=resolved.thresholds, sticky_distances=short_sticky.distances,
    )[1]
    assert {
        contact.contact_frame for contact in base_radius_contacts
        if contact.wrist_near is not False and contact.suppressed is not True
    } >= {73, 82}
    assert sum(
        contact.wrist_near is not False and contact.suppressed is not True
        for contact in resolved_contacts if contact.contact_frame in (73, 82)
    ) == 1

    short_window_contacts = base_radius_contacts
    full_contacts = segment_video(
        track, thresholds=base_radius_thresholds, sticky_distances=full_sticky.distances,
    )[1]
    short_contact = next(contact for contact in short_window_contacts if contact.contact_frame == 82)
    full_contact = next(contact for contact in full_contacts if contact.contact_frame == 82)
    assert (short_contact.wrist_near, short_contact.suppressed) == (True, False)
    assert (full_contact.wrist_near, full_contact.suppressed) == (False, False)

    # The final resolved contact uses the same track, sticky cache, and resolved constants for
    # attribution and landing.
    resolved_full_contacts = segment_video(
        track, thresholds=resolved.thresholds, sticky_distances=full_sticky.distances,
    )[1]
    final_contact = [
        contact.contact_frame for contact in resolved_full_contacts
        if contact.wrist_near is not False and contact.suppressed is not True
    ][-1]
    sticky = full_sticky
    assert attribute_half(final_contact, track, sticky, bboxes, (520.0, 560.0)) is Half.TOP
    # A 15-frame loss is longer than the unscaled base-30 10 but shorter than the resolved 20.
    # It exposes two post-contact descents: five samples (base 3 accepts; resolved 6 rejects),
    # then eight samples (resolved 6 accepts; double-scaled 12 rejects).
    landing_track = track.copy()
    landing_track[final_contact:135, 1] = 0.20
    landing_track[135:140, 1] = np.linspace(0.20, 0.24, 5)
    landing_track[140:143, 1] = (0.20, 0.80, 0.80)
    landing_track[143:158, 2] = 0
    landing_track[158:166, 1] = np.linspace(0.20, 0.27, 8)
    landing_track[166:, 1] = 0.20
    assert window_end(final_contact, n_frames, landing_track, np.zeros(n_frames, dtype=bool), 20) > 158
    assert window_end(final_contact, n_frames, landing_track, np.zeros(n_frames, dtype=bool), 10) == 143

    kin = LandingKinematics(np.full(n_frames, np.nan), np.full(n_frames, np.nan), np.zeros(n_frames))
    opts = LandingFilterOptions(1, 0.0, 1, 1, 0.0, use_settle=False, use_carry=False)
    dead = np.zeros(n_frames, dtype=bool)

    def pick_with(constants):
        end_frame = landing_window(
            final_contact, n_frames, landing_track, dead, constants.sustained_loss_frames,
        ).end_frame
        return pick_landing_to_end(
            final_contact, end_frame, landing_track, kin, opts, Half.TOP,
            (520.0, 560.0), (1920.0, 1080.0), court_info, constants, resolved.fps,
        )

    unscaled_landing = pick_with(
        replace(resolved.constants, sustained_loss_frames=10, min_descend_samples=3),
    )
    landing = pick_with(resolved.constants)
    double_scaled_landing = pick_with(replace(resolved.constants, min_descend_samples=12))
    assert unscaled_landing is not None
    assert unscaled_landing.frame == 139
    assert landing is not None
    assert landing.frame == 165
    assert double_scaled_landing is None
