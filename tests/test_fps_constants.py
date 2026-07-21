"""FPS-relativity regression tests for the scraper's base-30 public table."""
from __future__ import annotations

from dataclasses import replace
import subprocess

import numpy as np
import pandas as pd
import pytest

from src.annotator.config import SHIPPED_THRESHOLDS
from annotator.config import BaseAnnotatorConfig
from annotator.resolve import resolve
from src.annotator.fps_constants import probe_fps, scale_for_fps
from annotator.point_winner import (
    Half,
    LandingFilterOptions,
    LandingKinematics,
    attribute_half,
    convert_landing_options,
    pick_landing,
    window_end,
)
from annotator.rally_segmentation import scale_thresholds, segment_video
from annotator.replay_mask import combine_mask, court_absence_signal


def test_scale_for_fps_has_base_30_identity_for_every_scaled_row() -> None:
    values = scale_for_fps(30.0)
    assert values.rest_speed == 0.002
    # float64 evaluates 0.015 * 30.0 / 30.0 one ulp below the source literal.
    assert values.start_speed == 0.014999999999999998
    assert (
        values.rest_window, values.start_min_frames, values.smooth_window,
        values.end_rest_frames, values.court_absent_window,
        values.impulse_floor_half_window_frames, values.contact_dedup_radius_frames,
        values.contact_suppression_radius_frames, values.serve_start_lookback_frames,
        values.serve_stillness_window_frames,
        values.sustained_loss_frames,
        values.min_descend_samples, values.body_unit_half_window,
        values.composition_min_scene_len,
    ) == (5, 3, 3, 90, 15, 12, 3, 9, 25, 15, 10, 3, 12, 15)

    values25 = scale_for_fps(25.0)
    assert values25.rest_speed == 0.0024
    assert values25.start_speed == 0.018
    assert (
        values25.rest_window, values25.start_min_frames, values25.smooth_window,
        values25.end_rest_frames, values25.court_absent_window,
        values25.impulse_floor_half_window_frames, values25.contact_dedup_radius_frames,
        values25.contact_suppression_radius_frames, values25.serve_start_lookback_frames,
        values25.serve_stillness_window_frames,
        values25.sustained_loss_frames,
        values25.min_descend_samples, values25.body_unit_half_window,
        values25.composition_min_scene_len,
    ) == (4, 3, 3, 75, 13, 10, 3, 8, 21, 13, 8, 3, 10, 13)


def test_scale_for_fps_half_up_spots_and_floor_one() -> None:
    values50 = scale_for_fps(50.0)
    values60 = scale_for_fps(60.0)
    assert values50.impulse_floor_half_window_frames == 20
    assert values60.contact_dedup_radius_frames == 6
    assert values60.contact_suppression_radius_frames == 18
    assert values60.composition_min_scene_len == 30
    assert values60.court_absent_window == 30
    assert scale_for_fps(25.0).contact_suppression_radius_frames == 8
    assert scale_for_fps(25.0).court_absent_window == 13
    assert scale_for_fps(25.0).composition_min_scene_len == 13
    assert scale_for_fps(1.0).start_min_frames == 1


def test_resolution_keeps_inert_contact_fields_dimensionless() -> None:
    base = BaseAnnotatorConfig()
    for fps in (25.0, 50.0, 60.0):
        resolved = resolve(base, fps)
        assert resolved.thresholds.min_dir_change_deg == base.thresholds.min_dir_change_deg
    assert resolve(base, 25.0).constants.body_unit_half_window == 10
    assert resolve(base, 50.0).constants.body_unit_half_window == 20


def test_scale_for_fps_composition_scene_length_is_distinct_but_currently_equal() -> None:
    values60 = scale_for_fps(60.0)
    values25 = scale_for_fps(25.0)
    assert values60.composition_min_scene_len == values60.court_absent_window
    assert values25.composition_min_scene_len == values25.court_absent_window


def test_scale_for_fps_visible_sample_rows_floor_at_two_frames() -> None:
    values10 = scale_for_fps(10.0)
    assert values10.high_shot_oob_min_visible_frames == 2
    assert values10.reentry_min_visible_frames == 2
    assert scale_for_fps(25.0).high_shot_oob_min_visible_frames == 2
    assert scale_for_fps(60.0).high_shot_oob_min_visible_frames == 5
    assert scale_for_fps(25.0).reentry_min_visible_frames == 2
    assert scale_for_fps(60.0).reentry_min_visible_frames == 5


def test_probe_fps_rejects_vfr_and_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: subprocess.CompletedProcess(
        args, 0, '{"streams": [{"r_frame_rate": "25/1", "avg_frame_rate": "30/1"}]}', '',
    ))
    with pytest.raises(ValueError, match='variable frame rate'):
        probe_fps('video.mp4')  # type: ignore[arg-type]
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: subprocess.CompletedProcess(
        args, 0, '{"streams": [{"r_frame_rate": "0/1", "avg_frame_rate": "0/1"}]}', '',
    ))
    with pytest.raises(ValueError, match='invalid fps'):
        probe_fps('video.mp4')  # type: ignore[arg-type]


def test_stage8_scaled_preset_changes_segmentation_at_50fps() -> None:
    track = np.zeros((300, 3), dtype=float)
    track[:, 2] = 1
    track[20:40, 0] = np.arange(20) * 0.01
    track[40:, 0] = track[39, 0]
    unaware, _ = segment_video(
        track, thresholds=SHIPPED_THRESHOLDS,
        body_unit_half_window=scale_for_fps(30.0).body_unit_half_window,
    )
    aware, _ = segment_video(
        track, thresholds=scale_thresholds(SHIPPED_THRESHOLDS, 50.0),
        body_unit_half_window=scale_for_fps(50.0).body_unit_half_window,
    )
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


def test_resolved_60fps_seam_drives_replay_segmentation_attribution_and_landing(
) -> None:
    """One resolved config crosses every promoted FPS-sensitive boundary exactly once."""
    base = BaseAnnotatorConfig()
    resolved = resolve(base, 60.0)
    assert resolved.constants.court_absent_window == 30
    assert resolved.constants.sustained_loss_frames == 20
    assert resolved.constants.min_descend_samples == 6
    assert resolved.constants.body_unit_half_window == 24
    assert resolved.thresholds.min_dir_change_deg == base.thresholds.min_dir_change_deg

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
    plain_spans, _ = segment_video(
        track, thresholds=resolved.thresholds,
        body_unit_half_window=resolved.constants.body_unit_half_window,
    )
    masked_spans, _ = segment_video(
        track, thresholds=resolved.thresholds, replay_mask=replay,
        body_unit_half_window=resolved.constants.body_unit_half_window,
    )
    assert plain_spans[0][0] == 45
    assert masked_spans[0][0] == 75

    # This is the smallest real sticky-gate context: one in-court standing pose and identity
    # camera-to-court mapping.  The two short boxes sit just outside the base-12 association
    # window around contact 82, but inside the resolved-24 window.
    from annotator.rally_segmentation import CourtBox

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
    court_box = CourtBox((0.0, 1920.0), (0.0, 1080.0), (1.0, 1000.0), (0.0, 1080.0))
    court_info = {'H': np.eye(3), 'border_L': 0.0, 'border_R': 1920.0,
                  'border_U': 0.0, 'border_D': 1080.0}
    gate_kwargs = dict(
        thresholds=resolved.thresholds, pose_bboxes=bboxes, pose_scores=scores, pose_kps=kps,
        pose_ndet=ndet, court_box=court_box, gate_video_id='v', gate_court_info={'v': court_info},
        gate_resolution_table=pd.DataFrame({'width': [1920.0], 'height': [1080.0]}, index=['v']),
    )

    base_radius_contacts = segment_video(
        track, body_unit_half_window=12,
        **(gate_kwargs | {'thresholds': resolved.thresholds._replace(contact_suppression_radius_frames=9)}),
    )[1]
    resolved_contacts = segment_video(
        track, body_unit_half_window=12, **gate_kwargs,
    )[1]
    assert {
        contact.contact_frame for contact in base_radius_contacts
        if contact.wrist_near is not False and contact.suppressed is not True
    } >= {73, 82}
    assert sum(
        contact.wrist_near is not False and contact.suppressed is not True
        for contact in resolved_contacts if contact.contact_frame in (73, 82)
    ) == 1

    short_window_contacts = segment_video(
        track, body_unit_half_window=12,
        **(gate_kwargs | {'thresholds': resolved.thresholds._replace(contact_suppression_radius_frames=9)}),
    )[1]
    full_contacts = segment_video(
        track, body_unit_half_window=resolved.constants.body_unit_half_window,
        **(gate_kwargs | {'thresholds': resolved.thresholds._replace(contact_suppression_radius_frames=9)}),
    )[1]
    short_contact = next(contact for contact in short_window_contacts if contact.contact_frame == 82)
    full_contact = next(contact for contact in full_contacts if contact.contact_frame == 82)
    assert (short_contact.wrist_near, short_contact.suppressed) == (True, False)
    assert (full_contact.wrist_near, full_contact.suppressed) == (False, False)

    # The final resolved contact uses the same track, real pose gate, and resolved constants for
    # both attribution and landing.
    resolved_full_contacts = segment_video(
        track, body_unit_half_window=resolved.constants.body_unit_half_window, **gate_kwargs,
    )[1]
    final_contact = [
        contact.contact_frame for contact in resolved_full_contacts
        if contact.wrist_near is not False and contact.suppressed is not True
    ][-1]
    assert attribute_half(
        final_contact, track, bboxes, scores, kps, court_box, (520.0, 560.0), (1920.0, 1080.0),
        resolved.constants.body_unit_half_window,
    ) is Half.TOP
    # A 15-frame loss is longer than the unscaled base-30 10 but shorter than the resolved 20.
    # It exposes two post-contact descents: five samples (base 3 accepts; resolved 6 rejects),
    # then eight samples (resolved 6 accepts; double-scaled 12 rejects).
    landing_track = track.copy()
    landing_track[112:117, 1] = np.linspace(0.20, 0.24, 5)
    landing_track[117:120, 1] = (0.20, 0.80, 0.80)
    landing_track[120:135, 2] = 0
    landing_track[135:143, 1] = np.linspace(0.20, 0.27, 8)
    landing_track[143:, 1] = 0.20
    assert window_end(final_contact, n_frames, landing_track, np.zeros(n_frames, dtype=bool), 20) > 135
    assert window_end(final_contact, n_frames, landing_track, np.zeros(n_frames, dtype=bool), 10) == 120

    kin = LandingKinematics(np.full(n_frames, np.nan), np.full(n_frames, np.nan), np.zeros(n_frames))
    opts = LandingFilterOptions(1, 0.0, 1, 1, 0.0, use_settle=False, use_carry=False)
    unscaled_landing = pick_landing(
        final_contact, n_frames, landing_track, np.zeros(n_frames, dtype=bool), kin, opts, Half.TOP,
        (520.0, 560.0), (1920.0, 1080.0), court_info,
        replace(resolved.constants, sustained_loss_frames=10, min_descend_samples=3), resolved.fps,
    )
    landing = pick_landing(
        final_contact, n_frames, landing_track, np.zeros(n_frames, dtype=bool), kin, opts, Half.TOP,
        (520.0, 560.0), (1920.0, 1080.0), court_info, resolved.constants, resolved.fps,
    )
    double_scaled_landing = pick_landing(
        final_contact, n_frames, landing_track, np.zeros(n_frames, dtype=bool), kin, opts, Half.TOP,
        (520.0, 560.0), (1920.0, 1080.0), court_info,
        replace(resolved.constants, min_descend_samples=12), resolved.fps,
    )
    assert unscaled_landing is not None
    assert unscaled_landing.frame == 116
    assert landing is not None
    assert landing.frame == 142
    assert double_scaled_landing is None
