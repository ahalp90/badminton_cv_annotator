"""FPS-relativity regression tests for the scraper's tuned-25 public surface."""
from __future__ import annotations

import subprocess

import numpy as np
import pytest

from src.scraper.config import SHIPPED_THRESHOLDS
from src.scraper.fps_constants import probe_fps, scale_for_fps
from src.scraper.point_winner import LandingFilterOptions, convert_landing_options
from src.scraper.stage8_rally_segmentation import scale_thresholds, segment_video
from src.scraper.stage9_replay_mask import court_absence_signal


def test_scale_for_fps_has_tuned_25_identity_for_every_scaled_row() -> None:
    values = scale_for_fps(25.0)
    assert values.rest_speed == 0.002
    assert values.start_speed == 0.015
    assert (
        values.rest_window, values.start_min_frames, values.smooth_window,
        values.end_rest_frames, values.court_absent_window,
        values.impulse_floor_half_window_frames, values.contact_dedup_radius_frames,
        values.contact_suppression_radius_frames, values.serve_start_lookback_frames,
        values.wideshot_drift_end_frames, values.sustained_loss_frames,
        values.min_descend_samples, values.body_unit_half_window,
    ) == (5, 3, 3, 90, 15, 12, 3, 9, 25, 10, 10, 3, 12)


def test_scale_for_fps_half_up_spots_and_floor_one() -> None:
    values50 = scale_for_fps(50.0)
    values60 = scale_for_fps(60.0)
    assert values50.impulse_floor_half_window_frames == 24
    assert values60.contact_dedup_radius_frames == 7
    assert values60.contact_suppression_radius_frames == 22
    assert scale_for_fps(1.0).start_min_frames == 1


def test_scale_for_fps_composition_scene_length_is_distinct_from_court_absence() -> None:
    values60 = scale_for_fps(60.0)
    values25 = scale_for_fps(25.0)
    assert values60.composition_min_scene_len == 36
    assert values25.composition_min_scene_len == 15
    assert values60.court_absent_window == 36
    assert values25.court_absent_window == 15


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
    unaware, _ = segment_video(track, thresholds=SHIPPED_THRESHOLDS)
    aware, _ = segment_video(track, thresholds=scale_thresholds(SHIPPED_THRESHOLDS, 50.0))
    assert unaware == []
    assert aware == [(21, 41)]


def test_replay_court_absence_scales_at_50fps() -> None:
    present = np.ones(40, dtype=bool)
    present[5:25] = False
    assert court_absence_signal(present, 40).any()
    assert not court_absence_signal(present, 40, 50.0).any()


def test_landing_options_are_converted_once() -> None:
    opts = LandingFilterOptions(7, 0.004, 5, 7, 0.75)
    scaled = convert_landing_options(opts, 50.0)
    assert (scaled.settle_win, scaled.settle_thr, scaled.settle_min, scaled.carry_win, scaled.carry_thr) == (
        14, 0.002, 10, 14, 0.75,
    )
