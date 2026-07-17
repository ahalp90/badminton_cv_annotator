"""Frame-rate-relative scraper constants.

All caller-supplied frame counts and per-frame speeds on the scraper surface
are base-30 values. They are scaled exactly once when fps context exists.
Fields of :class:`FpsConstants` are final and never rescaled. YouTube sources
are assumed CFR; :func:`probe_fps` rejects variable-frame-rate files loudly.
"""
from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

BASE_FPS = 30.0
REST_SPEED_BASE30 = 0.002
START_SPEED_BASE30 = 0.015


@dataclass(frozen=True)
class FpsConstants:
    """Final fps-scaled values; the base-30 table is scaled exactly once."""
    rest_speed: float
    rest_window: int
    start_speed: float
    start_min_frames: int
    smooth_window: int
    end_rest_frames: int
    court_absent_window: int
    impulse_floor_half_window_frames: int
    contact_dedup_radius_frames: int
    contact_suppression_radius_frames: int
    serve_start_lookback_frames: int
    wideshot_drift_end_frames: int
    sustained_loss_frames: int
    min_descend_samples: int
    body_unit_half_window: int
    # Same value as court_absent_window today; distinct concept (PySceneDetect's
    # minimum scene length), so tuning one never silently moves the other.
    composition_min_scene_len: int


def _time(base30: float, fps: float) -> int:
    return max(1, math.floor(base30 * fps / BASE_FPS + 0.5))


def scale_for_fps(fps: float) -> FpsConstants:
    """Scale the base-30 table for a positive finite CFR frame rate."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f'fps must be positive and finite, got {fps!r}')
    return FpsConstants(
        rest_speed=REST_SPEED_BASE30 * BASE_FPS / fps,
        rest_window=_time(5.0, fps), start_speed=START_SPEED_BASE30 * BASE_FPS / fps,
        start_min_frames=_time(3.0, fps), smooth_window=_time(3.0, fps),
        end_rest_frames=_time(90.0, fps), court_absent_window=_time(15.0, fps),
        impulse_floor_half_window_frames=_time(12.0, fps), contact_dedup_radius_frames=_time(3.0, fps),
        contact_suppression_radius_frames=_time(9.0, fps), serve_start_lookback_frames=_time(25.0, fps),
        wideshot_drift_end_frames=_time(10.0, fps), sustained_loss_frames=_time(10.0, fps),
        min_descend_samples=_time(3.0, fps), body_unit_half_window=_time(12.0, fps),
        composition_min_scene_len=_time(15.0, fps),
    )


def probe_fps(video_path: Path) -> float:
    """Read a CFR rate with ffprobe, rejecting missing, invalid, and VFR streams."""
    try:
        completed = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
             'stream=r_frame_rate,avg_frame_rate', '-of', 'json', str(video_path)],
            check=True, capture_output=True, text=True,
        )
        stream = json.loads(completed.stdout)['streams'][0]
        rates = [float(Fraction(stream[key])) for key in ('r_frame_rate', 'avg_frame_rate')]
    except (KeyError, IndexError, ValueError, ZeroDivisionError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        raise ValueError(f'{video_path}: ffprobe could not read a valid video fps') from exc
    if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError(f'{video_path}: ffprobe returned a missing or invalid fps')
    if abs(rates[0] - rates[1]) > 1e-6:
        raise ValueError(f'{video_path}: variable frame rate is unsupported')
    return rates[0]
