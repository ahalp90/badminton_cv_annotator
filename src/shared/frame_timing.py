"""Convert frame counts and per-frame speeds between video frame rates."""

import math

BASE_FPS = 30.0


def frames_at_fps(frames_at_30_fps: float, fps: float) -> int:
    """Scale a base-30 frame count, rounding halves up and keeping at least one frame."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f'fps must be positive and finite, got {fps!r}')
    return max(1, math.floor(frames_at_30_fps * fps / BASE_FPS + 0.5))


def speed_at_fps(speed_at_30_fps: float, fps: float) -> float:
    """Convert speed per frame from 30 fps to the video's frame rate."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f'fps must be positive and finite, got {fps!r}')
    return speed_at_30_fps * BASE_FPS / fps


