"""Focused tests for the optional frame-rate motion scaling."""

from __future__ import annotations

import freeze_tree_contact_features as freezer
import numpy as np
import pytest


def _signals() -> dict[str, np.ndarray]:
    return {
        "shuttle_vx": np.asarray([1.0, 2.0], dtype=np.float32),
        "shuttle_vy": np.asarray([3.0, 4.0], dtype=np.float32),
        "shuttle_speed": np.asarray([5.0, 6.0], dtype=np.float32),
        "shuttle_impulse": np.asarray([7.0, 8.0], dtype=np.float32),
        "shuttle_impulse_ratio": np.asarray([9.0, 10.0], dtype=np.float32),
        "ankle_speed_top": np.asarray([11.0, 12.0], dtype=np.float32),
        "ankle_speed_bot": np.asarray([13.0, 14.0], dtype=np.float32),
    }


def test_raw_per_frame_is_unchanged() -> None:
    signals = _signals()

    scaled = freezer._scale_motion_signals(signals, 25.0, "raw_per_frame")

    for name, values in signals.items():
        np.testing.assert_array_equal(scaled[name], values)


def test_base30_per_frame_scales_linear_signals_at_25_fps() -> None:
    signals = _signals()
    factor = 25.0 / 30.0

    scaled = freezer._scale_motion_signals(signals, 25.0, "base30_per_frame")

    for name in freezer.MOTION_LINEAR_SIGNALS:
        np.testing.assert_allclose(scaled[name], signals[name] * factor)


def test_base30_per_frame_scales_impulse_quadratically_and_keeps_ratio() -> None:
    signals = _signals()
    factor = 25.0 / 30.0

    scaled = freezer._scale_motion_signals(signals, 25.0, "base30_per_frame")

    np.testing.assert_allclose(scaled["shuttle_impulse"], signals["shuttle_impulse"] * factor**2)
    np.testing.assert_array_equal(scaled["shuttle_impulse_ratio"], signals["shuttle_impulse_ratio"])


def test_base30_per_frame_is_an_identity_at_30_fps() -> None:
    signals = _signals()

    scaled = freezer._scale_motion_signals(signals, 30.0, "base30_per_frame")

    for name, values in signals.items():
        np.testing.assert_array_equal(scaled[name], values)


def test_motion_scaling_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unsupported motion mode"):
        freezer._scale_motion_signals(_signals(), 25.0, "unknown")
