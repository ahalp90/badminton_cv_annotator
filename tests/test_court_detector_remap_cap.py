"""cv2.remap rejects maps with 32,767 or more points, so the court samplers split larger requests."""

from __future__ import annotations

from contextlib import nullcontext

import cv2
import numpy as np
import pytest

from court_detector import measurements, stripe_refit
from court_detector.sampling import (
    REMAP_MAX_POINTS,
    prepared_measurements,
    remap_points,
)

FRAME_WIDTH, FRAME_HEIGHT = 80, 60
# One brightness_profiles call on ShuttleSet22 video 55, frame 53683, which stopped a court run:
# 1,398 fragments x 14 interior samples x 2 offsets = 39,144 points.
OVER_CAP_SHAPE = (1398, 14, 2)
REFERENCE_PIECE_POINTS = 1000


def random_points(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Points inside the frame, so every one is valid for every sampler."""
    return rng.uniform((0, 0), (FRAME_WIDTH - 1, FRAME_HEIGHT - 1), (*shape, 2)).astype(np.float32)


def remap_in_small_pieces(image: np.ndarray, xy: np.ndarray, border_mode: int) -> np.ndarray:
    """Reference samples from direct cv2.remap calls, each well under the cap."""
    pieces = [
        cv2.remap(image, xy[start:start + REFERENCE_PIECE_POINTS, 0], xy[start:start + REFERENCE_PIECE_POINTS, 1],
                  cv2.INTER_LINEAR, borderMode=border_mode)
        for start in range(0, len(xy), REFERENCE_PIECE_POINTS)
    ]
    return np.concatenate(pieces, axis=1)


@pytest.mark.parametrize("cached", [False, True])
def test_grey_samplers_accept_a_brightness_profile_over_the_cap(cached: bool) -> None:
    rng = np.random.default_rng(55)
    frame = rng.integers(0, 256, (FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    points = random_points(OVER_CAP_SHAPE, rng)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    expected = remap_in_small_pieces(grey, points.reshape(-1, 2), cv2.BORDER_REPLICATE).reshape(OVER_CAP_SHAPE)
    # The detector runs the cached sampler that prepared_measurements swaps in; research code runs the original.
    with prepared_measurements(measurements) if cached else nullcontext():
        values = measurements.grayscale_sample(frame, points)
    np.testing.assert_array_equal(values, expected)


@pytest.mark.parametrize(
    ("point_count", "expected_calls"),
    [(REMAP_MAX_POINTS, 1), (REMAP_MAX_POINTS + 1, 2), (2 * REMAP_MAX_POINTS, 2)],
)
def test_remap_points_matches_direct_calls_and_keeps_one_call_under_the_cap(
    monkeypatch: pytest.MonkeyPatch, point_count: int, expected_calls: int,
) -> None:
    rng = np.random.default_rng(point_count)
    grey = rng.uniform(0, 255, (FRAME_HEIGHT, FRAME_WIDTH)).astype(np.float32)
    xy = random_points((point_count,), rng)
    expected = remap_in_small_pieces(grey, xy, cv2.BORDER_REPLICATE)
    piece_sizes = []
    original_remap = cv2.remap

    def counting_remap(image: np.ndarray, map_x: np.ndarray, *args, **kwargs) -> np.ndarray:
        piece_sizes.append(len(map_x))
        return original_remap(image, map_x, *args, **kwargs)

    monkeypatch.setattr(cv2, "remap", counting_remap)
    sampled = remap_points(grey, xy, cv2.BORDER_REPLICATE)
    np.testing.assert_array_equal(sampled, expected)
    # The detector's scoring stays in float32 for speed.
    assert sampled.dtype == np.float32
    # One call under the cap is what keeps earlier court outputs identical.
    assert len(piece_sizes) == expected_calls
    assert sum(piece_sizes) == point_count


@pytest.mark.parametrize("plane_shape", [(FRAME_HEIGHT, FRAME_WIDTH), (FRAME_HEIGHT, FRAME_WIDTH, 3)],
                         ids=["grey", "lab"])
def test_stripe_sample_image_accepts_points_over_the_cap(plane_shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(len(plane_shape))
    planes = rng.uniform(0, 255, plane_shape).astype(np.float32)
    points = random_points(OVER_CAP_SHAPE, rng)
    sampled, valid = stripe_refit.sample_image(planes, points, np.empty((0, 4)))
    expected = remap_in_small_pieces(planes, points.reshape(-1, 2), cv2.BORDER_CONSTANT)
    # The stripe refit indexes samples by fragment, position and channel, so the shapes must hold.
    assert sampled.shape == points.shape[:-1] + planes.shape[2:]
    assert valid.shape == points.shape[:-1]
    assert valid.all()
    np.testing.assert_array_equal(sampled, expected.reshape(sampled.shape))
