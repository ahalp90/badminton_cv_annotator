"""What the D17 candidate search adds to the live generators.

- DIRECTION_SETTINGS: the vanishing-point estimator settings for the G0 and G1
  direction pairs, from the frozen record
  frozen_views/baseline_directions/gxBQ_window_00_frame_0.json.gz.
- The G1 paint filter: keep a line fragment when it sits on a bright, unsaturated
  ridge (ridge contrast >= PAINT_CONTRAST grey levels, peak saturation <=
  PAINT_SATURATION). line_identity/paint_profiles.py explains the two measures.
- The line-template seeds: where each pair of the three longest lengthwise lines
  meets. They join the template search's own vanishing points (Am1 recovery).

Research scripts import these back, so this module stays a leaf: it imports only
numpy, OpenCV and SciPy, and never edits sys.path.
"""

from __future__ import annotations

from itertools import combinations

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

DIRECTION_SETTINGS = {
    "angle_deg": 1.5, "direction_lines": 128, "pencils": 16, "overlap": 0.8, "rectangles": 16384,
    "candidate_batch": 256, "pencil_selection": "coverage",
}
SEED_LINE_COUNT = 3

PROFILE_HALF_WIDTH_WORKING_PX = 10
PROFILE_STEP_WORKING_PX = 0.5
RIDGE_SEARCH_WORKING_PX = 2.5
FLANK_WINDOWS_WORKING_PX = ((3., 6.), (6., 10.))
SAMPLES_ALONG = 12
PAINT_CONTRAST = 20.
PAINT_SATURATION = 90.


def profiles(frame: np.ndarray, fragments_native: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Brightness and saturation across each fragment: arrays (fragments, samples, offsets)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    brightness, saturation = hsv[..., 2], hsv[..., 1]
    starts, ends = fragments_native[:, :2], fragments_native[:, 2:]
    along = np.linspace(0.05, 0.95, SAMPLES_ALONG)
    centres = starts[:, None, :] + along[None, :, None] * (ends - starts)[:, None, :]
    directions = (ends - starts) / np.maximum(np.linalg.norm(ends - starts, axis=1), 1e-9)[:, None]
    normals = np.stack((-directions[:, 1], directions[:, 0]), axis=1)
    offsets_working = profile_offsets()
    # Offsets are in working pixels so the profile spans the same court-scale width on every view.
    offsets_native = offsets_working[None, None, :, None] * (normals * scale.mean())[:, None, None, :]
    points = centres[:, :, None, :] + offsets_native
    coordinates = np.stack((points[..., 1].ravel(), points[..., 0].ravel()))
    shape = points.shape[:3]
    sampled_brightness = map_coordinates(brightness, coordinates, order=1, mode='nearest').reshape(shape)
    sampled_saturation = map_coordinates(saturation, coordinates, order=1, mode='nearest').reshape(shape)
    return np.stack((sampled_brightness, sampled_saturation))


def profile_offsets() -> np.ndarray:
    return np.arange(-PROFILE_HALF_WIDTH_WORKING_PX, PROFILE_HALF_WIDTH_WORKING_PX + PROFILE_STEP_WORKING_PX / 2,
                     PROFILE_STEP_WORKING_PX)


def features(profile: np.ndarray) -> np.ndarray:
    """Per fragment: ridge contrast, saturation at the peak, peak offset, centre and flank brightness (medians over samples)."""
    brightness, saturation = profile
    offsets = profile_offsets()
    search = np.abs(offsets) <= RIDGE_SEARCH_WORKING_PX
    peak_index = brightness[..., search].argmax(axis=-1)
    peak = np.take_along_axis(brightness[..., search], peak_index[..., None], axis=-1)[..., 0]
    peak_saturation = np.take_along_axis(saturation[..., search], peak_index[..., None], axis=-1)[..., 0]
    # A ridge must fall away on both sides, so each window's contrast is measured against its brighter flank.
    flanks = []
    for near, far in FLANK_WINDOWS_WORKING_PX:
        left_flank = (offsets <= -near) & (offsets >= -far)
        right_flank = (offsets >= near) & (offsets <= far)
        flanks.append(np.maximum(brightness[..., left_flank].mean(axis=-1), brightness[..., right_flank].mean(axis=-1)))
    flank = np.min(flanks, axis=0)
    contrast = peak - flank
    centre = int(np.flatnonzero(offsets == 0)[0])
    return np.stack((np.median(contrast, axis=1), np.median(peak_saturation, axis=1),
                     np.median(offsets[search][peak_index], axis=1),
                     np.median(brightness[..., centre], axis=1), np.median(flank, axis=1)), axis=1)



def paint_mask(source: dict, frame: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Mark the source's fragments that sit on paint: one bool per fragment in segments_px.

    :param frame: Native BGR frame.
    :param scale: Native pixels per working pixel, as (x, y).
    """
    fragments = np.asarray(source['segments_px'], dtype=float)
    contrast, saturation = features(profiles(frame, fragments, scale))[:, :2].T
    return (contrast >= PAINT_CONTRAST) & (saturation <= PAINT_SATURATION)


def filtered_source(source: dict, keep: np.ndarray) -> dict:
    return {**source, 'segments_px': [segment for segment, kept in zip(source['segments_px'], keep, strict=True) if kept]}


def seed_points(lengthwise_lines: np.ndarray) -> np.ndarray:
    """Use each pair of the longest merged lengthwise lines, including infinity."""
    points = []
    for first, second in combinations(lengthwise_lines[:SEED_LINE_COUNT], 2):
        meeting = np.cross(first, second)
        norm = np.linalg.norm(meeting)
        if norm > 0:
            points.append(meeting / norm)
    return np.asarray(points, dtype=float).reshape(-1, 3)
