"""Check court shape, camera geometry and player positions."""


from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from . import geometry as detector


def gate_evidence(
    corners: np.ndarray, source: dict, scale: np.ndarray, size: tuple[int, int], families: tuple,
    maps: np.ndarray, zone: Any,
) -> dict:
    """Measure original gates independently; no retained-pool acceptance is inferred."""
    homography = cv2.getPerspectiveTransform(detector.CORNER_COURT_M, (corners / scale).astype(np.float32))
    feet = np.asarray([[[np.nan, np.nan] if foot is None else foot for foot in frame]
                       for frame in source['all_feet_px']], dtype=float) / scale
    one, two = zone.player_fractions(homography[None], feet)
    projected, scores, means, counts = detector._score(
        homography[None], maps, detector.Settings(wide_families=True, min_supported_lines=3), families,
    )
    if not len(projected):
        return {'geometry_valid': False, 'player_fractions': [float(one[0]), float(two[0])],
                'floor_score': None, 'camera_error': None}
    _, camera_error, _ = zone.net_segments(corners, (source['dimensions']['width'], source['dimensions']['height']))
    return {'geometry_valid': True, 'player_fractions': [float(one[0]), float(two[0])],
            'floor_score': float(scores[0]), 'family_support': means[0].tolist(), 'line_counts': counts[0].tolist(),
            'camera_error': float(camera_error) if np.isfinite(camera_error) else None}
