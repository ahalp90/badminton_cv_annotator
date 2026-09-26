"""Sample image colours across projected court markings."""


from __future__ import annotations

from pathlib import Path

import numpy as np

from . import geometry as detector
from . import line_observations as assignment


def ridge_mask(frame: np.ndarray, segments: np.ndarray) -> np.ndarray:
    accepted = detector._filter_painted_stripes(frame, segments)
    identities = {tuple(segment) for segment in accepted}
    return np.asarray([tuple(segment) in identities for segment in segments], dtype=bool)


def profiles(frame: np.ndarray, homographies: np.ndarray) -> list[dict]:
    """Measure finite visible intervals, retaining missing profiles separately."""
    height, width = frame.shape[:2]
    projected, _ = detector.project(homographies, detector.SEGMENTS_M)
    endpoints, visible = detector._visible_samples(projected.reshape(-1, 12, 2, 2), (width, height), 2)
    passed = np.zeros(visible.shape, dtype=bool)
    passed[visible] = ridge_mask(frame, endpoints[visible].reshape(-1, 4))
    results = []
    for interval_pass, interval_visible in zip(passed, visible, strict=True):
        markings = []
        for intervals in assignment.MARKING_INTERVALS:
            available = interval_visible[list(intervals)]
            markings.append(float(interval_pass[list(intervals)][available].mean()) if available.any() else None)
        available_scores = [value for value in markings if value is not None]
        results.append({'score': float(np.mean(available_scores)) if available_scores else None,
                        'marking_ridge': markings, 'interval_visible': interval_visible.tolist(),
                        'interval_ridge': interval_pass.tolist()})
    return results


def frame_path(source: dict, root: Path) -> Path:
    if source['id'].startswith('gxBQ'):
        return root / 'gx_extension/people' / source['image']
    if source['id'].startswith('shuttleset'):
        return root / 'original' / source['image']
    video = source['id'].split('_')[0]
    frame = int(source['id'].rsplit('_', 1)[1])
    return root / 'axis_matching_20260914/images' / video / f'frame_{frame:08d}.png'
