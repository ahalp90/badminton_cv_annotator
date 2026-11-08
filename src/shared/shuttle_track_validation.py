"""Check normalised shuttle positions and visibility before annotation."""

from __future__ import annotations

import numpy as np


def validate_shuttle_track(track: np.ndarray, frame_count: int | None = None) -> None:
    """Require a finite floating array of (x, y, visible), one row per video frame.

    Visibility must be 0 or 1. Visible xy must be within [0, 1]; invisible xy
    only needs to be finite. If frame_count is supplied, require that many rows.
    """
    values = np.asarray(track)
    expected_shape = (frame_count, 3) if frame_count is not None else None
    if expected_shape is not None and values.shape != expected_shape:
        raise ValueError(f"track shape {values.shape} != {expected_shape}")
    if expected_shape is None and (values.ndim != 2 or values.shape[1] != 3):
        raise ValueError(f"track must have shape (frames, 3), got {values.shape}")
    if not np.issubdtype(values.dtype, np.floating):
        raise ValueError(f"track must have floating dtype, got {values.dtype}")
    if not np.isfinite(values).all():
        raise ValueError("track must contain only finite values")
    if not np.isin(values[:, 2], (0.0, 1.0)).all():
        raise ValueError("track visibility must contain only 0 or 1")
    visible_coordinates = values[values[:, 2] == 1.0, :2]
    if ((visible_coordinates < 0.0) | (visible_coordinates > 1.0)).any():
        raise ValueError("visible shuttle coordinates must be within [0, 1]")
