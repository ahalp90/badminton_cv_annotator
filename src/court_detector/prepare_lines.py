"""Scale and group detected line fragments for court search."""


from __future__ import annotations

import numpy as np

from . import geometry as detector


def prepare(source: dict) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], tuple[int, int]]:
    """Reproduce detector scaling and original proposal families."""
    width, height = source['dimensions']['width'], source['dimensions']['height']
    settings = detector.Settings(wide_families=True, min_supported_lines=3)
    resize = min(1.0, settings.max_dimension / max(width, height))
    size = (round(width * resize), round(height * resize))
    native_scale = np.asarray([width, height], dtype=float) / size
    # Working fragments stay float64: direction estimates, line groups and line fits start from them.
    segments = np.asarray(source['segments_px'], dtype=float).reshape(-1, 4) / np.tile(native_scale, 2)
    raw_families = detector._wide_line_families(segments)
    families = (detector._merge_lines(raw_families[0], settings), detector._merge_lines(raw_families[1], settings))
    return segments, families, size
