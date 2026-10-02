"""Check whether players fit inside the court and its two halves."""


from __future__ import annotations

import numpy as np

from . import geometry as detector
from .camera import net_segments

__all__ = ['net_segments', 'player_fractions']


COURT_SIZE_M = detector.CORNER_COURT_M.max(axis=0)


def player_fractions(
    homographies: np.ndarray, feet_px: np.ndarray, margin: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure observed presence within each complete projected court.

    :param homographies: Court metres to image pixels, one matrix per hypothesis.
    :param feet_px: (sampled frames, two player slots, image xy); NaN means missing.
    :return: Fractions with at least one player, and with one player in each half.
    """
    inverse = np.linalg.inv(homographies)
    homogeneous = np.concatenate((feet_px, np.ones((*feet_px.shape[:-1], 1))), axis=-1)
    mapped = np.einsum("hij,fpj->hfpi", inverse, homogeneous)
    with np.errstate(divide="ignore", invalid="ignore"):
        court = mapped[..., :2] / mapped[..., 2:] / COURT_SIZE_M
    inside = np.isfinite(court).all(axis=-1) & (court >= -margin).all(axis=-1) & (court <= 1 + margin).all(axis=-1)
    far = inside & (court[..., 1] < 0.5)
    near = inside & (court[..., 1] >= 0.5)
    return inside.any(axis=-1).mean(axis=-1), (far.any(axis=-1) & near.any(axis=-1)).mean(axis=-1)
