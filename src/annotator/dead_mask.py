"""Public builder for the annotator's per-frame dead-time mask."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .composition_mask import build_composition_mask
from .config import COMPOSITION_KEEP_VOTE
from .replay_mask import combine_mask
from .types import DeadMaskMode


def _validate_composition_inputs(
    n_frames: int,
    keep_vote: np.ndarray | None,
    cut_frames: Sequence[int] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate composition inputs before calling the existing mask builder."""
    if cut_frames is None:
        raise ValueError('composition dead mask requires cut_frames')
    if keep_vote is None:
        raise ValueError('composition dead mask requires keep_vote')

    keep_vote_array = np.asarray(keep_vote)
    if keep_vote_array.dtype != np.dtype(bool):
        raise ValueError(f'keep_vote must have bool dtype, got {keep_vote_array.dtype}')
    if keep_vote_array.ndim != 1:
        raise ValueError('keep_vote must be a one-dimensional bool array')
    if len(keep_vote_array) != n_frames:
        raise ValueError(
            f'keep_vote length {len(keep_vote_array)} != n_frames {n_frames}'
        )

    cut_array = np.asarray(cut_frames)
    if cut_array.ndim != 1:
        raise ValueError('cut_frames must be a one-dimensional sequence of integers')
    if cut_array.size and not np.issubdtype(cut_array.dtype, np.integer):
        raise ValueError('cut_frames must contain integers')
    if np.any((cut_array < 0) | (cut_array > n_frames)):
        raise ValueError(f'cut_frames must be in [0, {n_frames}]')
    return cut_array.astype(int, copy=False), keep_vote_array


def build_dead_mask(
    mode: DeadMaskMode,
    n_frames: int,
    fps: float,
    *,
    court_present: np.ndarray | None = None,
    homography_rows: list[dict] | None = None,
    track: np.ndarray | None = None,
    rally_spans: list[tuple[int, int]] | None = None,
    cut_frames: Sequence[int] | np.ndarray | None = None,
    keep_vote: np.ndarray | None = None,
    vote: float | None = None,
    shuttle_hallucination_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Build a boolean dead-time mask using the selected producer policy.

    All modes consume ``n_frames``. ``REPLAY`` also consumes ``fps`` and the
    optional ``court_present``, ``homography_rows``, ``track``, ``rally_spans``,
    and ``shuttle_hallucination_mask`` signals. A missing replay signal
    contributes an all-False component. ``COMPOSITION`` requires ``cut_frames``
    and the frame-aligned boolean ``keep_vote``; ``vote`` overrides its default
    threshold. It ignores the replay inputs and ``fps``. ``UNION`` consumes both
    input groups and combines the resulting masks elementwise.

    :return: ``(n_frames,)`` boolean mask, True for excluded dead-time frames.
    """
    # Normalising here keeps the dispatch below on identity checks; a junk mode
    # string fails loudly in the cast.
    mode = DeadMaskMode(mode)
    if mode is DeadMaskMode.REPLAY:
        return combine_mask(
            court_present, homography_rows, track, rally_spans, n_frames, fps,
            non_evidence=shuttle_hallucination_mask,
        )

    cuts, votes = _validate_composition_inputs(n_frames, keep_vote, cut_frames)
    composition_vote = COMPOSITION_KEEP_VOTE if vote is None else vote
    composition, _ = build_composition_mask(cuts, votes, n_frames, composition_vote)
    if mode is DeadMaskMode.COMPOSITION:
        return composition

    replay = combine_mask(
        court_present, homography_rows, track, rally_spans, n_frames, fps,
        non_evidence=shuttle_hallucination_mask,
    )
    return composition | replay
