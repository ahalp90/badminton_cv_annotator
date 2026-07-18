"""Focused contracts for the public dead-mask builder."""
from __future__ import annotations

import numpy as np
import pytest

from annotator.config import COMPOSITION_KEEP_VOTE
from annotator.dead_mask import build_dead_mask
from annotator.replay_mask import combine_mask
from annotator.types import DeadMaskMode


def _composition_inputs() -> tuple[np.ndarray, np.ndarray]:
    return np.array([3, 5], dtype=int), np.array([True, True, True, False, False, True], dtype=bool)


def test_replay_mode_delegates_existing_union() -> None:
    present = np.array([True, False, False], dtype=bool)
    result = build_dead_mask(DeadMaskMode.REPLAY, 3, 30.0, court_present=present)
    expected = combine_mask(present, None, None, None, 3, 30.0)
    assert result.dtype == bool
    assert result.shape == (3,)
    np.testing.assert_array_equal(result, expected)


def test_composition_mode_uses_existing_default_vote() -> None:
    cuts, keep_vote = _composition_inputs()
    result = build_dead_mask(
        DeadMaskMode.COMPOSITION, len(keep_vote), 30.0,
        cut_frames=cuts, keep_vote=keep_vote,
    )
    # The first segment is live at the composition module's 0.5 default;
    # the second segment is dead.
    assert COMPOSITION_KEEP_VOTE == 0.5
    np.testing.assert_array_equal(result, [False, False, False, True, True, False])
    assert result.dtype == bool


def test_union_mode_is_elementwise_or() -> None:
    cuts, keep_vote = _composition_inputs()
    present = np.array([True, True, True, True, True, False], dtype=bool)
    composition = build_dead_mask(
        DeadMaskMode.COMPOSITION, 6, 30.0, cut_frames=cuts, keep_vote=keep_vote,
    )
    replay = build_dead_mask(DeadMaskMode.REPLAY, 6, 30.0, court_present=present)
    result = build_dead_mask(
        DeadMaskMode.UNION, 6, 30.0,
        court_present=present, cut_frames=cuts, keep_vote=keep_vote,
    )
    np.testing.assert_array_equal(result, composition | replay)
    assert result.dtype == bool
    assert result.shape == (6,)


@pytest.mark.parametrize('missing', ['cut_frames', 'keep_vote'])
def test_composition_validation_names_missing_input(missing: str) -> None:
    kwargs = {'cut_frames': np.array([], dtype=int), 'keep_vote': np.ones(3, dtype=bool)}
    kwargs[missing] = None
    with pytest.raises(ValueError, match=missing):
        build_dead_mask(DeadMaskMode.COMPOSITION, 3, 30.0, **kwargs)


def test_union_validates_before_building() -> None:
    with pytest.raises(ValueError, match='keep_vote.*bool'):
        build_dead_mask(
            DeadMaskMode.UNION, 3, 30.0,
            cut_frames=np.array([1], dtype=int), keep_vote=np.ones(3, dtype=np.int8),
        )


@pytest.mark.parametrize('cuts', [np.array([-1], dtype=int), np.array([4], dtype=int)])
def test_composition_rejects_out_of_range_cuts(cuts: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r'cut_frames.*\[0, 3\]'):
        build_dead_mask(
            DeadMaskMode.COMPOSITION, 3, 30.0,
            cut_frames=cuts, keep_vote=np.ones(3, dtype=bool),
        )


def test_composition_rejects_non_integer_cuts() -> None:
    with pytest.raises(ValueError, match='cut_frames.*integers'):
        build_dead_mask(
            DeadMaskMode.COMPOSITION, 3, 30.0,
            cut_frames=np.array([1.0]), keep_vote=np.ones(3, dtype=bool),
        )


def test_composition_rejects_non_vector_keep_vote() -> None:
    with pytest.raises(ValueError, match='keep_vote.*one-dimensional'):
        build_dead_mask(
            DeadMaskMode.COMPOSITION, 3, 30.0,
            cut_frames=[], keep_vote=np.ones((1, 3), dtype=bool),
        )
