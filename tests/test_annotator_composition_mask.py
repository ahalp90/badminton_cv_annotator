"""Annotator composition-mask tests for the per-segment court-view vote.

Synthetic cut frames and keep votes so each segment has an obvious live/dead
verdict. No video IO: the scenedetect cut pass is exercised elsewhere, these pin
the vote arithmetic that turns cuts + a court-view vote into the dead mask.
"""
import numpy as np
import pytest

from annotator.composition_mask import build_composition_mask


def _keep(n_frames: int, court_view_spans: list[tuple[int, int]]) -> np.ndarray:
    """A `(n_frames,)` court-view vote, True over each given span, False elsewhere."""
    keep = np.zeros(n_frames, dtype=bool)
    for start, end in court_view_spans:
        keep[start:end] = True
    return keep


def test_dead_segment_masks_its_whole_span():
    n_frames = 100
    cuts = np.array([40, 70])                     # segments [0,40) [40,70) [70,100)
    keep = _keep(n_frames, [(0, 40), (70, 100)])  # middle segment is not court view

    mask, segments = build_composition_mask(cuts, keep, n_frames, vote=0.5)

    assert not mask[:40].any()      # court-view segment stays live
    assert mask[40:70].all()        # cutaway segment fully dead
    assert not mask[70:100].any()
    assert [seg.is_dead for seg in segments] == [False, True, False]


def test_all_court_view_leaves_mask_all_false():
    n_frames = 60
    cuts = np.array([20, 40])
    keep = np.ones(n_frames, dtype=bool)

    mask, segments = build_composition_mask(cuts, keep, n_frames, vote=0.5)

    assert not mask.any()
    assert all(not seg.is_dead for seg in segments)


def test_vote_equality_is_live():
    """A segment exactly on the threshold stays live (is_dead = fraction < vote)."""
    n_frames = 30
    cuts = np.array([10])                         # segment A [0,10), segment B [10,30)
    # A votes court view on exactly half its frames; B always votes court view (an
    # anchor so nudging A dead does not trip the all-dead guard).
    keep = _keep(n_frames, [(0, 5), (10, 30)])

    mask, segments = build_composition_mask(cuts, keep, n_frames, vote=0.5)

    assert segments[0].keep_fraction == pytest.approx(0.5)
    assert not segments[0].is_dead                # 0.5 >= 0.5, so live
    assert not mask[:10].any()

    # Nudge the vote just above A's fraction and A alone flips to dead.
    dead_mask, dead_segments = build_composition_mask(cuts, keep, n_frames, vote=0.51)
    assert dead_segments[0].is_dead
    assert not dead_segments[1].is_dead
    assert dead_mask[:10].all()
    assert not dead_mask[10:30].any()


def test_boundaries_fold_zero_and_end_and_dedupe():
    """A cut coincident with 0 or n_frames, or a duplicate, must not spawn empty segments."""
    n_frames = 50
    cuts = np.array([0, 25, 25, 50])              # 0 and 50 coincide with the implicit ends; 25 doubled
    keep = _keep(n_frames, [(0, 25)])             # first half court view, second half not

    mask, segments = build_composition_mask(cuts, keep, n_frames, vote=0.5)

    assert len(segments) == 2                     # [0,25) and [25,50), no empties
    assert [(seg.start, seg.end) for seg in segments] == [(0, 25), (25, 50)]
    assert not mask[:25].any()
    assert mask[25:50].all()


def test_cut_order_does_not_matter():
    """np.unique sorts, so shuffled cut frames give the same mask (determinism)."""
    n_frames = 90
    keep = _keep(n_frames, [(0, 30), (60, 90)])
    ordered = build_composition_mask(np.array([30, 60]), keep, n_frames, vote=0.5)[0]
    shuffled = build_composition_mask(np.array([60, 30]), keep, n_frames, vote=0.5)[0]

    assert np.array_equal(ordered, shuffled)


def test_all_dead_fails_loud():
    n_frames = 40
    cuts = np.array([20])
    keep = np.zeros(n_frames, dtype=bool)         # nothing votes court view

    with pytest.raises(ValueError, match='all dead'):
        build_composition_mask(cuts, keep, n_frames, vote=0.5)
