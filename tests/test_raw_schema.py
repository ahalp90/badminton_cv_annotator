"""Synthetic, inference-free tests for raw_extract.extract_raw_frame's assembly.

Ported from the retired ``validation_scripts/rtmlib_migration/gate_raw_schema.py``
so the five-array assembly runs under CI. extract_raw_frame caps each frame to the
top ``n_max`` detections by bbox_score and NaN-pads the rest; these cover the
truncation / empty / partial edges a small real clip never hits, plus the dtype
contract (four float32 arrays, an int8-representable detection count).

Runs with NO rtmlib installed (CI has none). extract_raw_frame only attribute-reads
the detection object, so a duck-typed NamedTuple stands in for the real
``rtmlib_pose.FrameDetections`` (same four fields), matching the ``_FakeFrame``
pattern in test_extract_failure_guard.py.

Each synthetic detection ``i`` is tagged with marker ``i`` on every field, so the
marker read back from a padded slot says which detection landed there.
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np

from pipeline.config import COCO_N_JOINTS
from preparing_data.raw_extract import extract_raw_frame

_J = COCO_N_JOINTS  # 17, the COCO keypoint count the real adapter emits
# Duck-typed stand-in for rtmlib_pose.FrameDetections: extract_raw_frame only
# reads these four attributes off the detection object.
_FakeFrame = namedtuple("_FakeFrame", ["keypoints", "bboxes", "bbox_scores", "kp_scores"])


def _frame(scores: list[float]) -> _FakeFrame:
    """One synthetic frame of ``len(scores)`` detections, detection i tagged with
    marker i on every field."""
    m = len(scores)
    markers = np.arange(m, dtype=np.float32)
    kps = np.tile(markers.reshape(m, 1, 1), (1, _J, 2))  # (m, J, 2), every value == i
    bboxes = np.stack([markers, markers, markers + 1, markers + 1], axis=1)  # valid xyxy
    kp_scores = np.full((m, _J), 0.9, dtype=np.float32)
    return _FakeFrame(
        keypoints=kps, bboxes=bboxes.astype(np.float32),
        bbox_scores=np.asarray(scores, dtype=np.float32), kp_scores=kp_scores,
    )


def _markers(kps: np.ndarray, n: int) -> list[int]:
    """Recover the detection marker from each of the first ``n`` padded slots."""
    return [int(kps[i, 0, 0]) for i in range(n)]


def test_truncation_keeps_top_n_max_descending():
    """20 dets, ascending scores, n_max=16: keep the top-16 by score in DESCENDING
    order (markers 19..4), drop the 4 lowest, NaN-pad slots 16.., float32 throughout."""
    scores = [(i + 1) / 20 for i in range(20)]  # det0=0.05 (low) .. det19=1.0 (high)
    kps, bboxes, scores_out, kp_scores, n_dets = extract_raw_frame(
        _frame(scores), 16, "syn", 0, set()
    )

    assert n_dets == 16
    assert _markers(kps, 16) == list(range(19, 3, -1))  # [19, 18, ..., 4]
    assert np.all(np.diff(scores_out[:16]) <= 0)  # kept scores non-increasing
    for arr in (kps, bboxes, scores_out, kp_scores):
        assert arr.dtype == np.float32
    assert np.isnan(kps[16:]).all() and np.isnan(bboxes[16:]).all()
    assert np.isnan(scores_out[16:]).all() and np.isnan(kp_scores[16:]).all()


def test_stable_ties_keep_detector_order():
    """Equal scores hold detector order: [0.5, 0.9, 0.9] with n_max=2 keeps the two
    0.9s as markers [1, 2] (stable sort), not reordered."""
    frame = extract_raw_frame(_frame([0.5, 0.9, 0.9]), 2, "syn", 0, set())
    assert frame.n_dets == 2
    assert _markers(frame.kps, 2) == [1, 2]


def test_empty_frame_all_nan_ndet_zero():
    """0 dets: every array all-NaN, detection count 0."""
    kps, bboxes, scores, kp_scores, n_dets = extract_raw_frame(_frame([]), 16, "syn", 0, set())
    assert n_dets == 0
    assert np.isnan(kps).all() and np.isnan(bboxes).all()
    assert np.isnan(scores).all() and np.isnan(kp_scores).all()


def test_partial_frame_fills_slot_zero():
    """1 det, wide n_max=16: slot 0 filled (marker 0), slots 1.. NaN, count 1."""
    frame = extract_raw_frame(_frame([0.7]), 16, "syn", 0, set())
    assert frame.n_dets == 1
    assert _markers(frame.kps, 1) == [0]
    assert np.isnan(frame.kps[1:]).all()
