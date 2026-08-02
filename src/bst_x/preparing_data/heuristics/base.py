"""Shared types and doubles-guard helpers for the heuristics package.

Each heuristic variant (``current``, ``sticky_anchor``, etc.) lives in its
own module under this package and exposes an ``apply`` function with the
signature:

    apply(raw: RawClip, ctx: ClipContext, **hyperparams) -> HeuristicOutput

Kept separate from ``__init__.py`` so variant modules can import the shared
types without triggering the package-level registry build.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd


class RawClip(NamedTuple):
    """Per-clip raw pose-extract output, as written by ``preparing_data.raw_extract``.

    Real detections in frame ``f`` occupy indices ``0 .. ndet[f] - 1`` along
    the detect axis; entries at and beyond ``ndet[f]`` are NaN-padded. Shapes:

    - ``kps``       ``(F, N_max, J, 2)``  float32
    - ``bboxes``    ``(F, N_max, 4)``     float32
    - ``scores``    ``(F, N_max)``        float32
    - ``kp_scores`` ``(F, N_max, J)``     float32
    - ``ndet``      ``(F,)``              int
    """
    kps: np.ndarray
    bboxes: np.ndarray
    scores: np.ndarray
    kp_scores: np.ndarray
    ndet: np.ndarray


# Per-clip raw file suffixes, one per RawClip field above (kps, bboxes, scores,
# kp_scores, ndet). raw_extract writes all five; apply_heuristic requires all
# five present. Shared here so the writer/reader contract has one home.
RAW_SUFFIXES = (
    "_raw_kps.npy",
    "_raw_bboxes.npy",
    "_raw_scores.npy",
    "_raw_kp_scores.npy",
    "_raw_ndet.npy",
)


@dataclass
class ClipContext:
    """Per-clip context needed to project pixel coords into court coords.

    ``all_court_info`` is a ``{vid: court_info}`` map as returned by
    ``shared.court.get_court_info``; ``res_df`` is a DataFrame
    indexed by video id with at least ``width`` and ``height`` columns.
    """
    vid: int
    all_court_info: dict
    res_df: pd.DataFrame


class HeuristicOutput(NamedTuple):
    """Per-clip output matching the existing pipeline's filtered schema.

    - ``pos``    ``(F, 2, 2)``   normalised court positions, slot order (Top, Bottom).
    - ``joints`` ``(F, 2, J, 2)`` bbox-diagonal-normalised keypoints.
    - ``failed`` ``(F,)`` bool   True where the frame was zeroed.
    - ``overcount`` ``(F,)`` bool   True where >2 standing candidates projected within the doubles count margin (doubles evidence).
    """
    pos: np.ndarray
    joints: np.ndarray
    failed: np.ndarray
    overcount: np.ndarray


# COCO keypoint indices used by the sitting test.
SHOULDER_L, SHOULDER_R = 5, 6
HIP_L, HIP_R = 11, 12
KNEE_L, KNEE_R = 13, 14

# D26 doubles-guard head-count settings, measured on all 32,203 known-singles
# ShuttleSet clips. Margin 0.05 behaves identically to 0.0, so the headroom is
# free; the sitting exemption zeroes the residual seated-official false flags.
# sticky_anchor's hyperparams default to these; the detect path reads them directly.
DOUBLES_COUNT_MARGIN = 0.05
SITTING_THRESHOLD = -0.3


def is_sitting(kps: np.ndarray, sitting_threshold: float) -> np.ndarray:
    """Body-frame sitting test over a batch of poses.

    Projects the knee-offset-from-hip onto the hip-to-shoulder axis. A
    standing / airborne player has knees in the body-down direction
    (ratio around -0.7 to -0.9); a sitting person has knees roughly
    perpendicular to the body axis (ratio near 0). Returns True where the
    ratio exceeds ``sitting_threshold``.

    ``kps`` is (m, J, 2) pixel coords; returns a (m,) bool mask.
    """
    sh = (kps[:, SHOULDER_L] + kps[:, SHOULDER_R]) / 2  # (m, 2)
    hp = (kps[:, HIP_L] + kps[:, HIP_R]) / 2
    kn = (kps[:, KNEE_L] + kps[:, KNEE_R]) / 2
    body_up = sh - hp
    knee_vec = kn - hp
    torso_len_sq = (body_up * body_up).sum(axis=1)  # (m,)
    degenerate = torso_len_sq < 1e-6  # degenerate pose; defer to anchor distance
    ratio = (knee_vec * body_up).sum(axis=1) / np.where(degenerate, 1.0, torso_len_sq)
    return (ratio > sitting_threshold) & ~degenerate


def count_standing_in_court(
    pos: np.ndarray,
    sitting: np.ndarray,
    margin: float,
) -> int:
    """The doubles-guard head count: detections whose normalised court position
    lands within ``margin`` of the court on both axes AND that are not sitting.

    Seated officials 1-2 m behind the lines sit inside broad margins and are
    persistent, so windowing can't save them (D26). The exemption can only push
    the count down, away from a false doubles flag, and real doubles keeps
    tripping the guard because four players can't sit out more than half a rally.

    ``pos`` is (m, 2) normalised court coords; ``sitting`` is the (m,) bool mask
    from ``is_sitting``, index-aligned with ``pos``.
    """
    in_margin = ((pos >= -margin) & (pos <= 1 + margin)).all(axis=1)
    return int((in_margin & ~sitting).sum())
