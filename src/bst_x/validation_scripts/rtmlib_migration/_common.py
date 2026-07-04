"""Shared helpers for the rtmlib migration gates.

One place for the committed-raw loader, the greedy IoU box matcher, and the
IoU-matched per-keypoint L2 used by the keypoint-value gate (G1), the deployed
parity gate (G6) and the GPU parity gate (G7). Keeps the pixel-comparison logic
identical across gates so their verdicts are comparable.

Paths default to the local pool and are overridable by env so the same scripts
run on Bourbaki:
  RTMLIB_GATE_CLIPS  mp4 root (recursively searched by stem)
  RTMLIB_GATE_RAW    committed mmpose raw dir
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import numpy as np

J = 17            # COCO keypoints
N_MAX = 16        # raw_extract per-frame detection cap
CONF_THR = 0.5    # a joint is "confident" when both models score it above this
IOU_MATCH_MIN = 0.5  # min IoU to pair an rtmlib box with an mmpose box

# COCO-17 index groups for the reference-free order-sanity backstop.
HEAD = (0, 1, 2, 3, 4)   # nose, eyes, ears
KNEES = (13, 14)
ANKLES = (15, 16)

CLIPS = Path(os.environ.get(
    "RTMLIB_GATE_CLIPS",
    "/srv/mergerfs/main_pool/320_cosc594_data-bourbaki/ShuttleSet/clips",
))
RAW = Path(os.environ.get(
    "RTMLIB_GATE_RAW",
    "/srv/mergerfs/main_pool/320_cosc594_data-bourbaki/ShuttleSet_keypoints_raw",
))


class RawArrays(NamedTuple):
    """The five-array raw schema for one clip (N = N_MAX slots).

    Same shape/dtype whether loaded from the committed mmpose raw
    (``load_mmpose_raw``) or assembled from the rtmlib adapter
    (``assemble_raw_clip``), so a gate can compare like with like.
    """
    kps: np.ndarray          # (F, N, J, 2) float32; NaN-padded past ndet
    bboxes: np.ndarray       # (F, N, 4) float32; xyxy
    bbox_scores: np.ndarray  # (F, N) float32
    kp_scores: np.ndarray    # (F, N, J) float32
    ndet: np.ndarray         # (F,) int8; real detections per frame


def find_clip(stem: str, clips_root: Path = CLIPS) -> Path | None:
    """Locate ``<stem>.mp4`` anywhere under ``clips_root`` (train/val/test splits)."""
    return next(iter(clips_root.glob(f"**/{stem}.mp4")), None)


def load_mmpose_raw(stem: str, raw_root: Path = RAW) -> RawArrays:
    """Load one clip's committed five-array mmpose raw."""
    return RawArrays(
        kps=np.load(raw_root / f"{stem}_raw_kps.npy"),
        bboxes=np.load(raw_root / f"{stem}_raw_bboxes.npy"),
        bbox_scores=np.load(raw_root / f"{stem}_raw_scores.npy"),
        kp_scores=np.load(raw_root / f"{stem}_raw_kp_scores.npy"),
        ndet=np.load(raw_root / f"{stem}_raw_ndet.npy"),
    )


def assemble_raw_clip(frames: list, n_max: int = N_MAX) -> RawArrays:
    """Pack per-frame adapter detections into the five NaN-padded slot arrays.

    Calls the SHIPPED per-frame assembly ``raw_extract.extract_raw_frame`` (Batch
    2) rather than a replica, so the deployed-parity gates cannot drift from what
    ``raw_extract`` actually writes: real detections in detector order in slots
    ``0..n-1``; top-``n_max`` by descending ``bbox_score`` on overflow; NaN pad;
    int8 ndet. ``gate_raw_schema.py`` covers the truncation/empty/partial edges of
    ``extract_raw_frame`` directly.

    :param frames: rtmlib ``FrameDetections`` per frame, decode order.
    :param n_max: per-frame detection cap (16 in production).
    :return: the five-array ``RawArrays`` (F = len(frames)).
    """
    # Lazy: keeps _common (hence G1/G2/G4) importable without pulling raw_extract's
    # pipeline deps; only the deployed-parity gates (G6/G7/G8) call this.
    from preparing_data.raw_extract import extract_raw_frame

    F = len(frames)
    kps = np.empty((F, n_max, J, 2), dtype=np.float32)
    bboxes = np.empty((F, n_max, 4), dtype=np.float32)
    scores = np.empty((F, n_max), dtype=np.float32)
    kp_scores = np.empty((F, n_max, J), dtype=np.float32)
    ndet = np.empty(F, dtype=np.int8)
    warned: set[str] = set()
    for f, fr in enumerate(frames):
        k, b, s, ks, n = extract_raw_frame(fr, n_max, "gate", f, warned)
        kps[f], bboxes[f], scores[f], kp_scores[f], ndet[f] = k, b, s, ks, n
    return RawArrays(kps=kps, bboxes=bboxes, bbox_scores=scores, kp_scores=kp_scores, ndet=ndet)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two xyxy boxes."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_dets(
    mm_bboxes: np.ndarray,
    rt_bboxes: np.ndarray,
    min_iou: float = IOU_MATCH_MIN,
) -> list[tuple[int, int]]:
    """Greedily pair mmpose boxes to rtmlib boxes by IoU (each used once).

    :param mm_bboxes: (a, 4) mmpose xyxy boxes (real detections only).
    :param rt_bboxes: (b, 4) rtmlib xyxy boxes.
    :param min_iou: a pair is kept only if its IoU exceeds this.
    :return: list of ``(mm_index, rt_index)`` matched pairs.
    """
    pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    for i in range(len(mm_bboxes)):
        best_j, best_iou = -1, min_iou
        for j in range(len(rt_bboxes)):
            if j in used:
                continue
            v = iou_xyxy(mm_bboxes[i], rt_bboxes[j])
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0:
            used.add(best_j)
            pairs.append((i, best_j))
    return pairs


def matched_kp_l2(
    mm: RawArrays,
    frames: list,  # list[rtmlib_pose.FrameDetections]
) -> tuple[np.ndarray, np.ndarray]:
    """Per-keypoint pixel L2 over IoU-matched people, mmpose raw vs rtmlib.

    Matches each frame's real mmpose detections to the rtmlib detections by IoU,
    then stacks the per-joint Euclidean distance and a "confident in both models"
    mask. The mask lets a gate separate coordinate correctness (all joints) from
    model agreement on the signal-bearing joints (confident only): occluded
    extremities legitimately differ between body7 and the old RTMPose-L.

    :param mm: committed mmpose raw for the clip.
    :param frames: rtmlib ``FrameDetections`` per frame, decode order.
    :return: ``(l2, conf)`` each ``(matched_people, J)``; ``l2`` in pixels,
        ``conf`` boolean. Empty ``(0, J)`` arrays if nothing matched.
    """
    F = min(len(frames), mm.kps.shape[0])
    l2_rows: list[np.ndarray] = []
    conf_rows: list[np.ndarray] = []
    for f in range(F):
        rt = frames[f]
        n = int(mm.ndet[f])
        for i, j in match_dets(mm.bboxes[f, :n], rt.bboxes):
            l2_rows.append(np.linalg.norm(mm.kps[f, i] - rt.keypoints[j], axis=-1))  # (J,)
            conf_rows.append((mm.kp_scores[f, i] > CONF_THR) & (rt.kp_scores[j] > CONF_THR))
    if not l2_rows:
        return np.empty((0, J), np.float32), np.empty((0, J), bool)
    return np.stack(l2_rows), np.stack(conf_rows)
