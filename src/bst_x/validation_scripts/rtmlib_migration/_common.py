"""Shared helpers for the rtmlib migration gates.

One place for the committed-raw loader, the greedy IoU box matcher, and the
IoU-matched per-keypoint L2 used by the keypoint-value gate (G1), the deployed
parity gate (G6) and the GPU parity gate (G7). Keeps the pixel-comparison logic
identical across gates so their verdicts are comparable.

Paths default to the local pool and are overridable by env so the same scripts
run on Bourbaki:
  RTMLIB_GATE_CLIPS  mp4 root (recursively searched by stem)
  RTMLIB_GATE_RAW    committed mmpose raw dir
  RTMLIB_GATE_CLEAN  committed clean (sticky_anchor) dir
"""
from __future__ import annotations

import os
from dataclasses import fields
from pathlib import Path
from typing import NamedTuple

import numpy as np

from pipeline.config import COCO_N_JOINTS

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
CLEAN = Path(os.environ.get(
    "RTMLIB_GATE_CLEAN",
    "/srv/mergerfs/main_pool/320_cosc594_data-bourbaki/ShuttleSet_keypoints_clean_sticky_anchor",
))


class RawArrays(NamedTuple):
    """The five-array raw schema for one clip (N = N_MAX slots).

    Same shape/dtype whether loaded from the committed mmpose raw
    (``load_mmpose_raw``) or assembled from the rtmlib adapter
    (``assemble_raw_clip``), so a gate can compare like with like.

    ``J`` in the shape comments below is the COCO joint count (``COCO_N_JOINTS``, 17).
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
    int8 ndet. ``tests/test_raw_schema.py`` covers the truncation/empty/partial edges
    of ``extract_raw_frame`` directly.

    :param frames: rtmlib ``FrameDetections`` per frame, decode order.
    :param n_max: per-frame detection cap (16 in production).
    :return: the five-array ``RawArrays`` (F = len(frames)).
    """
    # Lazy: keeps _common (hence G1/G2/G4) importable without pulling raw_extract's
    # pipeline deps; only the deployed-parity gates (G6/G7/G8) call this.
    from preparing_data.raw_extract import extract_raw_frame

    F = len(frames)
    kps = np.empty((F, n_max, COCO_N_JOINTS, 2), dtype=np.float32)
    bboxes = np.empty((F, n_max, 4), dtype=np.float32)
    scores = np.empty((F, n_max), dtype=np.float32)
    kp_scores = np.empty((F, n_max, COCO_N_JOINTS), dtype=np.float32)
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
    extremities legitimately differ between the shipped RTMPose-L and the
    mmpose-era RTMPose-M (both body7).

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
        return np.empty((0, COCO_N_JOINTS), np.float32), np.empty((0, COCO_N_JOINTS), bool)
    return np.stack(l2_rows), np.stack(conf_rows)


def court_setup(print_versions: bool = False) -> tuple:
    """One-time court/resolution context + sticky_anchor default hyperparams.

    Shared by the deployed-parity gates (G5/G6/G7/G8). The heavy imports stay
    deferred so importing ``_common`` for the pure keypoint gates (G1/G2/G4) does
    not pull the pipeline/heuristic stack.

    :param print_versions: G5 alone prints the numpy/pandas banner (its byte-eq
        precondition is env-sensitive, so it records the versions it ran under);
        the other gates leave it off. Kept as a flag rather than forking the
        helper so both copies stay one function.
    :return: ``(res_df, court, params, RawClip, ClipContext, sticky_apply)``; the
        tuple every gate unpacks for a sticky_anchor call.
    """
    import pandas as pd
    from pipeline.config import RESOLUTION_CSV_PATH, SET_INFO_DIR
    from shared.court import get_court_info

    from preparing_data.heuristics.base import ClipContext, RawClip
    from preparing_data.heuristics.sticky_anchor import StickyAnchorParams
    from preparing_data.heuristics.sticky_anchor import apply as sticky_apply

    if print_versions:
        print(f"numpy {np.__version__} | pandas {pd.__version__}")
    res_df = pd.read_csv(RESOLUTION_CSV_PATH).set_index("id")
    homo_df = pd.read_csv(str(SET_INFO_DIR / "homography.csv")).set_index("id")
    court = {vid: get_court_info(homo_df, vid) for vid in res_df.index}
    params = {f.name: f.default for f in fields(StickyAnchorParams)}
    return res_df, court, params, RawClip, ClipContext, sticky_apply


class DeployedParity(NamedTuple):
    """Shared per-clip deployed-parity quantities (G6 and G8 build on these).

    ``out_failed`` and ``ref_failed`` are the raw masks, not just their lengths,
    because G8 reports failed-rate means over them and G6 reads their lengths.
    """
    out_failed: np.ndarray   # (F,) sticky_anchor failed mask over the adapter run
    ref_failed: np.ndarray   # (F,) committed clean failed mask
    raw_arr: RawArrays       # assembled adapter raw (for the ndet<2 dropped-player check)
    fmatch: float            # failed-frame agreement over the shared prefix
    rt_only_fail: int        # frames rtmlib fails that mmpose kept
    mm_only_fail: int        # the reverse (near-zero in practice)
    nb: int                  # frames both extractors keep (both-success)
    pos_med: float           # median |Δpos| over both-success frames (NaN if none)
    jnt_med: float           # median |Δjoints| over both-success frames (NaN if none)


def deployed_parity(frames: list, stem: str, setup: tuple) -> DeployedParity:
    """Score the adapter's detections through sticky_anchor against the clean.

    The per-clip parity body shared by the CPU deployed-parity gate (G6) and the
    GPU parity gate (G8); each wraps its own report columns around the return. It
    assembles the raw arrays exactly as ``raw_extract`` would, runs the unchanged
    ``sticky_anchor``, and compares the deployed output to the committed clean.

    :param frames: rtmlib ``FrameDetections`` per frame, decode order.
    :param stem: clip stem (``<vid>_...``); drives the vid and the clean lookup.
    :param setup: the ``court_setup`` tuple.
    :return: the shared parity quantities; see ``DeployedParity``.
    """
    res_df, court, params, RawClip, ClipContext, sticky_apply = setup
    raw_arr = assemble_raw_clip(frames)
    raw = RawClip(kps=raw_arr.kps, bboxes=raw_arr.bboxes, scores=raw_arr.bbox_scores,
                  kp_scores=raw_arr.kp_scores, ndet=raw_arr.ndet)
    ctx = ClipContext(vid=int(stem.split("_", 1)[0]), all_court_info=court, res_df=res_df)
    out = sticky_apply(raw, ctx, **params)

    ref_pos = np.load(CLEAN / f"{stem}_pos.npy")
    ref_joints = np.load(CLEAN / f"{stem}_joints.npy")
    ref_failed = np.load(CLEAN / f"{stem}_failed.npy")

    # sticky_anchor emits one row per input frame, so the shared prefix is the min
    # of the two failed-mask lengths. This equals G6's len(out.failed)/len(ref_failed)
    # exactly and G8's len(frames)/mm.kps.shape[0] by that same one-row invariant.
    F = min(len(out.failed), len(ref_failed))
    rt_f, mm_f = out.failed[:F], ref_failed[:F]
    fmatch = float((rt_f == mm_f).mean())
    # Directional failed-frame split: surface both directions separately, since
    # a one-directional excess is data loss the mean agreement hides.
    rt_only_fail = int((rt_f & ~mm_f).sum())  # rtmlib zeroes a frame mmpose kept
    mm_only_fail = int((~rt_f & mm_f).sum())  # the reverse (near-zero in practice)
    both = (~rt_f) & (~mm_f)
    nb = int(both.sum())
    if nb:
        pos_med = float(np.median(np.abs(out.pos[:F][both] - ref_pos[:F][both])))
        jnt_med = float(np.median(np.abs(out.joints[:F][both] - ref_joints[:F][both])))
    else:
        pos_med = jnt_med = float("nan")
    return DeployedParity(
        out_failed=out.failed, ref_failed=ref_failed, raw_arr=raw_arr,
        fmatch=fmatch, rt_only_fail=rt_only_fail, mm_only_fail=mm_only_fail,
        nb=nb, pos_med=pos_med, jnt_med=jnt_med,
    )
