"""Source extracts for issue #148; not a complete repository checkout.

Geometry predicates: src/courtkeynet/wrapper.py at 94bc9cc08808ca88cbc55749adf504ec257f0ee5.
Consensus arithmetic: src/courtkeynet/court_corners.py at the same revision.
Only comments/docstrings have been shortened; the executable statements are retained.
"""

from dataclasses import dataclass
import numpy as np

DEFAULT_AREA_BOUNDS = (0.01, 0.95)
CONSENSUS_FLAG_THRESHOLD_PX = 55.0

def _is_convex(corners_norm: np.ndarray) -> bool:
    edges = np.roll(corners_norm, -1, axis=0) - corners_norm
    next_edges = np.roll(edges, -1, axis=0)
    crosses = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]
    return bool((crosses >= 0).all() or (crosses <= 0).all())

def _shoelace_area(corners_norm: np.ndarray) -> float:
    x = corners_norm[:, 0]
    y = corners_norm[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

def _quadrants_ok(corners: np.ndarray) -> bool:
    cx, cy = corners.mean(axis=0)
    x = corners[:, 0]
    y = corners[:, 1]
    tl_ok = x[0] < cx and y[0] < cy
    tr_ok = x[1] > cx and y[1] < cy
    br_ok = x[2] > cx and y[2] > cy
    bl_ok = x[3] < cx and y[3] > cy
    return bool(tl_ok and tr_ok and br_ok and bl_ok)

def _geometry_flags(
    corners: np.ndarray,
    frame_wh: tuple[float, float],
    area_bounds: tuple[float, float] = DEFAULT_AREA_BOUNDS,
) -> tuple[str, ...]:
    flags: list[str] = []
    if not _is_convex(corners):
        flags.append("non_convex")
    lo, hi = area_bounds
    area_frac = _shoelace_area(corners) / (frame_wh[0] * frame_wh[1])
    if area_frac < lo or area_frac > hi:
        flags.append("bad_area")
    if not _quadrants_ok(corners):
        flags.append("bad_corner_order")
    return tuple(flags)

@dataclass(frozen=True)
class ConsensusRepair:
    consensus_quad: np.ndarray
    distances_px: np.ndarray
    flagged: np.ndarray
    repaired_quads: np.ndarray

def consensus_repair(quads: np.ndarray) -> ConsensusRepair:
    quads = np.asarray(quads, dtype=np.float64)
    if quads.ndim != 3 or quads.shape[1:] != (4, 2):
        raise ValueError(f"consensus_repair: expected (n_scenes, 4, 2) quads, got shape {quads.shape}")
    if quads.shape[0] == 0:
        raise ValueError("consensus_repair: at least one scene quad is required")
    consensus = np.median(quads, axis=0)
    per_corner_dist = np.linalg.norm(quads - consensus, axis=2)
    distances = per_corner_dist.max(axis=1)
    flagged = distances > CONSENSUS_FLAG_THRESHOLD_PX
    if flagged.sum() * 2 >= quads.shape[0]:
        raise ValueError(
            f"consensus_repair: {int(flagged.sum())} of {quads.shape[0]} scenes sit beyond "
            f"{CONSENSUS_FLAG_THRESHOLD_PX} px of the median; no trustworthy majority exists, "
            "keep the per-scene answers"
        )
    repaired_quads = np.where(flagged[:, None, None], consensus, quads)
    return ConsensusRepair(
        consensus_quad=consensus,
        distances_px=distances,
        flagged=flagged,
        repaired_quads=repaired_quads,
    )
