"""Analytical lower bounds on corner error for a fixed vanishing-point set.

These bounds require target corner coordinates. They are audit diagnostics, not
label-free selector scores, new fitted courts, or usability thresholds.
"""
from __future__ import annotations
from itertools import permutations
import numpy as np
from numpy.typing import ArrayLike, NDArray


def edge_bound(a: ArrayLike, b: ArrayLike, points: ArrayLike) -> NDArray[np.float64]:
    """Minimax normal displacement of endpoints onto a line through each VP.

    a, b: distinct finite 2-D endpoints in a common Euclidean image coordinate system.
    points: nonzero homogeneous 3-vectors, shape (N,3) or (3,).
    The answer is in the endpoint coordinate units. Finite and infinite VPs and
    arbitrary homogeneous scale/sign are supported.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    p = np.atleast_2d(np.asarray(points, dtype=float))
    if a.shape != (2,) or b.shape != (2,) or p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("Expected two endpoints of shape (2,) and VPs of shape (N,3).")
    if not all(np.isfinite(x).all() for x in (a,b,p)) or np.array_equal(a,b):
        raise ValueError("Endpoints must be distinct; all inputs must be finite.")
    scale = np.max(np.abs(p), axis=1)
    if np.any(scale == 0):
        raise ValueError("The zero homogeneous vector is not a vanishing point.")
    p = p / scale[:, None]
    ell = np.cross(np.r_[a,1.], np.r_[b,1.])
    ra = p[:, :2] - p[:, 2:] * a
    rb = p[:, :2] - p[:, 2:] * b
    denominator = np.maximum(np.linalg.norm(ra+rb,axis=1), np.linalg.norm(ra-rb,axis=1))
    if np.any(denominator == 0):
        raise ValueError("Degenerate endpoint/VP configuration.")
    return np.abs(p @ ell) / denominator


def axis_bounds(corners: ArrayLike, points: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Bounds for x and y VP roles, corners ordered FL, FR, NR, NL."""
    c = np.asarray(corners, dtype=float)
    if c.shape != (4,2):
        raise ValueError("Expected four 2-D corners in FL, FR, NR, NL order.")
    x = np.maximum(edge_bound(c[0],c[1],points), edge_bound(c[3],c[2],points))
    y = np.maximum(edge_bound(c[0],c[3],points), edge_bound(c[1],c[2],points))
    return x,y


def set_bound(corners: ArrayLike, points: ArrayLike) -> tuple[float, tuple[int,int]]:
    """Lower bound for any ordered distinct pair in the supplied VP set.

    A minimizing index pair is returned for reproducibility; it is not necessarily
    unique, does not certify attainability, and is not a fitted court.
    """
    p = np.atleast_2d(np.asarray(points, dtype=float))
    if len(p) < 2:
        raise ValueError("At least two VPs are required.")
    lx,ly = axis_bounds(corners,p)
    pairs = np.asarray(list(permutations(range(len(p)),2)),dtype=int)
    values = np.maximum(lx[pairs[:,0]],ly[pairs[:,1]])
    i = int(np.argmin(values))
    return float(values[i]),(int(pairs[i,0]),int(pairs[i,1]))
