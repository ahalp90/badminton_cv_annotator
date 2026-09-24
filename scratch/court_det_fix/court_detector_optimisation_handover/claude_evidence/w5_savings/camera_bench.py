"""Check and time an x/y/w form of vector_camera_errors against the current form (throwaway).

The candidate keeps each court axis's three homogeneous components as separate (courts, focal
lengths) arrays instead of one (courts, focal lengths, 3, 2) array. Every candidate must match
the current output byte for byte, on real saved batches and on random and degenerate homographies.

Usage: camera_bench.py CAMERA_BATCHES_NPZ
"""

import sys
import time
from pathlib import Path

import numpy as np

# This file sits in court_det_fix/<handover>/claude_evidence/w5_savings/.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "w5_holistic"))
from line_template_source import vector_camera_errors  # pyrefly: ignore[missing-import]


def split_camera_errors(homographies: np.ndarray, size: tuple[int, int], grouping: str) -> np.ndarray:
    width, height = size
    focals = np.geomspace(0.4 * width, 4.0 * width, 200)
    # One row per court, one column per focal length. The image-plane parts shift by the principal
    # point and then divide by the focal length, as in the current form; the w parts stay as they are.
    x_along = (homographies[:, 0, 0] - width / 2.0 * homographies[:, 2, 0])[:, None] / focals
    y_along = (homographies[:, 1, 0] - height / 2.0 * homographies[:, 2, 0])[:, None] / focals
    x_across = (homographies[:, 0, 1] - width / 2.0 * homographies[:, 2, 1])[:, None] / focals
    y_across = (homographies[:, 1, 1] - height / 2.0 * homographies[:, 2, 1])[:, None] / focals
    w_along, w_across = homographies[:, 2, 0, None], homographies[:, 2, 1, None]
    if grouping == "left":
        along = np.sqrt(np.square(x_along) + np.square(y_along) + np.square(w_along))
        across = np.sqrt(np.square(x_across) + np.square(y_across) + np.square(w_across))
        dot = x_along * x_across + y_along * y_across + w_along * w_across
    else:
        along = np.sqrt(np.square(x_along) + (np.square(y_along) + np.square(w_along)))
        across = np.sqrt(np.square(x_across) + (np.square(y_across) + np.square(w_across)))
        dot = x_along * x_across + (y_along * y_across + w_along * w_across)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = dot / (along * across)
        ratio = np.log(along / across)
        errors = np.hypot(cosine, ratio)
    usable = np.isfinite(errors) & np.isfinite(along) & np.isfinite(across) & (along > 0) & (across > 0)
    return np.where(usable, errors, np.inf).min(axis=1)


def same_bytes(left: np.ndarray, right: np.ndarray) -> bool:
    return (left.dtype, left.shape, left.tobytes()) == (right.dtype, right.shape, right.tobytes())


saved = np.load(sys.argv[1])
size = tuple(int(value) for value in saved["working_size"])
batches = [saved[name] for name in sorted(saved.files, key=lambda name: (len(name), name))
           if name.startswith("homographies_")]
random = np.random.default_rng(20260924)
extra = []
for trial in range(30):
    homographies = batches[trial % len(batches)].copy()
    homographies *= random.uniform(0.5, 2.0, (len(homographies), 1, 1))
    homographies[:, :2] += random.normal(0, 5, (len(homographies), 2, 3))
    if trial % 3 == 0:
        homographies[:5, 2, :2] = 0.0  # Affine courts: w is exactly zero.
        homographies[5:8, :, 0] = 0.0  # A collapsed court axis: zero norm.
        homographies[8, 0, 0] = np.inf
        homographies[9, 1, 1] = np.nan
        homographies[10, 2, 0] = -0.0
    extra.append(homographies)

for grouping in ("left", "right"):
    matches = [same_bytes(split_camera_errors(batch, size, grouping), vector_camera_errors(batch, size))
               for batch in batches + extra]
    print(f"{grouping} grouping: {sum(matches)} of {len(matches)} batches byte-identical")

for name, function in (("current", lambda batch: vector_camera_errors(batch, size)),
                       ("x/y/w left", lambda batch: split_camera_errors(batch, size, "left"))):
    started = time.perf_counter()
    for batch in batches:
        function(batch)
    elapsed = time.perf_counter() - started
    courts = sum(len(batch) for batch in batches)
    print(f"{name:12s} {elapsed:6.2f} s for {len(batches)} batches ({courts} courts)")
