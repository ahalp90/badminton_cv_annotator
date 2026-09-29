"""float16 vs float32 vs float64 camera errors on real batches: timing, error and gate flips (throwaway).

Usage: precision_probe.py ../w5_savings/camera_batches.npz (batches saved from a local D17 run; not committed).
"""

import sys
import time
from pathlib import Path

import numpy as np

# This file sits in scratch/court_det_fix/court_detector_optimisation_handover/claude_evidence/precision/.
REPO = Path(__file__).resolve().parents[5]
for folder in (REPO, REPO / "src", REPO / "scratch" / "court_det_fix" / "w5_holistic"):
    sys.path.insert(0, str(folder))
from line_template_source import CAMERA_LIMIT, CAMERA_RECHECK_MARGIN, vector_camera_errors  # pyrefly: ignore[missing-import]


def typed_errors(homographies: np.ndarray, size: tuple[int, int], dtype) -> np.ndarray:
    """The implemented form, computed throughout in the given float type."""
    homographies = homographies.astype(dtype)
    image_width, image_height = (dtype(value) for value in size)
    focals = np.geomspace(0.4 * image_width, 4.0 * image_width, 200, dtype=dtype)
    half = dtype(2.0)
    width_x = (homographies[:, 0, 0] - image_width / half * homographies[:, 2, 0])[:, None] / focals
    width_y = (homographies[:, 1, 0] - image_height / half * homographies[:, 2, 0])[:, None] / focals
    length_x = (homographies[:, 0, 1] - image_width / half * homographies[:, 2, 1])[:, None] / focals
    length_y = (homographies[:, 1, 1] - image_height / half * homographies[:, 2, 1])[:, None] / focals
    width_w, length_w = homographies[:, 2, 0, None], homographies[:, 2, 1, None]
    width_norm = np.sqrt(np.square(width_x) + np.square(width_y) + np.square(width_w))
    length_norm = np.sqrt(np.square(length_x) + np.square(length_y) + np.square(length_w))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        cosine = (width_x * length_x + width_y * length_y + width_w * length_w) / (width_norm * length_norm)
        errors = np.hypot(cosine, np.log(width_norm / length_norm))
    usable = (np.isfinite(errors) & np.isfinite(width_norm) & np.isfinite(length_norm)
              & (width_norm > 0) & (length_norm > 0))
    return np.where(usable, errors, np.inf).min(axis=1)


saved = np.load(sys.argv[1])
size = tuple(int(value) for value in saved["working_size"])
batches = [saved[name] for name in saved.files if name.startswith("homographies_")]
every_entry = np.concatenate([batch.reshape(-1) for batch in batches])
print(f"homography entries: max |value| {np.abs(every_entry).max():.3g}, min nonzero |value| "
      f"{np.abs(every_entry[every_entry != 0]).min():.3g}; courts {sum(len(batch) for batch in batches)}")

reference = np.concatenate([vector_camera_errors(batch, size) for batch in batches])
reference_passes = reference <= CAMERA_LIMIT
for dtype in (np.float32, np.float16):
    reduced = np.concatenate([typed_errors(batch, size, dtype) for batch in batches]).astype(np.float64)
    both_finite = np.isfinite(reference) & np.isfinite(reduced)
    difference = np.abs(reduced[both_finite] - reference[both_finite])
    flips = reference_passes != (reduced <= CAMERA_LIMIT)
    beyond_recheck = flips & (np.abs(reference - CAMERA_LIMIT) > CAMERA_RECHECK_MARGIN)
    print(f"{dtype.__name__}: max difference {difference.max():.2e}, median {np.median(difference):.2e}; "
          f"finite-pattern mismatches {np.count_nonzero(np.isfinite(reduced) != np.isfinite(reference))}; "
          f"gate flips {flips.sum()} (passes {reference_passes.sum()}), beyond the recheck margin {beyond_recheck.sum()}")

timings = {"float64": lambda batch: vector_camera_errors(batch, size),
           "float32": lambda batch: typed_errors(batch, size, np.float32),
           "float16": lambda batch: typed_errors(batch, size, np.float16)}
for name, function in timings.items():
    best = []
    for _ in range(3):
        started = time.perf_counter()
        for batch in batches:
            function(batch)
        best.append(time.perf_counter() - started)
    print(f"{name} best of 3: {min(best):.3f} s")
