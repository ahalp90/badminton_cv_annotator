"""Time axis scoring (projective_seed.score_axes) and its peak memory at several batch sizes.

The detector scores axis hypotheses 256 at a time. This checks whether larger batches (F4) would
be faster on a CPU. The inputs are synthetic but shaped like one real direction: 32,768
hypotheses, the court's 5 or 6 lines along the axis, and 32 or 128 line groups (128 is the cap).
Each time is the best of five passes on one thread; peak memory is from tracemalloc for one call.

Usage, from the repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python \
    scratch/court_det_fix/court_detector_optimisation_handover/claude_evidence/axis_batch/axis_batch.py
"""
import platform
import sys
import time
import tracemalloc

import numpy as np

sys.path[:0] = ['.', 'src', 'scratch/court_det_fix/frozen_helpers_20260914/vp_pruning',
                'scratch/court_det_fix/next_steps_20260916/webui_seed/source']
import projective_seed  # pyrefly: ignore[missing-import]

from experiments.annotator.independent_court import detector

HYPOTHESES = 32768
BATCHES = (64, 256, 1024, 4096, 16384)

rng = np.random.default_rng(0)
basis = np.array([[1., 0.02, 0.], [0.01, 1., 0.], [1e-5, 2e-5, 1.]])
parameters = np.column_stack((rng.uniform(20, 80, HYPOTHESES), rng.uniform(-200, 800, HYPOTHESES)))
print(platform.processor() or platform.machine(), f'numpy {np.__version__}')
for axis, coordinates in ((0, detector.X_COORDS), (1, detector.Y_COORDS)):
    for groups in (32, 128):
        endpoints = rng.uniform(0, 960, (groups, 2, 2))
        print(f'axis {axis}: {len(coordinates)} court lines, {groups} line groups')
        baseline = None
        for batch in BATCHES:
            tracemalloc.start()
            projective_seed.score_axes(parameters[:batch], coordinates, endpoints, basis, axis)
            peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            best = np.inf
            for _ in range(5):
                started = time.perf_counter()
                for start in range(0, HYPOTHESES, batch):
                    projective_seed.score_axes(parameters[start:start + batch], coordinates, endpoints, basis, axis)
                best = min(best, time.perf_counter() - started)
            baseline = best if batch == 256 else baseline
            ratio = f'{best / baseline:4.2f}x batch 256' if baseline else ''
            print(f'  batch {batch:>6}: {best * 1e3:7.1f} ms, peak {peak / 2**20:6.1f} MiB per call  {ratio}')
