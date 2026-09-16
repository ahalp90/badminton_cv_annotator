"""Measure distance from the camera cutoff for the nine frozen automatic inputs."""

from itertools import permutations
from pathlib import Path

import numpy as np
from run_automatic import (
    CAMERA_ERROR_LIMIT,
    CAMERA_ROUNDING_MARGIN,
    camera_direction_bound,
)
from run_diagnosis import read, write

CASE_IDS = [
    'gxBQ_window_00_frame_5', 'gxBQ_window_00_frame_0',
    'am2_window_00_frame_150', 'am2_window_01_frame_28019', 'am3_window_00_frame_0',
    'shuttleset_03_scene_0017', 'shuttleset_03_scene_0019', 'shuttleset_03_scene_0016', 'shuttleset_21_scene_0020',
]


def main() -> None:
    records = []
    for case_id in CASE_IDS:
        saved = read(Path('vp_pruning_20260914/coverage/results') / f'{case_id}.json.gz')
        points = np.asarray(saved['estimator']['points_working'])
        pairs, bounds = [], []
        for pair_id, pencil_ids in enumerate(permutations(range(len(points)), 2)):
            bound = camera_direction_bound(points[list(pencil_ids)], saved['working_size'])
            bounds.append(bound)
            if bound <= CAMERA_ERROR_LIMIT + CAMERA_ROUNDING_MARGIN:
                pairs.append(pair_id)
        distance = min(abs(bound - CAMERA_ERROR_LIMIT) for bound in bounds)
        records.append({'case_id': case_id, 'total': len(points) * (len(points) - 1),
                        'admitted_pairs': pairs, 'bounds': bounds, 'closest_threshold_distance': distance})
        print(case_id, len(pairs), 'admitted pairs; closest bound to .1:', distance, flush=True)
    write(Path('automatic_axes_20260914/bound_counts.json.gz'), records)


if __name__ == '__main__':
    main()
