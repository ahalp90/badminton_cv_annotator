"""Count GT contacts represented by positive candidate frames in saved training inputs.

Run from the checkout root with PYTHONPATH=src:. and --training RUN/training.
Writes a small JSON summary to stdout; it does not refit or modify saved inputs.
"""
import argparse
import gzip
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from annotator.config import ScalingKind
from annotator.contacts.features import REGION_FIELDS
from annotator.training.contact import (
    POSITIVE_RADIUS_AT_30_FPS,
    nearest_contact_distances,
)


def measure(arguments: tuple[Path, str, str]) -> dict[str, Any]:
    training, candidate, identity = arguments
    with threadpool_limits(limits=1):
        prepared = joblib.load(training / 'prepared' / candidate / 'videos' / f'{identity}.joblib')
        assert prepared.video.split == 'train'
        frames = prepared.features.rows['frame']
        assert np.all(np.diff(frames) >= 0)
        candidate_mask = np.zeros(len(frames), dtype=bool)
        for field in REGION_FIELDS:
            candidate_mask |= prepared.features.rows[field].astype(bool)
        candidates = frames[candidate_mask]
        labels = np.array(sorted(frame for rally in prepared.video.rallies for frame in rally.frames))
        radius = int(ScalingKind.FRAME_COUNT.scale(POSITIVE_RADIUS_AT_30_FPS, prepared.video.fps))
        covered = nearest_contact_distances(labels, candidates) <= radius
        available = nearest_contact_distances(labels, frames) <= radius
        positives = nearest_contact_distances(candidates, labels) <= radius
        return {'video': identity, 'gt_contacts': len(labels), 'covered_contacts': int(covered.sum()),
                'outside_feature_search': int((~available).sum()),
                'outside_candidate_regions': int((available & ~covered).sum()),
                'positive_rows': int(positives.sum()), 'candidate_rows': len(candidates)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--candidate', choices=('raw_baseline', 'body_window_36'), default='raw_baseline')
    parser.add_argument('--workers', type=int, choices=range(1, 29), default=8, metavar='1..28')
    args = parser.parse_args()
    with gzip.open(args.training / 'contacts' / args.candidate / 'seed-0/contacts/v_32.json.gz', 'rt') as stream:
        fit = json.load(stream)
    jobs = [(args.training, args.candidate, identity) for identity in fit['identity']['sampling_order']]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('fork')) as executor:
        rows = list(executor.map(measure, jobs))
    totals = {key: sum(row[key] for row in rows) for key in rows[0] if key != 'video'}
    assert totals['positive_rows'] == fit['positive_rows']
    assert (totals['covered_contacts'] + totals['outside_feature_search'] + totals['outside_candidate_regions']
            == totals['gt_contacts'])
    print(json.dumps({'candidate': args.candidate, 'training_videos': len(rows),
                      'positive_radius_at_30_fps': POSITIVE_RADIUS_AT_30_FPS, 'totals': totals, 'per_video': rows}))


if __name__ == '__main__':
    main()
