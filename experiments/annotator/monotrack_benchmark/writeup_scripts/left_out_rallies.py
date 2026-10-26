"""Facts about the ShuttleSet22 rallies left out of the test set, for the report's appendix.

The test set scores 3,327 of the ShuttleSet22 rallies in its 46 videos. The other 524 were left
out upstream because a hit in them is flagged ``flaw`` (or, once, out of time order). ``metrics.py``
ignores them in every condition. This script counts them, checks for raw validation rallies absent from scored labels, and
records how many discovery predictions they would add as extras if they were not ignored.

Usage: python left_out_rallies.py --data DIR --shots-master CSV
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Bound inline: ruff's E402 allows sys.path calls above imports, but not assignments.
sys.path.insert(0, str(REPO := Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(REPO / 'src'))

from experiments.annotator.heuristic_only_tune.data import load_original_videos
from experiments.annotator.monotrack_benchmark.benchmark_scripts.common import (
    load_manifest,
    read_json,
)
from experiments.annotator.monotrack_benchmark.writeup_scripts.metrics import (
    HEURISTIC,
    INPUTS,
    SEEDS,
    annotator_run,
    left_out_windows,
    test_left_out,
)

SHUTTLESET = REPO / 'data/shuttleset'


def validation_left_out(shots_master: Path) -> dict[str, list[dict]]:
    """Original ShuttleSet rallies in each validation video that shots_master.csv does not hold."""
    metadata = pd.read_csv(SHUTTLESET / 'video_metadata.csv').set_index('id')
    left_out = {}
    for video in load_original_videos(shots_master):
        if video.split != 'validation':
            continue
        video_id = int(video.identity.removeprefix('sset_'))
        held = {rally.identity for rally in video.rallies}
        rallies = []
        for path in sorted((SHUTTLESET / 'set' / metadata.loc[video_id, 'video']).glob('set*.csv')):
            labels = pd.read_csv(path)
            for rally, rows in labels.groupby('rally'):
                identity = f'{path.stem}:{rally}'
                if identity not in held:
                    rallies.append({'rally_id': identity, 'frames': tuple(sorted(int(frame) for frame in rows['frame_num']))})
        left_out[video.identity] = rallies
    return left_out


def extras_split(streams: dict[str, dict], left_out: dict[str, list[dict]]) -> dict[str, int]:
    """Predicted hits outside every scored window, split by whether a left-out rally's window holds them."""
    outside_scored = inside_left_out = 0
    for item in load_manifest(INPUTS)['items']:
        windows = left_out_windows(left_out[item['item_id']])
        for event in streams[item['item_id']]['events']:
            frame = event[0]
            if any(span['start_frame'] <= frame < span['end_frame'] for span in item['spans']):
                continue
            outside_scored += 1
            inside_left_out += any(start <= frame < end for start, end in windows)
    return {'outside_scored_windows': outside_scored, 'inside_left_out_windows': inside_left_out}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='metrics.py output folder')
    parser.add_argument('--annotator-runs', type=Path, required=True)
    parser.add_argument('--shots-master', type=Path, required=True)
    args = parser.parse_args()
    manifest = load_manifest(INPUTS)
    test = test_left_out()
    annotator = [extras_split({item['item_id']: read_json(annotator_run('discovery', '32_videos', seed, args.annotator_runs) / f'{item["item_id"]}.json.gz')
                               for item in manifest['items']}, test) for seed in SEEDS]
    trees_off = extras_split(read_json(HEURISTIC / 'test/predictions.json.gz')['streams'], test)
    validation = validation_left_out(args.shots_master)
    result = {
        'test_left_out_rallies': sum(len(rallies) for rallies in test.values()),
        'test_left_out_flaw_marked': sum(rally['flaw'] for rallies in test.values() for rally in rallies),
        'test_left_out_labelled_hits': sum(len(rally['frames']) for rallies in test.values() for rally in rallies),
        'validation_left_out_rallies': sum(len(rallies) for rallies in validation.values()),
        'annotator_discovery_extras': {
            'runs': annotator,
            'mean_outside_scored_windows': float(np.mean([run['outside_scored_windows'] for run in annotator])),
            'mean_inside_left_out_windows': float(np.mean([run['inside_left_out_windows'] for run in annotator])),
        },
        'trees_off_discovery_extras': trees_off,
    }
    (args.data / 'left_out_summary.json').write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
