"""Measure validation coverage with and without rallies containing ``flaw == 1`` labels."""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(REPO := Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(REPO / 'src'))

from annotator.sequence.contacts import scale_frames, stream_from_payload
from annotator.training.sequences import DEFAULT_RALLY_BOUNDARY_POLICY
from experiments.annotator.heuristic_only_tune.data import (
    SearchVideo,
    load_original_videos,
)
from experiments.annotator.monotrack_benchmark.benchmark_scripts.common import (
    read_json,
    write_json,
)
from experiments.annotator.monotrack_benchmark.writeup_scripts.metrics import (
    IGNORE_MARGIN,
    SEEDS,
    WINDOW_AFTER,
    WINDOW_BEFORE,
    ignore_left_out,
    judge_found_rallies,
)

SHUTTLESET = REPO / 'data/shuttleset'
TOLERANCE = 10


def flaw_labels(
    videos: Sequence[SearchVideo],
) -> tuple[dict[str, set[str]], dict[str, list[tuple[int, int]]], dict[str, int]]:
    """Join exact ``flaw == 1`` rows to loaded rally labels and make scaled windows."""
    metadata = pd.read_csv(SHUTTLESET / 'video_metadata.csv').set_index('id')
    flagged_rallies, windows = {}, {}
    flagged_hits = flagged_frames = missing_frames = missing_rallies = extra_rallies = 0
    for video in videos:
        video_id = int(video.identity.removeprefix('sset_'))
        folder = SHUTTLESET / 'set' / metadata.loc[video_id, 'video']
        held = {rally.identity: set(rally.frames) for rally in video.rallies}
        flagged_rallies[video.identity] = set()
        windows[video.identity] = []
        raw_identities = set()
        before, after = scale_frames(WINDOW_BEFORE, video.fps), scale_frames(WINDOW_AFTER, video.fps)
        for path in sorted(folder.glob('set*.csv')):
            labels = pd.read_csv(path)
            for rally_number, rows in labels.groupby('rally'):
                identity = f'{path.stem}:{int(rally_number)}'
                raw_identities.add(identity)
                if identity not in held:
                    missing_rallies += 1
                flaw_rows = rows['flaw'].eq(1)
                if not flaw_rows.any():
                    continue
                frames = tuple(sorted(int(frame) for frame in rows['frame_num']))
                identity_frames = {int(frame) for frame in rows.loc[flaw_rows, 'frame_num']}
                flagged_hits += int(flaw_rows.sum())
                flagged_frames += len(identity_frames)
                missing_frames += len(identity_frames - held.get(identity, set()))
                flagged_rallies[video.identity].add(identity)
                windows[video.identity].append((frames[0] - before, frames[-1] + after))
        extra_rallies += len(set(held) - raw_identities)
    if missing_rallies or extra_rallies or missing_frames:
        raise ValueError('raw validation flaw labels do not join to shots_master rallies and frames')
    counts = {'flaw_eq_1_hit_rows': flagged_hits, 'flaw_eq_1_unique_frames': flagged_frames,
              'rallies_with_flaw_eq_1': sum(map(len, flagged_rallies.values())),
              'raw_rallies_missing_from_loaded_labels': missing_rallies,
              'loaded_rallies_missing_from_raw_labels': extra_rallies,
              'flaw_eq_1_frames_missing_from_loaded_labels': missing_frames}
    return flagged_rallies, windows, counts


def fully_correct_rallies(judged: list[dict[str, Any]]) -> int:
    """Count labelled rallies with at least one fully correct discovered sequence."""
    return len({row['rally_id'] for row in judged
                if row['rally_id'] is not None and row['outcome'] == 'fully_correct'})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation', type=Path, required=True, help='saved validation root with seed-*/')
    parser.add_argument('--shots-master', type=Path, required=True, help='ShuttleSet shots_master.csv')
    parser.add_argument('--out', type=Path, required=True, help='output JSON.gz path')
    args = parser.parse_args()

    videos = [video for video in load_original_videos(args.shots_master) if video.split == 'validation']
    flagged_rallies, windows, label_counts = flaw_labels(videos)
    clean_labels = {
        video.identity: tuple(rally for rally in video.rallies
                              if rally.identity not in flagged_rallies[video.identity])
        for video in videos
    }
    all_denominator = sum(len(video.rallies) for video in videos)
    clean_denominator = sum(len(rallies) for rallies in clean_labels.values())
    clean_frames = {
        video.identity: [frame for rally in clean_labels[video.identity] for frame in rally.frames]
        for video in videos
    }

    per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        counts = {'all_labels_before_mask': 0, 'unflagged_before_mask': 0, 'unflagged_after_mask': 0}
        ignored_events = 0
        for video in videos:
            stream = read_json(args.validation / f'seed-{seed}' / f'{video.identity}.json.gz')
            sequences, _events = stream_from_payload(stream)
            tolerance = scale_frames(TOLERANCE, video.fps)
            all_judged = judge_found_rallies(
                sequences, video.rallies, tolerance, fps=video.fps,
                boundary_policy=DEFAULT_RALLY_BOUNDARY_POLICY,
            )
            clean_judged = judge_found_rallies(
                sequences, clean_labels[video.identity], tolerance, fps=video.fps,
                boundary_policy=DEFAULT_RALLY_BOUNDARY_POLICY,
            )
            counts['all_labels_before_mask'] += fully_correct_rallies(all_judged)
            counts['unflagged_before_mask'] += fully_correct_rallies(clean_judged)

            masked, ignored = ignore_left_out(
                stream, windows[video.identity], clean_frames[video.identity],
                scale_frames(IGNORE_MARGIN, video.fps),
            )
            masked_sequences, _masked_events = stream_from_payload(masked)
            masked_judged = judge_found_rallies(
                masked_sequences, clean_labels[video.identity], tolerance, fps=video.fps,
                boundary_policy=DEFAULT_RALLY_BOUNDARY_POLICY,
            )
            counts['unflagged_after_mask'] += fully_correct_rallies(masked_judged)
            ignored_events += ignored
        denominators = {'all_labels_before_mask': all_denominator,
                        'unflagged_before_mask': clean_denominator,
                        'unflagged_after_mask': clean_denominator}
        per_seed.append({
            'seed': seed,
            'counts': {name: {'fully_correct': count, 'denominator': denominators[name]}
                       for name, count in counts.items()},
            'ignored_prediction_events': ignored_events,
        })

    def aggregate(name: str, denominator: int) -> dict[str, float | int]:
        mean_correct = float(np.mean([row['counts'][name]['fully_correct'] for row in per_seed]))
        return {'mean_fully_correct': mean_correct, 'denominator_per_seed': denominator,
                'mean_coverage_percent': 100 * mean_correct / denominator}

    before = aggregate('unflagged_before_mask', clean_denominator)
    after = aggregate('unflagged_after_mask', clean_denominator)
    result = {
        'boundary_policy': asdict(DEFAULT_RALLY_BOUNDARY_POLICY),
        'tolerance_base_30_fps': TOLERANCE,
        'ignore_window_base_30_fps': {'before': WINDOW_BEFORE, 'after': WINDOW_AFTER, 'fps_scaled': True},
        'validation_videos': len(videos),
        'seeds': list(SEEDS),
        'label_counts': {'all_rallies': all_denominator, 'unflagged_rallies': clean_denominator,
                         **label_counts},
        'per_seed': per_seed,
        'aggregate': {
            'all_labels_before_mask': aggregate('all_labels_before_mask', all_denominator),
            'unflagged_before_mask': before,
            'unflagged_after_mask': after,
            'mask_effect_percentage_points': after['mean_coverage_percent'] - before['mean_coverage_percent'],
        },
    }
    write_json(args.out, result)
    print('wrote validation label sensitivity results')


if __name__ == '__main__':
    main()
