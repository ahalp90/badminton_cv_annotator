"""Count which validation rallies any choice from the saved option lists could get fully correct.

This made ``remote/candidate-pool-coverage.json.gz``, the probe behind the option-list ceiling in
``screening-evidence.json.gz``. It reads one option-list folder (``<pools>/videos/<video>.joblib``,
written by ``pools``) and checks each of the eight validation videos' labelled rallies:

- **Correct option with initial neighbours:** some option is fully correct for the rally when the
  neighbouring rally sections stay as they started. This is neither a ceiling nor a score.
- **Bounds possible:** some rally section could stretch to contain every labelled hit. A start can
  move back to the previous section's end and both ends can gain the finishing padding, so the
  check is generous.
- **Frame union possible:** within such a section, the frames of all its options together match
  every labelled hit at ±10 frames. The frames need not sit in one option, so this is generous too.

A rally failing either generous check is unreachable. The output is JSON on standard output.

Run from the repository root with ``PYTHONPATH=src:.``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from annotator.sequence.boundaries import BOUNDARY_PADDING_AT_30_FPS
from annotator.sequence.contacts import scale_frames
from annotator.training.sequences import match_contacts, overlapping_rallies
from experiments.annotator.tree_simplification.analysis.common import (
    SHOTS_MASTER,
    validation_videos,
)
from experiments.annotator.tree_simplification.options import (
    LabelledOptions,
    finish_in_initial_context,
)

TOLERANCE_AT_30_FPS = 10


def inspect_pool(pool: LabelledOptions) -> dict[str, Any]:
    """Coverage with initial neighbours, then the two generous reachability checks, for one video."""
    video = pool.video
    padding = scale_frames(BOUNDARY_PADDING_AT_30_FPS, video.fps)
    tolerance = scale_frames(TOLERANCE_AT_30_FPS, video.fps)
    stream_frames = np.asarray([event.frame for event in video.events], dtype=np.int64)
    covered: set[str] = set()
    sections_with_correct_option = 0
    # One (earliest start, latest end, all option frames) per initial rally section.
    envelopes: list[tuple[int, int, list[int]]] = []
    previous_end = 0
    for position, before in enumerate(video.initial):
        rows = pool.span_rows[before.span_id]
        correct_rows = rows[pool.targets[rows] == 1]
        sections_with_correct_option += bool(len(correct_rows))
        for row in correct_rows:
            finished = finish_in_initial_context(video, position, pool.sequences[row], stream_frames)
            overlaps = overlapping_rallies(finished, video.rallies)
            if len(overlaps) != 1:
                raise ValueError(f'{video.identity}: a correct option overlaps {len(overlaps)} labelled rallies')
            covered.add(overlaps[0].identity)
        options = [pool.sequences[row] for row in rows]
        earliest_start = max(previous_end, min(option.start_frame for option in options) - padding)
        latest_end = max(option.end_frame for option in options) + padding
        frames = sorted({event.frame for option in options for event in option.events})
        envelopes.append((earliest_start, latest_end, frames))
        previous_end = before.end_frame

    rallies = []
    for rally in video.rallies:
        containing = [frames for start, end, frames in envelopes if start <= rally.frames[0] and rally.frames[-1] < end]
        frame_union_possible = any(
            len(match_contacts(rally.frames, frames, tolerance)) == len(rally.frames) for frames in containing
        )
        rallies.append({
            'rally': rally.identity,
            'all_sides_known': all(side is not None for side in rally.sides),
            'correct_option_with_initial_neighbours': rally.identity in covered,
            'bounds_possible': bool(containing),
            'frame_union_possible': frame_union_possible,
        })
    return {
        'video': video.identity,
        'labelled_rallies': len(video.rallies),
        'sections_with_correct_option': sections_with_correct_option,
        'with_correct_option': len(covered),
        'bounds_impossible': sum(not row['bounds_possible'] for row in rallies),
        'frame_union_impossible': sum(row['bounds_possible'] and not row['frame_union_possible'] for row in rallies),
        'rallies': rallies,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pools', type=Path, required=True, help='option-list folder holding videos/')
    parser.add_argument('--shots-master', type=Path, default=SHOTS_MASTER, help='labels CSV')
    args = parser.parse_args()
    results = []
    for identity, video in sorted(validation_videos(args.shots_master).items()):
        pool: LabelledOptions = joblib.load(args.pools / 'videos' / f'{identity}.joblib')
        if pool.video.identity != identity or pool.video.rallies != video.rallies:
            raise ValueError(f'{identity}: the option list was built from different labels')
        results.append(inspect_pool(pool))
    print(json.dumps({'per_video': results}, indent=2))


if __name__ == '__main__':
    main()
