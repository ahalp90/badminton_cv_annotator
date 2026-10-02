"""Compare two saved refit outputs without fitting models or running inference.

The report retains gained/lost label IDs as well as aggregate rally counts so
sequence changes inside already-incorrect rallies remain visible.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from annotator.sequence.contacts import ContactEvent, scale_frames
from annotator.training.sequences import LabelledRally, match_contacts
from annotator.training.workflow import load_labels
from dataset_builder.vision import load_json_gz, save_json_gz
from experiments.annotator.old_court_regression.retrain import (
    COUNT_NAMES,
    TEST_VIDEO_IDS,
    score_stream,
    stream_from_payload,
)

REPOSITORY = Path(__file__).resolve().parents[3]
ARMS = ('base', 'veto', 'base_inference_veto')


def label_coverage(events: Sequence[ContactEvent], rallies: Sequence[LabelledRally], fps: float) -> dict[str, set]:
    """Return the same human label keys for timing and correct-side comparisons."""
    labels = []
    for rally in rallies:
        for frame, side in zip(rally.frames, rally.sides, strict=True):
            labels.append((frame, rally.identity, side))
    labels.sort(key=lambda row: row[0])
    matches = match_contacts([row[0] for row in labels], [event.frame for event in events], scale_frames(10, fps))
    timing, sides = set(), set()
    for label, prediction, _offset in matches:
        frame, rally, side = labels[label]
        key = (rally, frame)
        timing.add(key)
        if side is not None and events[prediction].side == side:
            sides.add(key)
    return {'timing': timing, 'correct_side': sides}


def compare_video(left: dict, right: dict, rallies: Sequence[LabelledRally], fps: float) -> dict[str, Any]:
    """Score both streams on identical labels and keep paired changes."""
    scores, coverage = {}, {}
    for arm, payload in (('left', left), ('right', right)):
        sequences, events = stream_from_payload(payload)
        scores[arm] = score_stream(sequences, events, rallies, fps, payload.get('confidence'))
        coverage[arm] = label_coverage(events, rallies, fps)
        reached = {rally for rally, _frame in coverage[arm]['timing']}
        correct = set(scores[arm]['correct_rallies'])
        scores[arm]['fully_correct_rallies'] = len(correct)
        scores[arm]['reached_but_not_correct_rallies'] = len(reached - correct)
        scores[arm]['wholly_missed_rallies'] = len(rallies) - len(reached)
        scores[arm]['correct_side_contacts'] = len(coverage[arm]['correct_side'])
    changes = {}
    for measure in ('timing', 'correct_side'):
        changes[measure] = {
            'gained': sorted(coverage['right'][measure] - coverage['left'][measure]),
            'lost': sorted(coverage['left'][measure] - coverage['right'][measure]),
        }
    changes['fully_correct_rallies'] = {
        'gained': sorted(set(scores['right']['correct_rallies']) - set(scores['left']['correct_rallies'])),
        'lost': sorted(set(scores['left']['correct_rallies']) - set(scores['right']['correct_rallies'])),
    }
    return {'scores': scores, 'changes': changes}


def expected_videos(split: str) -> tuple[str, ...]:
    """Use the same fixed populations as the paired refit."""
    if split == 'test':
        return tuple(str(video) for video in TEST_VIDEO_IDS if video != 15)
    path = REPOSITORY / 'scratch/contact_det_full_ds_fit/records/training_video_score_groups.json'
    return tuple(json.loads(path.read_text())['fixed_validation_videos'])


def compare_saved(run: Path, split: str, left_arm: str, right_arm: str) -> dict[str, Any]:
    """Require complete matching populations before aggregating saved streams."""
    identities = expected_videos(split)
    roots = [run / arm / 'eval' / split for arm in (left_arm, right_arm)]
    for root in roots:
        actual = {path.name for path in root.iterdir() if path.is_dir()}
        if actual != set(identities):
            raise ValueError(f'{root}: expected exactly the {len(identities)} pinned {split} videos; '
                             f'missing={sorted(set(identities) - actual)}, extra={sorted(actual - set(identities))}')
    rows = {}
    label_split = 'development' if split == 'validation' else 'test'
    for identity in identities:
        metadata = [load_json_gz(root / identity / 'scores.json.gz') for root in roots]
        fps, frame_count = metadata[0]['fps'], metadata[0]['frame_count']
        if any(row['fps'] != fps or row['frame_count'] != frame_count for row in metadata[1:]):
            raise ValueError(f'{identity}: saved outputs disagree on the video clock or length')
        rallies = load_labels(run / 'shared/labels' / label_split / f'{identity}.csv', frame_count)
        streams = [load_json_gz(root / identity / 'stream.json.gz') for root in roots]
        rows[identity] = compare_video(*streams, rallies, fps)
    totals = {}
    names = (*COUNT_NAMES, 'fully_correct_rallies', 'reached_but_not_correct_rallies',
             'wholly_missed_rallies', 'correct_side_contacts')
    for side in ('left', 'right'):
        total = {name: sum(row['scores'][side][name] for row in rows.values()) for name in names}
        total['contact_precision'] = (total['matched_contacts'] / total['predicted_contacts']
                                      if total['predicted_contacts'] else None)
        total['contact_recall'] = (total['matched_contacts'] / total['labelled_contacts']
                                  if total['labelled_contacts'] else None)
        totals[side] = total
    changes = {}
    for measure in ('fully_correct_rallies', 'timing', 'correct_side'):
        changes[measure] = {
            direction: sum(len(row['changes'][measure][direction]) for row in rows.values())
            for direction in ('gained', 'lost')
        }
    return {'split': split, 'left': left_arm, 'right': right_arm, 'videos': list(identities),
            'totals': totals, 'changes': changes, 'per_video': rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--split', choices=('validation', 'test'), required=True)
    parser.add_argument('--left', choices=ARMS, default='base')
    parser.add_argument('--right', choices=ARMS, default='veto')
    args = parser.parse_args(argv)
    if args.left == args.right:
        parser.error('choose two different saved outputs')
    result = compare_saved(args.run, args.split, args.left, args.right)
    output = args.run / 'comparisons' / f'{args.split}-{args.left}-vs-{args.right}.json.gz'
    save_json_gz(output, result)
    print(json.dumps({'output': str(output), 'totals': result['totals'], 'changes': result['changes']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
