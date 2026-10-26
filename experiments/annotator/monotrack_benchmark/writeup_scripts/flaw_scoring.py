"""Score discovery mode three ways for the ShuttleSet22 rallies left out of the test set.

The test set scores 3,327 rallies. 524 more rallies in the same videos were left out
because a hit in them is marked ``flaw`` (523) or listed out of time order (1). Discovery
mode searches whole videos, so it finds them. Three treatments:

- standard: retain all predictions. Left-out rallies have no labels, so predictions on them
  count as extras.
- ignore: the headline scoring. Predictions inside a left-out rally's window are dropped, and a
  found rally left with no hits is dropped. Denominators stay at 3,327 rallies.
- count: left-out rallies join the labels as ordinary rallies (3,851 in all).

Hitter sides for left-out rallies use the same rule as the scored labels: Top when the
hitter's court y is above the opponent's. The script checks that rule against every scored
label first.

Usage: python flaw_scoring.py --out DIR
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

# Bound inline: ruff's E402 allows sys.path calls above imports, but not assignments.
sys.path.insert(0, str(REPO := Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(REPO / 'src'))

from annotator.outcomes.point_winner import Half
from annotator.sequence.contacts import scale_frames, stream_from_payload
from annotator.training.sequences import LabelledRally
from experiments.annotator.annotator_closeout_new_courts.scripts.evaluate_saved import (
    score_video,
)
from experiments.annotator.monotrack_benchmark.benchmark_scripts.common import (
    labels_for_item,
    load_manifest,
    read_json,
)
from experiments.annotator.monotrack_benchmark.writeup_scripts.metrics import (
    HEURISTIC,
    IGNORE_MARGIN,
    INPUTS,
    SEEDS,
    WINDOW_AFTER,
    WINDOW_BEFORE,
    annotator_run,
    ignore_left_out,
    judge_found_rallies,
    rally_ladder,
)

TOLERANCE = 10
SHUTTLESET22 = REPO / 'data/shuttleset22/set'


def side_of(row: pd.Series) -> str | None:
    player_y, opponent_y = row['player_location_y'], row['opponent_location_y']
    if not np.isfinite(player_y) or not np.isfinite(opponent_y) or player_y == opponent_y:
        return None
    return 'Top' if player_y < opponent_y else 'Bot'


def left_out_labels(item: dict) -> tuple[list[LabelledRally], dict[str, int], list[tuple[int, bool]]]:
    """Left-out rallies in one test video, label checks, and each left-out hit's frame and flaw flag."""
    folder = SHUTTLESET22 / re.sub(r'^\d+ ', '', item['source_filename']).rsplit('.', 1)[0]
    scored = {span['rally_id']: span for span in item['spans']}
    rallies, hits = [], []
    check = {'scored_sides': 0, 'scored_sides_agree': 0, 'left_out_hits': 0, 'left_out_no_side': 0,
             'left_out_round_order_broken': 0, 'left_out_rallies_no_side': 0, 'flagged_hits': 0,
             'flagged_hits_no_side': 0, 'flagged_hits_after_first': 0, 'flagged_hits_not_after_previous': 0,
             'flagged_first_hits': 0}
    for path in sorted(folder.glob('set*.csv.gz')):
        labels = pd.read_csv(path)
        for rally, rows in labels.groupby('rally'):
            identity = f'{path.name.split(".")[0]}:{rally}'
            if identity in scored:
                sides = [side_of(row) for _, row in rows.iterrows()]
                check['scored_sides'] += len(sides)
                check['scored_sides_agree'] += sum(ours == theirs for ours, theirs
                                                   in zip(sides, scored[identity]['label_sides'], strict=True))
                continue
            by_round = rows.sort_values(['ball_round', 'frame_num'])
            frames_by_round = by_round['frame_num'].tolist()
            check['left_out_round_order_broken'] += any(right <= left for left, right in pairwise(frames_by_round))
            flagged = (by_round['flaw'] == 1).tolist()
            check['flagged_hits'] += sum(flagged)
            check['flagged_first_hits'] += flagged[0]
            check['flagged_hits_no_side'] += sum(is_flagged and side_of(row) is None
                                                 for is_flagged, (_, row) in zip(flagged, by_round.iterrows(), strict=True))
            for (previous, frame), is_flagged in zip(pairwise(frames_by_round), flagged[1:], strict=True):
                check['flagged_hits_after_first'] += is_flagged
                check['flagged_hits_not_after_previous'] += is_flagged and frame <= previous
            hits.extend(zip(frames_by_round, flagged, strict=True))
            ordered = rows.sort_values('frame_num')
            sides = [side_of(row) for _, row in ordered.iterrows()]
            check['left_out_hits'] += len(sides)
            check['left_out_no_side'] += sum(side is None for side in sides)
            check['left_out_rallies_no_side'] += any(side is None for side in sides)
            rallies.append(LabelledRally(f'left_out:{identity}', tuple(int(frame) for frame in ordered['frame_num']),
                                         tuple(None if side is None else Half(side) for side in sides)))
    return rallies, check, hits


def detection_near_hits(streams: dict[str, dict], items: list[dict], hits: dict[str, list[tuple[int, bool]]]) -> dict:
    """Share of flagged and unflagged left-out hits with any prediction within 10 frames."""
    near = {True: [0, 0], False: [0, 0]}
    for item in items:
        predictions = np.sort([event[0] for event in streams[item['item_id']]['events']])
        margin = scale_frames(TOLERANCE, item['fps'])
        for frame, is_flagged in hits[item['item_id']]:
            index = int(np.searchsorted(predictions, frame))
            neighbours = predictions[max(index - 1, 0):index + 1]
            near[is_flagged][0] += bool(len(neighbours)) and bool(np.abs(neighbours - frame).min() <= margin)
            near[is_flagged][1] += 1
    return {'flagged': 100 * near[True][0] / near[True][1], 'unflagged': 100 * near[False][0] / near[False][1]}


def score(item: dict, stream: dict, labels: tuple) -> dict:
    _, contacts, predictions, _ = score_video(item['item_id'], 'test', stream, labels, item['fps'], TOLERANCE)
    sequences, _ = stream_from_payload(stream)
    judged = judge_found_rallies(sequences, labels, scale_frames(TOLERANCE, item['fps']), fps=item['fps'])
    ladder = rally_ladder(judged, labels)
    left_out = {rally.identity for rally in labels if rally.identity.startswith('left_out:')}
    reached: dict[str, set[str]] = {}
    for record in judged:
        if record['rally_id'] is not None:
            reached.setdefault(record['rally_id'], set()).add(record['outcome'])
    return {
        'labelled_rallies': len(labels), 'labelled_contacts': len(contacts), 'predicted_contacts': len(predictions),
        'matched_contacts': sum(bool(row['matched']) for row in contacts),
        'correct_side_contacts': sum(bool(row['player_correct']) for row in contacts),
        'left_out_rallies': len(left_out),
        'left_out_fully_correct': sum('fully_correct' in reached.get(rally, set()) for rally in left_out),
        'left_out_matched': sum(bool(row['matched']) for row in contacts if row['rally_id'] in left_out),
        'left_out_contacts': sum(row['rally_id'] in left_out for row in contacts),
        **ladder,
    }


def score_task(task: tuple[str, dict, dict, list[LabelledRally]]) -> dict:
    treatment, item, stream, left_out = task
    scored = labels_for_item(item)
    if treatment == 'standard':
        return score(item, stream, scored)
    if treatment == 'count':
        return score(item, stream, scored + tuple(left_out))
    windows = [(rally.frames[0] - WINDOW_BEFORE, rally.frames[-1] + WINDOW_AFTER) for rally in left_out]
    scored_frames = [frame for rally in scored for frame in rally.frames]
    # The main scorer's rule, so the ignore treatment reproduces the headline discovery scores.
    masked, _ignored = ignore_left_out(stream, windows, scored_frames, scale_frames(IGNORE_MARGIN, item['fps']))
    return score(item, masked, scored)


def rates(totals: pd.Series) -> dict[str, float]:
    predicted, labelled = totals['predicted_contacts'], totals['labelled_contacts']
    rallies = totals['labelled_rallies']
    return {
        'contact_f1': 200 * totals['matched_contacts'] / (predicted + labelled),
        'player_f1': 200 * totals['correct_side_contacts'] / (predicted + labelled),
        'precision': 100 * totals['matched_contacts'] / predicted,
        'recall': 100 * totals['matched_contacts'] / labelled,
        'fully_correct': 100 * totals['fully_correct_matched'] / rallies,
        'boundaries_right': 100 * totals['boundaries_right'] / rallies,
        'labelled_rallies': float(rallies),
        'left_out_fully_correct': (100 * totals['left_out_fully_correct'] / totals['left_out_rallies']
                                   if totals['left_out_rallies'] else np.nan),
        'left_out_recall': (100 * totals['left_out_matched'] / totals['left_out_contacts']
                            if totals['left_out_contacts'] else np.nan),
        'scored_fully_correct': 100 * (totals['fully_correct_matched'] - totals['left_out_fully_correct'])
        / (rallies - totals['left_out_rallies']),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--annotator-runs', type=Path, required=True)
    args = parser.parse_args()
    items = load_manifest(INPUTS)['items']
    left_out, checks, hits = {}, [], {}
    for item in items:
        left_out[item['item_id']], check, hits[item['item_id']] = left_out_labels(item)
        checks.append(check)
    check = pd.DataFrame(checks).sum().to_dict()
    print('label checks:', check)
    if check['scored_sides_agree'] != check['scored_sides']:
        raise ValueError('side rule does not reproduce the scored labels')

    runs = {'annotator': [{item['item_id']: read_json(annotator_run('discovery', '32_videos', seed, args.annotator_runs) / f'{item["item_id"]}.json.gz')
                           for item in items} for seed in SEEDS],
            'trees_off': [read_json(HEURISTIC / 'test/predictions.json.gz')['streams']]}
    result = {'label_checks': {key: int(value) for key, value in check.items()}}
    for name, streams_by_run in runs.items():
        near = [detection_near_hits(streams, items, hits) for streams in streams_by_run]
        result[f'{name}_prediction_within_10_frames'] = {key: float(np.mean([run[key] for run in near]))
                                                         for key in near[0]}
        print(name, 'prediction within 10 frames of left-out hits:', result[f'{name}_prediction_within_10_frames'])
    with ProcessPoolExecutor() as pool:
        for name, streams_by_run in runs.items():
            result[name] = {}
            for treatment in ('standard', 'ignore', 'count'):
                by_run = []
                for streams in streams_by_run:
                    tasks = [(treatment, item, streams[item['item_id']], left_out[item['item_id']]) for item in items]
                    by_run.append(rates(pd.DataFrame(list(pool.map(score_task, tasks))).sum()))
                result[name][treatment] = {key: float(np.mean([run[key] for run in by_run])) for key in by_run[0]}
                print(name, treatment, {key: round(value, 2) for key, value in result[name][treatment].items()})
    (args.out / 'flaw_scoring_summary.json').write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
