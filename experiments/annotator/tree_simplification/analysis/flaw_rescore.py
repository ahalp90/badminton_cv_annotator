"""Score the old and new annotators' test predictions with and without the 524 left-out rallies.

The ShuttleSet22 test set scores 3,327 rallies. Its 46 videos hold 524 more, left out because a
hit is marked as flawed (523) or listed out of time order (1). Discovery mode searches whole
videos, so it also predicts hits inside them. Three treatments, from the MonoTrack write-up's
``flaw_scoring.py``:

- standard: the scoring behind every committed test result. Predictions inside left-out
  rallies count as extra hits.
- ignore: predictions inside a left-out rally's window (30 frames before its first hit to 30
  after its last) are dropped unless within 10 frames of a scored hit, then found rallies left
  empty are dropped. The write-up's headline figures use this treatment.
- count: left-out rallies join the labels as ordinary rallies (3,851 in all).

Each treatment reports fully correct rallies under two rules. ``fully_correct_rallies`` is this
experiment's rule: one found rally holds every labelled hit of exactly one labelled rally, by
frame range, with every hit matched to the right player and no extra hits.
``fully_correct_rallies_edge_allowance`` is the write-up's rule, which also accepts a found
rally whose edges fall up to 10 frames short of the first or last labelled hit.

Correct-player F1 and both rally counts use ±10 frames at 30 fps. Standard scoring must
reproduce the tracked per-video test scores of both systems, and the write-up's figures for the
old annotator. Run from the repository root with ``PYTHONPATH=src:.``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean
from typing import Any

from annotator.sequence.contacts import scale_frames, stream_from_payload
from annotator.training.sequences import LabelledRally
from experiments.annotator.annotator_closeout_new_courts.scripts.evaluate_saved import (
    score_video,
)
from experiments.annotator.monotrack_benchmark.benchmark_scripts.common import (
    labels_for_item,
    load_manifest,
)
from experiments.annotator.monotrack_benchmark.writeup_scripts.flaw_scoring import (
    TOLERANCE,
    left_out_labels,
)
from experiments.annotator.monotrack_benchmark.writeup_scripts.metrics import (
    IGNORE_MARGIN,
    INPUTS,
    WINDOW_AFTER,
    WINDOW_BEFORE,
    ignore_left_out,
    judge_found_rallies,
    rally_ladder,
)
from experiments.annotator.tree_simplification.analysis.common import (
    REPOSITORY,
    RESULTS,
    SEED_OFFSETS,
    check,
    read_json,
    write_json_gz,
)

OLD_ANNOTATOR = 'old-annotator-hgb-grid-017'
OLD_PREDICTIONS = REPOSITORY / 'experiments/annotator/tree_hp_tune/results/test_rerun/A/v32/grid-017'
NEW_CONFIGURATIONS = ('raw_baseline', 'body_window_36')
TREATMENTS = ('standard', 'ignore', 'count')
SCORED_RALLIES = 3327
LEFT_OUT_RALLIES = 524
COUNT_FIELDS = ('labelled_rallies', 'labelled_contacts', 'predicted_contacts', 'matched_contacts', 'correct_side_contacts')
# The write-up's five-seed means for the old annotator, from its flaw_scoring.py: correct-player F1 (%)
# and fully correct rallies under the edge allowance (% of all labelled rallies). It publishes them at one decimal.
WRITE_UP_OLD_ANNOTATOR = {'standard': (85.441, 51.482), 'ignore': (90.192, 51.548), 'count': (89.155, 44.414)}
WRITE_UP_DECIMALS = 3


def score_one(task: tuple[str, dict[str, Any], Path, list[LabelledRally]]) -> dict[str, int]:
    """Score one video's predictions under one treatment at ±10 frames."""
    treatment, item, path, left_out = task
    stream = read_json(path)
    scored = labels_for_item(item)
    labels = scored
    dropped = 0
    if treatment == 'ignore':
        windows = [(rally.frames[0] - WINDOW_BEFORE, rally.frames[-1] + WINDOW_AFTER) for rally in left_out]
        scored_frames = [frame for rally in scored for frame in rally.frames]
        stream, dropped = ignore_left_out(stream, windows, scored_frames, scale_frames(IGNORE_MARGIN, item['fps']))
    elif treatment == 'count':
        labels = scored + tuple(left_out)
    _proposals, contacts, predictions, rally_rows = score_video(item['item_id'], 'test', stream, labels, item['fps'], TOLERANCE)
    sequences, _events = stream_from_payload(stream)
    ladder = rally_ladder(judge_found_rallies(sequences, labels, scale_frames(TOLERANCE, item['fps'])), labels)
    return {
        'labelled_rallies': len(labels),
        'labelled_contacts': len(contacts),
        'predicted_contacts': len(predictions),
        'matched_contacts': sum(bool(row['matched']) for row in contacts),
        'correct_side_contacts': sum(bool(row['player_correct']) for row in contacts),
        'fully_correct_rallies': sum(bool(row['fully_correct']) for row in rally_rows),
        'fully_correct_rallies_edge_allowance': ladder['fully_correct_matched'],
        'dropped_predictions': dropped,
    }


def seed_summary(per_video: dict[str, dict[str, int]]) -> dict[str, Any]:
    totals = {name: sum(video[name] for video in per_video.values()) for name in next(iter(per_video.values()))}
    rallies = totals['labelled_rallies']
    return {
        **totals,
        'attributed_f1_percent': 200 * totals['correct_side_contacts'] / (totals['predicted_contacts'] + totals['labelled_contacts']),
        'fully_correct_rallies_percent': 100 * totals['fully_correct_rallies'] / rallies,
        'fully_correct_rallies_edge_allowance_percent': 100 * totals['fully_correct_rallies_edge_allowance'] / rallies,
    }


def prediction_folders(results: Path, old_predictions: Path) -> dict[str, Path]:
    folders = {OLD_ANNOTATOR: old_predictions}
    folders.update({configuration: results / 'streams/input-test' / configuration for configuration in NEW_CONFIGURATIONS})
    return folders


def check_standard(
    results: Path, old_predictions: Path, per_video: dict[str, dict[str, dict[str, Any]]],
) -> None:
    """Standard scoring must give the tracked per-video counts, under this experiment's rally rule."""
    def counts(score: dict[str, int]) -> dict[str, int]:
        return {name: score[name] for name in (*COUNT_FIELDS, 'fully_correct_rallies')}

    for seed in SEED_OFFSETS:
        old_saved = read_json(old_predictions / f'seed-{seed}' / 'scores.json.gz')['by_video']
        check({video: counts(score) for video, score in per_video[OLD_ANNOTATOR][str(seed)].items()},
              {video: saved['10'] for video, saved in old_saved.items()},
              f'{OLD_ANNOTATOR} seed {seed} standard scoring equals the saved per-video scores')
        for configuration in NEW_CONFIGURATIONS:
            folder = results / 'streams/input-test' / configuration / f'seed-{seed}' / 'scores'
            saved = {path.name.removesuffix('.json.gz'): read_json(path)['score']['10'] for path in folder.glob('*.json.gz')}
            check({video: counts(score) for video, score in per_video[configuration][str(seed)].items()}, saved,
                  f'{configuration} seed {seed} standard scoring equals the tracked per-video scores')


def check_write_up(summary: dict[str, Any]) -> None:
    for treatment, expected in WRITE_UP_OLD_ANNOTATOR.items():
        means = summary[OLD_ANNOTATOR][treatment]['mean']
        actual = (means['attributed_f1_percent'], means['fully_correct_rallies_edge_allowance_percent'])
        rounded = tuple(round(value, WRITE_UP_DECIMALS) for value in actual)
        if rounded != expected:
            raise ValueError(f'{OLD_ANNOTATOR} {treatment}: got {rounded}, the write-up reports {expected}')
        print(f'{OLD_ANNOTATOR} {treatment}: {rounded} equals the write-up', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', type=Path, default=RESULTS, help='tracked results folder')
    parser.add_argument('--old-predictions', type=Path, default=OLD_PREDICTIONS,
                        help='the old annotator test predictions, one seed-<N> folder per seed')
    parser.add_argument('--output', type=Path, required=True, help='gzip JSON file to write')
    args = parser.parse_args()

    items = load_manifest(INPUTS)['items']
    left_out = {}
    label_checks: dict[str, int] = {}
    for item in items:
        left_out[item['item_id']], checks, _hits = left_out_labels(item)
        for name, value in checks.items():
            label_checks[name] = label_checks.get(name, 0) + int(value)
    if label_checks['scored_sides_agree'] != label_checks['scored_sides']:
        raise ValueError('The hitter-side rule does not reproduce the scored labels')
    scored_total = sum(len(labels_for_item(item)) for item in items)
    left_out_total = sum(len(rallies) for rallies in left_out.values())
    if (scored_total, left_out_total) != (SCORED_RALLIES, LEFT_OUT_RALLIES):
        raise ValueError(f'Expected {SCORED_RALLIES} scored and {LEFT_OUT_RALLIES} left-out rallies')

    folders = prediction_folders(args.results, args.old_predictions)
    tasks = [
        (system, treatment, seed, item['item_id'], (treatment, item, folder / f'seed-{seed}' / f'{item["item_id"]}.json.gz',
                                                      left_out[item['item_id']]))
        for system, folder in folders.items() for treatment in TREATMENTS for seed in SEED_OFFSETS for item in items
    ]
    with ProcessPoolExecutor() as pool:
        scores = list(pool.map(score_one, [task[-1] for task in tasks], chunksize=4))
    per_video: dict[str, dict[str, dict[str, dict[str, dict[str, int]]]]] = {}
    for (system, treatment, seed, video, _task), score in zip(tasks, scores, strict=True):
        per_video.setdefault(treatment, {}).setdefault(system, {}).setdefault(str(seed), {})[video] = score
    check_standard(args.results, args.old_predictions, per_video['standard'])

    summary: dict[str, Any] = {}
    for system in folders:
        summary[system] = {}
        for treatment in TREATMENTS:
            per_seed = {seed: seed_summary(videos) for seed, videos in per_video[treatment][system].items()}
            summary[system][treatment] = {
                'per_seed': per_seed,
                'mean': {name: mean(values[name] for values in per_seed.values()) for name in per_seed['0']},
            }
    check_write_up(summary)
    document = {
        'schema': 'tree-simplification-test-flaw-scoring/1',
        'about': __doc__.split('Run from the repository root')[0].strip(),
        'tolerance_frames_at_30_fps': TOLERANCE,
        'test_videos': len(items),
        'scored_rallies': SCORED_RALLIES,
        'left_out_rallies': LEFT_OUT_RALLIES,
        'left_out_label_checks': label_checks,
        'predictions': {
            system: str(folder.resolve().relative_to(REPOSITORY)) for system, folder in folders.items()
        },
        'results': summary,
    }
    write_json_gz(args.output, document)
    for system in folders:
        for treatment in ('standard', 'ignore'):
            means = summary[system][treatment]['mean']
            print(f'{system} {treatment}: F1 {means["attributed_f1_percent"]:.3f}%, fully correct '
                  f'{means["fully_correct_rallies"]:.1f} ({means["fully_correct_rallies_percent"]:.3f}%), with edge '
                  f'allowance {means["fully_correct_rallies_edge_allowance"]:.1f} '
                  f'({means["fully_correct_rallies_edge_allowance_percent"]:.3f}%), dropped {means["dropped_predictions"]:.1f}',
                  flush=True)
    tracked = args.results / 'test-flaw-scoring.json.gz'
    if tracked.exists():
        check(document, read_json(tracked), 'test-flaw-scoring.json.gz')


if __name__ == '__main__':
    main()
