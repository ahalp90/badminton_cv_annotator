"""Contact, rally and confidence numbers for the v3 MonoTrack comparison report.

Reads committed prediction streams only. Every condition is rescored here with the
benchmark's own contact matcher, and the pooled totals are checked against each
run's saved ``scores.json.gz`` before anything is written.

Two rally rules are computed for discovery mode:

- ``strict``: the existing rule. One found rally must hold, by frame range, every
  labelled hit of exactly one labelled rally and no labelled hit of another.
- ``matched``: the report's rule, in three nested steps. Boundaries right: the strict
  rule, with the found rally's span widened by 10 frames at each end (the hit-timing
  allowance). Hits right: its hits also match every labelled hit one to one, with no
  extras. Fully correct: every hitter is also right.

With supplied windows the two rules agree, because each window holds exactly one
labelled rally. The script checks this.

The test videos also hold 524 ShuttleSet22 rallies left out of the test set, almost all
because a hit in them is flagged ``flaw``. Their flagged rows lack usable times and hitters,
so every condition ignores them: a prediction inside a left-out rally's window that is more
than 10 frames from every scored hit is dropped before scoring, and so is a found rally left
with no predictions. Saved-score checks add the dropped predictions back.

Usage: python metrics.py --confidence DIR --out DIR
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPO / 'src'), str(REPO)]

from annotator.sequence.contacts import scale_frames
from annotator.sequence.sides import alternate_sides
from annotator.training.sequences import match_contacts
from experiments.annotator.annotator_closeout_new_courts.scripts.evaluate_saved import (
    score_video,
)
from experiments.annotator.monotrack_benchmark.benchmark_scripts.common import (
    fixed_stream,
    labels_for_item,
    load_manifest,
    read_json,
)
from experiments.annotator.old_court_regression.retrain import stream_from_payload

BENCHMARK = REPO / 'experiments/annotator/monotrack_benchmark/benchmark_results'
INPUTS = BENCHMARK / 'inputs'
COMPARISON = BENCHMARK / 'comparison'
TREE = REPO / 'experiments/annotator/tree_hp_tune/results/test_rerun'
HEURISTIC = REPO / 'experiments/annotator/heuristic_only_tune/results'
SEEDS = (0, 100, 200, 300, 400)
# The tree retune setting with the best mean validation Correct-player F1; the report uses it throughout.
CONFIGURATION = 'grid-017'
TOLERANCES = (10, 5)
# The review cut-off carried over from the historical confidence model, fixed before this test.
REVIEW_THRESHOLD = 0.7570784853533734
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261019
SHUTTLESET22 = REPO / 'data/shuttleset22/set'
# Left-out rally windows run from 30 frames before the first hit to 30 after the last, as for scored rallies.
WINDOW_BEFORE, WINDOW_AFTER = 30, 31
# A prediction this close to a scored hit is never ignored, so it can still match or count as an extra.
IGNORE_MARGIN = 10

SUPPLIED_RUNS = {
    'monotrack_own_courts': [COMPARISON / 'mono_zeroed_own_off'],
    'monotrack_our_courts': [COMPARISON / 'mono_zeroed_our_courts_off'],
    'annotator_our_courts': [TREE / f'B/v32/{CONFIGURATION}/seed-{seed}' for seed in SEEDS],
    'annotator_monotrack_courts': [TREE / f'C/v32/{CONFIGURATION}/seed-{seed}' for seed in SEEDS],
    'heuristic_our_courts': [HEURISTIC / 'known_our_courts'],
    'heuristic_monotrack_courts': [HEURISTIC / 'known_mono_courts'],
    'annotator40_our_courts': [TREE / f'B/final40/{CONFIGURATION}/seed-{seed}' for seed in SEEDS],
    'annotator40_monotrack_courts': [TREE / f'C/final40/{CONFIGURATION}/seed-{seed}' for seed in SEEDS],
}
# Found-rally outcomes, from worst to best. The first three fail the boundary rung.
OUTCOMES = ('no_labelled_rally', 'merged_rallies', 'part_of_rally', 'missed_hits', 'extra_hits', 'wrong_player',
            'unlabelled_player', 'fully_correct')
BOUNDARY_RIGHT = frozenset(OUTCOMES[3:])
HITS_RIGHT = frozenset(OUTCOMES[5:])


def discovery_runs(confidence_root: Path) -> dict[str, list[dict[str, dict]]]:
    """Per-video discovery streams for each run, keyed by item ID."""
    runs: dict[str, list[dict[str, dict]]] = {'annotator_discovery': [], 'annotator40_discovery': []}
    manifest = load_manifest(INPUTS)
    for seed in SEEDS:
        streams = {}
        for item in manifest['items']:
            stream = read_json(TREE / f'A/v32/{CONFIGURATION}/seed-{seed}' / f'{item["item_id"]}.json.gz')
            rerun = read_json(confidence_root / f'seed-{seed}' / f'{item["item_id"]}.json.gz')
            if rerun['sequences'] != stream['sequences'] or rerun['events'] != stream['events']:
                raise ValueError(f'seed {seed}, video {item["item_id"]}: confidence rerun differs from committed stream')
            streams[item['item_id']] = rerun
        runs['annotator_discovery'].append(streams)
        runs['annotator40_discovery'].append({
            item['item_id']: read_json(TREE / f'A/final40/{CONFIGURATION}/seed-{seed}' / f'{item["item_id"]}.json.gz')
            for item in manifest['items']})
    runs['heuristic_discovery'] = [read_json(HEURISTIC / 'test/predictions.json.gz')['streams']]
    return runs


def test_left_out() -> dict[str, list[dict]]:
    """ShuttleSet22 rallies in each test video that the manifest does not score."""
    left_out = {}
    for item in load_manifest(INPUTS)['items']:
        folder = SHUTTLESET22 / re.sub(r'^\d+ ', '', item['source_filename']).rsplit('.', 1)[0]
        scored = {span['rally_id']: tuple(span['label_frames']) for span in item['spans']}
        rallies = []
        for path in sorted(folder.glob('set*.csv.gz')):
            labels = pd.read_csv(path)
            for rally, rows in labels.groupby('rally'):
                identity = f'{path.name.split(".")[0]}:{rally}'
                frames = tuple(int(frame) for frame in rows['frame_num'])
                if identity in scored:
                    # The manifest copies ShuttleSet22's frames, so a scored rally must match exactly.
                    if scored[identity] != frames:
                        raise ValueError(f'{item["item_id"]} {identity}: manifest frames differ from ShuttleSet22')
                    continue
                # 115 left-out rallies list frames out of time order, so sort before taking the extent.
                rallies.append({'rally_id': identity, 'frames': tuple(sorted(frames)),
                                'flaw': bool(rows['flaw'].notna().any())})
        if len(rallies) + len(scored) != sum(pd.read_csv(path)['rally'].nunique()
                                              for path in folder.glob('set*.csv.gz')):
            raise ValueError(f'{item["item_id"]}: scored rallies missing from ShuttleSet22')
        left_out[item['item_id']] = rallies
    return left_out


def left_out_windows(rallies: list[dict]) -> list[tuple[int, int]]:
    return [(rally['frames'][0] - WINDOW_BEFORE, rally['frames'][-1] + WINDOW_AFTER) for rally in rallies]


def ignore_left_out(stream: dict, windows: list[tuple[int, int]], scored_frames: list[int],
                    margin: int) -> tuple[dict, int]:
    """Drop predictions that belong to left-out rallies, then found rallies left with none."""
    def ignored(event: list) -> bool:
        frame = event[0]
        return (any(start <= frame < end for start, end in windows)
                and all(abs(frame - scored) > margin for scored in scored_frames))

    sequences = []
    for sequence in stream['sequences']:
        kept = [event for event in sequence['events'] if not ignored(event)]
        if kept or not sequence['events']:
            sequences.append({**sequence, 'events': kept})
    events = [event for event in stream['events'] if not ignored(event)]
    return {**stream, 'sequences': sequences, 'events': events}, len(stream['events']) - len(events)


def judge_found_rallies(sequences, rallies, tolerance: int) -> list[dict]:
    """Classify each found rally by the labelled hits inside its span, widened at each end by ``tolerance``.

    The widening gives span edges the same allowance as hit timing. Without it, a found rally that
    starts a few frames after the labelled serve would fail the boundary step.
    """
    judged = []
    for sequence in sequences:
        frames = [event.frame for event in sequence.events]
        start, end = sequence.start_frame - tolerance, sequence.end_frame + tolerance
        inside = [rally for rally in rallies if any(start <= frame < end for frame in rally.frames)]
        record = {'span_id': sequence.span_id, 'rally_id': None, 'hits': len(frames)}
        if not inside:
            judged.append(record | {'outcome': 'no_labelled_rally'})
            continue
        if len(inside) > 1:
            judged.append(record | {'outcome': 'merged_rallies'})
            continue
        rally = inside[0]
        record['rally_id'] = rally.identity
        covers = (sequence.start_frame - tolerance <= rally.frames[0]
                  and rally.frames[-1] < sequence.end_frame + tolerance)
        matches = match_contacts(rally.frames, frames, tolerance)
        if not covers:
            outcome = 'part_of_rally'
        elif len(matches) < len(rally.frames):
            outcome = 'missed_hits'
        elif len(frames) > len(rally.frames):
            outcome = 'extra_hits'
        else:
            revised = alternate_sides((sequence,), sequence.events)[0][0]
            wrong = any(rally.sides[label] is not None and revised.events[prediction].side != rally.sides[label]
                        for label, prediction, _offset in matches)
            if wrong:
                outcome = 'wrong_player'
            elif any(side is None for side in rally.sides):
                outcome = 'unlabelled_player'
            else:
                outcome = 'fully_correct'
        judged.append(record | {'outcome': outcome})
    return judged


def rally_ladder(judged: list[dict], rallies) -> dict[str, int]:
    """Count labelled rallies reaching each rung through any one found rally."""
    best: dict[str, set[str]] = {rally.identity: set() for rally in rallies}
    for record in judged:
        if record['rally_id'] is not None:
            best[record['rally_id']].add(record['outcome'])
    return {
        'boundaries_right': sum(bool(outcomes & BOUNDARY_RIGHT) for outcomes in best.values()),
        'hits_right': sum(bool(outcomes & HITS_RIGHT) for outcomes in best.values()),
        'fully_correct_matched': sum('fully_correct' in outcomes for outcomes in best.values()),
    }


def score_stream_for_item(item: dict, stream: dict, tolerance: int, windows: list[tuple[int, int]], *,
                          supplied: bool) -> tuple[dict, list[dict]]:
    """Counts for one video and tolerance, plus the found-rally judgements."""
    labels = labels_for_item(item)
    if supplied:
        events = [event for event in stream['events']
                  if any(span['start_frame'] <= event[0] < span['end_frame'] for span in item['spans'])]
        unmasked = fixed_stream(events, item['spans'])
    else:
        unmasked = stream
    scored_frames = [frame for rally in labels for frame in rally.frames]
    scored, ignored = ignore_left_out(unmasked, windows, scored_frames, scale_frames(IGNORE_MARGIN, item['fps']))
    if supplied:
        scored = fixed_stream(scored['events'], item['spans'])
    _, contacts, predictions, rally_rows = score_video(item['item_id'], 'test', scored, labels, item['fps'], tolerance)
    strict = sum(bool(row['fully_correct']) for row in rally_rows)
    # The saved scores count ignored predictions, so the strict rule is also scored without the mask.
    strict_unmasked = strict if not ignored else sum(
        bool(row['fully_correct'])
        for row in score_video(item['item_id'], 'test', unmasked, labels, item['fps'], tolerance)[3])
    sequences, _ = stream_from_payload(scored)
    judged = judge_found_rallies(sequences, labels, scale_frames(tolerance, item['fps']))
    outside = 0 if supplied else sum(
        not any(span['start_frame'] <= event[0] < span['end_frame'] for span in item['spans'])
        for event in scored['events'])
    counts = {
        'labelled_rallies': len(rally_rows), 'labelled_contacts': len(contacts),
        'predicted_contacts': len(predictions),
        'matched_contacts': sum(bool(row['matched']) for row in contacts),
        'correct_side_contacts': sum(bool(row['player_correct']) for row in contacts),
        'fully_correct_strict': strict,
        'fully_correct_strict_unmasked': strict_unmasked,
        'exact_hits_strict': sum(bool(row['exact_sequence']) for row in rally_rows),
        'predicted_outside_windows': outside,
        'ignored_predictions': ignored,
        'found_rallies': len(sequences),
        **rally_ladder(judged, labels),
    }
    confidence = stream.get('confidence')
    for record in judged:
        record['confidence'] = None if confidence is None else confidence[str(record['span_id'])]
    return counts, judged


def score_task(task: tuple[dict, dict, int, list[tuple[int, int]], bool]) -> tuple[dict, list[dict]]:
    item, stream, tolerance, windows, supplied = task
    return score_stream_for_item(item, stream, tolerance, windows, supplied=supplied)


def score_condition(name: str, streams_by_run: list[dict[str, dict]], *, supplied: bool,
                    saved: list[Path] | None, windows: dict[str, list[tuple[int, int]]],
                    pool: ProcessPoolExecutor) -> dict:
    manifest = load_manifest(INPUTS)
    per_run = []
    for run_index, streams in enumerate(streams_by_run):
        rows, found = [], []
        tasks = [(item, streams[item['item_id']], tolerance, windows[item['item_id']], supplied)
                 for item in manifest['items'] for tolerance in TOLERANCES]
        for (item, _stream, tolerance, _windows, _supplied), (counts, judged) in zip(
                tasks, pool.map(score_task, tasks, chunksize=4), strict=True):
                rows.append({'video': int(item['video_id']), 'tolerance': tolerance, **counts})
                if tolerance == 10:
                    found.extend({'video': int(item['video_id']), **record} for record in judged)
        frame = pd.DataFrame(rows)
        if saved is not None:
            check_against_saved(name, frame, read_json(saved[run_index] / 'scores.json.gz'))
        if supplied and not (frame['fully_correct_strict'] == frame['fully_correct_matched']).all():
            raise ValueError(f'{name}: the two rally rules disagree with supplied windows')
        per_run.append({'per_video': frame, 'found': pd.DataFrame(found)})
    return {'name': name, 'supplied': supplied, 'runs': per_run}


def check_against_saved(name: str, frame: pd.DataFrame, saved: dict) -> None:
    """Saved scores count every prediction, so ignored predictions are added back before comparing."""
    for tolerance in TOLERANCES:
        totals = frame[frame['tolerance'] == tolerance].sum()
        reference = saved['by_tolerance'][str(tolerance)]
        ours = {'labelled_contacts': totals['labelled_contacts'],
                'predicted_contacts': totals['predicted_contacts'] + totals['ignored_predictions'],
                'matched_contacts': totals['matched_contacts'],
                'correct_side_contacts': totals['correct_side_contacts'],
                'fully_correct_rallies': totals['fully_correct_strict_unmasked']}
        for key, value in ours.items():
            if int(value) != int(reference[key]):
                raise ValueError(f'{name} ±{tolerance}: {key} {int(value)} != saved {reference[key]}')


def rates(totals: pd.Series) -> dict[str, float]:
    labelled, predicted = totals['labelled_contacts'], totals['predicted_contacts']
    matched, attributed = totals['matched_contacts'], totals['correct_side_contacts']
    rallies = totals['labelled_rallies']
    return {
        'precision': 100 * matched / predicted, 'recall': 100 * matched / labelled,
        'f1': 100 * 2 * matched / (predicted + labelled),
        'player_precision': 100 * attributed / predicted, 'player_recall': 100 * attributed / labelled,
        'player_f1': 100 * 2 * attributed / (predicted + labelled),
        'fully_correct': 100 * totals['fully_correct_matched'] / rallies,
        'fully_correct_strict': 100 * totals['fully_correct_strict'] / rallies,
        'hits_right': 100 * totals['hits_right'] / rallies,
        'boundaries_right': 100 * totals['boundaries_right'] / rallies,
        'missed': labelled - matched, 'wrong_player': matched - attributed,
        'extra_inside_windows': predicted - matched - totals['predicted_outside_windows'],
        'extra_outside_windows': totals['predicted_outside_windows'],
        'rallies_not_fully_correct': rallies - totals['fully_correct_matched'],
        'found_rallies': totals['found_rallies'],
        'ignored_predictions': totals['ignored_predictions'],
        'fully_correct_count': totals['fully_correct_matched'],
        'hits_right_count': totals['hits_right'], 'boundaries_right_count': totals['boundaries_right'],
        'predicted_contacts': predicted, 'matched_contacts': matched,
    }


def summarise(condition: dict) -> dict:
    """Mean, lowest and highest run for every rate, at each tolerance."""
    result = {}
    for tolerance in TOLERANCES:
        by_run = [rates(run['per_video'][run['per_video']['tolerance'] == tolerance].sum()) for run in condition['runs']]
        result[str(tolerance)] = {key: {'mean': float(np.mean([run[key] for run in by_run])),
                                        'min': float(np.min([run[key] for run in by_run])),
                                        'max': float(np.max([run[key] for run in by_run])),
                                        'runs': [float(run[key]) for run in by_run]}
                                  for key in by_run[0]}
    return result


def per_video_mean(condition: dict, tolerance: int = 10) -> pd.DataFrame:
    """Per-video counts averaged over runs; counts are summed before rates are taken."""
    frames = [run['per_video'][run['per_video']['tolerance'] == tolerance].set_index('video') for run in condition['runs']]
    mean = sum(frame.drop(columns='tolerance') for frame in frames) / len(frames)
    mean['f1'] = 100 * 2 * mean['matched_contacts'] / (mean['predicted_contacts'] + mean['labelled_contacts'])
    mean['player_f1'] = 100 * 2 * mean['correct_side_contacts'] / (mean['predicted_contacts'] + mean['labelled_contacts'])
    mean['fully_correct'] = 100 * mean['fully_correct_matched'] / mean['labelled_rallies']
    return mean.sort_index()


def paired_bootstrap(first: pd.DataFrame, second: pd.DataFrame) -> dict:
    """Video-resampled 95% interval for pooled differences (first minus second)."""
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    draws = generator.integers(0, len(first), size=(BOOTSTRAP_DRAWS, len(first)))
    columns = ['labelled_contacts', 'predicted_contacts', 'matched_contacts', 'correct_side_contacts',
               'fully_correct_matched', 'labelled_rallies']

    def pooled(values: np.ndarray) -> dict[str, np.ndarray]:
        labelled, predicted, matched, attributed, correct, rallies = np.moveaxis(values, -1, 0)
        return {'f1': 200 * matched / (predicted + labelled), 'player_f1': 200 * attributed / (predicted + labelled),
                'fully_correct': 100 * correct / rallies}

    left, right = first[columns].to_numpy(), second[columns].to_numpy()
    observed_left, observed_right = pooled(left.sum(axis=0)), pooled(right.sum(axis=0))
    resampled_left, resampled_right = pooled(left[draws].sum(axis=1)), pooled(right[draws].sum(axis=1))
    result = {}
    for metric in ('f1', 'player_f1', 'fully_correct'):
        low, high = np.percentile(resampled_left[metric] - resampled_right[metric], [2.5, 97.5])
        per_video = first[metric] - second[metric]
        result[metric] = {'difference': float(observed_left[metric] - observed_right[metric]),
                          'interval': [float(low), float(high)],
                          'first_higher_videos': int((per_video > 0).sum()),
                          'second_higher_videos': int((per_video < 0).sum())}
    return result


def confidence_curve(condition: dict, total_rallies: int) -> dict:
    """What a reviewer gets by keeping only found rallies at or above each confidence cut-off."""
    thresholds = np.round(np.linspace(0, 1, 101), 2).tolist() + [REVIEW_THRESHOLD]
    curves = []
    for run in condition['runs']:
        found = run['found']
        rows = []
        for threshold in thresholds:
            kept = found[found['confidence'] >= threshold]
            outcomes = Counter(kept['outcome'])
            covered = kept[kept['outcome'] == 'fully_correct']
            rows.append({
                'threshold': threshold, 'kept': len(kept), **{outcome: outcomes.get(outcome, 0) for outcome in OUTCOMES},
                'fully_correct_share_of_kept': 100 * outcomes.get('fully_correct', 0) / len(kept) if len(kept) else np.nan,
                'boundaries_right_share_of_kept': (100 * sum(outcomes.get(name, 0) for name in BOUNDARY_RIGHT) / len(kept)
                                                   if len(kept) else np.nan),
                'labelled_rallies_fully_correct_in_kept': 100 * covered[['video', 'rally_id']].drop_duplicates().shape[0]
                / total_rallies,
            })
        curves.append(pd.DataFrame(rows))
    return {'thresholds': thresholds, 'runs': curves}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confidence', type=Path, required=True, help='confidence rerun output with seed-*/ folders')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(INPUTS)
    total_rallies = sum(len(item['spans']) for item in manifest['items'])
    windows = {video: left_out_windows(rallies) for video, rallies in test_left_out().items()}

    conditions = {}
    pool = ProcessPoolExecutor(max_workers=args.workers)
    for name, runs in SUPPLIED_RUNS.items():
        streams = [{item['item_id']: read_json(run / f'{item["item_id"]}.json.gz')['stream'] for item in manifest['items']}
                   for run in runs]
        conditions[name] = score_condition(name, streams, supplied=True, saved=runs, windows=windows, pool=pool)
        print(name, 'scored', flush=True)
    for name, streams in discovery_runs(args.confidence).items():
        conditions[name] = score_condition(name, streams, supplied=False, saved=None, windows=windows, pool=pool)
        print(name, 'scored', flush=True)

    # Discovery totals must reproduce the committed whole-video scores.
    for name, prefix in (('annotator_discovery', 'A/v32'), ('annotator40_discovery', 'A/final40')):
        for seed, run in zip(SEEDS, conditions[name]['runs'], strict=True):
            check_against_saved(name, run['per_video'], read_json(TREE / f'{prefix}/{CONFIGURATION}/seed-{seed}/scores.json.gz'))
    heuristic_saved = read_json(HEURISTIC / 'test/summary.json.gz')
    summary = {name: summarise(condition) for name, condition in conditions.items()}
    summary['_heuristic_discovery_saved_summary_keys'] = list(heuristic_saved)

    per_video = {name: per_video_mean(condition) for name, condition in conditions.items()}
    for name, frame in per_video.items():
        frame.to_csv(args.out / f'per_video_{name}.csv')
    paired = {
        'annotator_minus_monotrack_own_courts': paired_bootstrap(per_video['annotator_our_courts'],
                                                                 per_video['monotrack_own_courts']),
        'annotator_minus_monotrack_on_monotrack_courts': paired_bootstrap(per_video['annotator_monotrack_courts'],
                                                                          per_video['monotrack_own_courts']),
        'annotator_own_minus_monotrack_courts': paired_bootstrap(per_video['annotator_our_courts'],
                                                                 per_video['annotator_monotrack_courts']),
        'monotrack_own_minus_our_courts': paired_bootstrap(per_video['monotrack_own_courts'],
                                                           per_video['monotrack_our_courts']),
        'heuristic_minus_monotrack_own_courts': paired_bootstrap(per_video['heuristic_our_courts'],
                                                                 per_video['monotrack_own_courts']),
    }
    curve = confidence_curve(conditions['annotator_discovery'], total_rallies)
    for seed, frame in zip(SEEDS, curve['runs'], strict=True):
        frame.to_csv(args.out / f'confidence_curve_seed{seed}.csv', index=False)
    for name in ('annotator_discovery', 'annotator40_discovery', 'heuristic_discovery'):
        for seed, run in zip(SEEDS, conditions[name]['runs']):
            run['found'].to_csv(args.out / f'found_rallies_{name}_seed{seed}.csv.gz', index=False)
    (args.out / 'contact_summary.json').write_text(json.dumps({'summary': summary, 'paired': paired}, indent=1))
    print(json.dumps({name: {key: round(value['mean'], 2) for key, value in condition['10'].items()}
                      for name, condition in summary.items() if not name.startswith('_')}, indent=1))


if __name__ == '__main__':
    main()
