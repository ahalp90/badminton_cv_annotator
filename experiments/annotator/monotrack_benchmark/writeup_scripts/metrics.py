"""Score discovery contact endpoints and supplied rally windows.

Contact matching, excluded-label masking and player alternation use the benchmark's
existing rules. Saved-score checks compare contact totals, which are independent of
boundary policy. Report columns retain their established strict/matched names:
strict uses the whole-rally scorer and matched uses found-rally judgements, both
with the supplied policy. Supplied windows must agree under both scorers.

Usage: python metrics.py --annotator-runs DIR --out DIR
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path

import numpy as np
import pandas as pd

# Bound inline: ruff's E402 allows sys.path calls above imports, but not assignments.
sys.path.insert(0, str(REPO := Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(REPO / 'src'))

from annotator.sequence.contacts import scale_frames, stream_from_payload
from annotator.sequence.rally_model import alternate_sides
from annotator.training.sequences import (
    judge_rally_boundaries,
    match_contacts,
)
from experiments.annotator.annotator_closeout_new_courts.scripts.evaluate_saved import (
    score_video,
)
from experiments.annotator.monotrack_benchmark.benchmark_scripts.common import (
    fixed_stream,
    labels_for_item,
    load_manifest,
    read_json,
    write_json,
)

BENCHMARK = REPO / 'experiments/annotator/monotrack_benchmark/benchmark_results'
INPUTS = BENCHMARK / 'inputs'
MONOTRACK_RUNS = BENCHMARK / 'monotrack_five_seed'
HEURISTIC = REPO / 'experiments/annotator/heuristic_only_tune/results'
SEEDS = (0, 100, 200, 300, 400)
TOLERANCES = (10, 5)
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261019
SHUTTLESET22 = REPO / 'data/shuttleset22/set'
# Left-out rally windows run from 30 frames before the first hit to 30 after the last, as for scored rallies.
WINDOW_BEFORE, WINDOW_AFTER = 30, 31
# A prediction this close to a scored hit is never ignored, so it can still match or count as an extra.
IGNORE_MARGIN = 10


def annotator_run(condition: str, models: str, seed: int, root: Path) -> Path:
    """One seed's test run: condition ``discovery``, ``our_courts`` or ``monotrack_courts``.

    Models are ``32_videos`` (the headline), ``40_videos`` (contact model refitted on all 40 original
    videos) or ``body_window_36`` (the preprocessing setting with the most fully correct validation rallies).
    """
    return root / 'test' / condition / models / f'seed-{seed}'


def supplied_runs(annotator_root: Path, monotrack_root: Path) -> dict[str, list[Path]]:
    runs = {
        'monotrack_own_courts': [monotrack_root / 'test/monotrack' / f'seed-{seed}' for seed in SEEDS],
        'monotrack_our_courts': [monotrack_root / 'test/ours' / f'seed-{seed}' for seed in SEEDS],
        'heuristic_our_courts': [HEURISTIC / 'known_our_courts'],
        'heuristic_monotrack_courts': [HEURISTIC / 'known_mono_courts'],
    }
    for name, condition, family in (
        ('annotator_our_courts', 'our_courts', '32_videos'),
        ('annotator_monotrack_courts', 'monotrack_courts', '32_videos'),
        ('annotator40_our_courts', 'our_courts', '40_videos'),
        ('annotator40_monotrack_courts', 'monotrack_courts', '40_videos'),
        ('annotator_body_window_36_our_courts', 'our_courts', 'body_window_36'),
    ):
        runs[name] = [annotator_run(condition, family, seed, annotator_root) for seed in SEEDS]
    return runs


# Found-rally outcomes, from worst to best. Empty detections have no endpoints.
OUTCOMES = ('no_contacts', 'no_labelled_rally', 'merged_rallies', 'part_of_rally', 'overlong_rally', 'missed_hits', 'extra_hits', 'wrong_player',
            'unlabelled_player', 'fully_correct')
BOUNDARY_RIGHT = frozenset(('missed_hits', 'extra_hits', 'wrong_player', 'unlabelled_player', 'fully_correct'))
HITS_RIGHT = frozenset(('wrong_player', 'unlabelled_player', 'fully_correct'))
DISCOVERY_BOUNDARY_BASIS = 'first_and_last_accepted_contacts'


@dataclass(frozen=True)
class ContactEndpointLimits:
    """Outward context allowed by a boundary-only confidence assessment."""

    before_seconds: float
    after_seconds: float

    def __post_init__(self) -> None:
        if any(not isfinite(seconds) or seconds < 0 for seconds in (self.before_seconds, self.after_seconds)):
            raise ValueError('contact endpoint limits must be finite and non-negative')


def contact_endpoint_stream(stream: dict) -> dict:
    """Use contact extents for discovery scoring, preserving all full-stream events."""
    sequences = []
    for sequence in stream['sequences']:
        events = sequence['events']
        if events:
            sequence = sequence | {'start_frame': events[0][0], 'end_frame': events[-1][0] + 1}
        sequences.append(sequence)
    return stream | {'sequences': sequences}


def discovery_runs(annotator_root: Path) -> dict[str, list[dict[str, dict]]]:
    """Per-video discovery streams for each run, keyed by item ID, with each found rally's confidence."""
    runs: dict[str, list[dict[str, dict]]] = {'annotator_discovery': [], 'annotator40_discovery': []}
    manifest = load_manifest(INPUTS)
    for seed in SEEDS:
        streams = {}
        for item in manifest['items']:
            stream = read_json(annotator_run('discovery', '32_videos', seed, annotator_root) / f'{item["item_id"]}.json.gz')
            streams[item['item_id']] = stream
        runs['annotator_discovery'].append(streams)
        runs['annotator40_discovery'].append({
            item['item_id']: read_json(
                annotator_run('discovery', '40_videos', seed, annotator_root) / f'{item["item_id"]}.json.gz')
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


def judge_found_rallies(
    sequences, rallies, tolerance: int, *, fps: float,
    endpoint_limits: ContactEndpointLimits | None = None,
    supplied: bool = False,
) -> list[dict]:
    """Judge discovery contact endpoints, or the provided bounds in supplied mode."""
    judged = []
    for sequence in sequences:
        frames = [event.frame for event in sequence.events]
        record = {'span_id': sequence.span_id, 'rally_id': None, 'hits': len(frames)}
        if not supplied:
            if not frames:
                boundary = judge_rally_boundaries(sequence, rallies, tolerance=tolerance)
                outcome = 'unjudgeable_rally' if boundary.outcome == 'unjudgeable_rally' else 'no_contacts'
                judged.append(record | {'outcome': outcome})
                continue
            sequence = replace(sequence, start_frame=frames[0], end_frame=frames[-1] + 1)
        boundary = judge_rally_boundaries(sequence, rallies, tolerance=tolerance)
        rally = boundary.rally
        if rally is not None:
            record['rally_id'] = rally.identity
        if boundary.outcome != 'boundaries_right':
            judged.append(record | {'outcome': boundary.outcome})
            continue
        assert rally is not None
        if endpoint_limits is not None:
            before = scale_frames(round(endpoint_limits.before_seconds * 30), fps)
            after = scale_frames(round(endpoint_limits.after_seconds * 30), fps)
            if sequence.start_frame < rally.frames[0] - before or sequence.end_frame - 1 > rally.frames[-1] + after:
                judged.append(record | {'outcome': 'overlong_rally'})
                continue
        matches = match_contacts(rally.frames, frames, tolerance)
        revised = alternate_sides((sequence,), sequence.events)[0][0]
        # Every error is counted, not just the first, so the write-up can say how many fixes a rally needs.
        record |= {
            'missed': len(rally.frames) - len(matches),
            'extra': len(frames) - len(matches),
            'wrong_player': sum(rally.sides[label] is not None and revised.events[prediction].side != rally.sides[label]
                                for label, prediction, _offset in matches),
        }
        if record['missed']:
            outcome = 'missed_hits'
        elif len(frames) > len(rally.frames):
            outcome = 'extra_hits'
        elif record['wrong_player']:
            outcome = 'wrong_player'
        elif any(side is None for side in rally.sides):
            outcome = 'unlabelled_player'
        else:
            outcome = 'fully_correct'
        judged.append(record | {'outcome': outcome})
    return judged


def rally_ladder(judged: list[dict], rallies) -> dict[str, int]:
    """Count labelled rallies reaching each rung through any one found rally."""
    best: dict[str, set[str]] = {rally.identity: set() for rally in rallies if rally.judgeable}
    for record in judged:
        if record['rally_id'] in best:
            best[record['rally_id']].add(record['outcome'])
    return {
        'boundaries_right': sum(bool(outcomes & BOUNDARY_RIGHT) for outcomes in best.values()),
        'hits_right': sum(bool(outcomes & HITS_RIGHT) for outcomes in best.values()),
        'fully_correct_matched': sum('fully_correct' in outcomes for outcomes in best.values()),
    }


def prepare_scoring_stream(item: dict, stream: dict, labels, windows: list[tuple[int, int]], *,
                           supplied: bool) -> tuple[dict, dict, int]:
    """Apply the benchmark's supplied bounds and excluded-rally mask once."""
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
    else:
        scored = contact_endpoint_stream(scored)
    return scored, unmasked, ignored


def score_stream_for_item(item: dict, stream: dict, tolerance: int, windows: list[tuple[int, int]], *,
                          supplied: bool) -> tuple[dict, list[dict]]:
    """Counts for one video and tolerance, plus the found-rally judgements."""
    labels = labels_for_item(item)
    scored, unmasked, ignored = prepare_scoring_stream(item, stream, labels, windows, supplied=supplied)
    _, contacts, predictions, rally_rows = score_video(
        item['item_id'], 'test', scored, labels, item['fps'], tolerance,
    )
    strict = sum(bool(row['fully_correct']) for row in rally_rows)
    sequences, _ = stream_from_payload(scored)
    judged = judge_found_rallies(sequences, labels, scale_frames(tolerance, item['fps']), fps=item['fps'],
                                supplied=supplied)
    # Separate endpoint context from the contact-timing tolerance.
    no_allowance = rally_ladder(judge_found_rallies(
        sequences, labels, 0, fps=item['fps'], supplied=supplied,
    ), labels)['boundaries_right']
    outside = 0 if supplied else sum(
        not any(span['start_frame'] <= event[0] < span['end_frame'] for span in item['spans'])
        for event in scored['events'])
    unjudgeable = sum(record['outcome'] == 'unjudgeable_rally' for record in judged)
    counts = {
        'labelled_rallies': len(rally_rows), 'labelled_contacts': len(contacts),
        'predicted_contacts': len(predictions),
        'matched_contacts': sum(bool(row['matched']) for row in contacts),
        'correct_side_contacts': sum(bool(row['player_correct']) for row in contacts),
        'fully_correct_strict': strict,
        'exact_hits_strict': sum(bool(row['exact_sequence']) for row in rally_rows),
        'predicted_outside_windows': outside,
        'ignored_predictions': ignored,
        'found_rallies': len(sequences) - unjudgeable,
        # Subtract the same unjudgeable detections so the difference counts masking only.
        'found_rallies_before_ignore': len(unmasked['sequences']) - unjudgeable,
        'boundaries_right_no_allowance': no_allowance,
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
                'correct_side_contacts': totals['correct_side_contacts']}
        for key, value in ours.items():
            if int(value) != int(reference[key]):
                raise ValueError(f'{name} ±{tolerance}: {key} {int(value)} != saved {reference[key]}')


def rates(totals: pd.Series) -> dict[str, float]:
    labelled, predicted = totals['labelled_contacts'], totals['predicted_contacts']
    matched, attributed = totals['matched_contacts'], totals['correct_side_contacts']
    rallies = totals['labelled_rallies']
    return {
        'precision': 100 * matched / predicted if predicted else 0.0, 'recall': 100 * matched / labelled,
        'f1': 100 * 2 * matched / (predicted + labelled),
        'player_precision': 100 * attributed / predicted if predicted else 0.0,
        'player_recall': 100 * attributed / labelled,
        'player_f1': 100 * 2 * attributed / (predicted + labelled),
        'fully_correct': 100 * totals['fully_correct_matched'] / rallies,
        'fully_correct_strict': 100 * totals['fully_correct_strict'] / rallies,
        'hits_right': 100 * totals['hits_right'] / rallies,
        'boundaries_right': 100 * totals['boundaries_right'] / rallies,
        'boundaries_right_no_allowance': 100 * totals['boundaries_right_no_allowance'] / rallies,
        'ignored_found_rallies': totals['found_rallies_before_ignore'] - totals['found_rallies'],
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
    thresholds = np.round(np.linspace(0, 1, 101), 2).tolist()
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
    parser.add_argument('--annotator-runs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--monotrack-runs', type=Path, default=MONOTRACK_RUNS,
                        help='five_seed.py output with validation-selected test runs')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(INPUTS)
    total_rallies = sum(len(item['spans']) for item in manifest['items'])
    windows = {video: left_out_windows(rallies) for video, rallies in test_left_out().items()}

    conditions = {}
    pool = ProcessPoolExecutor(max_workers=args.workers)
    supplied = supplied_runs(args.annotator_runs, args.monotrack_runs)
    selected = read_json(args.monotrack_runs / 'selected.json.gz')
    if selected['selection_split'] != 'validation' or selected['seeds'] != list(SEEDS):
        raise ValueError('MonoTrack results must use all five validation-selected seeds')
    for name, runs in supplied.items():
        streams = [{item['item_id']: read_json(run / f'{item["item_id"]}.json.gz')['stream'] for item in manifest['items']}
                   for run in runs]
        conditions[name] = score_condition(
            name, streams, supplied=True, saved=runs, windows=windows, pool=pool,
        )
        print(name, 'scored', flush=True)
    for name, streams in discovery_runs(args.annotator_runs).items():
        conditions[name] = score_condition(
            name, streams, supplied=False, saved=None, windows=windows, pool=pool,
        )
        print(name, 'scored', flush=True)

    # Contact totals must match the originating run under either boundary policy.
    for name, models in (('annotator_discovery', '32_videos'), ('annotator40_discovery', '40_videos')):
        for seed, run in zip(SEEDS, conditions[name]['runs'], strict=True):
            saved = annotator_run('discovery', models, seed, args.annotator_runs) / 'scores.json.gz'
            check_against_saved(name, run['per_video'], read_json(saved))
    summary = {name: summarise(condition) for name, condition in conditions.items()}

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
    pool.shutdown()
    write_json(args.out / 'evaluation-policy.json.gz', {
        'annotator_runs': str(args.annotator_runs.resolve()),
        'boundary_policy': {'before_seconds': None, 'after_seconds': None},
        'boundary_basis': DISCOVERY_BOUNDARY_BASIS,
    })
    (args.out / 'contact_summary.json').write_text(json.dumps({'summary': summary, 'paired': paired}, indent=1))
    print(json.dumps({name: {key: round(value['mean'], 2) for key, value in condition['10'].items()}
                      for name, condition in summary.items() if not name.startswith('_')}, indent=1))


if __name__ == '__main__':
    main()
