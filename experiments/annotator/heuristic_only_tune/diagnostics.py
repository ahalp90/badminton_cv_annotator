"""Recompute the diagnostic counts quoted in RESULTS.md and KNOWN_RALLIES.md.

Reads only committed predictions, scores and benchmark inputs. Every count
uses the ±10 base-30 frame allowance. Rally judgements use the original
±1-second scoring windows, including for runs with wider inference context.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from annotator.sequence.contacts import scale_frames
from annotator.training.sequences import match_contacts
from experiments.annotator.annotator_closeout_new_courts.scripts.evaluate_saved import (
    score_video,
)
from experiments.annotator.heuristic_only_tune.data import load_test_videos
from experiments.annotator.monotrack_benchmark.benchmark_scripts.common import (
    fixed_stream,
    labels_for_item,
    load_manifest,
    read_json,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
CONTEXT = RESULTS / 'context'
BENCHMARK = HERE.parent / 'monotrack_benchmark' / 'benchmark_results'
INPUTS = BENCHMARK / 'inputs'
TOLERANCE_BASE30 = 10
EDGE_FRAMES = 10


@dataclass(frozen=True)
class WindowRally:
    """One labelled rally judged inside its supplied scoring window."""

    fully_correct: bool
    failure_reasons: tuple[str, ...]
    frames: tuple[int, ...]
    sides: tuple[str | None, ...]
    extra_frames: tuple[int, ...]
    first_label: int
    last_label: int
    start_frame: int
    end_frame: int


def boundary_failures(stage: str) -> dict[str, int]:
    """Count discovered sequences that touch one labelled rally but omit some of its contacts.

    :param stage: committed test stage, ``test`` (heuristic) or ``test_full``.
    :return: sequences touching exactly one rally, and how many of those omit a label.
    """
    streams = read_json(RESULTS / stage / 'predictions.json.gz')['streams']
    touching_one = 0
    omitting = 0
    for video in load_test_videos(INPUTS):
        proposals, _contacts, _predictions, _rallies = score_video(
            video.identity, video.split, streams[video.identity], video.rallies, video.fps, TOLERANCE_BASE30,
        )
        for row in proposals:
            if row['overlapping_rallies'] == 1:
                touching_one += 1
                omitting += row['boundary_error']
    return {'sequences_touching_one_rally': touching_one, 'omitting_a_labelled_contact': omitting}


def window_rallies(run: Path) -> dict[tuple[str, str], WindowRally]:
    """Judge each rally inside its original window, as the core benchmark scorer does.

    :param run: directory of per-video benchmark results.
    :return: one judgement per (video, rally) pair.
    """
    judged = {}
    for item in load_manifest(INPUTS)['items']:
        events = read_json(run / f'{item["item_id"]}.json.gz')['stream']['events']
        tolerance = scale_frames(TOLERANCE_BASE30, item['fps'])
        proposals, _contacts, _predictions, rally_rows = score_video(
            item['item_id'], 'test', fixed_stream(events, item['spans']), labels_for_item(item),
            item['fps'], TOLERANCE_BASE30,
        )
        proposal_by_span = {row['span_id']: row for row in proposals}
        correct_by_rally = {row['rally_id']: row['fully_correct'] for row in rally_rows}
        for span in item['spans']:
            proposal = proposal_by_span[span['span_id']]
            own = [event for event in events if span['start_frame'] <= event[0] < span['end_frame']]
            frames = tuple(event[0] for event in own)
            matches = match_contacts(span['label_frames'], frames, tolerance)
            matched = {prediction for _label, prediction, _offset in matches}
            judged[item['item_id'], span['rally_id']] = WindowRally(
                fully_correct=correct_by_rally[span['rally_id']],
                failure_reasons=tuple(name for name in ('missing', 'extra', 'wrong_player') if proposal[name]),
                frames=frames,
                sides=tuple(event[2] for event in own),
                extra_frames=tuple(frame for index, frame in enumerate(frames) if index not in matched),
                first_label=span['label_frames'][0],
                last_label=span['label_frames'][-1],
                start_frame=span['start_frame'],
                end_frame=span['end_frame'],
            )
    return judged


def compare_windows(before: Path, after: Path) -> dict[str, Any]:
    """Describe which rallies change outcome between two runs, and why the lost ones fail.

    :param before: per-video results for the reference condition.
    :param after: per-video results for the changed condition.
    :return: rally transitions, failure reasons and extra-contact locations.
    """
    old = window_rallies(before)
    new = window_rallies(after)
    gained = 0
    lost = []
    for key, rally in old.items():
        gained += not rally.fully_correct and new[key].fully_correct
        if rally.fully_correct and not new[key].fully_correct:
            lost.append(new[key])

    lost_with_extras = 0
    lost_only_from_extras = 0
    extras_near_edge = 0
    extra_positions: Counter[str] = Counter()
    for rally in lost:
        lost_with_extras += 'extra' in rally.failure_reasons
        lost_only_from_extras += rally.failure_reasons == ('extra',)
        for frame in rally.extra_frames:
            # Windows are half-open, so the last frame inside is end_frame - 1.
            extras_near_edge += min(frame - rally.start_frame, rally.end_frame - 1 - frame) <= EDGE_FRAMES
            if frame < rally.first_label:
                extra_positions['before_first_label'] += 1
            elif frame > rally.last_label:
                extra_positions['after_last_label'] += 1
            else:
                extra_positions['between_labels'] += 1

    different_frames = 0
    same_frames_different_players = 0
    for key, rally in old.items():
        if rally.frames != new[key].frames:
            different_frames += 1
        elif rally.sides != new[key].sides:
            same_frames_different_players += 1
    return {
        'gained_fully_correct': gained,
        'lost_fully_correct': len(lost),
        'lost_with_extra_contacts': lost_with_extras,
        'lost_only_from_extra_contacts': lost_only_from_extras,
        'extra_contacts_in_lost_rallies': extra_positions.total(),
        'extra_positions_in_lost_rallies': dict(extra_positions),
        'lost_extras_within_10_frames_of_window_edge': extras_near_edge,
        'rallies_with_different_contact_frames': different_frames,
        'rallies_with_same_frames_but_different_players': same_frames_different_players,
    }


def pooled_counts(before: Path, after: Path) -> dict[str, int]:
    """Pooled in-window contact counts for two scored runs.

    :param before: scored run directory for the reference condition.
    :param after: scored run directory for the changed condition.
    :return: matched, unmatched, missing and correctly attributed counts for each
        run, and the change between them.
    """
    runs = {}
    for label, run in (('before', before), ('after', after)):
        row = read_json(run / 'scores.json.gz')['by_tolerance'][str(TOLERANCE_BASE30)]
        # Labels lie well inside their windows, so every matched prediction is in a window.
        in_windows = row['predicted_contacts'] - row['predicted_contacts_outside_windows']
        runs[label] = {
            'matched': row['matched_contacts'],
            'unmatched_in_windows': in_windows - row['matched_contacts'],
            'missing': row['labelled_contacts'] - row['matched_contacts'],
            'correct_player': row['correct_side_contacts'],
            'fully_correct_rallies': row['fully_correct_rallies'],
        }
    counts = {}
    for name, value in runs['before'].items():
        counts[f'{name}_before'] = value
        counts[f'{name}_after'] = runs['after'][name]
        counts[f'{name}_change'] = runs['after'][name] - value
    return counts


def compare_runs(before: Path, after: Path) -> dict[str, Any]:
    """Combine pooled counts and per-rally transitions for two runs."""
    return {**pooled_counts(before, after), **compare_windows(before, after)}


def main() -> None:
    heuristic_one_second = RESULTS / 'known_our_courts'
    full_one_second = BENCHMARK.parent / 'archive' / 'results' / 'comparison' / 'ours_our_courts'
    report = {
        'RESULTS: heuristic discovered sequences': boundary_failures('test'),
        'RESULTS: full annotator discovered sequences': boundary_failures('test_full'),
        'KNOWN_RALLIES: heuristic discovery to supplied windows':
            compare_runs(RESULTS / 'discovery_benchmark', heuristic_one_second),
        'KNOWN_RALLIES: heuristic 1/1 to 1/2 seconds': pooled_counts(heuristic_one_second, CONTEXT / 'pre1_post2'),
        'KNOWN_RALLIES: heuristic 1/2 to 1/3 seconds': compare_runs(CONTEXT / 'pre1_post2', CONTEXT / 'pre1_post3'),
        'KNOWN_RALLIES: heuristic 2/2 to 2/3 seconds': compare_runs(CONTEXT / 'pre2_post2', CONTEXT / 'pre2_post3'),
        'KNOWN_RALLIES: full annotator 1/1 to 2/3 seconds':
            compare_runs(full_one_second, CONTEXT / 'pre2_post3_full_ours'),
    }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
