"""Score all contacts and judge rallies using the common supplied windows."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from common import fixed_stream, labels_for_item, load_manifest, read_json, write_json

COUNT_FIELDS = ('labelled_rallies', 'labelled_contacts', 'predicted_contacts', 'matched_contacts',
                'correct_side_contacts', 'fully_correct_rallies', 'predicted_contacts_outside_windows')


def add_rates(counts: dict[str, Any]) -> dict[str, Any]:
    matched, predicted, labelled = (counts[key] for key in
                                   ('matched_contacts', 'predicted_contacts', 'labelled_contacts'))
    return {**counts, 'precision': matched / predicted if predicted else 0.0,
            'recall': matched / labelled if labelled else 0.0,
            'f1': 2 * matched / (predicted + labelled) if predicted + labelled else 0.0}


def score_item(item: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    from experiments.annotator.annotator_closeout_new_courts.scripts.evaluate_saved import (
        score_video,
    )

    events = result['stream']['events']
    inside_windows = [any(span['start_frame'] <= event[0] < span['end_frame'] for span in item['spans'])
                      for event in events]
    outside_count = len(events) - sum(inside_windows)
    if result['boundaries'] == 'provided':
        events = [event for event, inside in zip(events, inside_windows, strict=True) if inside]
        outside_count = 0
    elif result['boundaries'] != 'discovered':
        raise ValueError('Unknown rally-boundary condition')
    # Discovery keeps out-of-window events here: they must count as false positives.
    stream = fixed_stream(events, item['spans'])
    scores = {}
    for tolerance in (10, 5):
        _, contacts, predictions, rallies = score_video(
            item['item_id'], 'test', stream, labels_for_item(item), item['fps'], tolerance,
        )
        scores[str(tolerance)] = add_rates({
            'labelled_rallies': len(rallies), 'labelled_contacts': len(contacts),
            'predicted_contacts': len(predictions), 'matched_contacts': sum(bool(row['matched']) for row in contacts),
            'correct_side_contacts': sum(bool(row['player_correct']) for row in contacts),
            'fully_correct_rallies': sum(bool(row['fully_correct']) for row in rallies),
            'predicted_contacts_outside_windows': outside_count,
        })
    return scores


def score_run(inputs: Path, run: Path) -> dict[str, Any]:
    manifest = load_manifest(inputs)
    totals = {str(tolerance): dict.fromkeys(COUNT_FIELDS, 0) for tolerance in (10, 5)}
    by_video = {}
    failed_videos = []
    failed_rallies = 0
    failure_stages: Counter[str] = Counter()
    for item in manifest['items']:
        # Missing files indicate an unfinished run; explicit failures contain an empty stream.
        result = read_json(run / f'{item["item_id"]}.json.gz')
        if result['item_id'] != item['item_id']:
            raise ValueError('Result video does not match its filename')
        by_video[item['item_id']] = score_item(item, result)
        if result['status'] == 'failed':
            failed_videos.append(item['item_id'])
        failed_rallies += len(result.get('failed_rallies', []))
        failure_stages.update(failure.get('stage', failure['reason'])
                              for failure in result.get('failed_rallies', []))
        for tolerance, counts in totals.items():
            for key in COUNT_FIELDS:
                counts[key] += by_video[item['item_id']][tolerance][key]
    summary = {'by_tolerance': {key: add_rates(value) for key, value in totals.items()}, 'by_video': by_video,
               'failed_videos': failed_videos, 'failed_rallies': failed_rallies,
               'failure_stages': dict(failure_stages)}
    write_json(run / 'scores.json.gz', summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    summary = score_run(args.inputs, args.run)
    print(summary['by_tolerance'], flush=True)


if __name__ == '__main__':
    main()
