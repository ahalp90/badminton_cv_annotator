"""Check whether the saved shuttle guard flags mark the false final hits.

Compares guard codes at the 233 false final hits with the real final hits of
fully correct rallies, then counts what dropping a flagged final hit would
rescue and break. Reads ``results/shuttle_guard_runs.json.gz`` from
``pull_shuttle_guard_runs.py`` and the evaluation tables.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / 'results'
# Video 22's guard was unavailable, so its saved codes are all zero rather than clean.
GUARD_UNAVAILABLE = {'22'}
WINDOWS = (0, 2, 5)


def test_rows(name: str) -> pd.DataFrame:
    table = pd.read_csv(RESULTS / f'{name}.csv.gz', dtype={'video': str})
    return table[(table.split == 'test') & (table.tolerance_base30 == 10)]


def frame_values(info: dict[str, Any], key: str) -> np.ndarray:
    """Expand one video's saved runs into one value per source frame."""
    values = np.zeros(info['frame_count'], dtype=np.int8)
    for run in info[key]:
        values[run[0]:run[1]] = run[2] if len(run) == 3 else 1
    return values


def flag_hits(hits: pd.DataFrame) -> pd.DataFrame:
    """Add the guard code, nearby flags, fill and visibility at each emitted hit."""
    with gzip.open(RESULTS / 'shuttle_guard_runs.json.gz', 'rt', encoding='utf-8') as handle:
        runs = json.load(handle)['videos']
    hits = hits.assign(code=0, filled=False, hidden=False, **{f'flagged_{radius}': False for radius in WINDOWS})
    for video, part in hits.groupby('video'):
        codes = frame_values(runs[video], 'guard_runs')
        filled = frame_values(runs[video], 'inpaint_runs').astype(bool)
        hidden = frame_values(runs[video], 'hidden_runs').astype(bool)
        frames = part.source_frame.to_numpy()
        hits.loc[part.index, 'code'] = codes[frames]
        hits.loc[part.index, 'filled'] = filled[frames]
        hits.loc[part.index, 'hidden'] = hidden[frames]
        for radius in WINDOWS:
            near = [bool((codes[max(frame - radius, 0):frame + radius + 1] > 0).any()) for frame in frames]
            hits.loc[part.index, f'flagged_{radius}'] = near
    return hits


def final_hits(hits: pd.DataFrame, rallies: pd.DataFrame, tail: pd.DataFrame) -> pd.DataFrame:
    """Return the final hits of spans holding a false or a fully correct rally.

    Also returns the real hit just before each false one, which shares its rally and track.
    """
    in_span_order = hits.sort_values('source_frame').groupby(['video', 'span_id'])
    last_in_span = in_span_order.tail(1)

    false_hits = last_in_span.merge(tail[['video', 'query_frame']], left_on=['video', 'source_frame'],
                                    right_on=['video', 'query_frame'], validate='one_to_one')
    # Every sole-error extra follows the last label, so it must be its span's final emitted hit.
    assert len(false_hits) == len(tail), (len(false_hits), len(tail))
    false_spans = false_hits[['video', 'span_id']]
    real_before_false = in_span_order.nth(-2).merge(false_spans, on=['video', 'span_id'])
    assert len(real_before_false) == len(tail) and real_before_false.matched.all()

    correct = rallies.loc[rallies.fully_correct.astype(bool), ['video', 'rally_id']]
    correct_spans = hits.merge(correct, on=['video', 'rally_id'])[['video', 'span_id']].drop_duplicates()
    real_finals = last_in_span.merge(correct_spans, on=['video', 'span_id'])
    assert real_finals.matched.all()
    return pd.concat([false_hits.assign(group='false final hit'),
                      real_before_false.assign(group='real hit just before the false one'),
                      real_finals.assign(group='real final hit, fully correct rally')], ignore_index=True)


def main() -> None:
    hits = flag_hits(test_rows('predictions'))
    rallies = test_rows('rallies')
    tail = pd.read_csv(RESULTS / 'tail_rallies.csv.gz', dtype={'video': str})
    tail = tail[tail.sole_error]
    finals = final_hits(hits, rallies, tail)
    finals = finals[~finals.video.isin(GUARD_UNAVAILABLE)]
    guarded_hits = hits[~hits.video.isin(GUARD_UNAVAILABLE)]

    print('Guard flags at final hits (video 22 left out)')
    rows = []
    for name, part in [*finals.groupby('group'), ('every emitted hit that matches a label', guarded_hits[guarded_hits.matched])]:
        row = {'group': name, 'hits': len(part)}
        for radius in WINDOWS:
            row[f'flagged within {radius}'] = f'{part[f"flagged_{radius}"].sum()} ({100 * part[f"flagged_{radius}"].mean():.1f}%)'
        row['proof code at frame'] = int((part.code == 1).sum())
        row['inpainted at frame'] = f'{part.filled.sum()} ({100 * part.filled.mean():.1f}%)'
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))

    print('\nDropping a span\'s last hit when it is flagged (fully correct rallies now: '
          f'{int(rallies.fully_correct.sum())} of {len(rallies)})')
    is_false = finals.group == 'false final hit'
    is_correct_final = finals.group == 'real final hit, fully correct rally'
    every_last_hit = guarded_hits.sort_values('source_frame').groupby(['video', 'span_id']).tail(1)
    rows = []
    for radius in WINDOWS:
        flagged = finals[f'flagged_{radius}']
        rescued, broken = int((flagged & is_false).sum()), int((flagged & is_correct_final).sum())
        dropped = every_last_hit[every_last_hit[f'flagged_{radius}']]
        rows.append({'flag window': f'±{radius} frames', 'rallies rescued': rescued, 'rallies broken': broken,
                     'net': rescued - broken, 'fully correct after': int(rallies.fully_correct.sum()) + rescued - broken,
                     'all spans: false hits dropped': int((~dropped.matched).sum()),
                     'all spans: real hits dropped': int(dropped.matched.sum())})
    print(pd.DataFrame(rows).to_string(index=False))

    codes = finals[is_false & finals.flagged_0].code.value_counts().sort_index()
    print('\nGuard code at flagged false final hits:', {f'code {code}': int(count) for code, count in codes.items()})

    print('\nFalse final hits flagged on the frame, by official ending')
    false_hits = finals[is_false].merge(tail[['video', 'query_frame', 'ending']], left_on=['video', 'source_frame'],
                                        right_on=['video', 'query_frame'])
    endings = false_hits.groupby('ending').agg(rallies=('flagged_0', 'size'), flagged=('flagged_0', 'sum'))
    print(endings.sort_values('rallies', ascending=False).to_string())


if __name__ == '__main__':
    main()
