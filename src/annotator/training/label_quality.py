"""Rallies whose contact labels cannot support whole-rally supervision."""

import csv
import gzip
from dataclasses import replace
from functools import cache
from pathlib import Path

from annotator.training.sequences import LabelledRally

RALLY_EXCLUSIONS = Path(__file__).resolve().parents[3] / 'data/shuttleset/rally_exclusions.csv.gz'
ORIGINAL_SETS = RALLY_EXCLUSIONS.parent / 'set'


@cache
def load_rally_exclusions(path: Path = RALLY_EXCLUSIONS) -> frozenset[tuple[int, str]]:
    """Read the original ShuttleSet video and set:rally keys, retaining all reasons on disk."""
    with gzip.open(path, 'rt', newline='') as handle:
        return frozenset((int(row['vid']), f"set{int(row['set'])}:{int(row['rally'])}")
                         for row in csv.DictReader(handle))


def mark_unjudgeable_rallies(rallies: tuple[LabelledRally, ...], video_id: int) -> tuple[LabelledRally, ...]:
    """Attach whole-rally reliability without removing any contact supervision."""
    spans = unjudgeable_spans()
    return tuple(replace(rally, judgeable=(video_id, rally.identity) not in spans,
                         unjudgeable_span=spans.get((video_id, rally.identity))) for rally in rallies)


@cache
def unjudgeable_spans() -> dict[tuple[int, str], tuple[int, int]]:
    """Read original contact extents without restoring dropped contact targets."""
    exclusions = load_rally_exclusions()
    spans = {}
    with (ORIGINAL_SETS / 'match.csv').open(newline='') as handle:
        matches = list(csv.DictReader(handle))
    for match in matches:
        video = int(match['id'])
        for path in sorted((ORIGINAL_SETS / match['video']).glob('set*.csv')):
            with path.open(newline='') as handle:
                for row in csv.DictReader(handle):
                    key = (video, f"{path.stem}:{int(row['rally'])}")
                    if key not in exclusions:
                        continue
                    frame = int(float(row['frame_num']))
                    first, last = spans.get(key, (frame, frame + 1))
                    spans[key] = (min(first, frame), max(last, frame + 1))
    if set(spans) != exclusions:
        raise ValueError('Rally exclusions do not match the original ShuttleSet annotations')
    return spans
