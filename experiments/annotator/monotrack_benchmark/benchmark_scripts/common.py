"""Shared input and result formats for the MonoTrack comparison."""
from __future__ import annotations

import gzip
import json
import lzma
import sys
import tomllib
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path[:0] = [str(REPO / 'src'), str(REPO)]
VIDEO_IDS = (*range(8, 14), *range(16, 45), *range(46, 56), 57)
POPULATION = {'videos': 46, 'rallies': 3327, 'contacts': 37184}


def read_json(path: Path) -> Any:
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt') as stream:
        json.dump(value, stream, allow_nan=False, indent=2)


def lock_record(path: Path, value: Any) -> None:
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f'{path}: inputs or selection protocol changed; use a fresh output directory')
    else:
        write_json(path, value)


def read_array(path: Path) -> Any:
    import numpy as np

    with lzma.open(path, 'rb') as stream:
        return np.load(stream, allow_pickle=False)


def load_config(path: Path) -> dict[str, Any]:
    with path.open('rb') as stream:
        config = tomllib.load(stream)
    for key, value in config['paths'].items():
        candidate = Path(value).expanduser()
        config['paths'][key] = candidate if candidate.is_absolute() else REPO / candidate
    return config


def load_manifest(root: Path) -> dict[str, Any]:
    manifest = read_json(root / 'manifest.json.gz')
    items = manifest['items']
    counts = {'videos': len(items), 'rallies': sum(len(item['spans']) for item in items),
              'contacts': sum(len(span['label_frames']) for item in items for span in item['spans'])}
    if tuple(sorted(item['video_id'] for item in items)) != VIDEO_IDS or counts != POPULATION:
        raise ValueError(f'Evaluation population changed: {counts}')
    return manifest


def labels_for_item(item: dict[str, Any]) -> tuple[Any, ...]:
    from annotator.outcomes.point_winner import Half
    from annotator.training.sequences import LabelledRally

    return tuple(LabelledRally(span['rally_id'], tuple(span['label_frames']),
                              tuple(None if side is None else Half(side) for side in span['label_sides']))
                 for span in item['spans'])


def fixed_stream(events: list[list[Any]], spans: list[dict[str, Any]]) -> dict[str, Any]:
    """Use supplied windows for rally judgement, retaining all events for contact scoring."""
    sequences = []
    for span in spans:
        own_events = [event for event in events if span['start_frame'] <= event[0] < span['end_frame']]
        sequences.append({**{key: span[key] for key in ('span_id', 'start_frame', 'end_frame')}, 'events': own_events})
    return {'sequences': sequences, 'events': events}


def court_records(root: Path, item: dict[str, Any]) -> list[dict[str, Any]]:
    scenes = read_json(root / item['item_id'] / 'court_evidence.json.gz')['scene_records']
    by_bounds = {(scene['start_frame'], scene['end_frame']): scene for scene in scenes}
    return [by_bounds[span['start_frame'], span['end_frame']] for span in item['spans']]
