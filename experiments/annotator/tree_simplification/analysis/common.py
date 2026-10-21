"""Helpers shared by the analysis scripts: reading files, re-scoring predictions and checking evidence.

Every checker uses one rule. Integer counts must be equal. Floats may differ by at most
``FLOAT_TOLERANCE``, which only absorbs a different order of the same arithmetic.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import re
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib

from experiments.annotator.heuristic_only_tune.data import (
    COUNT_FIELDS,
    SearchVideo,
    load_original_videos,
    score_stream,
)

REPOSITORY = Path(__file__).resolve().parents[4]
RESULTS = REPOSITORY / 'experiments/annotator/tree_simplification/results'
SHOTS_MASTER = REPOSITORY / 'training/data/shuttleset/annotations/shots_master.csv'
SEED_OFFSETS = (0, 100, 200, 300, 400)
TOLERANCES = ('5', '10')
FLOAT_TOLERANCE = 1e-12
# Pooled per-seed metrics that rank the preprocessing configurations, in their recorded order.
METRICS = ('f1_10', 'fully_correct_rallies_10', 'f1_5', 'fully_correct_rallies_5')
# Matplotlib names markers and clip paths with random hex ids unless a salt is set.
GENERATED_SVG_ID = re.compile(r'(id="|#)([mp])[0-9a-f]{10}')
VALIDATION_VIDEOS = ('sset_18', 'sset_22', 'sset_24', 'sset_25', 'sset_30', 'sset_31', 'sset_39', 'sset_40')


def read_json(path: Path) -> Any:
    """Read a JSON file, gzip-compressed when its name ends in ``.gz``."""
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def write_json_gz(path: Path, document: Any) -> None:
    """Write sorted, indented JSON with a fixed gzip timestamp, so a rebuild gives the same bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(document, indent=2, sort_keys=True) + '\n').encode()
    path.write_bytes(gzip.compress(payload, mtime=0))


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def attributed_f1(counts: dict[str, Any]) -> float:
    """Correct-player F1: twice the correct-player matches over predicted plus labelled contacts."""
    return 2 * counts['correct_side_contacts'] / (counts['predicted_contacts'] + counts['labelled_contacts'])


def pooled_counts(per_video: dict[str, Any], tolerance: str) -> dict[str, Any]:
    """Sum one tolerance's counts over videos, then add the pooled correct-player F1."""
    counts: dict[str, Any] = {name: sum(video[tolerance][name] for video in per_video.values()) for name in COUNT_FIELDS}
    counts['attributed_f1'] = attributed_f1(counts)
    return counts


def pooled_both(per_video: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {tolerance: pooled_counts(per_video, tolerance) for tolerance in TOLERANCES}


def seed_metrics(per_video: dict[str, Any]) -> dict[str, Any]:
    """Pooled correct-player F1 and fully correct rallies at both tolerances, in the recorded key order."""
    values: dict[str, Any] = {}
    for tolerance in ('10', '5'):
        counts = [video[tolerance] for video in per_video.values()]
        totals = {
            name: sum(row[name] for row in counts)
            for name in ('correct_side_contacts', 'predicted_contacts', 'labelled_contacts')
        }
        values[f'f1_{tolerance}'] = attributed_f1(totals)
        values[f'fully_correct_rallies_{tolerance}'] = sum(row['fully_correct_rallies'] for row in counts)
    return values


def validation_videos(shots_master: Path) -> dict[str, SearchVideo]:
    """The eight validation videos and their labels, from the tracked 32/8 split."""
    videos = {video.identity: video for video in load_original_videos(shots_master) if video.split == 'validation'}
    if tuple(sorted(videos)) != VALIDATION_VIDEOS:
        raise ValueError(f'Expected the eight validation videos, found {sorted(videos)}')
    return videos


def _score_file(task: tuple[SearchVideo, Path]) -> dict[str, dict[str, int]]:
    video, path = task
    return score_stream(video, read_json(path))


def rescore_folders(folders: dict[str, Path], videos: dict[str, SearchVideo]) -> dict[str, dict[str, Any]]:
    """Score every prediction file in each folder; each folder must hold one file per video.

    Returns per-video counts keyed first by the caller's folder key, then by video.
    """
    tasks = []
    for key, folder in folders.items():
        found = sorted(path.name.removesuffix('.json.gz') for path in folder.glob('*.json.gz'))
        if found != sorted(videos):
            raise ValueError(f'{folder}: expected one prediction file per video, found {found}')
        tasks.extend((key, identity, folder / f'{identity}.json.gz') for identity in sorted(videos))
    with ProcessPoolExecutor() as pool:
        scores = list(pool.map(_score_file, [(videos[identity], path) for _key, identity, path in tasks], chunksize=4))
    per_folder: dict[str, dict[str, Any]] = {key: {} for key in folders}
    for (key, identity, _path), score in zip(tasks, scores, strict=True):
        per_folder[key][identity] = score
    return per_folder


def _compare(actual: Any, expected: Any, where: str, skip: frozenset[str], problems: list[str]) -> int:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            problems.append(f'{where}: expected an object, found {type(actual).__name__}')
            return 0
        expected_keys = set(expected) - skip
        actual_keys = set(actual) - skip
        if expected_keys != actual_keys:
            problems.append(
                f'{where}: keys differ; missing {sorted(expected_keys - actual_keys)}, '
                f'extra {sorted(actual_keys - expected_keys)}'
            )
        return sum(
            _compare(actual[key], expected[key], f'{where}.{key}', skip, problems)
            for key in sorted(expected_keys & actual_keys)
        )
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            problems.append(f'{where}: expected a list of {len(expected)}, found {actual!r:.80}')
            return 0
        return sum(
            _compare(actual_item, expected_item, f'{where}[{index}]', skip, problems)
            for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True))
        )
    # bool is a subclass of int, so it must be compared before the numeric rule.
    expected_is_number = isinstance(expected, (int, float)) and not isinstance(expected, bool)
    actual_is_number = isinstance(actual, (int, float)) and not isinstance(actual, bool)
    if expected_is_number and actual_is_number:
        if isinstance(expected, int) and isinstance(actual, int):
            same = expected == actual
        else:
            same = math.isfinite(actual) and abs(actual - expected) <= FLOAT_TOLERANCE
    else:
        same = type(actual) is type(expected) and actual == expected
    if not same:
        problems.append(f'{where}: expected {expected!r}, found {actual!r}')
    return 1


def check(actual: Any, expected: Any, label: str, *, skip: Iterable[str] = ()) -> int:
    """Compare a rebuilt value with the committed one and fail loudly on any difference.

    ``skip`` names keys, at any depth, that record how a run was made rather than a result.
    Returns the number of values compared.
    """
    problems: list[str] = []
    compared = _compare(actual, expected, label, frozenset(skip), problems)
    if problems:
        shown = '\n  '.join(problems[:20])
        raise ValueError(f'{label}: {len(problems)} differences from the committed evidence:\n  {shown}')
    print(f'{label}: {compared} values match', flush=True)
    return compared


def normalised_svg(text: str) -> str:
    """SVG text without trailing spaces or Matplotlib's generated ids, which carry no drawing."""
    stripped = '\n'.join(line.rstrip() for line in text.splitlines())
    return GENERATED_SVG_ID.sub(r'\1\2-generated', stripped)


def save_svg(figure: Any, output: Path) -> None:
    """Write a figure as SVG with no date, fixed ids and no trailing spaces, so rebuilds give the same bytes."""
    buffer = io.StringIO()
    with matplotlib.rc_context({'svg.hashsalt': 'tree-simplification'}):
        figure.savefig(buffer, format='svg', metadata={'Date': None, 'Creator': 'Matplotlib'})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(line.rstrip() for line in buffer.getvalue().splitlines()) + '\n')


def check_svg(rebuilt: Path, tracked: Path) -> None:
    """Fail unless the rebuilt figure draws the same thing as the tracked one."""
    if normalised_svg(rebuilt.read_text()) != normalised_svg(tracked.read_text()):
        raise ValueError(f'{rebuilt} draws something different from {tracked}')
    print(f'{rebuilt.name}: draws the same figure as {tracked.name}', flush=True)
