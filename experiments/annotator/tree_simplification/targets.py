"""Prepare a five-frame target view aligned to fixed expanded rally pools."""

from __future__ import annotations

import argparse
import lzma
import multiprocessing
import os
import sys
import tempfile
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))

from experiments.annotator.tree_simplification import pilot

# isort: split

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from annotator.sequence.boundaries import BOUNDARY_PADDING_AT_30_FPS
from annotator.sequence.contacts import scale_frames
from annotator.training.workflow import whole_rally_correctness
from experiments.annotator.tree_simplification.options import (
    ExpandedPool,
    finish_in_initial_context,
)

SCHEMA = 'tree-simplification-five-frame-targets/1'


@dataclass(frozen=True)
class TargetJob:
    identity: str
    group: str
    split: str
    pool_path: Path
    pools_root: Path
    pools_manifest: dict[str, Any]
    output: Path


def raw_five_targets(pool: ExpandedPool) -> np.ndarray:
    """Return canonical whole-rally labels at five frames per 30 fps, one per row."""
    video = pool.original
    stream_frames = np.asarray([event.frame for event in video.events], dtype=np.int64)
    position_by_span = {sequence.span_id: position for position, sequence in enumerate(video.initial)}
    targets = np.empty(len(pool.sequences), dtype=np.int8)
    padding = scale_frames(BOUNDARY_PADDING_AT_30_FPS, video.fps)
    for span_id, rows in pool.span_rows.items():
        span_sequences = tuple(pool.sequences[int(row)] for row in rows)
        label_start = min(sequence.start_frame for sequence in span_sequences) - padding
        label_end = max(sequence.end_frame for sequence in span_sequences) + padding + 1
        rallies = tuple(
            rally for rally in video.rallies
            if any(label_start <= frame < label_end for frame in rally.frames)
        )
        for row in rows:
            sequence = pool.sequences[int(row)]
            finished = finish_in_initial_context(
                video, position_by_span[span_id], sequence, stream_frames,
            )
            targets[row] = whole_rally_correctness(
                finished, rallies, video.fps, tolerance_at_30_fps=5,
            )
    return targets


def tightest_targets(pool: ExpandedPool, five: np.ndarray) -> np.ndarray:
    """Use five-frame labels per span when they retain a known positive option."""
    if pool.targets.shape != (len(pool.sequences),) or five.shape != pool.targets.shape:
        raise ValueError('five-frame targets must align with every expanded pool row')

    targets = pool.targets.copy()
    for rows in pool.span_rows.values():
        included = rows[pool.targets[rows] >= 0]
        if np.any(five[included] == 1):
            targets[included] = five[included]
    return targets


def _metadata_path(array_path: Path) -> Path:
    return array_path.parent / f'{array_path.name.removesuffix(".npy.xz")}.json.gz'


def _save_array_atomically(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix='.npy.xz')
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with lzma.open(temporary, 'wb', preset=9) as stream:
            np.save(stream, values, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _identity_fields(job: TargetJob) -> dict[str, Any]:
    return {
        'video': job.identity,
        'group': job.group,
        'split': job.split,
        'source_pool': str(job.pool_path.resolve()),
        'pools': str(job.pools_root.resolve()),
        'pools_manifest': job.pools_manifest,
    }


def _target_job(job: TargetJob) -> dict[str, Any] | None:
    artifact = job.output / 'videos' / f'{job.identity}.npy.xz'
    metadata_path = _metadata_path(artifact)
    identity_fields = _identity_fields(job)
    if artifact.is_file() and metadata_path.is_file():
        metadata = pilot.read_json(metadata_path)
        if any(metadata.get(key) != value for key, value in identity_fields.items()):
            raise ValueError(f'{job.identity}: cached targets name a different source pool or manifest')
        with lzma.open(artifact, 'rb') as stream:
            cached = np.load(stream, allow_pickle=False)
        if cached.shape != (metadata['rows'],) or not np.isin(cached, (-1, 0, 1)).all():
            raise ValueError(f'{job.identity}: cached five-frame targets are incomplete or invalid')
        return None

    with threadpool_limits(limits=1):
        started = time.perf_counter()
        pool: ExpandedPool = joblib.load(job.pool_path)
        if pool.original.identity != job.identity or pool.original.group != job.group:
            raise ValueError(f'{job.identity}: expanded pool identity or group differs from its record')
        targets = raw_five_targets(pool)
        if targets.shape != pool.targets.shape:
            raise ValueError(f'{job.identity}: five-frame target rows do not align to its expanded pool')
        positive_spans = sum(
            bool(np.any(targets[rows] == 1)) for rows in pool.span_rows.values()
        )
        summary = {
            **identity_fields,
            'rows': len(targets),
            'positive_rows': int(np.count_nonzero(targets == 1)),
            'positive_spans': int(positive_spans),
            'total_spans': len(pool.span_rows),
            'target_counts': {str(value): int(np.count_nonzero(targets == value)) for value in (-1, 0, 1)},
            'wall_s': time.perf_counter() - started,
        }
        _save_array_atomically(artifact, targets)
        pilot.write_json(metadata_path, summary)
        return summary


def prepare_targets(
    pools_root: Path, output: Path, *, workers: int = 8, only: str | None = None,
) -> list[dict[str, Any]]:
    """Prepare five-frame labels for existing pools and resume matching sidecars."""
    if not 1 <= workers <= 28:
        raise ValueError('workers must be between 1 and 28')
    pools_root = pools_root.resolve()
    pools_manifest = pilot.read_json(pools_root / 'pools.json.gz')
    records = {}
    for pool_path in sorted((pools_root / 'videos').glob('*.joblib')):
        identity = pool_path.stem
        record = pilot.read_json(pool_path.with_suffix('.json.gz'))
        if record.get('video') != identity:
            raise ValueError(f'{identity}: pool sidecar identity differs from its filename')
        records[identity] = (pool_path, record)
    if only is not None and only not in records:
        raise ValueError(f'--only does not name a prepared pool: {only}')
    if not records:
        raise ValueError('No expanded pools were found')

    manifest = {
        'schema': SCHEMA,
        'pools': str(pools_root),
        'pools_manifest': pools_manifest,
        'videos': {
            identity: {
                'pool': str(pool_path.resolve()),
                'group': record['group'],
                'split': record['split'],
            }
            for identity, (pool_path, record) in records.items()
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'targets.json.gz'
    if manifest_path.is_file() and pilot.read_json(manifest_path) != manifest:
        raise ValueError('Expanded pool manifest changed; use a fresh target output directory')
    if not manifest_path.is_file():
        pilot.write_json(manifest_path, manifest)

    selected = [only] if only is not None else sorted(records)
    jobs = [
        TargetJob(
            identity, records[identity][1]['group'], records[identity][1]['split'],
            records[identity][0], pools_root, pools_manifest, output,
        )
        for identity in selected
    ]
    summaries = []
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
    ) as executor:
        futures = [executor.submit(_target_job, job) for job in jobs]
        for future in as_completed(futures):
            summary = future.result()
            if summary is not None:
                summaries.append(summary)
                print(summary, flush=True)
    return summaries


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pools', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--only', help='Process one video as a smoke test')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = argument_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 28:
        parser.error('--workers must be between 1 and 28')
    prepare_targets(args.pools, args.output, workers=args.workers, only=args.only)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
