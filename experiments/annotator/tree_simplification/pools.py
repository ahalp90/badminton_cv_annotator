"""Prepare labelled rally options for the current selector experiments."""

from __future__ import annotations

import argparse
import multiprocessing
import shutil
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))

from experiments.annotator import development as experiment

# isort: split

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from annotator.contacts.model import ContactModelConfig
from annotator.sequence.contacts import ContactSequence
from annotator.training import rally_targets
from annotator.training.sequences import (
    DEFAULT_RALLY_BOUNDARY_POLICY,
    RallyBoundaryPolicy,
    SequenceTrainingVideo,
)
from annotator.training.workflow import TrainingSettings
from experiments.annotator.tree_simplification.data import PreparedVideo
from experiments.annotator.tree_simplification.options import (
    LabelledOptions,
    expand_pool,
)

SCHEMA = 'tree-simplification-option-pools/1'
TARGET_VALUES = (-1, 0, 1)
EDIT_KINDS = ('keep', 'add', 'delete', 'mixed')


@dataclass(frozen=True)
class PoolJob:
    video: str
    group: str
    split: str
    prepared_path: Path
    contact_path: Path
    output: Path
    contact_prediction: dict[str, Any]
    max_edits: int
    deletion_candidates: int
    boundary_policy: RallyBoundaryPolicy
    reuse_options: Path | None = None


def _span_rows(sequences: Sequence[ContactSequence], span_ids: Sequence[int]) -> dict[int, np.ndarray]:
    rows: dict[int, list[int]] = {span_id: [] for span_id in span_ids}
    for row, sequence in enumerate(sequences):
        rows[sequence.span_id].append(row)
    return {span_id: np.asarray(indices, dtype=np.int64) for span_id, indices in rows.items()}


def original_rows(options: LabelledOptions) -> np.ndarray:
    """Locate the original options so features and alternate targets use identical rows."""
    return rally_targets.original_option_rows(options.runtime)


def original_options(options: LabelledOptions, *, selected: np.ndarray | None = None) -> LabelledOptions:
    """Select the initial and original options, keeping their feature and target rows."""
    if selected is None:
        selected = original_rows(options)
    sequences = tuple(options.sequences[row] for row in selected)
    return LabelledOptions(
        sequences,
        options.matrix[selected],
        options.targets[selected],
        _span_rows(sequences, tuple(options.span_rows)),
        options.video,
        options.boundary_policy,
    )


def edit_kind(sequence: ContactSequence, initial_by_span: dict[int, ContactSequence]) -> str:
    """Classify a proposal by whether it adds or removes detector contacts."""
    before = {event.frame for event in initial_by_span[sequence.span_id].events}
    after = {event.frame for event in sequence.events}
    added = bool(after - before)
    deleted = bool(before - after)
    if added and deleted:
        return 'mixed'
    if added:
        return 'add'
    if deleted:
        return 'delete'
    return 'keep'


def pool_summary(options: LabelledOptions, split: str) -> dict[str, object]:
    """Summarise target balance and positive-target span coverage."""
    initial_by_span = {sequence.span_id: sequence for sequence in options.video.initial}
    target_counts = {kind: Counter() for kind in EDIT_KINDS}
    for sequence, target in zip(options.sequences, options.targets, strict=True):
        target_counts[edit_kind(sequence, initial_by_span)][str(int(target))] += 1

    original = original_options(options)

    def positive_spans(pool: LabelledOptions) -> int:
        return sum(bool(np.any(pool.targets[rows] == 1)) for rows in pool.span_rows.values())

    return {
        'video': options.video.identity,
        'group': options.video.group,
        'split': split,
        'boundary_policy': asdict(options.boundary_policy),
        'options': len(options.sequences),
        'feature_count': int(options.matrix.shape[1]),
        'target_counts_by_edit_kind': {
            kind: {str(target): target_counts[kind][str(target)] for target in TARGET_VALUES}
            for kind in EDIT_KINDS
        },
        'positive_target_span_coverage': {
            'original_options': positive_spans(original),
            'all_options': positive_spans(options),
            'total_spans': len(options.span_rows),
        },
    }


def _prepare_pool(job: PoolJob) -> dict[str, object]:
    """Build and atomically cache one video's labelled option pool."""
    with threadpool_limits(limits=1):
        if job.reuse_options is not None:
            saved: LabelledOptions = joblib.load(job.reuse_options)
            if saved.video.identity != job.video or saved.video.group != job.group:
                raise ValueError(f'{job.video}: reused proposal identity or group differs')
            targets = rally_targets.option_targets(saved.video, saved.runtime, boundary_policy=job.boundary_policy)
            options = replace(saved, targets=targets, boundary_policy=job.boundary_policy)
        else:
            prepared: PreparedVideo = joblib.load(job.prepared_path)
            if prepared.video.identity != job.video or prepared.group != job.group:
                raise ValueError(f'{job.video}: prepared identity or group differs from its record')
            if prepared.video.split != job.split:
                raise ValueError(f'{job.video}: prepared split differs from its record')
            contact = joblib.load(job.contact_path)
            prediction = ContactModelConfig(**job.contact_prediction)
            settings = TrainingSettings(contact_prediction=prediction)
            video: SequenceTrainingVideo = experiment.training_pool(prepared, contact, settings)
            options = expand_pool(video, job.max_edits, job.deletion_candidates, boundary_policy=job.boundary_policy)
        if np.isinf(options.matrix).any():
            raise ValueError(f'{job.video}: option feature matrix contains infinity')

        summary = pool_summary(options, job.split)
        artifact = job.output / 'videos' / f'{job.video}.joblib'
        experiment.dump_atomically(options, artifact)
        experiment.write_json(artifact.with_suffix('.json.gz'), summary)
        return summary


def _reuse_validation_pool(source: Path, artifact: Path, video: str, group: str) -> None:
    metadata = experiment.read_json(source.with_suffix('.json.gz'))
    if (metadata['video'], metadata['group'], metadata['split']) != (video, group, 'validation'):
        raise ValueError(f'{video}: reused validation metadata differs')
    artifact.parent.mkdir(parents=True, exist_ok=True)
    pairs = ((source, artifact), (source.with_suffix('.json.gz'), artifact.with_suffix('.json.gz')))
    for source_path, destination in pairs:
        temporary = destination.with_name(destination.name + '.tmp')
        shutil.copyfile(source_path, temporary)
        temporary.replace(destination)


def prepare_pools(
    prepared_root: Path,
    contact_run_output: Path,
    output: Path,
    *,
    workers: int = 8,
    only: str | None = None,
    max_edits: int = 3,
    deletion_candidates: int = 6,
    contact_paths: Mapping[str, Path] | None = None,
    contact_folds: Mapping[str, Sequence[str]] | None = None,
    validation_source: Path | None = None,
    boundary_policy: RallyBoundaryPolicy = DEFAULT_RALLY_BOUNDARY_POLICY,
    reuse_options: Path | None = None,
) -> list[dict[str, object]]:
    """Build selected training and validation pools from a completed contact run."""
    if not 1 <= workers <= 28:
        raise ValueError('workers must be between 1 and 28')
    if max_edits < 0 or deletion_candidates < 0:
        raise ValueError('edit and deletion budgets must be non-negative')

    prepared_root = prepared_root.resolve()
    contact_run_output = contact_run_output.resolve()
    contact_run_config = experiment.read_json(contact_run_output / 'settings.json.gz')
    if contact_run_config['prepared'] != str(prepared_root):
        raise ValueError('Contact-run settings name a different prepared directory')
    settings = contact_run_config['settings']
    preparation = experiment.read_json(prepared_root / 'preparation.json.gz')['inputs']
    if settings['preprocessing'] != preparation['preprocessing']:
        raise ValueError('Contact-run settings and prepared evidence use different preprocessing')

    contacts = contact_run_output / 'contacts'
    if contact_paths is not None:
        training_ids = {identity for identity, group in preparation['groups'].items() if group != 'V'}
        if set(contact_paths) != training_ids:
            raise ValueError('Explicit contact paths must name every training video exactly once')
    if validation_source is not None:
        validation_source = validation_source.resolve()
        original = experiment.read_json(validation_source / 'pools.json.gz')
        expected = {
            'schema': SCHEMA, 'prepared': str(prepared_root), 'contact_run': str(contact_run_output),
            'settings': settings, 'max_edits': max_edits, 'deletion_candidates': deletion_candidates,
            'boundary_policy': asdict(boundary_policy),
        }
        if original != expected:
            raise ValueError('Reused validation pools use different preparation or proposal settings')
    records = []
    for prepared_path in sorted((prepared_root / 'videos').glob('*.joblib')):
        record = experiment.read_json(prepared_path.with_suffix('.json.gz'))
        video = prepared_path.stem
        if record['video'] != video or preparation['groups'][video] != record['group']:
            raise ValueError(f'{video}: prepared record differs from the preparation manifest')
        if only is not None and video != only:
            continue
        if record['split'] not in ('train', 'validation'):
            raise ValueError(f'{video}: unsupported prepared split {record["split"]!r}')
        contact_name = f'held_{record["group"]}' if record['split'] == 'train' else 'v_32'
        contact_path = contacts / f'{contact_name}.joblib'
        if contact_paths is not None and record['split'] == 'train':
            contact_path = contact_paths[video].resolve()
        if not contact_path.is_file():
            raise FileNotFoundError(f'Missing contact model in the contact run: {contact_path}')
        records.append((video, record, prepared_path, contact_path))
    if only is not None and not records:
        raise ValueError(f'--only does not name a prepared training or validation video: {only}')
    if not records:
        raise ValueError('No prepared training or validation videos were found')

    manifest = {
        'schema': SCHEMA,
        'prepared': str(prepared_root),
        'contact_run': str(contact_run_output),
        'settings': settings,
        'max_edits': max_edits,
        'deletion_candidates': deletion_candidates,
        'boundary_policy': asdict(boundary_policy),
    }
    if contact_paths is not None:
        manifest['contact_paths'] = {video: str(path.resolve()) for video, path in contact_paths.items()}
    if contact_folds is not None:
        manifest['contact_folds'] = {fold: list(videos) for fold, videos in contact_folds.items()}
    if validation_source is not None:
        manifest['reused_validation'] = {'source': str(validation_source), 'pools_manifest': original}
    if reuse_options is not None:
        reuse_options = reuse_options.resolve()
        reused_manifest = experiment.read_json(reuse_options / 'pools.json.gz')
        for name in ('schema', 'prepared', 'contact_run', 'settings', 'max_edits', 'deletion_candidates'):
            if reused_manifest[name] != manifest[name]:
                raise ValueError(f'Reused proposal {name} differs from this run')
        manifest['reused_options'] = {'source': str(reuse_options), 'pools_manifest': reused_manifest}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'pools.json.gz'
    if manifest_path.exists() and experiment.read_json(manifest_path) != manifest:
        raise ValueError('Pool settings changed; use a fresh output directory')
    if not manifest_path.exists():
        experiment.write_json(manifest_path, manifest)

    jobs = []
    for video, record, prepared_path, contact_path in records:
        artifact = output / 'videos' / f'{video}.joblib'
        if artifact.is_file() and artifact.with_suffix('.json.gz').is_file():
            continue
        if validation_source is not None and record['split'] == 'validation':
            source = validation_source / 'videos' / artifact.name
            _reuse_validation_pool(source, artifact, video, record['group'])
            continue
        jobs.append(PoolJob(
            video, record['group'], record['split'], prepared_path, contact_path, output,
            settings['contact_prediction'], max_edits, deletion_candidates, boundary_policy,
            None if reuse_options is None else reuse_options / 'videos' / f'{video}.joblib',
        ))

    summaries = []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context('spawn'),
    ) as executor:
        futures = [executor.submit(_prepare_pool, job) for job in jobs]
        for future in as_completed(futures):
            summary = future.result()
            summaries.append(summary)
            print(summary, flush=True)
    return summaries


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument(
        '--contact-run', type=Path, required=True,
        help='Run folder containing the settings and contact models',
    )
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--only', help='Prepare one video as a smoke test')
    parser.add_argument('--max-edits', type=int, default=3)
    parser.add_argument('--deletion-candidates', type=int, default=6)
    parser.add_argument('--boundary-before-seconds', type=float, default=DEFAULT_RALLY_BOUNDARY_POLICY.before_seconds)
    parser.add_argument('--boundary-after-seconds', type=float, default=DEFAULT_RALLY_BOUNDARY_POLICY.after_seconds)
    parser.add_argument('--no-max-buffer', action='store_true', help='Train rally completeness without context limits')
    parser.add_argument('--reuse-options', type=Path, help='Relabel saved proposals without recomputing their features')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = argument_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 28:
        parser.error('--workers must be between 1 and 28')
    if args.max_edits < 0 or args.deletion_candidates < 0:
        parser.error('edit and deletion budgets must be non-negative')
    prepare_pools(
        args.prepared, args.contact_run, args.output, workers=args.workers, only=args.only,
        max_edits=args.max_edits, deletion_candidates=args.deletion_candidates,
        boundary_policy=(RallyBoundaryPolicy(None, None) if args.no_max_buffer
                         else RallyBoundaryPolicy(args.boundary_before_seconds, args.boundary_after_seconds)),
        reuse_options=args.reuse_options,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
