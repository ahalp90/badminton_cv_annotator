"""Build eight-fold contact training pools while reusing final contact validation."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))

from experiments.annotator.tree_simplification import pilot

# isort: split

import numpy as np
from threadpoolctl import threadpool_limits

from annotator.training.contact import (
    ContactFitConfig,
    ContactTrainingVideo,
    fit_contact_model,
)
from experiments.annotator.heuristic_only_tune.data import load_original_videos
from experiments.annotator.tree_hp_tune.run import development_for
from experiments.annotator.tree_simplification.pools import prepare_pools

CONTACT_FOLDS = {
    'A1': ('sset_01', 'sset_11', 'sset_21', 'sset_32'),
    'A2': ('sset_05', 'sset_19', 'sset_29', 'sset_41'),
    'B1': ('sset_02', 'sset_13', 'sset_26', 'sset_34'),
    'B2': ('sset_06', 'sset_15', 'sset_33', 'sset_38'),
    'C1': ('sset_03', 'sset_16', 'sset_28', 'sset_35'),
    'C2': ('sset_07', 'sset_17', 'sset_36', 'sset_42'),
    'D1': ('sset_04', 'sset_14', 'sset_23', 'sset_43'),
    'D2': ('sset_08', 'sset_20', 'sset_37', 'sset_44'),
}
SCHEMA = 'tree-simplification-contact-folds/1'


def fold_assignments(development: pilot.retrain.Development) -> dict[str, str]:
    """Require the frozen 32/8 split and preserve the downstream A–D groups."""
    training = development.videos(pilot.retrain.TRAINING_GROUPS)
    validation = development.videos((pilot.retrain.VALIDATION_GROUP,))
    assignments = {}
    for fold, identities in CONTACT_FOLDS.items():
        if len(identities) != 4:
            raise ValueError('Each contact fold must contain four videos')
        for identity in identities:
            if identity in assignments or development.group_by_video.get(identity) != fold[0]:
                raise ValueError('Contact folds must be disjoint subdivisions of the original A–D groups')
            assignments[identity] = fold
    if len(training) != 32 or len(validation) != 8 or set(assignments) != set(training):
        raise ValueError('Contact folds require the original 32 training and 8 validation videos')
    return assignments


def prepare_contact_folds(
    prepared_root: Path,
    pilot_output: Path,
    original_pools: Path,
    output: Path,
    *,
    threads: int = 2,
    workers: int = 4,
    only: str | None = None,
) -> None:
    """Fit eight contact complements, then rebuild training pools in a fresh family.

    Fits and pool preparation run consecutively. Validation pool files are copied
    exactly; their final contact32 model remains the original pilot's model.
    """
    if not 1 <= threads <= 28 or not 1 <= workers <= 28:
        raise ValueError('threads and workers must each be between 1 and 28')
    prepared_root, pilot_output = prepared_root.resolve(), pilot_output.resolve()
    original_pools, output = original_pools.resolve(), output.resolve()
    pilot_settings = pilot.read_json(pilot_output / 'settings.json.gz')
    preparation = pilot.read_json(prepared_root / 'preparation.json.gz')['inputs']
    if pilot_settings['prepared'] != str(prepared_root):
        raise ValueError('Pilot settings name a different prepared directory')
    settings = pilot_settings['settings']
    pool_settings = pilot.read_json(original_pools / 'pools.json.gz')
    if (pool_settings['prepared'], pool_settings['pilot'], pool_settings['settings']) != (
        str(prepared_root), str(pilot_output), settings,
    ):
        raise ValueError('Original pools and pilot settings differ')
    videos = load_original_videos(Path(preparation['paths']['shots_master']))
    development = development_for(videos)
    assignments = fold_assignments(development)
    if preparation['groups'] != development.group_by_video:
        raise ValueError('Prepared videos differ from the fixed development groups')
    final_contact = pilot_output / 'contacts' / 'v_32.joblib'
    if not final_contact.is_file():
        raise FileNotFoundError(final_contact)
    manifest: dict[str, Any] = {
        'schema': SCHEMA, 'prepared': str(prepared_root), 'pilot': str(pilot_output),
        'settings': settings, 'original_pools': str(original_pools), 'pools_manifest': pool_settings,
        'folds': {fold: list(identities) for fold, identities in CONTACT_FOLDS.items()},
        'assignments': assignments, 'final_contact': str(final_contact),
    }
    manifest_path = output / 'contact-folds.json.gz'
    if manifest_path.exists() and pilot.read_json(manifest_path) != manifest:
        raise ValueError('Contact folds or settings changed; use a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        pilot.write_json(manifest_path, manifest)
    config = ContactFitConfig(**settings['contact'])
    training_ids = development.videos(pilot.retrain.TRAINING_GROUPS)
    training = {}
    for identity in training_ids:
        prepared = pilot.load_prepared(prepared_root, identity)
        if prepared.video.split != 'train' or prepared.group != development.group_by_video[identity]:
            raise ValueError(f'{identity}: training membership differs')
        frames = np.asarray(sorted(frame for rally in prepared.video.rallies for frame in rally.frames), dtype=np.int32)
        training[identity] = ContactTrainingVideo(identity, prepared.features.rows, frames, prepared.video.fps)
    del prepared
    with threadpool_limits(limits=threads):
        for fold, held in CONTACT_FOLDS.items():
            path = output / 'contacts' / f'held_{fold}.joblib'
            if path.is_file() and path.with_suffix('.json.gz').is_file():
                continue
            sampling_order = tuple(identity for identity in training_ids if identity not in held)
            fit_order = development.id_order(sampling_order)
            pilot.progress(output, 'contact-fit', fold=fold, training_videos=len(sampling_order))
            started = time.perf_counter()
            fitted = fit_contact_model(
                [training[identity] for identity in sampling_order], config, fit_video_order=fit_order,
            )
            pilot.retrain.dump_atomically(fitted.model, path)
            pilot.write_json(path.with_suffix('.json.gz'), {
                'fold': fold, 'held_videos': held, 'training_videos': sampling_order,
                'fit_video_order': fit_order, 'rows': len(fitted.selection.labels),
                'positive_rows': int(fitted.selection.labels.sum()), 'wall_s': time.perf_counter() - started,
            })
            del fitted
    del training
    paths = {identity: output / 'contacts' / f'held_{fold}.joblib' for identity, fold in assignments.items()}
    prepare_pools(
        prepared_root, pilot_output, output / 'pools', workers=workers, only=only,
        max_edits=pool_settings['max_edits'], deletion_candidates=pool_settings['deletion_candidates'],
        contact_paths=paths, contact_folds=CONTACT_FOLDS, validation_source=original_pools,
    )
    pilot.progress(output, 'complete', contact_fits=8, only=only)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--original-pools', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--only', help='Build one pool after fitting the eight contact folds')
    args = parser.parse_args(argv)
    prepare_contact_folds(
        args.prepared, args.pilot, args.original_pools, args.output,
        threads=args.threads, workers=args.workers, only=args.only,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
