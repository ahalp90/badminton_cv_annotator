"""Evaluate base-30 contact motion units without changing production feature semantics.

Prepared rows retain their original regions and frame windows. Experimental models
must consume these transformed rows; they are never exported as production bundles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from annotator.config import SHIPPED_THRESHOLDS
from annotator.contacts.features import (
    WINDOW_OFFSETS_BASE30,
    ContactFeatures,
    shift_within_interval,
)
from annotator.fps_constants import BASE_FPS, ScalingKind, scale_for_fps
from annotator.rally.contacts import FLOOR_EPS, rolling_floor, span_impulses
from annotator.training.contact import ContactTrainingVideo, fit_contact_model
from annotator.training.trees import TreeBackend
from annotator.types import SmoothingMode
from experiments.annotator.heuristic_only_tune.data import load_original_videos
from experiments.annotator.tree_hp_tune.run import development_for
from experiments.annotator.tree_simplification import pilot
from experiments.annotator.tree_simplification.data import PreparedVideo

SCHEMA = 'contact-motion-base30/1'
VELOCITY_SCHEMA = 'contact-velocity-base30/1'
VELOCITY_SIGNALS = ('shuttle_vx', 'shuttle_vy', 'shuttle_speed', 'ankle_speed_top', 'ankle_speed_bot')


def file_md5(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'md5').hexdigest()


def preparation_receipt(prepared_root: Path) -> dict[str, Any]:
    """Reject partial or modified prepared families before fitting or replay."""
    preparation = pilot.read_json(prepared_root / 'preparation.json.gz')
    schema = preparation.get('motion_units')
    if schema not in (SCHEMA, VELOCITY_SCHEMA):
        raise ValueError('Expected the normalised prepared family')
    expected = set(preparation['inputs']['groups'])
    paths = sorted((prepared_root / 'videos').glob('*.joblib'))
    if {path.stem for path in paths} != expected:
        raise ValueError('Normalised preparation is incomplete')
    artifacts = {}
    for path in paths:
        record = pilot.read_json(path.with_suffix('.json.gz'))
        actual = file_md5(path)
        if record.get('motion_units') != schema or record.get('prepared_md5') != actual:
            raise ValueError(f'{path.stem}: prepared artifact differs from its receipt')
        artifacts[path.stem] = actual
    return {'schema': schema, 'preparation': preparation, 'artifacts': artifacts}


def normalise_contact_features(
    features: ContactFeatures, track: np.ndarray, fps: float, *, velocity_only: bool = False,
) -> ContactFeatures:
    """Convert learned motion inputs to base-30 units, preserving candidate regions.

    First differences scale by fps/30; second differences scale by its square.
    The impulse ratio uses a floor in the same base-30 units. Its smoothing and
    missing-data policy remain those of the original contact feature builder.
    """
    constants = scale_for_fps(fps)
    if fps == BASE_FPS:
        return features
    factor = fps / BASE_FPS
    rows = features.rows.copy()
    for signal in VELOCITY_SIGNALS:
        for offset in WINDOW_OFFSETS_BASE30:
            rows[f'{signal}_t{offset:+d}'] *= factor
    if velocity_only:
        return features._replace(rows=rows)
    for offset in WINDOW_OFFSETS_BASE30:
        rows[f'shuttle_impulse_t{offset:+d}'] *= factor ** 2

    ratio = np.full(len(track), np.nan, dtype=np.float64)
    thresholds = SHIPPED_THRESHOLDS._replace(smooth_window=constants.smooth_window)
    for start, end in features.search_intervals:
        impulse = span_impulses(track, start, end, thresholds, smoothing_mode=SmoothingMode.ZERO_FILL)
        if impulse is None:
            continue
        impulse = impulse * factor ** 2
        visible = track[start:end, 2] == 1
        around_visible = visible[:-2] & visible[1:-1] & visible[2:]
        floor = rolling_floor(impulse, around_visible, constants.impulse_floor_half_window_frames)
        ratio[start + 1:start + 1 + len(impulse)] = impulse / np.maximum(floor, FLOOR_EPS)
    for interval_id, (start, end) in enumerate(features.search_intervals):
        selected = rows['interval_id'] == interval_id
        frames = rows['frame'][selected]
        for base_offset in WINDOW_OFFSETS_BASE30:
            offset = 0
            if base_offset:
                count = ScalingKind.FRAME_COUNT.scale(abs(base_offset), fps)
                offset = int(math.copysign(count, base_offset))
            rows[f'shuttle_impulse_ratio_t{base_offset:+d}'][selected] = shift_within_interval(
                ratio, frames, offset, start, end,
            )
    return features._replace(rows=rows)


def _prepare_one(arguments: tuple[Path, Path, bool]) -> dict[str, Any]:
    source, output, velocity_only = arguments
    record = pilot.read_json(source.with_suffix('.json.gz'))
    source_md5 = file_md5(source)
    destination = output / 'videos' / source.name
    record_path = destination.with_suffix('.json.gz')
    if destination.exists() and record_path.exists():
        cached = pilot.read_json(record_path)
        if cached.get('source_md5') != source_md5 or cached.get('prepared_md5') != file_md5(destination):
            raise ValueError('Prepared source or cached artifact changed; use a fresh directory')
        return cached
    with threadpool_limits(limits=1):
        started = time.perf_counter()
        prepared: PreparedVideo = joblib.load(source)
        if prepared.video.identity != record['video'] or prepared.group != record['group']:
            raise ValueError('Prepared identity differs from its sidecar')
        if prepared.video.fps != prepared.evidence.fps:
            raise ValueError('Video and evidence FPS differ')
        features = normalise_contact_features(
            prepared.features, prepared.evidence.track, prepared.video.fps, velocity_only=velocity_only,
        )
        pilot.retrain.dump_atomically(replace(prepared, features=features), destination)
        schema = VELOCITY_SCHEMA if velocity_only else SCHEMA
        record.update(fps=prepared.video.fps, motion_units=schema, source_md5=source_md5,
                      prepared_md5=file_md5(destination), wall_s=time.perf_counter() - started)
        pilot.write_json(record_path, record)
        return record


def prepare(source: Path, output: Path, workers: int, only: str | None = None, *, velocity_only: bool = False) -> None:
    """Write a separate prepared family; retain the fixed development membership."""
    source, output = source.resolve(), output.resolve()
    if source == output:
        raise ValueError('Normalised preparation requires a separate output directory')
    original = pilot.read_json(source / 'preparation.json.gz')
    if 'motion_units' in original:
        raise ValueError('Source is already transformed')
    manifest = {
        **original, 'motion_units': VELOCITY_SCHEMA if velocity_only else SCHEMA,
        'raw_prepared': str(source), 'transform_source': pilot.source_record(),
        'transform_md5': file_md5(Path(__file__)),
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'preparation.json.gz'
    if manifest_path.exists() and pilot.read_json(manifest_path) != manifest:
        raise ValueError('Preparation inputs or transformation changed; use a fresh directory')
    pilot.write_json(manifest_path, manifest)
    paths = sorted((source / 'videos').glob('*.joblib'))
    if {path.stem for path in paths} != set(original['inputs']['groups']):
        raise ValueError('Source preparation is incomplete')
    if only is not None:
        paths = [path for path in paths if path.stem == only]
        if not paths:
            raise ValueError('Requested smoke video is absent')
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as executor:
        for record in executor.map(_prepare_one, [(path, output, velocity_only) for path in paths]):
            print(record, flush=True)
    pilot.progress(output, 'prepared', videos=len(paths), complete=only is None)


def fit_contacts(prepared_root: Path, output: Path, seed_offset: int, threads: int) -> None:
    """Fit the same four held-group models and final 32-video contact model as the raw pilot."""
    prepared_root = prepared_root.resolve()
    units = preparation_receipt(prepared_root)
    preparation = units['preparation']
    settings = pilot.settings_for(TreeBackend.XGBOOST, 'cuda', 5.0, seed_offset=seed_offset)
    legacy_settings = asdict(settings)
    legacy_settings.pop('rally_layers')
    config = json.loads(json.dumps({'settings': legacy_settings, 'prepared': str(prepared_root)}, default=sorted))
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / 'settings.json.gz'
    if receipt.exists() and pilot.read_json(receipt) != config:
        raise ValueError('Contact settings changed; use a fresh output directory')
    units_path = output / 'motion-units.json.gz'
    if units_path.exists() and pilot.read_json(units_path) != units:
        raise ValueError('Contact preparation changed; use a fresh output directory')
    pilot.write_json(receipt, config)
    pilot.write_json(units_path, units)
    videos = load_original_videos(Path(preparation['inputs']['paths']['shots_master']))
    development = development_for(videos)
    if preparation['inputs']['groups'] != development.group_by_video:
        raise ValueError('Prepared membership differs from the fixed development split')
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
        for name, sampling_order in pilot.retrain.contact_fit_videos(development).items():
            if name == 'final_40':
                continue
            path = output / 'contacts' / f'{name}.joblib'
            if path.exists() and path.with_suffix('.json.gz').exists():
                continue
            started = time.perf_counter()
            pilot.progress(output, 'contact-fit', fit=name)
            fitted = fit_contact_model(
                [training[identity] for identity in sampling_order], settings.contact,
                fit_video_order=development.id_order(sampling_order),
            )
            pilot.retrain.dump_atomically(fitted.model, path)
            pilot.write_json(path.with_suffix('.json.gz'), {
                'wall_s': time.perf_counter() - started, 'rows': len(fitted.selection.labels),
                'positive_rows': int(fitted.selection.labels.sum()), 'training_videos': sampling_order,
                'motion_units': units['schema'],
            })
            del fitted
    pilot.progress(output, 'complete', fits=5, motion_units=units['schema'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='stage', required=True)
    preparation = subparsers.add_parser('prepare')
    preparation.add_argument('--source', type=Path, required=True)
    preparation.add_argument('--output', type=Path, required=True)
    preparation.add_argument('--workers', type=int, default=4)
    preparation.add_argument('--only')
    preparation.add_argument('--velocity-only', action='store_true', help='Leave impulse and its ratio in raw units')
    contact = subparsers.add_parser('contacts')
    contact.add_argument('--prepared', type=Path, required=True)
    contact.add_argument('--output', type=Path, required=True)
    contact.add_argument('--seed-offset', type=int, default=0)
    contact.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.stage == 'prepare':
        if not 1 <= args.workers <= 28:
            parser.error('--workers must be between 1 and 28')
        prepare(args.source, args.output, args.workers, args.only, velocity_only=args.velocity_only)
    else:
        if not 1 <= args.threads <= 28:
            parser.error('--threads must be between 1 and 28')
        fit_contacts(args.prepared, args.output, args.seed_offset, args.threads)


if __name__ == '__main__':
    main()
