"""Fit contact inputs and summarise confidence scores for saved predictions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(REPOSITORY := Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(REPOSITORY / 'src'))

from experiments.annotator.heuristic_only_tune.run import (
    read_json,
    write_json,
)

# isort: split

import numpy as np
from threadpoolctl import threadpool_limits

from annotator.inference.learned import (
    LearnedModelPrediction,
)
from annotator.training.contact import ContactTrainingVideo, fit_contact_model
from annotator.training.sequences import (
    whole_rally_correctness,
)
from annotator.training.workflow import TrainingSettings
from experiments.annotator import development as experiment
from experiments.annotator.development import development_for
from experiments.annotator.heuristic_only_tune.data import (
    load_original_videos,
)
from experiments.annotator.tree_simplification.prepare import (
    preprocessing_from_record,
)
from experiments.annotator.tree_simplification.review_metrics import ranking_metrics
from experiments.annotator.tree_simplification.selector_review import (
    score_selector_review,
)


def _prepared_inputs(prepared_root: Path) -> dict[str, Any]:
    """Load prepared inputs and check their preprocessing settings."""
    experiment.require_raw_features(prepared_root)
    manifest = read_json(prepared_root / 'preparation.json.gz')
    inputs = manifest['inputs']
    preprocessing_from_record(inputs['preprocessing'])
    if len(inputs['groups']) != 40:
        raise ValueError('Prepared development input must name all 40 fixed videos')
    return inputs


def _training_records(
    prepared_root: Path, inputs: dict[str, Any], development: Any,
) -> dict[str, ContactTrainingVideo]:
    """Load contact rows from the fixed 32-video development training split."""
    training_ids = development.videos(experiment.TRAINING_GROUPS)
    training: dict[str, ContactTrainingVideo] = {}
    for identity in training_ids:
        path = prepared_root / 'videos' / f'{identity}.joblib'
        sidecar = read_json(path.with_suffix('.json.gz'))
        if sidecar.get('video') != identity or sidecar.get('preprocessing') != inputs['preprocessing']:
            raise ValueError(f'{identity}: prepared sidecar differs from the saved preprocessing')
        prepared = experiment.load_prepared(prepared_root, identity)
        expected_group = development.group_by_video[identity]
        if prepared.video.identity != identity or prepared.video.split != 'train' or prepared.group != expected_group:
            raise ValueError(f'{identity}: prepared identity or training group differs')
        frames = np.asarray(sorted(frame for rally in prepared.video.rallies for frame in rally.frames), dtype=np.int32)
        training[identity] = ContactTrainingVideo(
            identity, prepared.features.rows, frames, prepared.video.fps,
        )
    return training


def _pool_settings(settings: TrainingSettings, prepared_root: Path) -> dict[str, Any]:
    training_settings = asdict(settings)
    training_settings.pop('rally')
    training_settings = json.loads(json.dumps(training_settings, default=sorted))
    return {'settings': training_settings, 'prepared': str(prepared_root.resolve())}


def fit_contacts(
    prepared_root: Path,
    output: Path,
    *,
    seed_offset: int = 0,
    threads: int = 2,
    device: str = 'cuda',
) -> dict[str, Any]:
    """Fit held-A/B/C/D and v_32 contacts using the preparation's config."""
    if not 1 <= threads <= 28:
        raise ValueError('threads must be between 1 and 28')
    if device not in ('cpu', 'cuda'):
        raise ValueError('device must be cpu or cuda')
    prepared_root = prepared_root.resolve()
    inputs = _prepared_inputs(prepared_root)
    videos = load_original_videos(Path(inputs['paths']['shots_master']))
    development = development_for(videos)
    training_ids = development.videos(experiment.TRAINING_GROUPS)
    training = _training_records(prepared_root, inputs, development)
    if set(training) != set(training_ids):
        raise ValueError('Contact fit inputs do not match the fixed 32-video training split')

    settings = replace(
        experiment.settings_for(device, 5.0, seed_offset=seed_offset),
        preprocessing=preprocessing_from_record(inputs['preprocessing']),
    )
    pool_settings = _pool_settings(settings, prepared_root)
    output.mkdir(parents=True, exist_ok=True)
    settings_path = output / 'settings.json.gz'
    if settings_path.exists() and read_json(settings_path) != pool_settings:
        raise ValueError('Contact settings changed; use a fresh output directory')
    write_json(settings_path, pool_settings)

    fits = experiment.contact_fit_videos(development)
    expected_names = {'held_A', 'held_B', 'held_C', 'held_D', 'v_32'}
    if set(fits) != expected_names | {'final_40'}:
        raise ValueError('Contact fit definitions differ from the five production contact models')
    fit_identities = {
        name: {
            'fit': name,
            'sampling_order': list(sampling_order),
            'fit_video_order': list(development.id_order(sampling_order)),
        }
        for name, sampling_order in fits.items()
        if name != 'final_40'
    }
    stage = {
        'schema': 'tree-simplification-contacts/1',
        'seed_offset': seed_offset,
        'device': device,
        'threads': threads,
        'prepared': str(prepared_root),
        'settings': pool_settings,
        'fits': {name: fit_identities[name] for name in sorted(fit_identities)},
    }
    stage_path = output / 'contact-stage.json.gz'
    if stage_path.exists() and read_json(stage_path) != stage:
        raise ValueError('Contact stage identity changed; use a fresh output directory')
    (output / 'contacts').mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=threads):
        for name, sampling_order in fits.items():
            if name == 'final_40':
                continue
            fit_order = development.id_order(sampling_order)
            identity = {
                **fit_identities[name],
                'seed_offset': seed_offset,
                'device': device,
            }
            model_path = output / 'contacts' / f'{name}.joblib'
            record_path = model_path.with_suffix('.json.gz')
            if model_path.exists() != record_path.exists():
                raise ValueError(f'{name}: contact model and metadata must be present together')
            if model_path.exists():
                cached = read_json(record_path)
                if cached.get('identity') != identity:
                    raise ValueError(f'{name}: cached contact fit differs; choose a fresh output directory')
                continue

            started = time.perf_counter()
            experiment.progress(output, 'contact-fit-start', fit=name, videos=len(sampling_order))
            fitted = fit_contact_model(
                [training[identity] for identity in sampling_order], settings.contact,
                fit_video_order=fit_order,
            )
            experiment.dump_atomically(fitted.model, model_path)
            record = {
                'identity': identity,
                'rows': len(fitted.selection.labels),
                'positive_rows': int(fitted.selection.labels.sum()),
                'wall_s': time.perf_counter() - started,
            }
            write_json(record_path, record)
            experiment.progress(output, 'contact-fit-complete', fit=name, rows=record['rows'])

    write_json(stage_path, stage)
    return stage


def _review_rows(
    video: Any,
    identity: str,
    prediction: LearnedModelPrediction,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selector = score_selector_review(prediction.review.features)
    rankings: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    sequences = prediction.sequences
    span_ids = tuple(sequence.span_id for sequence in sequences)
    for tolerance in (5, 10):
        labels = np.asarray([
            whole_rally_correctness(sequence, video.rallies, video.fps, tolerance_at_30_fps=tolerance)
            for sequence in sequences
        ], dtype=np.int8)
        for name, confidence in (('selector', selector), ('fitted', prediction.review)):
            scores = np.asarray(prediction.scores_for_rallies(confidence))
            rankings[f'{name}-{tolerance}'] = ranking_metrics(labels, scores)
            for span, label, score in zip(span_ids, labels, scores, strict=True):
                rows.append({
                    'video': identity,
                    'span': span,
                    'tolerance': tolerance,
                    'ranking': name,
                    'correct': int(label),
                    'score': float(score),
                })
    return rankings, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)

    contacts = commands.add_parser('contacts', help='fit held-group and full-training contact trees')
    contacts.add_argument('--prepared', type=Path, required=True)
    contacts.add_argument('--output', type=Path, required=True)
    contacts.add_argument('--seed-offset', type=int, default=0)
    contacts.add_argument('--threads', type=int, default=2)
    contacts.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')

    args = parser.parse_args()
    fit_contacts(
        args.prepared, args.output, seed_offset=args.seed_offset,
        threads=args.threads, device=args.device,
    )


if __name__ == '__main__':
    main()
