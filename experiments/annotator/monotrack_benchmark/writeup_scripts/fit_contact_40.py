"""Refit the annotator's contact model on all 40 original ShuttleSet videos.

This is the training-set-size check in the MonoTrack write-up. For each of the five test
bundles, the contact model is refitted on the 32 training and 8 validation videos with
the same settings and seed as its 32-video fit. The rally selector and review ranker keep
their 32-video fits.

Before the 40-video fit, the script refits the 32-video contact model from the same
prepared inputs and requires it to predict exactly what the bundle's contact model
predicts on the validation videos. A 40-video model therefore differs from its bundle's
contact model only in its training videos.

Usage: python fit_contact_40.py --prepared DIR --out DIR
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

# Bound inline: ruff's E402 allows sys.path calls above imports, but not assignments.
sys.path.insert(0, str(REPO := Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(REPO / 'src'))

import numpy as np
from threadpoolctl import threadpool_limits

from annotator.contacts.model import contact_feature_matrix
from annotator.models import AnnotatorModels, fitted_trees, load_models, save_models
from annotator.training.contact import ContactTrainingVideo, fit_contact_model
from experiments.annotator import development as experiment
from experiments.annotator.development import development_for
from experiments.annotator.heuristic_only_tune.data import load_original_videos
from experiments.annotator.heuristic_only_tune.run import read_json, write_json
from experiments.annotator.tree_simplification.data import PreparedVideo
from experiments.annotator.tree_simplification.prepare import preprocessing_record

SEEDS = (0, 100, 200, 300, 400)
TREE_RESULTS = REPO / 'experiments/annotator/tree_simplification/results'
SHOTS_MASTER = REPO / 'training/data/shuttleset/annotations/shots_master.csv'
# The contact fits behind the test bundles ran on the GPU with four threads.
DEVICE, THREADS = 'cuda', 4


def bundle_path(seed: int) -> Path:
    return TREE_RESULTS / f'models/raw_baseline-seed-{seed}'


def contact_stage(seed: int) -> dict:
    return read_json(TREE_RESULTS / f'test-run/candidates/raw_baseline/seed-{seed}/contacts-fit/contact-stage.json.gz')


def training_settings_record(seed: int, models: AnnotatorModels) -> dict:
    """The fit settings in the form the contact-stage record saves them."""
    settings = replace(experiment.settings_for(DEVICE, 5.0, seed_offset=seed),
                       preprocessing=models.preprocessing)
    record = asdict(settings)
    record = experiment.contact_stage_settings(record)
    return json.loads(json.dumps(record, default=sorted))


def load_training_videos(prepared: Path, preprocessing: dict) -> dict[str, tuple[PreparedVideo, ContactTrainingVideo]]:
    """All 40 prepared development videos, checked against the bundle's preprocessing."""
    manifest = read_json(prepared / 'preparation.json.gz')
    if manifest.get('motion_units') is not None or manifest['inputs']['preprocessing'] != preprocessing:
        raise ValueError('prepared inputs differ from the bundle preprocessing or use scaled motion units')
    videos = {}
    for identity in manifest['inputs']['groups']:
        sidecar = read_json(prepared / 'videos' / f'{identity}.json.gz')
        if sidecar.get('video') != identity or sidecar.get('preprocessing') != preprocessing:
            raise ValueError(f'{identity}: prepared sidecar differs from the bundle preprocessing')
        video = experiment.load_prepared(prepared, identity)
        frames = np.asarray(sorted(frame for rally in video.video.rallies for frame in rally.frames), dtype=np.int32)
        videos[identity] = (video, ContactTrainingVideo(identity, video.features.rows, frames, video.video.fps))
    if len(videos) != 40:
        raise ValueError('expected all 40 original development videos')
    return videos


def predictions(model: object, videos: list[PreparedVideo]) -> np.ndarray:
    matrix = np.concatenate([contact_feature_matrix(video.features.rows) for video in videos])
    return model.predict_proba(matrix)[:, 1]  # type: ignore[attr-defined]


def refit(seed: int, prepared: Path, output: Path) -> dict:
    models = load_models(bundle_path(seed))
    preprocessing = preprocessing_record(models.preprocessing)
    stage = contact_stage(seed)
    settings = training_settings_record(seed, models)
    if stage['seed_offset'] != seed or experiment.contact_stage_settings(stage['settings']['settings']) != settings:
        raise ValueError(f'seed {seed}: contact settings differ from the test bundle contact stage')
    contact_settings = experiment.settings_for(DEVICE, 5.0, seed_offset=seed).contact
    videos = load_training_videos(prepared, preprocessing)
    development = development_for(load_original_videos(SHOTS_MASTER))
    fits = experiment.contact_fit_videos(development)
    if list(fits['v_32']) != stage['fits']['v_32']['sampling_order']:
        raise ValueError(f'seed {seed}: 32-video sampling order differs from the contact stage')
    validation = [videos[identity][0] for identity in development.videos((experiment.VALIDATION_GROUP,))]

    def fit(name: str) -> tuple[object, int, int, float]:
        order = fits[name]
        started = time.perf_counter()
        with threadpool_limits(limits=THREADS):
            result = fit_contact_model([videos[identity][1] for identity in order], contact_settings,
                                       fit_video_order=development.id_order(order))
        result.model.set_params(device='cpu')
        return (result.model, len(result.selection.labels), int(result.selection.labels.sum()),
                time.perf_counter() - started)

    model_32, rows_32, positives_32, seconds_32 = fit('v_32')
    expected, actual = predictions(models.contact, validation), predictions(model_32, validation)
    if not np.array_equal(expected, actual):
        raise ValueError(f'seed {seed}: refitted 32-video contact model differs from the bundle '
                         f'(max difference {np.abs(expected - actual).max():.3g})')
    model_40, rows_40, positives_40, seconds_40 = fit('final_40')
    combined = replace(models, contact=model_40)
    for tree in fitted_trees(combined):
        if tree.get_params().get('device') != 'cpu':
            raise ValueError('bundle trees must predict on the CPU')
    save_models(combined, output / f'seed-{seed}')
    reloaded = load_models(output / f'seed-{seed}')
    if not np.array_equal(predictions(reloaded.contact, validation), predictions(model_40, validation)):
        raise ValueError(f'seed {seed}: saved 40-video bundle does not reload to the fitted model')
    record = {
        'seed_offset': seed, 'source_bundle': f'raw_baseline-seed-{seed}', 'device': DEVICE, 'threads': THREADS,
        'contact_settings': settings['contact'],
        'check_32_videos': {'validation_rows': len(expected), 'identical_predictions': True,
                            'selected_rows': rows_32, 'positive_rows': positives_32, 'seconds': seconds_32},
        'fit_40_videos': {'sampling_order': list(fits['final_40']),
                          'fit_order': list(development.id_order(fits['final_40'])),
                          'selected_rows': rows_40, 'positive_rows': positives_40, 'seconds': seconds_40},
    }
    write_json(output / f'seed-{seed}' / 'contact-fit.json.gz', record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True,
                        help='prepared baseline inputs for the 40 original videos (tree_simplification prepare)')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='*', default=list(SEEDS))
    args = parser.parse_args()
    for seed in args.seeds:
        record = refit(seed, args.prepared.resolve(), args.out)
        print(json.dumps({key: record[key] for key in ('seed_offset', 'check_32_videos')}), flush=True)


if __name__ == '__main__':
    main()
