"""Fit split correction from saved contact models and evaluate the complete bundle.

Keep the saved full contact model and first selector. Use the normal training
implementation to fit the follow-up selector and final confidence model. Run
one seed per process; the caller can bind each process to six CPU cores.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, replace
from pathlib import Path

import joblib
from threadpoolctl import threadpool_limits

from annotator.models import AnnotatorModels, fitted_trees, save_models
from annotator.training.followup import (
    PreparedTrainingVideo,
    ScoredTrainingVideo,
    fit_followup_models,
)
from annotator.training.rally import RallyTrainingPool
from annotator.training.trees import TreeFitConfig
from annotator.training.workflow import DEFAULT_TRAINING_SETTINGS, TrainingSettings
from dataset_builder.vision import save_json_gz
from experiments.annotator import development
from experiments.annotator.heuristic_only_tune.data import load_original_videos
from experiments.annotator.heuristic_only_tune.run import load_paths, write_json
from experiments.annotator.tree_simplification.court_acceptance import (
    CANDIDATE,
    cpu_pool,
    evaluate,
)
from experiments.annotator.tree_simplification.hp_search_workers import (
    contact_directory,
    pool_directory,
)


def import_training_inputs(
    paths_file: Path, prepared_root: Path, baseline: Path, output: Path, seed: int,
) -> tuple[list[PreparedTrainingVideo], dict[str, Path], dict[str, Path]]:
    """Reuse prepared videos, contact scores and labelled ordinary proposal arrays."""
    paths = load_paths(paths_file)
    originals = load_original_videos(paths['shots_master'])
    split = development.development_for(originals)
    by_identity = {video.identity: video for video in originals}
    contacts = contact_directory(baseline, CANDIDATE, seed)
    pools = pool_directory(baseline, CANDIDATE, seed, 'train')
    videos, ordinary_paths = [], {}
    directory = output / 'ordinary'
    directory.mkdir(parents=True, exist_ok=True)
    for identity in split.id_order(split.videos(development.TRAINING_GROUPS)):
        group = split.group_by_video[identity]
        video = PreparedTrainingVideo(
            identity, group, prepared_root / 'development' / 'videos' / f'{identity}.joblib',
            by_identity[identity].rallies,
        )
        videos.append(video)
        saved = joblib.load(pools / f'{identity}.joblib')
        # Ordinary pools have no piece bounds. Keep their existing rows and labels.
        context = replace(saved.video.context, piece_bounds={})
        options = replace(saved.runtime, context=context)
        pool = RallyTrainingPool(identity, group, options, saved.targets)
        scored = joblib.load(contacts / 'scores' / f'{identity}.joblib')
        destination = directory / f'{identity}.joblib'
        joblib.dump(ScoredTrainingVideo(scored, pool), destination, compress=3)
        ordinary_paths[identity] = destination
    held_contacts = {group: contacts / f'held_{group}.joblib' for group in development.TRAINING_GROUPS}
    return videos, ordinary_paths, held_contacts


def run(
    paths_file: Path, prepared_root: Path, baseline: Path, output_root: Path,
    seed: int, workers: int, gpu_fits: int, device: str,
) -> None:
    started = time.perf_counter()
    output = output_root / f'seed-{seed}'
    output.mkdir(parents=True, exist_ok=True)
    # This experiment starts from its saved three-model baseline; runtime
    # loading accepts only complete current-format bundles.
    original = joblib.load(baseline / 'models' / f'seed-{seed}' / 'bundle' / 'models.joblib')
    videos, ordinary, held_contacts = import_training_inputs(paths_file, prepared_root, baseline, output, seed)
    defaults = DEFAULT_TRAINING_SETTINGS
    rally_settings = replace(
        defaults.rally,
        tree=TreeFitConfig(max_iter=CANDIDATE.selector_rounds, max_leaf_nodes=CANDIDATE.selector_leaves,
                           random_seed=defaults.rally.tree.random_seed + seed, device=device),
        max_edits=original.rally_selector.max_edits,
        deletion_candidates=original.rally_selector.deletion_candidates,
    )
    confidence_settings = replace(defaults.confidence, random_seed=defaults.confidence.random_seed + seed, device=device)
    with threadpool_limits(limits=2):
        followup, review = fit_followup_models(
            videos, ordinary, held_contacts, original.contact_settings, rally_settings, confidence_settings,
            output / 'training', workers=workers, gpu_fits=gpu_fits,
        )
    models = AnnotatorModels(
        original.contact_model, original.rally_selector, followup, review,
        original.contact_settings, original.preprocessing,
    )
    for model in fitted_trees(models):
        model.set_params(device='cpu', n_jobs=1)
    bundle = output / 'bundle'
    save_models(models, bundle)
    settings = TrainingSettings(
        contact=replace(defaults.contact, max_leaf_nodes=CANDIDATE.contact_leaves,
                        max_iter=CANDIDATE.contact_rounds, random_seed=defaults.contact.random_seed + seed, device=device),
        contact_prediction=original.contact_settings, preprocessing=original.preprocessing,
        rally=rally_settings, confidence=confidence_settings,
    )
    save_json_gz(bundle / 'fit-settings.json.gz', asdict(settings))
    write_json(output / 'training-times.json.gz', {'training_wall': time.perf_counter() - started})
    with cpu_pool(workers) as cpu:
        evaluate(paths_file, prepared_root, bundle, output_root / 'evaluation', seed, cpu, device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True, help='Completed contact and first-selector training run')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--gpu-fits', type=int, default=3)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    if min(args.workers, args.gpu_fits) < 1:
        parser.error('--workers and --gpu-fits must be positive')
    run(args.paths, args.prepared, args.baseline, args.output, args.seed, args.workers, args.gpu_fits, args.device)


if __name__ == '__main__':
    main()
