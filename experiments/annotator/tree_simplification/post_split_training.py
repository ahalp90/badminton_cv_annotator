"""Compare two ways to train the follow-up selector and final confidence model.

Both recipes keep the deployed contact model and first selector fixed. With
``reuse-contact``, each outer fold uses one saved contact model throughout its
training. With ``nested-contact``, auxiliary contact fits also exclude the video
group whose intermediate predictions become training examples.

Run one recipe and seed per process. Use the existing six-core affinity and
three concurrent GPU fits; the caller runs the five seeds in parallel.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from annotator.models import AnnotatorModels, fitted_trees, save_models
from annotator.rallies.refine_rallies import RallySelector, predict_rally_options
from annotator.rallies.split_and_correct_rallies import (
    build_split_context,
    split_and_correct_rallies,
)
from annotator.training.contact import ContactFitConfig
from annotator.training.rally import RallyFitConfig, fit_review_model, fit_selector
from annotator.training.sequences import whole_rally_correctness
from annotator.training.trees import TreeFitConfig
from annotator.training.workflow import DEFAULT_TRAINING_SETTINGS
from experiments.annotator import development
from experiments.annotator.heuristic_only_tune.data import load_original_videos
from experiments.annotator.heuristic_only_tune.run import (
    load_paths,
    read_json,
    write_json,
)
from experiments.annotator.tree_simplification.court_acceptance import (
    CANDIDATE,
    cpu_pool,
    evaluate,
    finish_jobs,
)
from experiments.annotator.tree_simplification.hp_search_workers import (
    cache_contact_scores,
    contact_directory,
    contacts_at_cutoff,
    fit_contact,
    pool_directory,
    prepare_pool,
    prepared_video,
)
from experiments.annotator.tree_simplification.input_stages import _training_records
from experiments.annotator.tree_simplification.options import (
    LabelledOptions,
    expand_pool,
)
from experiments.annotator.tree_simplification.targets import (
    preferred_targets,
    raw_five_targets,
)

Recipe = Literal['reuse-contact', 'nested-contact']
GROUPS = development.TRAINING_GROUPS


@dataclass(frozen=True)
class OrdinaryPoolJob:
    """Score one video with the named contact-training groups, then build options."""

    video_id: str
    video_group: str
    contact_groups: tuple[str, ...]
    prepared: Path
    contact_model: Path
    scores: Path
    pool: Path
    reuse_saved: bool


@dataclass(frozen=True)
class PiecePoolJob:
    """Split a first selector's finished prediction into follow-up examples."""

    ordinary: OrdinaryPoolJob
    first_selector: Path
    pool: Path


@dataclass(frozen=True)
class SelectorFitJob:
    """Ordinary examples and optional split examples for one selector fit."""

    groups: tuple[str, ...]
    ordinary: tuple[OrdinaryPoolJob, ...]
    pieces: tuple[PiecePoolJob, ...]
    destination: Path


def ordinary_pool_job(
    baseline: Path, output: Path, prepared: Path, split: development.Development,
    seed: int, identity: str, contact_groups: tuple[str, ...],
) -> OrdinaryPoolJob:
    """Reuse original out-of-group pools; keep every new pool in this run."""
    owner = split.group_by_video[identity]
    contact_root = contact_directory(baseline, CANDIDATE, seed)
    if len(contact_groups) == 3:
        held = next(group for group in GROUPS if group not in contact_groups)
        contact_model = contact_root / f'held_{held}.joblib'
    else:
        contact_model = output / 'contacts' / f'{"".join(contact_groups)}.joblib'
    reuse_saved = len(contact_groups) == 3 and owner not in contact_groups
    if reuse_saved:
        scores = contact_root / 'scores' / f'{identity}.joblib'
        pool = pool_directory(baseline, CANDIDATE, seed, 'train') / f'{identity}.joblib'
    else:
        scores = output / 'scores' / ''.join(contact_groups) / f'{identity}.joblib'
        pool = output / 'ordinary' / ''.join(contact_groups) / f'{identity}.joblib'
    return OrdinaryPoolJob(
        identity, owner, contact_groups, prepared / 'videos' / f'{identity}.joblib',
        contact_model, scores, pool, reuse_saved,
    )


def first_selector_path(output: Path, recipe: Recipe, outer_group: str, inner_group: str | None = None) -> Path:
    groups = tuple(group for group in GROUPS if group not in (outer_group, inner_group))
    directory = output / 'first_selectors'
    if recipe == 'reuse-contact':
        directory /= f'held-{outer_group}'
    return directory / ''.join(groups) / 'selector.joblib'


def first_selector_jobs(
    baseline: Path, output: Path, prepared: Path, split: development.Development, seed: int, recipe: Recipe,
) -> tuple[SelectorFitJob, ...]:
    """Plan the first selectors needed to keep each final confidence group out.

    For outer A, the triple selector uses B/C/D and inner selectors use C/D,
    B/D and B/C. Nested contact scores exclude each training video's own group;
    reused contact scores always come from the saved B/C/D contact model.
    """
    jobs = {}
    for outer in GROUPS:
        outer_training = tuple(group for group in GROUPS if group != outer)
        for inner in (None, *outer_training):
            selected = tuple(group for group in outer_training if group != inner)
            ordinary = []
            for identity in split.id_order(split.videos(selected)):
                owner = split.group_by_video[identity]
                contact_groups = outer_training if recipe == 'reuse-contact' else tuple(
                    group for group in selected if group != owner
                )
                ordinary.append(ordinary_pool_job(
                    baseline, output, prepared, split, seed, identity, contact_groups,
                ))
            destination = first_selector_path(output, recipe, outer, inner)
            jobs[destination] = SelectorFitJob(selected, tuple(ordinary), (), destination)
    return tuple(jobs.values())


def followup_selector_jobs(
    baseline: Path, output: Path, prepared: Path, split: development.Development, seed: int, recipe: Recipe,
) -> tuple[SelectorFitJob, ...]:
    """Mix ordinary pools with pieces from selectors that did not train on them."""
    jobs = []
    for outer in (None, *GROUPS):
        selected = tuple(group for group in GROUPS if group != outer)
        ordinary, pieces = [], []
        for identity in split.id_order(split.videos(selected)):
            owner = split.group_by_video[identity]
            if outer is None:
                contact_groups = tuple(group for group in GROUPS if group != owner)
                first_selector = first_selector_path(output, recipe, owner)
            else:
                contact_groups = selected if recipe == 'reuse-contact' else tuple(
                    group for group in selected if group != owner
                )
                first_selector = first_selector_path(output, recipe, outer, owner)
            source = ordinary_pool_job(baseline, output, prepared, split, seed, identity, contact_groups)
            ordinary.append(source)
            piece_path = first_selector.parent / 'pieces' / f'{identity}.joblib'
            pieces.append(PiecePoolJob(source, first_selector, piece_path))
        destination = output / 'followup_selectors' / ''.join(selected) / 'selector.joblib'
        jobs.append(SelectorFitJob(selected, tuple(ordinary), tuple(pieces), destination))
    return tuple(jobs)


def prepare_ordinary_pool(job: OrdinaryPoolJob) -> None:
    if job.reuse_saved:
        for path in (job.contact_model, job.scores, job.pool):
            if not path.is_file():
                raise FileNotFoundError(f'Missing saved baseline input: {path}')
        return
    cache_contact_scores(job.prepared, job.contact_model, job.scores, float('inf'))
    prepare_pool(job.prepared, job.scores, CANDIDATE.cutoff, job.pool, float('inf'))


def load_ordinary_pool(path: Path) -> LabelledOptions:
    """Restore the original ordinary context with unrestricted contact ownership.

    Saved baseline contexts predate the split-piece ownership field. Their
    arrays and targets remain unchanged; ordinary rallies have no such limits.
    """
    pool = joblib.load(path)
    context = replace(pool.video.context, piece_bounds={})
    return replace(pool, video=replace(pool.video, context=context))


def prepare_piece_pool(job: PiecePoolJob) -> None:
    if job.pool.is_file():
        return
    started = time.perf_counter()
    ordinary = load_ordinary_pool(job.ordinary.pool)
    prepared = prepared_video(job.ordinary.prepared)
    contacts = contacts_at_cutoff(joblib.load(job.ordinary.scores), CANDIDATE.cutoff, prepared.evidence.fps)
    model = joblib.load(job.first_selector)
    model.set_params(device='cpu', n_jobs=1)
    prediction = predict_rally_options(RallySelector(model), ordinary.runtime)
    context, _, _, _ = build_split_context(prepared.evidence, contacts, prediction)
    pieces = None
    if context is not None:
        video = replace(ordinary.video, context=context)
        pieces = expand_pool(video)
        pieces = replace(pieces, targets=preferred_targets(pieces, raw_five_targets(pieces)))
    development.dump_atomically(pieces, job.pool)
    write_json(job.pool.with_suffix('.json.gz'), {
        'video': job.ordinary.video_id, 'pieces': 0 if pieces is None else len(pieces.span_rows),
        'rows': 0 if pieces is None else len(pieces.targets), 'seconds': time.perf_counter() - started,
    })


def fit_selector_job(job: SelectorFitJob, seed: int, device: str) -> Path:
    record = job.destination.with_suffix('.json.gz')
    if job.destination.is_file() and record.is_file():
        return job.destination
    started = time.perf_counter()
    pools = [load_ordinary_pool(source.pool) for source in job.ordinary]
    ordinary_spans = sum(len(pool.span_rows) for pool in pools)
    piece_spans = 0
    for source in job.pieces:
        pieces = joblib.load(source.pool)
        if pieces is not None:
            piece_spans += len(pieces.span_rows)
            pools.append(pieces)
    config = TreeFitConfig(
        max_iter=CANDIDATE.selector_rounds, max_leaf_nodes=CANDIDATE.selector_leaves,
        random_seed=20260905 + seed, device=device,
    )
    model = fit_selector(
        [pool.runtime for pool in pools], [pool.targets for pool in pools],
        config, span_weight=RallyFitConfig().span_weight,
    )
    model.set_params(device='cpu', n_jobs=1)
    development.dump_atomically(model, job.destination)
    write_json(record, {
        'groups': job.groups, 'training_videos': [source.video_id for source in job.ordinary],
        'ordinary_spans': ordinary_spans, 'piece_spans': piece_spans,
        'rows': sum(int((pool.targets >= 0).sum()) for pool in pools),
        'seconds': time.perf_counter() - started,
    })
    return job.destination


def prepare_confidence_rows(
    source: OrdinaryPoolJob, first_selector: Path, followup_selector: Path, destination: Path,
) -> None:
    if destination.is_file():
        return
    ordinary = load_ordinary_pool(source.pool)
    prepared = prepared_video(source.prepared)
    contacts = contacts_at_cutoff(joblib.load(source.scores), CANDIDATE.cutoff, prepared.evidence.fps)
    first_model = joblib.load(first_selector)
    followup_model = joblib.load(followup_selector)
    for model in (first_model, followup_model):
        model.set_params(device='cpu', n_jobs=1)
    first_prediction = predict_rally_options(RallySelector(first_model), ordinary.runtime)
    rallies, _, _, features = split_and_correct_rallies(
        prepared.evidence, contacts, first_prediction, RallySelector(followup_model),
    )
    sequences = {rally.sequence.span_id: rally.sequence for rally in rallies}
    labels = np.asarray([
        whole_rally_correctness(sequences[span_id], ordinary.video.rallies, prepared.evidence.fps,
                               tolerance_at_30_fps=10)
        for span_id in features.span_ids
    ], dtype=np.int8)
    development.dump_atomically((features.matrix, labels), destination)


def fit_confidence(paths: list[Path], seed: int, device: str, destination: Path) -> Any:
    if destination.is_file():
        return joblib.load(destination)
    matrices, labels = [], []
    for path in paths:
        matrix, target = joblib.load(path)
        matrices.append(matrix)
        labels.append(target)
    settings = DEFAULT_TRAINING_SETTINGS.confidence
    config = replace(settings, random_seed=settings.random_seed + seed, device=device)
    model = fit_review_model(np.concatenate(matrices), np.concatenate(labels), config)
    model.set_params(device='cpu', n_jobs=1)
    development.dump_atomically(model, destination)
    return model


def export_bundle(baseline: Path, output: Path, seed: int, review_model: Any) -> Path:
    """Keep the original deployed first pass and add the fitted correction layer."""
    # Read this experiment's fixed three-model baseline directly. The runtime
    # loader accepts only the current bundle format.
    original = joblib.load(baseline / 'models' / f'seed-{seed}' / 'bundle' / 'models.joblib')
    followup_selector = output / 'followup_selectors' / 'ABCD' / 'selector.joblib'
    models = AnnotatorModels(
        contact_model=original.contact_model, rally_selector=original.rally_selector,
        followup_selector=RallySelector(joblib.load(followup_selector)),
        review_model=review_model, contact_settings=original.contact_settings, preprocessing=original.preprocessing,
    )
    for model in fitted_trees(models):
        model.set_params(device='cpu', n_jobs=1)
    bundle = output / 'bundle'
    save_models(models, bundle)
    return bundle


def run(
    paths_file: Path, prepared_root: Path, baseline: Path, output_root: Path,
    seed: int, recipe: Recipe, workers: int, gpu_fits: int, device: str,
) -> None:
    """Fit auxiliary inputs, first selectors, follow-up selectors, then confidence."""
    started = time.perf_counter()
    paths = load_paths(paths_file)
    videos = load_original_videos(paths['shots_master'])
    split = development.development_for(videos)
    prepared = prepared_root / 'development'
    inputs = read_json(prepared / 'preparation.json.gz')['inputs']
    output = output_root / recipe / f'seed-{seed}'
    output.mkdir(parents=True, exist_ok=True)
    stage_times = {}
    settings = {'recipe': recipe, 'seed': seed, 'baseline': str(baseline.resolve()),
                'prepared': str(prepared_root.resolve()), 'device': device, 'workers': workers, 'gpu_fits': gpu_fits}
    settings_path = output / 'settings.json.gz'
    if settings_path.is_file() and read_json(settings_path) != settings:
        raise ValueError('Training inputs or settings changed; use a fresh output directory')
    write_json(settings_path, settings)

    with threadpool_limits(limits=2), ThreadPoolExecutor(max_workers=gpu_fits) as gpu, cpu_pool(workers) as cpu:
        stage_started = time.perf_counter()
        if recipe == 'nested-contact':
            records = _training_records(prepared, inputs, split)
            config = replace(ContactFitConfig(), max_leaf_nodes=CANDIDATE.contact_leaves,
                             max_iter=CANDIDATE.contact_rounds, random_seed=20260824 + seed, device=device)
            jobs = []
            for count in (1, 2):
                for groups in combinations(GROUPS, count):
                    sampling_order = split.videos(groups)
                    destination = output / 'contacts' / f'{"".join(groups)}.joblib'
                    jobs.append(gpu.submit(fit_contact, records, sampling_order, split.id_order(sampling_order),
                                           config, destination, float('inf')))
            finish_jobs(jobs, 'auxiliary contact fits')
            del records
        stage_times['auxiliary_contacts'] = time.perf_counter() - stage_started

        first_jobs = first_selector_jobs(baseline, output, prepared, split, seed, recipe)
        followup_jobs = followup_selector_jobs(baseline, output, prepared, split, seed, recipe)
        ordinary_jobs = {source.pool: source for job in (*first_jobs, *followup_jobs) for source in job.ordinary}
        stage_started = time.perf_counter()
        finish_jobs([cpu.submit(prepare_ordinary_pool, job) for job in ordinary_jobs.values()], 'ordinary pools')
        stage_times['ordinary_pools'] = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        finish_jobs([gpu.submit(fit_selector_job, job, seed, device) for job in first_jobs], 'first selector fits')
        stage_times['first_selectors'] = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        piece_jobs = {source.pool: source for job in followup_jobs for source in job.pieces}
        finish_jobs([cpu.submit(prepare_piece_pool, job) for job in piece_jobs.values()], 'split-piece pools')
        stage_times['piece_pools'] = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        finish_jobs([gpu.submit(fit_selector_job, job, seed, device) for job in followup_jobs], 'follow-up selector fits')
        stage_times['followup_selectors'] = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        confidence_paths, jobs = [], []
        for outer in GROUPS:
            training_groups = tuple(group for group in GROUPS if group != outer)
            first_selector = first_selector_path(output, recipe, outer)
            followup_selector = output / 'followup_selectors' / ''.join(training_groups) / 'selector.joblib'
            for identity in split.id_order(split.videos((outer,))):
                source = ordinary_pool_job(baseline, output, prepared, split, seed, identity, training_groups)
                destination = output / 'confidence_rows' / f'{identity}.joblib'
                confidence_paths.append(destination)
                jobs.append(cpu.submit(prepare_confidence_rows, source, first_selector, followup_selector, destination))
        finish_jobs(jobs, 'final confidence rows')
        review_model = fit_confidence(confidence_paths, seed, device, output / 'confidence.joblib')
        bundle = export_bundle(baseline, output, seed, review_model)
        stage_times['confidence_and_export'] = time.perf_counter() - stage_started
        stage_times['training_wall'] = time.perf_counter() - started
        write_json(output / 'training-times.json.gz', stage_times)

        evaluate(paths_file, prepared_root, bundle, output_root / recipe / 'evaluation', seed, cpu, device=device)
    write_json(output / 'complete.json.gz', {'complete': True, 'seconds': time.perf_counter() - started})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True, help='Completed high-shot-3 refit directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recipe', choices=('reuse-contact', 'nested-contact'), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--gpu-fits', type=int, default=3)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    run(args.paths, args.prepared, args.baseline, args.output, args.seed, args.recipe,
        args.workers, args.gpu_fits, args.device)


if __name__ == '__main__':
    main()
