"""Fit split-piece correction and confidence while keeping each judged group out.

For a held group, use its contact model throughout the temporary selector fits.
Save prepared evidence and scored candidates on disk, then load the videos and
proposal pools needed for the current step.
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from annotator.contacts.build_features import ContactFeatures
from annotator.contacts.score_candidates import (
    ContactModelConfig,
    ScoredContacts,
    score_contact_features,
)
from annotator.inference.prepare_annotation_inputs import PreparedAnnotationInputs
from annotator.rallies.build_contact_proposals import (
    build_rally_context,
    build_rally_options,
)
from annotator.rallies.refine_rallies import RallySelector, predict_rally_options
from annotator.rallies.split_and_correct_rallies import (
    build_split_context,
    split_and_correct_rallies,
)
from annotator.training.rally import (
    RallyFitConfig,
    RallyTrainingPool,
    fit_review_model,
    fit_selector,
    label_rally_options,
)
from annotator.training.sequences import (
    LabelledRally,
    SequenceTrainingVideo,
    whole_rally_correctness,
)
from annotator.training.trees import TreeFitConfig


@dataclass(frozen=True)
class PreparedTrainingInputs:
    """Prepared detector evidence and contact features reused by every fit."""

    evidence: PreparedAnnotationInputs
    features: ContactFeatures


@dataclass(frozen=True)
class PreparedTrainingVideo:
    """One labelled video's prepared inputs, saved once for reuse.

    ``inputs_path`` stores a record with ``evidence`` and ``features`` fields.
    Video order supplied to the fitter determines proposal-row order.
    """

    video_id: str
    group: str
    inputs_path: Path
    rallies: tuple[LabelledRally, ...]


@dataclass(frozen=True)
class ScoredTrainingVideo:
    """All scored contact candidates and the ordinary labelled rally options."""

    contacts: ScoredContacts
    pool: RallyTrainingPool


def save_training_inputs(
    video_id: str, group: str, evidence: PreparedAnnotationInputs, features: ContactFeatures,
    rallies: tuple[LabelledRally, ...], destination: Path,
) -> PreparedTrainingVideo:
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(PreparedTrainingInputs(evidence, features), destination, compress=3)
    return PreparedTrainingVideo(video_id, group, destination, rallies)


def prepare_training_data(
    video: PreparedTrainingVideo, contact_model: Path, contact_settings: ContactModelConfig,
    rally_settings: RallyFitConfig, destination: Path,
) -> Path:
    """Score a video's candidates once and retain them beside its ordinary pool."""
    prepared = joblib.load(video.inputs_path)
    evidence, features = prepared.evidence, prepared.features
    model = joblib.load(contact_model)
    model.set_params(device='cpu', n_jobs=1)
    contacts = score_contact_features(features.rows, model, evidence.fps, contact_settings)
    context = build_rally_context(evidence, contacts)
    training = SequenceTrainingVideo(video.video_id, video.group, context, video.rallies)
    options = build_rally_options(context, rally_settings.max_edits, rally_settings.deletion_candidates)
    pool = label_rally_options(training, options, rally_settings)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ScoredTrainingVideo(contacts, pool), destination, compress=3)
    return destination


def fit_selector_from_files(
    ordinary_paths: Sequence[Path], piece_paths: Sequence[Path], settings: RallyFitConfig, destination: Path,
) -> Path:
    """Fit ordinary pools followed by piece pools, preserving the supplied order."""
    pools = [joblib.load(path).pool for path in ordinary_paths]
    for path in piece_paths:
        pool = joblib.load(path)
        if pool is not None:
            pools.append(pool)
    model = fit_selector(
        [pool.options for pool in pools], [pool.targets for pool in pools], settings.tree,
        span_weight=settings.span_weight,
    )
    model.set_params(device='cpu', n_jobs=1)
    selector = RallySelector(model, settings.max_edits, settings.deletion_candidates)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selector, destination, compress=3)
    return destination


def prepare_piece_training_data(
    video: PreparedTrainingVideo, ordinary_path: Path, first_selector: Path,
    settings: RallyFitConfig, destination: Path,
) -> Path:
    """Label alternatives for pieces from a first selector that withheld this group."""
    evidence = joblib.load(video.inputs_path).evidence
    ordinary = joblib.load(ordinary_path)
    prediction = predict_rally_options(joblib.load(first_selector), ordinary.pool.options)
    context, _, _, _ = build_split_context(evidence, ordinary.contacts, prediction)
    pool = None
    if context is not None:
        training = SequenceTrainingVideo(video.video_id, video.group, context, video.rallies)
        options = build_rally_options(context, settings.max_edits, settings.deletion_candidates)
        pool = label_rally_options(training, options, settings)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pool, destination, compress=3)
    return destination


def prepare_held_group_results(
    video: PreparedTrainingVideo, ordinary_path: Path, first_selector: Path, followup_selector: Path,
    settings: RallyFitConfig, pieces_path: Path, confidence_path: Path,
) -> None:
    """Use one held-out prediction for final confidence and full follow-up training."""
    evidence = joblib.load(video.inputs_path).evidence
    ordinary = joblib.load(ordinary_path)
    prediction = predict_rally_options(joblib.load(first_selector), ordinary.pool.options)
    corrected = split_and_correct_rallies(evidence, ordinary.contacts, prediction, joblib.load(followup_selector))
    pool = None
    if corrected.piece_options is not None:
        training = SequenceTrainingVideo(video.video_id, video.group, corrected.piece_options.context, video.rallies)
        pool = label_rally_options(training, corrected.piece_options, settings)
    pieces_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pool, pieces_path, compress=3)
    final_by_span = {rally.sequence.span_id: rally.sequence for rally in corrected.rallies}
    labels = np.asarray([
        whole_rally_correctness(final_by_span[span_id], video.rallies, evidence.fps,
                               tolerance_at_30_fps=settings.review_tolerance_at_30_fps)
        for span_id in corrected.review_features.span_ids
    ], dtype=np.int8)
    confidence_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump((corrected.review_features.matrix, labels), confidence_path, compress=3)


def initialise_preparation_worker() -> None:
    threadpool_limits(limits=1)


def fit_followup_models(
    videos: Sequence[PreparedTrainingVideo], ordinary_paths: Mapping[str, Path],
    held_contact_models: Mapping[str, Path], contact_settings: ContactModelConfig,
    rally_settings: RallyFitConfig, confidence_settings: TreeFitConfig, working_directory: Path,
    *, workers: int = 1, gpu_fits: int = 1,
) -> tuple[RallySelector, Any]:
    """Fit correction and confidence with one fixed contact model per held group.

    Temporary first selectors omit the group they predict. Their training
    contacts use the fixed model, including in-sample contacts for the remaining
    groups. The final judged group stays outside every contributing fit.
    """
    groups = tuple(sorted({video.group for video in videos}))
    if len(groups) < 3:
        raise ValueError('follow-up training needs at least three distinct training groups')
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
        initializer=initialise_preparation_worker,
    ) as cpu, ThreadPoolExecutor(max_workers=gpu_fits) as gpu:
        for held_group in groups:
            fit_held_group(
                videos, ordinary_paths, held_contact_models[held_group], held_group,
                contact_settings, rally_settings, working_directory, cpu, gpu,
            )

    first_pass_pools = [ordinary_paths[video.video_id] for video in videos]
    piece_pools = [working_directory / 'pieces' / f'{video.video_id}.joblib' for video in videos]
    followup_path = fit_selector_from_files(
        first_pass_pools, piece_pools, rally_settings, working_directory / 'followup-selector.joblib',
    )
    matrices, labels = [], []
    for group in groups:
        for video in videos:
            if video.group == group:
                matrix, targets = joblib.load(working_directory / 'confidence' / f'{video.video_id}.joblib')
                matrices.append(matrix)
                labels.append(targets)
    review = fit_review_model(np.concatenate(matrices), np.concatenate(labels), confidence_settings)
    return joblib.load(followup_path), review


def fit_held_group(
    videos: Sequence[PreparedTrainingVideo], ordinary_paths: Mapping[str, Path], contact_model: Path, held_group: str,
    contact_settings: ContactModelConfig, settings: RallyFitConfig, working_directory: Path,
    cpu: ProcessPoolExecutor, gpu: ThreadPoolExecutor,
) -> None:
    """Make one group's final predictions using fits that never saw that group.

    With A held out, keep the B/C/D contact model fixed. The B/C/D first selector
    predicts A. First selectors fitted on C/D, B/D and B/C supply pieces from B,
    C and D respectively. Fit the B/C/D follow-up on ordinary rallies and those
    pieces, then use it to correct A's split rallies.
    """
    training = tuple(video for video in videos if video.group != held_group)
    held_videos = tuple(video for video in videos if video.group == held_group)
    training_groups = tuple(dict.fromkeys(video.group for video in training))
    directory = working_directory / f'held-{held_group}'
    prepared_paths = {}
    jobs = []
    for video in training:
        destination = directory / 'ordinary' / f'{video.video_id}.joblib'
        prepared_paths[video.video_id] = destination
        jobs.append(cpu.submit(prepare_training_data, video, contact_model, contact_settings, settings, destination))
    for job in jobs:
        job.result()

    first_selector = fit_selector_from_files(
        list(prepared_paths.values()), (), settings, directory / 'first-selector.joblib',
    )
    selector_jobs = {}
    for predicted_group in training_groups:
        selected_paths = [prepared_paths[video.video_id] for video in training if video.group != predicted_group]
        destination = directory / f'without-{predicted_group}' / 'first-selector.joblib'
        selector_jobs[predicted_group] = gpu.submit(fit_selector_from_files, selected_paths, (), settings, destination)
    for job in selector_jobs.values():
        job.result()

    piece_paths, jobs = [], []
    for video in training:
        destination = directory / 'pieces' / f'{video.video_id}.joblib'
        piece_paths.append(destination)
        jobs.append(cpu.submit(
            prepare_piece_training_data, video, prepared_paths[video.video_id],
            selector_jobs[video.group].result(), settings, destination,
        ))
    for job in jobs:
        job.result()
    followup_selector = fit_selector_from_files(
        list(prepared_paths.values()), piece_paths, settings, directory / 'followup-selector.joblib',
    )
    jobs = []
    for video in held_videos:
        jobs.append(cpu.submit(
            prepare_held_group_results, video, ordinary_paths[video.video_id], first_selector, followup_selector, settings,
            working_directory / 'pieces' / f'{video.video_id}.joblib',
            working_directory / 'confidence' / f'{video.video_id}.joblib',
        ))
    for job in jobs:
        job.result()
