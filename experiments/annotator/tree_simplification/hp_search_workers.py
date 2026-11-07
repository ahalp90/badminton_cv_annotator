"""Reusable fitting, proposal and scoring jobs for the joint settings search."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
from threadpoolctl import threadpool_limits

from annotator.contacts.score_candidates import (
    NEARBY_CONTACT_DISTANCE_AT_30_FPS,
    ScoredContacts,
    remove_nearby_contacts,
    score_contact_features,
)
from annotator.rallies.build_contact_proposals import build_rally_context
from annotator.rallies.refine_rallies import RallySelector, predict_rally_options
from annotator.training.contact import (
    ContactFitConfig,
    ContactTrainingVideo,
    fit_contact_model,
)
from annotator.training.rally import RallyFitConfig
from annotator.training.sequences import SequenceTrainingVideo
from annotator.training.trees import TreeFitConfig
from experiments.annotator.contact_records import stream_payload
from experiments.annotator.development import dump_atomically, pooled_scores
from experiments.annotator.heuristic_only_tune.data import SearchVideo, score_stream
from experiments.annotator.heuristic_only_tune.run import read_json, write_json
from experiments.annotator.monotrack_benchmark.writeup_scripts.metrics import (
    contact_endpoint_stream,
)
from experiments.annotator.tree_simplification.data import PreparedVideo
from experiments.annotator.tree_simplification.options import (
    LabelledOptions,
    expand_pool,
)
from experiments.annotator.tree_simplification.targets import (
    preferred_targets,
    raw_five_targets,
)
from experiments.annotator.tree_simplification.train import SelectorTrainer
from shared.frame_timing import frames_at_fps


@dataclass(frozen=True)
class Candidate:
    body_window: int = 12
    contact_leaves: int = 31
    contact_rounds: int = 180
    cutoff: float = 0.9
    selector_leaves: int = 31
    selector_rounds: int = 400

    @property
    def contact_label(self) -> str:
        return f'w{self.body_window}-c{self.contact_leaves}-r{self.contact_rounds}'

    @property
    def family_label(self) -> str:
        return f'{self.contact_label}-p{round(self.cutoff * 100):03d}'

    @property
    def label(self) -> str:
        return f'{self.family_label}-s{self.selector_leaves}-r{self.selector_rounds}'

    @property
    def changes(self) -> int:
        return sum((self.body_window != 12, self.contact_leaves != 31, self.contact_rounds != 180,
                    self.cutoff != 0.9, self.selector_leaves != 31, self.selector_rounds != 400))


def check_deadline(deadline: float) -> None:
    if time.time() >= deadline:
        raise TimeoutError('The search reached its time limit')


def initialise_cpu_worker() -> None:
    threadpool_limits(limits=1)


def contact_directory(root: Path, candidate: Candidate, seed: int) -> Path:
    return root / 'contacts' / candidate.contact_label / f'seed-{seed}'


def pool_directory(root: Path, candidate: Candidate, seed: int, split: str) -> Path:
    return root / 'pools' / candidate.family_label / f'seed-{seed}' / split


def fit_directory(root: Path, candidate: Candidate, seed: int) -> Path:
    return root / 'fits' / candidate.label / f'seed-{seed}'


@lru_cache(maxsize=2)
def prepared_video(path: Path) -> PreparedVideo:
    return joblib.load(path)


@lru_cache(maxsize=8)
def labelled_pool(path: Path) -> LabelledOptions:
    return joblib.load(path)


def fit_contact(
    records: dict[str, ContactTrainingVideo], sampling_order: tuple[str, ...], fit_order: tuple[str, ...],
    config: ContactFitConfig, destination: Path, deadline: float,
) -> Path:
    check_deadline(deadline)
    if destination.exists() and destination.with_suffix('.json.gz').exists():
        return destination
    started = time.perf_counter()
    fitted = fit_contact_model([records[identity] for identity in sampling_order], config, fit_video_order=fit_order)
    fitted.model.set_params(device='cpu', n_jobs=1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dump_atomically(fitted.model, destination)
    write_json(destination.with_suffix('.json.gz'), {
        'sampling_order': sampling_order, 'fit_order': fit_order,
        'rows': len(fitted.selection.labels), 'seconds': time.perf_counter() - started,
        'best_iteration': getattr(fitted.model, 'best_iteration', None),
    })
    return destination


def cache_contact_scores(
    prepared_path: Path, model_path: Path, destination: Path, deadline: float,
) -> str:
    check_deadline(deadline)
    if destination.exists():
        return prepared_path.stem
    prepared = prepared_video(prepared_path)
    model = joblib.load(model_path)
    model.set_params(device='cpu', n_jobs=1)
    scored = score_contact_features(prepared.features.rows, model, prepared.evidence.fps)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dump_atomically(scored, destination)
    return prepared_path.stem


def contacts_at_cutoff(scored: ScoredContacts, cutoff: float, fps: float) -> ScoredContacts:
    distance = frames_at_fps(NEARBY_CONTACT_DISTANCE_AT_30_FPS, fps)
    kept = remove_nearby_contacts(
        scored.candidates['frame'], scored.candidates['interval_id'], scored.probabilities, cutoff, distance,
    )
    return scored._replace(kept_indices=kept)


def prepare_pool(
    prepared_path: Path, scores_path: Path, cutoff: float, destination: Path, deadline: float,
) -> str:
    check_deadline(deadline)
    if destination.exists() and destination.with_suffix('.json.gz').exists():
        return prepared_path.stem
    started = time.perf_counter()
    prepared = prepared_video(prepared_path)
    scored = contacts_at_cutoff(joblib.load(scores_path), cutoff, prepared.evidence.fps)
    context = build_rally_context(prepared.evidence, scored)
    video = SequenceTrainingVideo(prepared.video.identity, prepared.group, context, prepared.video.rallies)
    pool = expand_pool(video)
    pool = replace(pool, targets=preferred_targets(pool, raw_five_targets(pool)))
    before = score_stream(prepared.video, stream_payload(context.initial, context.events))
    destination.parent.mkdir(parents=True, exist_ok=True)
    dump_atomically(pool, destination)
    write_json(destination.with_suffix('.json.gz'), {
        'video': prepared.video.identity, 'group': prepared.group, 'cutoff': cutoff,
        'rows': len(pool.targets), 'before': before, 'seconds': time.perf_counter() - started,
    })
    return prepared_path.stem


def load_pools(directory: Path, identities: tuple[str, ...]) -> dict[str, LabelledOptions]:
    return {identity: joblib.load(directory / f'{identity}.joblib') for identity in identities}


def fit_selector(
    pools: dict[str, LabelledOptions], candidate: Candidate, seed: int, groups: tuple[str, ...],
    root: Path, device: str, deadline: float,
) -> Path:
    check_deadline(deadline)
    config = TreeFitConfig(
        max_iter=candidate.selector_rounds, max_leaf_nodes=candidate.selector_leaves,
        random_seed=20260905 + seed, device=device,
    )
    output = fit_directory(root, candidate, seed)
    output.mkdir(parents=True, exist_ok=True)
    model = SelectorTrainer(pools, config, output, RallyFitConfig().span_weight).fit(groups)
    model.set_params(device='cpu', n_jobs=1)
    destination = output / 'fits' / ''.join(groups) / 'selector.joblib'
    dump_atomically(model, destination)
    return destination


def score_fold(
    model_path: Path, directory: Path, videos: tuple[SearchVideo, ...], output: Path, deadline: float,
) -> dict[str, Any]:
    check_deadline(deadline)
    result_path = output / 'result.json.gz'
    if result_path.exists():
        return read_json(result_path)
    model = joblib.load(model_path)
    model.set_params(device='cpu', n_jobs=1)
    per_video = {}
    before = {}
    (output / 'streams').mkdir(parents=True, exist_ok=True)
    for video in videos:
        check_deadline(deadline)
        pool_path = directory / f'{video.identity}.joblib'
        pool = labelled_pool(pool_path)
        prediction = predict_rally_options(RallySelector(model), pool.runtime)
        stream = stream_payload(prediction.sequences, prediction.events)
        write_json(output / 'streams' / f'{video.identity}.json.gz', stream)
        per_video[video.identity] = score_stream(video, contact_endpoint_stream(stream))
        before[video.identity] = read_json(pool_path.with_suffix('.json.gz'))['before']
    result = {'per_video': per_video, 'totals': pooled_scores(per_video), 'before': pooled_scores(before)}
    write_json(result_path, result)
    return result
