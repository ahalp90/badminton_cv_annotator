"""Fit one small rally-stack configuration and score each prefix on validation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(REPOSITORY := Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(REPOSITORY / 'src'))

from experiments.annotator.heuristic_only_tune.run import (
    read_json,
    source_record,
    write_json,
)

# isort: split

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from annotator.training.sequences import TreeFitConfig
from annotator.training.trees import TreeBackend
from experiments.annotator.heuristic_only_tune.data import (
    SearchVideo,
    load_original_videos,
    score_stream,
)
from experiments.annotator.old_court_regression.retrain import (
    dump_atomically,
    stream_payload,
)
from experiments.annotator.tree_hp_tune.run import pooled_scores
from experiments.annotator.tree_simplification.layers import (
    choose_rows,
    finish_selected,
    fit_layer,
    predict_layer,
)
from experiments.annotator.tree_simplification.options import ExpandedPool
from experiments.annotator.tree_simplification.pilot import progress
from experiments.annotator.tree_simplification.pools import edit_kind, legacy_pool


def score_layers(models: tuple[Any, ...], pool: ExpandedPool) -> tuple[np.ndarray, ...]:
    scores = []
    for model in models:
        scores.append(predict_layer(model, pool, scores))
    return tuple(scores)


@dataclass
class LayerTrainer:
    """Recursively exclude each rally group while keeping contact pools fixed.

    Contact scores use the historical one-level held-group procedure. They
    retain indirect paths between training labels and other groups' features.
    Only the separate validation videos estimate generalisation here.
    """

    pools: dict[str, ExpandedPool]
    config: TreeFitConfig
    output: Path
    stacks: dict[tuple[tuple[str, ...], int], tuple[Any, ...]] = field(default_factory=dict)
    scores: dict[tuple[tuple[str, ...], int, str], tuple[np.ndarray, ...]] = field(default_factory=dict)

    def held_scores(self, groups: tuple[str, ...], depth: int, identity: str) -> tuple[np.ndarray, ...]:
        key = groups, depth, identity
        if key not in self.scores:
            self.scores[key] = score_layers(self.fit(groups, depth), self.pools[identity])
        return self.scores[key]

    def fit(self, groups: tuple[str, ...], depth: int) -> tuple[Any, ...]:
        if depth == 0:
            return ()
        if depth > len(groups):
            raise ValueError('recursive rally training needs at least one group per layer')
        key = groups, depth
        if key in self.stacks:
            return self.stacks[key]
        prefix = self.fit(groups, depth - 1)
        destination = self.output / 'fits' / ''.join(groups) / f'layer-{depth}.joblib'
        record_path = destination.with_suffix('.json.gz')
        if destination.exists() and record_path.exists():
            model = joblib.load(destination)
        else:
            selected = [pool for pool in self.pools.values() if pool.original.group in groups]
            histories = [self.held_scores(
                tuple(group for group in groups if group != pool.original.group), depth - 1, pool.original.identity,
            ) for pool in selected]
            progress(self.output, 'rally-fit', groups=groups, layer=depth, videos=len(selected))
            started = time.perf_counter()
            model = fit_layer(selected, histories, self.config)
            dump_atomically(model, destination)
            write_json(record_path, {
                'groups': groups, 'layer': depth, 'training_videos': [pool.original.identity for pool in selected],
                'rows': sum(int((pool.targets >= 0).sum()) for pool in selected),
                'fit_wall_s': time.perf_counter() - started,
            })
        self.stacks[key] = (*prefix, model)
        return self.stacks[key]


def evaluate(
    models: tuple[Any, ...], pools: dict[str, ExpandedPool], videos: dict[str, SearchVideo], output: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    (output / 'streams').mkdir(parents=True, exist_ok=True)
    per_video = {}
    picks = {}
    for identity, pool in pools.items():
        scores = score_layers(models, pool)[-1]
        sequences, events = finish_selected(pool, scores)
        stream = stream_payload(sequences, events)
        write_json(output / 'streams' / f'{identity}.json.gz', stream)
        per_video[identity] = score_stream(videos[identity], stream)
        initial = {sequence.span_id: sequence for sequence in pool.original.initial}
        picks[identity] = [
            {'span': span, 'row': row, 'score': float(scores[row]), 'target': int(pool.targets[row]),
             'edit_kind': edit_kind(pool.sequences[row], initial)}
            for span, row in choose_rows(pool, scores).items()
        ]
    result = {'per_video': per_video, 'totals': pooled_scores(per_video), 'wall_s': time.perf_counter() - started}
    write_json(output / 'picks.json.gz', picks)
    write_json(output / 'validation.json.gz', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pools', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=list(TreeBackend), default=TreeBackend.XGBOOST)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--leaves', type=int, default=15)
    parser.add_argument('--rounds', type=int, default=200)
    parser.add_argument('--min-child-weight', type=float, default=5.0)
    parser.add_argument('--l2', type=float, default=1.0)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=20260905)
    parser.add_argument('--layers', type=int, choices=(1, 2, 3), default=3)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--legacy-pool', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.threads <= 28:
        parser.error('--threads must be between 1 and 28')
    if args.backend == TreeBackend.HIST_GRADIENT_BOOSTING and args.device != 'cpu':
        parser.error('HGB requires --device cpu')
    config = TreeFitConfig(
        max_iter=args.rounds, max_leaf_nodes=args.leaves, learning_rate=args.learning_rate,
        l2_regularization=args.l2, random_seed=args.seed, backend=TreeBackend(args.backend),
        device=args.device, min_child_weight=args.min_child_weight,
    )
    manifest = read_json(args.pools / 'pools.json.gz')
    settings = {
        'tree': asdict(config), 'pools': str(args.pools.resolve()), 'pool_settings': manifest,
        'layers': args.layers, 'legacy_pool': args.legacy_pool,
    }
    settings = json.loads(json.dumps(settings, default=sorted))
    args.output.mkdir(parents=True, exist_ok=True)
    settings_path = args.output / 'settings.json.gz'
    if settings_path.exists() and read_json(settings_path) != settings:
        raise ValueError('Training configuration changed; use a fresh output directory')
    write_json(settings_path, settings)
    source = source_record()
    started = time.perf_counter()
    prepared = read_json(Path(manifest['prepared']) / 'preparation.json.gz')['inputs']
    videos = {video.identity: video for video in load_original_videos(Path(prepared['paths']['shots_master']))}
    training, validation = {}, {}
    for identity, video in videos.items():
        path = args.pools / 'videos' / f'{identity}.joblib'
        pool: ExpandedPool = joblib.load(path)
        metadata = read_json(path.with_suffix('.json.gz'))
        if metadata['split'] != video.split or pool.original.group != prepared['groups'][identity]:
            raise ValueError(f'{identity}: prepared split or group differs')
        if args.legacy_pool:
            pool = legacy_pool(pool)
        (training if video.split == 'train' else validation)[identity] = pool
    if len(training) != 32 or len(validation) != 8:
        raise ValueError('Comparison requires the original 32 training and 8 validation videos')
    groups = tuple(sorted({pool.original.group for pool in training.values()}))
    trainer = LayerTrainer(training, config, args.output)
    results = {}
    with threadpool_limits(limits=args.threads):
        for depth in range(1, args.layers + 1):
            models = trainer.fit(groups, depth)
            destination = args.output / f'layers-{depth}'
            result = evaluate(models, validation, videos, destination)
            dump_atomically(models, destination / 'rally-models.joblib')
            results[str(depth)] = result['totals']
            progress(args.output, 'validated', layers=depth, total_models=depth + 1, totals=result['totals'])
            write_json(args.output / 'result.json.gz', {
                'source': source, 'settings': settings, 'layers': results, 'complete': depth == args.layers,
                'wall_s': time.perf_counter() - started, 'threads': args.threads,
            })


if __name__ == '__main__':
    main()
