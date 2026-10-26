"""Fit one small rally-stack configuration and score each prefix on validation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field, replace
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

from annotator.sequence.contacts import stream_payload
from annotator.training.trees import TreeFitConfig
from dataset_builder.vision import load_npy_xz, save_npy_xz
from experiments.annotator.development import dump_atomically, pooled_scores, progress
from experiments.annotator.heuristic_only_tune.data import (
    SearchVideo,
    load_original_videos,
    score_stream,
)
from experiments.annotator.tree_simplification.layers import (
    choose_rows,
    finish_selected,
    fit_layer,
    predict_layer,
)
from experiments.annotator.tree_simplification.options import LabelledOptions
from experiments.annotator.tree_simplification.pools import (
    SCHEMA as POOL_SCHEMA,
)
from experiments.annotator.tree_simplification.pools import (
    edit_kind,
    original_options,
    original_rows,
)
from experiments.annotator.tree_simplification.targets import (
    five_frame_preferred_targets,
)


def score_layers(models: tuple[Any, ...], pool: LabelledOptions) -> tuple[np.ndarray, ...]:
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

    pools: dict[str, LabelledOptions]
    config: TreeFitConfig
    output: Path
    span_weight: float
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
            selected = [pool for pool in self.pools.values() if pool.video.group in groups]
            histories = [self.held_scores(
                tuple(group for group in groups if group != pool.video.group), depth - 1, pool.video.identity,
            ) for pool in selected]
            progress(self.output, 'rally-fit', groups=groups, layer=depth, videos=len(selected))
            started = time.perf_counter()
            model = fit_layer(selected, histories, self.config, span_weight=self.span_weight)
            dump_atomically(model, destination)
            write_json(record_path, {
                'groups': groups, 'layer': depth, 'training_videos': [pool.video.identity for pool in selected],
                'rows': sum(int((pool.targets >= 0).sum()) for pool in selected),
                'span_weight': self.span_weight,
                'fit_wall_s': time.perf_counter() - started,
            })
        self.stacks[key] = (*prefix, model)
        return self.stacks[key]


def evaluate(
    models: tuple[Any, ...], pools: dict[str, LabelledOptions], videos: dict[str, SearchVideo], output: Path,
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
        initial = {sequence.span_id: sequence for sequence in pool.video.initial}
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
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--leaves', type=int, default=15)
    parser.add_argument('--rounds', type=int, default=200)
    parser.add_argument('--min-child-weight', type=float, default=5.0)
    parser.add_argument('--l2', type=float, default=1.0)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=20260905)
    parser.add_argument('--layers', type=int, choices=(1, 2, 3), default=3)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--original-options', action='store_true')
    parser.add_argument('--five-frame-targets', type=Path, help='Use five-frame-preferred targets from these sidecars')
    parser.add_argument('--span-weight', type=float, help='Keep a matched comparison’s fixed training span weight')
    args = parser.parse_args()
    if not 1 <= args.threads <= 28:
        parser.error('--threads must be between 1 and 28')
    if args.span_weight is not None and (not np.isfinite(args.span_weight) or args.span_weight <= 0):
        parser.error('--span-weight must be finite and positive')
    config = TreeFitConfig(
        max_iter=args.rounds, max_leaf_nodes=args.leaves, learning_rate=args.learning_rate,
        l2_regularization=args.l2, random_seed=args.seed,
        device=args.device, min_child_weight=args.min_child_weight,
    )
    manifest = read_json(args.pools / 'pools.json.gz')
    if manifest['schema'] != POOL_SCHEMA:
        raise ValueError('Option pools use an unsupported target rule; prepare fresh pools')
    target_manifest = None
    if args.five_frame_targets:
        target_manifest = read_json(args.five_frame_targets / 'targets.json.gz')
        if target_manifest['pools'] != str(args.pools.resolve()) or target_manifest['pools_manifest'] != manifest:
            raise ValueError('Five-frame targets were prepared from different option pools')
    args.output.mkdir(parents=True, exist_ok=True)
    source = source_record()
    started = time.perf_counter()
    prepared = read_json(Path(manifest['prepared']) / 'preparation.json.gz')['inputs']
    videos = {video.identity: video for video in load_original_videos(Path(prepared['paths']['shots_master']))}
    training, validation = {}, {}
    counts = {}
    for identity, video in videos.items():
        path = args.pools / 'videos' / f'{identity}.joblib'
        pool: LabelledOptions = joblib.load(path)
        metadata = read_json(path.with_suffix('.json.gz'))
        if metadata['split'] != video.split or pool.video.group != prepared['groups'][identity]:
            raise ValueError(f'{identity}: prepared split or group differs')
        selected = original_rows(pool)
        original = original_options(pool, selected=selected)
        counts[identity] = {
            'split': video.split, 'original_options': len(original.targets), 'all_options': len(pool.targets),
            'original_judged_rows': int((original.targets >= 0).sum()),
            'all_judged_rows': int((pool.targets >= 0).sum()),
            'original_judged_spans': sum(
                bool(np.any(original.targets[rows] >= 0)) for rows in original.span_rows.values()
            ),
        }
        five = None
        if args.five_frame_targets:
            five = load_npy_xz(args.five_frame_targets / 'videos' / f'{identity}.npy.xz')
            if five.shape != pool.targets.shape:
                raise ValueError(f'{identity}: five-frame targets do not align with the option pool')
        if args.original_options:
            pool = original
            if five is not None:
                five = five[selected]
        if five is not None:
            pool = replace(pool, targets=five_frame_preferred_targets(pool, five))
        save_npy_xz(args.output / 'targets' / f'{identity}.npy.xz', pool.targets)
        (training if video.split == 'train' else validation)[identity] = pool
    if len(training) != 32 or len(validation) != 8:
        raise ValueError('Comparison requires the original 32 training and 8 validation videos')
    # Anchor every arm/subset to the same full-training original-option scale.
    span_weight = (
        sum(counts[identity]['original_judged_rows'] for identity in training)
        / sum(counts[identity]['original_judged_spans'] for identity in training)
    )
    if args.span_weight is not None:
        span_weight = args.span_weight
    settings = {
        'tree': asdict(config), 'pools': str(args.pools.resolve()), 'pool_settings': manifest,
        'layers': args.layers, 'original_options': args.original_options, 'span_weight': span_weight,
    }
    trainer = LayerTrainer(training, config, args.output, span_weight)
    if target_manifest is not None:
        settings['target_policy'] = {'name': 'five-frame-preferred', 'manifest': target_manifest}
    settings = json.loads(json.dumps(settings, default=sorted))
    settings_path = args.output / 'settings.json.gz'
    if settings_path.exists():
        saved = read_json(settings_path)
        previous_depth = saved.pop('layers')
        requested = {key: value for key, value in settings.items() if key != 'layers'}
        if saved != requested or previous_depth > args.layers:
            raise ValueError('Training configuration changed; use a fresh output directory')
    write_json(settings_path, settings)
    write_json(args.output / 'pool-counts.json.gz', counts)
    groups = tuple(sorted({pool.video.group for pool in training.values()}))
    results = {}
    with threadpool_limits(limits=args.threads):
        for depth in range(1, args.layers + 1):
            models = trainer.fit(groups, depth)
            destination = args.output / f'layers-{depth}'
            result_path = destination / 'validation.json.gz'
            if result_path.exists():
                result = read_json(result_path)
            else:
                result = evaluate(models, validation, videos, destination)
            if not (destination / 'rally-models.joblib').exists():
                dump_atomically(models, destination / 'rally-models.joblib')
            results[str(depth)] = result['totals']
            progress(args.output, 'validated', layers=depth, total_models=depth + 1, totals=result['totals'])
            write_json(args.output / 'result.json.gz', {
                'source': source, 'settings': settings, 'layers': results, 'complete': depth == args.layers,
                'wall_s': time.perf_counter() - started, 'threads': args.threads,
                'non_rally_models': 1,
            })


if __name__ == '__main__':
    main()
