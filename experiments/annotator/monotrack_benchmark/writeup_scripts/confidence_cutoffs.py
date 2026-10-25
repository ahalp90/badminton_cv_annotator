"""Select review cut-offs on fresh validation discovery runs, then evaluate on test.

Every distinct validation confidence score is an attainable cut-off (keep scores >=
cut-off). Balanced maximises mean seed F1 for retaining fully correct found rallies.
Keep half minimises the pooled distance from half of all found validation rallies.
Boundaries selects the lowest cut-off reaching 99% mean seed boundary precision,
requiring a non-empty selection in every seed. Ties use the lowest cut-off.

Outputs keep the report's existing curve columns and cut-off names. An unattainable
99% target has a null cut-off and null validation/test statistics. Test curves are
reported at validation cut-offs only; test outcomes never choose a cut-off.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(REPO := Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(REPO / 'src'))

from annotator.sequence.contacts import scale_frames, stream_from_payload
from annotator.training.sequences import (
    DEFAULT_RALLY_BOUNDARY_POLICY,
    RallyBoundaryPolicy,
)
from experiments.annotator.heuristic_only_tune.data import load_original_videos
from experiments.annotator.monotrack_benchmark.benchmark_scripts.common import (
    read_json,
    write_json,
)
from experiments.annotator.monotrack_benchmark.writeup_scripts.metrics import (
    BOUNDARY_RIGHT,
    SEEDS,
    judge_found_rallies,
)

TOLERANCE = 10
BOUNDARY_TARGET = 99.0


def validation_found(
    validation: Path, shots_master: Path, boundary_policy: RallyBoundaryPolicy = DEFAULT_RALLY_BOUNDARY_POLICY,
) -> tuple[pd.DataFrame, int]:
    """Judge fresh validation streams against every labelled development rally."""
    videos = [video for video in load_original_videos(shots_master) if video.split == 'validation']
    rows = []
    for seed in SEEDS:
        for video in videos:
            stream = read_json(validation / f'seed-{seed}' / f'{video.identity}.json.gz')
            sequences, _ = stream_from_payload(stream)
            judged = judge_found_rallies(
                sequences, video.rallies, scale_frames(TOLERANCE, video.fps), fps=video.fps,
                boundary_policy=boundary_policy,
            )
            for record in judged:
                rows.append({'seed': seed, 'video': video.identity, **record,
                             'confidence': stream['confidence'][str(record['span_id'])]})
    return pd.DataFrame(rows), sum(len(video.rallies) for video in videos)


def attainable_thresholds(found: pd.DataFrame) -> np.ndarray:
    """Every distinct validation score, in ascending order; tied scores stay together."""
    scores = found['confidence'].to_numpy(dtype=float)
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError('validation confidence scores must be non-empty and finite')
    return np.unique(scores)


def counts_at_thresholds(scores: np.ndarray, thresholds: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Count weighted retained rows for all cut-offs with one sort and cumulative sum."""
    order = np.argsort(scores)
    sorted_scores = scores[order]
    # Prefix zero also handles a cut-off above every score without a special branch.
    cumulative = np.concatenate(([0], np.cumsum(weights[order], dtype=np.int64)))
    return cumulative[-1] - cumulative[np.searchsorted(sorted_scores, thresholds, side='left')]


def covered_at_thresholds(run: pd.DataFrame, mask: pd.Series, thresholds: np.ndarray) -> np.ndarray:
    """Count labelled rallies represented by at least one retained correct prediction."""
    best = run.loc[mask].groupby(['video', 'rally_id'])['confidence'].max().to_numpy(dtype=float)
    return counts_at_thresholds(best, thresholds, np.ones(len(best), dtype=np.int64))


def curve(found: pd.DataFrame, total_rallies: int, thresholds) -> pd.DataFrame:
    """Existing curve schema, computed in O(rows log rows + seeds * cut-offs)."""
    thresholds = np.asarray(thresholds, dtype=float)
    runs = []
    for seed, run in found.groupby('seed'):
        scores = run['confidence'].to_numpy(dtype=float)
        correct = run['outcome'] == 'fully_correct'
        boundaries = run['outcome'].isin(BOUNDARY_RIGHT)
        kept = counts_at_thresholds(scores, thresholds, np.ones(len(run), dtype=np.int64))
        fully_correct = counts_at_thresholds(scores, thresholds, correct.to_numpy(dtype=np.int64))
        boundary_correct = counts_at_thresholds(scores, thresholds, boundaries.to_numpy(dtype=np.int64))
        no_label = counts_at_thresholds(
            scores, thresholds, (run['outcome'] == 'no_labelled_rally').to_numpy(dtype=np.int64),
        )
        precision = np.divide(100 * fully_correct, kept, out=np.full(len(kept), np.nan), where=kept > 0)
        boundary_precision = np.divide(100 * boundary_correct, kept, out=np.full(len(kept), np.nan), where=kept > 0)
        denominator = kept + int(correct.sum())
        f1 = np.divide(200 * fully_correct, denominator, out=np.zeros(len(kept)), where=denominator > 0)
        runs.append(pd.DataFrame({
            'seed': seed, 'threshold': thresholds, 'found': len(run), 'kept': kept,
            'fully_correct_kept': fully_correct, 'no_labelled_rally_kept': no_label,
            'boundaries_right_kept': boundary_correct, 'fully_correct_share': precision,
            'boundaries_right_share': boundary_precision,
            'coverage': 100 * covered_at_thresholds(run, correct, thresholds) / total_rallies,
            'boundary_coverage': 100 * covered_at_thresholds(run, boundaries, thresholds) / total_rallies,
            'f1': f1,
        }))
    return pd.concat(runs, ignore_index=True)


def pareto_mask(precision: pd.Series, coverage: pd.Series) -> np.ndarray:
    """Whether any threshold improves one mean without worsening the other."""
    points = pd.DataFrame({'precision': precision.to_numpy(), 'coverage': coverage.to_numpy()})
    best_by_precision = points.groupby('precision')['coverage'].max().sort_index(ascending=False)
    better_precision = best_by_precision.cummax().shift(1, fill_value=-np.inf)
    same_best = points['precision'].map(best_by_precision)
    better_best = points['precision'].map(better_precision)
    return ((points['coverage'] >= same_best) & (points['coverage'] > better_best)).to_numpy()


def seed_summary(curves: pd.DataFrame) -> pd.DataFrame:
    """Mean, lowest and highest seed for each cut-off, with pooled counts."""
    columns = ['found', 'kept', 'fully_correct_kept', 'boundaries_right_kept', 'fully_correct_share',
               'boundaries_right_share', 'coverage', 'boundary_coverage', 'f1', 'no_labelled_rally_kept']
    summary = curves.groupby('threshold')[columns].agg(['mean', 'min', 'max'])
    pooled = curves.groupby('threshold')[['boundaries_right_kept', 'kept']].sum()
    summary[('pooled_boundaries_right_share', 'mean')] = 100 * pooled['boundaries_right_kept'] / pooled['kept']
    summary[('kept', 'sum')] = pooled['kept']
    summary[('found', 'sum')] = curves.groupby('threshold')['found'].sum()
    summary[('nonempty_seeds', 'count')] = curves.groupby('threshold')['boundaries_right_share'].count()
    summary[('seeds', 'count')] = curves.groupby('threshold')['seed'].nunique()
    summary[('pareto', 'mean')] = pareto_mask(summary[('fully_correct_share', 'mean')], summary[('coverage', 'mean')])
    return summary


def choose_cutoffs(validation: pd.DataFrame) -> dict[str, float | None]:
    """Choose only from validation; lower thresholds win tied objectives."""
    validation = validation.sort_index()
    balanced = float(validation[('f1', 'mean')].idxmax())
    half_distance = (2 * validation[('kept', 'sum')] - validation[('found', 'sum')]).abs()
    keep_half = float(half_distance.idxmin())
    feasible = (validation[('boundaries_right_share', 'mean')] >= BOUNDARY_TARGET) & (
        validation[('nonempty_seeds', 'count')] == validation[('seeds', 'count')]
    )
    qualifying = validation.index[feasible]
    boundaries = float(qualifying.min()) if len(qualifying) else None
    return {'balanced': balanced, 'keep_half': keep_half, 'boundaries_99': boundaries}


def at_cutoff(summary: pd.DataFrame, threshold: float | None) -> dict[str, float] | None:
    if threshold is None:
        return None
    row = summary.loc[threshold]
    return {f'{column}_{statistic}': float(row[(column, statistic)]) for column, statistic in summary.columns}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation', type=Path, required=True, help='fresh 32-video validation root with seed-*/')
    parser.add_argument('--metrics', type=Path, required=True, help='fresh metrics.py output')
    parser.add_argument('--boundary-before-seconds', type=float, default=2.0)
    parser.add_argument('--boundary-after-seconds', type=float, default=3.0)
    parser.add_argument('--test-rallies', type=int, default=3327)
    parser.add_argument('--shots-master', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    policy = RallyBoundaryPolicy(args.boundary_before_seconds, args.boundary_after_seconds)
    metrics_policy = read_json(args.metrics / 'evaluation-policy.json.gz')['boundary_policy']
    if metrics_policy != asdict(policy):
        raise ValueError('validation and test metrics must use the same boundary policy')
    args.out.mkdir(parents=True, exist_ok=True)
    found, validation_rallies = validation_found(args.validation, args.shots_master, policy)
    if set(found['seed']) != set(SEEDS):
        raise ValueError('validation must contain found rallies for all five seeds')
    found.to_csv(args.out / 'validation_found_rallies.csv.gz', index=False)
    thresholds = attainable_thresholds(found)
    validation_curves = curve(found, validation_rallies, thresholds)
    validation_curves.to_csv(args.out / 'validation_confidence_curve.csv.gz', index=False)
    validation = seed_summary(validation_curves)
    cutoffs = choose_cutoffs(validation)
    test_found = pd.concat([
        pd.read_csv(args.metrics / f'found_rallies_annotator_discovery_seed{seed}.csv.gz').assign(seed=seed)
        for seed in SEEDS
    ])
    test_curves = curve(test_found, args.test_rallies, thresholds)
    test_curves.to_csv(args.out / 'test_confidence_curve.csv.gz', index=False)
    test = seed_summary(test_curves)
    result = {
        'boundary_policy': asdict(policy), 'selection_split': 'validation',
        'thresholds_searched': len(thresholds), 'boundary_target_feasible': cutoffs['boundaries_99'] is not None,
        'validation_rallies': validation_rallies,
        'validation_found_rallies_per_seed': found.groupby('seed').size().tolist(),
        'cutoffs': cutoffs,
        'validation': {name: at_cutoff(validation, value) for name, value in cutoffs.items()},
        'test': {name: at_cutoff(test, value) for name, value in cutoffs.items()},
    }
    write_json(args.out / 'confidence_cutoffs.json.gz', result)
    print(json.dumps({'cutoffs': cutoffs, 'thresholds_searched': len(thresholds)}, indent=1))


if __name__ == '__main__':
    main()
