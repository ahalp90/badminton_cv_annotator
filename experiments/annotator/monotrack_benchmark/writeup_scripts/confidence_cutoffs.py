"""Choose discovery-mode confidence cut-offs on the eight validation videos, then read them on test.

The confidence score is the review ranker's score, which ranks found rallies by how likely
they are to be fully correct. Keeping only rallies at or above a cut-off trades coverage
(labelled rallies delivered) for accuracy (share of kept rallies that are right). All three
cut-offs come from validation, never from test:

- balanced: the highest F1 for "fully correct", where precision is the fully correct share of
  kept rallies and recall is the share of all fully correct found rallies that are kept;
- keep half: the review threshold that keeps half of the found rallies on validation, pooled
  over the five seeds. It is read from the review-threshold search in
  ``tree_simplification/results/input-review-thresholds.json.gz``;
- boundaries: the lowest cut-off where at least 99% of kept rallies, pooled over the five
  seeds, have the right boundaries.

A cut-off is Pareto-optimal when no other cut-off gives both a higher fully correct share and
higher coverage (seed means).

Found rallies are judged with ``metrics.judge_found_rallies`` at ±10 frames, the same rule as test.

Test videos also hold 524 ShuttleSet22 rallies left out of the test set. ``metrics.py`` already
drops found rallies that sit only on those, so every kept test rally can be scored. Validation
labels hold every rally in ShuttleSet's release; ``left_out_rallies.py`` (run first) checks this.

Usage: python confidence_cutoffs.py --data DIR --shots-master CSV --out DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Bound inline: ruff's E402 allows sys.path calls above imports, but not assignments.
sys.path.insert(0, str(REPO := Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(REPO / 'src'))

from annotator.sequence.contacts import scale_frames, stream_from_payload
from experiments.annotator.annotator_closeout_new_courts.scripts.evaluate_saved import (
    score_video,
)
from experiments.annotator.heuristic_only_tune.data import load_original_videos
from experiments.annotator.monotrack_benchmark.benchmark_scripts.common import read_json
from experiments.annotator.monotrack_benchmark.writeup_scripts.metrics import (
    BOUNDARY_RIGHT,
    RUNS,
    SEEDS,
    judge_found_rallies,
)

TOLERANCE = 10
THRESHOLDS = np.round(np.arange(0, 1.0001, 0.005), 3)
BOUNDARY_TARGET = 99.0
TREE_RESULTS = REPO / 'experiments/annotator/tree_simplification/results'


def keep_half_cutoff() -> float:
    """The fitted review ranker's threshold that keeps half of the validation rallies, baseline preprocessing."""
    record = read_json(TREE_RESULTS / 'input-review-thresholds.json.gz')
    (ranking,) = [ranking for ranking in record['candidate_rankings']
                  if (ranking['candidate'], ranking['ranking']) == ('raw_baseline', 'fitted')]
    (threshold,) = [threshold for threshold in ranking['thresholds'] if threshold['target_coverage'] == 0.5]
    return float(threshold['cutoff'])


def validation_found(validation: Path, shots_master: Path) -> tuple[pd.DataFrame, int]:
    """Judged found rallies for every seed and validation video, plus the labelled rally count."""
    videos = [video for video in load_original_videos(shots_master) if video.split == 'validation']
    rows = []
    for seed in SEEDS:
        saved = read_json(TREE_RESULTS / f'test-run/candidates/raw_baseline/seed-{seed}/export/result.json.gz')
        strict_total = 0
        for video in videos:
            stream = read_json(validation / f'seed-{seed}' / f'{video.identity}.json.gz')
            sequences, _ = stream_from_payload(stream)
            judged = judge_found_rallies(sequences, video.rallies, scale_frames(TOLERANCE, video.fps))
            for record in judged:
                rows.append({'seed': seed, 'video': video.identity, **record,
                             'confidence': stream['confidence'][str(record['span_id'])]})
            _, _, _, rally_rows = score_video(video.identity, video.split, stream, video.rallies, video.fps, TOLERANCE)
            strict_total += sum(bool(row['fully_correct']) for row in rally_rows)
        # The validation runs match the tracked predictions, so the strict count must match the saved score.
        if strict_total != saved['totals'][str(TOLERANCE)]['fully_correct_rallies']:
            raise ValueError(f'seed {seed}: strict fully correct {strict_total} != saved validation score')
    return pd.DataFrame(rows), sum(len(video.rallies) for video in videos)


def curve(found: pd.DataFrame, total_rallies: int, thresholds) -> pd.DataFrame:
    """Kept-rally shares at each cut-off, one row per seed and cut-off."""
    found = found.assign(fully_correct=found['outcome'] == 'fully_correct',
                         boundaries_right=found['outcome'].isin(BOUNDARY_RIGHT))
    rows = []
    for seed, run in found.groupby('seed'):
        all_correct = run['fully_correct'].sum()
        for threshold in thresholds:
            kept = run[run['confidence'] >= threshold]
            correct = kept['fully_correct'].sum()
            precision = 100 * correct / len(kept) if len(kept) else np.nan
            recall = 100 * correct / all_correct
            rows.append({
                'seed': seed, 'threshold': float(threshold), 'kept': len(kept),
                'fully_correct_kept': int(correct),
                'no_labelled_rally_kept': int((kept['outcome'] == 'no_labelled_rally').sum()),
                'boundaries_right_kept': int(kept['boundaries_right'].sum()),
                'fully_correct_share': precision,
                'boundaries_right_share': 100 * kept['boundaries_right'].mean() if len(kept) else np.nan,
                'coverage': 100 * kept.loc[kept['fully_correct'], ['video', 'rally_id']].drop_duplicates().shape[0]
                / total_rallies,
                'boundary_coverage': 100 * kept.loc[kept['boundaries_right'], ['video', 'rally_id']]
                .drop_duplicates().shape[0] / total_rallies,
                'f1': 2 * precision * recall / (precision + recall) if correct else 0.0,
            })
    return pd.DataFrame(rows)


def seed_summary(curves: pd.DataFrame) -> pd.DataFrame:
    """Mean, lowest and highest seed for each cut-off."""
    columns = ['kept', 'fully_correct_kept', 'boundaries_right_kept', 'fully_correct_share', 'boundaries_right_share',
               'coverage', 'boundary_coverage', 'f1', 'no_labelled_rally_kept']
    summary = curves.groupby('threshold')[columns].agg(['mean', 'min', 'max'])
    pooled = curves.groupby('threshold')[['boundaries_right_kept', 'kept']].sum()
    summary[('pooled_boundaries_right_share', 'mean')] = 100 * pooled['boundaries_right_kept'] / pooled['kept']
    precision, coverage = summary[('fully_correct_share', 'mean')], summary[('coverage', 'mean')]
    summary[('pareto', 'mean')] = [
        not ((precision >= share) & (coverage >= covered) & ((precision > share) | (coverage > covered))).any()
        for share, covered in zip(precision, coverage, strict=True)]
    return summary


def choose_cutoffs(validation: pd.DataFrame) -> dict[str, float]:
    balanced = float(validation[('f1', 'mean')].idxmax())
    qualifying = validation.index[validation[('pooled_boundaries_right_share', 'mean')] >= BOUNDARY_TARGET]
    boundaries = float(qualifying.min())
    return {'balanced': balanced, 'keep_half': keep_half_cutoff(), 'boundaries_99': boundaries}


def at_cutoff(summary: pd.DataFrame, threshold: float) -> dict[str, dict[str, float]]:
    row = summary.loc[threshold]
    return {f'{column}_{statistic}': float(row[(column, statistic)]) for column, statistic in summary.columns}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation', type=Path, default=RUNS / 'validation/32_videos',
                        help='validation discovery runs with seed-*/ folders')
    parser.add_argument('--data', type=Path, required=True, help='metrics.py output with found_rallies_*_seed*.csv.gz')
    parser.add_argument('--test-rallies', type=int, default=3327)
    parser.add_argument('--shots-master', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    left_out = json.loads((args.data / 'left_out_summary.json').read_text())
    if left_out['validation_left_out_rallies'] != 0:
        raise ValueError('validation videos have left-out rallies; flag their found rallies before choosing cut-offs')
    found, validation_rallies = validation_found(args.validation, args.shots_master)
    found.to_csv(args.out / 'validation_found_rallies.csv.gz', index=False)
    grid = seed_summary(curve(found, validation_rallies, THRESHOLDS.tolist()))
    cutoffs = choose_cutoffs(grid)
    validation_curves = curve(found, validation_rallies, sorted(set(THRESHOLDS.tolist()) | set(cutoffs.values())))
    validation_curves.to_csv(args.out / 'validation_confidence_curve.csv', index=False)
    validation = seed_summary(validation_curves)

    test_found = pd.concat([pd.read_csv(args.data / f'found_rallies_annotator_discovery_seed{seed}.csv.gz').assign(seed=seed)
                            for seed in SEEDS])
    test_thresholds = sorted(set(THRESHOLDS.tolist()) | set(cutoffs.values()))
    test_curves = curve(test_found, args.test_rallies, test_thresholds)
    test_curves.to_csv(args.out / 'test_confidence_curve.csv', index=False)
    test = seed_summary(test_curves)
    validation_at = {name: at_cutoff(validation, value) for name, value in cutoffs.items()}
    result = {
        'validation_rallies': validation_rallies, 'validation_found_rallies_per_seed': found.groupby('seed').size()
        .tolist(), 'cutoffs': cutoffs, 'validation': validation_at,
        'test': {name: at_cutoff(test, value) for name, value in cutoffs.items()},
    }
    (args.out / 'confidence_cutoffs.json').write_text(json.dumps(result, indent=1))
    print(json.dumps({'cutoffs': cutoffs}, indent=1))
    for name, value in cutoffs.items():
        for split, table in (('validation', validation), ('test', test)):
            row = table.loc[value]
            print(f'{name:>13} {value:.3f} {split:>10}: kept {row[("kept", "mean")]:.0f}, fully correct '
                  f'{row[("fully_correct_share", "mean")]:.1f}% ({row[("fully_correct_kept", "mean")]:.0f}), boundaries '
                  f'{row[("boundaries_right_share", "mean")]:.1f}% (pooled '
                  f'{row[("pooled_boundaries_right_share", "mean")]:.1f}%), pareto {bool(row[("pareto", "mean")])}, '
                  f'coverage {row[("coverage", "mean")]:.1f}%, boundary coverage {row[("boundary_coverage", "mean")]:.1f}%')


if __name__ == '__main__':
    main()
