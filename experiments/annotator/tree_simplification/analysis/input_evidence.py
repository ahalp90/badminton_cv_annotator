"""Rebuild the preprocessing sweep's results, test analysis, review thresholds and costs from tracked files.

Checks these parts of ``input-preprocessing-evidence.json.gz``:

- ``validation_results``: re-scored from ``streams/input-validation/``.
- ``test_results`` and ``test_analysis``: from the per-video test scores in
  ``streams/input-test/<configuration>/seed-<N>/scores/``. Each score file must name the md5 of
  its tracked prediction file. ``flaw_rescore.py`` re-scores those prediction files.
- ``test_selection``: from ``test-run/selection.json.gz``.
- ``review_thresholds`` and ``test_review_evaluation``: rebuilt with ``review_thresholds`` from the
  two tracked row files. Choosing thresholds again from the validation rows must also give the
  tracked threshold file.
- ``validation_costs``: from ``input-stage-costs.csv``.

Run from the repository root with ``PYTHONPATH=src:.``.
"""

from __future__ import annotations

import argparse
import csv
import math
from decimal import Decimal
from pathlib import Path
from statistics import mean, median
from typing import Any

from annotator.models import model_files
from experiments.annotator.development import pooled_scores
from experiments.annotator.tree_simplification import review_thresholds
from experiments.annotator.tree_simplification.analysis.common import (
    RESULTS,
    SEED_OFFSETS,
    SHOTS_MASTER,
    attributed_f1,
    check,
    md5,
    read_json,
    rescore_folders,
    seed_metrics,
    validation_videos,
)

BASELINE = 'raw_baseline'
TEST_VIDEOS = 46
STAGES = ('prepare', 'contacts', 'pools', 'targets', 'selector', 'export')
KIB_PER_GIB = 1024 ** 2
# Fields that record how a run was made. They are copied facts, not results to rebuild.
COPIED_FACTS = ('source',)


def check_validation_results(results: Path, evidence: dict[str, Any], shots_master: Path) -> None:
    root = results / 'streams/input-validation'
    folders = {
        f'{configuration}/{offset}': root / configuration / f'seed-{offset}'
        for configuration, seeds in evidence['validation_results'].items() for offset in seeds
    }
    scored = rescore_folders(folders, validation_videos(shots_master))
    rebuilt: dict[str, dict[str, Any]] = {}
    for key, per_video in scored.items():
        configuration, offset = key.split('/')
        rebuilt.setdefault(configuration, {})[offset] = {'per_video': per_video, 'totals': pooled_scores(per_video)}
    check(rebuilt, evidence['validation_results'], 'validation_results', skip=COPIED_FACTS)


def test_scores(results: Path, selection: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Each test run's per-video counts, after checking every score file against its prediction file."""
    scores: dict[str, dict[str, Any]] = {}
    for finalist in selection['finalists']:
        candidate = finalist['candidate']
        scores[candidate] = {}
        for offset in SEED_OFFSETS:
            folder = results / 'streams/input-test' / candidate / f'seed-{offset}'
            records = {path.name.removesuffix('.json.gz'): read_json(path) for path in sorted((folder / 'scores').glob('*.json.gz'))}
            if len(records) != TEST_VIDEOS:
                raise ValueError(f'{folder}: expected {TEST_VIDEOS} score files, found {len(records)}')
            hashes = {record['evaluation_sha256'] for record in records.values()}
            if len(hashes) != 1:
                raise ValueError(f'{folder}: score files name more than one evaluation')
            for video, record in records.items():
                if record['complete'] is not True or record['video'] != video:
                    raise ValueError(f'{folder}: score file for {video} is incomplete or names another video')
                if record['stream_md5'] != md5(folder / f'{video}.json.gz'):
                    raise ValueError(f'{folder}: score file for {video} names another prediction file')
            bundle = results / 'models' / f'{candidate}-seed-{offset}'
            per_video = {video: record['score'] for video, record in records.items()}
            scores[candidate][str(offset)] = {
                'per_video': per_video,
                'totals': pooled_scores(per_video),
                'evaluation_sha256': hashes.pop(),
                'bundle_md5': {name: md5(path) for name, path in model_files(bundle).items()},
            }
    return scores


def test_analysis(scores: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Per-seed metrics, means and paired differences from the baseline, overall and per video."""
    videos = sorted(scores[BASELINE]['0']['per_video'])
    rows = {}
    for candidate, seeds in scores.items():
        per_seed = {offset: seed_metrics(run['per_video']) for offset, run in seeds.items()}
        baseline_seeds = {offset: seed_metrics(run['per_video']) for offset, run in scores[BASELINE].items()}
        per_video_delta = {}
        for video in videos:
            delta = {}
            for tolerance in ('10', '5'):
                rally_changes = []
                f1_changes = []
                for offset in seeds:
                    actual = seeds[offset]['per_video'][video][tolerance]
                    baseline = scores[BASELINE][offset]['per_video'][video][tolerance]
                    rally_changes.append(actual['fully_correct_rallies'] - baseline['fully_correct_rallies'])
                    f1_changes.append(attributed_f1(actual) - attributed_f1(baseline))
                delta[f'f1_{tolerance}'] = mean(f1_changes)
                delta[f'fully_correct_rallies_{tolerance}'] = mean(rally_changes)
            per_video_delta[video] = delta
        rows[candidate] = {
            'per_seed': per_seed,
            'mean': {key: mean(values[key] for values in per_seed.values()) for key in per_seed['0']},
            'paired_delta': {
                offset: {key: value - baseline_seeds[offset][key] for key, value in values.items()}
                for offset, values in per_seed.items()
            },
            'per_video_mean_delta': per_video_delta,
        }
    return {'videos': videos, 'rows': rows}


def check_test(results: Path, evidence: dict[str, Any], selection: dict[str, Any]) -> None:
    scores = test_scores(results, selection)
    check(scores, evidence['test_results'], 'test_results', skip=COPIED_FACTS)
    for finalist in selection['finalists']:
        for seed in finalist['seeds']:
            check(scores[finalist['candidate']][str(seed['offset'])]['bundle_md5'], seed['bundle_md5'],
                  f'{finalist["candidate"]} seed {seed["offset"]} test bundle md5 equals the selection')
    check(test_analysis(scores), evidence['test_analysis'], 'test_analysis')


def check_selection(evidence: dict[str, Any], selection: dict[str, Any]) -> None:
    summary = {
        name: selection[name] for name in (
            'schema', 'chosen_at', 'analysis_sha256', 'evaluation_source', 'feature_units', 'fit_policy',
            'primary_tolerance', 'seed_offsets',
        )
    }
    summary['finalists'] = [
        {
            'candidate': finalist['candidate'],
            'objectives': finalist['objectives'],
            'seeds': [
                {
                    'offset': seed['offset'], 'bundle_md5': seed['bundle_md5'],
                    'source': seed['settings']['source'], 'preprocessing': seed['settings']['preprocessing'],
                }
                for seed in finalist['seeds']
            ],
        }
        for finalist in selection['finalists']
    ]
    check(summary, evidence['test_selection'], 'test_selection')


def check_review(results: Path, evidence: dict[str, Any]) -> None:
    validation_path = results / 'input-validation-review-rows.json.gz'
    test_path = results / 'input-test-review-rows.json.gz'
    thresholds_path = results / 'input-review-thresholds.json.gz'
    thresholds = read_json(thresholds_path)
    chosen = review_thresholds.choose_review_thresholds(read_json(validation_path), input_md5=md5(validation_path))
    check(chosen, thresholds, 'input-review-thresholds.json.gz chosen again from the validation rows')
    summary = {name: thresholds[name] for name in ('schema', 'policy', 'validation_rows_md5', 'validation_row_count')}
    summary['candidate_rankings'] = [
        {name: group[name] for name in ('candidate', 'ranking', 'thresholds')} for group in thresholds['candidate_rankings']
    ]
    check(summary, evidence['review_thresholds'], 'review_thresholds')
    evaluation = review_thresholds.evaluate_thresholds(
        read_json(test_path), thresholds, input_md5=md5(test_path), thresholds_md5=md5(thresholds_path),
    )
    check(evaluation, evidence['test_review_evaluation'], 'test_review_evaluation')


def nearest_rank(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def check_costs(results: Path, evidence: dict[str, Any]) -> None:
    with (results / 'input-stage-costs.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    # Decimal keeps the two-decimal times exact through the sums.
    sweep = [row for row in rows if row['stage'] != 'test']
    distributions = {}
    for stage in STAGES:
        stage_rows = [row for row in sweep if row['stage'] == stage]
        elapsed = [float(row['elapsed_s']) for row in stage_rows]
        rss_gib = [int(row['max_rss_kib']) / KIB_PER_GIB for row in stage_rows]
        peak = max(stage_rows, key=lambda row: int(row['max_rss_kib']))
        seed_part = f'/seed-{peak["seed_offset"]}' if peak['seed_offset'] else ''
        distributions[stage] = {
            'n': len(stage_rows),
            'elapsed_p50_s': round(median(elapsed), 2),
            'elapsed_p90_nearest_rank_s': round(nearest_rank(elapsed, 0.9), 2),
            'elapsed_max_s': round(max(elapsed), 2),
            'elapsed_sum_s': float(sum(Decimal(row['elapsed_s']) for row in stage_rows)),
            'user_cpu_sum_h': round(float(sum(Decimal(row['user_cpu_s']) for row in stage_rows)) / 3600, 4),
            'reported_rss_max_gib': round(max(rss_gib), 3),
            'reported_rss_p50_gib': round(median(rss_gib), 3),
            'reported_rss_peak_file': f'remote/candidates/{peak["configuration"]}{seed_part}/{stage}.time',
        }
    elapsed_sum = sum(Decimal(row['elapsed_s']) for row in sweep)
    user_sum = sum(Decimal(row['user_cpu_s']) for row in sweep)
    baseline_seed_zero = [row for row in sweep if row['configuration'] == BASELINE and row['seed_offset'] in ('', '0')]
    timing = {
        'baseline_seed0_accumulated_stage_elapsed_s': float(sum(Decimal(row['elapsed_s']) for row in baseline_seed_zero)),
        'per_stage_elapsed_sum_s': float(elapsed_sum),
        'per_stage_elapsed_sum_h': round(float(elapsed_sum) / 3600, 4),
        'per_stage_user_cpu_sum_h': round(float(user_sum) / 3600, 4),
    }
    lanes = sorted({(row['configuration'], row['seed_offset']) for row in sweep if row['seed_offset']})
    repeats = [(configuration, int(seed)) for configuration, seed in lanes if seed != '0']
    repeated: dict[str, list[int]] = {}
    for configuration, seed in repeats:
        repeated.setdefault(configuration, []).append(seed)
    accounting = {
        'candidate_config_count': len({row['configuration'] for row in sweep}),
        'candidate_preparations': sum(row['stage'] == 'prepare' for row in sweep),
        'fitted_lane_total': len(lanes),
        'matched_repeat_lanes': {
            'baseline': sum(configuration == BASELINE for configuration, _seed in repeats),
            'challengers': sum(configuration != BASELINE for configuration, _seed in repeats),
            'total': len(repeats),
        },
        'preparation_reuse_for_repeat_lanes': len(repeats),
        'repeated_candidates': {configuration: sorted(seeds) for configuration, seeds in repeated.items()},
    }
    recorded = evidence['validation_costs']
    check(distributions, recorded['stage_distributions'], 'validation_costs.stage_distributions')
    check(timing, {key: recorded['timing_semantics'][key] for key in timing}, 'validation_costs.timing_semantics')
    # Receipt counts and the screen/refinement phase split are copied facts from the run records.
    check(accounting, {key: recorded['lane_accounting'][key] for key in accounting}, 'validation_costs.lane_accounting')
    check(sum(seed == '0' for _configuration, seed in lanes), recorded['lane_accounting']['seed0_lanes']['total'],
          'validation_costs.lane_accounting.seed0_lanes.total')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', type=Path, default=RESULTS, help='tracked results folder')
    parser.add_argument('--shots-master', type=Path, default=SHOTS_MASTER, help='ShuttleSet labels')
    args = parser.parse_args()
    evidence = read_json(args.results / 'input-preprocessing-evidence.json.gz')
    selection = read_json(args.results / 'test-run/selection.json.gz')
    check_validation_results(args.results, evidence, args.shots_master)
    check_test(args.results, evidence, selection)
    check_selection(evidence, selection)
    check_review(args.results, evidence)
    check_costs(args.results, evidence)


if __name__ == '__main__':
    main()
