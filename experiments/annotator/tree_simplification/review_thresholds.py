"""Choose review-score thresholds from validation rows and apply them unchanged to test rows."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

SCHEMA = 'annotator-review-thresholds/1'
SEED_OFFSETS = (0, 100, 200, 300, 400)
TOLERANCES = (5, 10)
TARGET_COVERAGES = (0.5, 0.7, 0.8, 0.9, 0.95)
LABELS = (1, 0, -1)
ROW_FIELDS = ('candidate', 'seed_offset', 'split', 'video', 'span', 'tolerance', 'ranking', 'correct', 'score')


def _read_json(path: Path) -> Any:
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt') as stream:
        json.dump(value, stream, sort_keys=True, separators=(',', ':'))
        stream.write('\n')


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_rows(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, list) or not values:
        raise ValueError('Review rows must be a non-empty JSON list')
    rows: list[dict[str, Any]] = []
    seen: dict[tuple[Any, ...], dict[str, Any]] = {}
    for index, value in enumerate(values):
        if not isinstance(value, dict) or any(field not in value for field in ROW_FIELDS):
            raise ValueError(f'Row {index} is missing required review fields')
        row = {field: value[field] for field in ROW_FIELDS}
        for field in ('candidate', 'split', 'video', 'ranking'):
            if not isinstance(row[field], str) or not row[field]:
                raise ValueError(f'Row {index} has an invalid {field}')
        if isinstance(row['span'], bool) or not isinstance(row['span'], int):
            raise TypeError(f'Row {index} has an invalid span')
        for field in ('seed_offset', 'tolerance', 'correct'):
            if isinstance(row[field], bool) or not isinstance(row[field], int):
                raise TypeError(f'Row {index} has an invalid {field}')
        if row['tolerance'] not in TOLERANCES or row['correct'] not in (-1, 0, 1):
            raise ValueError(f'Row {index} has an unsupported tolerance or correctness label')
        score = row['score']
        if score is not None:
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError(f'Row {index} has a non-finite or non-numeric score')
            row['score'] = float(score)
        key = tuple(row[field] for field in ROW_FIELDS if field not in ('correct', 'score'))
        previous = seen.get(key)
        if previous is not None:
            if previous != row:
                raise ValueError(f'Conflicting duplicate review row: {key}')
            continue
        seen[key] = row
        rows.append(row)

    tolerance_pairs: dict[tuple[Any, ...], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        base_key = tuple(
            row[field] for field in ROW_FIELDS if field not in ('tolerance', 'correct', 'score')
        )
        pair = tolerance_pairs[base_key]
        if row['tolerance'] in pair:
            raise ValueError(f'Duplicate tolerance row: {base_key}, {row["tolerance"]}')
        pair[row['tolerance']] = row
    for key, pair in tolerance_pairs.items():
        if set(pair) != set(TOLERANCES):
            raise ValueError(f'Each review span must have both tolerances: {key}')
        if pair[5]['score'] != pair[10]['score']:
            raise ValueError(f'Tolerance rows have different scores: {key}')
    return rows


def _candidate_rankings(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return sorted({(row['candidate'], row['ranking']) for row in rows})


def _judgement(counts: dict[int, int]) -> dict[str, Any]:
    judged = counts[1] + counts[0]
    return {
        'correct': counts[1],
        'wrong': counts[0],
        'unknown': counts[-1],
        'error': counts[0] / judged if judged else None,
    }


def _summary_from_counts(
    total: int,
    missing: dict[int, int],
    scored: dict[int, int],
    accepted: dict[int, int],
    *,
    unfiltered: bool = False,
) -> dict[str, Any]:
    if unfiltered:
        accepted = {label: 0 for label in LABELS}
        retained = {label: missing[label] + scored[label] for label in LABELS}
        accepted_count = 0
        filtered_count = 0
        manual_count = total
    else:
        retained = {label: missing[label] + accepted[label] for label in LABELS}
        accepted_count = sum(accepted.values())
        filtered_count = sum(scored.values()) - accepted_count
        manual_count = sum(missing.values())
    retained_count = sum(retained.values())
    return {
        'total': total,
        'scored': sum(scored.values()),
        'missing': sum(missing.values()),
        'retained': retained_count,
        'accepted': accepted_count,
        'filtered': filtered_count,
        'manual_review': manual_count,
        'retained_judgement': _judgement(retained),
        'accepted_judgement': _judgement(accepted),
    }


def _summarise(rows: list[dict[str, Any]], cutoff: float | None) -> dict[str, Any]:
    missing = {label: 0 for label in LABELS}
    scored = {label: 0 for label in LABELS}
    accepted = {label: 0 for label in LABELS}
    for row in rows:
        label = row['correct']
        if row['score'] is None:
            missing[label] += 1
        else:
            scored[label] += 1
            if cutoff is not None and row['score'] >= cutoff:
                accepted[label] += 1
    return _summary_from_counts(len(rows), missing, scored, accepted, unfiltered=cutoff is None)


def _validation_tradeoff(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[tuple[float, float]]]:
    scored_by_score: dict[float, dict[int, int]] = defaultdict(lambda: {label: 0 for label in LABELS})
    missing = {label: 0 for label in LABELS}
    scored = {label: 0 for label in LABELS}
    for row in rows:
        label = row['correct']
        score = row['score']
        if score is None:
            missing[label] += 1
        else:
            scored[label] += 1
            scored_by_score[score][label] += 1

    scores = sorted(scored_by_score)
    cutoffs = sorted({0.0, 1.0, *scores})
    below = {label: 0 for label in LABELS}
    score_index = 0
    curve = [_summary_from_counts(len(rows), missing, scored, scored, unfiltered=True) | {
        'cutoff': None, 'achieved_coverage': 1.0,
    }]
    coverages = []
    for cutoff in cutoffs:
        while score_index < len(scores) and scores[score_index] < cutoff:
            for label in LABELS:
                below[label] += scored_by_score[scores[score_index]][label]
            score_index += 1
        accepted = {label: scored[label] - below[label] for label in LABELS}
        coverage = (sum(missing.values()) + sum(accepted.values())) / len(rows)
        coverages.append((cutoff, coverage))
        curve.append(_summary_from_counts(len(rows), missing, scored, accepted) | {
            'cutoff': cutoff, 'achieved_coverage': coverage,
        })
    return curve, coverages


def choose_review_thresholds(rows: Any, *, input_md5: str | None = None) -> dict[str, Any]:
    """Choose coverage-target thresholds from validation scores only."""
    clean_rows = _normalise_rows(rows)
    if {row['split'] for row in clean_rows} != {'validation'}:
        raise ValueError('Thresholds can only be chosen from validation rows')
    pairs = _candidate_rankings(clean_rows)
    chosen_pairs = []
    for candidate, ranking in pairs:
        pair_rows = [row for row in clean_rows if row['candidate'] == candidate and row['ranking'] == ranking]
        seeds = {row['seed_offset'] for row in pair_rows}
        if seeds != set(SEED_OFFSETS):
            raise ValueError(f'{candidate}/{ranking}: expected seed offsets {SEED_OFFSETS}, got {sorted(seeds)}')
        primary_rows = [row for row in pair_rows if row['tolerance'] == 10]
        tradeoff, coverages = _validation_tradeoff(primary_rows)
        thresholds = [{'name': 'unfiltered', 'target_coverage': None, 'cutoff': None,
                       'achieved_validation_coverage': 1.0}]
        for target in TARGET_COVERAGES:
            closest = min(abs(coverage - target) for _, coverage in coverages)
            tied = [item for item in coverages if math.isclose(
                abs(item[1] - target), closest, rel_tol=0.0, abs_tol=1e-12,
            )]
            cutoff, coverage = max(tied, key=lambda item: item[1])
            thresholds.append({
                'name': f'retain_{target:g}', 'target_coverage': target, 'cutoff': cutoff,
                'achieved_validation_coverage': coverage,
            })
        for threshold in thresholds:
            cutoff = threshold['cutoff']
            threshold['validation_by_tolerance'] = {
                str(tolerance): _tolerance_report(
                    [row for row in pair_rows if row['tolerance'] == tolerance], cutoff,
                )
                for tolerance in TOLERANCES
            }
        chosen_pairs.append({
            'candidate': candidate, 'ranking': ranking, 'thresholds': thresholds,
            'validation_tolerance_10_tradeoff': tradeoff,
        })
    return {
        'schema': SCHEMA,
        'policy': {
            'seed_offsets': list(SEED_OFFSETS),
            'threshold_tolerance': 10,
            'report_tolerances': list(TOLERANCES),
            'target_coverages': list(TARGET_COVERAGES),
            'cutoff_rule': 'score < cutoff filtered; score >= cutoff accepted; missing score retained for manual review',
            'coverage_denominator': 'all distinct tolerance-10 validation rows, including missing scores',
            'seed_note': 'Repeated seeds share videos; no confidence intervals treat seeds as independent data.',
        },
        'validation_rows_md5': input_md5,
        'validation_row_count': len(clean_rows),
        'candidate_rankings': chosen_pairs,
    }


def _validate_thresholds(thresholds: Any) -> dict[tuple[str, str], dict[str, Any]]:
    if not isinstance(thresholds, dict) or thresholds.get('schema') != SCHEMA:
        raise ValueError(f'Expected review threshold schema {SCHEMA!r}')
    policy = thresholds.get('policy')
    if not isinstance(policy, dict) or policy.get('seed_offsets') != list(SEED_OFFSETS):
        raise ValueError('Thresholds do not name the required five seed offsets')
    if policy.get('threshold_tolerance') != 10:
        raise ValueError('Thresholds must be chosen at tolerance 10')
    entries = thresholds.get('candidate_rankings')
    if not isinstance(entries, list) or not entries:
        raise ValueError('Thresholds must contain candidate/ranking entries')
    if any(not isinstance(entry, dict) for entry in entries):
        raise TypeError('Threshold candidate/ranking entries must be objects')
    indexed = {}
    for entry in entries:
        pair = (entry.get('candidate'), entry.get('ranking'))
        if not all(isinstance(value, str) and value for value in pair) or pair in indexed:
            raise ValueError('Threshold candidate/ranking identities must be unique non-empty strings')
        pair_thresholds = entry.get('thresholds')
        expected_names = ['unfiltered', *(f'retain_{target:g}' for target in TARGET_COVERAGES)]
        if not isinstance(pair_thresholds, list) or any(not isinstance(item, dict) for item in pair_thresholds):
            raise TypeError(f'{pair}: threshold entries must be objects')
        if [item.get('name') for item in pair_thresholds] != expected_names:
            raise ValueError(f'{pair}: threshold list is missing')
        for threshold in pair_thresholds:
            name = threshold['name']
            cutoff = threshold.get('cutoff')
            if cutoff is not None and (
                isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)) or not math.isfinite(cutoff)
            ):
                raise ValueError(f"{pair}: invalid threshold 'cutoff' value")
            if (name == 'unfiltered') != (cutoff is None):
                raise ValueError(f'{pair}: only the unfiltered threshold may have a null cutoff')
            expected_target = None if name == 'unfiltered' else float(name.removeprefix('retain_'))
            if threshold.get('target_coverage') != expected_target:
                raise ValueError(f'{pair}: threshold target does not match its canonical name')
            achieved = threshold.get('achieved_validation_coverage')
            if not isinstance(achieved, (int, float)) or not math.isfinite(achieved) or not 0 <= achieved <= 1:
                raise ValueError(f'{pair}: invalid achieved validation coverage')
        indexed[pair] = entry
    return indexed


def _tolerance_report(rows: list[dict[str, Any]], cutoff: float | None) -> dict[str, Any]:
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[row['seed_offset']].append(row)
        by_video[row['video']].append(row)
    seeds = {}
    for seed, seed_rows in sorted(by_seed.items()):
        seed_videos: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in seed_rows:
            seed_videos[row['video']].append(row)
        seeds[str(seed)] = {
            'pooled': _summarise(seed_rows, cutoff),
            'by_video': {video: _summarise(video_rows, cutoff) for video, video_rows in sorted(seed_videos.items())},
        }
    return {
        'pooled': _summarise(rows, cutoff),
        'by_seed': seeds,
        'by_video': {video: _summarise(video_rows, cutoff) for video, video_rows in sorted(by_video.items())},
    }


def evaluate_thresholds(
    rows: Any,
    thresholds: Any,
    *,
    input_md5: str | None = None,
    thresholds_md5: str | None = None,
) -> dict[str, Any]:
    """Apply saved thresholds to test rows without recalibrating them."""
    clean_rows = _normalise_rows(rows)
    if {row['split'] for row in clean_rows} != {'test'}:
        raise ValueError('Threshold application requires test rows')
    chosen_pairs = _validate_thresholds(thresholds)
    row_pairs = set(_candidate_rankings(clean_rows))
    if row_pairs != set(chosen_pairs):
        unknown = sorted(row_pairs - set(chosen_pairs))
        missing = sorted(set(chosen_pairs) - row_pairs)
        raise ValueError(f'Test candidate/ranking set differs from the chosen thresholds; unknown={unknown}, missing={missing}')
    reports = []
    for candidate, ranking in sorted(chosen_pairs):
        pair_rows = [row for row in clean_rows if row['candidate'] == candidate and row['ranking'] == ranking]
        seeds = {row['seed_offset'] for row in pair_rows}
        if seeds != set(SEED_OFFSETS):
            raise ValueError(f'{candidate}/{ranking}: expected test seed offsets {SEED_OFFSETS}, got {sorted(seeds)}')
        for threshold in chosen_pairs[(candidate, ranking)]['thresholds']:
            cutoff = threshold['cutoff']
            by_tolerance = {}
            for tolerance in TOLERANCES:
                tolerance_rows = [row for row in pair_rows if row['tolerance'] == tolerance]
                by_tolerance[str(tolerance)] = _tolerance_report(tolerance_rows, cutoff)
            reports.append({
                'candidate': candidate,
                'ranking': ranking,
                'threshold': threshold['name'],
                'target_coverage': threshold.get('target_coverage'),
                'cutoff': cutoff,
                'achieved_validation_coverage': threshold['achieved_validation_coverage'],
                'by_tolerance': by_tolerance,
            })
    return {
        'schema': 'annotator-review-threshold-evaluation/1',
        'thresholds_md5': thresholds_md5,
        'test_rows_md5': input_md5,
        'reports': reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    choose_parser = commands.add_parser('choose', help='choose thresholds from validation review rows')
    choose_parser.add_argument('--rows', type=Path, required=True)
    choose_parser.add_argument('--output', type=Path, required=True)
    evaluate_parser = commands.add_parser('evaluate', help='apply chosen thresholds to test review rows')
    evaluate_parser.add_argument('--rows', type=Path, required=True)
    evaluate_parser.add_argument('--thresholds', type=Path, required=True)
    evaluate_parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'choose':
        thresholds = choose_review_thresholds(_read_json(args.rows), input_md5=_md5(args.rows))
        _write_json(args.output, thresholds)
    else:
        result = evaluate_thresholds(
            _read_json(args.rows), _read_json(args.thresholds),
            input_md5=_md5(args.rows), thresholds_md5=_md5(args.thresholds),
        )
        _write_json(args.output, result)


if __name__ == '__main__':
    main()
