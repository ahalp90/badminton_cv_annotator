"""Build the training search, freeze a shortlist, and select on validation."""

from __future__ import annotations

import argparse
import random
from itertools import product
from pathlib import Path
from typing import Any

from experiments.annotator.heuristic_only_tune.run import read_json, write_json
from experiments.annotator.heuristic_tuning.corpus_sweep import candidate_id
from experiments.annotator.heuristic_tuning.sweep import CandidateSpec, serialise_spec

# Speeds and frame counts use the annotator's base-30 contract.
AXES: dict[str, tuple[float, ...]] = {
    'rest_speed': (1 / 600, 1 / 400, 1 / 240, 1 / 120, 0.01, 1 / 60),
    'rest_window': (5, 9, 15, 18, 25),
    'start_speed': (1 / 120, 1 / 80, 1 / 60, 0.025, 0.03, 1 / 24),
    'start_min_frames': (1, 2, 3, 4, 6),
    'end_rest_frames': (18, 24, 30, 36, 54, 72, 90, 108),
    'smooth_window': (3, 5, 7, 9),
    'impulse_floor_half_window_frames': (6, 9, 12, 18, 24),
    'contact_impulse_multiple': (1.5, 2, 3, 4, 6, 8),
    'contact_dedup_radius_frames': (1, 2, 3, 5, 7),
    'contact_suppression_radius_frames': (3, 5, 7, 9, 12),
    'body_unit_half_window': (6, 12, 24),
    'replay_mask_min_frames': (5, 15, 30),
    'blip_max_frames': (6, 12, 24),
    'gap_state_demotion_bound': (30, 75, 150),
}
COUPLED_AXES = (
    ('rest_speed', 'start_speed'),
    ('rest_window', 'end_rest_frames'),
    ('smooth_window', 'contact_impulse_multiple'),
    ('smooth_window', 'impulse_floor_half_window_frames'),
    ('contact_dedup_radius_frames', 'contact_suppression_radius_frames'),
    ('contact_impulse_multiple', 'contact_suppression_radius_frames'),
)
OBJECTIVES = ('contacts', 'rallies')


def spec_from_row(row: dict[str, Any], label: str = 'grid') -> CandidateSpec:
    return CandidateSpec(label, dict(row['overrides_base30']), dict(row.get('strategies', {})))


def seed_specs() -> list[CandidateSpec]:
    fixed = {
        'impulse_floor_half_window_frames': 12, 'contact_impulse_multiple': 4,
        'contact_dedup_radius_frames': 3, 'contact_suppression_radius_frames': 9,
        'body_unit_half_window': 12, 'replay_mask_min_frames': 15,
        'blip_max_frames': 12, 'gap_state_demotion_bound': 75,
    }
    earlier = {
        **fixed, 'rest_speed': 0.01, 'rest_window': 15, 'start_speed': 0.03,
        'start_min_frames': 3, 'smooth_window': 5, 'end_rest_frames': 30,
    }
    scaled = dict(earlier)
    for key in ('rest_speed', 'start_speed'):
        scaled[key] *= 25 / 30
    for key in (
        'rest_window', 'start_min_frames', 'smooth_window', 'end_rest_frames',
        'impulse_floor_half_window_frames', 'contact_dedup_radius_frames', 'contact_suppression_radius_frames',
    ):
        scaled[key] *= 30 / 25
    moderate = {
        **fixed, 'rest_speed': 1 / 240, 'rest_window': 9, 'start_speed': 1 / 60,
        'start_min_frames': 3, 'smooth_window': 5, 'end_rest_frames': 54,
    }
    return [CandidateSpec(label, values, {'span_open': 'BACK_FILL'}) for label, values in (
        ('earlier_integer', earlier), ('earlier_scaled', scaled), ('moderate', moderate),
    )]


def valid_spec(spec: CandidateSpec) -> bool:
    values = spec.overrides_base30
    return (
        values['rest_speed'] < values['start_speed']
        and values['contact_dedup_radius_frames'] <= values['contact_suppression_radius_frames']
    )


def coarse_specs(count: int, seed: int) -> list[CandidateSpec]:
    if count < 3:
        raise ValueError('coarse search needs at least the three historical seeds')
    generator = random.Random(seed)
    specs = seed_specs()
    seen = {candidate_id(spec) for spec in specs}
    historical = specs[0]
    for key, choices in AXES.items():
        for value in choices:
            spec = CandidateSpec('sensitivity', {**historical.overrides_base30, key: value}, historical.strategies)
            identity = candidate_id(spec)
            if len(specs) < count and valid_spec(spec) and identity not in seen:
                seen.add(identity)
                specs.append(spec)
    while len(specs) < count:
        spec = CandidateSpec('coarse', {key: generator.choice(values) for key, values in AXES.items()}, {
            'span_open': generator.choice(('BACK_FILL', 'REGION_START')),
        })
        identity = candidate_id(spec)
        if valid_spec(spec) and identity not in seen:
            seen.add(identity)
            specs.append(spec)
    return specs


def rank_key(row: dict[str, Any], objective: str) -> tuple[Any, ...]:
    main = row['by_tolerance']['10']
    narrow = row['by_tolerance']['5']
    contact = main['attributed_f1']
    rally = main['fully_correct_rallies_percent']
    first, second = (contact, rally) if objective == 'contacts' else (rally, contact)
    return (-first, -second, -main['f1'], -narrow['attributed_f1'], row['candidate_id'])


def load_summaries(paths: list[Path], split: str) -> list[dict[str, Any]]:
    rows = {}
    expected_count = 32 if split == 'train' else 8
    population = None
    for path in paths:
        summary = read_json(path)
        if (summary['split'] != split or summary['mode'] != 'heuristic' or summary['smoke_test']
                or len(summary['videos']) != expected_count):
            raise ValueError(f'{path}: selection requires the complete {split} split')
        videos = sorted(summary['videos'])
        if population is not None and videos != population:
            raise ValueError('summaries do not use the same videos')
        population = videos
        for row in summary['candidates']:
            identity = row['candidate_id']
            if identity in rows and rows[identity]['by_tolerance'] != row['by_tolerance']:
                raise ValueError(f'{identity}: repeated configuration has different scores')
            rows[identity] = row
    return list(rows.values())


def adjacent_values(value: float, values: tuple[float, ...]) -> tuple[float, ...]:
    below = [candidate for candidate in values if candidate < value]
    above = [candidate for candidate in values if candidate > value]
    return tuple(([max(below)] if below else []) + ([min(above)] if above else []))


def refine_specs(rows: list[dict[str, Any]], objective: str, parents: int) -> list[CandidateSpec]:
    seen = {row['candidate_id'] for row in rows}
    specs = []

    def append(spec: CandidateSpec) -> None:
        identity = candidate_id(spec)
        if valid_spec(spec) and identity not in seen:
            seen.add(identity)
            specs.append(spec)

    for row in sorted(rows, key=lambda item: rank_key(item, objective))[:parents]:
        parent = spec_from_row(row['spec'])
        for key, choices in AXES.items():
            for value in adjacent_values(parent.overrides_base30[key], choices):
                append(CandidateSpec('refine', {**parent.overrides_base30, key: value}, parent.strategies))
        for first, second in COUPLED_AXES:
            first_values = adjacent_values(parent.overrides_base30[first], AXES[first])
            second_values = adjacent_values(parent.overrides_base30[second], AXES[second])
            for first_value, second_value in product(first_values, second_values):
                append(CandidateSpec('refine', {
                    **parent.overrides_base30, first: first_value, second: second_value,
                }, parent.strategies))
        other_open = 'REGION_START' if parent.strategies['span_open'] == 'BACK_FILL' else 'BACK_FILL'
        append(CandidateSpec('refine', parent.overrides_base30, {'span_open': other_open}))
    return specs


def write_candidates(output: Path, stage: str, specs: list[CandidateSpec], **metadata: Any) -> None:
    if output.exists():
        raise FileExistsError(f'refusing to replace a candidate file: {output}')
    if not specs:
        raise ValueError('no new configurations remain')
    write_json(output, {
        'schema': 'heuristic-only-tune-candidates/1', 'stage': stage, **metadata,
        'candidates': [{'label': spec.label, **serialise_spec(spec)} for spec in specs],
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    coarse = subparsers.add_parser('coarse')
    coarse.add_argument('--count', type=int, default=128)
    coarse.add_argument('--seed', type=int, default=20261017)
    coarse.add_argument('--output', type=Path, required=True)
    refine = subparsers.add_parser('refine')
    refine.add_argument('--results', type=Path, nargs='+', required=True)
    refine.add_argument('--objective', choices=OBJECTIVES, required=True)
    refine.add_argument('--parents', type=int, default=3)
    refine.add_argument('--output', type=Path, required=True)
    shortlist = subparsers.add_parser('shortlist')
    shortlist.add_argument('--results', type=Path, nargs='+', required=True)
    shortlist.add_argument('--per-objective', type=int, default=3)
    shortlist.add_argument('--output', type=Path, required=True)
    select = subparsers.add_parser('select')
    select.add_argument('--results', type=Path, required=True)
    select.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'coarse':
        write_candidates(args.output, 'coarse', coarse_specs(args.count, args.seed), seed=args.seed, axes=AXES)
        return
    if args.command == 'select':
        rows = load_summaries([args.results], 'validation')
    else:
        rows = load_summaries(args.results, 'train')
    if args.command == 'refine':
        if args.parents < 1:
            raise ValueError('parents must be positive')
        specs = refine_specs(rows, args.objective, args.parents)
        write_candidates(args.output, 'refine', specs, objective=args.objective,
                         previous_candidate_ids=sorted(row['candidate_id'] for row in rows))
        return
    per_objective = 1 if args.command == 'select' else args.per_objective
    if per_objective < 1:
        raise ValueError('per-objective must be positive')
    specs_by_id = {}
    choices = {}
    for objective in OBJECTIVES:
        selected = sorted(rows, key=lambda row: rank_key(row, objective))[:per_objective]
        choices[objective] = [row['candidate_id'] for row in selected]
        for row in selected:
            specs_by_id[row['candidate_id']] = spec_from_row(row['spec'], objective)
    write_candidates(args.output, 'selected' if args.command == 'select' else 'shortlist',
                     list(specs_by_id.values()), objectives=choices)


if __name__ == '__main__':
    main()
