"""Checks for frozen shortlist repeats and original-ShuttleSet selection."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.annotator.tree_hp_tune import repeats


def result(identity: str, score: float, label: str = 'candidate') -> dict:
    return {
        'id': identity, 'label': label, 'per_video': {'original': {'10': {'attributed_f1': score}}},
        'totals': {
            '10': {'attributed_f1': score, 'fully_correct_rallies': 3},
            '5': {'attributed_f1': score},
        },
    }


def candidates_and_results() -> tuple[list, list]:
    specs, results = [], []
    for index in range(48):
        label = 'shipped' if index == 0 else f'candidate-{index}'
        spec = repeats.grid.CandidateSpec(label, {'smooth_window': index + 1}, {})
        specs.append(spec)
        results.append(result(repeats.grid.candidate_id(spec), index / 100, label))
    return specs, results


def test_every_seed_changes_and_other_settings_stay_identical() -> None:
    original = repeats.ORIGINAL_SETTINGS
    assert repeats.seeded_settings(0) == original
    for offset in repeats.SEED_OFFSETS[1:]:
        changed = repeats.seeded_settings(offset)
        assert changed.contact.random_seed == original.contact.random_seed + offset
        assert changed.sequences.serve.random_seed == original.sequences.serve.random_seed + offset
        assert changed.sequences.sequence.random_seed == original.sequences.sequence.random_seed + offset
        assert changed.confidence.random_seed == original.confidence.random_seed + offset
        restored = replace(
            changed,
            contact=replace(changed.contact, random_seed=original.contact.random_seed),
            sequences=replace(
                changed.sequences,
                serve=replace(changed.sequences.serve, random_seed=original.sequences.serve.random_seed),
                sequence=replace(changed.sequences.sequence, random_seed=original.sequences.sequence.random_seed),
            ),
            confidence=replace(changed.confidence, random_seed=original.confidence.random_seed),
        )
        assert restored == original
        record = repeats.settings_record(changed)
        assert json.loads(json.dumps(record)) == record


def test_top_four_training_and_baseline_are_frozen() -> None:
    specs, results = candidates_and_results()
    chosen = repeats.freeze_candidates(results, specs)
    assert [row['label'] for row in chosen] == ['candidate-47', 'candidate-46', 'candidate-45', 'candidate-44', 'shipped']
    results[0]['totals']['10']['attributed_f1'] = 1.0
    chosen = repeats.freeze_candidates(results, specs)
    assert len(chosen) == 4
    assert chosen[0]['label'] == 'shipped'
    with pytest.raises(ValueError, match='all 48'):
        repeats.freeze_candidates(results[:-1], specs)


def write_trial(root: Path, candidate: dict, offset: int, train: float, validation: float) -> None:
    trial = repeats.seed_root(root, offset) / 'trials' / candidate['id']
    (trial / 'bundle_v32').mkdir(parents=True)
    repeats.grid.write_json(trial / 'result.json.gz', result(candidate['id'], train, candidate['label']))
    repeats.grid.write_json(trial / 'validation.json.gz', result(candidate['id'], validation, candidate['label']))
    if offset:
        repeats.grid.write_json(trial / 'repeat-seed.json.gz', {
            'schema': repeats.SCHEMA, 'id': candidate['id'], 'seed_offset': offset,
            'settings': repeats.settings_record(repeats.seeded_settings(offset)),
        })


def test_means_use_validation_only_and_preserve_all_seed_bundles(tmp_path: Path) -> None:
    baseline = {'id': 'baseline', 'label': 'shipped'}
    candidate = {'id': 'tuned', 'label': 'tuned'}
    plan = {'candidates': [candidate, baseline], 'selection': 'validation means'}
    scores = [0.9, 0.4, 0.4, 0.4, 0.4]
    for offset, score in zip(repeats.SEED_OFFSETS, scores, strict=True):
        write_trial(tmp_path, candidate, offset, 1.0, score)
        write_trial(tmp_path, baseline, offset, 0.2, 0.55)
    summary, selected = repeats.summarise(tmp_path, plan)
    assert selected['config_id'] == 'baseline'
    assert selected['mean_validation']['10']['attributed_f1'] == pytest.approx(0.55)
    tuned = next(row for row in summary['results'] if row['id'] == 'tuned')
    assert tuned['mean_validation']['10']['attributed_f1'] == pytest.approx(0.5)
    assert tuned['spread_validation']['10']['attributed_f1']['max'] == 0.9
    assert len(tuned['paired_validation_delta_from_baseline']) == 5
    assert len(selected['bundles']) == len(selected['baseline']['bundles']) == 5
    assert selected['bundles'][0]['bundle'].endswith('grid48/trials/baseline/bundle_v32')
    assert selected['bundles'][1]['bundle'].endswith('repeats/seed-100/trials/baseline/bundle_v32')


def test_exact_mean_ties_prefer_baseline(tmp_path: Path) -> None:
    candidates = [{'id': 'tuned', 'label': 'tuned'}, {'id': 'baseline', 'label': 'shipped'}]
    for candidate in candidates:
        for offset in repeats.SEED_OFFSETS:
            write_trial(tmp_path, candidate, offset, 0.9, 0.6)
    _, selected = repeats.summarise(tmp_path, {'candidates': candidates, 'selection': 'validation means'})
    assert selected['config_id'] == 'baseline'


@pytest.mark.parametrize('threads', [1, 2])
def test_seeded_worker_reuses_existing_fits_and_original_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, threads: int,
) -> None:
    spec = repeats.grid.CandidateSpec('shipped', {}, {})
    candidate = {'id': repeats.grid.candidate_id(spec), 'label': spec.label, 'overrides_base30': {}, 'strategies': {}}
    paths = {name: tmp_path / name for name in ('shots_master', 'stages', 'development_courts')}
    seen = []

    def fit(actual_spec: object, actual_paths: dict, output: Path, actual_threads: int) -> None:
        assert actual_paths == paths and 'test_inputs' not in actual_paths
        assert repeats.grid.DEFAULT_TRAINING_SETTINGS == repeats.seeded_settings(100)
        assert repeats.grid.retrain.DEFAULT_TRAINING_SETTINGS == repeats.seeded_settings(100)
        assert actual_spec == spec and actual_threads == threads
        repeats.grid.write_json(output / 'trials' / candidate['id'] / 'result.json.gz',
                                {'id': candidate['id'], 'label': candidate['label'], 'threads': actual_threads})
        seen.append(('fit', output))

    def validate(actual_spec: object, actual_paths: dict, output: Path, actual_threads: int) -> None:
        assert actual_paths == paths and 'test_inputs' not in actual_paths
        assert actual_spec == spec and actual_threads == threads
        seen.append(('validation', output))

    monkeypatch.setattr(repeats.grid, 'DEFAULT_TRAINING_SETTINGS', repeats.ORIGINAL_SETTINGS)
    monkeypatch.setattr(repeats.grid.retrain, 'DEFAULT_TRAINING_SETTINGS', repeats.ORIGINAL_SETTINGS)
    monkeypatch.setattr(repeats.grid, 'fit_trial', fit)
    monkeypatch.setattr(repeats.grid, 'validate_trial', validate)
    repeats.repeat_job(tmp_path, candidate, 100, paths, 'fit', threads)
    repeats.repeat_job(tmp_path, candidate, 100, paths, 'validate', threads)
    assert seen == [('fit', tmp_path / 'repeats/seed-100'), ('validation', tmp_path / 'repeats/seed-100')]
    fitted = repeats.read_result(tmp_path, candidate, 100, 'fit')
    assert fitted is not None and fitted['threads'] == threads


def test_freeze_checks_snapshot_and_rejects_changed_repeat_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    specs, results = candidates_and_results()
    grid_root = tmp_path / 'grid48'
    (grid_root / 'sticky').mkdir(parents=True)
    repeats.grid.write_json(grid_root / 'baseline-parity.json.gz', {'videos': 32, 'different_selections': []})
    paths = {name: tmp_path / name for name in ('shots_master', 'stages', 'development_courts', 'test_inputs')}
    snapshot = {'source': {'files_md5': {'source': 'fixed'}}, 'inputs': {'original': 'fixed'}}
    monkeypatch.setattr(repeats, 'verify_snapshot', lambda *_args: (snapshot, specs, paths))
    monkeypatch.setattr(repeats.grid, 'completed', lambda *_args: results)
    plan, actual_paths = repeats.freeze_plan(tmp_path, tmp_path)
    assert 'test_inputs' not in actual_paths
    assert plan['original_snapshot'] == snapshot
    assert (tmp_path / 'repeats/seed-100/sticky').resolve() == (grid_root / 'sticky').resolve()
    assert repeats.freeze_plan(tmp_path, tmp_path)[0] == plan
    plan['script_md5'] = 'changed'
    repeats.grid.write_json(tmp_path / 'repeats/plan.json.gz', plan)
    with pytest.raises(ValueError, match='frozen repeat plan changed'):
        repeats.freeze_plan(tmp_path, tmp_path)
