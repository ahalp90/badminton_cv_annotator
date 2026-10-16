"""Keep the search reproducible and selection separate from held-out data."""

from pathlib import Path

import pytest

from experiments.annotator.heuristic_only_tune.run import read_json, write_json
from experiments.annotator.heuristic_only_tune.search import (
    AXES,
    coarse_specs,
    load_summaries,
    rank_key,
    refine_specs,
    seed_specs,
    valid_spec,
    write_candidates,
)
from experiments.annotator.heuristic_tuning.corpus_sweep import candidate_id
from experiments.annotator.heuristic_tuning.sweep import serialise_spec


def row(identity: str, attributed: float, rallies: float, contact: float = 0.8) -> dict:
    metrics = {'attributed_f1': attributed, 'fully_correct_rallies_percent': rallies, 'f1': contact}
    return {'candidate_id': identity, 'by_tolerance': {'10': metrics, '5': metrics}}


def test_objectives_keep_their_distinct_tradeoff() -> None:
    better_contacts = row('contacts', 0.8, 40)
    better_rallies = row('rallies', 0.79, 45)
    assert rank_key(better_contacts, 'contacts') < rank_key(better_rallies, 'contacts')
    assert rank_key(better_rallies, 'rallies') < rank_key(better_contacts, 'rallies')
    assert rank_key(row('a', 0.8, 40), 'contacts') < rank_key(row('b', 0.8, 40), 'contacts')


def test_coarse_search_reproduces_and_covers_all_axes() -> None:
    first = coarse_specs(128, 20261017)
    second = coarse_specs(128, 20261017)
    assert first == second
    assert len({candidate_id(spec) for spec in first}) == 128
    assert first[:3] == seed_specs()
    assert all(valid_spec(spec) for spec in first)
    for key, choices in AXES.items():
        observed = {spec.overrides_base30[key] for spec in first}
        assert set(choices) <= observed


def test_refinement_changes_coupled_axes_and_skips_completed_settings() -> None:
    seed = seed_specs()[0]
    prior = {**row(candidate_id(seed), 0.8, 40), 'spec': serialise_spec(seed)}
    refined = refine_specs([prior], 'contacts', 1)
    assert refined
    assert len({candidate_id(spec) for spec in refined}) == len(refined)
    assert all(candidate_id(spec) != candidate_id(seed) for spec in refined)
    assert any(sum(spec.overrides_base30[key] != seed.overrides_base30[key] for key in AXES) == 2
               for spec in refined)


@pytest.mark.parametrize(('split', 'smoke', 'count'), [('test', False, 32), ('train', True, 32), ('train', False, 8)])
def test_training_selection_refuses_test_and_partial_results(tmp_path: Path, split: str, smoke: bool, count: int) -> None:
    path = tmp_path / 'summary.json.gz'
    write_json(path, {'split': split, 'mode': 'heuristic', 'smoke_test': smoke,
                      'videos': [str(index) for index in range(count)], 'candidates': []})
    with pytest.raises(ValueError, match='complete train split'):
        load_summaries([path], 'train')


def test_candidate_freeze_does_not_overwrite_existing_file(tmp_path: Path) -> None:
    path = tmp_path / 'candidates.json.gz'
    write_candidates(path, 'coarse', seed_specs())
    with pytest.raises(FileExistsError):
        write_candidates(path, 'coarse', seed_specs())
    assert read_json(path)['stage'] == 'coarse'
