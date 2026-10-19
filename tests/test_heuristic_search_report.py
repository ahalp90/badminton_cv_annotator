"""Check integer pooling, selected candidates and stored-summary consistency."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from experiments.annotator.heuristic_only_tune.data import COUNT_FIELDS, add_rates
from experiments.annotator.heuristic_only_tune.report import main, result_rows
from experiments.annotator.heuristic_only_tune.run import read_json, write_json


def _candidate(identity: str, videos: tuple[str, str] = ('sset_01', 'sset_21')) -> dict[str, Any]:
    counts = [
        dict(zip(COUNT_FIELDS, (10, 100, 100, 50, 40, 2), strict=True)),
        dict(zip(COUNT_FIELDS, (90, 900, 900, 90, 80, 9), strict=True)),
    ]
    total = {name: sum(row[name] for row in counts) for name in COUNT_FIELDS}
    by_video = {}
    for video, own in zip(videos, counts, strict=True):
        by_video[video] = {tolerance: add_rates(own) for tolerance in ('5', '10')}
    return {
        'candidate_id': identity,
        'by_tolerance': {tolerance: add_rates(total) for tolerance in ('5', '10')},
        'by_video': by_video,
    }


def _summary(
    path: Path, candidates: list[dict[str, Any]], *, split: str = 'train', mode: str = 'heuristic',
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, {
        'schema': 'heuristic-only-tune-summary/1', 'split': split, 'mode': mode,
        'videos': list(candidates[0]['by_video']), 'candidates': candidates,
    })


def test_pools_counts_before_calculating_rates_and_separates_fps(tmp_path: Path) -> None:
    path = tmp_path / 'contacts' / 'summary.json.gz'
    _summary(path, [_candidate('chosen')])
    rows = result_rows([path])
    assert len(rows) == 6
    groups = {row['fps_group']: row for row in rows if row['tolerance_base30'] == 10}
    assert groups['all']['precision'] == 0.14
    assert groups['all']['attributed_f1'] == 0.12
    assert groups['all']['n_videos'] == 2
    assert groups['25']['precision'] == 0.5
    assert groups['30']['precision'] == 0.1
    assert groups['25']['labelled_rallies'] == 10
    assert groups['30']['labelled_rallies'] == 90
    assert {row['stage'] for row in rows} == {'contacts'}


def test_selected_objectives_filter_heuristics_but_keep_full_mode(tmp_path: Path) -> None:
    heuristic = tmp_path / 'heuristic' / 'summary.json.gz'
    full = tmp_path / 'full' / 'summary.json.gz'
    selected = tmp_path / 'selected.json.gz'
    _summary(heuristic, [_candidate('contacts'), _candidate('rallies'), _candidate('other')])
    _summary(full, [_candidate('full_model')], mode='full')
    write_json(selected, {'objectives': {'contacts': ['contacts'], 'rallies': ['rallies']}})
    rows = result_rows([heuristic, full], selected)
    assert {row['candidate_id'] for row in rows} == {'contacts', 'rallies', 'full_model'}
    assert len(rows) == 18


def test_test_split_keeps_only_aggregate_rows(tmp_path: Path) -> None:
    path = tmp_path / 'test' / 'summary.json.gz'
    _summary(path, [_candidate('chosen', ('8', '9'))], split='test')
    rows = result_rows([path])
    assert len(rows) == 2
    assert {row['fps_group'] for row in rows} == {'all'}


@pytest.mark.parametrize('field', ['predicted_contacts', 'attributed_f1'])
def test_stored_totals_or_rates_must_match_video_counts(tmp_path: Path, field: str) -> None:
    path = tmp_path / 'broken' / 'summary.json.gz'
    candidate = _candidate('chosen')
    candidate['by_tolerance']['10'][field] += 1
    _summary(path, [candidate])
    with pytest.raises(ValueError, match='inconsistent totals at ±10'):
        result_rows([path])


def test_video_population_must_match_the_summary(tmp_path: Path) -> None:
    path = tmp_path / 'broken' / 'summary.json.gz'
    _summary(path, [_candidate('chosen')])
    summary = read_json(path)
    summary['videos'].pop()
    write_json(path, summary)
    with pytest.raises(ValueError, match='different video population'):
        result_rows([path])


def test_cli_writes_self_contained_rows_without_source_paths(tmp_path: Path) -> None:
    source = tmp_path / 'contacts' / 'summary.json.gz'
    output = tmp_path / 'table.json.gz'
    _summary(source, [_candidate('chosen')])
    main(['--summaries', str(source), '--output', str(output)])
    document = read_json(output)
    assert document['schema'] == 'heuristic-only-tune-result-table/1'
    assert document['rows'] == result_rows([source])
    assert str(tmp_path) not in str(document)
