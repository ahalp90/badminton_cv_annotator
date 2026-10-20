from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from annotator.outcomes.point_winner import Half
from annotator.sequence.contacts import ContactEvent, ContactSequence
from experiments.annotator.tree_simplification import pools
from experiments.annotator.tree_simplification.options import ExpandedPool


def example_pool() -> ExpandedPool:
    first = ContactEvent(10, 0.9, Half.TOP)
    second = ContactEvent(20, 0.8, Half.BOT)
    addition = ContactEvent(15, 0.7, Half.TOP)
    other = ContactEvent(40, 0.85, Half.BOT)
    other_addition = ContactEvent(35, 0.6, Half.TOP)
    initial = (
        ContactSequence(0, 5, 25, (first, second)),
        ContactSequence(1, 30, 45, (other,)),
    )
    sequences = (
        initial[0],
        replace(initial[0], events=(first, addition, second)),
        replace(initial[0], events=(second,)),
        replace(initial[0], events=(addition, second)),
        initial[1],
        replace(initial[1], events=(other_addition, other)),
    )
    legacy_options = (SimpleNamespace(span_id=0, sequence=sequences[1]),)
    original = SimpleNamespace(
        identity='fixture',
        group='A',
        initial=initial,
        pool=SimpleNamespace(options=legacy_options),
    )
    return ExpandedPool(
        sequences,
        np.arange(12, dtype=np.float32).reshape(6, 2),
        np.asarray([1, -1, 0, 1, 0, 1], dtype=np.int8),
        {0: np.asarray([0, 1, 2, 3]), 1: np.asarray([4, 5])},
        original,
    )


def test_legacy_pool_keeps_original_choices_and_preserves_row_alignment() -> None:
    expanded = example_pool()

    legacy = pools.legacy_pool(expanded)

    assert legacy.sequences == (expanded.sequences[0], expanded.sequences[1], expanded.sequences[4])
    np.testing.assert_array_equal(legacy.matrix, expanded.matrix[[0, 1, 4]])
    np.testing.assert_array_equal(legacy.targets, expanded.targets[[0, 1, 4]])
    np.testing.assert_array_equal(legacy.span_rows[0], [0, 1])
    np.testing.assert_array_equal(legacy.span_rows[1], [2])
    assert legacy.original is expanded.original


def test_pool_summary_counts_targets_and_positive_span_coverage() -> None:
    summary = pools.pool_summary(example_pool(), 'train')

    assert (summary['video'], summary['group'], summary['split']) == ('fixture', 'A', 'train')
    assert summary['target_counts_by_edit_kind'] == {
        'keep': {'-1': 0, '0': 1, '1': 1},
        'add': {'-1': 1, '0': 0, '1': 1},
        'delete': {'-1': 0, '0': 1, '1': 0},
        'mixed': {'-1': 0, '0': 0, '1': 1},
    }
    assert summary['positive_target_span_coverage'] == {
        'legacy': 1,
        'expanded': 2,
        'total_spans': 2,
    }


def test_cli_wires_pool_arguments_and_help(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    captured: dict[str, Any] = {}

    def prepare(*args: Any, **kwargs: Any) -> list[dict[str, object]]:
        captured['args'] = args
        captured['kwargs'] = kwargs
        return []

    monkeypatch.setattr(pools, 'prepare_pools', prepare)
    assert pools.main([
        '--prepared', 'prepared', '--pilot', 'pilot', '--output', 'output',
        '--workers', '2', '--only', 'fixture', '--max-edits', '2',
        '--deletion-candidates', '4',
    ]) == 0
    assert captured['args'] == (Path('prepared'), Path('pilot'), Path('output'))
    assert captured['kwargs'] == {
        'workers': 2,
        'only': 'fixture',
        'max_edits': 2,
        'deletion_candidates': 4,
    }

    with pytest.raises(SystemExit) as error:
        pools.main(['--help'])
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert '--prepared' in help_text
    assert '--pilot' in help_text
    assert '--only' in help_text
