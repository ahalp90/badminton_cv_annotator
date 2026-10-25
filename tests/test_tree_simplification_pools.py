from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from annotator.outcomes.point_winner import Half
from annotator.sequence.contacts import ContactEvent, ContactSequence
from annotator.sequence.repairs import RallyRepairs
from annotator.training.sequences import (
    DEFAULT_RALLY_BOUNDARY_POLICY,
    SequenceTrainingVideo,
)
from experiments.annotator.tree_simplification import pools
from experiments.annotator.tree_simplification.options import LabelledOptions


def example_pool() -> LabelledOptions:
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
    original = SequenceTrainingVideo(
        'fixture', 'A', 30.0, initial, (first, second, other),
        RallyRepairs((sequences[1],), {}, {}), (), 50,
    )
    return LabelledOptions(
        sequences,
        np.arange(12, dtype=np.float32).reshape(6, 2),
        np.asarray([1, -1, 0, 1, 0, 1], dtype=np.int8),
        {0: np.asarray([0, 1, 2, 3]), 1: np.asarray([4, 5])},
        original,
    )


def test_original_options_keep_choices_and_row_alignment() -> None:
    options = example_pool()

    original = pools.original_options(options)

    assert original.sequences == (options.sequences[0], options.sequences[1], options.sequences[4])
    np.testing.assert_array_equal(original.matrix, options.matrix[[0, 1, 4]])
    np.testing.assert_array_equal(original.targets, options.targets[[0, 1, 4]])
    np.testing.assert_array_equal(original.span_rows[0], [0, 1])
    np.testing.assert_array_equal(original.span_rows[1], [2])
    assert original.video is options.video


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
        'original_options': 1,
        'all_options': 2,
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
        '--prepared', 'prepared', '--contact-run', 'contact-run', '--output', 'output',
        '--workers', '2', '--only', 'fixture', '--max-edits', '2',
        '--deletion-candidates', '4',
    ]) == 0
    assert captured['args'] == (Path('prepared'), Path('contact-run'), Path('output'))
    assert captured['kwargs'] == {
        'workers': 2,
        'only': 'fixture',
        'max_edits': 2,
        'deletion_candidates': 4,
        'boundary_policy': DEFAULT_RALLY_BOUNDARY_POLICY,
    }

    with pytest.raises(SystemExit) as error:
        pools.main(['--help'])
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert '--prepared' in help_text
    assert '--contact-run' in help_text
    assert '--only' in help_text
