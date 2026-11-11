"""Every final confidence fold must exclude its group through all fitted inputs."""

from pathlib import Path

import pytest

from experiments.annotator.development import Development
from experiments.annotator.tree_simplification.post_split_training import (
    first_selector_jobs,
    first_selector_path,
    followup_selector_jobs,
)


@pytest.mark.parametrize('recipe,first_fit_count', [('reuse-contact', 16), ('nested-contact', 10)])
def test_confidence_training_excludes_outer_group_at_every_stage(tmp_path: Path, recipe, first_fit_count):
    owners = {f'{group}-{index}': group for group in 'ABCD' for index in range(2)}
    split = Development(owners, dict.fromkeys(owners, 30.0), dict(zip(owners, range(len(owners)), strict=True)))
    arguments = (tmp_path / 'baseline', tmp_path / 'run', tmp_path / 'prepared', split, 100, recipe)
    first_jobs = first_selector_jobs(*arguments)
    first_by_path = {job.destination: job for job in first_jobs}
    followup_jobs = followup_selector_jobs(*arguments)
    assert len(first_jobs) == first_fit_count
    assert len(followup_jobs) == 5

    for followup in followup_jobs:
        outer = set('ABCD') - set(followup.groups)
        assert len(followup.ordinary) == len(followup.pieces) == 2 * len(followup.groups)
        for ordinary, piece in zip(followup.ordinary, followup.pieces, strict=True):
            assert ordinary == piece.ordinary
            assert not outer.intersection(ordinary.contact_groups)
            first = first_by_path[piece.first_selector]
            assert ordinary.video_group not in first.groups
            assert not outer.intersection(first.groups)
            for training_input in first.ordinary:
                assert not outer.intersection(training_input.contact_groups)
                if recipe == 'nested-contact' or not outer:
                    assert ordinary.video_group not in training_input.contact_groups
                if recipe == 'nested-contact':
                    assert training_input.video_group not in training_input.contact_groups
            if not outer:
                assert ordinary.reuse_saved
                assert ordinary.video_group not in ordinary.contact_groups
            elif recipe == 'nested-contact':
                assert ordinary.video_group not in ordinary.contact_groups
            else:
                assert ordinary.video_group in ordinary.contact_groups

    for outer in 'ABCD':
        first = first_by_path[first_selector_path(tmp_path / 'run', recipe, outer)]
        assert outer not in first.groups
        assert all(outer not in source.contact_groups for source in first.ordinary)


def test_nested_pair_selectors_use_other_groups_contact_model(tmp_path: Path):
    owners = {group: group for group in 'ABCD'}
    split = Development(owners, dict.fromkeys(owners, 30.0), dict(zip(owners, range(4), strict=True)))
    jobs = first_selector_jobs(tmp_path / 'baseline', tmp_path / 'run', tmp_path, split, 0, 'nested-contact')
    pair = next(job for job in jobs if job.groups == ('C', 'D'))
    assert [(source.video_group, source.contact_groups) for source in pair.ordinary] == [
        ('C', ('D',)), ('D', ('C',)),
    ]
    triple = next(job for job in jobs if job.groups == ('B', 'C', 'D'))
    assert [(source.video_group, source.contact_groups) for source in triple.ordinary] == [
        ('B', ('C', 'D')), ('C', ('B', 'D')), ('D', ('B', 'C')),
    ]
