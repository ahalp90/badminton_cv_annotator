"""Focused tests for the sequential accepted-contact opener search."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest
from accepted_contact_trace import (
    AcceptedContactEvidence,
    IncomingVerdict,
    OpenerCategory,
    closest_post_contact_run,
    has_credible_outgoing_motion,
    search_accepted_contacts,
)


class Motion(NamedTuple):
    n_frames: int
    largest_step_ratio: float


class Trend(NamedTuple):
    fitted_decrease_bh: float


def _motion(*, n_frames: int = 5, largest_step_ratio: float = 4.0) -> Motion:
    return Motion(n_frames, largest_step_ratio)


def _trend(fitted_decrease_bh: float) -> Trend:
    return Trend(fitted_decrease_bh)


def test_post_run_starts_immediately_after_contact() -> None:
    usable = np.array([False, True, False, True, True, False])

    run = closest_post_contact_run(usable, contact_frame=2, lookahead_frames=3)

    assert run == (3, 5, 1)


def test_post_run_chooses_earliest_run_and_excludes_contact() -> None:
    usable = np.array([False, True, True, False, True, True])

    run = closest_post_contact_run(usable, contact_frame=1, lookahead_frames=4)

    assert run == (2, 3, 1)


def test_post_run_obeys_lookahead_and_scene_mask() -> None:
    usable = np.ones(8, dtype=bool)
    same_scene = np.array([True, True, True, False, False, True, True, True])

    run = closest_post_contact_run(usable, 2, 4, same_scene)

    assert run == (5, 7, 3)
    assert closest_post_contact_run(usable, 2, 2, same_scene) is None


@pytest.mark.parametrize(
    ("motion", "trend", "gap", "expected"),
    [
        (_motion(), _trend(-0.05), 2, True),
        (_motion(n_frames=4), _trend(-0.05), 2, False),
        (_motion(largest_step_ratio=4.01), _trend(-0.05), 2, False),
        (_motion(), _trend(-0.049), 2, False),
        (_motion(), _trend(-0.05), 3, False),
    ],
)
def test_credible_outgoing_uses_fixed_inclusive_boundaries(
    motion: Motion,
    trend: Trend,
    gap: int,
    expected: bool,
) -> None:
    assert has_credible_outgoing_motion(motion, trend, gap, 2) is expected


@pytest.mark.parametrize(
    ("motion", "trend", "gap"),
    [
        (None, _trend(-0.05), 2),
        (_motion(), None, 2),
        (_motion(), _trend(-0.05), None),
        (None, None, None),
    ],
)
def test_unavailable_post_evidence_is_not_credible_outgoing(
    motion: Motion | None,
    trend: Trend | None,
    gap: int | None,
) -> None:
    assert has_credible_outgoing_motion(motion, trend, gap, 2) is False


def test_search_skips_non_credible_contacts_and_stops_at_first_credible() -> None:
    contacts = [
        AcceptedContactEvidence(10, False),
        AcceptedContactEvidence(20, True),
        AcceptedContactEvidence(30, True),
    ]
    checked_frames: list[int] = []

    def incoming_check(frame: int) -> IncomingVerdict:
        checked_frames.append(frame)
        return IncomingVerdict.NOT_INCOMING

    result = search_accepted_contacts(contacts, incoming_check)

    assert result.category is OpenerCategory.VISIBLE_SERVE
    assert result.selected_frame == 20
    assert result.selected_rank == 2
    assert result.skipped_contacts == 1
    assert checked_frames == [20]


def test_incoming_selected_contact_implies_an_unshown_serve() -> None:
    result = search_accepted_contacts(
        [AcceptedContactEvidence(20, True)],
        lambda _frame: IncomingVerdict.INCOMING,
    )

    assert result.category is OpenerCategory.FIRST_VISIBLE_POST_SERVE
    assert result.selected_frame == 20


def test_unavailable_pre_contact_result_remains_unknown() -> None:
    result = search_accepted_contacts(
        [AcceptedContactEvidence(20, True)],
        lambda _frame: IncomingVerdict.UNAVAILABLE,
    )

    assert result.category is OpenerCategory.NOT_ENOUGH_TRAJECTORY
    assert result.selected_frame == 20


def test_no_credible_contact_has_no_selected_frame() -> None:
    incoming_check_called = False

    def incoming_check(_frame: int) -> IncomingVerdict:
        nonlocal incoming_check_called
        incoming_check_called = True
        return IncomingVerdict.INCOMING

    result = search_accepted_contacts(
        [
            AcceptedContactEvidence(10, False),
            AcceptedContactEvidence(20, False),
        ],
        incoming_check,
    )

    assert result.category is OpenerCategory.NO_CREDIBLE_CONTACT
    assert result.selected_frame is None
    assert result.selected_rank is None
    assert result.skipped_contacts == 2
    assert incoming_check_called is False


def test_search_requires_chronological_contacts() -> None:
    contacts = [
        AcceptedContactEvidence(20, False),
        AcceptedContactEvidence(10, True),
    ]

    with pytest.raises(ValueError, match="strictly chronological"):
        search_accepted_contacts(contacts, lambda _frame: IncomingVerdict.NOT_INCOMING)
