"""Focused tests for the sequential accepted-contact opener search."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

from scratch.serve_id_by_lookback_followup.accepted_contact_trace import (
    AcceptedContactEvidence,
    IncomingVerdict,
    OpenerCategory,
    closest_post_contact_run,
    has_credible_outgoing_motion,
    search_accepted_contacts,
)
from scratch.serve_id_by_lookback_followup.analyse_accepted_contact_trace import (
    SearchRow,
    TruthRow,
    build_summary,
    read_csv_gz,
    read_json_gz,
    score_search_rows,
    write_csv_gz,
    write_json_gz,
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


def _search_row(
    *,
    rally: int,
    accepted: tuple[int, ...],
    selected_frame: int | None,
    selected_rank: int | None,
    skipped_contacts: int,
    pre_verdict: str | None,
    category: OpenerCategory,
) -> SearchRow:
    return SearchRow(
        fixture="fixture",
        video_id=1,
        set_id="set1",
        rally=rally,
        fps=30.0,
        span_id=rally,
        accepted_contact_frames=accepted,
        credible_outgoing=tuple(frame == selected_frame for frame in accepted),
        selected_frame=selected_frame,
        selected_rank=selected_rank,
        skipped_contacts=skipped_contacts,
        selected_player="Top" if selected_frame is not None else None,
        pre_contact_verdict=pre_verdict,
        opener_category=category.value,
    )


def test_gt_join_scores_visible_implied_unknown_and_exhausted_results() -> None:
    search_rows = [
        _search_row(
            rally=1,
            accepted=(80, 100),
            selected_frame=100,
            selected_rank=2,
            skipped_contacts=1,
            pre_verdict=IncomingVerdict.NOT_INCOMING.value,
            category=OpenerCategory.VISIBLE_SERVE,
        ),
        _search_row(
            rally=2,
            accepted=(100, 130),
            selected_frame=130,
            selected_rank=2,
            skipped_contacts=1,
            pre_verdict=IncomingVerdict.INCOMING.value,
            category=OpenerCategory.FIRST_VISIBLE_POST_SERVE,
        ),
        _search_row(
            rally=3,
            accepted=(100,),
            selected_frame=100,
            selected_rank=1,
            skipped_contacts=0,
            pre_verdict=IncomingVerdict.UNAVAILABLE.value,
            category=OpenerCategory.NOT_ENOUGH_TRAJECTORY,
        ),
        _search_row(
            rally=4,
            accepted=(80,),
            selected_frame=None,
            selected_rank=None,
            skipped_contacts=1,
            pre_verdict=None,
            category=OpenerCategory.NO_CREDIBLE_CONTACT,
        ),
    ]
    truth_by_key = {
        row.key: TruthRow((100, 130))
        for row in search_rows
    }

    scored = score_search_rows(search_rows, truth_by_key)

    assert [row["tolerance_10_transition"] for row in scored] == [
        "fixed",
        "unchanged_correct",
        "pre_contact_unknown",
        "no_credible_contact",
    ]
    assert scored[0]["tolerance_10_selected_label"] == "contact_1"
    assert scored[1]["tolerance_10_selected_label"] == "contact_2"


def test_summary_keeps_the_primary_unmatched_slice() -> None:
    search_row = _search_row(
        rally=1,
        accepted=(80, 100),
        selected_frame=100,
        selected_rank=2,
        skipped_contacts=1,
        pre_verdict=IncomingVerdict.NOT_INCOMING.value,
        category=OpenerCategory.VISIBLE_SERVE,
    )
    scored = score_search_rows([search_row], {search_row.key: TruthRow((100, 130))})

    summary = build_summary(scored)

    tolerances = summary["tolerances"]
    assert isinstance(tolerances, dict)
    tolerance_10 = tolerances["10"]
    assert isinstance(tolerance_10, dict)
    unmatched_slice = tolerance_10["baseline_unmatched_slice"]
    assert unmatched_slice == {"population": 1, "transitions": {"fixed": 1}}


def test_compressed_evidence_round_trip(tmp_path: Path) -> None:
    rows_path = tmp_path / "rows.csv.gz"
    summary_path = tmp_path / "summary.json.gz"
    rows = [{"key": "a", "count": 1}, {"key": "b", "count": 2}]
    summary = {"population": 2, "counts": {"a": 1, "b": 1}}

    write_csv_gz(rows_path, rows)
    write_json_gz(summary_path, summary)

    assert read_csv_gz(rows_path) == [
        {"key": "a", "count": "1"},
        {"key": "b", "count": "2"},
    ]
    assert read_json_gz(summary_path) == summary
