"""Focused tests for the H3/R8 accepted-contact opener helpers."""

from __future__ import annotations

import json
from typing import NamedTuple

import numpy as np
import pytest

from annotator.inpaint_guard import (
    DEGRADED,
    FABRICATED,
    NO_FLAG,
    SUSPECT_FLAT,
    grade_track,
)
from scratch.serve_id_by_lookback_followup.accepted_contact_trace_variants import (
    FrozenContactEvidence,
    HighShotState,
    IncomingSearchCategory,
    IncomingStopReason,
    PreContactVerdict,
    PredecessorAdmission,
    SequentialCategory,
    classify_pre_motion,
    has_outgoing_motion,
    incoming_predecessor_search,
    path_is_eligible,
    rebuild_guard_codes,
    sequential_outgoing_search,
)
from scratch.serve_id_by_lookback_followup.analyse_accepted_contact_trace import (
    TruthRow,
)
from scratch.serve_id_by_lookback_followup.analyse_accepted_contact_trace_variants import (
    ContactEvidenceRow,
    FixtureRunStats,
    _closest_bracketing_high_shot,
    build_summary,
    derive_search_rows,
    score_search_rows,
)


class Motion(NamedTuple):
    n_frames: int
    largest_step_ratio: float


class Trend(NamedTuple):
    fitted_decrease_bh: float


def _motion(*, n_frames: int = 5, largest_step_ratio: float = 8.0) -> Motion:
    return Motion(n_frames, largest_step_ratio)


def _contact(
    frame: int,
    pre_verdict: PreContactVerdict,
    credible_outgoing: bool = False,
) -> FrozenContactEvidence:
    return FrozenContactEvidence(frame, pre_verdict, credible_outgoing)


def test_halo_three_keeps_core_attractors_and_blanks_but_clears_distant_halo() -> None:
    track = np.column_stack((np.arange(30), np.arange(30))).astype(float)
    track[3, :2] = track[12, :2]
    track[6, :2] = (0, 0)
    existing = np.zeros(30, dtype=np.uint8)
    existing[2:5] = DEGRADED
    existing[10] = FABRICATED
    existing[12] = SUSPECT_FLAT
    existing[6] = DEGRADED

    rebuilt = rebuild_guard_codes(track, existing, halo_frames=3)

    assert rebuilt[2] == NO_FLAG
    assert rebuilt[3] == DEGRADED
    assert rebuilt[6] == NO_FLAG
    assert rebuilt[10] == FABRICATED
    assert rebuilt[12] == SUSPECT_FLAT
    assert np.all(rebuilt[7:10] == DEGRADED)
    assert np.all(rebuilt[13:16] == DEGRADED)


def test_halo_fifteen_reconstructs_production_grade_codes() -> None:
    varying_pattern = np.column_stack((np.arange(16), np.arange(16) * 3 + 1)).astype(float)
    track_parts: list[np.ndarray] = []
    unique_offset = 10_000
    for episode in range(30):
        separator = np.column_stack(
            (
                np.arange(unique_offset + episode * 40, unique_offset + episode * 40 + 35),
                np.arange(unique_offset + episode * 50, unique_offset + episode * 50 + 35),
            )
        ).astype(float)
        track_parts.extend((varying_pattern.copy(), separator))
    rare_pattern = np.column_stack((np.arange(500, 516), np.arange(800, 816))).astype(float)
    track_parts.extend((rare_pattern, np.full((35, 2), 20_000.0), rare_pattern.copy()))
    track = np.concatenate(track_parts)

    production_codes, _info = grade_track(track)
    rebuilt = rebuild_guard_codes(track, production_codes, halo_frames=15)

    assert np.any(production_codes == FABRICATED)
    np.testing.assert_array_equal(rebuilt, production_codes)


@pytest.mark.parametrize(
    ("motion", "gap", "expected"),
    [
        (_motion(), 2, True),
        (_motion(largest_step_ratio=8.01), 2, False),
        (_motion(n_frames=4), 2, False),
        (_motion(), 3, False),
    ],
)
def test_path_eligibility_uses_fixed_inclusive_boundaries(motion: Motion, gap: int, expected: bool) -> None:
    assert path_is_eligible(motion, gap, maximum_contact_gap=2) is expected


@pytest.mark.parametrize(
    ("motion", "trend", "gap", "expected"),
    [
        (_motion(), Trend(0.05), 2, PreContactVerdict.INCOMING),
        (_motion(), Trend(0.049), 2, PreContactVerdict.NOT_INCOMING),
        (_motion(largest_step_ratio=8.01), Trend(0.2), 2, PreContactVerdict.UNAVAILABLE),
        (None, Trend(0.2), 2, PreContactVerdict.UNAVAILABLE),
        (_motion(), None, 2, PreContactVerdict.UNAVAILABLE),
        (_motion(), Trend(0.2), None, PreContactVerdict.UNAVAILABLE),
    ],
)
def test_pre_contact_verdict_remains_three_way(
    motion: Motion | None,
    trend: Trend | None,
    gap: int | None,
    expected: PreContactVerdict,
) -> None:
    assert classify_pre_motion(motion, trend, gap, maximum_contact_gap=2) is expected


@pytest.mark.parametrize(
    ("motion", "trend", "gap", "expected"),
    [
        (_motion(), Trend(-0.05), 2, True),
        (_motion(), Trend(-0.049), 2, False),
        (_motion(largest_step_ratio=8.01), Trend(-0.2), 2, False),
        (None, Trend(-0.2), 2, False),
        (_motion(), None, 2, False),
        (_motion(), Trend(-0.2), None, False),
    ],
)
def test_outgoing_motion_is_binary(
    motion: Motion | None,
    trend: Trend | None,
    gap: int | None,
    expected: bool,
) -> None:
    assert has_outgoing_motion(motion, trend, gap, maximum_contact_gap=2) is expected


@pytest.mark.parametrize(
    ("verdict", "category"),
    [
        (PreContactVerdict.INCOMING, SequentialCategory.FIRST_VISIBLE_POST_SERVE),
        (PreContactVerdict.NOT_INCOMING, SequentialCategory.VISIBLE_SERVE),
        (PreContactVerdict.UNAVAILABLE, SequentialCategory.NOT_ENOUGH_TRAJECTORY),
    ],
)
def test_sequential_search_classifies_first_outgoing_contact(
    verdict: PreContactVerdict,
    category: SequentialCategory,
) -> None:
    contacts = [
        _contact(10, PreContactVerdict.NOT_INCOMING),
        _contact(20, verdict, credible_outgoing=True),
        _contact(30, PreContactVerdict.INCOMING, credible_outgoing=True),
    ]

    result = sequential_outgoing_search(contacts)

    assert result.category is category
    assert result.selected_frame == 20
    assert result.selected_rank == 2
    assert result.skipped_contacts == 1


def test_sequential_search_reports_no_credible_contact() -> None:
    result = sequential_outgoing_search([_contact(10, PreContactVerdict.INCOMING)])

    assert result.category is SequentialCategory.NO_CREDIBLE_CONTACT
    assert result.selected_frame is None
    assert result.skipped_contacts == 1


def test_earliest_incoming_anchor_ignores_outgoing_evidence() -> None:
    contacts = [
        _contact(100, PreContactVerdict.NOT_INCOMING, credible_outgoing=True),
        _contact(130, PreContactVerdict.INCOMING, credible_outgoing=False),
        _contact(150, PreContactVerdict.INCOMING, credible_outgoing=True),
    ]

    result = incoming_predecessor_search(
        contacts,
        ordinary_max_gap_frames=60,
        high_shot_state=None,
        high_shot_endpoint_buffer_frames=12,
    )

    assert result.category is IncomingSearchCategory.VISIBLE_SERVE
    assert result.anchor_frame == 130
    assert result.anchor_rank == 2
    assert result.predecessor_frame == 100
    assert result.admission is PredecessorAdmission.ORDINARY


@pytest.mark.parametrize(("gap", "admission"), [(60, PredecessorAdmission.ORDINARY), (61, PredecessorAdmission.NONE)])
def test_ordinary_predecessor_cap_is_inclusive(gap: int, admission: PredecessorAdmission) -> None:
    result = incoming_predecessor_search(
        [
            _contact(100, PreContactVerdict.NOT_INCOMING),
            _contact(100 + gap, PreContactVerdict.INCOMING),
        ],
        ordinary_max_gap_frames=60,
        high_shot_state=None,
        high_shot_endpoint_buffer_frames=12,
    )

    assert result.admission is admission
    expected_category = (
        IncomingSearchCategory.VISIBLE_SERVE
        if admission is PredecessorAdmission.ORDINARY
        else IncomingSearchCategory.FIRST_VISIBLE_POST_SERVE
    )
    assert result.category is expected_category


@pytest.mark.parametrize(
    ("state", "admission"),
    [
        (HighShotState(112, 188), PredecessorAdmission.HIGH_SHOT),
        (HighShotState(113, 188), PredecessorAdmission.NONE),
        (HighShotState(112, 187), PredecessorAdmission.NONE),
    ],
)
def test_high_shot_endpoint_buffer_is_inclusive(state: HighShotState, admission: PredecessorAdmission) -> None:
    result = incoming_predecessor_search(
        [
            _contact(100, PreContactVerdict.NOT_INCOMING),
            _contact(200, PreContactVerdict.INCOMING),
        ],
        ordinary_max_gap_frames=60,
        high_shot_state=state,
        high_shot_endpoint_buffer_frames=12,
    )

    assert result.admission is admission


def test_high_shot_only_admits_predecessor_for_evidence_inspection() -> None:
    result = incoming_predecessor_search(
        [
            _contact(100, PreContactVerdict.UNAVAILABLE),
            _contact(200, PreContactVerdict.INCOMING),
        ],
        ordinary_max_gap_frames=60,
        high_shot_state=HighShotState(112, 188),
        high_shot_endpoint_buffer_frames=12,
    )

    assert result.category is IncomingSearchCategory.PREDECESSOR_EVIDENCE_UNAVAILABLE
    assert result.stop_reason is IncomingStopReason.PREDECESSOR_ADMITTED_HIGH_SHOT
    assert result.predecessor_frame == 100


@pytest.mark.parametrize(
    ("contacts", "category", "reason"),
    [
        ([], IncomingSearchCategory.NO_ACCEPTED_CONTACT, IncomingStopReason.NO_ACCEPTED_CONTACT),
        (
            [_contact(10, PreContactVerdict.NOT_INCOMING)],
            IncomingSearchCategory.NO_MEASURED_INCOMING,
            IncomingStopReason.NO_MEASURED_INCOMING,
        ),
        (
            [_contact(10, PreContactVerdict.UNAVAILABLE)],
            IncomingSearchCategory.NO_INCOMING_WITH_UNAVAILABLE,
            IncomingStopReason.NO_INCOMING_WITH_UNAVAILABLE,
        ),
    ],
)
def test_no_anchor_terminal_states_remain_distinct(
    contacts: list[FrozenContactEvidence],
    category: IncomingSearchCategory,
    reason: IncomingStopReason,
) -> None:
    result = incoming_predecessor_search(
        contacts,
        ordinary_max_gap_frames=60,
        high_shot_state=None,
        high_shot_endpoint_buffer_frames=12,
    )

    assert result.category is category
    assert result.stop_reason is reason
    assert result.anchor_frame is None


def test_first_incoming_contact_implies_unshown_serve_without_inventing_frame() -> None:
    result = incoming_predecessor_search(
        [_contact(100, PreContactVerdict.INCOMING)],
        ordinary_max_gap_frames=60,
        high_shot_state=None,
        high_shot_endpoint_buffer_frames=12,
    )

    assert result.category is IncomingSearchCategory.FIRST_VISIBLE_POST_SERVE
    assert result.stop_reason is IncomingStopReason.NO_PREDECESSOR
    assert result.anchor_frame == 100
    assert result.predecessor_frame is None


def test_searches_require_chronological_contacts() -> None:
    contacts = [
        _contact(20, PreContactVerdict.NOT_INCOMING),
        _contact(10, PreContactVerdict.INCOMING),
    ]

    with pytest.raises(ValueError, match="strictly chronological"):
        sequential_outgoing_search(contacts)
    with pytest.raises(ValueError, match="strictly chronological"):
        incoming_predecessor_search(
            contacts,
            ordinary_max_gap_frames=60,
            high_shot_state=None,
            high_shot_endpoint_buffer_frames=12,
        )


def _contact_row(
    *,
    accepted_rank: int,
    frame: int,
    verdict: PreContactVerdict,
    outgoing: bool,
) -> ContactEvidenceRow:
    return ContactEvidenceRow(
        fixture="fixture",
        video_id=1,
        set_id="set1",
        rally=1,
        fps=30.0,
        span_id=2,
        accepted_rank=accepted_rank,
        contact_frame=frame,
        player="Top",
        pre_run_start=None,
        pre_run_end=None,
        pre_contact_gap=None,
        pre_n_frames=None,
        pre_largest_step_ratio=None,
        pre_fitted_decrease_bh=None,
        pre_path_status="no_usable_run",
        pre_verdict=verdict.value,
        post_run_start=None,
        post_run_end=None,
        post_contact_gap=None,
        post_n_frames=None,
        post_largest_step_ratio=None,
        post_fitted_decrease_bh=None,
        post_path_status="no_usable_run",
        credible_outgoing=outgoing,
        preceding_high_shot_start=None,
        preceding_high_shot_end=None,
        preceding_high_shot_left_gap=None,
        preceding_high_shot_right_gap=None,
    )


def test_driver_derives_both_searches_only_from_frozen_contact_rows() -> None:
    rows = [
        _contact_row(
            accepted_rank=1,
            frame=100,
            verdict=PreContactVerdict.UNAVAILABLE,
            outgoing=False,
        ),
        _contact_row(
            accepted_rank=2,
            frame=130,
            verdict=PreContactVerdict.INCOMING,
            outgoing=True,
        ),
    ]

    result = derive_search_rows(rows)[0]

    assert result.sequential_category == SequentialCategory.FIRST_VISIBLE_POST_SERVE.value
    assert result.sequential_selected_frame == 130
    assert result.incoming_category == IncomingSearchCategory.PREDECESSOR_EVIDENCE_UNAVAILABLE.value
    assert result.incoming_anchor_frame == 130
    assert result.incoming_predecessor_frame == 100


def test_driver_scores_visible_serve_and_first_post_serve_against_distinct_gt_contacts() -> None:
    rows = [
        _contact_row(
            accepted_rank=1,
            frame=100,
            verdict=PreContactVerdict.NOT_INCOMING,
            outgoing=False,
        ),
        _contact_row(
            accepted_rank=2,
            frame=130,
            verdict=PreContactVerdict.INCOMING,
            outgoing=True,
        ),
    ]
    result = derive_search_rows(rows)[0]

    scored = score_search_rows(
        [result],
        {result.key: TruthRow((100, 130))},
    )[0]

    assert scored["tolerance_10_sequential_selected_label"] == "contact_2"
    assert scored["tolerance_10_sequential_final_correct"] is True
    assert scored["tolerance_10_incoming_selected_label"] == "contact_1"
    assert scored["tolerance_10_incoming_final_correct"] is True


def test_closest_high_shot_state_minimises_the_worst_endpoint_distance() -> None:
    selected = _closest_bracketing_high_shot(
        100,
        200,
        [
            HighShotState(105, 170),
            HighShotState(112, 188),
            HighShotState(120, 195),
        ],
    )

    assert selected == HighShotState(112, 188)


def test_summary_survives_a_json_round_trip() -> None:
    stats = FixtureRunStats(
        fixture="fixture",
        production_guard_counts={0: 10, 1: 2},
        h3_guard_counts={0: 11, 1: 1},
        changed_guard_frames=1,
        halo15_exact_match=True,
        high_shot_state_count=3,
    )
    evidence = [
        _contact_row(
            accepted_rank=1,
            frame=100,
            verdict=PreContactVerdict.INCOMING,
            outgoing=True,
        )
    ]
    result = derive_search_rows(evidence)[0]
    scored = score_search_rows([result], {result.key: TruthRow((90, 100))})

    summary = build_summary(evidence, scored, [stats])

    assert json.loads(json.dumps(summary)) == summary
