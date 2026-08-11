"""Run the fixed halo-3, ratio-8 accepted-contact opener experiments."""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import io
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import TypeAlias

import numpy as np

from annotator.calibration.fixtures import FIXTURES
from annotator.calibration.gt_scoring import load_gt_tables
from annotator.config import BaseAnnotatorConfig
from annotator.inpaint_guard import code_counts
from annotator.rally.spans import _gap_is_high_shot_oob, _gap_passes_reentry_guard
from annotator.resolve import resolve
from annotator.types import true_runs
from scratch.serve_id_by_lookback_followup.accepted_contact_trace import (
    closest_post_contact_run,
)
from scratch.serve_id_by_lookback_followup.accepted_contact_trace_variants import (
    MAX_LARGEST_STEP_RATIO,
    MIN_PATH_FRAMES,
    FrozenContactEvidence,
    HighShotState,
    IncomingSearchCategory,
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
    ContactContext,
    SearchInputs,
    SearchKey,
    TruthRow,
    _contact_context,
    _measure_run,
    _primary_crosswalk,
    _scaled_frames,
)
from scratch.serve_start_trajectory_exploration.experiment_data import load_video_data
from scratch.serve_start_trajectory_exploration.trajectory_features import (
    IncomingMotion,
    RobustDistanceTrend,
    align_anchor_to_gt,
    closest_pre_contact_run,
)

RUN_DIR = Path(__file__).resolve().parent
CONTACT_EVIDENCE_PATH = RUN_DIR / "h3_r8_contact_evidence.csv.gz"
SEARCH_RESULTS_PATH = RUN_DIR / "h3_r8_search_results.csv.gz"
SUMMARY_PATH = RUN_DIR / "h3_r8_summary.json.gz"

HALO_SOURCE_FRAMES = 3
PRODUCTION_HALO_SOURCE_FRAMES = 15
LOOKAROUND_BASE30_FRAMES = 30
MAX_LOCAL_GAP_BASE30_FRAMES = 2
PREDECESSOR_MAX_GAP_BASE30_FRAMES = 60
HIGH_SHOT_ENDPOINT_BUFFER_BASE30_FRAMES = 12
CONTACT_TOLERANCES_BASE30 = (5, 10, 30)
EXPECTED_PRIMARY_BY_FIXTURE = {"sset_01": 104, "sset_15": 84, "sset_21": 51}
EXPECTED_CONTACT_EVIDENCE_ROWS = 3_200
MIN_DIRECTIONAL_CHANGE_BH = 0.05

PathStatus: TypeAlias = str


@dataclass(frozen=True, slots=True)
class PathEvidence:
    """One local shuttle run and its fixed eligibility measurements."""

    run_start: int | None
    run_end: int | None
    contact_gap: int | None
    n_frames: int | None
    largest_step_ratio: float | None
    fitted_decrease_bh: float | None
    status: PathStatus


@dataclass(frozen=True, slots=True)
class ContactEvidenceRow:
    """GT-free trajectory evidence for one chronological accepted contact."""

    fixture: str
    video_id: int
    set_id: str
    rally: int
    fps: float
    span_id: int
    accepted_rank: int
    contact_frame: int
    player: str | None
    pre_run_start: int | None
    pre_run_end: int | None
    pre_contact_gap: int | None
    pre_n_frames: int | None
    pre_largest_step_ratio: float | None
    pre_fitted_decrease_bh: float | None
    pre_path_status: str
    pre_verdict: str
    post_run_start: int | None
    post_run_end: int | None
    post_contact_gap: int | None
    post_n_frames: int | None
    post_largest_step_ratio: float | None
    post_fitted_decrease_bh: float | None
    post_path_status: str
    credible_outgoing: bool
    preceding_high_shot_start: int | None
    preceding_high_shot_end: int | None
    preceding_high_shot_left_gap: int | None
    preceding_high_shot_right_gap: int | None

    @property
    def key(self) -> SearchKey:
        """Return the fixed rally identity."""
        return self.fixture, self.video_id, self.set_id, self.rally


@dataclass(frozen=True, slots=True)
class DualSearchRow:
    """Both GT-free opener results for one rally."""

    fixture: str
    video_id: int
    set_id: str
    rally: int
    fps: float
    span_id: int
    accepted_count: int
    accepted_contact_frames: tuple[int, ...]
    sequential_category: str
    sequential_selected_frame: int | None
    sequential_selected_rank: int | None
    sequential_selected_player: str | None
    sequential_selected_pre_verdict: str | None
    sequential_skipped_contacts: int
    incoming_category: str
    incoming_anchor_frame: int | None
    incoming_anchor_rank: int | None
    incoming_anchor_player: str | None
    incoming_predecessor_frame: int | None
    incoming_predecessor_rank: int | None
    incoming_predecessor_player: str | None
    incoming_predecessor_verdict: str | None
    incoming_predecessor_gap: int | None
    incoming_admission: str | None
    incoming_stop_reason: str
    incoming_high_shot_start: int | None
    incoming_high_shot_end: int | None

    @property
    def key(self) -> SearchKey:
        """Return the fixed rally identity."""
        return self.fixture, self.video_id, self.set_id, self.rally


@dataclass(frozen=True, slots=True)
class FixtureRunStats:
    """Guard and high-shot checks recorded for one fixture."""

    fixture: str
    production_guard_counts: dict[int, int]
    h3_guard_counts: dict[int, int]
    changed_guard_frames: int
    halo15_exact_match: bool
    high_shot_state_count: int


def measured_high_shot_states(track: np.ndarray, fps: float) -> tuple[HighShotState, ...]:
    """Rebuild the release configuration's measured high-shot gap states."""
    resolved = resolve(BaseAnnotatorConfig(), fps)
    demotion_bound = resolved.gap_state_demotion_bound
    if demotion_bound is None:
        raise ValueError("the fixed release configuration must have a high-shot demotion bound")
    variant = resolved.reentry_guard_variant
    buffer = resolved.reentry_guard_buffer
    if variant is None or buffer is None:
        raise ValueError("the fixed release configuration must have the two-sided re-entry guard")

    states: list[HighShotState] = []
    for gap_start, gap_end in true_runs(track[:, 2] != 1):
        if not _gap_is_high_shot_oob(track, gap_start, resolved.constants):
            continue
        if not _gap_passes_reentry_guard(
            track,
            gap_start,
            gap_end,
            variant,
            buffer,
            resolved.constants,
        ):
            continue
        state_end = min(gap_start + demotion_bound, gap_end)
        if state_end > gap_start:
            states.append(HighShotState(gap_start, state_end))
    return tuple(states)


def _closest_bracketing_high_shot(
    previous_frame: int,
    current_frame: int,
    states: Sequence[HighShotState],
) -> HighShotState | None:
    """Return the bracketing state with the closest worst endpoint distance."""
    candidates = [
        state
        for state in states
        if previous_frame <= state.start < state.end <= current_frame
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda state: (
            max(state.start - previous_frame, current_frame - state.end),
            state.start - previous_frame + current_frame - state.end,
            state.start,
            state.end,
        ),
    )


def _path_status(
    motion: IncomingMotion,
    contact_gap: int,
    maximum_contact_gap: int,
) -> str:
    """Name the first failed fixed eligibility condition."""
    if motion.n_frames < MIN_PATH_FRAMES:
        return "too_few_frames"
    if contact_gap > maximum_contact_gap:
        return "contact_gap_too_large"
    if motion.largest_step_ratio > MAX_LARGEST_STEP_RATIO:
        return "largest_step_ratio_too_large"
    return "eligible"


def _measure_local_path(
    data: SearchInputs,
    context: ContactContext | None,
    contact_frame: int,
    lookaround_frames: int,
    maximum_contact_gap: int,
    *,
    before_contact: bool,
) -> tuple[PathEvidence, IncomingMotion | None, RobustDistanceTrend | None]:
    """Measure the closest strict local run on one side of a contact."""
    if context is None:
        return PathEvidence(None, None, None, None, None, None, "contact_context_unavailable"), None, None

    if before_contact:
        run = closest_pre_contact_run(context.usable, contact_frame, lookaround_frames)
    else:
        run = closest_post_contact_run(context.usable, contact_frame, lookaround_frames)
    if run is None:
        return PathEvidence(None, None, None, None, None, None, "no_usable_run"), None, None

    gap = run.frames_to_contact if before_contact else run.frames_from_contact
    measurements = _measure_run(data, context, run.start, run.end)
    if measurements is None:
        return PathEvidence(run.start, run.end, gap, None, None, None, "measurement_unavailable"), None, None
    motion, trend = measurements
    status = _path_status(motion, gap, maximum_contact_gap)
    if path_is_eligible(motion, gap, maximum_contact_gap) != (status == "eligible"):
        raise ValueError("path status differs from the fixed eligibility helper")
    return (
        PathEvidence(
            run.start,
            run.end,
            gap,
            motion.n_frames,
            motion.largest_step_ratio,
            trend.fitted_decrease_bh,
            status,
        ),
        motion,
        trend,
    )


def build_span_contact_evidence(
    data: SearchInputs,
    span_id: int,
    key: SearchKey,
    high_shot_states: Sequence[HighShotState],
) -> list[ContactEvidenceRow]:
    """Measure every accepted contact in one fixed primary span."""
    accepted = tuple(sorted(data.accepted_by_span.get(span_id, [])))
    if len(accepted) != len(set(accepted)):
        raise ValueError(f"{data.fixture.name} span {span_id}: accepted contacts must be unique")
    span_start, span_end = data.spans[span_id]
    if any(not span_start <= frame < span_end for frame in accepted):
        raise ValueError(f"{data.fixture.name} span {span_id}: accepted contact is outside its span")

    lookaround_frames = _scaled_frames(LOOKAROUND_BASE30_FRAMES, data.fixture.fps)
    maximum_contact_gap = _scaled_frames(MAX_LOCAL_GAP_BASE30_FRAMES, data.fixture.fps)
    fixture, video_id, set_id, rally = key
    rows: list[ContactEvidenceRow] = []
    previous_frame: int | None = None
    for accepted_rank, contact_frame in enumerate(accepted, start=1):
        context = _contact_context(data, span_id, contact_frame)
        pre, pre_motion, pre_trend = _measure_local_path(
            data,
            context,
            contact_frame,
            lookaround_frames,
            maximum_contact_gap,
            before_contact=True,
        )
        post, post_motion, post_trend = _measure_local_path(
            data,
            context,
            contact_frame,
            lookaround_frames,
            maximum_contact_gap,
            before_contact=False,
        )
        pre_verdict = classify_pre_motion(
            pre_motion,
            pre_trend,
            pre.contact_gap,
            maximum_contact_gap,
        )
        credible_outgoing = has_outgoing_motion(
            post_motion,
            post_trend,
            post.contact_gap,
            maximum_contact_gap,
        )

        high_shot = (
            _closest_bracketing_high_shot(previous_frame, contact_frame, high_shot_states)
            if previous_frame is not None
            else None
        )
        left_gap = high_shot.start - previous_frame if high_shot is not None and previous_frame is not None else None
        right_gap = contact_frame - high_shot.end if high_shot is not None else None
        rows.append(
            ContactEvidenceRow(
                fixture=fixture,
                video_id=video_id,
                set_id=set_id,
                rally=rally,
                fps=data.fixture.fps,
                span_id=span_id,
                accepted_rank=accepted_rank,
                contact_frame=contact_frame,
                player=context.player.value if context is not None else None,
                pre_run_start=pre.run_start,
                pre_run_end=pre.run_end,
                pre_contact_gap=pre.contact_gap,
                pre_n_frames=pre.n_frames,
                pre_largest_step_ratio=pre.largest_step_ratio,
                pre_fitted_decrease_bh=pre.fitted_decrease_bh,
                pre_path_status=pre.status,
                pre_verdict=pre_verdict.value,
                post_run_start=post.run_start,
                post_run_end=post.run_end,
                post_contact_gap=post.contact_gap,
                post_n_frames=post.n_frames,
                post_largest_step_ratio=post.largest_step_ratio,
                post_fitted_decrease_bh=post.fitted_decrease_bh,
                post_path_status=post.status,
                credible_outgoing=credible_outgoing,
                preceding_high_shot_start=high_shot.start if high_shot is not None else None,
                preceding_high_shot_end=high_shot.end if high_shot is not None else None,
                preceding_high_shot_left_gap=left_gap,
                preceding_high_shot_right_gap=right_gap,
            )
        )
        previous_frame = contact_frame
    return rows


def build_contact_evidence() -> tuple[
    list[ContactEvidenceRow],
    dict[SearchKey, TruthRow],
    list[FixtureRunStats],
]:
    """Build all GT-free contact rows and the separate scoring table."""
    shared_gt_tables = load_gt_tables()
    evidence_rows: list[ContactEvidenceRow] = []
    truth_by_key: dict[SearchKey, TruthRow] = {}
    fixture_stats: list[FixtureRunStats] = []
    primary_counts: Counter[str] = Counter()
    for fixture in FIXTURES:
        print(f"{fixture.name}: loading fixed inputs")
        data = load_video_data(fixture, shared_gt_tables)
        reconstructed_h15 = rebuild_guard_codes(
            data.track,
            data.guard_codes,
            PRODUCTION_HALO_SOURCE_FRAMES,
        )
        halo15_exact_match = bool(np.array_equal(reconstructed_h15, data.guard_codes))
        if not halo15_exact_match:
            raise ValueError(f"{fixture.name}: halo-15 reconstruction differs from production")
        h3_codes = rebuild_guard_codes(data.track, data.guard_codes, HALO_SOURCE_FRAMES)
        search_inputs = replace(SearchInputs.from_video_data(data), guard_codes=h3_codes)
        high_shot_states = measured_high_shot_states(data.track, fixture.fps)

        for rally, span_id in _primary_crosswalk(data):
            key = (fixture.name, fixture.video_id, rally.set_id, rally.rally)
            evidence_rows.extend(
                build_span_contact_evidence(
                    search_inputs,
                    span_id,
                    key,
                    high_shot_states,
                )
            )
            truth_by_key[key] = TruthRow(tuple(int(frame) for frame in rally.stroke_frames))
            primary_counts[fixture.name] += 1

        fixture_stats.append(
            FixtureRunStats(
                fixture=fixture.name,
                production_guard_counts=code_counts(data.guard_codes),
                h3_guard_counts=code_counts(h3_codes),
                changed_guard_frames=int(np.count_nonzero(h3_codes != data.guard_codes)),
                halo15_exact_match=halo15_exact_match,
                high_shot_state_count=len(high_shot_states),
            )
        )
        print(
            f"{fixture.name}: froze {primary_counts[fixture.name]} rallies and "
            f"{sum(row.fixture == fixture.name for row in evidence_rows)} contacts"
        )
        del data, search_inputs, h3_codes, reconstructed_h15
        gc.collect()

    if primary_counts != Counter(EXPECTED_PRIMARY_BY_FIXTURE) or len(truth_by_key) != 239:
        raise ValueError("primary population differs from the fixed 239-rally contract")
    if len(evidence_rows) != EXPECTED_CONTACT_EVIDENCE_ROWS:
        raise ValueError(
            f"accepted-contact population changed: expected {EXPECTED_CONTACT_EVIDENCE_ROWS}, "
            f"got {len(evidence_rows)}"
        )
    _validate_contact_evidence(evidence_rows, truth_by_key)
    return evidence_rows, truth_by_key, fixture_stats


def _validate_contact_evidence(
    rows: Sequence[ContactEvidenceRow],
    truth_by_key: dict[SearchKey, TruthRow] | None = None,
) -> None:
    """Check stable keys, chronological contacts, and contiguous ranks."""
    grouped: dict[SearchKey, list[ContactEvidenceRow]] = defaultdict(list)
    for row in rows:
        grouped[row.key].append(row)
    if truth_by_key is not None and set(grouped) != set(truth_by_key):
        raise ValueError("contact evidence and truth tables must have the same keys")
    for key, contacts in grouped.items():
        ranks = [row.accepted_rank for row in contacts]
        frames = [row.contact_frame for row in contacts]
        if ranks != list(range(1, len(contacts) + 1)):
            raise ValueError(f"{key}: accepted ranks are not contiguous")
        if frames != sorted(set(frames)):
            raise ValueError(f"{key}: accepted frames are not strictly chronological")


def derive_search_rows(rows: Sequence[ContactEvidenceRow]) -> list[DualSearchRow]:
    """Derive both searches only from frozen per-contact evidence."""
    _validate_contact_evidence(rows)
    grouped: dict[SearchKey, list[ContactEvidenceRow]] = defaultdict(list)
    for row in rows:
        grouped[row.key].append(row)

    results: list[DualSearchRow] = []
    for key, contacts in grouped.items():
        frozen = tuple(
            FrozenContactEvidence(
                frame=row.contact_frame,
                pre_verdict=PreContactVerdict(row.pre_verdict),
                credible_outgoing=row.credible_outgoing,
            )
            for row in contacts
        )
        sequential = sequential_outgoing_search(frozen)

        incoming_anchor = next(
            (row for row in contacts if row.pre_verdict == PreContactVerdict.INCOMING.value),
            None,
        )
        high_shot_state = None
        if (
            incoming_anchor is not None
            and incoming_anchor.preceding_high_shot_start is not None
            and incoming_anchor.preceding_high_shot_end is not None
        ):
            high_shot_state = HighShotState(
                incoming_anchor.preceding_high_shot_start,
                incoming_anchor.preceding_high_shot_end,
            )
        ordinary_cap = _scaled_frames(PREDECESSOR_MAX_GAP_BASE30_FRAMES, contacts[0].fps)
        endpoint_buffer = _scaled_frames(HIGH_SHOT_ENDPOINT_BUFFER_BASE30_FRAMES, contacts[0].fps)
        incoming = incoming_predecessor_search(
            frozen,
            ordinary_max_gap_frames=ordinary_cap,
            high_shot_state=high_shot_state,
            high_shot_endpoint_buffer_frames=endpoint_buffer,
        )

        by_rank = {row.accepted_rank: row for row in contacts}
        sequential_row = by_rank.get(sequential.selected_rank)
        anchor_row = by_rank.get(incoming.anchor_rank)
        predecessor_row = by_rank.get(incoming.predecessor_rank)
        fixture, video_id, set_id, rally = key
        results.append(
            DualSearchRow(
                fixture=fixture,
                video_id=video_id,
                set_id=set_id,
                rally=rally,
                fps=contacts[0].fps,
                span_id=contacts[0].span_id,
                accepted_count=len(contacts),
                accepted_contact_frames=tuple(row.contact_frame for row in contacts),
                sequential_category=sequential.category.value,
                sequential_selected_frame=sequential.selected_frame,
                sequential_selected_rank=sequential.selected_rank,
                sequential_selected_player=sequential_row.player if sequential_row is not None else None,
                sequential_selected_pre_verdict=(
                    sequential_row.pre_verdict if sequential_row is not None else None
                ),
                sequential_skipped_contacts=sequential.skipped_contacts,
                incoming_category=incoming.category.value,
                incoming_anchor_frame=incoming.anchor_frame,
                incoming_anchor_rank=incoming.anchor_rank,
                incoming_anchor_player=anchor_row.player if anchor_row is not None else None,
                incoming_predecessor_frame=incoming.predecessor_frame,
                incoming_predecessor_rank=incoming.predecessor_rank,
                incoming_predecessor_player=(
                    predecessor_row.player if predecessor_row is not None else None
                ),
                incoming_predecessor_verdict=(
                    predecessor_row.pre_verdict if predecessor_row is not None else None
                ),
                incoming_predecessor_gap=incoming.contact_gap,
                incoming_admission=(
                    incoming.admission.value
                ),
                incoming_stop_reason=incoming.stop_reason.value,
                incoming_high_shot_start=(
                    high_shot_state.start
                    if incoming.admission is PredecessorAdmission.HIGH_SHOT
                    and high_shot_state is not None
                    else None
                ),
                incoming_high_shot_end=(
                    high_shot_state.end
                    if incoming.admission is PredecessorAdmission.HIGH_SHOT
                    and high_shot_state is not None
                    else None
                ),
            )
        )
    return results


def _search_dict(row: DualSearchRow) -> dict[str, object]:
    """Convert one GT-free result to its stable CSV representation."""
    values = asdict(row)
    values["accepted_contact_frames"] = json.dumps(
        row.accepted_contact_frames,
        separators=(",", ":"),
    )
    return values


def _expected_frame_and_label(
    row: DualSearchRow,
    search: str,
) -> tuple[int | None, str | None]:
    """Return the frame and GT ordinal expected by one terminal category."""
    if search == "sequential":
        if row.sequential_category == SequentialCategory.VISIBLE_SERVE.value:
            return row.sequential_selected_frame, "contact_1"
        if row.sequential_category == SequentialCategory.FIRST_VISIBLE_POST_SERVE.value:
            return row.sequential_selected_frame, "contact_2"
        return None, None
    if search == "incoming":
        if row.incoming_category == IncomingSearchCategory.VISIBLE_SERVE.value:
            return row.incoming_predecessor_frame, "contact_1"
        if row.incoming_category == IncomingSearchCategory.FIRST_VISIBLE_POST_SERVE.value:
            return row.incoming_anchor_frame, "contact_2"
        return None, None
    raise ValueError(f"unknown search {search!r}")


def _terminal_transition(
    row: DualSearchRow,
    search: str,
    baseline_correct: bool,
    final_correct: bool,
) -> str:
    """Name correctness transitions while retaining terminal unknown causes."""
    if search == "sequential":
        if row.sequential_category == SequentialCategory.NOT_ENOUGH_TRAJECTORY.value:
            return "pre_contact_unavailable"
        if row.sequential_category == SequentialCategory.NO_CREDIBLE_CONTACT.value:
            return "no_credible_contact"
    else:
        terminal_names = {
            IncomingSearchCategory.PREDECESSOR_EVIDENCE_UNAVAILABLE.value,
            IncomingSearchCategory.NO_MEASURED_INCOMING.value,
            IncomingSearchCategory.NO_INCOMING_WITH_UNAVAILABLE.value,
            IncomingSearchCategory.NO_ACCEPTED_CONTACT.value,
        }
        if row.incoming_category in terminal_names:
            return row.incoming_category
    if final_correct and not baseline_correct:
        return "fixed"
    if baseline_correct and not final_correct:
        return "damaged"
    return "unchanged_correct" if final_correct else "unchanged_wrong"


def score_search_rows(
    search_rows: Sequence[DualSearchRow],
    truth_by_key: dict[SearchKey, TruthRow],
) -> list[dict[str, object]]:
    """Append GT scoring only after both search results are frozen."""
    scored: list[dict[str, object]] = []
    for search_row in search_rows:
        row = _search_dict(search_row)
        truth = truth_by_key[search_row.key]
        row["gt_stroke_frames"] = json.dumps(truth.stroke_frames, separators=(",", ":"))
        baseline_frame = search_row.accepted_contact_frames[0]
        for tolerance in CONTACT_TOLERANCES_BASE30:
            baseline = align_anchor_to_gt(
                baseline_frame,
                truth.stroke_frames,
                search_row.fps,
                tolerance,
            )
            prefix = f"tolerance_{tolerance}"
            row[f"{prefix}_baseline_label"] = baseline.label
            row[f"{prefix}_baseline_multiple"] = baseline.multiple_within_tolerance
            baseline_correct = baseline.label == "contact_1"

            for search in ("sequential", "incoming"):
                selected_frame, expected_label = _expected_frame_and_label(search_row, search)
                selected_label = "no_anchor"
                nearest_ordinal: int | None = None
                signed_offset: float | None = None
                multiple = False
                if selected_frame is not None:
                    alignment = align_anchor_to_gt(
                        selected_frame,
                        truth.stroke_frames,
                        search_row.fps,
                        tolerance,
                    )
                    selected_label = alignment.label
                    nearest_ordinal = alignment.nearest_gt_ordinal
                    signed_offset = alignment.signed_offset_base30
                    multiple = alignment.multiple_within_tolerance
                final_correct = expected_label is not None and selected_label == expected_label
                row[f"{prefix}_{search}_selected_label"] = selected_label
                row[f"{prefix}_{search}_selected_nearest_gt_ordinal"] = nearest_ordinal
                row[f"{prefix}_{search}_selected_signed_offset_base30"] = signed_offset
                row[f"{prefix}_{search}_selected_multiple"] = multiple
                row[f"{prefix}_{search}_final_correct"] = final_correct
                row[f"{prefix}_{search}_transition"] = _terminal_transition(
                    search_row,
                    search,
                    baseline_correct,
                    final_correct,
                )
        scored.append(row)
    return scored


def build_summary(
    evidence_rows: Sequence[ContactEvidenceRow],
    scored_rows: Sequence[dict[str, object]],
    fixture_stats: Sequence[FixtureRunStats],
) -> dict[str, object]:
    """Build fixed descriptive counts for the evidence and both searches."""
    summary: dict[str, object] = {
        "schema": "accepted_contact_h3_r8_summary/1",
        "population": len(scored_rows),
        "accepted_contacts": len(evidence_rows),
        "fixed_rule": {
            "halo_source_frames_per_side": HALO_SOURCE_FRAMES,
            "lookaround_base30_frames": LOOKAROUND_BASE30_FRAMES,
            "maximum_local_gap_base30_frames": MAX_LOCAL_GAP_BASE30_FRAMES,
            "minimum_path_frames": MIN_PATH_FRAMES,
            "maximum_largest_step_ratio": MAX_LARGEST_STEP_RATIO,
            "minimum_directional_change_bh": MIN_DIRECTIONAL_CHANGE_BH,
            "predecessor_max_gap_base30_frames": PREDECESSOR_MAX_GAP_BASE30_FRAMES,
            "high_shot_endpoint_buffer_base30_frames": HIGH_SHOT_ENDPOINT_BUFFER_BASE30_FRAMES,
        },
        "fixture_checks": [
            {
                "fixture": stats.fixture,
                "production_guard_counts": {
                    str(code): count
                    for code, count in stats.production_guard_counts.items()
                },
                "h3_guard_counts": {
                    str(code): count
                    for code, count in stats.h3_guard_counts.items()
                },
                "changed_guard_frames": stats.changed_guard_frames,
                "halo15_exact_match": stats.halo15_exact_match,
                "high_shot_state_count": stats.high_shot_state_count,
            }
            for stats in fixture_stats
        ],
        "pre_verdicts": dict(sorted(Counter(row.pre_verdict for row in evidence_rows).items())),
        "pre_path_statuses": dict(
            sorted(Counter(row.pre_path_status for row in evidence_rows).items())
        ),
        "post_path_statuses": dict(
            sorted(Counter(row.post_path_status for row in evidence_rows).items())
        ),
        "credible_outgoing_contacts": sum(row.credible_outgoing for row in evidence_rows),
        "sequential_categories": dict(
            sorted(Counter(row["sequential_category"] for row in scored_rows).items())
        ),
        "incoming_categories": dict(
            sorted(Counter(row["incoming_category"] for row in scored_rows).items())
        ),
        "incoming_admissions": dict(
            sorted(Counter(str(row["incoming_admission"]) for row in scored_rows).items())
        ),
        "incoming_stop_reasons": dict(
            sorted(Counter(row["incoming_stop_reason"] for row in scored_rows).items())
        ),
        "tolerances": {},
    }
    tolerance_summaries: dict[str, object] = {}
    for tolerance in CONTACT_TOLERANCES_BASE30:
        prefix = f"tolerance_{tolerance}"
        tolerance_summary: dict[str, object] = {
            "baseline_labels": dict(
                sorted(Counter(row[f"{prefix}_baseline_label"] for row in scored_rows).items())
            )
        }
        for search in ("sequential", "incoming"):
            tolerance_summary[search] = {
                "transitions": dict(
                    sorted(
                        Counter(
                            row[f"{prefix}_{search}_transition"]
                            for row in scored_rows
                        ).items()
                    )
                ),
                "final_correct": sum(
                    bool(row[f"{prefix}_{search}_final_correct"])
                    for row in scored_rows
                ),
                "selected_multiple": sum(
                    bool(row[f"{prefix}_{search}_selected_multiple"])
                    for row in scored_rows
                ),
            }
        tolerance_summaries[str(tolerance)] = tolerance_summary
    summary["tolerances"] = tolerance_summaries
    return summary


def _csv_text(rows: Sequence[dict[str, object]]) -> str:
    """Serialise rows with stable field order."""
    if not rows:
        raise ValueError("cannot serialise an empty row table")
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ValueError("all output rows must have the same ordered fields")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _write_csv_gz(path: Path, rows: Sequence[dict[str, object]]) -> None:
    """Write deterministic compressed CSV evidence."""
    payload = _csv_text(rows).encode("utf-8")
    with (
        path.open("wb") as raw_handle,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as handle,
    ):
        handle.write(payload)


def _read_csv_gz(path: Path) -> list[dict[str, str]]:
    """Read compressed CSV evidence."""
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _normalised_csv_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, str]]:
    """Return the strings used by the saved CSV."""
    return list(csv.DictReader(io.StringIO(_csv_text(rows), newline="")))


def _write_json_gz(path: Path, payload: dict[str, object]) -> None:
    """Write deterministic compressed JSON evidence."""
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with (
        path.open("wb") as raw_handle,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as handle,
    ):
        handle.write(encoded)


def _read_json_gz(path: Path) -> object:
    """Read compressed JSON evidence."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _contact_dict(row: ContactEvidenceRow) -> dict[str, object]:
    """Convert one contact row to CSV-compatible values."""
    return asdict(row)


def _optional_int(value: str) -> int | None:
    return None if value == "" else int(value)


def _optional_float(value: str) -> float | None:
    return None if value == "" else float(value)


def _parse_contact_row(row: dict[str, str]) -> ContactEvidenceRow:
    """Parse one saved contact row back into its typed GT-free form."""
    return ContactEvidenceRow(
        fixture=row["fixture"],
        video_id=int(row["video_id"]),
        set_id=row["set_id"],
        rally=int(row["rally"]),
        fps=float(row["fps"]),
        span_id=int(row["span_id"]),
        accepted_rank=int(row["accepted_rank"]),
        contact_frame=int(row["contact_frame"]),
        player=row["player"] or None,
        pre_run_start=_optional_int(row["pre_run_start"]),
        pre_run_end=_optional_int(row["pre_run_end"]),
        pre_contact_gap=_optional_int(row["pre_contact_gap"]),
        pre_n_frames=_optional_int(row["pre_n_frames"]),
        pre_largest_step_ratio=_optional_float(row["pre_largest_step_ratio"]),
        pre_fitted_decrease_bh=_optional_float(row["pre_fitted_decrease_bh"]),
        pre_path_status=row["pre_path_status"],
        pre_verdict=row["pre_verdict"],
        post_run_start=_optional_int(row["post_run_start"]),
        post_run_end=_optional_int(row["post_run_end"]),
        post_contact_gap=_optional_int(row["post_contact_gap"]),
        post_n_frames=_optional_int(row["post_n_frames"]),
        post_largest_step_ratio=_optional_float(row["post_largest_step_ratio"]),
        post_fitted_decrease_bh=_optional_float(row["post_fitted_decrease_bh"]),
        post_path_status=row["post_path_status"],
        credible_outgoing=row["credible_outgoing"] == "True",
        preceding_high_shot_start=_optional_int(row["preceding_high_shot_start"]),
        preceding_high_shot_end=_optional_int(row["preceding_high_shot_end"]),
        preceding_high_shot_left_gap=_optional_int(row["preceding_high_shot_left_gap"]),
        preceding_high_shot_right_gap=_optional_int(row["preceding_high_shot_right_gap"]),
    )


def _result_fields() -> tuple[str, ...]:
    """Return the GT-free result prefix."""
    return tuple(field.name for field in fields(DualSearchRow))


def _check_saved(
    rebuilt_evidence: Sequence[ContactEvidenceRow],
    scored_rows: Sequence[dict[str, object]],
    summary: dict[str, object],
) -> None:
    """Compare rebuilt evidence, results, and summary with saved outputs."""
    saved_evidence = _read_csv_gz(CONTACT_EVIDENCE_PATH)
    normalised_evidence = _normalised_csv_rows([_contact_dict(row) for row in rebuilt_evidence])
    if saved_evidence != normalised_evidence:
        raise ValueError("saved contact evidence differs from rebuilt evidence")

    saved_results = _read_csv_gz(SEARCH_RESULTS_PATH)
    rebuilt_results = _normalised_csv_rows(scored_rows)
    if len(saved_results) != 239 or len(rebuilt_results) != 239:
        raise ValueError("saved and rebuilt search results must each contain 239 rows")
    result_fields = _result_fields()
    saved_prefix = [{field: row[field] for field in result_fields} for row in saved_results]
    rebuilt_prefix = [{field: row[field] for field in result_fields} for row in rebuilt_results]
    if saved_prefix != rebuilt_prefix:
        raise ValueError("saved GT-free search results differ from rebuilt results")
    if saved_results != rebuilt_results:
        raise ValueError("saved scored results differ from rebuilt results")
    if _read_json_gz(SUMMARY_PATH) != summary:
        raise ValueError("saved summary differs from rebuilt summary")


def run(*, write: bool, check: bool) -> None:
    """Build evidence, then write or check the fixed experiment outputs."""
    if write == check:
        raise ValueError("choose exactly one of write or check")
    rebuilt_evidence, truth_by_key, fixture_stats = build_contact_evidence()

    if write:
        _write_csv_gz(
            CONTACT_EVIDENCE_PATH,
            [_contact_dict(row) for row in rebuilt_evidence],
        )
        frozen_evidence = [
            _parse_contact_row(row)
            for row in _read_csv_gz(CONTACT_EVIDENCE_PATH)
        ]
    else:
        saved_evidence = _read_csv_gz(CONTACT_EVIDENCE_PATH)
        normalised_evidence = _normalised_csv_rows(
            [_contact_dict(row) for row in rebuilt_evidence]
        )
        if saved_evidence != normalised_evidence:
            raise ValueError("saved contact evidence differs from rebuilt evidence")
        frozen_evidence = [_parse_contact_row(row) for row in saved_evidence]

    search_rows = derive_search_rows(frozen_evidence)
    scored_rows = score_search_rows(search_rows, truth_by_key)
    summary = build_summary(frozen_evidence, scored_rows, fixture_stats)
    if write:
        _write_csv_gz(SEARCH_RESULTS_PATH, scored_rows)
        _write_json_gz(SUMMARY_PATH, summary)
        print(
            f"wrote {len(frozen_evidence)} contact rows and "
            f"{len(scored_rows)} search rows"
        )
        return
    _check_saved(rebuilt_evidence, scored_rows, summary)
    print("checked 3200 contact rows and 239 search rows")


def main(argv: Iterable[str] | None = None) -> None:
    """Parse the explicit write/check mode and run the experiment."""
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    arguments = parser.parse_args(argv)
    run(write=arguments.write, check=arguments.check)


if __name__ == "__main__":
    main()
