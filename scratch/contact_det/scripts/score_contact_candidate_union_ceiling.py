"""Measure the label-aware ceiling of the frozen Phase 3 candidate union.

The candidate union is fixed while the scorer is label-blind.  A deliberately
non-deployable oracle then chooses one candidate per labelled contact, if an
injective assignment exists.  Every accepted choice is sent through the
unchanged strict rally evaluator, so the result measures headroom rather than
proposing a production selection rule.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

MODULE_ROOT = Path(__file__).resolve().parent
CONTACT_DET_ROOT = MODULE_ROOT.parent
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import freeze_contact_evidence as evidence_freezer
import score_contact_decision_trials as decision_scorer
import score_contact_evidence as evidence_scorer
import score_contact_player_attribution as attribution_scorer
import score_contact_rallies as rally_scorer
import score_contact_shortlist as shortlist_scorer
import score_tree_contact_detector as tree_scorer

RESULTS_SCHEMA = "contact-candidate-union-ceiling/1"
PRIMARY_TOLERANCE_BASE30 = 10
SENSITIVITY_TOLERANCE_BASE30 = 5
TOLERANCES_BASE30 = (PRIMARY_TOLERANCE_BASE30, SENSITIVITY_TOLERANCE_BASE30)
ZERO_CONFIDENCE = 0.0
EXPECTED_UNASSIGNED_COUNT = 53
MAX_FULL_CEILING = 260

SOURCE_ANCHOR = "anchor"
SOURCE_ALTERNATIVE = "alternative"
SOURCE_ANCHOR_AND_ALTERNATIVE = "anchor_and_alternative"

TIMING_REJECTION_REASONS = frozenset(
    {
        rally_scorer.REASON_NO_EVENTS,
        rally_scorer.REASON_NO_RALLY,
        rally_scorer.REASON_MULTIPLE_RALLIES,
        rally_scorer.REASON_MISSING_CONTACT,
        rally_scorer.REASON_EXTRA_EVENT,
        rally_scorer.REASON_TIMING_MISMATCH,
    }
)


@dataclass(frozen=True)
class FrozenUnion:
    """All label-free event inputs used by the ceiling measurement."""

    frozen_shortlist: shortlist_scorer.FrozenShortlist
    union_rows: np.ndarray
    union_events: dict[str, tuple[rally_scorer.FixedEvent, ...]]
    union_spans: tuple[rally_scorer.FixedSpan, ...]
    baseline_spans: tuple[rally_scorer.FixedSpan, ...]
    unassigned: tuple[rally_scorer.FixedEvent, ...]
    attribution: dict[tuple[str, int], str | None]
    source_flags: dict[tuple[str, int], str]
    provenance: dict[str, str]


@dataclass(frozen=True)
class AssignmentSearch:
    """The first deterministic component-wise assignment and search counts.

    ``attempts`` counts candidate subsets checked against the unchanged
    production greedy matcher.  The other counters make the exact search
    auditable without turning them into a stopping condition: memoisation and
    the matching feasibility check only remove states whose future choices are
    provably identical or impossible.
    """

    selected_event_indices: tuple[int, ...] | None
    attempts: int
    rejected_assignments: int
    score: rally_scorer.SpanScore | None
    memo_prunes: int = 0
    feasibility_prunes: int = 0
    visited_states: int = 0
    component_count: int = 0
    maximum_component_contacts: int = 0
    maximum_component_candidates: int = 0


def _event_identity(event: rally_scorer.FixedEvent) -> tuple[str, int]:
    return event.fixture, event.frame


def _row_identity(row: np.void) -> tuple[str, int]:
    fixture = row["fixture"]
    if isinstance(fixture, bytes):
        fixture = fixture.decode("ascii")
    return str(fixture), int(row["frame"])


def _source_flag(is_anchor: bool, is_alternative: bool) -> str:
    if is_anchor and is_alternative:
        return SOURCE_ANCHOR_AND_ALTERNATIVE
    if is_anchor:
        return SOURCE_ANCHOR
    if is_alternative:
        return SOURCE_ALTERNATIVE
    raise ValueError("union event must have an anchor or alternative source")


def _provenance(
    arguments: argparse.Namespace,
    verified_features: tree_scorer.VerifiedFeatures,
    verified_candidates: tree_scorer.VerifiedCandidateScores,
    evidence: evidence_scorer.VerifiedFreeze,
) -> dict[str, str]:
    """Record content hashes without recording local or remote path details."""
    return {
        "evidence_manifest_sha256": tree_scorer._sha256(arguments.evidence_manifest),
        "contact_evidence_sha256": str(evidence.manifest["evidence_sha256"]),
        "feature_manifest_sha256": tree_scorer._sha256(arguments.feature_manifest),
        "feature_sha256": str(verified_features.manifest["feature_sha256"]),
        "tree_results_sha256": tree_scorer._sha256(arguments.tree_results),
        "candidate_manifest_sha256": tree_scorer._sha256(arguments.candidate_manifest),
        "candidate_scores_sha256": str(verified_candidates.manifest["candidate_sha256"]),
        "region_v1_manifest_sha256": tree_scorer._sha256(arguments.region_v1_manifest),
        "region_v1_results_sha256": tree_scorer._sha256(arguments.region_v1_results),
    }


def _source_flags(frozen: shortlist_scorer.FrozenShortlist) -> dict[tuple[str, int], str]:
    """Mark every union identity as anchor, alternative, or both."""
    anchor_ids = {
        _row_identity(row) for row in frozen.nplus_rows[frozen.anchor_indices]
    }
    alternative_ids = {
        _row_identity(row) for row in frozen.nplus_rows[frozen.alternative_indices]
    }
    flags: dict[tuple[str, int], str] = {}
    for row in frozen.nplus_rows[frozen.shortlist_indices]:
        identity = _row_identity(row)
        flags[identity] = _source_flag(identity in anchor_ids, identity in alternative_ids)
    if set(flags) != {
        _row_identity(row) for row in frozen.nplus_rows[frozen.shortlist_indices]
    }:
        raise ValueError("source flags do not cover the frozen shortlist")
    return flags


def _union_rows(
    frozen: shortlist_scorer.FrozenShortlist,
) -> np.ndarray:
    """Mark exactly the frozen union as retained for strict rally helpers."""
    rows = frozen.nplus_rows.copy()
    rows["decision"] = tree_scorer.CANDIDATE_BELOW_THRESHOLD
    rows["decision"][frozen.shortlist_indices] = tree_scorer.CANDIDATE_RETAINED
    return rows


def _prediction_frames(rows: np.ndarray, indices: np.ndarray) -> dict[str, np.ndarray]:
    return shortlist_scorer._prediction_frames(rows, indices)


def _replay_union_attribution(
    arguments: argparse.Namespace,
    predictions: Mapping[str, np.ndarray],
) -> dict[tuple[str, int], str | None]:
    """Replay shipped Top/Bottom attribution over every union frame."""
    freeze_arguments = argparse.Namespace(
        region_v2_manifest=arguments.feature_manifest,
        region_v2_results=arguments.tree_results,
        region_v1_manifest=arguments.region_v1_manifest,
        region_v1_results=arguments.region_v1_results,
    )
    freezes = attribution_scorer._load_tree_freezes(freeze_arguments)
    attribution = attribution_scorer._shipped_attribution_map(
        arguments.data_root,
        freezes,
        {"region_v2/candidate_union": dict(predictions)},
    )
    expected = {
        (fixture, int(frame))
        for fixture, frames in predictions.items()
        for frame in frames
    }
    if set(attribution) != expected:
        raise ValueError("shipped attribution does not cover exactly the union frames")
    return attribution


def _assert_union_identity_coverage(
    frozen: shortlist_scorer.FrozenShortlist,
    events: Mapping[str, Sequence[rally_scorer.FixedEvent]],
    attribution: Mapping[tuple[str, int], str | None],
) -> None:
    expected = {
        _row_identity(row) for row in frozen.nplus_rows[frozen.shortlist_indices]
    }
    actual = {
        _event_identity(event)
        for fixture_events in events.values()
        for event in fixture_events
    }
    if actual != expected:
        raise ValueError("half-open span assignment changed union event identities")
    if set(attribution) != expected:
        raise ValueError("attribution identity coverage differs from the union")


def _freeze_label_blind(arguments: argparse.Namespace) -> FrozenUnion:
    """Verify and freeze every prediction input before loading labels."""
    evidence = evidence_scorer.verify_freeze(arguments.evidence_manifest)
    verified_features = tree_scorer.verify_freeze(arguments.feature_manifest)
    verified_candidates = tree_scorer.verify_candidate_scores(
        arguments.candidate_manifest,
        verified_features,
        arguments.tree_results,
    )
    if tree_scorer._result_variant(verified_candidates.tree_result) != tree_scorer.CANDIDATE_VARIANT:
        raise ValueError("candidate union requires the retained baseline HGB physics scores")

    frozen_shortlist = shortlist_scorer.freeze_shortlist(
        verified_candidates.rows,
        decision_scorer._model_rows(verified_features),
    )
    shortlist_scorer.validate_frozen_shortlist(frozen_shortlist)
    union_rows = _union_rows(frozen_shortlist)
    predictions = _prediction_frames(
        frozen_shortlist.nplus_rows,
        frozen_shortlist.shortlist_indices,
    )
    attribution = _replay_union_attribution(arguments, predictions)
    union_events = rally_scorer.retained_events_from_scores(union_rows, attribution)
    _assert_union_identity_coverage(frozen_shortlist, union_events, attribution)
    union_spans = rally_scorer.fixed_spans_from_evidence(evidence.evidence, union_events)

    baseline_events = rally_scorer.retained_events_from_scores(
        frozen_shortlist.nplus_rows,
        attribution,
    )
    baseline_spans = rally_scorer.fixed_spans_from_evidence(evidence.evidence, baseline_events)
    unassigned = rally_scorer.unassigned_events(union_spans, union_events)
    source_flags = _source_flags(frozen_shortlist)
    if len(unassigned) != EXPECTED_UNASSIGNED_COUNT:
        raise ValueError(
            f"expected {EXPECTED_UNASSIGNED_COUNT} out-of-span union candidates, found {len(unassigned)}"
        )
    if set(source_flags) != {
        _event_identity(event)
        for fixture_events in union_events.values()
        for event in fixture_events
    }:
        raise ValueError("source flags do not cover assigned union candidates")
    return FrozenUnion(
        frozen_shortlist,
        union_rows,
        {fixture: tuple(events) for fixture, events in union_events.items()},
        union_spans,
        baseline_spans,
        unassigned,
        dict(attribution),
        source_flags,
        _provenance(arguments, verified_features, verified_candidates, evidence),
    )


def _load_labels() -> tuple[
    dict[str, tuple[rally_scorer.RallyReference, ...]],
    dict[tuple[str, int], str],
]:
    """Read timing labels, then side labels, after all predictions are fixed."""
    return rally_scorer._load_timing_rallies(), rally_scorer._load_side_labels()


def _tolerance_frames_by_fixture(tolerance_base30: int) -> dict[str, int]:
    return {
        fixture: evidence_scorer.scale_base30_frames(tolerance_base30, fps)
        for fixture, (_video_id, fps) in evidence_freezer.FIXTURE_SPECS.items()
    }


def _single_rally(
    span: rally_scorer.FixedSpan,
    rallies_by_fixture: Mapping[str, Sequence[rally_scorer.RallyReference]],
) -> rally_scorer.RallyReference | None:
    candidates = rally_scorer._span_rally_candidates(
        span,
        rallies_by_fixture.get(span.fixture, ()),
    )
    return candidates[0] if len(candidates) == 1 else None


def _timing_exact(score: rally_scorer.SpanScore, rally: rally_scorer.RallyReference) -> bool:
    return (
        score.rally_id == rally.rally_id
        and score.event_count == len(rally.frames)
        and score.ground_truth_contacts == len(rally.frames)
        and score.timing_matches == len(rally.frames)
        and not TIMING_REJECTION_REASONS.intersection(score.rejection_reasons)
    )


def _assignment_domains(
    span: rally_scorer.FixedSpan,
    rally: rally_scorer.RallyReference,
    tolerance_frames: int,
    target_sides: Mapping[tuple[str, int], str],
    require_correct_side: bool,
) -> tuple[tuple[int, ...], ...]:
    domains: list[tuple[int, ...]] = []
    for gt_frame in rally.frames:
        matching = []
        for event_index, event in enumerate(span.events):
            if abs(event.frame - gt_frame) > tolerance_frames:
                continue
            if require_correct_side:
                target = target_sides[(span.fixture, gt_frame)]
                if event.predicted_side != target:
                    continue
            matching.append(event_index)
        domains.append(
            tuple(
                sorted(
                    matching,
                    key=lambda index: (span.events[index].frame, index),
                )
            )
        )
    return tuple(domains)


def _assignment_components(
    domains: Sequence[Sequence[int]],
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Split the GT/candidate graph into deterministic connected components.

    A component contains every GT contact reachable through a timing-tolerance
    candidate edge.  Components share neither a GT contact nor a candidate
    event, so their injective choices and production-greedy matches cannot
    conflict.  Solving them separately turns a product of independent
    alternatives into a sum of exact searches.
    """
    event_to_contacts: dict[int, list[int]] = {}
    for contact_index, domain in enumerate(domains):
        for event_index in domain:
            event_to_contacts.setdefault(event_index, []).append(contact_index)
    for contacts in event_to_contacts.values():
        contacts.sort()

    unseen_contacts = set(range(len(domains)))
    components: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    while unseen_contacts:
        first_contact = min(unseen_contacts)
        unseen_contacts.remove(first_contact)
        contacts = {first_contact}
        events: set[int] = set()
        pending_contacts = [first_contact]
        while pending_contacts:
            contact_index = pending_contacts.pop()
            for event_index in domains[contact_index]:
                if event_index in events:
                    continue
                events.add(event_index)
                for neighbouring_contact in event_to_contacts[event_index]:
                    if neighbouring_contact in contacts:
                        continue
                    contacts.add(neighbouring_contact)
                    unseen_contacts.remove(neighbouring_contact)
                    pending_contacts.append(neighbouring_contact)
        components.append((tuple(sorted(contacts)), tuple(sorted(events))))
    return tuple(components)


def _remaining_matching_exists(
    domains: Sequence[Sequence[int]],
    order: Sequence[int],
    position: int,
    used: set[int],
) -> bool:
    """Return whether the remaining contacts have an injective matching.

    This is an exact augmenting-path bipartite matching check.  It is a
    feasibility prune only.  The search still calls the production evaluator
    for every candidate subset that survives this check, because a matching
    under the tolerance is not necessarily accepted by the evaluator's
    ordered greedy matcher.
    """
    if position == len(order):
        return True

    event_to_contact: dict[int, int] = {}

    def augment(contact_index: int, seen_events: set[int]) -> bool:
        for event_index in domains[contact_index]:
            if event_index in used or event_index in seen_events:
                continue
            seen_events.add(event_index)
            previous_contact = event_to_contact.get(event_index)
            if previous_contact is None or augment(previous_contact, seen_events):
                event_to_contact[event_index] = contact_index
                return True
        return False

    for contact_index in order[position:]:
        if not augment(contact_index, set()):
            return False
    return True


def _production_match_accepts(
    span: rally_scorer.FixedSpan,
    rally: rally_scorer.RallyReference,
    contact_indices: Sequence[int],
    selected_indices: Sequence[int],
    tolerance_frames: int,
    target_sides: Mapping[tuple[str, int], str],
    require_correct_side: bool,
) -> bool:
    """Check the unchanged production greedy matcher for one component.

    Assignment domains describe a possible GT-to-event pairing.  They do not
    force the production matcher to choose that pairing.  In particular, a
    side-compatible assignment can still fail when the greedy distance order
    pairs the selected events to different contacts.  Check the actual pairs
    here before combining components; the final full-span evaluator remains
    the authoritative acceptance check.
    """
    selected_events = tuple(
        span.events[index]
        for index in sorted(
            selected_indices,
            key=lambda index: (span.events[index].frame, index),
        )
    )
    matches = rally_scorer._greedy_matches(
        tuple(rally.frames[index] for index in contact_indices),
        tuple(event.frame for event in selected_events),
        tolerance_frames,
    )
    if len(matches) != len(contact_indices) or len(matches) != len(selected_events):
        return False
    if not require_correct_side:
        return True
    return all(
        selected_events[event_index].predicted_side
        == target_sides[(span.fixture, rally.frames[contact_indices[gt_index]])]
        for gt_index, event_index in matches
    )


def _search_component(
    span: rally_scorer.FixedSpan,
    rally: rally_scorer.RallyReference,
    tolerance_frames: int,
    target_sides: Mapping[tuple[str, int], str],
    require_correct_side: bool,
    contact_indices: Sequence[int],
    component_event_indices: Sequence[int],
    domains: Sequence[Sequence[int]],
    order: Sequence[int],
) -> AssignmentSearch:
    """Exhaustively search one disconnected component in fixed order."""
    used: set[int] = set()
    assignment: dict[int, int] = {}
    attempts = 0
    rejected = 0
    memo_prunes = 0
    feasibility_prunes = 0
    visited_states = 0
    accepted_indices: tuple[int, ...] | None = None
    seen_states: set[tuple[int, frozenset[int]]] = set()

    def visit(position: int) -> None:
        nonlocal attempts, rejected, memo_prunes, feasibility_prunes
        nonlocal visited_states, accepted_indices
        if accepted_indices is not None:
            return

        state = (position, frozenset(used))
        if state in seen_states:
            memo_prunes += 1
            return
        seen_states.add(state)
        visited_states += 1
        if not _remaining_matching_exists(domains, order, position, used):
            feasibility_prunes += 1
            return

        if position == len(order):
            attempts += 1
            selected_indices = tuple(
                sorted(
                    assignment.values(),
                    key=lambda index: (span.events[index].frame, index),
                )
            )
            if _production_match_accepts(
                span,
                rally,
                contact_indices,
                selected_indices,
                tolerance_frames,
                target_sides,
                require_correct_side,
            ):
                accepted_indices = selected_indices
            else:
                rejected += 1
            return

        gt_index = order[position]
        for event_index in domains[gt_index]:
            if event_index in used:
                continue
            used.add(event_index)
            assignment[gt_index] = event_index
            visit(position + 1)
            del assignment[gt_index]
            used.remove(event_index)
            if accepted_indices is not None:
                return

    visit(0)
    return AssignmentSearch(
        accepted_indices,
        attempts,
        rejected,
        None,
        memo_prunes,
        feasibility_prunes,
        visited_states,
        1,
        len(order),
        len(component_event_indices),
    )


def _search_assignment(
    span: rally_scorer.FixedSpan,
    rally: rally_scorer.RallyReference,
    tolerance_frames: int,
    target_sides: Mapping[tuple[str, int], str],
    require_correct_side: bool,
) -> AssignmentSearch:
    """Search all disconnected components, then run the full strict check.

    The production matcher builds a globally sorted list of GT/event pairs.
    With disconnected tolerance domains, that list is the disjoint union of
    the component lists.  Its greedy decisions therefore cannot consume an
    event or GT contact from another component.  A locally accepted choice in
    every component is consequently equivalent to their combined choice.
    """
    if tolerance_frames < 0:
        raise ValueError("tolerance must be non-negative")
    timing_domains = _assignment_domains(
        span,
        rally,
        tolerance_frames,
        target_sides,
        False,
    )
    domains = _assignment_domains(
        span,
        rally,
        tolerance_frames,
        target_sides,
        require_correct_side,
    )
    if any(not domain for domain in domains):
        return AssignmentSearch(None, 0, 0, None)

    # Components must use every timing edge.  Side-filtering the graph would
    # falsely split contacts whose production greedy matcher can still pair
    # through an opposite-side event; side constraints belong only to the
    # assignment choices within each timing component.
    components = _assignment_components(timing_domains)
    selected_indices: list[int] = []
    attempts = 0
    rejected = 0
    memo_prunes = 0
    feasibility_prunes = 0
    visited_states = 0
    maximum_contacts = 0
    maximum_candidates = 0
    for contact_indices, event_indices in components:
        order = tuple(
            sorted(
                contact_indices,
                key=lambda index: (
                    len(domains[index]),
                    rally.frames[index],
                    index,
                ),
            )
        )
        component_result = _search_component(
            span,
            rally,
            tolerance_frames,
            target_sides,
            require_correct_side,
            contact_indices,
            event_indices,
            domains,
            order,
        )
        attempts += component_result.attempts
        rejected += component_result.rejected_assignments
        memo_prunes += component_result.memo_prunes
        feasibility_prunes += component_result.feasibility_prunes
        visited_states += component_result.visited_states
        maximum_contacts = max(maximum_contacts, len(contact_indices))
        maximum_candidates = max(maximum_candidates, len(event_indices))
        if component_result.selected_event_indices is None:
            return AssignmentSearch(
                None,
                attempts,
                rejected,
                None,
                memo_prunes,
                feasibility_prunes,
                visited_states,
                len(components),
                maximum_contacts,
                maximum_candidates,
            )
        selected_indices.extend(component_result.selected_event_indices)

    selected_indices_tuple = tuple(
        sorted(
            selected_indices,
            key=lambda index: (span.events[index].frame, index),
        )
    )
    selected_span = replace(
        span,
        events=tuple(span.events[index] for index in selected_indices_tuple),
    )
    score = rally_scorer.evaluate_span(
        selected_span,
        (rally,),
        target_sides,
        tolerance_frames,
        ZERO_CONFIDENCE,
    )
    accepted = score.fully_correct if require_correct_side else _timing_exact(score, rally)
    if not accepted or (require_correct_side and score.rejection_reasons):
        raise AssertionError(
            "component production matches must survive the full strict evaluator"
        )
    return AssignmentSearch(
        selected_indices_tuple,
        attempts,
        rejected,
        score,
        memo_prunes,
        feasibility_prunes,
        visited_states,
        len(components),
        maximum_contacts,
        maximum_candidates,
    )


def _span_detail(
    span: rally_scorer.FixedSpan,
    rally: rally_scorer.RallyReference,
    search: AssignmentSearch,
    source_flags: Mapping[tuple[str, int], str],
) -> dict[str, Any]:
    if search.selected_event_indices is None:
        raise ValueError("cannot report a span without a feasible assignment")
    selected_events = tuple(span.events[index] for index in search.selected_event_indices)
    if search.score is None:
        raise ValueError("accepted assignment is missing its strict evaluation")
    if not search.score.fully_correct and not _timing_exact(search.score, rally):
        raise ValueError("accepted assignment is missing its strict evaluation")
    if search.score.fully_correct and search.score.rejection_reasons:
        raise ValueError("fully correct assignment has rejection reasons")
    if search.score.event_count != len(rally.frames) or search.score.timing_matches != len(rally.frames):
        raise ValueError("accepted assignment does not contain exactly one event per contact")
    flags = [source_flags[_event_identity(event)] for event in selected_events]
    dropped_events = tuple(
        event
        for index, event in enumerate(span.events)
        if index not in search.selected_event_indices
    )
    selected_anchor_count = sum(
        flag in {SOURCE_ANCHOR, SOURCE_ANCHOR_AND_ALTERNATIVE} for flag in flags
    )
    selected_non_anchor_alternative_count = sum(
        flag == SOURCE_ALTERNATIVE for flag in flags
    )
    dropped_anchor_count = sum(
        source_flags[_event_identity(event)]
        in {SOURCE_ANCHOR, SOURCE_ANCHOR_AND_ALTERNATIVE}
        for event in dropped_events
    )
    dropped_non_anchor_alternative_count = sum(
        source_flags[_event_identity(event)] == SOURCE_ALTERNATIVE
        for event in dropped_events
    )
    return {
        "fixture": span.fixture,
        "span_id": span.span_id,
        "start_frame": span.start_frame,
        "end_frame": span.end_frame,
        "rally_id": rally.rally_id,
        "ground_truth_contacts": len(rally.frames),
        "candidate_events_in_span": len(span.events),
        "selected_frames": [event.frame for event in selected_events],
        "selected_predicted_sides": [event.predicted_side for event in selected_events],
        "selected_source_flags": flags,
        "source_flag_counts": dict(Counter(flags)),
        "action_counts": {
            "retain": len(selected_events),
            "drop": len(span.events) - len(selected_events),
            "selected_anchors": selected_anchor_count,
            "selected_non_anchor_alternatives": selected_non_anchor_alternative_count,
            "selected_dual_source": flags.count(SOURCE_ANCHOR_AND_ALTERNATIVE),
            "dropped_anchors": dropped_anchor_count,
            "dropped_non_anchor_alternatives": dropped_non_anchor_alternative_count,
        },
        "search_attempts": search.attempts,
        "rejected_assignments": search.rejected_assignments,
        "memo_prunes": search.memo_prunes,
        "feasibility_prunes": search.feasibility_prunes,
        "visited_search_states": search.visited_states,
        "component_count": search.component_count,
        "maximum_component_contacts": search.maximum_component_contacts,
        "maximum_component_candidates": search.maximum_component_candidates,
        "strict_rejection_reasons": list(search.score.rejection_reasons),
    }


def _ceiling_for_tolerance(
    spans: Sequence[rally_scorer.FixedSpan],
    rallies_by_fixture: Mapping[str, Sequence[rally_scorer.RallyReference]],
    target_sides: Mapping[tuple[str, int], str],
    tolerance_base30: int,
    source_flags: Mapping[tuple[str, int], str],
    *,
    require_correct_side: bool,
) -> dict[str, Any]:
    tolerance_frames = _tolerance_frames_by_fixture(tolerance_base30)
    feasible: list[dict[str, Any]] = []
    single_rally_spans = 0
    false_spans = 0
    multiple_rally_spans = 0
    search_attempts = 0
    rejected_assignments = 0
    memo_prunes = 0
    feasibility_prunes = 0
    visited_search_states = 0
    total_components = 0
    maximum_component_contacts = 0
    maximum_component_candidates = 0
    maximum_candidate_events = 0
    maximum_search_attempts = 0
    for span in spans:
        maximum_candidate_events = max(maximum_candidate_events, len(span.events))
        candidates = rally_scorer._span_rally_candidates(
            span,
            rallies_by_fixture.get(span.fixture, ()),
        )
        if not candidates:
            false_spans += 1
            continue
        if len(candidates) != 1:
            multiple_rally_spans += 1
            continue
        single_rally_spans += 1
        rally = candidates[0]
        search = _search_assignment(
            span,
            rally,
            tolerance_frames[span.fixture],
            target_sides,
            require_correct_side,
        )
        search_attempts += search.attempts
        rejected_assignments += search.rejected_assignments
        memo_prunes += search.memo_prunes
        feasibility_prunes += search.feasibility_prunes
        visited_search_states += search.visited_states
        total_components += search.component_count
        maximum_component_contacts = max(
            maximum_component_contacts,
            search.maximum_component_contacts,
        )
        maximum_component_candidates = max(
            maximum_component_candidates,
            search.maximum_component_candidates,
        )
        maximum_search_attempts = max(maximum_search_attempts, search.attempts)
        if search.selected_event_indices is not None:
            feasible.append(_span_detail(span, rally, search, source_flags))

    identities = {
        (str(row["fixture"]), int(row["span_id"])) for row in feasible
    }
    return {
        "tolerance_base30": tolerance_base30,
        "tolerance_frames": tolerance_frames,
        "require_correct_side": require_correct_side,
        "single_rally_spans": single_rally_spans,
        "false_spans": false_spans,
        "multiple_rally_spans": multiple_rally_spans,
        "false_or_multi_rally_spans": false_spans + multiple_rally_spans,
        "total_assignment_attempts": search_attempts,
        "total_rejected_assignments": rejected_assignments,
        "total_memo_prunes": memo_prunes,
        "total_feasibility_prunes": feasibility_prunes,
        "total_visited_search_states": visited_search_states,
        "total_components": total_components,
        "maximum_component_contacts": maximum_component_contacts,
        "maximum_component_candidates": maximum_component_candidates,
        "maximum_candidate_events_in_span": maximum_candidate_events,
        "maximum_assignment_attempts_in_span": maximum_search_attempts,
        "feasible_span_count": len(feasible),
        "feasible_identities": [
            {"fixture": fixture, "span_id": span_id}
            for fixture, span_id in sorted(identities)
        ],
        "feasible_spans": feasible,
    }


def _identity_set(report: Mapping[str, Any]) -> set[tuple[str, int]]:
    return {
        (str(row["fixture"]), int(row["span_id"]))
        for row in report["feasible_identities"]
    }


def _baseline_report(
    spans: Sequence[rally_scorer.FixedSpan],
    rallies_by_fixture: Mapping[str, Sequence[rally_scorer.RallyReference]],
    target_sides: Mapping[tuple[str, int], str],
) -> dict[str, Any]:
    fps_by_fixture = {
        fixture: fps
        for fixture, (_video_id, fps) in evidence_freezer.FIXTURE_SPECS.items()
    }
    output: dict[str, Any] = {}
    for tolerance in TOLERANCES_BASE30:
        report = rally_scorer.score_strict_rallies(
            spans,
            rallies_by_fixture,
            target_sides,
            fps_by_fixture,
            tolerance_base30=tolerance,
            requirements=(ZERO_CONFIDENCE,),
            detail_requirement=ZERO_CONFIDENCE,
        )
        detail = report["confidence_curve"][0]
        identities = {
            (str(row["fixture"]), int(row["span_id"]))
            for row in report["spans"]
            if row["fully_correct"]
        }
        output[str(tolerance)] = {
            "summary": detail,
            "fully_correct_identities": [
                {"fixture": fixture, "span_id": span_id}
                for fixture, span_id in sorted(identities)
            ],
            "spans": report["spans"],
        }
    return output


def _assert_real_input_invariants(
    frozen: FrozenUnion,
    baseline: Mapping[str, Any],
    timing_ceiling: Mapping[str, Any],
    full_ceiling: Mapping[str, Any],
) -> None:
    """Guard the pinned real fixture against silent identity drift."""
    if len(frozen.frozen_shortlist.anchor_indices) != shortlist_scorer.EXPECTED_ANCHOR_COUNT:
        raise ValueError("real-input anchor count differs from the Phase 3 contract")
    if len(frozen.frozen_shortlist.shortlist_indices) != shortlist_scorer.EXPECTED_SHORTLIST_COUNT:
        raise ValueError("real-input union count differs from the Phase 3 contract")
    if len(frozen.unassigned) != EXPECTED_UNASSIGNED_COUNT:
        raise ValueError("real-input out-of-span count differs")
    for tolerance in TOLERANCES_BASE30:
        key = str(tolerance)
        baseline_count = sum(
            bool(row["fully_correct"]) for row in baseline[key]["spans"]
        )
        expected = 27 if tolerance == PRIMARY_TOLERANCE_BASE30 else 24
        if baseline_count != expected:
            raise ValueError(
                f"N+ baseline fully-correct count changed at ±{tolerance}: {baseline_count}"
            )
        timing_ids = _identity_set(timing_ceiling[key])
        full_ids = _identity_set(full_ceiling[key])
        baseline_ids = {
            (str(row["fixture"]), int(row["span_id"]))
            for row in baseline[key]["spans"]
            if row["fully_correct"]
        }
        if not baseline_ids <= full_ids:
            raise ValueError(f"baseline fully-correct identities are missing from the ±{tolerance} ceiling")
        if not full_ids <= timing_ids:
            raise ValueError(f"full ceiling identities exceed timing-only identities at ±{tolerance}")
        if len(full_ids) > MAX_FULL_CEILING:
            raise ValueError(f"full ceiling is implausibly large at ±{tolerance}")


def _result_payload(
    frozen: FrozenUnion,
    rallies_by_fixture: Mapping[str, Sequence[rally_scorer.RallyReference]],
    target_sides: Mapping[tuple[str, int], str],
) -> dict[str, Any]:
    baseline = _baseline_report(frozen.baseline_spans, rallies_by_fixture, target_sides)
    timing_ceiling = {
        str(tolerance): _ceiling_for_tolerance(
            frozen.union_spans,
            rallies_by_fixture,
            target_sides,
            tolerance,
            frozen.source_flags,
            require_correct_side=False,
        )
        for tolerance in TOLERANCES_BASE30
    }
    full_ceiling = {
        str(tolerance): _ceiling_for_tolerance(
            frozen.union_spans,
            rallies_by_fixture,
            target_sides,
            tolerance,
            frozen.source_flags,
            require_correct_side=True,
        )
        for tolerance in TOLERANCES_BASE30
    }
    _assert_real_input_invariants(frozen, baseline, timing_ceiling, full_ceiling)
    source_counts = Counter(frozen.source_flags.values())
    unassigned_source_counts = Counter(
        frozen.source_flags[_event_identity(event)] for event in frozen.unassigned
    )
    unassigned_details = [
        {
            "fixture": event.fixture,
            "frame": event.frame,
            "timing_score": event.timing_score,
            "predicted_side": event.predicted_side,
            "source_flag": frozen.source_flags[_event_identity(event)],
        }
        for event in frozen.unassigned
    ]
    output: dict[str, Any] = {
        "schema": RESULTS_SCHEMA,
        "fixture_set": list(evidence_freezer.FIXTURE_SPECS),
        "labels_read_after_union_fixed": True,
        "oracle": {
            "deployable": False,
            "description": (
                "label-aware exhaustive candidate selection; this is an upper bound, "
                "not a production rule"
            ),
            "assignment": "one injective candidate per labelled contact",
            "span_policy": "false and multi-rally spans abstain",
            "confidence_requirement": ZERO_CONFIDENCE,
        },
        "label_free_counts": {
            "anchors": len(frozen.frozen_shortlist.anchor_indices),
            "alternative_selections": len(frozen.frozen_shortlist.alternative_indices),
            "distinct_alternatives": len(
                np.unique(frozen.frozen_shortlist.alternative_indices)
            ),
            "shortlist_union": len(frozen.frozen_shortlist.shortlist_indices),
            "unassigned_candidates": len(frozen.unassigned),
            "source_flag_counts": dict(source_counts),
            "unassigned_source_flag_counts": dict(unassigned_source_counts),
            "alternatives_already_anchors": shortlist_scorer.EXPECTED_ALTERNATIVES_ALREADY_ANCHORS,
        },
        "inputs": dict(frozen.provenance),
        "unassigned_candidates": unassigned_details,
        "baseline_nplus": baseline,
        "timing_only_ceiling": timing_ceiling,
        "timing_and_side_ceiling": full_ceiling,
        "delta_from_baseline": {},
    }
    for tolerance in TOLERANCES_BASE30:
        key = str(tolerance)
        baseline_ids = _identity_set(
            {"feasible_identities": baseline[key]["fully_correct_identities"]}
        )
        timing_ids = _identity_set(timing_ceiling[key])
        full_ids = _identity_set(full_ceiling[key])
        output["delta_from_baseline"][key] = {
            "baseline_fully_correct": len(baseline_ids),
            "timing_only_feasible": len(timing_ids),
            "timing_and_side_feasible": len(full_ids),
            "timing_only_delta": len(timing_ids) - len(baseline_ids),
            "timing_and_side_delta": len(full_ids) - len(baseline_ids),
            "new_timing_only_identities": [
                {"fixture": fixture, "span_id": span_id}
                for fixture, span_id in sorted(timing_ids - baseline_ids)
            ],
            "new_timing_and_side_identities": [
                {"fixture": fixture, "span_id": span_id}
                for fixture, span_id in sorted(full_ids - baseline_ids)
            ],
            "lost_baseline_identities": [
                {"fixture": fixture, "span_id": span_id}
                for fixture, span_id in sorted(baseline_ids - full_ids)
            ],
        }
    return output


def score(arguments: argparse.Namespace) -> dict[str, Any]:
    """Freeze predictions, load labels, then score both ceiling variants."""
    frozen = _freeze_label_blind(arguments)
    rallies_by_fixture, target_sides = _load_labels()
    return _result_payload(frozen, rallies_by_fixture, target_sides)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    raw_root = CONTACT_DET_ROOT / "raw"
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=raw_root / "region_v2" / "run_a" / "tree_contact_features_manifest.json",
    )
    parser.add_argument(
        "--tree-results",
        type=Path,
        default=raw_root / "region_v2" / "tree_contact_results.json.gz",
    )
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument(
        "--evidence-manifest",
        type=Path,
        default=raw_root / "contact_evidence_manifest.json",
    )
    parser.add_argument(
        "--region-v1-manifest",
        type=Path,
        default=raw_root / "tree_trial" / "tree_contact_features_manifest.json",
    )
    parser.add_argument(
        "--region-v1-results",
        type=Path,
        default=raw_root / "tree_trial" / "tree_contact_results_with_frames.json.gz",
    )
    parser.add_argument("--data-root", type=Path, default=raw_root / "region_v2_inputs")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    payload = score(arguments)
    rally_scorer.write_results(arguments.output, payload)
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
