"""Label-free player-side rules for a fixed contact stream."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from annotator.point_winner import Half, fit_alternation
from scratch.contact_det.scripts.score_contact_rallies import FixedEvent, FixedSpan

SectionIdentity = tuple[str, int]


@dataclass(frozen=True)
class SideDecision:
    """One whole-section alternating side choice."""

    fixture: str
    span_id: int
    event_frames: tuple[int, ...]
    sides_before: tuple[str | None, ...]
    sides_after: tuple[str, ...]
    top_final_score: int
    bot_final_score: int

    @property
    def score_gap(self) -> int:
        return abs(self.top_final_score - self.bot_final_score)


def alternating_pattern(final_half: Half, event_count: int) -> tuple[str, ...]:
    """Build the alternating pattern that ends on ``final_half``."""
    return tuple(
        str(final_half if (event_count - 1 - index) % 2 == 0 else Half.TOP if final_half == Half.BOT else Half.BOT)
        for index in range(event_count)
    )


def choose_simple_alternation(span: FixedSpan) -> SideDecision | None:
    """Choose the better alternating phase, or leave a tied section alone."""
    guesses = [None if event.predicted_side is None else Half(event.predicted_side) for event in span.events]
    final_half = fit_alternation(guesses)
    if final_half is None:
        return None
    top_pattern = alternating_pattern(Half.TOP, len(span.events))
    bot_pattern = alternating_pattern(Half.BOT, len(span.events))
    sides_before = tuple(event.predicted_side for event in span.events)
    top_score = sum(side is not None and side == assigned for side, assigned in zip(sides_before, top_pattern, strict=True))
    bot_score = sum(side is not None and side == assigned for side, assigned in zip(sides_before, bot_pattern, strict=True))
    return SideDecision(
        fixture=span.fixture,
        span_id=span.span_id,
        event_frames=tuple(event.frame for event in span.events),
        sides_before=sides_before,
        sides_after=alternating_pattern(final_half, len(span.events)),
        top_final_score=top_score,
        bot_final_score=bot_score,
    )


def simple_alternation_decisions(spans: Sequence[FixedSpan]) -> tuple[SideDecision, ...]:
    """Apply the existing categorical vote to every section it can resolve."""
    decisions: list[SideDecision] = []
    for span in spans:
        decision = choose_simple_alternation(span)
        if decision is not None and decision.sides_after != decision.sides_before:
            decisions.append(decision)
    return tuple(decisions)


def side_decisions_from_payload(
    payload: Mapping[str, Any],
    expected_schema: str,
) -> tuple[SideDecision, ...]:
    """Read side decisions from one of the compact saved records."""
    raw_decisions = payload.get("decisions")
    if payload.get("schema") != expected_schema or not isinstance(raw_decisions, list):
        raise ValueError("Side-decision record has another schema")
    return tuple(
        SideDecision(
            fixture=str(row["fixture"]),
            span_id=int(row["span_id"]),
            event_frames=tuple(int(frame) for frame in row["event_frames"]),
            sides_before=tuple(row["sides_before"]),
            sides_after=tuple(str(side) for side in row["sides_after"]),
            top_final_score=int(row["top_final_score"]),
            bot_final_score=int(row["bot_final_score"]),
        )
        for row in raw_decisions
    )


def apply_side_decisions(
    spans: Sequence[FixedSpan],
    events_by_fixture: Mapping[str, Sequence[FixedEvent]],
    decisions: Sequence[SideDecision],
) -> tuple[tuple[FixedSpan, ...], Mapping[str, tuple[FixedEvent, ...]]]:
    """Return new streams with only the named player sides changed."""
    overrides: dict[tuple[str, int], str] = {}
    span_lookup = {(span.fixture, span.span_id): span for span in spans}
    for decision in decisions:
        span = span_lookup[(decision.fixture, decision.span_id)]
        frames = tuple(event.frame for event in span.events)
        sides = tuple(event.predicted_side for event in span.events)
        if frames != decision.event_frames or sides != decision.sides_before:
            raise ValueError(f"{decision.fixture}/{decision.span_id}: side decision source differs")
        for frame, side in zip(decision.event_frames, decision.sides_after, strict=True):
            identity = (decision.fixture, frame)
            if identity in overrides:
                raise ValueError(f"{decision.fixture}/{frame}: side changed twice")
            overrides[identity] = side

    revised_events: dict[str, tuple[FixedEvent, ...]] = {}
    for fixture, events in events_by_fixture.items():
        revised_events[fixture] = tuple(
            FixedEvent(
                fixture=event.fixture,
                frame=event.frame,
                timing_score=event.timing_score,
                predicted_side=overrides.get((fixture, event.frame), event.predicted_side),
            )
            for event in events
        )
    revised_by_identity = {
        (event.fixture, event.frame): event
        for fixture_events in revised_events.values()
        for event in fixture_events
    }
    revised_spans = tuple(
        FixedSpan(
            fixture=span.fixture,
            span_id=span.span_id,
            start_frame=span.start_frame,
            end_frame=span.end_frame,
            events=tuple(revised_by_identity[(event.fixture, event.frame)] for event in span.events),
        )
        for span in spans
    )
    return revised_spans, MappingProxyType(revised_events)
