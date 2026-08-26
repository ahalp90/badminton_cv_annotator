"""Freeze and score the bounded serve-prefix candidate experiment.

The candidate list is deliberately built from label-blind artefacts.  The
timing oracle and strict rally score are loaded only after that list, the fixed
heuristic action, and all Top/Bottom predictions have been frozen.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
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

import score_contact_decision_trials as decision_scorer
import score_contact_evidence as evidence_scorer
import score_contact_player_attribution as attribution_scorer
import score_contact_rallies as rally_scorer
import score_tree_contact_detector as tree_scorer

RESULTS_SCHEMA = "contact-serve-prefix-score/3"
CONSTRUCTION_SCHEMA = "contact-serve-prefix-construction/1"
SELECTED_TRIAL_ID = "N+"
NPLUS_RADIUS_BASE30 = 6
PEAK_COUNT = 3
MAX_CANDIDATES = 5
PRIMARY_TOLERANCE_BASE30 = 10
SENSITIVITY_TOLERANCE_BASE30 = 5

EXPECTED_NPLUS_EVENT_COUNT = 3_238
EXPECTED_SPAN_COUNT = 311
EXPECTED_RALLY_COUNT = 292
EXPECTED_CONTACT_COUNT = 3_128

SOURCE_HGB_PEAK_1 = "hgb_peak_1"
SOURCE_HGB_PEAK_2 = "hgb_peak_2"
SOURCE_HGB_PEAK_3 = "hgb_peak_3"
SOURCE_FILTERED_HEURISTIC = "filtered_heuristic"
SOURCE_ANCHOR = "anchor"
ABSTENTION_NO_ANCHOR = "no_anchor"
ABSTENTION_MALFORMED_BOUNDS = "malformed_prefix_bounds"
OUTPUT_START_DETECTED_SPAN = "detected_span"
OUTPUT_START_SERVE_PREPEND = "serve_prepend"


@dataclass(frozen=True)
class PrefixCandidate:
    """One label-blind candidate in a detected span's prefix."""

    fixture: str
    span_id: int
    interval_id: int
    frame: int
    timing_score: float
    rank: int
    source_flags: tuple[str, ...]
    heuristic_span_ids: tuple[int, ...] = ()

    @property
    def identity(self) -> tuple[str, int, int]:
        """Return the exact frozen candidate identity."""
        return self.fixture, self.interval_id, self.frame

    @property
    def is_anchor(self) -> bool:
        """Return whether this row is the span anchor."""
        return SOURCE_ANCHOR in self.source_flags

    @property
    def is_filtered_heuristic(self) -> bool:
        """Return whether this row came from the filtered heuristic stream."""
        return SOURCE_FILTERED_HEURISTIC in self.source_flags


@dataclass(frozen=True)
class PrefixAction:
    """A fixed or oracle action attached to one detected span."""

    action: str
    candidate: PrefixCandidate | None
    anchor_frame: int | None
    reason: str

    @property
    def selected_frame(self) -> int | None:
        """Return the selected frame, if this action selects one."""
        return None if self.candidate is None else self.candidate.frame


@dataclass(frozen=True)
class PrefixRecord:
    """The frozen prefix bounds, candidates and fixed action for one span."""

    fixture: str
    span_id: int
    interval_id: int | None
    lower_frame: int | None
    upper_frame: int | None
    anchor_frame: int | None
    anchor_score: float | None
    radius_frames: int
    candidates: tuple[PrefixCandidate, ...]
    fixed_action: PrefixAction
    abstention_reason: str | None = None

    @property
    def prefix_length(self) -> int | None:
        """Return the inclusive prefix length in source frames."""
        if self.lower_frame is None or self.upper_frame is None:
            return None
        return self.upper_frame - self.lower_frame + 1


@dataclass(frozen=True)
class FrozenPrefixSet:
    """A complete label-blind construction and its N+ source rows."""

    prefixes: tuple[PrefixRecord, ...]
    nplus_rows: np.ndarray
    nplus_event_count: int
    candidate_source_count: int
    candidate_count: int
    exact_deduplication_count: int
    malformed_count: int


@dataclass(frozen=True)
class VerifiedLabelBlindInputs:
    """All verified input artefacts needed before labels are loaded."""

    features: tree_scorer.VerifiedFeatures
    candidates: tree_scorer.VerifiedCandidateScores
    evidence: evidence_scorer.VerifiedFreeze
    nplus_rows: np.ndarray
    model_rows: np.ndarray
    search_intervals: dict[str, tuple[tuple[int, int], ...]]


@dataclass(frozen=True)
class ServeOutputSpanBounds:
    """Detected and output bounds for one serve-prefix result."""

    fixture: str
    span_id: int
    detected_span_start: int
    detected_span_end: int
    serve_prepend_frame: int | None
    output_span_start: int
    output_span_end: int
    output_start_source: str


@dataclass(frozen=True)
class AttachedSpanSet:
    """Scoring spans and the bounds used to make them."""

    spans: tuple[rally_scorer.FixedSpan, ...]
    bounds: tuple[ServeOutputSpanBounds, ...]


def scaled_nplus_radius(fps: float) -> int:
    """Scale N+'s six base-30 frames using the repository frame rule."""
    return tree_scorer._scaled_frames(NPLUS_RADIUS_BASE30, float(fps))


def _decode_fixture_rows(rows: np.ndarray) -> np.ndarray:
    """Decode the fixed-width fixture field used by the score sidecar."""
    values = rows["fixture"]
    if values.dtype.kind == "S":
        return np.char.decode(values, "ascii")
    if values.dtype.kind == "U":
        return values.astype(str)
    raise TypeError("score rows fixture field must be fixed-width text")


def _retained_mask(rows: np.ndarray) -> np.ndarray:
    """Return N+ retained rows, accepting small pure-test tables without decisions."""
    if rows.dtype.names is None or "frame" not in rows.dtype.names:
        raise ValueError("score rows must contain frame")
    if "decision" not in rows.dtype.names:
        return np.ones(len(rows), dtype=bool)
    return rows["decision"] == tree_scorer.CANDIDATE_RETAINED


def _identity_rows(rows: np.ndarray) -> dict[tuple[str, int, int], int]:
    """Index rows by the exact fixture, interval and frame identity."""
    required = {"fixture", "interval_id", "frame", "timing_score"}
    if rows.dtype.names is None or not required.issubset(rows.dtype.names):
        raise ValueError(f"score rows are missing fields: {sorted(required)}")
    fixtures = _decode_fixture_rows(rows)
    output: dict[tuple[str, int, int], int] = {}
    for index, (fixture, interval_id, frame) in enumerate(
        zip(fixtures, rows["interval_id"], rows["frame"], strict=True)
    ):
        identity = (str(fixture), int(interval_id), int(frame))
        if identity in output:
            raise ValueError(f"duplicate score-row identity: {identity}")
        score = float(rows["timing_score"][index])
        if not math.isfinite(score):
            raise ValueError(f"non-finite HGB score at {identity}")
        output[identity] = index
    return output


def _normalise_intervals(
    intervals_by_fixture: Mapping[str, Sequence[Sequence[int]]],
) -> dict[str, tuple[tuple[int, int], ...]]:
    """Validate and normalise half-open frozen search intervals."""
    output: dict[str, tuple[tuple[int, int], ...]] = {}
    for fixture, raw_intervals in intervals_by_fixture.items():
        parsed: list[tuple[int, int]] = []
        for interval in raw_intervals:
            if len(interval) != 2:
                raise ValueError(f"{fixture}: search interval must have two bounds")
            start, end = (int(interval[0]), int(interval[1]))
            if start < 0 or start >= end:
                raise ValueError(f"{fixture}: malformed search interval {(start, end)}")
            if parsed and start < parsed[-1][1]:
                raise ValueError(f"{fixture}: search intervals overlap")
            parsed.append((start, end))
        output[str(fixture)] = tuple(parsed)
    return output


def _interval_for_frame(
    intervals: Sequence[tuple[int, int]],
    frame: int,
    *,
    name: str,
) -> tuple[int, tuple[int, int]]:
    """Find the one frozen interval containing a source frame."""
    matches = [
        (interval_id, bounds)
        for interval_id, bounds in enumerate(intervals)
        if bounds[0] <= frame < bounds[1]
    ]
    if len(matches) != 1:
        raise ValueError(f"{name}: frame {frame} belongs to {len(matches)} search intervals")
    return matches[0]


def make_prefix_bounds(
    fixture: str,
    span_id: int,
    span_start: int,
    span_end: int,
    anchor_frame: int,
    interval_id: int,
    interval_bounds: tuple[int, int],
    preceding_span_end: int | None,
) -> tuple[int, int]:
    """Build the inclusive prefix bounds from one frozen interval and span."""
    if span_start < 0 or span_start >= span_end:
        raise ValueError(f"{fixture}/{span_id}: detected span bounds are malformed")
    interval_start, interval_end = interval_bounds
    if interval_start < 0 or interval_start >= interval_end:
        raise ValueError(f"{fixture}/{span_id}: search interval bounds are malformed")
    if not interval_start <= anchor_frame < interval_end:
        raise ValueError(f"{fixture}/{span_id}: anchor lies outside its search interval")
    lower = interval_start
    if preceding_span_end is not None and interval_start <= preceding_span_end < interval_end:
        lower = max(lower, preceding_span_end)
    upper = anchor_frame
    if lower > upper:
        raise ValueError(f"{fixture}/{span_id}: prefix lower bound exceeds anchor")
    return lower, upper


def _peak_indices(
    rows: np.ndarray,
    fixture: str,
    interval_id: int,
    lower_frame: int,
    upper_frame: int,
    radius_frames: int,
    *,
    limit: int = PEAK_COUNT,
    fixture_names: np.ndarray | None = None,
    interval_row_indices: np.ndarray | None = None,
) -> tuple[int, ...]:
    """Return strongest threshold-free NMS peaks in one inclusive prefix."""
    if limit <= 0:
        return ()
    if radius_frames < 0:
        raise ValueError("N+ radius must be non-negative")
    if interval_row_indices is None:
        fixtures = _decode_fixture_rows(rows) if fixture_names is None else fixture_names
        local = np.flatnonzero(
            (fixtures == fixture)
            & (rows["interval_id"] == interval_id)
            & (rows["frame"] >= lower_frame)
            & (rows["frame"] <= upper_frame)
        )
    else:
        local = interval_row_indices[
            (rows["frame"][interval_row_indices] >= lower_frame)
            & (rows["frame"][interval_row_indices] <= upper_frame)
        ]
    order = sorted(
        (int(index) for index in local),
        key=lambda index: (-float(rows["timing_score"][index]), int(rows["frame"][index]), index),
    )
    kept: list[int] = []
    for index in order:
        frame = int(rows["frame"][index])
        if all(abs(frame - int(rows["frame"][other])) > radius_frames for other in kept):
            kept.append(index)
    return tuple(kept[:limit])


def threshold_free_peaks(
    rows: np.ndarray,
    fixture: str,
    interval_id: int,
    lower_frame: int,
    upper_frame: int,
    radius_frames: int,
    *,
    limit: int = PEAK_COUNT,
) -> tuple[tuple[int, float], ...]:
    """Return peak frames and scores, ordered by score then earlier frame."""
    indices = _peak_indices(
        rows,
        fixture,
        interval_id,
        lower_frame,
        upper_frame,
        radius_frames,
        limit=limit,
    )
    return tuple((int(rows["frame"][index]), float(rows["timing_score"][index])) for index in indices)


def _span_rows(evidence: Mapping[str, Any], fixture: str) -> list[tuple[int, int, int]]:
    """Return detected spans in frame order as ``(id, start, end)`` tuples."""
    fixture_rows = [row for row in evidence.get("fixtures", ()) if row.get("fixture") == fixture]
    if len(fixture_rows) != 1:
        raise ValueError(f"expected one evidence row for {fixture}")
    raw_spans = fixture_rows[0].get("spans")
    if not isinstance(raw_spans, Sequence) or isinstance(raw_spans, (str, bytes)):
        raise TypeError(f"{fixture}: evidence spans must be a sequence")
    spans: list[tuple[int, int, int]] = []
    for raw_span in raw_spans:
        if not isinstance(raw_span, Mapping):
            raise TypeError(f"{fixture}: evidence span must be an object")
        span_id = int(raw_span["span_id"])
        start = int(raw_span["start_frame"])
        end = int(raw_span["end_frame"])
        if start < 0 or start >= end:
            raise ValueError(f"{fixture}/{span_id}: detected span bounds are malformed")
        spans.append((span_id, start, end))
    if [row[0] for row in spans] != [row[0] for row in sorted(spans, key=lambda row: row[1])]:
        raise ValueError(f"{fixture}: detected spans are not in frame order")
    if any(left[2] > right[1] for left, right in pairwise(spans)):
        raise ValueError(f"{fixture}: detected spans overlap or are out of order")
    return spans


def _filtered_heuristic_rows(
    evidence: Mapping[str, Any],
    intervals_by_fixture: Mapping[str, tuple[tuple[int, int], ...]],
    score_indices: Mapping[tuple[str, int, int], int],
    rows: np.ndarray,
) -> dict[tuple[str, int, int], tuple[int, ...]]:
    """Index filtered heuristic contacts by exact HGB identity."""
    output: dict[tuple[str, int, int], list[int]] = {}
    for fixture_row in evidence.get("fixtures", ()):
        fixture = str(fixture_row["fixture"])
        intervals = intervals_by_fixture[fixture]
        for span in fixture_row["spans"]:
            span_id = int(span["span_id"])
            for contact in span["contacts"]:
                if not bool(contact["filtered"]):
                    continue
                frame = int(contact["contact_frame"])
                try:
                    interval_id, _bounds = _interval_for_frame(
                        intervals, frame, name=f"{fixture}/{span_id} filtered heuristic"
                    )
                except ValueError as error:
                    # Contacts outside the frozen HGB search surface cannot be
                    # a prefix candidate, so they are not silently cross-joined.
                    if "belongs to 0 search intervals" in str(error):
                        continue
                    raise
                identity = (fixture, interval_id, frame)
                if identity not in score_indices:
                    raise ValueError(
                        f"{fixture}/{span_id}/{frame}: filtered heuristic has no exact HGB row"
                    )
                output.setdefault(identity, []).append(span_id)
    return {identity: tuple(sorted(set(span_ids))) for identity, span_ids in output.items()}


def _candidate_for_index(
    rows: np.ndarray,
    index: int,
    fixture: str,
    span_id: int,
    rank: int,
    source_flags: Sequence[str],
    heuristic_span_ids: Sequence[int] = (),
) -> PrefixCandidate:
    """Create a candidate from one exact score row."""
    return PrefixCandidate(
        fixture=fixture,
        span_id=span_id,
        interval_id=int(rows["interval_id"][index]),
        frame=int(rows["frame"][index]),
        timing_score=float(rows["timing_score"][index]),
        rank=rank,
        source_flags=tuple(source_flags),
        heuristic_span_ids=tuple(sorted({int(value) for value in heuristic_span_ids})),
    )


def _merge_candidate(
    candidates: list[PrefixCandidate],
    candidate: PrefixCandidate,
) -> None:
    """Merge source flags by exact identity while preserving candidate order."""
    for index, existing in enumerate(candidates):
        if existing.identity != candidate.identity:
            continue
        flags = tuple(dict.fromkeys((*existing.source_flags, *candidate.source_flags)))
        spans = tuple(sorted({*existing.heuristic_span_ids, *candidate.heuristic_span_ids}))
        candidates[index] = PrefixCandidate(
            existing.fixture,
            existing.span_id,
            existing.interval_id,
            existing.frame,
            existing.timing_score,
            existing.rank,
            flags,
            spans,
        )
        return
    candidates.append(candidate)


def _renumber_candidates(candidates: Sequence[PrefixCandidate]) -> tuple[PrefixCandidate, ...]:
    """Assign final one-based list ranks after exact deduplication."""
    return tuple(
        PrefixCandidate(
            candidate.fixture,
            candidate.span_id,
            candidate.interval_id,
            candidate.frame,
            candidate.timing_score,
            rank,
            candidate.source_flags,
            candidate.heuristic_span_ids,
        )
        for rank, candidate in enumerate(candidates, start=1)
    )


def choose_fixed_action(
    prefix: PrefixRecord,
    nplus_rows: np.ndarray,
) -> PrefixAction:
    """Apply the predeclared heuristic-agreement rule without labels."""
    if prefix.anchor_frame is None or prefix.interval_id is None:
        return PrefixAction("abstain", None, prefix.anchor_frame, prefix.abstention_reason or ABSTENTION_NO_ANCHOR)
    heuristic = next(
        (candidate for candidate in prefix.candidates if candidate.is_filtered_heuristic),
        None,
    )
    if heuristic is None:
        return PrefixAction("abstain", None, prefix.anchor_frame, "no_filtered_heuristic_candidate")
    if heuristic.frame >= prefix.anchor_frame:
        return PrefixAction("abstain", None, prefix.anchor_frame, "heuristic_not_earlier")
    retained = _retained_mask(nplus_rows)
    fixtures = _decode_fixture_rows(nplus_rows)
    same_interval = np.flatnonzero(
        retained
        & (fixtures == prefix.fixture)
        & (nplus_rows["interval_id"] == prefix.interval_id)
    )
    if any(
        abs(heuristic.frame - int(nplus_rows["frame"][index])) <= prefix.radius_frames
        for index in same_interval
    ):
        return PrefixAction("abstain", None, prefix.anchor_frame, "heuristic_within_nplus_radius")
    return PrefixAction("insert", heuristic, prefix.anchor_frame, "heuristic_agreement")


def _abstained_prefix(
    fixture: str,
    span_id: int,
    radius_frames: int,
    reason: str,
) -> PrefixRecord:
    """Create a construction record for an abstained span."""
    action = PrefixAction("abstain", None, None, reason)
    return PrefixRecord(
        fixture,
        span_id,
        None,
        None,
        None,
        None,
        None,
        radius_frames,
        (),
        action,
        reason,
    )


def build_prefix_record(
    fixture: str,
    span_id: int,
    span_start: int,
    span_end: int,
    nplus_rows: np.ndarray,
    score_rows: np.ndarray,
    intervals: tuple[tuple[int, int], ...],
    preceding_span_end: int | None,
    heuristic_rows: Mapping[tuple[str, int, int], tuple[int, ...]],
    *,
    score_indices: Mapping[tuple[str, int, int], int] | None = None,
    score_fixture_names: np.ndarray | None = None,
    score_interval_indices: Mapping[tuple[str, int], np.ndarray] | None = None,
) -> PrefixRecord:
    """Build one frozen prefix record from label-blind rows."""
    radius = scaled_nplus_radius(float(tree_scorer.FIXTURE_SPECS[fixture][1]))
    nplus_fixtures = _decode_fixture_rows(nplus_rows)
    retained = _retained_mask(nplus_rows)
    anchor_indices = np.flatnonzero(
        retained
        & (nplus_fixtures == fixture)
        & (nplus_rows["frame"] >= span_start)
        & (nplus_rows["frame"] < span_end)
    )
    if not len(anchor_indices):
        return _abstained_prefix(fixture, span_id, radius, ABSTENTION_NO_ANCHOR)
    anchor_index = int(anchor_indices[np.argmin(nplus_rows["frame"][anchor_indices])])
    anchor_frame = int(nplus_rows["frame"][anchor_index])
    interval_id, interval_bounds = _interval_for_frame(
        intervals,
        anchor_frame,
        name=f"{fixture}/{span_id} anchor",
    )
    lower, upper = make_prefix_bounds(
        fixture,
        span_id,
        span_start,
        span_end,
        anchor_frame,
        interval_id,
        interval_bounds,
        preceding_span_end,
    )
    if score_indices is None:
        score_indices = _identity_rows(score_rows)
    peak_indices = _peak_indices(
        score_rows,
        fixture,
        interval_id,
        lower,
        upper,
        radius,
        fixture_names=score_fixture_names,
        interval_row_indices=(
            None
            if score_interval_indices is None
            else score_interval_indices.get((fixture, interval_id), np.empty(0, dtype=np.int32))
        ),
    )
    candidates: list[PrefixCandidate] = []
    for peak_rank, index in enumerate(peak_indices, start=1):
        source = (SOURCE_HGB_PEAK_1, SOURCE_HGB_PEAK_2, SOURCE_HGB_PEAK_3)[peak_rank - 1]
        _merge_candidate(
            candidates,
            _candidate_for_index(score_rows, index, fixture, span_id, peak_rank, (source,)),
        )

    heuristic_identities = [
        identity
        for identity in heuristic_rows
        if identity[0] == fixture
        and identity[1] == interval_id
        and lower <= identity[2] <= upper
    ]
    if heuristic_identities:
        heuristic_identity = min(
            heuristic_identities,
            key=lambda identity: (-float(score_rows[score_indices[identity]]["timing_score"]), identity[2]),
        )
        index = score_indices[heuristic_identity]
        _merge_candidate(
            candidates,
            _candidate_for_index(
                score_rows,
                index,
                fixture,
                span_id,
                0,
                (SOURCE_FILTERED_HEURISTIC,),
                heuristic_rows[heuristic_identity],
            ),
        )
    anchor_identity = (fixture, interval_id, anchor_frame)
    if anchor_identity not in score_indices:
        raise ValueError(f"{fixture}/{span_id}/{anchor_frame}: N+ anchor has no exact HGB row")
    _merge_candidate(
        candidates,
        _candidate_for_index(
            score_rows,
            score_indices[anchor_identity],
            fixture,
            span_id,
            0,
            (SOURCE_ANCHOR,),
        ),
    )
    candidates = list(_renumber_candidates(candidates[:MAX_CANDIDATES]))
    record = PrefixRecord(
        fixture,
        span_id,
        interval_id,
        lower,
        upper,
        anchor_frame,
        float(nplus_rows["timing_score"][anchor_index]),
        radius,
        tuple(candidates),
        PrefixAction("abstain", None, anchor_frame, "pending_fixed_rule"),
        None,
    )
    return PrefixRecord(
        record.fixture,
        record.span_id,
        record.interval_id,
        record.lower_frame,
        record.upper_frame,
        record.anchor_frame,
        record.anchor_score,
        record.radius_frames,
        record.candidates,
        choose_fixed_action(record, nplus_rows),
        record.abstention_reason,
    )


def construct_prefixes(
    nplus_rows: np.ndarray,
    score_rows: np.ndarray,
    evidence: Mapping[str, Any],
    search_intervals: Mapping[str, Sequence[Sequence[int]]],
) -> FrozenPrefixSet:
    """Construct all prefixes, candidates and fixed actions without labels."""
    intervals_by_fixture = _normalise_intervals(search_intervals)
    score_indices = _identity_rows(score_rows)
    score_fixture_names = _decode_fixture_rows(score_rows)
    score_interval_indices: dict[tuple[str, int], list[int]] = {}
    for index, (fixture, interval_id) in enumerate(
        zip(score_fixture_names, score_rows["interval_id"], strict=True)
    ):
        score_interval_indices.setdefault((str(fixture), int(interval_id)), []).append(index)
    frozen_interval_indices = {
        identity: np.asarray(indices, dtype=np.int32)
        for identity, indices in score_interval_indices.items()
    }
    heuristic_rows = _filtered_heuristic_rows(evidence, intervals_by_fixture, score_indices, score_rows)
    prefixes: list[PrefixRecord] = []
    candidate_source_count = 0
    candidate_count = 0
    malformed_count = 0
    for fixture in tree_scorer.FIXTURE_SPECS:
        spans = _span_rows(evidence, fixture)
        previous_end: int | None = None
        for span_id, start, end in spans:
            try:
                prefix = build_prefix_record(
                    fixture,
                    span_id,
                    start,
                    end,
                    nplus_rows,
                    score_rows,
                    intervals_by_fixture[fixture],
                    previous_end,
                    heuristic_rows,
                    score_indices=score_indices,
                    score_fixture_names=score_fixture_names,
                    score_interval_indices=frozen_interval_indices,
                )
            except ValueError as error:
                # The record is retained for structural diagnostics.  The
                # real-data gate below rejects this state before labels load.
                prefix = _abstained_prefix(fixture, span_id, scaled_nplus_radius(tree_scorer.FIXTURE_SPECS[fixture][1]), ABSTENTION_MALFORMED_BOUNDS)
                malformed_count += 1
                prefix = PrefixRecord(
                    prefix.fixture,
                    prefix.span_id,
                    prefix.interval_id,
                    prefix.lower_frame,
                    prefix.upper_frame,
                    prefix.anchor_frame,
                    prefix.anchor_score,
                    prefix.radius_frames,
                    prefix.candidates,
                    PrefixAction("abstain", None, None, f"{ABSTENTION_MALFORMED_BOUNDS}: {error}"),
                    ABSTENTION_MALFORMED_BOUNDS,
                )
            prefixes.append(prefix)
            candidate_source_count += sum(len(candidate.source_flags) for candidate in prefix.candidates)
            candidate_count += len(prefix.candidates)
            previous_end = end
    return FrozenPrefixSet(
        tuple(prefixes),
        nplus_rows,
        int(np.count_nonzero(_retained_mask(nplus_rows))),
        candidate_source_count,
        candidate_count,
        candidate_source_count - candidate_count,
        malformed_count,
    )


def validate_real_construction(frozen: FrozenPrefixSet) -> None:
    """Apply the label-blind real-data gate before any labels are imported."""
    if frozen.nplus_event_count != EXPECTED_NPLUS_EVENT_COUNT:
        raise ValueError(
            f"N+ baseline event count differs: {frozen.nplus_event_count} != {EXPECTED_NPLUS_EVENT_COUNT}"
        )
    if len(frozen.prefixes) != EXPECTED_SPAN_COUNT:
        raise ValueError(f"detected span count differs: {len(frozen.prefixes)} != {EXPECTED_SPAN_COUNT}")
    if frozen.malformed_count:
        raise ValueError(f"real construction has {frozen.malformed_count} malformed prefix bounds")
    for prefix in frozen.prefixes:
        if prefix.anchor_frame is not None and not prefix.candidates:
            raise ValueError(f"{prefix.fixture}/{prefix.span_id}: anchored prefix has no candidates")


def _candidate_json(candidate: PrefixCandidate) -> dict[str, Any]:
    """Serialise one candidate with every frozen identity field."""
    return {
        "fixture": candidate.fixture,
        "span_id": candidate.span_id,
        "interval_id": candidate.interval_id,
        "frame": candidate.frame,
        "timing_score": candidate.timing_score,
        "rank": candidate.rank,
        "source_flags": list(candidate.source_flags),
        "heuristic_span_ids": list(candidate.heuristic_span_ids),
    }


def _action_json(action: PrefixAction) -> dict[str, Any]:
    """Serialise a fixed action and its optional candidate."""
    return {
        "action": action.action,
        "anchor_frame": action.anchor_frame,
        "reason": action.reason,
        "candidate": None if action.candidate is None else _candidate_json(action.candidate),
    }


def _output_span_json(bounds: ServeOutputSpanBounds) -> dict[str, Any]:
    """Serialise one detected span and its final output bounds."""
    return {
        "fixture": bounds.fixture,
        "span_id": bounds.span_id,
        "detected_span_start": bounds.detected_span_start,
        "detected_span_end": bounds.detected_span_end,
        "serve_prepend_frame": bounds.serve_prepend_frame,
        "output_span_start": bounds.output_span_start,
        "output_span_end": bounds.output_span_end,
        "output_start_source": bounds.output_start_source,
    }


def construction_payload(frozen: FrozenPrefixSet) -> dict[str, Any]:
    """Return a deterministic, label-free construction payload."""
    return {
        "schema": CONSTRUCTION_SCHEMA,
        "labels_read": False,
        "nplus_event_count": frozen.nplus_event_count,
        "candidate_source_count": frozen.candidate_source_count,
        "candidate_count": frozen.candidate_count,
        "exact_deduplicated_candidates": frozen.exact_deduplication_count,
        "malformed_count": frozen.malformed_count,
        "prefixes": [
            {
                "fixture": prefix.fixture,
                "span_id": prefix.span_id,
                "interval_id": prefix.interval_id,
                "lower_frame": prefix.lower_frame,
                "upper_frame": prefix.upper_frame,
                "prefix_length": prefix.prefix_length,
                "anchor_frame": prefix.anchor_frame,
                "anchor_score": prefix.anchor_score,
                "radius_frames": prefix.radius_frames,
                "abstention_reason": prefix.abstention_reason,
                "candidates": [_candidate_json(candidate) for candidate in prefix.candidates],
                "fixed_action": _action_json(prefix.fixed_action),
            }
            for prefix in frozen.prefixes
        ],
    }


def deterministic_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Encode JSON with stable ordering and a final newline."""
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def freeze_construction_twice(
    nplus_rows: np.ndarray,
    score_rows: np.ndarray,
    evidence: Mapping[str, Any],
    search_intervals: Mapping[str, Sequence[Sequence[int]]],
) -> tuple[FrozenPrefixSet, bytes]:
    """Construct twice and require byte-identical label-blind JSON."""
    first = construct_prefixes(nplus_rows, score_rows, evidence, search_intervals)
    second = construct_prefixes(nplus_rows, score_rows, evidence, search_intervals)
    first_bytes = deterministic_json_bytes(construction_payload(first))
    second_bytes = deterministic_json_bytes(construction_payload(second))
    if first_bytes != second_bytes:
        raise ValueError("label-blind prefix construction is not byte-identical")
    return first, first_bytes


def load_label_blind_inputs(arguments: argparse.Namespace) -> VerifiedLabelBlindInputs:
    """Verify every committed input and replay N+ before labels are loaded."""
    features = tree_scorer.verify_freeze(arguments.feature_manifest)
    candidates = tree_scorer.verify_candidate_scores(
        arguments.candidate_manifest,
        features,
        arguments.tree_results,
    )
    if tree_scorer._result_variant(candidates.tree_result) != tree_scorer.CANDIDATE_VARIANT:
        raise ValueError("serve-prefix experiment requires retained baseline HGB physics scores")
    evidence = evidence_scorer.verify_freeze(arguments.evidence_manifest)
    model_rows = decision_scorer._model_rows(features)
    rows_by_trial = decision_scorer.replay_all_trials(candidates.rows, model_rows)
    nplus_rows = rows_by_trial[SELECTED_TRIAL_ID]
    search_intervals = tree_scorer._manifest_intervals(features.manifest, "search_intervals")
    return VerifiedLabelBlindInputs(features, candidates, evidence, nplus_rows, model_rows, search_intervals)


def _candidate_event(
    candidate: PrefixCandidate,
    attribution: Mapping[tuple[str, int], str | None],
) -> rally_scorer.FixedEvent:
    """Convert a frozen candidate and its replayed side into a strict event."""
    if (candidate.fixture, candidate.frame) not in attribution:
        raise ValueError(f"missing replayed side prediction for {candidate.fixture}/{candidate.frame}")
    side = rally_scorer._normalise_half(
        attribution[(candidate.fixture, candidate.frame)],
        f"{candidate.fixture}/{candidate.frame} attribution",
    )
    return rally_scorer.FixedEvent(candidate.fixture, candidate.frame, candidate.timing_score, side)


def apply_insert_with_local_dedup(
    events: Sequence[rally_scorer.FixedEvent],
    candidate: PrefixCandidate,
    attribution: Mapping[tuple[str, int], str | None],
    anchor_frame: int,
    radius_frames: int,
) -> tuple[rally_scorer.FixedEvent, ...]:
    """Insert a candidate while replacing only its local anchor when nearby."""
    if radius_frames < 0:
        raise ValueError("local deduplication radius must be non-negative")
    candidate_event = _candidate_event(candidate, attribution)
    updated = list(events)
    anchor_indices = [index for index, event in enumerate(updated) if event.frame == anchor_frame]
    if abs(candidate.frame - anchor_frame) <= radius_frames:
        if len(anchor_indices) != 1:
            raise ValueError(f"expected one anchor event at frame {anchor_frame}")
        updated[anchor_indices[0]] = candidate_event
    else:
        duplicate_indices = [index for index, event in enumerate(updated) if event.frame == candidate.frame]
        if len(duplicate_indices) > 1:
            raise ValueError(f"duplicate event frame {candidate.frame}")
        if duplicate_indices:
            updated[duplicate_indices[0]] = candidate_event
        else:
            updated.append(candidate_event)
    return tuple(sorted(updated, key=lambda event: event.frame))


def _actions_by_span(prefixes: Sequence[PrefixRecord]) -> dict[tuple[str, int], PrefixAction]:
    """Index one frozen action per fixture and detected span."""
    output: dict[tuple[str, int], PrefixAction] = {}
    for prefix in prefixes:
        identity = (prefix.fixture, prefix.span_id)
        if identity in output:
            raise ValueError(f"duplicate prefix action identity: {identity}")
        output[identity] = prefix.fixed_action
    return output


def apply_actions_to_stream(
    events_by_fixture: Mapping[str, Sequence[rally_scorer.FixedEvent]],
    prefixes: Sequence[PrefixRecord],
    attribution: Mapping[tuple[str, int], str | None],
) -> dict[str, tuple[rally_scorer.FixedEvent, ...]]:
    """Apply selected actions while retaining all later N+ events."""
    output = {fixture: list(events) for fixture, events in events_by_fixture.items()}
    selected_identities: set[tuple[str, int]] = set()
    for prefix in prefixes:
        action = prefix.fixed_action
        if action.candidate is None:
            continue
        if prefix.anchor_frame is None:
            raise ValueError(f"{prefix.fixture}/{prefix.span_id}: selected action has no anchor")
        identity = (action.candidate.fixture, action.candidate.frame)
        if identity in selected_identities:
            raise ValueError(f"candidate selected for more than one span: {identity}")
        selected_identities.add(identity)
        expected_action = (
            "replace"
            if abs(action.candidate.frame - prefix.anchor_frame) <= prefix.radius_frames
            else "insert"
        )
        if action.action != expected_action:
            raise ValueError(
                f"{prefix.fixture}/{prefix.span_id}: action {action.action!r} "
                f"does not match {expected_action!r}"
            )
        current = output.setdefault(prefix.fixture, [])
        output[prefix.fixture] = list(
            apply_insert_with_local_dedup(
                current,
                action.candidate,
                attribution,
                prefix.anchor_frame,
                prefix.radius_frames,
            )
        )
    return {
        fixture: tuple(sorted(events, key=lambda event: event.frame))
        for fixture, events in output.items()
    }


def attach_actions_to_spans(
    evidence: Mapping[str, Any],
    events_by_fixture: Mapping[str, Sequence[rally_scorer.FixedEvent]],
    prefixes: Sequence[PrefixRecord],
) -> AttachedSpanSet:
    """Build output spans from final events after choosing each output start."""
    base_spans = rally_scorer.fixed_spans_from_evidence(evidence, events_by_fixture)
    actions = _actions_by_span(prefixes)
    attached: list[rally_scorer.FixedSpan] = []
    output_bounds: list[ServeOutputSpanBounds] = []
    for span in base_spans:
        action = actions[(span.fixture, span.span_id)]
        candidate = action.candidate
        serve_prepend_frame = (
            candidate.frame
            if candidate is not None and candidate.frame < span.start_frame
            else None
        )
        output_span_start = (
            span.start_frame
            if serve_prepend_frame is None
            else serve_prepend_frame
        )
        output_start_source = (
            OUTPUT_START_DETECTED_SPAN
            if serve_prepend_frame is None
            else OUTPUT_START_SERVE_PREPEND
        )
        bounds = ServeOutputSpanBounds(
            fixture=span.fixture,
            span_id=span.span_id,
            detected_span_start=span.start_frame,
            detected_span_end=span.end_frame,
            serve_prepend_frame=serve_prepend_frame,
            output_span_start=output_span_start,
            output_span_end=span.end_frame,
            output_start_source=output_start_source,
        )
        output_bounds.append(bounds)
        fixture_events = events_by_fixture.get(span.fixture, ())
        events = tuple(
            event
            for event in fixture_events
            if bounds.output_span_start <= event.frame < bounds.output_span_end
        )
        attached.append(
            rally_scorer.FixedSpan(
                span.fixture,
                span.span_id,
                bounds.output_span_start,
                bounds.output_span_end,
                tuple(sorted(events, key=lambda event: event.frame)),
            )
        )
    return AttachedSpanSet(tuple(attached), tuple(output_bounds))


def candidate_prediction_frames(
    nplus_rows: np.ndarray,
    prefixes: Sequence[PrefixRecord],
) -> dict[str, np.ndarray]:
    """Return the union of baseline N+ and every frozen candidate frame."""
    retained = _retained_mask(nplus_rows)
    names = _decode_fixture_rows(nplus_rows)
    frames: dict[str, set[int]] = {fixture: set() for fixture in tree_scorer.FIXTURE_SPECS}
    for fixture, frame in zip(names[retained], nplus_rows["frame"][retained], strict=True):
        frames[str(fixture)].add(int(frame))
    for prefix in prefixes:
        for candidate in prefix.candidates:
            frames[candidate.fixture].add(candidate.frame)
    return {
        fixture: np.asarray(sorted(values), dtype=np.int32)
        for fixture, values in frames.items()
    }


def replay_candidate_sides(
    arguments: argparse.Namespace,
    nplus_rows: np.ndarray,
    prefixes: Sequence[PrefixRecord],
) -> dict[tuple[str, int], str | None]:
    """Replay shipped Top/Bottom predictions before either label table loads."""
    frames = candidate_prediction_frames(nplus_rows, prefixes)
    expected = {
        (fixture, int(frame))
        for fixture, fixture_frames in frames.items()
        for frame in fixture_frames
    }
    freeze_arguments = argparse.Namespace(
        region_v2_manifest=arguments.feature_manifest,
        region_v2_results=arguments.tree_results,
        region_v1_manifest=arguments.region_v1_manifest,
        region_v1_results=arguments.region_v1_results,
    )
    freezes = attribution_scorer._load_tree_freezes(freeze_arguments)
    variants = {"region_v2/serve_prefix": frames}
    attribution = attribution_scorer._shipped_attribution_map(arguments.data_root, freezes, variants)
    if set(attribution) != expected:
        raise ValueError("Top/Bottom replay does not cover every baseline or candidate frame")
    return attribution


def _span_rally_candidates(
    span: rally_scorer.FixedSpan,
    rallies: Sequence[rally_scorer.RallyReference],
) -> tuple[rally_scorer.RallyReference, ...]:
    """Return rallies that overlap a frozen detected span."""
    return tuple(
        rally
        for rally in rallies
        if any(span.start_frame <= frame < span.end_frame for frame in rally.frames)
    )


def choose_timing_oracle_action(
    prefix: PrefixRecord,
    rallies: Sequence[rally_scorer.RallyReference],
    tolerance_frames: int,
    *,
    span_start: int | None = None,
    span_end: int | None = None,
    baseline_event_frames: Sequence[int] = (),
) -> PrefixAction:
    """Choose an earlier candidate only when the baseline misses the serve."""
    if prefix.anchor_frame is None:
        return PrefixAction("abstain", None, None, ABSTENTION_NO_ANCHOR)
    if tolerance_frames < 0:
        raise ValueError("oracle tolerance must be non-negative")
    start = prefix.lower_frame if span_start is None else span_start
    end = prefix.upper_frame + 1 if span_end is None else span_end
    if start is None or end is None:
        return PrefixAction("abstain", None, prefix.anchor_frame, "missing_span_bounds")
    candidates = tuple(
        rally
        for rally in rallies
        if any(start <= frame < end for frame in rally.frames)
    )
    if len(candidates) != 1:
        reason = "no_rally" if not candidates else "multiple_rallies"
        return PrefixAction("abstain", None, prefix.anchor_frame, reason)
    rally = candidates[0]
    baseline_matches = tree_scorer._greedy_matches(
        np.asarray(rally.frames, dtype=np.int32),
        np.asarray(baseline_event_frames, dtype=np.int32),
        tolerance_frames,
    )
    if any(gt_index == 0 for gt_index, _prediction_index, _offset in baseline_matches):
        return PrefixAction("abstain", None, prefix.anchor_frame, "serve_already_matched")
    target_frame = rally.frames[0]
    earlier = [candidate for candidate in prefix.candidates if candidate.frame < prefix.anchor_frame]
    eligible = [candidate for candidate in earlier if abs(candidate.frame - target_frame) <= tolerance_frames]
    if not eligible:
        return PrefixAction("abstain", None, prefix.anchor_frame, "no_candidate_within_tolerance")
    selected = min(eligible, key=lambda candidate: (abs(candidate.frame - target_frame), -candidate.timing_score, candidate.frame))
    action = "replace" if abs(selected.frame - prefix.anchor_frame) <= prefix.radius_frames else "insert"
    return PrefixAction(action, selected, prefix.anchor_frame, "timing_oracle")


def _oracle_prefixes(
    frozen: FrozenPrefixSet,
    baseline_spans: Sequence[rally_scorer.FixedSpan],
    rallies_by_fixture: Mapping[str, Sequence[rally_scorer.RallyReference]],
    evidence: Mapping[str, Any],
) -> tuple[PrefixRecord, ...]:
    """Return copies of prefixes whose action is the timing oracle choice."""
    spans_by_fixture = {
        fixture: _span_rows(evidence, fixture)
        for fixture in tree_scorer.FIXTURE_SPECS
    }
    baseline_by_identity = {
        (span.fixture, span.span_id): span
        for span in baseline_spans
    }
    output: list[PrefixRecord] = []
    for prefix in frozen.prefixes:
        span_rows = {span_id: (start, end) for span_id, start, end in spans_by_fixture[prefix.fixture]}
        span_start, span_end = span_rows[prefix.span_id][0:2]
        tolerance = tree_scorer._scaled_frames(
            PRIMARY_TOLERANCE_BASE30,
            tree_scorer.FIXTURE_SPECS[prefix.fixture][1],
        )
        action = choose_timing_oracle_action(
            prefix,
            rallies_by_fixture.get(prefix.fixture, ()),
            tolerance,
            span_start=span_start,
            span_end=span_end,
            baseline_event_frames=tuple(
                event.frame
                for event in baseline_by_identity[(prefix.fixture, prefix.span_id)].events
            ),
        )
        output.append(
            PrefixRecord(
                prefix.fixture,
                prefix.span_id,
                prefix.interval_id,
                prefix.lower_frame,
                prefix.upper_frame,
                prefix.anchor_frame,
                prefix.anchor_score,
                prefix.radius_frames,
                prefix.candidates,
                action,
                prefix.abstention_reason,
            )
        )
    return tuple(output)


def _ground_truth_from_rallies(
    rallies_by_fixture: Mapping[str, Sequence[rally_scorer.RallyReference]],
) -> tree_scorer.GroundTruth:
    """Build production event ground truth from the already loaded timing table."""
    frames = {
        fixture: np.asarray(
            [frame for rally in rallies_by_fixture[fixture] for frame in rally.frames],
            dtype=np.int32,
        )
        for fixture in tree_scorer.FIXTURE_SPECS
    }
    serves = {
        fixture: {rally.frames[0] for rally in rallies_by_fixture[fixture]}
        for fixture in tree_scorer.FIXTURE_SPECS
    }
    return tree_scorer.GroundTruth(
        frames,
        serves,
        sum(len(rallies_by_fixture[fixture]) for fixture in tree_scorer.FIXTURE_SPECS),
    )


def _matched_identities(
    ground_truth: tree_scorer.GroundTruth,
    predictions: Mapping[str, np.ndarray],
    tolerance_base30: int,
) -> tuple[
    set[tuple[str, int]],
    set[tuple[str, int]],
    set[tuple[str, int]],
    set[tuple[str, int]],
]:
    """Return matched contact, serve, prediction and unmatched prediction identities."""
    matched_contacts: set[tuple[str, int]] = set()
    matched_serves: set[tuple[str, int]] = set()
    matched_predictions: set[tuple[str, int]] = set()
    all_predictions: set[tuple[str, int]] = set()
    for fixture in tree_scorer.FIXTURE_SPECS:
        tolerance = tree_scorer._scaled_frames(tolerance_base30, tree_scorer.FIXTURE_SPECS[fixture][1])
        fixture_predictions = predictions.get(fixture, np.empty(0, dtype=np.int32))
        matches = tree_scorer._greedy_matches(
            ground_truth.frames[fixture],
            fixture_predictions,
            tolerance,
        )
        all_predictions.update((fixture, int(frame)) for frame in fixture_predictions)
        for gt_index, prediction_index, _offset in matches:
            gt_frame = int(ground_truth.frames[fixture][gt_index])
            contact_identity = (fixture, gt_frame)
            matched_contacts.add(contact_identity)
            matched_predictions.add((fixture, int(fixture_predictions[prediction_index])))
            if gt_frame in ground_truth.serves[fixture]:
                matched_serves.add(contact_identity)
    return (
        matched_contacts,
        matched_serves,
        matched_predictions,
        all_predictions - matched_predictions,
    )


def _event_comparison(
    ground_truth: tree_scorer.GroundTruth,
    baseline: Mapping[str, np.ndarray],
    alternative: Mapping[str, np.ndarray],
    tolerance_base30: int,
) -> dict[str, Any]:
    """Report production timing metrics and matched identity changes."""
    base_metrics = tree_scorer._event_counts(ground_truth, baseline, tolerance_base30)
    alternative_metrics = tree_scorer._event_counts(ground_truth, alternative, tolerance_base30)
    base_contacts, base_serves, base_predictions, base_unmatched = _matched_identities(
        ground_truth, baseline, tolerance_base30
    )
    alt_contacts, alt_serves, alt_predictions, alt_unmatched = _matched_identities(
        ground_truth, alternative, tolerance_base30
    )
    baseline_frame_sets = {
        fixture: {int(value) for value in baseline.get(fixture, ())}
        for fixture in tree_scorer.FIXTURE_SPECS
    }
    added_frames = {
        (fixture, int(frame))
        for fixture in tree_scorer.FIXTURE_SPECS
        for frame in alternative.get(fixture, np.empty(0, dtype=np.int32))
        if int(frame) not in baseline_frame_sets[fixture]
    }
    near_existing: list[tuple[str, int]] = []
    for fixture, frame in added_frames:
        radius = tree_scorer._scaled_frames(tolerance_base30, tree_scorer.FIXTURE_SPECS[fixture][1])
        if any(abs(frame - int(existing)) <= radius for existing in baseline.get(fixture, ())):
            near_existing.append((fixture, frame))
    added_unmatched = alt_unmatched - base_unmatched
    return {
        "baseline": base_metrics,
        "alternative": alternative_metrics,
        "newly_matched_contact_identities": sorted(alt_contacts - base_contacts),
        "lost_contact_identities": sorted(base_contacts - alt_contacts),
        "newly_matched_serve_identities": sorted(alt_serves - base_serves),
        "lost_serve_identities": sorted(base_serves - alt_serves),
        "newly_matched_prediction_identities": sorted(alt_predictions - base_predictions),
        "lost_matched_prediction_identities": sorted(base_predictions - alt_predictions),
        "added_unmatched_event_identities": sorted(added_unmatched),
        "added_unmatched_event_count": len(added_unmatched),
        "added_event_count": len(added_frames),
        "added_events_within_tolerance_of_baseline": sorted(near_existing),
    }


def _strict_scores_at_requirement(
    spans: Sequence[rally_scorer.FixedSpan],
    rallies_by_fixture: Mapping[str, Sequence[rally_scorer.RallyReference]],
    target_sides: Mapping[tuple[str, int], str],
    tolerance_base30: int,
    requirement: float,
) -> dict[tuple[str, int], rally_scorer.SpanScore]:
    """Return strict scores keyed by stable span identity."""
    output: dict[tuple[str, int], rally_scorer.SpanScore] = {}
    for span in spans:
        tolerance = tree_scorer._scaled_frames(
            tolerance_base30,
            tree_scorer.FIXTURE_SPECS[span.fixture][1],
        )
        result = rally_scorer.evaluate_span(
            span,
            rallies_by_fixture.get(span.fixture, ()),
            target_sides,
            tolerance,
            requirement,
        )
        identity = (span.fixture, span.span_id)
        if identity in output:
            raise ValueError(f"duplicate strict span identity: {identity}")
        output[identity] = result
    return output


def _strict_comparison(
    baseline_spans: Sequence[rally_scorer.FixedSpan],
    alternative_spans: Sequence[rally_scorer.FixedSpan],
    rallies_by_fixture: Mapping[str, Sequence[rally_scorer.RallyReference]],
    target_sides: Mapping[tuple[str, int], str],
    fps_by_fixture: Mapping[str, float],
    tolerance_base30: int,
) -> dict[str, Any]:
    """Return full confidence curves and fully-correct identity deltas."""
    baseline = rally_scorer.score_strict_rallies(
        baseline_spans,
        rallies_by_fixture,
        target_sides,
        fps_by_fixture,
        tolerance_base30=tolerance_base30,
    )
    alternative = rally_scorer.score_strict_rallies(
        alternative_spans,
        rallies_by_fixture,
        target_sides,
        fps_by_fixture,
        tolerance_base30=tolerance_base30,
    )
    identity_deltas: dict[str, Any] = {}
    for requirement in (0.0, 0.9):
        baseline_scores = _strict_scores_at_requirement(
            baseline_spans,
            rallies_by_fixture,
            target_sides,
            tolerance_base30,
            requirement,
        )
        alternative_scores = _strict_scores_at_requirement(
            alternative_spans,
            rallies_by_fixture,
            target_sides,
            tolerance_base30,
            requirement,
        )
        baseline_ids = {
            identity for identity, score in baseline_scores.items() if score.fully_correct
        }
        alternative_ids = {
            identity for identity, score in alternative_scores.items() if score.fully_correct
        }
        baseline_curve_row = next(
            row for row in baseline["confidence_curve"] if row["confidence_requirement"] == requirement
        )
        alternative_curve_row = next(
            row for row in alternative["confidence_curve"] if row["confidence_requirement"] == requirement
        )
        if len(baseline_ids) != baseline_curve_row["fully_correct_kept_rallies"]:
            raise ValueError("baseline fully-correct identities differ from the confidence curve")
        if len(alternative_ids) != alternative_curve_row["fully_correct_kept_rallies"]:
            raise ValueError("alternative fully-correct identities differ from the confidence curve")
        baseline_reasons = Counter(
            reason
            for score in baseline_scores.values()
            for reason in score.rejection_reasons
        )
        alternative_reasons = Counter(
            reason
            for score in alternative_scores.values()
            for reason in score.rejection_reasons
        )
        changed_reasons = [
            {
                "fixture": fixture,
                "span_id": span_id,
                "baseline": list(baseline_scores[(fixture, span_id)].rejection_reasons),
                "alternative": list(alternative_scores[(fixture, span_id)].rejection_reasons),
            }
            for fixture, span_id in sorted(baseline_scores)
            if baseline_scores[(fixture, span_id)].rejection_reasons
            != alternative_scores[(fixture, span_id)].rejection_reasons
        ]
        identity_deltas[f"{requirement:.2f}"] = {
            "newly_fully_correct_span_identities": sorted(alternative_ids - baseline_ids),
            "lost_fully_correct_span_identities": sorted(baseline_ids - alternative_ids),
            "net_fully_correct_change": len(alternative_ids) - len(baseline_ids),
            "rejection_reason_counts": {
                "baseline": dict(sorted(baseline_reasons.items())),
                "alternative": dict(sorted(alternative_reasons.items())),
            },
            "changed_rejection_reasons": changed_reasons,
        }
    return {
        "tolerance_base30": tolerance_base30,
        "baseline": baseline,
        "alternative": alternative,
        "fully_correct_identity_changes": identity_deltas,
        "baseline_at_zero": next(
            row for row in baseline["confidence_curve"] if row["confidence_requirement"] == 0.0
        ),
        "alternative_at_zero": next(
            row for row in alternative["confidence_curve"] if row["confidence_requirement"] == 0.0
        ),
        "baseline_at_0.90": next(
            row for row in baseline["confidence_curve"] if row["confidence_requirement"] == 0.9
        ),
        "alternative_at_0.90": next(
            row for row in alternative["confidence_curve"] if row["confidence_requirement"] == 0.9
        ),
    }


def _prediction_frames_from_events(
    events_by_fixture: Mapping[str, Sequence[rally_scorer.FixedEvent]],
) -> dict[str, np.ndarray]:
    """Convert fixed event objects to sorted frame arrays."""
    return {
        fixture: np.asarray(
            sorted(event.frame for event in events_by_fixture.get(fixture, ())),
            dtype=np.int32,
        )
        for fixture in tree_scorer.FIXTURE_SPECS
    }


def _structural_report(
    frozen: FrozenPrefixSet,
    spans: Sequence[rally_scorer.FixedSpan],
    rallies_by_fixture: Mapping[str, Sequence[rally_scorer.RallyReference]],
    ground_truth: tree_scorer.GroundTruth,
) -> dict[str, Any]:
    """Summarise span coverage and label-blind candidate structure."""
    prefix_by_identity = {
        (prefix.fixture, prefix.span_id): prefix
        for prefix in frozen.prefixes
    }
    mapped_by_identity = {
        (span.fixture, span.span_id): _span_rally_candidates(
            span, rallies_by_fixture.get(span.fixture, ())
        )
        for span in spans
    }
    mapped_counts = [len(rallies) for rallies in mapped_by_identity.values()]
    mapped_rallies = {
        (span.fixture, rally.rally_index)
        for span in spans
        for rally in mapped_by_identity[(span.fixture, span.span_id)]
    }
    all_rallies = {
        (fixture, rally.rally_index)
        for fixture, rallies in rallies_by_fixture.items()
        for rally in rallies
    }
    anchored = [prefix for prefix in frozen.prefixes if prefix.anchor_frame is not None]
    source_overlap = sum(len(set(candidate.source_flags)) > 1 for prefix in frozen.prefixes for candidate in prefix.candidates)
    candidate_counts = [len(prefix.candidates) for prefix in frozen.prefixes]
    labelled_contacts_per_prefix: dict[tuple[str, int], int] = {}
    for prefix in frozen.prefixes:
        if prefix.lower_frame is None or prefix.upper_frame is None:
            labelled_contacts_per_prefix[(prefix.fixture, prefix.span_id)] = 0
            continue
        labelled_contacts_per_prefix[(prefix.fixture, prefix.span_id)] = sum(
            prefix.lower_frame <= int(frame) <= prefix.upper_frame
            for frame in ground_truth.frames[prefix.fixture]
        )

    baseline_predictions = {
        fixture: np.asarray(
            sorted(
                int(frame)
                for frame in frozen.nplus_rows["frame"][
                    _retained_mask(frozen.nplus_rows)
                    & (_decode_fixture_rows(frozen.nplus_rows) == fixture)
                ]
            ),
            dtype=np.int32,
        )
        for fixture in tree_scorer.FIXTURE_SPECS
    }
    _contacts, matched_serves, _predictions, _unmatched = _matched_identities(
        ground_truth,
        baseline_predictions,
        PRIMARY_TOLERANCE_BASE30,
    )
    missed_serves = {
        (fixture, int(frame))
        for fixture, frames in ground_truth.serves.items()
        for frame in frames
    } - matched_serves
    addressable: list[dict[str, Any]] = []
    for span in spans:
        identity = (span.fixture, span.span_id)
        prefix = prefix_by_identity[identity]
        rallies = mapped_by_identity[identity]
        if prefix.anchor_frame is None or len(rallies) != 1:
            continue
        serve_frame = rallies[0].frames[0]
        if (span.fixture, serve_frame) not in missed_serves:
            continue
        tolerance = tree_scorer._scaled_frames(
            PRIMARY_TOLERANCE_BASE30,
            tree_scorer.FIXTURE_SPECS[span.fixture][1],
        )
        candidates = [
            candidate
            for candidate in prefix.candidates
            if abs(candidate.frame - serve_frame) <= tolerance
        ]
        if candidates:
            addressable.append(
                {
                    "fixture": span.fixture,
                    "span_id": span.span_id,
                    "rally_id": rallies[0].rally_id,
                    "serve_frame": serve_frame,
                    "candidate_ranks": [candidate.rank for candidate in candidates],
                    "candidate_frames": [candidate.frame for candidate in candidates],
                }
            )

    fixtures: dict[str, dict[str, Any]] = {}
    for fixture in tree_scorer.FIXTURE_SPECS:
        fixture_spans = [span for span in spans if span.fixture == fixture]
        fixture_prefixes = [prefix for prefix in frozen.prefixes if prefix.fixture == fixture]
        fixture_mapped = [mapped_by_identity[(span.fixture, span.span_id)] for span in fixture_spans]
        fixture_mapped_rallies = {
            rally.rally_index
            for rallies in fixture_mapped
            for rally in rallies
        }
        fixture_addressable = [row for row in addressable if row["fixture"] == fixture]
        fixtures[fixture] = {
            "real_rallies": len(rallies_by_fixture[fixture]),
            "detected_spans": len(fixture_spans),
            "spans_mapping_to_zero_rallies": sum(not rallies for rallies in fixture_mapped),
            "spans_mapping_to_one_rally": sum(len(rallies) == 1 for rallies in fixture_mapped),
            "spans_mapping_to_multiple_rallies": sum(len(rallies) > 1 for rallies in fixture_mapped),
            "real_rallies_outside_every_detected_span": len(rallies_by_fixture[fixture])
            - len(fixture_mapped_rallies),
            "anchored_spans": sum(prefix.anchor_frame is not None for prefix in fixture_prefixes),
            "unanchored_spans": sum(prefix.anchor_frame is None for prefix in fixture_prefixes),
            "addressable_missed_serves": len(fixture_addressable),
        }
    return {
        "real_rallies": ground_truth.rally_count,
        "real_contacts": int(sum(len(values) for values in ground_truth.frames.values())),
        "detected_spans": len(spans),
        "spans_mapping_to_zero_rallies": mapped_counts.count(0),
        "spans_mapping_to_one_rally": mapped_counts.count(1),
        "spans_mapping_to_multiple_rallies": sum(count > 1 for count in mapped_counts),
        "real_rallies_outside_every_detected_span": len(all_rallies - mapped_rallies),
        "anchored_spans": len(anchored),
        "unanchored_spans": len(frozen.prefixes) - len(anchored),
        "prefix_length": {
            "count": len([prefix.prefix_length for prefix in frozen.prefixes if prefix.prefix_length is not None]),
            "values": [prefix.prefix_length for prefix in frozen.prefixes if prefix.prefix_length is not None],
        },
        "candidate_count": {"values": candidate_counts},
        "candidate_source_overlap_count": source_overlap,
        "exact_candidate_deduplication_count": frozen.exact_deduplication_count,
        "prefixes_with_more_than_one_labelled_contact": sum(
            count > 1 for count in labelled_contacts_per_prefix.values()
        ),
        "missed_serve_count": len(missed_serves),
        "addressable_missed_serve_count": len(addressable),
        "addressable_fraction_of_missed_serves": len(addressable) / len(missed_serves),
        "addressable_missed_serves": addressable,
        "fixtures": fixtures,
    }


def _prospective_gate_report(
    structural: Mapping[str, Any],
    event_reports: Mapping[str, Any],
    strict_reports: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the development stop rules frozen in the runbook."""
    oracle_primary = strict_reports["oracle"][str(PRIMARY_TOLERANCE_BASE30)]
    fixed_primary = strict_reports["fixed"][str(PRIMARY_TOLERANCE_BASE30)]
    oracle_zero = oracle_primary["fully_correct_identity_changes"]["0.00"]
    fixed_zero = fixed_primary["fully_correct_identity_changes"]["0.00"]
    fixed_ninety = fixed_primary["fully_correct_identity_changes"]["0.90"]
    fixed_zero_new = len(fixed_zero["newly_fully_correct_span_identities"])
    fixed_zero_lost = len(fixed_zero["lost_fully_correct_span_identities"])
    fixed_ninety_lost = len(fixed_ninety["lost_fully_correct_span_identities"])
    oracle_new_serves = len(
        event_reports["oracle"][str(PRIMARY_TOLERANCE_BASE30)]["newly_matched_serve_identities"]
    )
    fixed_new_serves = len(
        event_reports["fixed"][str(PRIMARY_TOLERANCE_BASE30)]["newly_matched_serve_identities"]
    )
    serve_recovery_fraction = fixed_new_serves / oracle_new_serves if oracle_new_serves else 0.0

    opportunity_fraction_pass = structural["addressable_fraction_of_missed_serves"] >= 0.10
    fixture_opportunity_pass = all(
        fixture["addressable_missed_serves"] >= 2
        for fixture in structural["fixtures"].values()
    )
    oracle_rally_pass = (
        oracle_zero["net_fully_correct_change"] >= 2
        and not oracle_zero["lost_fully_correct_span_identities"]
    )
    headroom_checks = {
        "at_least_ten_percent_of_missed_serves_have_a_candidate": opportunity_fraction_pass,
        "at_least_two_candidate_opportunities_per_fixture": fixture_opportunity_pass,
        "oracle_adds_two_fully_correct_rallies_without_loss_at_zero": oracle_rally_pass,
    }

    fixed_zero_accuracy_pass = (
        fixed_primary["alternative_at_zero"]["kept_rally_accuracy"]
        >= fixed_primary["baseline_at_zero"]["kept_rally_accuracy"]
    )
    fixed_ninety_accuracy_pass = (
        fixed_primary["alternative_at_0.90"]["kept_rally_accuracy"]
        >= fixed_primary["baseline_at_0.90"]["kept_rally_accuracy"]
    )
    fixed_checks = {
        "at_least_one_new_fully_correct_rally_at_zero": fixed_zero_new >= 1,
        "no_fully_correct_rally_lost_at_zero": fixed_zero_lost == 0,
        "no_fully_correct_rally_lost_at_0.90": fixed_ninety_lost == 0,
        "kept_rally_accuracy_does_not_fall_at_zero": fixed_zero_accuracy_pass,
        "kept_rally_accuracy_does_not_fall_at_0.90": fixed_ninety_accuracy_pass,
        "no_fixture_loses_a_fully_correct_rally": fixed_zero_lost == 0,
        "recovers_at_least_quarter_of_oracle_new_serves": serve_recovery_fraction >= 0.25,
    }
    return {
        "headroom": {
            "pass": all(headroom_checks.values()),
            "checks": headroom_checks,
            "addressable_fraction_of_missed_serves": structural[
                "addressable_fraction_of_missed_serves"
            ],
            "oracle_net_fully_correct_change_at_zero": oracle_zero["net_fully_correct_change"],
        },
        "fixed_rule": {
            "pass": all(fixed_checks.values()),
            "checks": fixed_checks,
            "new_fully_correct_at_zero": fixed_zero_new,
            "lost_fully_correct_at_zero": fixed_zero_lost,
            "lost_fully_correct_at_0.90": fixed_ninety_lost,
            "serve_recovery_fraction_of_oracle": serve_recovery_fraction,
        },
    }


def score(arguments: argparse.Namespace, *, construction_only: bool | None = None) -> dict[str, Any]:
    """Run the frozen prefix experiment in its required label-load order."""
    verified = load_label_blind_inputs(arguments)
    frozen, construction_bytes = freeze_construction_twice(
        verified.nplus_rows,
        verified.candidates.rows,
        verified.evidence.evidence,
        verified.search_intervals,
    )
    validate_real_construction(frozen)
    if construction_only is None:
        construction_only = bool(getattr(arguments, "construction_only", False))
    payload: dict[str, Any] = {
        "schema": RESULTS_SCHEMA,
        "fixture_set": list(tree_scorer.FIXTURE_SPECS),
        "selected_trial": SELECTED_TRIAL_ID,
        "labels_read_after_predictions_fixed": True,
        "construction": construction_payload(frozen),
        "construction_json_sha256": hashlib.sha256(construction_bytes).hexdigest(),
        "inputs": {
            "feature_manifest_sha256": tree_scorer._sha256(arguments.feature_manifest),
            "feature_sha256": verified.features.manifest["feature_sha256"],
            "tree_result_sha256": verified.candidates.manifest["tree_result_sha256"],
            "candidate_manifest_sha256": tree_scorer._sha256(arguments.candidate_manifest),
            "candidate_scores_sha256": verified.candidates.manifest["candidate_sha256"],
            "evidence_manifest_sha256": tree_scorer._sha256(arguments.evidence_manifest),
            "evidence_sha256": verified.evidence.manifest["evidence_sha256"],
            "region_v1_manifest_sha256": tree_scorer._sha256(arguments.region_v1_manifest),
            "region_v1_result_sha256": tree_scorer._sha256(arguments.region_v1_results),
        },
    }
    if construction_only:
        payload["construction_only"] = True
        return payload

    # This is the only side-prediction boundary.  It runs before either
    # ShuttleSet timing or player-side labels are loaded.
    attribution = replay_candidate_sides(arguments, verified.nplus_rows, frozen.prefixes)
    nplus_events = rally_scorer.retained_events_from_scores(verified.nplus_rows, attribution)
    fixed_events = apply_actions_to_stream(nplus_events, frozen.prefixes, attribution)
    nplus_spans = rally_scorer.fixed_spans_from_evidence(verified.evidence.evidence, nplus_events)
    fixed_output = attach_actions_to_spans(
        verified.evidence.evidence,
        fixed_events,
        frozen.prefixes,
    )

    # Timing labels are now allowed.  The oracle can only inspect frozen rows.
    rallies_by_fixture = rally_scorer._load_timing_rallies()
    oracle_prefixes = _oracle_prefixes(
        frozen,
        nplus_spans,
        rallies_by_fixture,
        verified.evidence.evidence,
    )
    oracle_events = apply_actions_to_stream(nplus_events, oracle_prefixes, attribution)
    oracle_output = attach_actions_to_spans(
        verified.evidence.evidence,
        oracle_events,
        oracle_prefixes,
    )

    # Player-side labels load last, after the oracle's actions are final.
    target_sides = rally_scorer._load_side_labels()
    ground_truth = _ground_truth_from_rallies(rallies_by_fixture)
    if frozen.nplus_event_count != EXPECTED_NPLUS_EVENT_COUNT:
        raise ValueError("N+ event count changed after stream construction")
    fps_by_fixture = {
        fixture: float(fps)
        for fixture, (_video_id, fps) in tree_scorer.FIXTURE_SPECS.items()
    }
    baseline_strict = _strict_comparison(
        nplus_spans,
        nplus_spans,
        rallies_by_fixture,
        target_sides,
        fps_by_fixture,
        PRIMARY_TOLERANCE_BASE30,
    )
    baseline_zero = baseline_strict["baseline_at_zero"]
    if baseline_zero["fully_correct_kept_rallies"] != 27:
        raise ValueError("N+ baseline does not reproduce 27 fully correct spans at confidence 0.00")
    streams = {
        "nplus": nplus_events,
        "fixed": fixed_events,
        "oracle": oracle_events,
    }
    spans_by_stream = {
        "nplus": nplus_spans,
        "fixed": fixed_output.spans,
        "oracle": oracle_output.spans,
    }
    baseline_predictions = _prediction_frames_from_events(nplus_events)
    event_reports: dict[str, Any] = {}
    strict_reports: dict[str, Any] = {}
    for name in ("fixed", "oracle"):
        predictions = _prediction_frames_from_events(streams[name])
        event_reports[name] = {
            str(tolerance): _event_comparison(
                ground_truth,
                baseline_predictions,
                predictions,
                tolerance,
            )
            for tolerance in (PRIMARY_TOLERANCE_BASE30, SENSITIVITY_TOLERANCE_BASE30)
        }
        strict_reports[name] = {
            str(tolerance): _strict_comparison(
                nplus_spans,
                spans_by_stream[name],
                rallies_by_fixture,
                target_sides,
                fps_by_fixture,
                tolerance,
            )
            for tolerance in (PRIMARY_TOLERANCE_BASE30, SENSITIVITY_TOLERANCE_BASE30)
        }
    structural = _structural_report(frozen, nplus_spans, rallies_by_fixture, ground_truth)
    payload.update(
        {
            "construction_only": False,
            "structural": structural,
            "event_reports": event_reports,
            "strict_reports": strict_reports,
            "prospective_gates": _prospective_gate_report(
                structural,
                event_reports,
                strict_reports,
            ),
            "baseline_nplus": {
                "event_count": frozen.nplus_event_count,
                "fully_correct_at_zero": baseline_zero["fully_correct_kept_rallies"],
                "primary_tolerance_base30": PRIMARY_TOLERANCE_BASE30,
            },
            "actions": {
                "fixed_insert_count": sum(prefix.fixed_action.action == "insert" for prefix in frozen.prefixes),
                "fixed_abstention_count": sum(prefix.fixed_action.candidate is None for prefix in frozen.prefixes),
                "fixed_selected_count": sum(prefix.fixed_action.candidate is not None for prefix in frozen.prefixes),
                "fixed_pre_span_choice_count": sum(
                    bounds.serve_prepend_frame is not None
                    for bounds in fixed_output.bounds
                ),
                "oracle_insert_count": sum(prefix.fixed_action.action == "insert" for prefix in oracle_prefixes),
                "oracle_selected_count": sum(prefix.fixed_action.candidate is not None for prefix in oracle_prefixes),
                "oracle_pre_span_choice_count": sum(
                    bounds.serve_prepend_frame is not None
                    for bounds in oracle_output.bounds
                ),
                "fixed_output_spans": [
                    _output_span_json(bounds)
                    for bounds in fixed_output.bounds
                ],
                "oracle_output_spans": [
                    _output_span_json(bounds)
                    for bounds in oracle_output.bounds
                ],
                "oracle": [_action_json(prefix.fixed_action) for prefix in oracle_prefixes],
            },
        }
    )
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the frozen input paths and output mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    raw_root = CONTACT_DET_ROOT / "raw"
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=raw_root / "followups" / "phase2" / "baseline_freeze" / "tree_contact_features_manifest.json",
    )
    parser.add_argument(
        "--tree-results",
        type=Path,
        default=raw_root / "followups" / "phase2" / "baseline_score" / "tree_contact_results.json.gz",
    )
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        default=raw_root / "followups" / "phase2" / "baseline_score" / "tree_contact_candidate_scores_manifest.json",
    )
    parser.add_argument("--evidence-manifest", type=Path, default=raw_root / "contact_evidence_manifest.json")
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
    parser.add_argument("--construction-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the experiment and write deterministic JSON or gzip JSON."""
    arguments = parse_args(argv)
    payload = score(arguments)
    rally_scorer.write_results(arguments.output, payload)
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
