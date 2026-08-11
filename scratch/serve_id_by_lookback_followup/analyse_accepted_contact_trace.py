"""Run and check the fixed sequential opener search over all 239 primary rallies."""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import io
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TypeAlias

import numpy as np

from annotator import point_winner
from annotator.calibration.fixtures import FIXTURES, Fixture
from annotator.calibration.gt_scoring import load_gt_tables
from annotator.calibration.scoring import RallyBoundary
from annotator.fps_constants import ScalingKind
from annotator.inpaint_guard import NO_FLAG
from annotator.types import Slot, StickyResult
from scratch.serve_id_by_lookback_followup.accepted_contact_trace import (
    AcceptedContactEvidence,
    IncomingVerdict,
    OpenerCategory,
    closest_post_contact_run,
    has_credible_outgoing_motion,
    search_accepted_contacts,
)
from scratch.serve_start_trajectory_exploration.experiment_data import (
    VideoData,
    load_video_data,
)
from scratch.serve_start_trajectory_exploration.trajectory_features import (
    IncomingMotion,
    RobustDistanceTrend,
    align_anchor_to_gt,
    closest_pre_contact_run,
    decide_fixed_motion_rules,
    fit_robust_distance_trend,
    measure_incoming_motion,
)

RUN_DIR = Path(__file__).resolve().parent
ROWS_PATH = RUN_DIR / "accepted_contact_trace_rows.csv.gz"
SUMMARY_PATH = RUN_DIR / "accepted_contact_trace_summary.json.gz"

LOOKAROUND_BASE30_FRAMES = 30
MAX_LOCAL_GAP_BASE30_FRAMES = 2
CONTACT_TOLERANCES_BASE30 = (5, 10, 30)
EXPECTED_PRIMARY_BY_FIXTURE = {"sset_01": 104, "sset_15": 84, "sset_21": 51}

SearchKey: TypeAlias = tuple[str, int, str, int]


@dataclass(frozen=True, slots=True)
class SearchRow:
    """One frozen GT-free search result with its stable crosswalk key."""

    fixture: str
    video_id: int
    set_id: str
    rally: int
    fps: float
    span_id: int
    accepted_contact_frames: tuple[int, ...]
    credible_outgoing: tuple[bool, ...]
    selected_frame: int | None
    selected_rank: int | None
    skipped_contacts: int
    selected_player: str | None
    pre_contact_verdict: str | None
    opener_category: str

    @property
    def key(self) -> SearchKey:
        """Return the fixed row key without any stroke labels."""
        return self.fixture, self.video_id, self.set_id, self.rally


@dataclass(frozen=True, slots=True)
class TruthRow:
    """GT stroke frames held outside the search path until scoring."""

    stroke_frames: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SearchInputs:
    """Trajectory and accepted-contact fields with all GT fields removed."""

    fixture: Fixture
    track: np.ndarray
    bboxes: np.ndarray
    sticky: StickyResult
    segments: tuple[tuple[int, int], ...]
    guard_codes: np.ndarray
    court_present: np.ndarray
    spans: tuple[tuple[int, int], ...]
    accepted_by_span: dict[int, tuple[int, ...]]

    @classmethod
    def from_video_data(cls, data: VideoData) -> SearchInputs:
        """Copy only fields the GT-free search is allowed to inspect."""
        return cls(
            fixture=data.fixture,
            track=data.track,
            bboxes=data.bboxes,
            sticky=data.sticky,
            segments=tuple(data.segments),
            guard_codes=data.guard_codes,
            court_present=data.court_present,
            spans=tuple(data.spans),
            accepted_by_span={
                span_id: tuple(frames)
                for span_id, frames in data.accepted_by_span.items()
            },
        )

    def segment_for_frame(self, frame: int) -> tuple[int, int] | None:
        """Return the half-open tracker segment containing a source frame."""
        for start, end in self.segments:
            if start <= frame < end:
                return start, end
        return None


@dataclass(frozen=True, slots=True)
class ContactContext:
    """Player and recurrence-clean mask for one accepted contact."""

    player: point_winner.Half
    slot: Slot
    usable: np.ndarray


def _scaled_frames(base30_frames: int, fps: float) -> int:
    """Scale a base-30fps frame count with the production convention."""
    return int(ScalingKind.FRAME_COUNT.scale(base30_frames, fps))


def _contact_context(
    data: SearchInputs,
    span_id: int,
    contact_frame: int,
) -> ContactContext | None:
    """Build the fixed player-specific recurrence-clean path mask."""
    player = point_winner.attribute_half(
        contact_frame,
        data.track,
        data.sticky,
        data.bboxes,
        data.fixture.net_band,
    )
    segment = data.segment_for_frame(contact_frame)
    if player is None or segment is None:
        return None

    slot = Slot.TOP if player is point_winner.Half.TOP else Slot.BOTTOM
    coordinate_valid = np.isfinite(data.track[:, :2]).all(axis=1)
    non_zero_coordinate = ~((data.track[:, 0] == 0) & (data.track[:, 1] == 0))
    usable = (
        (data.track[:, 2] == 1)
        & coordinate_valid
        & non_zero_coordinate
        & data.court_present
        & np.isfinite(data.sticky.distances_per_slot[:, slot])
        & np.isfinite(data.sticky.bbox_height[:, slot])
        & (data.sticky.bbox_height[:, slot] > 0)
        & (data.guard_codes == NO_FLAG)
    )

    span_start, span_end = data.spans[span_id]
    segment_start, segment_end = segment
    local_start = max(span_start, segment_start)
    local_end = min(span_end, segment_end)
    local_mask = np.zeros(len(data.track), dtype=bool)
    local_mask[local_start:local_end] = True
    return ContactContext(player, slot, usable & local_mask)


def _measure_run(
    data: SearchInputs,
    context: ContactContext,
    start: int,
    end: int,
) -> tuple[IncomingMotion, RobustDistanceTrend] | None:
    """Measure one fixed local path using the PR #82 primitives."""
    if end - start < 2:
        return None
    run_slice = slice(start, end)
    distances_bh = data.sticky.distances_per_slot[run_slice, context.slot]
    motion = measure_incoming_motion(
        distances_bh,
        data.track[run_slice, :2],
        data.sticky.bbox_height[run_slice, context.slot],
        data.fixture.resolution,
    )
    return motion, fit_robust_distance_trend(distances_bh)


def _credible_outgoing(
    data: SearchInputs,
    contact_frame: int,
    context: ContactContext | None,
    lookahead_frames: int,
    maximum_gap_frames: int,
) -> bool:
    """Return the binary post-contact predicate for one accepted contact."""
    if context is None:
        return False
    run = closest_post_contact_run(context.usable, contact_frame, lookahead_frames)
    if run is None:
        return False
    measurements = _measure_run(data, context, run.start, run.end)
    if measurements is None:
        return False
    motion, trend = measurements
    return has_credible_outgoing_motion(
        motion,
        trend,
        run.frames_from_contact,
        maximum_gap_frames,
    )


def _incoming_verdict(
    data: SearchInputs,
    contact_frame: int,
    context: ContactContext,
    lookback_frames: int,
    maximum_gap_frames: int,
) -> IncomingVerdict:
    """Apply the existing PR #82 three-way pre-contact check."""
    run = closest_pre_contact_run(context.usable, contact_frame, lookback_frames)
    if run is None:
        return IncomingVerdict.UNAVAILABLE
    measurements = _measure_run(data, context, run.start, run.end)
    if measurements is None:
        return IncomingVerdict.UNAVAILABLE
    motion, trend = measurements
    decisions = decide_fixed_motion_rules(
        motion,
        trend,
        run.frames_to_contact,
        maximum_gap_frames,
    )
    if not decisions.common_path_eligible:
        return IncomingVerdict.UNAVAILABLE
    if decisions.robust_trend_incoming:
        return IncomingVerdict.INCOMING
    return IncomingVerdict.NOT_INCOMING


def _half_text(player: point_winner.Half | None) -> str | None:
    """Return the stable spelling for a selected player."""
    return player.value if player is not None else None


def build_span_search_row(
    data: SearchInputs,
    span_id: int,
    key: SearchKey,
) -> SearchRow:
    """Run the GT-free opener search for one fixed primary span."""
    accepted = tuple(sorted(data.accepted_by_span.get(span_id, [])))
    if len(accepted) != len(set(accepted)):
        raise ValueError(f"{data.fixture.name} span {span_id}: accepted contacts must be unique")
    span_start, span_end = data.spans[span_id]
    if any(not span_start <= frame < span_end for frame in accepted):
        raise ValueError(f"{data.fixture.name} span {span_id}: accepted contact is outside its span")

    lookaround_frames = _scaled_frames(LOOKAROUND_BASE30_FRAMES, data.fixture.fps)
    maximum_gap_frames = _scaled_frames(MAX_LOCAL_GAP_BASE30_FRAMES, data.fixture.fps)
    contexts = {
        frame: _contact_context(data, span_id, frame)
        for frame in accepted
    }
    outgoing = tuple(
        _credible_outgoing(
            data,
            frame,
            contexts[frame],
            lookaround_frames,
            maximum_gap_frames,
        )
        for frame in accepted
    )
    evidence = [
        AcceptedContactEvidence(frame, credible)
        for frame, credible in zip(accepted, outgoing, strict=True)
    ]
    selected_pre: IncomingVerdict | None = None

    def incoming_check(frame: int) -> IncomingVerdict:
        nonlocal selected_pre
        context = contexts[frame]
        if context is None:
            raise ValueError("a credible outgoing contact must have a player and tracker scene")
        selected_pre = _incoming_verdict(
            data,
            frame,
            context,
            lookaround_frames,
            maximum_gap_frames,
        )
        return selected_pre

    result = search_accepted_contacts(evidence, incoming_check)
    selected_player = (
        contexts[result.selected_frame].player
        if result.selected_frame is not None and contexts[result.selected_frame] is not None
        else None
    )
    fixture, video_id, set_id, rally = key
    return SearchRow(
        fixture=fixture,
        video_id=video_id,
        set_id=set_id,
        rally=rally,
        fps=data.fixture.fps,
        span_id=span_id,
        accepted_contact_frames=accepted,
        credible_outgoing=outgoing,
        selected_frame=result.selected_frame,
        selected_rank=result.selected_rank,
        skipped_contacts=result.skipped_contacts,
        selected_player=_half_text(selected_player),
        pre_contact_verdict=selected_pre.value if selected_pre is not None else None,
        opener_category=result.category.value,
    )


def _primary_crosswalk(data: VideoData) -> list[tuple[object, int]]:
    """Return the fixed one-to-one rally-to-span crosswalk for one fixture."""
    covered = [
        (rally, span_id)
        for rally, (boundary, span_id) in zip(data.gt_rallies, data.boundaries, strict=True)
        if boundary is RallyBoundary.COVERED and span_id is not None
    ]
    multiplicity = Counter(span_id for _rally, span_id in covered)
    return [(rally, span_id) for rally, span_id in covered if multiplicity[span_id] == 1]


def build_search_rows() -> tuple[list[SearchRow], dict[SearchKey, TruthRow]]:
    """Freeze all search rows before returning the separate GT frame table."""
    shared_gt_tables = load_gt_tables()
    search_rows: list[SearchRow] = []
    truth_by_key: dict[SearchKey, TruthRow] = {}
    counts_by_fixture: Counter[str] = Counter()
    for fixture in FIXTURES:
        print(f"{fixture.name}: loading fixed inputs")
        data = load_video_data(fixture, shared_gt_tables)
        search_inputs = SearchInputs.from_video_data(data)
        for rally, span_id in _primary_crosswalk(data):
            key = (fixture.name, fixture.video_id, rally.set_id, rally.rally)
            search_rows.append(build_span_search_row(search_inputs, span_id, key))
            truth_by_key[key] = TruthRow(tuple(int(frame) for frame in rally.stroke_frames))
            counts_by_fixture[fixture.name] += 1
        print(f"{fixture.name}: froze {counts_by_fixture[fixture.name]} primary search rows")
        del data
        gc.collect()

    if len(search_rows) != 239 or counts_by_fixture != Counter(EXPECTED_PRIMARY_BY_FIXTURE):
        raise ValueError("primary population differs from the fixed 239-rally contract")
    keys = [row.key for row in search_rows]
    if len(keys) != len(set(keys)) or set(keys) != set(truth_by_key):
        raise ValueError("search and truth tables must have the same 239 unique keys")
    return search_rows, truth_by_key


def _search_dict(row: SearchRow) -> dict[str, object]:
    """Convert one frozen row to its stable CSV representation."""
    values = asdict(row)
    values["accepted_contact_frames"] = json.dumps(
        row.accepted_contact_frames,
        separators=(",", ":"),
    )
    values["credible_outgoing"] = json.dumps(row.credible_outgoing, separators=(",", ":"))
    return values


def _score_category(category: str, selected_label: str) -> bool:
    """Score visible and implied serves against the expected visible GT contact."""
    if category == OpenerCategory.VISIBLE_SERVE.value:
        return selected_label == "contact_1"
    if category == OpenerCategory.FIRST_VISIBLE_POST_SERVE.value:
        return selected_label == "contact_2"
    return False


def _transition(category: str, baseline_correct: bool, final_correct: bool) -> str:
    """Name the fixed correctness transition for one tolerance."""
    if category == OpenerCategory.NOT_ENOUGH_TRAJECTORY.value:
        return "pre_contact_unknown"
    if category == OpenerCategory.NO_CREDIBLE_CONTACT.value:
        return "no_credible_contact"
    if final_correct and not baseline_correct:
        return "fixed"
    if baseline_correct and not final_correct:
        return "damaged"
    return "unchanged_correct" if final_correct else "unchanged_wrong"


def score_search_rows(
    search_rows: Sequence[SearchRow],
    truth_by_key: dict[SearchKey, TruthRow],
) -> list[dict[str, object]]:
    """Join GT only after the search rows have been frozen."""
    scored: list[dict[str, object]] = []
    for search_row in search_rows:
        row = _search_dict(search_row)
        truth = truth_by_key[search_row.key]
        row["gt_stroke_frames"] = json.dumps(truth.stroke_frames, separators=(",", ":"))
        baseline_frame = (
            search_row.accepted_contact_frames[0]
            if search_row.accepted_contact_frames
            else None
        )
        for tolerance in CONTACT_TOLERANCES_BASE30:
            baseline_label = "no_anchor"
            baseline_multiple = False
            if baseline_frame is not None:
                baseline = align_anchor_to_gt(
                    baseline_frame,
                    truth.stroke_frames,
                    search_row.fps,
                    tolerance,
                )
                baseline_label = baseline.label
                baseline_multiple = baseline.multiple_within_tolerance

            selected_label = "no_anchor"
            selected_ordinal: int | None = None
            selected_offset: float | None = None
            selected_multiple = False
            if search_row.selected_frame is not None:
                selected = align_anchor_to_gt(
                    search_row.selected_frame,
                    truth.stroke_frames,
                    search_row.fps,
                    tolerance,
                )
                selected_label = selected.label
                selected_ordinal = selected.nearest_gt_ordinal
                selected_offset = selected.signed_offset_base30
                selected_multiple = selected.multiple_within_tolerance

            final_correct = _score_category(search_row.opener_category, selected_label)
            baseline_correct = baseline_label == "contact_1"
            prefix = f"tolerance_{tolerance}"
            row[f"{prefix}_baseline_label"] = baseline_label
            row[f"{prefix}_baseline_multiple"] = baseline_multiple
            row[f"{prefix}_selected_label"] = selected_label
            row[f"{prefix}_selected_nearest_gt_ordinal"] = selected_ordinal
            row[f"{prefix}_selected_signed_offset_base30"] = selected_offset
            row[f"{prefix}_selected_multiple"] = selected_multiple
            row[f"{prefix}_final_correct"] = final_correct
            row[f"{prefix}_transition"] = _transition(
                search_row.opener_category,
                baseline_correct,
                final_correct,
            )
        scored.append(row)
    return scored


def build_summary(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    """Build compact fixed counts without selecting any rule or threshold."""
    summary: dict[str, object] = {
        "schema": "accepted_contact_trace_summary/1",
        "population": len(rows),
        "fixed_rule": {
            "lookaround_base30_frames": LOOKAROUND_BASE30_FRAMES,
            "maximum_local_gap_base30_frames": MAX_LOCAL_GAP_BASE30_FRAMES,
            "minimum_path_frames": 5,
            "maximum_largest_step_ratio": 4.0,
            "minimum_directional_change_bh": 0.05,
        },
        "opener_categories": dict(sorted(Counter(row["opener_category"] for row in rows).items())),
        "selected_ranks": dict(
            sorted(Counter(str(row["selected_rank"]) for row in rows).items())
        ),
        "skipped_contacts_total": sum(int(row["skipped_contacts"]) for row in rows),
        "tolerances": {},
    }
    tolerance_summaries: dict[str, object] = {}
    for tolerance in CONTACT_TOLERANCES_BASE30:
        prefix = f"tolerance_{tolerance}"
        tolerance_summaries[str(tolerance)] = {
            "baseline_labels": dict(
                sorted(Counter(row[f"{prefix}_baseline_label"] for row in rows).items())
            ),
            "transitions": dict(
                sorted(Counter(row[f"{prefix}_transition"] for row in rows).items())
            ),
            "final_correct": sum(bool(row[f"{prefix}_final_correct"]) for row in rows),
            "baseline_multiple": sum(bool(row[f"{prefix}_baseline_multiple"]) for row in rows),
            "selected_multiple": sum(bool(row[f"{prefix}_selected_multiple"]) for row in rows),
        }
    primary_unmatched = [
        row for row in rows if row["tolerance_10_baseline_label"] == "unmatched"
    ]
    tolerance_summaries["10"]["baseline_unmatched_slice"] = {
        "population": len(primary_unmatched),
        "transitions": dict(
            sorted(Counter(row["tolerance_10_transition"] for row in primary_unmatched).items())
        ),
    }
    summary["tolerances"] = tolerance_summaries
    return summary


def _csv_text(rows: Sequence[dict[str, object]]) -> str:
    """Serialise rows with a stable field order."""
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


def write_csv_gz(path: Path, rows: Sequence[dict[str, object]]) -> None:
    """Write deterministic compressed CSV evidence."""
    payload = _csv_text(rows).encode("utf-8")
    with (
        path.open("wb") as raw_handle,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as handle,
    ):
        handle.write(payload)


def read_csv_gz(path: Path) -> list[dict[str, str]]:
    """Read one compressed CSV evidence table."""
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _normalised_csv_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, str]]:
    """Return the exact string representation used by the saved CSV."""
    return list(csv.DictReader(io.StringIO(_csv_text(rows), newline="")))


def write_json_gz(path: Path, payload: dict[str, object]) -> None:
    """Write deterministic compressed JSON evidence."""
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with (
        path.open("wb") as raw_handle,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as handle,
    ):
        handle.write(encoded)


def read_json_gz(path: Path) -> object:
    """Read compressed JSON evidence."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _search_fields() -> tuple[str, ...]:
    """Return the GT-free prefix of the saved row schema."""
    return tuple(field.name for field in fields(SearchRow))


def check_saved_outputs(
    scored_rows: Sequence[dict[str, object]],
    summary: dict[str, object],
) -> None:
    """Rebuild and compare saved decompressed search and scored evidence."""
    saved_rows = read_csv_gz(ROWS_PATH)
    rebuilt_rows = _normalised_csv_rows(scored_rows)
    if len(saved_rows) != 239 or len(rebuilt_rows) != 239:
        raise ValueError("saved and rebuilt evidence must each contain 239 rows")
    search_fields = _search_fields()
    saved_search = [{field: row[field] for field in search_fields} for row in saved_rows]
    rebuilt_search = [{field: row[field] for field in search_fields} for row in rebuilt_rows]
    if saved_search != rebuilt_search:
        raise ValueError("saved GT-free search rows differ from the rebuilt search rows")
    if saved_rows != rebuilt_rows:
        raise ValueError("saved scored rows differ from the rebuilt scored rows")
    if read_json_gz(SUMMARY_PATH) != summary:
        raise ValueError("saved summary differs from the rebuilt summary")


def run(*, check: bool) -> None:
    """Build all rows, then write or check the fixed evidence files."""
    search_rows, truth_by_key = build_search_rows()
    scored_rows = score_search_rows(search_rows, truth_by_key)
    summary = build_summary(scored_rows)
    if check:
        check_saved_outputs(scored_rows, summary)
        print("checked 239 saved search and scored rows")
        return
    write_csv_gz(ROWS_PATH, scored_rows)
    write_json_gz(SUMMARY_PATH, summary)
    print(f"wrote {len(scored_rows)} rows to {ROWS_PATH}")


def main(argv: Iterable[str] | None = None) -> None:
    """Parse the one supported check mode and run the analysis."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args(argv)
    run(check=arguments.check)


if __name__ == "__main__":
    main()
