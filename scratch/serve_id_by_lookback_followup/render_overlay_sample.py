"""Select and render a GT-free diagnostic sample for accepted-contact evidence.

The 239-row trace is the selection population.  This script only uses its
frozen accepted-contact and verdict fields when choosing clips.  It loads the
underlying measurements afterwards to explain and draw each chosen contact.
"""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import json
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from annotator import point_winner
from annotator.calibration.fixtures import FIXTURES, Fixture
from annotator.calibration.gt_scoring import load_gt_tables
from annotator.fps_constants import ScalingKind
from annotator.inpaint_guard import CODE_NAMES, NO_FLAG, grade_track
from annotator.rally.evidence import (
    BODY_UNIT_HALF_WINDOW,
    build_sticky_result,
    tracker_segments,
)
from annotator.replay_mask import _read_homography_rows
from annotator.types import Slot, StickyResult
from annotator.validation_overlay.core.cli import DrawFn, make_render_plan, render
from annotator.validation_overlay.core.decode import probe_video
from annotator.validation_overlay.core.hud import FONT, HudStyle
from annotator.validation_overlay.core.timeline import Segment
from scratch.serve_id_by_lookback_followup.accepted_contact_trace import (
    IncomingVerdict,
    closest_post_contact_run,
    has_credible_outgoing_motion,
)
from scratch.serve_id_by_lookback_followup.analyse_accepted_contact_trace import (
    LOOKAROUND_BASE30_FRAMES,
    MAX_LOCAL_GAP_BASE30_FRAMES,
    _contact_context,
    _measure_run,
)
from scratch.serve_start_trajectory_exploration.experiment_data import (
    FIXTURE_ROOT,
    RELEASE_RESULTS,
)
from scratch.serve_start_trajectory_exploration.trajectory_features import (
    closest_pre_contact_run,
    decide_fixed_motion_rules,
)

RUN_DIR = Path(__file__).resolve().parent
TRACE_PATH = RUN_DIR / "accepted_contact_trace_rows.csv.gz"
OUTPUT_DIR = RUN_DIR / "overlay_sample"
MANIFEST_PATH = OUTPUT_DIR / "manifest.csv"
README_PATH = OUTPUT_DIR / "README.md"
ROW_CACHE_DIR = Path("/tmp/serve_id_overlay_rows")
STRIDE1_TRACK = (
    RUN_DIR.parent / "serve_start_trajectory_exploration" / "assets"
    / "shuttleset-current-annotator-reference-v1" / "inputs" / "track_overrides"
    / "sset_01_tracknet_stride1_weight.npy"
)
STRIDE1_RELEASE_DIR = (
    RUN_DIR.parent / "serve_start_trajectory_exploration" / "assets"
    / "shuttleset-current-annotator-reference-v1" / "measurement"
    / "current_annotator_8config_288p" / "static_shuttleset_homography" / "sset_01" / "tracknet-stride-1"
)
COMPARISON_DIR = OUTPUT_DIR / "stride1_comparisons"
CLIP_HALF_WINDOW_BASE30 = 60
SAMPLE_PER_STRATUM = 5
# The standalone clips need enough physical pixels for the explanatory panel.
# Comparison panels stay at their established width so their existing layout is
# unchanged.
SAMPLE_RENDER_WIDTH = 1920
COMPARISON_PANEL_WIDTH = 960
HUD_HEIGHT = 32
VIDEO_PATHS = {
    "sset_01": Path("local_scratch/autograder_architecture/videos_288p/sset_01_288p.mp4"),
    "sset_15": Path("local_scratch/autograder_architecture/videos_288p/sset_15_288p.mp4"),
    "sset_21": Path("local_scratch/autograder_architecture/videos_288p/sset_21_288p.mp4"),
}
FIXTURE_ORDER = tuple(fixture.name for fixture in FIXTURES)
_GATE_INPUTS: tuple[dict[str, dict], object] | None = None

# Protan-safe, plus a shape or line-style distinction for every meaning.
FOCAL_COLOUR = (255, 220, 0)  # cyan
OTHER_CONTACT_COLOUR = (0, 210, 255)  # yellow
SHUTTLE_COLOUR = (220, 80, 255)  # magenta
ANCHOR_COLOUR = (255, 160, 40)  # blue-orange
USABLE_COLOUR = (230, 230, 230)  # white
UNUSABLE_COLOUR = (75, 75, 75)  # dark grey


@dataclass(frozen=True, slots=True)
class TraceRow:
    """The GT-free fields needed to nominate a diagnostic contact."""

    fixture: str
    video_id: int
    set_id: str
    rally: int
    span_id: int
    fps: float
    accepted: tuple[int, ...]
    outgoing: tuple[bool, ...]
    selected_frame: int | None
    selected_rank: int | None
    selected_player: str | None
    pre_verdict: str | None

    @property
    def stable_key(self) -> str:
        return f"{self.fixture}:{self.video_id}:{self.set_id}:{self.rally}"


@dataclass(frozen=True, slots=True)
class Candidate:
    """One selected contact, before metric explanation or drawing."""

    stratum: str
    trace: TraceRow
    frame: int
    accepted_rank: int
    player: str


@dataclass(frozen=True, slots=True)
class ContactDetails:
    """Measurements and explanatory text for an already selected contact."""

    candidate: Candidate
    usable: np.ndarray
    pre_run: object | None
    post_run: object | None
    pre_motion: object | None
    post_motion: object | None
    pre_failure: str
    post_failure: str
    max_gap: int


@dataclass(frozen=True, slots=True)
class WindowData:
    """Global-indexed arrays with sticky evidence rebuilt for one padded window."""

    fixture: Fixture
    track: np.ndarray
    bboxes: np.ndarray
    sticky: StickyResult
    segments: list[tuple[int, int]]
    guard_codes: np.ndarray
    court_present: np.ndarray
    spans: list[tuple[int, int]]

    def segment_for_frame(self, frame: int) -> tuple[int, int] | None:
        for start, end in self.segments:
            if start <= frame < end:
                return start, end
        return None


def scaled_frames(base30_frames: int, fps: float) -> int:
    """Scale a base-30 frame count with the fixed production convention."""
    return int(ScalingKind.FRAME_COUNT.scale(base30_frames, fps))


def read_trace_rows(path: Path = TRACE_PATH) -> list[TraceRow]:
    """Read the frozen trace without consulting its scoring-only GT columns."""
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows: list[TraceRow] = []
    for raw in raw_rows:
        rows.append(
            TraceRow(
                fixture=raw["fixture"],
                video_id=int(raw["video_id"]),
                set_id=raw["set_id"],
                rally=int(raw["rally"]),
                span_id=int(raw["span_id"]),
                fps=float(raw["fps"]),
                accepted=tuple(json.loads(raw["accepted_contact_frames"])),
                outgoing=tuple(json.loads(raw["credible_outgoing"])),
                selected_frame=int(raw["selected_frame"]) if raw["selected_frame"] else None,
                selected_rank=int(raw["selected_rank"]) if raw["selected_rank"] else None,
                selected_player=raw["selected_player"] or None,
                pre_verdict=raw["pre_contact_verdict"] or None,
            )
        )
    if len(rows) != 239:
        raise ValueError(f"expected the fixed 239-row population, got {len(rows)}")
    return rows


def player_name(player: point_winner.Half | None) -> str | None:
    """Return the required display spelling for an attributed player."""
    if player is point_winner.Half.TOP:
        return "Top"
    if player is point_winner.Half.BOT:
        return "Bottom"
    return None


def _spans_for_fixture(fixture: Fixture) -> list[tuple[int, int]]:
    release_dir = RELEASE_RESULTS / fixture.name / "tracknet-stride-8"
    annotations = json.loads((release_dir / "annotations.json").read_text(encoding="utf-8"))
    spans: list[tuple[int, int]] = []
    for start, end in annotations["spans"]:
        spans.append((int(start), int(end)))
    return spans


def _empty_global_sticky(n_frames: int) -> StickyResult:
    return StickyResult(
        np.full(n_frames, np.inf),
        np.full((n_frames, 2), -1, dtype=int),
        np.zeros(n_frames, dtype=int),
        np.full((n_frames, 2, 2), np.nan),
        np.full((n_frames, 2), np.nan),
        np.full((n_frames, 2), np.inf),
        np.full((n_frames, 2), np.inf),
        np.zeros(n_frames, dtype=bool),
    )


def load_window_data(
    fixture: Fixture,
    frame: int,
    *,
    track_path: Path | None = None,
    release_dir: Path | None = None,
) -> WindowData:
    """Rebuild sticky evidence from a segment start through one clip's padded end.

    Pose inputs stay memory-mapped. Only the required prefix of the tracker
    segment is copied into RAM, retaining the production picker's EMA history.
    """
    track = np.load(track_path or FIXTURE_ROOT / fixture.track_path, mmap_mode="r", allow_pickle=False)
    n_frames = len(track)
    release_dir = release_dir or RELEASE_RESULTS / fixture.name / "tracknet-stride-8"
    court_present = np.load(release_dir / "court_present.npy", mmap_mode="r", allow_pickle=False)
    scene_rows = _read_homography_rows(release_dir / "scene_rows.csv", str(fixture.video_id))
    segments = tracker_segments(scene_rows, court_present, n_frames)
    segment = next(((start, end) for start, end in segments if start <= frame < end), None)
    if segment is None:
        raise ValueError(f"{fixture.name} frame {frame}: missing tracker scene")
    start, segment_end = segment
    clip_end = min(n_frames, frame + scaled_frames(CLIP_HALF_WINDOW_BASE30, fixture.fps) + BODY_UNIT_HALF_WINDOW + 1)
    end = min(segment_end, clip_end)
    if end <= start:
        raise ValueError(f"{fixture.name} frame {frame}: empty sticky window")

    bboxes = np.load(FIXTURE_ROOT / fixture.pose_path("bboxes"), mmap_mode="r", allow_pickle=False)
    scores = np.load(FIXTURE_ROOT / fixture.pose_path("scores"), mmap_mode="r", allow_pickle=False)
    kps = np.load(FIXTURE_ROOT / fixture.pose_path("kps"), mmap_mode="r", allow_pickle=False)
    ndet = np.load(FIXTURE_ROOT / fixture.pose_path("ndet"), mmap_mode="r", allow_pickle=False)
    global _GATE_INPUTS
    if _GATE_INPUTS is None:
        _master, _homography, courts, resolution_table = load_gt_tables()
        _GATE_INPUTS = ({str(video_id): court for video_id, court in courts.items()}, resolution_table)
    gate_courts, resolution_table = _GATE_INPUTS
    gate_resolution = resolution_table.copy()
    gate_resolution.index = gate_resolution.index.astype(str)
    local_sticky = build_sticky_result(
        np.asarray(track[start:end]),
        [(0, end - start)],
        np.asarray(bboxes[start:end]),
        np.asarray(scores[start:end]),
        np.asarray(kps[start:end]),
        np.asarray(ndet[start:end]),
        str(fixture.video_id),
        gate_courts,
        gate_resolution,
        fixture.resolution,
        BODY_UNIT_HALF_WINDOW,
    )
    sticky = _empty_global_sticky(n_frames)
    for field_index, global_field in enumerate(sticky):
        local_field = local_sticky[field_index]
        global_field[start:end] = local_field
    guard_codes, _guard_info = grade_track(np.asarray(track))
    return WindowData(fixture, track, bboxes, sticky, segments, guard_codes, court_present, _spans_for_fixture(fixture))


def round_robin_by_fixture(candidates: Sequence[Candidate], count: int) -> list[Candidate]:
    """Choose stable candidates in a fixture round-robin, then key order."""
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in sorted(candidates, key=lambda item: (item.trace.fixture, item.trace.set_id, item.trace.rally, item.frame)):
        grouped[candidate.trace.fixture].append(candidate)
    selected: list[Candidate] = []
    index = 0
    while len(selected) < count:
        added = False
        for fixture in FIXTURE_ORDER:
            values = grouped[fixture]
            if index < len(values):
                selected.append(values[index])
                added = True
                if len(selected) == count:
                    break
        if not added:
            raise ValueError(f"only found {len(selected)} candidates, needed {count}")
        index += 1
    return selected


def build_candidates(rows: Sequence[TraceRow]) -> list[Candidate]:
    """Make the four strata using only trace fields and player attribution."""
    first_false: list[Candidate] = []
    selected_unavailable: list[Candidate] = []
    by_fixture: dict[str, list[TraceRow]] = defaultdict(list)
    for row in rows:
        if row.accepted and not row.outgoing[0]:
            by_fixture[row.fixture].append(row)
        if row.selected_frame is not None and row.pre_verdict == IncomingVerdict.UNAVAILABLE.value:
            if row.selected_player not in {"Top", "Bot"} or row.selected_rank is None:
                raise ValueError(f"{row.stable_key}: unavailable selection has incomplete player fields")
            player = "Bottom" if row.selected_player == "Bot" else "Top"
            selected_unavailable.append(Candidate("", row, row.selected_frame, row.selected_rank, player))
    for values in by_fixture.values():
        values.sort(key=lambda item: (item.set_id, item.rally, item.accepted[0]))

    fixture_by_name = {fixture.name: fixture for fixture in FIXTURES}
    offsets = {fixture: 0 for fixture in FIXTURE_ORDER}
    player_counts = {"Top": 0, "Bottom": 0}
    while min(player_counts.values()) < SAMPLE_PER_STRATUM:
        progressed = False
        for fixture_name in FIXTURE_ORDER:
            values = by_fixture[fixture_name]
            while offsets[fixture_name] < len(values):
                row = values[offsets[fixture_name]]
                offsets[fixture_name] += 1
                data = load_window_data(fixture_by_name[fixture_name], row.accepted[0])
                context = _contact_context(data, row.span_id, row.accepted[0])
                player = player_name(context.player) if context is not None else None
                del data
                gc.collect()
                if player is None or player_counts[player] >= SAMPLE_PER_STRATUM:
                    continue
                first_false.append(Candidate("", row, row.accepted[0], 1, player))
                player_counts[player] += 1
                progressed = True
                break
        if not progressed:
            raise ValueError(f"could not find five first outgoing-false contacts per player: {player_counts}")

    requested = (
        ("bottom_first_outgoing_false", first_false, "Bottom"),
        ("top_first_outgoing_false", first_false, "Top"),
        ("bottom_selected_pre_unavailable", selected_unavailable, "Bottom"),
        ("top_selected_pre_unavailable", selected_unavailable, "Top"),
    )
    chosen: list[Candidate] = []
    used_keys: set[str] = set()
    used_contacts: set[tuple[str, int]] = set()
    for stratum, pool, player in requested:
        eligible = [
            Candidate(stratum, item.trace, item.frame, item.accepted_rank, item.player)
            for item in pool
            if item.player == player
            and item.trace.stable_key not in used_keys
            and (item.trace.stable_key, item.frame) not in used_contacts
        ]
        selection = round_robin_by_fixture(eligible, SAMPLE_PER_STRATUM)
        chosen.extend(selection)
        used_keys.update(item.trace.stable_key for item in selection)
        used_contacts.update((item.trace.stable_key, item.frame) for item in selection)
    if len(chosen) != 20:
        raise ValueError(f"selected {len(chosen)} clips instead of 20")
    return chosen


def smoke_candidate(rows: Sequence[TraceRow]) -> Candidate:
    """Return the first deterministic outgoing-false sample without building all pools."""
    row = next(
        item
        for item in rows
        if item.fixture == "sset_01" and item.set_id == "set1" and item.rally == 8
    )
    fixture = next(item for item in FIXTURES if item.name == row.fixture)
    data = load_window_data(fixture, row.accepted[0])
    context = _contact_context(data, row.span_id, row.accepted[0])
    player = player_name(context.player) if context is not None else None
    del data
    gc.collect()
    if player != "Bottom" or row.outgoing[0]:
        raise ValueError("smoke candidate no longer matches the fixed outgoing-false population")
    return Candidate("bottom_first_outgoing_false", row, row.accepted[0], 1, player)


def usable_mask(data: WindowData, slot: Slot, span_id: int, frame: int) -> np.ndarray:
    """Rebuild the recurrence-clean local mask used by the fixed checks."""
    segment = data.segment_for_frame(frame)
    if segment is None:
        return np.zeros(len(data.track), dtype=bool)
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
    local = np.zeros(len(data.track), dtype=bool)
    local[max(span_start, segment_start):min(span_end, segment_end)] = True
    return usable & local


def failure_reason(run: object | None, motion: object | None, max_gap: int, direction: str) -> str:
    """Name the first fixed eligibility failure, keeping unavailable distinct."""
    if run is None:
        return "no run"
    frames = run_length(run)
    if frames < 5:
        return f"fewer than 5 frames ({frames})"
    gap = run_gap(run)
    if gap > max_gap:
        return f"local gap {gap} over scaled {max_gap}"
    ratio = float(getattr(motion, "largest_step_ratio", float("nan")))
    if ratio > 4.0:
        return f"largest-step ratio {ratio:.2f} over 4.0"
    return f"{direction} trend did not meet the fixed directional threshold"


def contact_details(data: WindowData, candidate: Candidate) -> ContactDetails:
    """Measure the selected local paths for the manifest and overlay."""
    context = _contact_context(data, candidate.trace.span_id, candidate.frame)
    max_gap = scaled_frames(MAX_LOCAL_GAP_BASE30_FRAMES, candidate.trace.fps)
    if context is None:
        empty = np.zeros(len(data.track), dtype=bool)
        return ContactDetails(candidate, empty, None, None, None, None, "missing player/scene", "missing player/scene", max_gap)
    lookaround = scaled_frames(LOOKAROUND_BASE30_FRAMES, candidate.trace.fps)
    usable = usable_mask(data, context.slot, candidate.trace.span_id, candidate.frame)
    pre_run = closest_pre_contact_run(usable, candidate.frame, lookaround)
    post_run = closest_post_contact_run(usable, candidate.frame, lookaround)
    pre_measurements = _measure_run(data, context, pre_run.start, pre_run.end) if pre_run else None
    post_measurements = _measure_run(data, context, post_run.start, post_run.end) if post_run else None
    pre_motion = pre_measurements[0] if pre_measurements else None
    post_motion = post_measurements[0] if post_measurements else None
    return ContactDetails(
        candidate,
        usable,
        pre_run,
        post_run,
        pre_motion,
        post_motion,
        failure_reason(pre_run, pre_motion, max_gap, "incoming") if candidate.stratum.endswith("unavailable") else "not sampled",
        failure_reason(post_run, post_motion, max_gap, "outgoing") if candidate.stratum.endswith("outgoing_false") else "not sampled",
        max_gap,
    )


def run_gap(run: object) -> int:
    """Return the pre or post run's contact distance with its shared meaning."""
    pre_gap = getattr(run, "frames_to_contact", None)
    return int(pre_gap) if pre_gap is not None else int(run.frames_from_contact)


def run_length(run: object | None) -> int | None:
    """Return a run's actual half-open source-frame length."""
    if run is None:
        return None
    return int(run.end - run.start)


def draw_text_panel(image: np.ndarray, lines: Sequence[str], style: HudStyle) -> None:
    """Draw a compact opaque panel on the right without obscuring the source HUD."""
    scale = style.font_scale
    thickness = max(1, style.text_thickness)
    sizes = [cv2.getTextSize(line, FONT, scale, thickness) for line in lines]
    width = max(size[0][0] for size in sizes) + style.padding * 2
    height = sum(size[0][1] + size[1] + style.padding for size in sizes) + style.padding
    left = image.shape[1] - width - style.inset
    top = style.inset
    cv2.rectangle(image, (left, top), (left + width, top + height), (0, 0, 0), -1)
    y = top + style.padding
    for line, (text_size, baseline) in zip(lines, sizes, strict=True):
        y += text_size[1]
        cv2.putText(image, line, (left + style.padding, y), FONT, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += baseline + style.padding


def make_draw(
    data: WindowData,
    details: ContactDetails,
    clip_start: int,
    clip_end: int,
    style: HudStyle,
    *,
    outgoing_verdict: bool | None = None,
    pre_verdict: str | None = None,
    reason_text: str | None = None,
) -> DrawFn:
    """Bind one contact's fixed evidence to the established overlay renderer."""
    candidate = details.candidate
    accepted = candidate.trace.accepted
    source_width, source_height = data.fixture.resolution
    trail_frames = 10

    def source_to_output(x: float, y: float) -> tuple[int, int]:
        return (
            round(x / source_width * style.output_width),
            round(y / source_height * style.output_height),
        )

    def draw(image: np.ndarray, source_idx: int, in_target_span: bool) -> list[str]:
        del in_target_span
        for frame in accepted:
            if clip_start <= frame <= clip_end:
                x = round((frame - clip_start) / max(1, clip_end - clip_start) * (style.output_width - 1))
                colour = FOCAL_COLOUR if frame == candidate.frame else OTHER_CONTACT_COLOUR
                thickness = 5 if frame == candidate.frame else 2
                cv2.line(image, (x, 0), (x, int(34 * style.scale)), colour, thickness)
                cv2.circle(image, (x, int(40 * style.scale)), 5 if frame == candidate.frame else 3, colour, -1)

        trail_start = max(clip_start, source_idx - trail_frames)
        trail_points: list[tuple[int, int]] = []
        previous_usable_point: tuple[int, int] | None = None
        for frame in range(trail_start, source_idx + 1):
            x, y, visible = data.track[frame]
            if visible == 1 and np.isfinite((x, y)).all():
                point = (round(x * style.output_width), round(y * style.output_height))
                if details.usable[frame]:
                    trail_points.append(point)
                    if previous_usable_point is not None:
                        cv2.line(image, previous_usable_point, point, SHUTTLE_COLOUR, max(1, int(3 * style.scale)))
                    previous_usable_point = point
                else:
                    cv2.drawMarker(image, point, UNUSABLE_COLOUR, cv2.MARKER_TILTED_CROSS, 12, 2)
                    previous_usable_point = None
        if trail_points:
            cv2.circle(image, trail_points[-1], max(4, int(8 * style.scale)), SHUTTLE_COLOUR, 2)

        slot = Slot.TOP if candidate.player == "Top" else Slot.BOTTOM
        picked = int(data.sticky.picks[source_idx, slot])
        if picked >= 0:
            x1, y1, x2, y2 = data.bboxes[source_idx, picked]
            if np.isfinite((x1, y1, x2, y2)).all():
                top_left = source_to_output(float(x1), float(y1))
                bottom_right = source_to_output(float(x2), float(y2))
                cv2.rectangle(image, top_left, bottom_right, ANCHOR_COLOUR, max(2, int(3 * style.scale)))
                anchor = data.sticky.ankle_pos[source_idx, slot]
                if np.isfinite(anchor).all():
                    anchor_point = (round(anchor[0] * style.output_width), round(anchor[1] * style.output_height))
                    cv2.drawMarker(image, anchor_point, ANCHOR_COLOUR, cv2.MARKER_CROSS, 16, 2)

        bar_top = image.shape[0] - int(36 * style.scale)
        for frame in range(clip_start, clip_end + 1):
            x = round((frame - clip_start) / max(1, clip_end - clip_start) * (style.output_width - 1))
            colour = USABLE_COLOUR if details.usable[frame] else UNUSABLE_COLOUR
            cv2.line(image, (x, bar_top), (x, image.shape[0] - 1), colour, 1)
        focus_x = round((candidate.frame - clip_start) / max(1, clip_end - clip_start) * (style.output_width - 1))
        cv2.line(image, (focus_x, bar_top), (focus_x, image.shape[0] - 1), FOCAL_COLOUR, 3)

        pre_text = run_text("pre", details.pre_run, details.pre_motion)
        post_text = run_text("post", details.post_run, details.post_motion)
        verdict = pre_verdict if pre_verdict is not None else candidate.trace.pre_verdict or "n/a"
        outgoing = outgoing_verdict if outgoing_verdict is not None else candidate.trace.outgoing[candidate.accepted_rank - 1]
        guard_code = int(data.guard_codes[source_idx])
        lines = [
            f"source f{source_idx}; rel {source_idx - candidate.frame:+d}",
            f"FOCAL f{candidate.frame}; rank {candidate.accepted_rank}; {candidate.player}",
            f"recurrence clean: {'usable' if details.usable[source_idx] else 'unusable'}",
            f"guard={guard_code} ({CODE_NAMES[guard_code]}); used by calculation: {'yes' if details.usable[source_idx] else 'no'}",
            pre_text,
            post_text,
            f"outgoing={outgoing}; pre={verdict}",
            f"reason: {reason_text or (details.pre_failure if candidate.stratum.endswith('unavailable') else details.post_failure)}",
        ]
        draw_text_panel(image, lines, style)
        return []

    return draw


def run_text(label: str, run: object | None, motion: object | None) -> str:
    """Format a local run's source-frame bounds and essential fixed metrics."""
    if run is None:
        return f"{label}: no run"
    gap = run_gap(run)
    frames = run_length(run)
    ratio = float(getattr(motion, "largest_step_ratio", float("nan"))) if motion is not None else float("nan")
    return f"{label}: [{run.start},{run.end}) n={frames} gap={gap} ratio={ratio:.2f}"


def manifest_row(details: ContactDetails, clip_start: int, clip_end: int, filename: str) -> dict[str, object]:
    """Return a row whose GT-free diagnostics remain separate from scoring."""
    candidate = details.candidate
    trace = candidate.trace
    return {
        "stratum": candidate.stratum,
        "clip_file": filename,
        "stable_rally_key": trace.stable_key,
        "span_id": trace.span_id,
        "contact_frame": candidate.frame,
        "accepted_rank": candidate.accepted_rank,
        "player": candidate.player,
        "clip_start_frame": clip_start,
        "clip_end_frame": clip_end,
        "binary_outgoing_verdict": trace.outgoing[candidate.accepted_rank - 1],
        "three_way_pre_verdict": trace.pre_verdict or "not_applicable",
        "sampled_failure_reason": details.pre_failure if candidate.stratum.endswith("unavailable") else details.post_failure,
        "pre_run": run_text("pre", details.pre_run, details.pre_motion),
        "post_run": run_text("post", details.post_run, details.post_motion),
        "pre_frames": run_length(details.pre_run) or "",
        "pre_gap": getattr(details.pre_run, "frames_to_contact", ""),
        "pre_largest_step_ratio": getattr(details.pre_motion, "largest_step_ratio", ""),
        "post_frames": run_length(details.post_run) or "",
        "post_gap": getattr(details.post_run, "frames_from_contact", ""),
        "post_largest_step_ratio": getattr(details.post_motion, "largest_step_ratio", ""),
        "scoring_only_gt_stroke_frames": "not included",
    }


def write_manifest(rows: Sequence[dict[str, object]]) -> None:
    """Write the human-readable sample crosswalk."""
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_readme(rows: Sequence[dict[str, object]]) -> None:
    """Explain clip order, selection, and the on-frame visual legend."""
    counts: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        counts[str(row["stratum"])].append(str(row["clip_file"]))
    clip_description = (
        "Each 1920x1080 clip has approximately 60 base-30fps frames on either side of the focal contact, "
        "scaled to the fixture FPS and clipped to source-video bounds. The standalone clips use a 32 px "
        "reference HUD and the same libx264 fast/CRF 18 base encoding as the comparison panels."
    )
    body = [
        "# Accepted-contact diagnostic overlay sample",
        "",
        "Twenty clips selected deterministically from the fixed 239 one-to-one rows. Selection uses only accepted-contact order, binary outgoing verdicts, three-way pre verdicts, and player attribution. The GT stroke frames were not used and are not included here.",
        "",
        "## Viewing order",
        "",
    ]
    for stratum, files in counts.items():
        body.append(f"- `{stratum}`: {', '.join(f'`{name}`' for name in files)}")
    body.extend([
        "",
        "The deterministic selector cycles fixtures in `sset_01`, `sset_15`, `sset_21` order, then uses set/rally/contact order. It excludes any rally already chosen by an earlier stratum, so no clip shares a rally or focal contact.",
        "",
        "## Legend",
        "",
        "- Cyan thick tick and circle: focal accepted contact. Yellow thin ticks and circles: other accepted impulses in the clip.",
        "- Magenta line and circle: recurrence-clean usable TrackNet trail and current point. The line never crosses an unusable frame.",
        "- Dark grey X: raw TrackNet reported a point, but recurrence-clean rejected it. These points never join the magenta trail or enter a calculation.",
        "- Blue-orange rectangle and cross: focal player's sticky picked bbox and ankle anchor, when available.",
        "- Bottom strip: light grey means recurrence-clean usable. Dark grey means unusable. The cyan vertical line is the focal frame.",
        "- Right panel: local pre/post runs use half-open source-frame bounds. It shows the inpaint-guard code/name and whether the current frame enters the recurrence-clean calculation. `pre=unavailable` means the pre-contact path failed a common eligibility check. It does not mean `not_incoming`.",
        "",
        "The overlay deliberately does not apply the separate `producer_inpaint` sidecar. The fixed primary calculation uses the PR #82 `recurrence_clean` mask, including its inpaint guard, rather than the `producer_original` path.",
        "",
        clip_description,
        "",
        "## How the fixed checks work",
        "",
        "A frame is recurrence-clean usable only when TrackNet marked it visible with finite, non-zero coordinates; the court is present; the focal player has a finite positive sticky bbox height and finite shuttle-to-player distance; and the recomputed recurrence guard is clear. The matching local path must then have at least five frames, reach the contact within the scaled two-frame gap, and have a largest-step ratio no greater than 4.0. A three-way pre result is `unavailable` when that shared path eligibility fails. `not_incoming` is a different result and is never used as a synonym for unavailable.",
        "",
        "## Per-video guide",
        "",
    ])
    for row in rows:
        stratum = str(row["stratum"])
        if stratum.endswith("outgoing_false"):
            selection = "It is the first accepted contact for the named player whose fixed binary outgoing check is false."
            inspect = "Inspect whether a real outgoing shot was wrongly skipped, then compare the post-contact run and recurrence-clean strip with the stated failure."
        else:
            selection = "It is the selected accepted contact for the named player whose fixed three-way pre verdict is unavailable."
            inspect = "Inspect whether incoming or not-incoming looks visually obvious despite the stated metric ineligibility, then compare the pre-contact run and recurrence-clean strip."
        body.extend([
            f"### `{row['clip_file']}`",
            "",
            f"{selection} It occupies its deterministic fixture-round-robin position within `{stratum}`.",
            "",
            f"The failed fixed check is: `{row['sampled_failure_reason']}`.",
            "",
            inspect,
            "",
        ])
    README_PATH.write_text("\n".join(body) + "\n", encoding="utf-8")


def record_manifest_row(index: int) -> None:
    """Calculate one manifest row in a fresh process without encoding a clip."""
    selection_path = ROW_CACHE_DIR / "selection.json"
    candidates = load_selection(selection_path)
    if not 1 <= index <= len(candidates):
        raise ValueError(f"clip index must be in [1, {len(candidates)}], got {index}")
    candidate = candidates[index - 1]
    fixture = next(item for item in FIXTURES if item.name == candidate.trace.fixture)
    data = load_window_data(fixture, candidate.frame)
    details = contact_details(data, candidate)
    info = probe_video(VIDEO_PATHS[candidate.trace.fixture])
    half_window = scaled_frames(CLIP_HALF_WINDOW_BASE30, candidate.trace.fps)
    clip_start = max(0, candidate.frame - half_window)
    clip_end = min(info.nb_frames - 1, candidate.frame + half_window)
    filename = f"{index:02d}_{candidate.stratum}_{candidate.trace.fixture}_{candidate.trace.set_id}_r{candidate.trace.rally}_f{candidate.frame}.mp4"
    ROW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (ROW_CACHE_DIR / f"{index:02d}.json").write_text(
        json.dumps(manifest_row(details, clip_start, clip_end, filename), sort_keys=True), encoding="utf-8"
    )


def write_selection() -> None:
    """Freeze the selected GT-free candidates for fresh-process manifest rows."""
    rows = []
    for candidate in build_candidates(read_trace_rows()):
        rows.append(
            {
                "stratum": candidate.stratum,
                "trace": {
                    "fixture": candidate.trace.fixture,
                    "video_id": candidate.trace.video_id,
                    "set_id": candidate.trace.set_id,
                    "rally": candidate.trace.rally,
                    "span_id": candidate.trace.span_id,
                    "fps": candidate.trace.fps,
                    "accepted": candidate.trace.accepted,
                    "outgoing": candidate.trace.outgoing,
                    "selected_frame": candidate.trace.selected_frame,
                    "selected_rank": candidate.trace.selected_rank,
                    "selected_player": candidate.trace.selected_player,
                    "pre_verdict": candidate.trace.pre_verdict,
                },
                "frame": candidate.frame,
                "accepted_rank": candidate.accepted_rank,
                "player": candidate.player,
            }
        )
    ROW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (ROW_CACHE_DIR / "selection.json").write_text(json.dumps(rows), encoding="utf-8")


def load_selection(path: Path) -> list[Candidate]:
    """Read the frozen selection without rerunning its player-attribution scan."""
    if not path.is_file():
        raise ValueError("selection cache is missing; run --write-selection first")
    selected: list[Candidate] = []
    for raw in json.loads(path.read_text(encoding="utf-8")):
        trace = TraceRow(**raw["trace"])
        selected.append(Candidate(raw["stratum"], trace, raw["frame"], raw["accepted_rank"], raw["player"]))
    if len(selected) != 20:
        raise ValueError(f"selection cache must have 20 candidates, got {len(selected)}")
    return selected


def assemble_manifest() -> None:
    """Combine fresh-process metric rows into the final CSV and README."""
    paths = [ROW_CACHE_DIR / f"{index:02d}.json" for index in range(1, 21)]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"missing cached manifest rows: {missing}")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    write_manifest(rows)
    write_readme(rows)


def render_comparison(index: int) -> None:
    """Render one literal retained-stride-8 and reconstructed-stride-1 comparison."""
    candidate = load_selection(ROW_CACHE_DIR / "selection.json")[index - 1]
    if candidate.trace.fixture != "sset_01":
        raise ValueError(f"comparison index {index} is not an sset_01 sample")
    fixture = next(item for item in FIXTURES if item.name == "sset_01")
    info = probe_video(VIDEO_PATHS[fixture.name])
    half_window = scaled_frames(CLIP_HALF_WINDOW_BASE30, candidate.trace.fps)
    clip_start = max(0, candidate.frame - half_window)
    clip_end = min(info.nb_frames - 1, candidate.frame + half_window)
    stride1 = np.load(STRIDE1_TRACK, mmap_mode="r", allow_pickle=False)
    if stride1.shape != (info.nb_frames, 3):
        raise ValueError(f"stride-1 track is not frame-aligned: {stride1.shape} vs {(info.nb_frames, 3)}")
    stride8_data = load_window_data(fixture, candidate.frame)
    stride8_details = contact_details(stride8_data, candidate)
    data = load_window_data(fixture, candidate.frame, track_path=STRIDE1_TRACK, release_dir=STRIDE1_RELEASE_DIR)
    details = contact_details(data, candidate)
    context = _contact_context(data, candidate.trace.span_id, candidate.frame)
    if context is None:
        stride1_pre = IncomingVerdict.UNAVAILABLE.value
        stride1_outgoing = False
        pre_trend = None
        post_trend = None
    else:
        pre_measurement = _measure_run(data, context, details.pre_run.start, details.pre_run.end) if details.pre_run else None
        post_measurement = _measure_run(data, context, details.post_run.start, details.post_run.end) if details.post_run else None
        pre_motion, pre_trend = pre_measurement if pre_measurement else (None, None)
        post_motion, post_trend = post_measurement if post_measurement else (None, None)
        if details.pre_run is None or pre_motion is None or pre_trend is None:
            stride1_pre = IncomingVerdict.UNAVAILABLE.value
        else:
            decisions = decide_fixed_motion_rules(pre_motion, pre_trend, details.pre_run.frames_to_contact, details.max_gap)
            stride1_pre = (
                IncomingVerdict.INCOMING.value if decisions.robust_trend_incoming
                else IncomingVerdict.NOT_INCOMING.value if decisions.common_path_eligible
                else IncomingVerdict.UNAVAILABLE.value
            )
        stride1_outgoing = has_credible_outgoing_motion(
            post_motion, post_trend,
            details.post_run.frames_from_contact if details.post_run else None,
            details.max_gap,
        )
    stride1_pre_status = (
        details.pre_failure if stride1_pre == IncomingVerdict.UNAVAILABLE.value
        else f"eligible; {'incoming' if stride1_pre == IncomingVerdict.INCOMING.value else 'not incoming'}"
    )
    stride1_outgoing_status = "credible outgoing" if stride1_outgoing else details.post_failure
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    left_render = Path("/tmp") / f"stride8_overlay_{index:02d}.mp4"
    right = Path("/tmp") / f"stride1_overlay_{index:02d}.mp4"
    left_plan = make_render_plan(
        info,
        [Segment(clip_start, clip_end, candidate.trace.stable_key)],
        left_render,
        render_width=COMPARISON_PANEL_WIDTH,
        hud_height=HUD_HEIGHT,
        lead_in=0,
        lead_out=0,
        spacer=0,
    )
    render(left_plan, make_draw(stride8_data, stride8_details, clip_start, clip_end, left_plan.hud_style))
    plan = make_render_plan(
        info,
        [Segment(clip_start, clip_end, candidate.trace.stable_key)],
        right,
        render_width=COMPARISON_PANEL_WIDTH,
        hud_height=HUD_HEIGHT,
        lead_in=0,
        lead_out=0,
        spacer=0,
    )
    reason_text = stride1_pre_status if candidate.stratum.endswith("unavailable") else stride1_outgoing_status
    render(plan, make_draw(data, details, clip_start, clip_end, plan.hud_style, outgoing_verdict=stride1_outgoing, pre_verdict=stride1_pre, reason_text=reason_text))
    left = left_render
    output = COMPARISON_DIR / f"{index:02d}_stride8_vs_stride1_{candidate.trace.set_id}_r{candidate.trace.rally}_f{candidate.frame}.mp4"
    command = [
        "ffmpeg", "-y", "-v", "error", "-i", str(left), "-i", str(right), "-filter_complex",
        (
            "[0:v]drawtext=text='stride 8':x=16:y=16:fontsize=30:fontcolor=white:box=1:boxcolor=black@0.8[left];"
            "[1:v]drawtext=text='stride 1':x=16:y=16:fontsize=30:fontcolor=white:box=1:boxcolor=black@0.8[right];"
            "[left][right]hstack=inputs=2"
        ),
        "-an", str(output),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg hstack failed with status {completed.returncode}")
    right.unlink(missing_ok=True)
    left_render.unlink(missing_ok=True)
    row = {
        "comparison_file": output.name,
        "source_clip": f"{index:02d}_{candidate.stratum}_{candidate.trace.fixture}_{candidate.trace.set_id}_r{candidate.trace.rally}_f{candidate.frame}.mp4",
        "stable_rally_key": candidate.trace.stable_key,
        "contact_frame": candidate.frame,
        "stride1_pre_run": run_text("pre", details.pre_run, details.pre_motion),
        "stride1_post_run": run_text("post", details.post_run, details.post_motion),
        "stride1_pre_status": stride1_pre_status,
        "stride1_outgoing_status": stride1_outgoing_status,
        "stride8_outgoing": candidate.trace.outgoing[candidate.accepted_rank - 1],
        "stride8_pre": candidate.trace.pre_verdict or "not_applicable",
        "stride1_outgoing": stride1_outgoing,
        "stride1_pre": stride1_pre,
        "stride1_pre_fitted_decrease_bh": getattr(pre_trend, "fitted_decrease_bh", ""),
        "stride1_post_fitted_decrease_bh": getattr(post_trend, "fitted_decrease_bh", ""),
    }
    (ROW_CACHE_DIR / f"comparison_{index:02d}.json").write_text(json.dumps(row), encoding="utf-8")


def assemble_comparisons() -> None:
    """Write the comparison manifest after all eight fresh-process renders."""
    indexes = (1, 4, 6, 9, 11, 14, 16, 19)
    rows = [json.loads((ROW_CACHE_DIR / f"comparison_{index:02d}.json").read_text(encoding="utf-8")) for index in indexes]
    for row in rows:
        if "stride1_pre_status" not in row:
            row["stride1_pre_status"] = (
                row["stride1_pre_failure"] if row["stride1_pre_failure"] != "not sampled"
                else "unavailable; see pre run"
                if row["stride1_pre"] == IncomingVerdict.UNAVAILABLE.value
                else f"eligible; {'incoming' if row['stride1_pre'] == IncomingVerdict.INCOMING.value else 'not incoming'}"
            )
            row["stride1_outgoing_status"] = (
                "credible outgoing" if row["stride1_outgoing"] else row["stride1_post_failure"]
            )
            del row["stride1_pre_failure"]
            del row["stride1_post_failure"]
    path = COMPARISON_DIR / "manifest.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    existing = README_PATH.read_text(encoding="utf-8")
    comparison_header = "\n## Stride-8 and stride-1 comparisons\n"
    if comparison_header in existing:
        README_PATH.write_text(existing.split(comparison_header, maxsplit=1)[0].rstrip() + "\n", encoding="utf-8")
    with README_PATH.open("a", encoding="utf-8") as handle:
        handle.write("\n## Stride-8 and stride-1 comparisons\n\n")
        handle.write(
            "These eight files show a fresh large-HUD stride-8 rerender on the left and an "
            "independently reconstructed stride-1 WEIGHT-track calculation on the right. Both panels use "
            "the same source-frame range and focal accepted contact. The stride-1 track has 154393 rows, "
            "matching sset_01 and the source video. The right panel recomputes the recurrence guard, usable "
            "mask and local runs. It does not apply the producer inpaint sidecar.\n\n"
        )
        for row in rows:
            handle.write(
                f"- `{row['comparison_file']}`: outgoing `{row['stride8_outgoing']}` to "
                f"`{row['stride1_outgoing']}` ({row['stride1_outgoing_status']}); pre "
                f"`{row['stride8_pre']}` to `{row['stride1_pre']}` ({row['stride1_pre_status']}). "
                f"Stride-1 runs: pre `{row['stride1_pre_run']}`; post `{row['stride1_post_run']}`.\n"
            )


def render_sample(*, smoke_only: bool = False, index: int | None = None) -> None:
    """Select, render and document the requested sample."""
    rows = read_trace_rows()
    if smoke_only:
        candidates = [smoke_candidate(rows)]
    elif index is not None:
        candidates = load_selection(ROW_CACHE_DIR / "selection.json")
    else:
        candidates = build_candidates(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    if index is not None and not 1 <= index <= len(candidates):
        raise ValueError(f"clip index must be in [1, {len(candidates)}], got {index}")
    selected = candidates[:1] if smoke_only else candidates
    for order, candidate in enumerate(selected, start=1):
        if index is not None and order != index:
            continue
        fixture = next(fixture for fixture in FIXTURES if fixture.name == candidate.trace.fixture)
        data = load_window_data(fixture, candidate.frame)
        details = contact_details(data, candidate)
        source_video = VIDEO_PATHS[candidate.trace.fixture]
        info = probe_video(source_video)
        if info.nb_frames != len(data.track):
            raise ValueError(f"{candidate.trace.fixture}: video/track frame counts differ")
        half_window = scaled_frames(CLIP_HALF_WINDOW_BASE30, candidate.trace.fps)
        clip_start = max(0, candidate.frame - half_window)
        clip_end = min(info.nb_frames - 1, candidate.frame + half_window)
        filename = f"{order:02d}_{candidate.stratum}_{candidate.trace.fixture}_{candidate.trace.set_id}_r{candidate.trace.rally}_f{candidate.frame}.mp4"
        output = OUTPUT_DIR / filename
        plan = make_render_plan(
            info,
            [Segment(clip_start, clip_end, candidate.trace.stable_key)],
            output,
            render_width=SAMPLE_RENDER_WIDTH,
            hud_height=HUD_HEIGHT,
            lead_in=0,
            lead_out=0,
            spacer=0,
            verify=smoke_only,
        )
        result = render(plan, make_draw(data, details, clip_start, clip_end, plan.hud_style))
        expected = clip_end - clip_start + 1
        if result.output_frames != expected:
            raise RuntimeError(f"{output}: wrote {result.output_frames}, expected {expected}")
        manifest.append(manifest_row(details, clip_start, clip_end, filename))
        print(f"rendered {output} ({result.output_frames} frames)")
        del data
        gc.collect()
    if not smoke_only and index is None:
        write_manifest(manifest)
        write_readme(manifest)


def main(argv: Iterable[str] | None = None) -> int:
    """Run either a one-clip identity-gated smoke or the full 20-clip render."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="render only the first clip with identity verification")
    parser.add_argument("--index", type=int, help="render one 1-based clip index in a fresh process")
    parser.add_argument("--record-index", type=int, help="write one fresh-process manifest metric row")
    parser.add_argument("--write-selection", action="store_true", help="freeze fresh-process manifest candidates")
    parser.add_argument("--assemble-manifest", action="store_true", help="write manifest and README from metric rows")
    parser.add_argument("--comparison-index", type=int, help="render one sset_01 stride comparison")
    parser.add_argument("--assemble-comparisons", action="store_true", help="write stride comparison manifest")
    arguments = parser.parse_args(argv)
    if arguments.smoke:
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    modes = sum(bool(value) for value in (arguments.smoke, arguments.index, arguments.record_index, arguments.write_selection, arguments.assemble_manifest, arguments.comparison_index, arguments.assemble_comparisons))
    if modes > 1:
        parser.error("choose one execution mode")
    if arguments.record_index is not None:
        record_manifest_row(arguments.record_index)
        return 0
    if arguments.write_selection:
        write_selection()
        return 0
    if arguments.assemble_manifest:
        assemble_manifest()
        return 0
    if arguments.comparison_index is not None:
        render_comparison(arguments.comparison_index)
        return 0
    if arguments.assemble_comparisons:
        assemble_comparisons()
        return 0
    render_sample(smoke_only=arguments.smoke, index=arguments.index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
