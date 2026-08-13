"""Freeze PR #82 and PR #88 predictions without reading ShuttleSet labels."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from enum import StrEnum
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from annotator import point_winner
from annotator.fps_constants import ScalingKind, scale_for_fps
from annotator.inpaint_guard import DEGRADED, FABRICATED, NO_FLAG, SUSPECT_FLAT, grade_track
from annotator.rally.evidence import build_sticky_result, tracker_segments
from annotator.types import Slot, StickyResult, true_runs
from dataset_builder.vision import PoseArrays, load_court_vision, load_npy_xz
from scratch.serve_start_trajectory_exploration.trajectory_features import (
    IncomingMotion,
    RobustDistanceTrend,
    closest_pre_contact_run,
    decide_fixed_motion_rules,
    fit_robust_distance_trend,
    measure_incoming_motion,
)


LOOKAROUND_BASE30_FRAMES = 30
MAX_LOCAL_GAP_BASE30_FRAMES = 2
H3_HALO_SOURCE_FRAMES = 3
PRODUCTION_HALO_SOURCE_FRAMES = 15
H3_MAX_LARGEST_STEP_RATIO = 8.0
MIN_PATH_FRAMES = 5
MIN_DIRECTIONAL_CHANGE_BH = 0.05


@dataclass(frozen=True)
class VideoSpec:
    """Validated canonical metadata fixed before prediction generation."""

    video_id: str
    fps: float
    frame_count: int
    width: int
    height: int


VIDEO_SPECS = (
    VideoSpec("sset_20", 25.0, 81_650, 1920, 1080),
    VideoSpec("sset_22", 30.0, 100_896, 1920, 1080),
)


def load_pose_arrays(data_root: Path, video_id: str) -> PoseArrays:
    """Load raw pose arrays without importing the GPU court runner."""
    run_id = f"issue90_{video_id.replace('_', '')}_v2"
    root = data_root / "stages" / "pose_raw_v2" / video_id / f"publish_{run_id}"
    return PoseArrays(
        kps=np.load(root / "pose_raw_kps.npy", allow_pickle=False),
        bboxes=np.load(root / "pose_raw_bboxes.npy", allow_pickle=False),
        scores=np.load(root / "pose_raw_scores.npy", allow_pickle=False),
        kp_scores=np.load(root / "pose_raw_kp_scores.npy", allow_pickle=False),
        ndet=np.load(root / "pose_raw_ndet.npy", allow_pickle=False),
    )


class PreVerdict(StrEnum):
    """Frozen pre-contact trajectory verdict."""

    INCOMING = "incoming"
    NOT_INCOMING = "not_incoming"
    UNAVAILABLE = "unavailable"


class SearchCategory(StrEnum):
    """Terminal PR #88 outgoing-search category."""

    VISIBLE_SERVE = "visible_serve"
    FIRST_VISIBLE_POST_SERVE = "first_visible_post_serve_contact"
    NOT_ENOUGH_TRAJECTORY = "not_enough_shuttle_trajectory_to_tell"
    NO_CREDIBLE_CONTACT = "no_credible_accepted_contact"


@dataclass(frozen=True)
class ContactContext:
    """Player identity and usable path mask for one accepted contact."""

    player: point_winner.Half
    slot: Slot
    usable: np.ndarray


@dataclass(frozen=True)
class PredictionInputs:
    """All label-blind arrays and annotation fields for one video."""

    spec: VideoSpec
    track: np.ndarray
    pose: PoseArrays
    sticky: StickyResult
    segments: tuple[tuple[int, int], ...]
    production_guard_codes: np.ndarray
    h3_guard_codes: np.ndarray
    court_present: np.ndarray
    net_band: tuple[float, float]
    spans: tuple[tuple[int, int], ...]
    accepted_by_span: dict[int, tuple[int, ...]]


@dataclass(frozen=True)
class PredictionRow:
    """One frozen span prediction with no ground-truth fields."""

    video_id: str
    span_id: int
    fps: float
    span_start: int
    span_end: int
    accepted_contact_frames: str
    pr82_anchor_frame: int | None
    pr82_anchor_player: str | None
    pr82_pre_verdict: str
    pr82_server: str | None
    pr88_category: str
    pr88_selected_frame: int | None
    pr88_selected_rank: int | None
    pr88_selected_player: str | None
    pr88_pre_verdict: str | None
    pr88_credible_outgoing: str
    pr88_branch: str
    pr88_server: str | None


@dataclass(frozen=True)
class MeasuredPath:
    """One local path and its distance trend."""

    motion: IncomingMotion
    trend: RobustDistanceTrend
    contact_gap: int


def other_side(player: point_winner.Half) -> point_winner.Half:
    """Return the opposite court half."""
    return point_winner.OTHER_HALF[player]


def rebuild_guard_codes(track: np.ndarray, existing_codes: np.ndarray, halo_frames: int) -> np.ndarray:
    """Rebuild recurrence grades with a fixed literal source-frame halo."""
    core = (existing_codes == FABRICATED) | (existing_codes == SUSPECT_FLAT)
    halo = np.zeros(len(track), dtype=bool)
    edges = np.diff(np.concatenate(([False], core, [False])).astype(np.int8))
    for start in np.flatnonzero(edges == 1):
        halo[max(0, int(start) - halo_frames) : int(start)] = True
    for stop in np.flatnonzero(edges == -1):
        halo[int(stop) : min(len(track), int(stop) + halo_frames)] = True

    on_attractor = np.zeros(len(track), dtype=bool)
    for pos_x, pos_y in np.unique(track[core, :2], axis=0):
        on_attractor |= (track[:, 0] == pos_x) & (track[:, 1] == pos_y)

    rebuilt = np.zeros(len(track), dtype=np.uint8)
    rebuilt[(halo | on_attractor) & ~core] = DEGRADED
    rebuilt[existing_codes == SUSPECT_FLAT] = SUSPECT_FLAT
    rebuilt[existing_codes == FABRICATED] = FABRICATED
    stored_blank = (track[:, 0] == 0) & (track[:, 1] == 0)
    rebuilt[stored_blank] = NO_FLAG
    return rebuilt


def closest_post_contact_run(
    usable: np.ndarray, contact_frame: int, lookahead_frames: int
) -> tuple[int, int, int] | None:
    """Return the earliest usable run after a contact and its contact gap."""
    window_start = contact_frame + 1
    window_end = min(len(usable), contact_frame + lookahead_frames + 1)
    runs = true_runs(usable[window_start:window_end])
    if not runs:
        return None
    relative_start, relative_end = runs[0]
    start = window_start + relative_start
    end = window_start + relative_end
    return start, end, start - contact_frame


def _segment_for_frame(inputs: PredictionInputs, frame: int) -> tuple[int, int] | None:
    for start, end in inputs.segments:
        if start <= frame < end:
            return start, end
    return None


def contact_context(
    inputs: PredictionInputs,
    span_id: int,
    contact_frame: int,
    guard_codes: np.ndarray,
) -> ContactContext | None:
    """Build the fixed player-specific recurrence-clean path mask."""
    player = point_winner.attribute_half(
        contact_frame,
        inputs.track,
        inputs.sticky,
        inputs.pose.bboxes,
        inputs.net_band,
    )
    segment = _segment_for_frame(inputs, contact_frame)
    if player is None or segment is None:
        return None

    slot = Slot.TOP if player is point_winner.Half.TOP else Slot.BOTTOM
    coordinate_valid = np.isfinite(inputs.track[:, :2]).all(axis=1)
    coordinate_valid &= ~((inputs.track[:, 0] == 0) & (inputs.track[:, 1] == 0))
    usable = (
        (inputs.track[:, 2] == 1)
        & coordinate_valid
        & inputs.court_present
        & np.isfinite(inputs.sticky.distances_per_slot[:, slot])
        & np.isfinite(inputs.sticky.bbox_height[:, slot])
        & (inputs.sticky.bbox_height[:, slot] > 0)
        & (guard_codes == NO_FLAG)
    )
    span_start, span_end = inputs.spans[span_id]
    segment_start, segment_end = segment
    local = np.zeros(len(inputs.track), dtype=bool)
    local[max(span_start, segment_start) : min(span_end, segment_end)] = True
    return ContactContext(player, slot, usable & local)


def measure_path(
    inputs: PredictionInputs,
    context: ContactContext | None,
    contact_frame: int,
    lookaround_frames: int,
    *,
    before_contact: bool,
) -> MeasuredPath | None:
    """Measure the closest strict local path on one side of a contact."""
    if context is None:
        return None
    if before_contact:
        run = closest_pre_contact_run(context.usable, contact_frame, lookaround_frames)
        if run is None:
            return None
        start, end, gap = run.start, run.end, run.frames_to_contact
    else:
        run = closest_post_contact_run(context.usable, contact_frame, lookaround_frames)
        if run is None:
            return None
        start, end, gap = run
    if end - start < 2:
        return None

    run_slice = slice(start, end)
    distances = inputs.sticky.distances_per_slot[run_slice, context.slot]
    motion = measure_incoming_motion(
        distances,
        inputs.track[run_slice, :2],
        inputs.sticky.bbox_height[run_slice, context.slot],
        (float(inputs.spec.width), float(inputs.spec.height)),
    )
    return MeasuredPath(motion, fit_robust_distance_trend(distances), gap)


def h3_verdict(path: MeasuredPath | None, maximum_gap: int) -> PreVerdict:
    """Apply the frozen H3/R8 pre-contact rule."""
    if path is None:
        return PreVerdict.UNAVAILABLE
    eligible = (
        path.motion.n_frames >= MIN_PATH_FRAMES
        and path.contact_gap <= maximum_gap
        and path.motion.largest_step_ratio <= H3_MAX_LARGEST_STEP_RATIO
    )
    if not eligible:
        return PreVerdict.UNAVAILABLE
    if path.trend.fitted_decrease_bh >= MIN_DIRECTIONAL_CHANGE_BH:
        return PreVerdict.INCOMING
    return PreVerdict.NOT_INCOMING


def h3_outgoing(path: MeasuredPath | None, maximum_gap: int) -> bool:
    """Apply the frozen H3/R8 outgoing predicate."""
    if path is None:
        return False
    return (
        path.motion.n_frames >= MIN_PATH_FRAMES
        and path.contact_gap <= maximum_gap
        and path.motion.largest_step_ratio <= H3_MAX_LARGEST_STEP_RATIO
        and path.trend.fitted_decrease_bh <= -MIN_DIRECTIONAL_CHANGE_BH
    )


def pr82_prediction(
    inputs: PredictionInputs,
    span_id: int,
    accepted: tuple[int, ...],
    lookaround_frames: int,
    maximum_gap: int,
) -> tuple[int | None, str | None, PreVerdict, point_winner.Half | None]:
    """Apply the frozen PR #82 rank-one incoming-motion rule."""
    if not accepted:
        return None, None, PreVerdict.UNAVAILABLE, None
    anchor = accepted[0]
    context = contact_context(inputs, span_id, anchor, inputs.production_guard_codes)
    if context is None:
        return anchor, None, PreVerdict.UNAVAILABLE, None
    path = measure_path(inputs, context, anchor, lookaround_frames, before_contact=True)
    verdict = PreVerdict.UNAVAILABLE
    if path is not None:
        decisions = decide_fixed_motion_rules(
            path.motion,
            path.trend,
            path.contact_gap,
            maximum_gap,
        )
        if decisions.common_path_eligible:
            verdict = (
                PreVerdict.INCOMING
                if decisions.robust_trend_incoming
                else PreVerdict.NOT_INCOMING
            )
    server = other_side(context.player) if verdict is PreVerdict.INCOMING else context.player
    return anchor, context.player.value, verdict, server


def build_prediction_row(inputs: PredictionInputs, span_id: int) -> PredictionRow:
    """Freeze both rule outputs for one predicted span."""
    accepted = tuple(sorted(inputs.accepted_by_span.get(span_id, ())))
    span_start, span_end = inputs.spans[span_id]
    if any(not span_start <= frame < span_end for frame in accepted):
        raise ValueError(f"{inputs.spec.video_id} span {span_id}: contact outside span")
    lookaround = int(ScalingKind.FRAME_COUNT.scale(LOOKAROUND_BASE30_FRAMES, inputs.spec.fps))
    maximum_gap = int(ScalingKind.FRAME_COUNT.scale(MAX_LOCAL_GAP_BASE30_FRAMES, inputs.spec.fps))
    anchor, anchor_player, baseline_verdict, baseline_server = pr82_prediction(
        inputs,
        span_id,
        accepted,
        lookaround,
        maximum_gap,
    )

    selected_frame: int | None = None
    selected_rank: int | None = None
    selected_player: point_winner.Half | None = None
    selected_verdict: PreVerdict | None = None
    outgoing_values: list[bool] = []
    for rank, frame in enumerate(accepted, start=1):
        context = contact_context(inputs, span_id, frame, inputs.h3_guard_codes)
        pre_path = measure_path(inputs, context, frame, lookaround, before_contact=True)
        post_path = measure_path(inputs, context, frame, lookaround, before_contact=False)
        credible = h3_outgoing(post_path, maximum_gap)
        outgoing_values.append(credible)
        if selected_frame is None and credible:
            selected_frame = frame
            selected_rank = rank
            selected_player = context.player if context is not None else None
            selected_verdict = h3_verdict(pre_path, maximum_gap)

    if selected_frame is None:
        category = SearchCategory.NO_CREDIBLE_CONTACT
    elif selected_verdict is PreVerdict.INCOMING:
        category = SearchCategory.FIRST_VISIBLE_POST_SERVE
    elif selected_verdict is PreVerdict.NOT_INCOMING:
        category = SearchCategory.VISIBLE_SERVE
    else:
        category = SearchCategory.NOT_ENOUGH_TRAJECTORY

    if category is SearchCategory.FIRST_VISIBLE_POST_SERVE and selected_player is not None:
        branch = "selected_outgoing_contact__incoming__other_side"
        preferred_server = other_side(selected_player)
    elif category is SearchCategory.VISIBLE_SERVE and selected_player is not None:
        branch = "selected_outgoing_contact__not_incoming__selected_side"
        preferred_server = selected_player
    else:
        branch = "pr82_fallback"
        preferred_server = baseline_server

    return PredictionRow(
        video_id=inputs.spec.video_id,
        span_id=span_id,
        fps=inputs.spec.fps,
        span_start=span_start,
        span_end=span_end,
        accepted_contact_frames=json.dumps(accepted, separators=(",", ":")),
        pr82_anchor_frame=anchor,
        pr82_anchor_player=anchor_player,
        pr82_pre_verdict=baseline_verdict.value,
        pr82_server=baseline_server.value if baseline_server is not None else None,
        pr88_category=category.value,
        pr88_selected_frame=selected_frame,
        pr88_selected_rank=selected_rank,
        pr88_selected_player=selected_player.value if selected_player is not None else None,
        pr88_pre_verdict=selected_verdict.value if selected_verdict is not None else None,
        pr88_credible_outgoing=json.dumps(outgoing_values, separators=(",", ":")),
        pr88_branch=branch,
        pr88_server=preferred_server.value if preferred_server is not None else None,
    )


def read_annotation(data_root: Path, spec: VideoSpec) -> tuple[
    tuple[tuple[int, int], ...], dict[int, tuple[int, ...]]
]:
    """Read only label-blind spans and accepted contacts."""
    path = data_root / "stages" / "annotation" / spec.video_id / "annotator_result.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema") != "annotator-result/0.1" or payload.get("video_id") != spec.video_id:
        raise ValueError(f"unexpected annotation identity in {path}")
    result = payload["result"]
    spans = tuple((int(start), int(end)) for start, end in result["spans"])
    accepted = {
        int(span_id): tuple(int(frame) for frame in frames)
        for span_id, frames in result["filtered_by_rally"].items()
    }
    return spans, accepted


def load_prediction_inputs(data_root: Path, spec: VideoSpec) -> PredictionInputs:
    """Rebuild label-blind sticky and recurrence evidence from saved stages."""
    track = load_npy_xz(
        data_root / "stages" / "shuttle" / spec.video_id / "shuttle_track.npy.xz"
    )
    if track.shape != (spec.frame_count, 3):
        raise ValueError(f"{spec.video_id}: shuttle shape {track.shape} is invalid")
    pose = load_pose_arrays(data_root, spec.video_id)
    court = load_court_vision(
        data_root / "stages" / "court" / spec.video_id,
        video_id=spec.video_id,
        frame_count=spec.frame_count,
        resolution=(float(spec.width), float(spec.height)),
    )
    court_inputs = court.evidence.inputs
    if court_inputs is None:
        raise ValueError(f"{spec.video_id}: court inputs unavailable")
    homography_rows = court_inputs.homography_rows.to_dict("records")
    segments = tracker_segments(homography_rows, court.evidence.court_present, spec.frame_count)
    sticky = build_sticky_result(
        track,
        segments,
        pose.bboxes,
        pose.scores,
        pose.kps,
        pose.ndet,
        spec.video_id,
        court_inputs.gate_court_info,
        court_inputs.gate_resolution_table,
        (float(spec.width), float(spec.height)),
        scale_for_fps(spec.fps).body_unit_half_window,
    )
    production_codes, _guard_info = grade_track(track)
    reconstructed = rebuild_guard_codes(track, production_codes, PRODUCTION_HALO_SOURCE_FRAMES)
    if not np.array_equal(reconstructed, production_codes):
        raise ValueError(f"{spec.video_id}: production halo reconstruction differs")
    spans, accepted = read_annotation(data_root, spec)
    return PredictionInputs(
        spec=spec,
        track=track,
        pose=pose,
        sticky=sticky,
        segments=tuple(segments),
        production_guard_codes=production_codes,
        h3_guard_codes=rebuild_guard_codes(track, production_codes, H3_HALO_SOURCE_FRAMES),
        court_present=court.evidence.court_present,
        net_band=court_inputs.net_band,
        spans=spans,
        accepted_by_span=accepted,
    )


def _csv_bytes(rows: list[PredictionRow]) -> bytes:
    stream = io.StringIO(newline="")
    fieldnames = list(asdict(rows[0]))
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(asdict(row) for row in rows)
    return stream.getvalue().encode("utf-8")


def write_predictions(output_dir: Path, rows: list[PredictionRow]) -> None:
    """Write deterministic predictions and their freeze checksum."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_bytes = _csv_bytes(rows)
    prediction_path = output_dir / "predictions.csv.gz"
    with prediction_path.open("wb") as raw_handle, gzip.GzipFile(
        filename="", mode="wb", fileobj=raw_handle, mtime=0
    ) as zipped:
        zipped.write(csv_bytes)
    digest = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    manifest = {
        "schema": "serve-rule-holdout-prediction-freeze/1",
        "labels_read": False,
        "population": len(rows),
        "videos": [asdict(spec) for spec in VIDEO_SPECS],
        "prediction_file": prediction_path.name,
        "prediction_sha256": digest,
    }
    (output_dir / "prediction_freeze.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"froze {len(rows)} predictions sha256={digest}")


def run(data_root: Path, output_dir: Path) -> None:
    """Build and freeze all predictions before any label join exists."""
    rows: list[PredictionRow] = []
    for spec in VIDEO_SPECS:
        inputs = load_prediction_inputs(data_root, spec)
        rows.extend(build_prediction_row(inputs, span_id) for span_id in range(len(inputs.spans)))
        print(f"{spec.video_id}: froze {len(inputs.spans)} span predictions", flush=True)
    write_predictions(output_dir, rows)


def main() -> None:
    """Parse paths and freeze label-blind predictions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.data_root, arguments.output_dir)


if __name__ == "__main__":
    main()
