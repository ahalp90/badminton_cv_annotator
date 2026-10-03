"""Compare three ways to sample one scene's court from the first, middle and last frames.

This is an experiment, not a detector feature. Every method reads the same inputs:
the midpoint's scheduled 3 s people window (feet.window_frames), that window's
first, middle and last frames with each frame's own lines and person boxes, and
the midpoint's standing feet. README.md states each method's exact behaviour and
the output format.

The frame phases below copy CourtDetector.detect's order without its feet step,
so every frame uses the midpoint's feet rather than a window recentred on itself.
Each phase calls the detector's own search, line templates, scoring, net choice,
stripe refit and reuse checks.

Each method first tries the courts it established in earlier scenes on the middle
frame, as run_video's reuse does. Only a scene that reuses none of them reaches
the method's three-frame work.
"""

from __future__ import annotations

import copy
import dataclasses
import gzip
import json
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from typing import Any, NamedTuple

import cv2
import numpy as np

from annotator import court_views
from court_detector import feet, proposals, reuse, search
from court_detector.composition import ENDPOINT_ROLES, FRAME_ROLES
from court_detector.detect import (
    MAX_HORIZON_TILT_DEG,
    VISIBILITY_FLOOR,
    CourtDetector,
    CourtFitError,
    CourtResult,
    Laps,
    LiveModules,
    Switches,
    freeze_arrays,
    source_record,
)
from court_detector.feet import FeetWindow
from court_detector.inputs import (
    FrameReader,
    PeopleSource,
    ViewInputs,
    same_frame_provenance,
)
from court_detector.line_sources import LineSource
from court_detector.reuse import KnownCourt
from court_detector.scene_sources import SceneInfo
from shared.court import HOMOGRAPHY_RESOLUTION
from shared.court_model import CORNER_COURT_M

MANIFEST_SCHEMA = "court-scene-sampling-manifest/1"
METHODS = ("full_three", "cheap_first", "seed_refit")
# The original middle-frame-only detector, with its own reuse history.
BASELINE = "baseline"
ARMS = (BASELINE, *METHODS)
# Every choice between frames breaks exact ties in FRAME_ROLES order: middle, then first, then last.
# cheap_first's searches fully score only each direction pair's best 2048 courts by 16-sample support.
CHEAP_FIRST_SCORE_LIMIT = 2048
# run_video.scene_courts keeps a video's 8 newest searched courts and tries the first 3 on a new scene.
KNOWN_COURTS_KEPT = 8
KNOWN_COURTS_TRIED = 3
# The per-marking evidence kept for each accepted court; its paint score is built from these.
MARKING_FIELDS = ("marking", "visible_samples", "known_photometry_samples", "projected_visible_span_px",
                  "q_paint10", "exclusive_fragment_count")
# The keys prepare.py wrote to the saved full-video cut pass; only "scenes" is read.
CUT_PASS_KEYS = {"video", "frame_count", "fps", "native_size", "scene_seconds", "scenes"}


class Route(StrEnum):
    """How one frame's evaluation ended."""

    PLAYER_CHECK = "player_check"  # detect()'s early exit: the shared feet cannot pass the player checks
    HISTORY_REUSE = "history_reuse"  # a court this arm established in an earlier scene passed the reuse checks
    FULL_SEARCH = "full_search"  # search, then scoring, in one go
    PREPARED_FINISH = "prepared_finish"  # scoring of a search cheap_first saved in its early phase
    SEED_REUSE = "seed_reuse"  # this scene's first accepted court passed the reuse checks on this frame


# Courts found by searching this frame. Only these may join an arm's history.
SEARCHED_ROUTES = (Route.FULL_SEARCH, Route.PREPARED_FINISH)


@dataclass(frozen=True)
class ManifestScene:
    scene_id: str
    start_frame: int
    end_frame: int  # exclusive
    midpoint_override: int | None  # the reviewed frame, when it differs from the scheduled midpoint
    # (HISTOGRAM_BINS,) luma histogram of the scheduled midpoint from the cut pass. Like
    # run_video, it only orders which earlier courts to try first.
    histogram: np.ndarray | None = field(default=None, compare=False)

    @property
    def scheduled_midpoint(self) -> int:
        return SceneInfo(self.start_frame, self.end_frame).middle_frame

    @property
    def midpoint(self) -> int:
        return self.scheduled_midpoint if self.midpoint_override is None else self.midpoint_override


@dataclass(frozen=True)
class ManifestVideo:
    video_id: str
    video: Path
    people: Path | None  # saved pose directory; None runs RTMLib on the frames the scenes need
    # Reviewed scenes, which every arm evaluates without its history; or, for a full
    # video, every cut-pass scene in order, which every arm starts with its history.
    scenes: list[ManifestScene]
    later_scenes: list[ManifestScene]  # returning views, which every arm starts with its history
    full_video: bool  # scenes come from a saved cut pass and cover the whole video
    cut_pass_seconds: float | None  # the saved cut pass's measured time, spent once for every arm


def read_json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    with path.open() as stream:
        return json.load(stream)


def checked_keys(entry: Any, allowed: set[str], required: set[str], where: str) -> dict:
    if not isinstance(entry, dict) or not required <= entry.keys() <= allowed:
        raise ValueError(f"{where} needs keys {sorted(required)} and allows only {sorted(allowed)}")
    return entry


def plain_name(value: Any, where: str) -> str:
    """IDs name overlay files, so each must be a plain file name."""
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError(f"{where}: ID {value!r} must be a plain file name")
    return value


def frame_number(value: Any, where: str) -> int:
    # bool is an int subclass, so a stray true would otherwise become frame 1.
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{where}: frame {value!r} must be a non-negative integer")
    return value


def read_scene(entry: Any, where: str) -> ManifestScene:
    entry = checked_keys(entry, {"id", "start_frame", "end_frame", "midpoint_frame"},
                         {"id", "start_frame", "end_frame"}, f"{where} scene")
    scene_id = plain_name(entry["id"], where)
    start_frame = frame_number(entry["start_frame"], scene_id)
    end_frame = frame_number(entry["end_frame"], scene_id)
    if not start_frame < end_frame:
        raise ValueError(f"{scene_id}: [{start_frame}, {end_frame}) holds no frame; end_frame is exclusive")
    override = entry.get("midpoint_frame")
    if override is not None:
        override = frame_number(override, scene_id)
        if not start_frame <= override < end_frame:
            raise ValueError(f"{scene_id}: midpoint {override} lies outside [{start_frame}, {end_frame})")
    return ManifestScene(scene_id, start_frame, end_frame, override)


def read_cut_pass(path: Path, video_id: str) -> tuple[list[ManifestScene], float | None]:
    """A saved full-video cut pass: its scenes in order, named like run_video's, and its measured time."""
    record = checked_keys(read_json(path), CUT_PASS_KEYS, {"scenes"}, str(path))
    scenes = []
    for index, entry in enumerate(record["scenes"]):
        entry = checked_keys(entry, {"start_frame", "end_frame", "histogram"}, {"start_frame", "end_frame"},
                             f"{path} scene {index}")
        scene = read_scene({"id": f"{video_id}_scene_{index:04d}", "start_frame": entry["start_frame"],
                            "end_frame": entry["end_frame"]}, video_id)
        if entry.get("histogram") is not None:
            scene = dataclasses.replace(scene, histogram=np.asarray(entry["histogram"], dtype=float))
        scenes.append(scene)
    return scenes, record.get("scene_seconds")


def read_manifest(path: Path) -> list[ManifestVideo]:
    """The comparison's videos and scenes; README.md describes the layout."""
    manifest = checked_keys(read_json(path), {"schema", "videos"}, {"schema", "videos"}, str(path))
    if manifest["schema"] != MANIFEST_SCHEMA:
        raise ValueError(f"{path}: schema must be {MANIFEST_SCHEMA!r}")
    videos = []
    for entry in manifest["videos"]:
        entry = checked_keys(entry, {"id", "video", "people", "scenes", "later_scenes", "full_video_scenes"},
                             {"id", "video"}, f"{path} video")
        video_id = plain_name(entry["id"], "video")
        full_video = "full_video_scenes" in entry
        if full_video == ("scenes" in entry) or (full_video and "later_scenes" in entry):
            raise ValueError(f"{video_id}: give either \"scenes\" with optional \"later_scenes\", "
                             f"or \"full_video_scenes\" alone")
        cut_pass_seconds = None
        if full_video:
            scenes, cut_pass_seconds = read_cut_pass(Path(entry["full_video_scenes"]), video_id)
        else:
            scenes = [read_scene(scene, video_id) for scene in entry["scenes"]]
        if not scenes:
            raise ValueError(f"{video_id}: lists no scenes")
        later_scenes = [read_scene(scene, video_id) for scene in entry.get("later_scenes", [])]
        people = Path(entry["people"]) if "people" in entry else None
        videos.append(ManifestVideo(video_id, Path(entry["video"]), people, scenes, later_scenes, full_video,
                                    cut_pass_seconds))
    if not videos:
        raise ValueError(f"{path} lists no videos")
    video_ids = [video.video_id for video in videos]
    scene_ids = [scene.scene_id for video in videos for scene in video.scenes + video.later_scenes]
    for name, ids in (("video", video_ids), ("scene", scene_ids)):
        if len(set(ids)) != len(ids):
            raise ValueError(f"Manifest {name} IDs must be unique")
    return videos


@dataclass(frozen=True)
class SceneInputs:
    """One scene's inputs, prepared once and shared by every method."""

    scene: ManifestScene
    feet: FeetWindow  # the midpoint's window and standing feet, used for all three frames
    views: dict[str, ViewInputs]  # by FRAME_ROLES name; each frame keeps its own image, lines and boxes
    # run_video's reuse alignment image: the median of the three frames' grey images, which
    # keeps the static court while players move. Earlier scenes' courts align to it.
    alignment_median: np.ndarray
    seconds: dict[str, Any]  # measured once: window decode, feet, median, and each frame's lines and boxes


def scene_window(scene: ManifestScene, fps: float) -> list[int] | None:
    """The midpoint's 31-frame people window, or None when the scene is too short for it."""
    try:
        return feet.window_frames(scene.midpoint, fps, scene.start_frame, scene.end_frame)
    except ValueError:
        return None


def frame_view(scene: ManifestScene, role: str, frame_index: int, frames: FrameReader, people: PeopleSource,
               lines: LineSource, seconds: dict[str, Any]) -> ViewInputs:
    """One frame's image, lines and person boxes, timing the lines and boxes separately."""
    frame, = frames.read([frame_index])
    started = perf_counter()
    segments = lines.segments(frame, frame_index)
    seconds["lines"][role] = perf_counter() - started
    started = perf_counter()
    sample, = people.samples([frame_index])
    seconds["boxes"][role] = perf_counter() - started
    if sample.frame_index != frame_index:
        raise ValueError(f"{scene.scene_id}: people source returned frame {sample.frame_index}, not {frame_index}")
    view_id = f"{scene.scene_id}_{role}_frame_{frame_index}"
    return ViewInputs(view_id, frame, frame_index, (scene.start_frame, scene.end_frame), segments,
                      sample.boxes_px, same_frame_provenance(view_id, frame_index))


def prepare_scene(scene: ManifestScene, window: list[int], frames: FrameReader, people: PeopleSource,
                  lines: LineSource, switches: Switches) -> SceneInputs:
    """Decode the window, then prepare the middle frame, its feet and the two endpoint frames.

    The endpoints are the scheduled window's first and last frames, even when the grey
    check drops them from the feet. The feet come before the endpoints, so an
    endpoint's person boxes cost only what the feet step did not already compute.
    """
    seconds: dict[str, Any] = {"lines": {}, "boxes": {}}
    started = perf_counter()
    frames.read(window)  # in order, once; later reads come from VideoFrames' cache
    seconds["decode_window"] = perf_counter() - started
    middle = frame_view(scene, "middle", scene.midpoint, frames, people, lines, seconds)
    started = perf_counter()
    feet_window = feet.window_feet(middle, people, frames, switches.enforce_scene_consistency)
    seconds["feet"] = perf_counter() - started
    views = {"middle": middle}
    for role, frame_index in zip(ENDPOINT_ROLES, (window[0], window[-1]), strict=True):
        views[role] = frame_view(scene, role, frame_index, frames, people, lines, seconds)
    started = perf_counter()
    images = [reuse.view_image(views[role].frame) for role in ("first", "middle", "last")]
    median = np.median(images, axis=0).astype(np.uint8)
    median.flags.writeable = False
    seconds["alignment_median"] = perf_counter() - started
    return SceneInputs(scene, feet_window, views, median, seconds)


def players_can_pass(switches: Switches, feet_window: FeetWindow) -> bool:
    """The detector's shared-feet count check, before reuse or search."""
    return not (switches.require_people and switches.artefacts_dir is None
                and not feet.can_satisfy_player_requirement(feet_window.all_feet_px))


@dataclass
class FramePrep:
    """One frame's detector inputs, kept so later phases reuse them rather than rebuild them."""

    role: str  # one of FRAME_ROLES
    view: ViewInputs
    source: dict  # detect.source_record with the midpoint's feet
    native_frame: np.ndarray  # read-only view of view.frame
    context: Any  # measurements.ViewContext, frozen
    populations: dict[str, list[dict]] | None = None  # both searches' entries, once searched
    templates: list[dict] | None = None  # the line-template entries, once generated


@dataclass
class FrameAttempt:
    """One completed evaluation of one frame within one method."""

    prep: FramePrep
    route: Route
    result: CourtResult
    stage_seconds: dict[str, float]
    reuse_records: list[dict]  # try_reuse's record for each known court tried on this frame, in order


@dataclass
class MethodRun:
    """One method's evaluations of one scene, in the order they ran."""

    method: str
    attempts: list[FrameAttempt]
    frames_used: list[str]  # frames whose lines and boxes the method needed, in FRAME_ROLES order
    # The method's first evaluation found no court, so it evaluated no other frame. A
    # scene that reused an earlier court also ends after one frame, but with a court.
    stopped: bool
    seed: KnownCourt | None = None  # the accepted court carried to the other frames
    cheap_scores: dict[str, dict] | None = None  # cheap_first's frame ranking, by role
    leading_role: str | None = None  # the frame cheap_first finished first


class StoredCourt(NamedTuple):
    """One court in an arm's history."""

    court: KnownCourt
    histogram: np.ndarray | None  # the donor scene's; orders later attempts only


def frame_context(detector: CourtDetector, role: str, view: ViewInputs, feet_window: FeetWindow) -> FramePrep:
    """detect()'s context step, with the midpoint's feet in place of the frame's own."""
    source = source_record(view, feet_window.all_feet_px)
    native_frame = view.frame.view()
    native_frame.flags.writeable = False
    context = detector.live.verifier.view_context(view.view_id, source, view.provenance, native_frame, view.view_id)
    freeze_arrays(context)
    return FramePrep(role, view, source, native_frame, context)


@contextmanager
def fit_errors(view_id: str) -> Iterator[None]:
    """detect()'s rule: after input validation, a ValueError or ArithmeticError is this frame's fit failure."""
    try:
        yield
    except (ValueError, ArithmeticError) as error:
        raise CourtFitError(f"{view_id}: {error}") from error


def search_frame(detector: CourtDetector, prep: FramePrep, laps: Laps) -> None:
    """detect()'s two line searches and line templates. The entries stay on prep for scoring."""
    if prep.populations is not None:
        raise RuntimeError(f"{prep.view.view_id} was already searched; its saved entries must be reused")
    live, switches = detector.live, detector.switches
    with fit_errors(prep.view.view_id):
        populations = detector.search(prep.context, prep.source, prep.native_frame, laps)
        generated = live.line_template_source.generate(
            prep.context, live.runtime, live.court_model, min_visible_lengthwise=VISIBILITY_FLOOR[0],
            min_visible_cross_court=VISIBILITY_FLOOR[1], seed_points=search.seed_points(prep.context.families[0]),
            device=switches.template_device,
        )
    prep.populations, prep.templates = populations, list(generated.entries)
    laps.lap("line_templates")


def finish_frame(detector: CourtDetector, prep: FramePrep, laps: Laps) -> CourtResult:
    """detect()'s candidate scoring, net choice and stripe refit, from prep's saved search."""
    if prep.populations is None or prep.templates is None:
        raise RuntimeError(f"{prep.view.view_id} must be searched before scoring")
    live = detector.live
    with fit_errors(prep.view.view_id), live.prepared_measurements(live.verifier):
        return detector.score_and_choose(prep.view, prep.context, prep.populations, prep.templates,
                                         prep.native_frame, laps, {})


def reused_result(view_id: str, court: reuse.ReusedCourt) -> CourtResult:
    return CourtResult(view_id, court.corners_native_px, None, "reuse", None, court.paint_score, court.source_view_id)


def first_reused(detector: CourtDetector, known_courts: Sequence[KnownCourt], prep: FramePrep,
                 alignment_image: np.ndarray) -> tuple[CourtResult | None, list[dict]]:
    """detect()'s reuse loop: the first known court that passes, and each tried court's record in order.

    :param alignment_image: the grey image each known court's image aligns to.
    """
    switches = detector.switches
    records = []
    for known in known_courts:
        with fit_errors(prep.view.view_id):
            attempt = reuse.try_reuse(known, prep.context, prep.native_frame, detector.live,
                                      max_horizon_tilt_deg=MAX_HORIZON_TILT_DEG if switches.upright_camera else None,
                                      require_people=switches.require_people, alignment_image=alignment_image)
        records.append(attempt.record)
        if attempt.court is not None:
            return reused_result(prep.view.view_id, attempt.court), records
    return None, records


def detect_fixed_feet(detector: CourtDetector, role: str, view: ViewInputs, feet_window: FeetWindow,
                      known_courts: Sequence[KnownCourt] = (), alignment_image: np.ndarray | None = None,
                      reuse_route: Route = Route.HISTORY_REUSE) -> FrameAttempt:
    """CourtDetector.detect with the scene's shared feet in place of its own feet step.

    :param known_courts: courts to try before searching, as detect()'s known_courts.
    :param alignment_image: what they align to; None uses this frame's own image, as detect() does.
    :param reuse_route: the route a passing known court records.
    """
    laps = Laps()
    prep = frame_context(detector, role, view, feet_window)
    laps.lap("context")
    if not players_can_pass(detector.switches, feet_window):
        result = CourtResult(view.view_id, None, "no_gated_court", None, None)
        return FrameAttempt(prep, Route.PLAYER_CHECK, result, laps.seconds, [])
    image = reuse.view_image(prep.native_frame) if alignment_image is None else alignment_image
    reused, records = first_reused(detector, known_courts, prep, image)
    if known_courts:
        laps.lap("reuse")
    if reused is not None:
        return FrameAttempt(prep, reuse_route, reused, laps.seconds, records)
    search_frame(detector, prep, laps)
    result = finish_frame(detector, prep, laps)
    return FrameAttempt(prep, Route.FULL_SEARCH, result, laps.seconds, records)


def preference_key(score: float | None, role: str) -> tuple[bool, float, int]:
    """Sort key for max(): a higher score wins, a missing score loses, and exact ties go by FRAME_ROLES."""
    return score is not None, -np.inf if score is None else score, -FRAME_ROLES.index(role)


def chosen_attempt(run: MethodRun) -> FrameAttempt | None:
    """The selection rule under test: the accepted court with the highest final paint score."""
    accepted = [attempt for attempt in run.attempts if attempt.result.corners_native_px is not None]
    if not accepted:
        return None
    return max(accepted, key=lambda attempt: preference_key(attempt.result.paint_score, attempt.prep.role))


def seed_court(attempt: FrameAttempt) -> KnownCourt | None:
    """An accepted court as a known court, with its own frame's image as the reference.

    Reuse compares paint support, so run_video keeps only courts with positive paint;
    None marks a court that cannot be carried to another frame.
    """
    result = attempt.result
    if result.corners_native_px is None or result.paint_score is None or not result.paint_score > 0:
        return None
    return reuse.make_known_court(attempt.prep.view.view_id, attempt.prep.native_frame, result.corners_native_px,
                                  result.paint_score)


def middle_detection(detector: CourtDetector, inputs: SceneInputs,
                     known_courts: Sequence[KnownCourt]) -> FrameAttempt:
    """The middle-frame-only baseline: try earlier courts against the median image, then search."""
    return detect_fixed_feet(detector, "middle", inputs.views["middle"], inputs.feet, known_courts,
                             inputs.alignment_median)


def ended_at_middle(method: str, middle: FrameAttempt) -> MethodRun:
    """A method that needs no other frame: the middle reused an earlier court, or found no court."""
    return MethodRun(method, [middle], ["middle"], stopped=middle.result.corners_native_px is None)


def full_three(detector: CourtDetector, inputs: SceneInputs, known_courts: Sequence[KnownCourt] = ()) -> MethodRun:
    """Full middle-frame detection; after a searched court, full independent searches of both endpoints."""
    middle = middle_detection(detector, inputs, known_courts)
    if middle.route != Route.FULL_SEARCH or middle.result.corners_native_px is None:
        return ended_at_middle("full_three", middle)
    endpoints = [detect_fixed_feet(detector, role, inputs.views[role], inputs.feet) for role in ENDPOINT_ROLES]
    return MethodRun("full_three", [middle, *endpoints], list(FRAME_ROLES), stopped=False)


def seed_refit(detector: CourtDetector, inputs: SceneInputs, known_courts: Sequence[KnownCourt] = ()) -> MethodRun:
    """Full middle-frame detection; each endpoint tries the middle court's reuse checks, then a full search."""
    middle = middle_detection(detector, inputs, known_courts)
    if middle.route != Route.FULL_SEARCH or middle.result.corners_native_px is None:
        return ended_at_middle("seed_refit", middle)
    seed = seed_court(middle)
    seeds = () if seed is None else (seed,)
    endpoints = []
    for role in ENDPOINT_ROLES:
        endpoints.append(detect_fixed_feet(detector, role, inputs.views[role], inputs.feet, seeds,
                                           reuse_route=Route.SEED_REUSE))
    return MethodRun("seed_refit", [middle, *endpoints], list(FRAME_ROLES), stopped=False, seed=seed)


def cheap_frame_score(live: LiveModules, prep: FramePrep) -> dict[str, Any]:
    """The best 16-sample line support among the frame's plausible prepared candidates.

    This is the search's cheap score (continuous_support at CHEAP_SAMPLES) over both
    searches' entries and the line templates that pass the detector's hard-validity
    rule, which scoring applies first. It measures against the frame's two wide-family
    distance maps, the maps that scoring and the court checks use. The search's own
    16-sample tier scores against per-direction-pair maps it does not keep.
    """
    if prep.populations is None or prep.templates is None:
        raise RuntimeError(f"{prep.view.view_id} must be searched before its cheap score")
    entries = [*prep.populations["all_lines"], *prep.populations["painted_lines"], *prep.templates]
    plausible = [entry for entry in entries if live.verifier.hard_validity(entry)[0]]
    record: dict[str, Any] = {"score": None, "candidates": len(entries), "plausible": len(plausible),
                              "best_candidate_id": None}
    if not plausible:
        return record
    homographies = np.asarray([entry["homography_working"] for entry in plausible], dtype=np.float32)
    # A writeable copy matches the search's own calls, so Numba reuses their compiled code
    # rather than compiling a read-only variant inside cheap_first's timing.
    maps = np.array(live.scoring.view_line_maps(prep.context))
    scores = proposals.finite_scores(homographies, maps, prep.context.size, proposals.CHEAP_SAMPLES)
    best = int(np.argmax(scores))
    record.update(score=float(scores[best]), best_candidate_id=plausible[best]["candidate_id"])
    return record


def early_phase(detector: CourtDetector, prep: FramePrep, laps: Laps) -> dict[str, Any]:
    """cheap_first's early work on one frame: both searches, line templates and the cheap score."""
    search_frame(detector, prep, laps)
    score = cheap_frame_score(detector.live, prep)
    laps.lap("cheap_frame_score")
    return score


def cheap_first(detector: CourtDetector, inputs: SceneInputs, known_courts: Sequence[KnownCourt] = ()) -> MethodRun:
    """Search all three frames, finish the frame with the best cheap score, then carry its court.

    Each frame's context and search run once and stay on its FramePrep. A frame whose
    reuse check fails is scored from that saved search, never searched again.

    :param detector: A detector whose switches set CHEAP_FIRST_SCORE_LIMIT (cheap_first_detector).
    """
    if detector.switches.full_score_limit != CHEAP_FIRST_SCORE_LIMIT:
        raise ValueError("cheap_first needs cheap_first_detector's search limit")
    laps = Laps()
    middle = frame_context(detector, "middle", inputs.views["middle"], inputs.feet)
    laps.lap("context")
    if not players_can_pass(detector.switches, inputs.feet):
        result = CourtResult(middle.view.view_id, None, "no_gated_court", None, None)
        return ended_at_middle("cheap_first", FrameAttempt(middle, Route.PLAYER_CHECK, result, laps.seconds, []))
    reused, history_records = first_reused(detector, known_courts, middle, inputs.alignment_median)
    if known_courts:
        laps.lap("reuse")
    if reused is not None:
        return ended_at_middle("cheap_first", FrameAttempt(middle, Route.HISTORY_REUSE, reused, laps.seconds,
                                                           history_records))

    preps = {"middle": middle}
    cheap_scores = {"middle": early_phase(detector, middle, laps)}
    early_seconds = {"middle": laps.seconds}
    for role in ENDPOINT_ROLES:
        laps = Laps()
        preps[role] = frame_context(detector, role, inputs.views[role], inputs.feet)
        laps.lap("context")
        cheap_scores[role] = early_phase(detector, preps[role], laps)
        early_seconds[role] = laps.seconds
    leading = max(FRAME_ROLES, key=lambda role: preference_key(cheap_scores[role]["score"], role))
    reuse_records = {"middle": history_records, "first": [], "last": []}

    def finished(role: str, laps: Laps) -> FrameAttempt:
        result = finish_frame(detector, preps[role], laps)
        return FrameAttempt(preps[role], Route.PREPARED_FINISH, result, {**early_seconds[role], **laps.seconds},
                            reuse_records[role])

    lead = finished(leading, Laps())
    run = MethodRun("cheap_first", [lead], list(FRAME_ROLES), stopped=lead.result.corners_native_px is None,
                    cheap_scores=cheap_scores, leading_role=leading)
    if run.stopped:
        return run
    run.seed = seed_court(lead)
    for role in FRAME_ROLES:
        if role == leading:
            continue
        laps = Laps()
        if run.seed is not None:
            own_image = reuse.view_image(preps[role].native_frame)
            reused, records = first_reused(detector, [run.seed], preps[role], own_image)
            # The middle may already hold a "reuse" lap from its history.
            laps.lap("seed_reuse")
            reuse_records[role].extend(records)
            if reused is not None:
                run.attempts.append(FrameAttempt(preps[role], Route.SEED_REUSE, reused,
                                                 {**early_seconds[role], **laps.seconds}, reuse_records[role]))
                continue
        run.attempts.append(finished(role, laps))
    return run


def cheap_first_detector(detector: CourtDetector) -> CourtDetector:
    """A second handle on the loaded detector and its open worker pool, with cheap_first's search limit.

    Only the switches differ. Make it inside the detector's with block: it shares that
    block's pool, which the original detector closes.
    """
    twin = copy.copy(detector)
    twin.switches = dataclasses.replace(detector.switches, full_score_limit=CHEAP_FIRST_SCORE_LIMIT)
    return twin


METHOD_FUNCTIONS: dict[str, Callable[[CourtDetector, SceneInputs, Sequence[KnownCourt]], MethodRun]] = {
    "full_three": full_three, "cheap_first": cheap_first, "seed_refit": seed_refit,
}


def histogram_distance(histogram: np.ndarray, stored: StoredCourt) -> float:
    """run_video's ordering distance; a court without a histogram goes last."""
    return float("inf") if stored.histogram is None else float(np.abs(histogram - stored.histogram).sum())


def courts_to_try(history: list[StoredCourt], histogram: np.ndarray | None) -> list[KnownCourt]:
    """run_video's attempts: nearest histogram first, else newest first; then the first KNOWN_COURTS_TRIED."""
    ordered = history
    if histogram is not None:
        ordered = sorted(history, key=lambda stored: histogram_distance(histogram, stored))
    return [stored.court for stored in ordered[:KNOWN_COURTS_TRIED]]


def remember(history: list[StoredCourt], court: KnownCourt, histogram: np.ndarray | None) -> None:
    """Add a searched court to an arm's history, newest first, keeping run_video's KNOWN_COURTS_KEPT."""
    history.insert(0, StoredCourt(court, histogram))
    del history[KNOWN_COURTS_KEPT:]


def donor_attempt(run: MethodRun) -> FrameAttempt | None:
    """The court a method's scene adds to its history: its best court found by a search.

    Reused courts never donate, as in run_video, because chained refits would
    accumulate movement. This is the chosen court unless a seed refit scored higher.
    """
    searched = [attempt for attempt in run.attempts
                if attempt.route in SEARCHED_ROUTES and attempt.result.corners_native_px is not None]
    if not searched:
        return None
    return max(searched, key=lambda attempt: preference_key(attempt.result.paint_score, attempt.prep.role))


def establish(history: list[StoredCourt], run: MethodRun, inputs: SceneInputs) -> dict[str, Any] | None:
    """Keep a method's donor court for later scenes, with the reference image its corners fit.

    A middle-frame court keeps run_video's median image. An endpoint court keeps its own
    frame's image, because the median can mix in camera movement its corners do not fit.
    """
    donor = donor_attempt(run)
    # Reuse compares paint support, so run_video keeps only courts with positive paint.
    if donor is None or donor.result.paint_score is None or not donor.result.paint_score > 0:
        return None
    middle = donor.prep.role == "middle"
    court = reuse.make_known_court(donor.prep.view.view_id, donor.prep.native_frame, donor.result.corners_native_px,
                                   donor.result.paint_score, alignment_image=inputs.alignment_median if middle else None)
    remember(history, court, inputs.scene.histogram)
    return {"view_id": court.view_id, "role": donor.prep.role, "frame_index": donor.prep.view.frame_index,
            "route": donor.route, "is_chosen": donor is chosen_attempt(run), "paint_score": court.paint_score,
            "reference_image": "window_median" if middle else "own_frame"}


def paint_evidence(live: LiveModules, context: Any, corners_native: np.ndarray) -> dict[str, Any]:
    """The paint and line evidence behind a court's paint score, measured in the court's own frame."""
    scale = np.asarray(context.native_size, dtype=float) / np.asarray(context.size, dtype=float)
    working = np.asarray(corners_native, dtype=float) / scale
    homography = cv2.getPerspectiveTransform(CORNER_COURT_M, working.astype(np.float32)).astype(float)
    with live.prepared_measurements(live.verifier):
        evidence, _ = live.verifier.measure_candidate(context, {"homography_working": homography}, {})
    markings = []
    for marking in evidence["markings"]:
        markings.append({name: marking[name] for name in MARKING_FIELDS})
    return {"q_paint10_span_weighted": evidence["q_paint10_span_weighted"],
            "q_geom_span_weighted": evidence["q_geom_span_weighted"],
            "assigned_fragment_count": evidence["assigned_fragment_count"],
            "photometry_occlusion_aware": evidence["photometry_occlusion_aware"],
            "directional": evidence["directional"], "markings": markings}


def in_middle_frame(corners_native: np.ndarray, frame: np.ndarray, middle_frame: np.ndarray,
                    role: str) -> dict[str, Any]:
    """A court carried into the middle frame by image alignment inside that court.

    The warp is the ECC homography behind reuse's same-camera test. Its correlation and
    largest corner movement record camera motion; no motion limit applies here.
    Comparable means ECC converged and the images correlate at the reuse test's level.
    """
    corners = np.asarray(corners_native, dtype=float)
    if role == "middle":
        return {"corners_native_px": corners.tolist(), "alignment": None, "comparable": True}
    native_size = (frame.shape[1], frame.shape[0])
    corners_refpx = corners * np.asarray(HOMOGRAPHY_RESOLUTION) / np.asarray(native_size)
    alignment = court_views.measure_view_alignment(reuse.view_image(frame), reuse.view_image(middle_frame),
                                                   corners_refpx)
    if alignment is None:
        return {"corners_native_px": None, "alignment": None, "comparable": False}
    correlation = float(alignment.correlation)
    return {"corners_native_px": reuse.moved_corners_native(alignment, native_size).tolist(),
            "alignment": {"correlation": correlation, "max_corner_shift_refpx": float(alignment.shift_refpx),
                          "same_camera": alignment.matches},
            "comparable": correlation >= court_views.MIN_ALIGNMENT_CORRELATION}


def disagreement(reference_native: Sequence, corners_native: Sequence) -> dict[str, float]:
    """How far a court sits from a reference court in the same frame. Not an error against ground truth.

    Reports the largest corner distance in pixels, and reuse.floor_shift_m's largest
    floor-point movement in metres on the compared court's floor. A court turned 180
    degrees is the same court, so the closer corner order counts.
    """
    reference = np.asarray(reference_native, dtype=np.float32)
    corners = np.asarray(corners_native, dtype=np.float32)
    turned = np.roll(corners, 2, axis=0)  # CORNER_COURT_M's TL TR BR BL order, turned 180 degrees
    distances = [float(np.linalg.norm(order - reference, axis=1).max()) for order in (corners, turned)]
    closer = corners if distances[0] <= distances[1] else turned
    before = cv2.getPerspectiveTransform(CORNER_COURT_M, reference).astype(float)
    after = cv2.getPerspectiveTransform(CORNER_COURT_M, closer).astype(float)
    shift_m, _ = reuse.floor_shift_m(before, after)
    return {"max_corner_px": min(distances), "max_floor_shift_m": shift_m}


def result_fields(result: CourtResult) -> dict[str, Any]:
    corners = result.corners_native_px
    return {"status": "no_court" if corners is None else "court",
            "corners_native_px": None if corners is None else np.asarray(corners).tolist(),
            "no_court_reason": result.no_court_reason, "chosen_key": result.chosen_key,
            "paint_score": result.paint_score, "reused_from": result.reused_from}


def attempt_row(live: LiveModules, attempt: FrameAttempt, middle_frame: np.ndarray,
                baseline_corners: Sequence | None) -> dict[str, Any]:
    """One frame's evaluation: its court in its own frame, its evidence, and its place in the middle frame."""
    view = attempt.prep.view
    records = attempt.reuse_records
    row = {"role": attempt.prep.role, "frame_index": view.frame_index, "view_id": view.view_id,
           "route": attempt.route, **result_fields(attempt.result), "stage_seconds": attempt.stage_seconds,
           "reuse_rejection": records[-1]["rejection"] if records else None, "reuse": records}
    corners = attempt.result.corners_native_px
    if corners is None:
        return row
    row["evidence"] = paint_evidence(live, attempt.prep.context, corners)
    placed = in_middle_frame(corners, view.frame, middle_frame, attempt.prep.role)
    row["in_middle_frame"] = placed
    if baseline_corners is not None and placed["comparable"]:
        row["vs_baseline"] = disagreement(baseline_corners, placed["corners_native_px"])
    return row


def charged_inputs(seconds: dict[str, Any], roles: Sequence[str]) -> dict[str, Any]:
    """The shared inputs one arm needed: the window decode, feet and median, and its frames' lines and boxes."""
    return {"decode_window": seconds["decode_window"], "feet": seconds["feet"],
            "alignment_median": seconds["alignment_median"],
            "lines": {role: seconds["lines"][role] for role in roles},
            "boxes": {role: seconds["boxes"][role] for role in roles}}


def timing(detector_seconds: float, seconds: dict[str, Any], roles: Sequence[str]) -> dict[str, Any]:
    """An arm's measured detector wall time, and a total assembled with its share of the shared inputs.

    The assembled total adds separate measurements. It is not an end-to-end wall time.
    """
    charged = charged_inputs(seconds, roles)
    inputs_total = (charged["decode_window"] + charged["feet"] + charged["alignment_median"]
                    + sum(charged["lines"].values()) + sum(charged["boxes"].values()))
    return {"detector_walltime_measured": detector_seconds, "shared_inputs_charged": charged,
            "assembled_total_not_walltime": detector_seconds + inputs_total}


def method_row(live: LiveModules, run: MethodRun, inputs: SceneInputs,
               baseline_corners: Sequence | None) -> dict[str, Any]:
    """One method's outcome on one scene, without its timing."""
    middle_frame = inputs.views["middle"].frame
    frames = [attempt_row(live, attempt, middle_frame, baseline_corners) for attempt in run.attempts]
    chosen = chosen_attempt(run)
    row: dict[str, Any] = {
        "status": "no_court" if chosen is None else "court",
        "stopped_after_first_evaluation": run.stopped, "frames_used": run.frames_used,
        "chosen_position": None if chosen is None else run.attempts.index(chosen),
        "seed_view_id": None if run.seed is None else run.seed.view_id, "frames": frames,
    }
    if run.cheap_scores is not None:
        row.update(cheap_frame_scores=run.cheap_scores, leading_role=run.leading_role)
    return row
