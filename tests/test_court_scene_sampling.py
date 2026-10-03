"""The three-frame court sampling experiment: shared inputs, stop rule, reuse of prepared work, selection and timing."""

from __future__ import annotations

import dataclasses
import gzip
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from court_detector import feet, reuse, stripe_refit
from court_detector.detect import (
    CourtDetector,
    CourtResult,
    Switches,
    freeze_arrays,
    load_live_modules,
)
from court_detector.inputs import PersonSample
from court_detector.reuse import KnownCourt
from experiments.annotator.court_scene_sampling import run, sampling
from experiments.annotator.court_scene_sampling.render import (
    draw_outline,
    overlay_items,
)
from experiments.annotator.court_scene_sampling.sampling import (
    CHEAP_FIRST_SCORE_LIMIT,
    FrameAttempt,
    ManifestScene,
    ManifestVideo,
    MethodRun,
    Route,
)

WIDTH, HEIGHT = 64, 48
FPS = 25.0
SCENE = ManifestScene("clip_s0", 0, 100, None)  # midpoint 50; its window runs from 12 to 88
# Court metres to working pixels for a court well inside the 64x48 test image.
VISIBLE_COURT = np.array([[8., 0., 5.], [0., 3., 3.], [0., 0., 1.]], dtype=np.float32)


class Frames:
    """Each frame is filled with its own index, so any image shows which frame it came from."""

    fps = FPS
    size = (WIDTH, HEIGHT)

    def read(self, indices):
        return [np.full((HEIGHT, WIDTH, 3), index, dtype=np.uint8) for index in indices]


class People:
    def __init__(self, count: int = 2) -> None:
        self.count = count

    def samples(self, indices):
        boxes = np.array([[10., 5., 20., 40.], [40., 5., 50., 40.]])[:self.count]
        return [PersonSample(index, boxes, np.zeros((self.count, 17, 2))) for index in indices]


class Lines:
    def __init__(self) -> None:
        self.frames: list[int] = []

    def segments(self, frame, frame_index):
        assert np.all(frame == frame_index)
        self.frames.append(frame_index)
        return np.array([[1., 2., 30., 40.]], dtype=np.float32)


def role_of(view_id: str) -> str:
    return view_id.split("_frame_")[0].rsplit("_", 1)[1]


def corners_for(frame_index: int) -> np.ndarray:
    return np.array([[10., 10.], [50., 10.], [60., 40.], [4., 40.]]) + frame_index / 100


class FakeDetector(CourtDetector):
    """The experiment's real phase order around stub search and scoring. Copies share one call log."""

    def __init__(self, paints: dict[str, float | None], line_distances: dict[str, float] | None = None,
                 switches: Switches | None = None) -> None:
        self.switches = switches or Switches(timing=True)
        self.pool = None
        self.paints = paints  # final paint score by frame role; None means no court
        self.calls: list[tuple] = []
        distances = line_distances or {}

        def view_context(view_id, source, provenance, frame, case_id):
            maps = np.full((2, HEIGHT, WIDTH), distances.get(role_of(view_id), 0.), dtype=np.float32)
            return SimpleNamespace(case_id=case_id, source=source, frame=frame, families=[np.empty((0, 3))],
                                   size=(WIDTH, HEIGHT), native_size=(WIDTH, HEIGHT), maps=maps)

        def counted_view_context(view_id, source, provenance, frame, case_id):
            self.contexts.append(view_id)
            return view_context(view_id, source, provenance, frame, case_id)

        self.contexts: list[str] = []  # the view_id of every context built
        self.live = SimpleNamespace(
            verifier=SimpleNamespace(view_context=counted_view_context, hard_validity=lambda entry: (True, None)),
            runtime={}, court_model=None,
            line_template_source=SimpleNamespace(generate=lambda *args, **kwargs: SimpleNamespace(entries=[], metadata={})),
            prepared_measurements=lambda verifier: nullcontext(),
            scoring=SimpleNamespace(view_line_maps=lambda context: context.maps),
        )

    def search(self, context, source, native_frame, laps, artefacts=None):
        populations = {"all_lines": [{"candidate_id": "0:0", "homography_working": VISIBLE_COURT.tolist()}],
                       "painted_lines": []}
        self.calls.append(("search", role_of(context.case_id), self.switches.full_score_limit, id(populations)))
        laps.lap("all_lines_search")
        return populations

    def score_and_choose(self, view, context, populations, templates, native_frame, laps, artefacts):
        role = role_of(view.view_id)
        self.calls.append(("score", role, self.switches.full_score_limit, id(populations)))
        laps.lap("scoring")
        if self.paints[role] is None:
            return CourtResult(view.view_id, None, "no_gated_court", None, None)
        return CourtResult(view.view_id, corners_for(view.frame_index), None, "key", None, self.paints[role])


def scene_inputs(people: People | None = None, lines: Lines | None = None) -> sampling.SceneInputs:
    window = sampling.scene_window(SCENE, FPS)
    assert window is not None
    return sampling.prepare_scene(SCENE, window, Frames(), people or People(), lines or Lines(), Switches())


def fake_reuse(monkeypatch: pytest.MonkeyPatch, passes) -> list[tuple]:
    """Replace the reuse checks: passes(view_id, known) says which known courts move onto which views.

    A passing court lands on the frame's own corners with paint .45. Returns each call's
    (view_id, known court, alignment image) in order.
    """
    calls = []

    def try_reuse(known, context, frame, live, **kwargs):
        calls.append((context.case_id, known, kwargs["alignment_image"]))
        if not passes(context.case_id, known):
            return reuse.ReuseAttempt(None, {"source_view_id": known.view_id, "rejection": "alignment_mismatch"})
        court = reuse.ReusedCourt(known.view_id, corners_for(int(frame[0, 0, 0])), .45, .9, .01)
        return reuse.ReuseAttempt(court, {"source_view_id": known.view_id, "rejection": None})

    monkeypatch.setattr(reuse, "try_reuse", try_reuse)
    return calls


def accept_reuse(monkeypatch: pytest.MonkeyPatch, accepted: set[str]) -> list[tuple]:
    """Reuse passes on frames whose role is in accepted, whatever the known court."""
    return fake_reuse(monkeypatch, lambda view_id, known: role_of(view_id) in accepted)


def rejections(records: list[dict]) -> list[str | None]:
    return [record["rejection"] for record in records]


def searched_roles(detector: FakeDetector) -> list[str]:
    return [call[1] for call in detector.calls if call[0] == "search"]


def scored_roles(detector: FakeDetector) -> list[str]:
    return [call[1] for call in detector.calls if call[0] == "score"]


def test_endpoints_are_the_scheduled_window_ends_and_share_the_midpoint_feet() -> None:
    lines = Lines()
    inputs = scene_inputs(lines=lines)
    window = inputs.feet.frames
    assert [inputs.views[role].frame_index for role in ("first", "middle", "last")] == [window[0], 50, window[-1]]
    # The grey check drops both endpoints from the feet, and they are still the sampled frames.
    assert window[0] not in inputs.feet.kept_frames and window[-1] not in inputs.feet.kept_frames
    assert lines.frames == [50, window[0], window[-1]]
    detector = FakeDetector({"middle": .5, "first": .4, "last": .3})
    run_record = sampling.full_three(detector, inputs)
    for attempt in run_record.attempts:
        assert attempt.prep.source["all_feet_px"] is inputs.feet.all_feet_px
        assert np.all(attempt.prep.native_frame == attempt.prep.view.frame_index)
        np.testing.assert_array_equal(attempt.result.corners_native_px, corners_for(attempt.prep.view.frame_index))


def test_short_scene_has_no_window() -> None:
    assert sampling.scene_window(ManifestScene("short", 0, 20, None), FPS) is None


@pytest.mark.parametrize("method", sampling.METHODS)
def test_middle_no_court_stops_every_method_after_its_first_full_evaluation(monkeypatch, method) -> None:
    reuse_calls = accept_reuse(monkeypatch, {"first", "last"})
    detector = FakeDetector({"middle": None, "first": .9, "last": .9}, {"middle": 0., "first": 2., "last": 2.})
    method_detector = sampling.cheap_first_detector(detector) if method == "cheap_first" else detector
    run_record = sampling.METHOD_FUNCTIONS[method](method_detector, scene_inputs())
    assert run_record.stopped and sampling.chosen_attempt(run_record) is None
    assert [attempt.prep.role for attempt in run_record.attempts] == ["middle"]
    assert scored_roles(detector) == ["middle"] and reuse_calls == []
    # cheap_first searched all three frames before finishing the one it ranked first.
    assert searched_roles(detector) == (["middle", "first", "last"] if method == "cheap_first" else ["middle"])
    assert run_record.frames_used == (["middle", "first", "last"] if method == "cheap_first" else ["middle"])


def test_cheap_first_stops_when_its_leading_endpoint_finds_no_court(monkeypatch) -> None:
    reuse_calls = accept_reuse(monkeypatch, {"middle", "last"})
    detector = FakeDetector({"middle": .9, "first": None, "last": .9}, {"middle": 2., "first": 0., "last": 2.})
    run_record = sampling.cheap_first(sampling.cheap_first_detector(detector), scene_inputs())
    assert run_record.leading_role == "first" and run_record.stopped
    assert scored_roles(detector) == ["first"] and reuse_calls == []


@pytest.mark.parametrize("method", sampling.METHODS)
def test_too_few_players_stop_every_method_before_any_search(method) -> None:
    detector = FakeDetector({"middle": .9, "first": .9, "last": .9})
    method_detector = sampling.cheap_first_detector(detector) if method == "cheap_first" else detector
    run_record = sampling.METHOD_FUNCTIONS[method](method_detector, scene_inputs(People(count=1)))
    assert [(attempt.route, attempt.result.no_court_reason) for attempt in run_record.attempts] == [
        ("player_check", "no_gated_court")]
    assert detector.calls == [] and run_record.frames_used == ["middle"]


def test_cheap_first_searches_each_frame_once_and_finishes_from_the_saved_search(monkeypatch) -> None:
    reuse_calls = accept_reuse(monkeypatch, {"middle"})
    detector = FakeDetector({"middle": .5, "first": .6, "last": .7}, {"middle": 1., "first": 0., "last": 2.})
    inputs = scene_inputs()
    run_record = sampling.cheap_first(sampling.cheap_first_detector(detector), inputs)
    assert run_record.leading_role == "first"
    assert run_record.cheap_scores is not None
    assert run_record.cheap_scores["first"]["score"] > run_record.cheap_scores["middle"]["score"]
    searches = {call[1]: call for call in detector.calls if call[0] == "search"}
    assert list(searches) == ["middle", "first", "last"]
    assert {call[2] for call in detector.calls} == {CHEAP_FIRST_SCORE_LIMIT}
    # The lead finishes, the middle passes reuse, and the last fails reuse and finishes its saved search.
    assert scored_roles(detector) == ["first", "last"]
    for call in detector.calls:
        if call[0] == "score":
            assert call[3] == searches[call[1]][3]
    assert [(attempt.prep.role, attempt.route) for attempt in run_record.attempts] == [
        ("first", Route.PREPARED_FINISH), ("middle", Route.SEED_REUSE), ("last", Route.PREPARED_FINISH)]
    assert [rejections(attempt.reuse_records) for attempt in run_record.attempts] == [
        [], [None], ["alignment_mismatch"]]
    # One context per frame: the early phase's work carries through to the finish.
    assert sorted(role_of(view_id) for view_id in detector.contexts) == ["first", "last", "middle"]
    # The seed is the lead's court with the lead frame's own image, and each frame aligns its own image.
    seed = reuse_calls[0][1]
    np.testing.assert_array_equal(seed.corners_native_px, corners_for(inputs.views["first"].frame_index))
    np.testing.assert_array_equal(seed.image, reuse.view_image(inputs.views["first"].frame))
    for view_id, _, alignment_image in reuse_calls:
        np.testing.assert_array_equal(alignment_image, reuse.view_image(inputs.views[role_of(view_id)].frame))
    assert sampling.chosen_attempt(run_record) is run_record.attempts[2]
    assert "seed_reuse" in run_record.attempts[1].stage_seconds
    with pytest.raises(RuntimeError, match="already searched"):
        sampling.search_frame(detector, run_record.attempts[2].prep, sampling.Laps())


def test_seed_refit_searches_only_the_endpoint_whose_reuse_failed(monkeypatch) -> None:
    reuse_calls = accept_reuse(monkeypatch, {"first"})
    detector = FakeDetector({"middle": .5, "first": .9, "last": .6})
    inputs = scene_inputs()
    run_record = sampling.seed_refit(detector, inputs)
    assert searched_roles(detector) == ["middle", "last"]
    assert [(attempt.prep.role, attempt.route) for attempt in run_record.attempts] == [
        ("middle", Route.FULL_SEARCH), ("first", Route.SEED_REUSE), ("last", Route.FULL_SEARCH)]
    assert rejections(run_record.attempts[2].reuse_records) == ["alignment_mismatch"]
    for view_id, _, alignment_image in reuse_calls:
        np.testing.assert_array_equal(alignment_image, reuse.view_image(inputs.views[role_of(view_id)].frame))
    # The reused first-frame court scored .45, below the searched last frame's .6.
    assert sampling.chosen_attempt(run_record) is run_record.attempts[2]


def attempt(role: str, paint: float | None, court: bool = True) -> FrameAttempt:
    prep = SimpleNamespace(role=role)
    result = CourtResult(role, np.zeros((4, 2)) if court else None, None, None, None, paint)
    return FrameAttempt(prep, Route.FULL_SEARCH, result, {}, [])  # type: ignore[arg-type]


@pytest.mark.parametrize(("attempts", "expected"), [
    ([attempt("middle", .5), attempt("first", .7), attempt("last", .7)], 1),
    ([attempt("first", .7), attempt("last", .7), attempt("middle", .7)], 2),
    ([attempt("middle", None), attempt("last", .1)], 1),
    ([attempt("middle", .9, court=False), attempt("first", .2)], 1),
])
def test_selection_takes_the_highest_final_paint_then_middle_first_last(attempts, expected) -> None:
    assert sampling.chosen_attempt(MethodRun("test", attempts, [], False)) is attempts[expected]


def test_timing_charges_shared_inputs_only_for_the_frames_used() -> None:
    seconds = {"decode_window": 1., "feet": 2., "alignment_median": .125,
               "lines": {"middle": 3., "first": 5., "last": 7.}, "boxes": {"middle": .25, "first": .5, "last": .75}}
    middle_only = sampling.timing(10., seconds, ["middle"])
    assert middle_only["shared_inputs_charged"]["lines"] == {"middle": 3.}
    assert middle_only["detector_walltime_measured"] == 10.
    assert middle_only["assembled_total_not_walltime"] == 10. + 1. + 2. + .125 + 3. + .25
    assert sampling.timing(10., seconds, list(sampling.FRAME_ROLES))["assembled_total_not_walltime"] == 29.625


@pytest.mark.parametrize("count", [1, 2])
def test_shared_feet_detection_matches_ordinary_detect_on_real_minimal_inputs(count) -> None:
    detector = CourtDetector(Switches(timing=True))
    frames, people = Frames(), People(count)
    inputs = scene_inputs(people)
    view = inputs.views["middle"]
    ordinary = detector.detect(view, people, frames)
    shared = sampling.detect_fixed_feet(detector, "middle", view, feet.window_feet(view, people, frames, True))
    assert sampling.result_fields(shared.result) == sampling.result_fields(ordinary)
    assert ordinary.stage_seconds is not None
    assert list(shared.stage_seconds) == [name for name in ordinary.stage_seconds if name != "feet"]
    assert shared.route == ("player_check" if count == 1 else "full_search")


@pytest.mark.parametrize("passes", [True, False])
def test_middle_detection_matches_run_videos_detect_with_known_courts(monkeypatch, passes) -> None:
    calls = fake_reuse(monkeypatch, lambda view_id, known: passes)
    detector = CourtDetector(Switches(timing=True))
    frames, people = Frames(), People()
    inputs = scene_inputs(people)
    known = [reuse.make_known_court("earlier", inputs.views["middle"].frame, corners_for(1), .5)]
    view = dataclasses.replace(inputs.views["middle"], alignment_image=inputs.alignment_median)
    ordinary = detector.detect(view, people, frames, known_courts=known)
    shared = sampling.middle_detection(detector, inputs, known)
    assert sampling.result_fields(shared.result) == sampling.result_fields(ordinary)
    assert shared.route == (Route.HISTORY_REUSE if passes else Route.FULL_SEARCH)
    assert [call[2] is inputs.alignment_median for call in calls] == [True, True]
    assert ordinary.stage_seconds is not None
    assert list(shared.stage_seconds) == [name for name in ordinary.stage_seconds if name != "feet"]


def textured_frame(shift: tuple[float, float] = (0., 0.)) -> np.ndarray:
    rng = np.random.default_rng(7)
    grey = cv2.GaussianBlur(rng.integers(0, 255, (1080, 1920), dtype=np.uint8), (0, 0), 6)
    grey = cv2.normalize(grey, None, 30, 200, cv2.NORM_MINMAX)
    moved = cv2.warpAffine(grey, np.float32([[1, 0, shift[0]], [0, 1, shift[1]]]), (1920, 1080))
    return cv2.cvtColor(moved, cv2.COLOR_GRAY2BGR)


def test_endpoint_court_is_carried_into_the_middle_frame_or_marked_incomparable() -> None:
    middle_corners = np.array([[640., 300.], [1280., 300.], [1560., 960.], [360., 960.]])
    shift = (8., -4.)
    endpoint = textured_frame(shift)
    placed = sampling.in_middle_frame(middle_corners + shift, endpoint, textured_frame(), "first")
    np.testing.assert_allclose(placed["corners_native_px"], middle_corners, atol=0.3)
    assert placed["comparable"] and not placed["alignment"]["same_camera"]
    blank = sampling.in_middle_frame(middle_corners, np.zeros_like(endpoint), textured_frame(), "last")
    assert blank == {"corners_native_px": None, "alignment": None, "comparable": False}
    turned = np.roll(middle_corners + [3., 4.], 2, axis=0)
    assert sampling.disagreement(middle_corners, turned)["max_corner_px"] == pytest.approx(5.)


def test_paint_evidence_reproduces_the_final_refit_paint_score() -> None:
    from experiments.court_detector.saved_views import frozen_cases

    root = frozen_cases.ROOT
    case_id = "shuttleset_03_scene_0017"
    saved = Path(__file__).resolve().parents[1] / f"tests/fixtures/court_detector/saved_courts/{case_id}.json"
    corners = np.asarray(json.loads(saved.read_text())["corners_native_px"])
    live = load_live_modules()
    context = frozen_cases.prepare_view(root, case_id)
    freeze_arrays(context)
    scale = np.asarray(context.native_size) / np.asarray(context.size)
    with live.prepared_measurements(live.verifier):
        described = stripe_refit.describe(corners / scale, context, live.verifier, live.runtime,
                                          live.scoring.view_line_maps(context))
    evidence = sampling.paint_evidence(live, context, corners)
    assert evidence["q_paint10_span_weighted"] == described["paint_score"]
    assert len(evidence["markings"]) == 11
    assert {"visible_samples", "known_photometry_samples"} <= evidence["markings"][0].keys()


def test_scene_run_records_baseline_agreement_timing_and_established_courts(monkeypatch) -> None:
    accept_reuse(monkeypatch, {"first", "last", "middle"})
    monkeypatch.setattr(sampling, "paint_evidence", lambda live, context, corners: {})
    detector = FakeDetector({"middle": .5, "first": .6, "last": .4}, {"middle": 0., "first": 1., "last": 2.})
    inputs = scene_inputs()
    histories = {arm: [] for arm in sampling.ARMS}
    row = run.scene_row(inputs, Frames(), People(), detector, sampling.cheap_first_detector(detector), histories,
                        use_history=False)
    assert row["baseline"]["status"] == "court"
    assert row["baseline_agreement"] == {"full_three": True, "seed_refit": True}
    full = row["methods"]["full_three"]
    assert full["frames"][full["chosen_position"]]["role"] == "first"
    assert full["established"]["view_id"] == inputs.views["first"].view_id
    assert full["established"]["reference_image"] == "own_frame"
    np.testing.assert_array_equal(histories["full_three"][0].court.image, reuse.view_image(inputs.views["first"].frame))
    assert set(full["seconds"]["shared_inputs_charged"]["lines"]) == {"middle", "first", "last"}
    # Endpoint courts in these flat test frames cannot be aligned, so they are marked, not moved.
    assert full["frames"][1]["in_middle_frame"]["comparable"] is False
    # seed_refit's endpoints reuse the middle court at .45, so the searched middle is chosen and kept
    # with the median image, as run_video keeps a middle court.
    for arm in ("baseline", "seed_refit"):
        assert [stored.court.view_id for stored in histories[arm]] == [inputs.views["middle"].view_id]
        assert histories[arm][0].court.image is inputs.alignment_median
    baseline = row["baseline"]
    assert baseline["seconds"]["detector_walltime_measured"] == pytest.approx(
        baseline["seconds"]["detect_walltime_measured_with_repeat_feet"] - baseline["stage_seconds"]["feet"])
    json.dumps(run.jsonable(row), allow_nan=False)


def test_a_seed_refit_is_chosen_but_the_searched_court_joins_the_history(monkeypatch) -> None:
    accept_reuse(monkeypatch, {"first"})
    detector = FakeDetector({"middle": .4, "first": .9, "last": .3})
    inputs = scene_inputs()
    run_record = sampling.seed_refit(detector, inputs)
    assert sampling.chosen_attempt(run_record) is run_record.attempts[1]  # the .45 refit beats the .4 search
    history: list[sampling.StoredCourt] = []
    established = sampling.establish(history, run_record, inputs)
    assert established is not None and not established["is_chosen"]
    assert established["route"] == Route.FULL_SEARCH and established["role"] == "middle"
    assert [stored.court.view_id for stored in history] == [inputs.views["middle"].view_id]


def test_returning_view_tries_each_arms_history_on_the_median_then_runs_the_method(monkeypatch) -> None:
    calls = fake_reuse(monkeypatch, lambda view_id, known: known.view_id != "earlier" and role_of(view_id) == "first")
    monkeypatch.setattr(sampling, "paint_evidence", lambda live, context, corners: {})
    detector = FakeDetector({"middle": .5, "first": .6, "last": .4})
    inputs = scene_inputs()
    earlier = reuse.make_known_court("earlier", inputs.views["middle"].frame, corners_for(1), .5)
    histories = {arm: [sampling.StoredCourt(earlier, None)] for arm in sampling.ARMS}
    row = run.scene_row(inputs, Frames(), People(), detector, sampling.cheap_first_detector(detector), histories,
                        use_history=True)
    # Each arm tries its earlier court once, on the middle, against the median image. Seeds
    # inside a method align to each frame's own image.
    history_calls = [call for call in calls if call[1] is earlier]
    assert [role_of(view_id) for view_id, _, _ in history_calls] == ["middle"] * len(sampling.ARMS)
    for view_id, known, image in calls:
        if known is earlier:
            assert image is inputs.alignment_median
        else:
            np.testing.assert_array_equal(image, reuse.view_image(inputs.views[role_of(view_id)].frame))
    seed_refit_row = row["methods"]["seed_refit"]
    assert seed_refit_row["known_courts_tried"] == ["earlier"]
    middle = seed_refit_row["frames"][0]
    assert middle["route"] == "full_search" and rejections(middle["reuse"]) == ["alignment_mismatch"]
    assert [stored.court.view_id for stored in histories["seed_refit"]] == [inputs.views["middle"].view_id, "earlier"]
    assert "baseline_agreement" not in row


def test_histories_keep_eight_newest_courts_and_try_three_nearest_by_histogram() -> None:
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    history: list[sampling.StoredCourt] = []
    for index in range(10):
        court = reuse.make_known_court(f"court_{index}", frame, corners_for(index), .5)
        sampling.remember(history, court, np.array([index, 0.]))
    assert [stored.court.view_id for stored in history] == [f"court_{index}" for index in range(9, 1, -1)]
    assert [court.view_id for court in sampling.courts_to_try(history, None)] == ["court_9", "court_8", "court_7"]
    nearest = sampling.courts_to_try(history, np.array([3.2, 0.]))
    assert [court.view_id for court in nearest] == ["court_3", "court_4", "court_2"]


def scene_of(view_id: str) -> str:
    return view_id.split("_frame_")[0].rsplit("_", 1)[0]


def test_full_video_arms_keep_their_own_histories_and_reuse_before_any_three_frame_work(monkeypatch) -> None:
    # Scene 2's histogram sits nearest scene 0's. Scene 0's courts pass only on scene 2; within
    # a scene, seeds pass on the middle and first frames.
    def passes(view_id: str, known: KnownCourt) -> bool:
        if scene_of(known.view_id) == scene_of(view_id):
            return role_of(view_id) in {"middle", "first"}
        return scene_of(view_id) == "clip_scene_0002" and scene_of(known.view_id) == "clip_scene_0000"

    fake_reuse(monkeypatch, passes)
    monkeypatch.setattr(sampling, "paint_evidence", lambda live, context, corners: {})
    histograms = [np.array([1., 0.]), np.array([0., 1.]), np.array([.9, .1])]
    scenes = [ManifestScene(f"clip_scene_{index:04d}", 80 * index, 80 * index + 80, None, histogram)
              for index, histogram in enumerate(histograms)]
    scenes.append(ManifestScene("clip_scene_0003", 240, 250, None))
    video = ManifestVideo("clip", Path("clip.mp4"), None, scenes, [], True, 12.5)
    detector = FakeDetector({"middle": .4, "first": .6, "last": .3}, {"middle": 1., "first": 0., "last": 2.})
    video_row = {"scenes": [], "later_scenes": []}
    work_after_scene = []

    def save() -> None:
        work_after_scene.append((len(detector.calls), len(detector.contexts)))

    run.compare_video(video, Frames(), People(), Lines(), detector, video_row, save)
    first_frame, middle_frame = "clip_scene_{:04d}_first_frame_{}", "clip_scene_{:04d}_middle_frame_{}"
    endpoint_donors = [first_frame.format(1, 82), first_frame.format(0, 2)]
    middle_donors = [middle_frame.format(1, 120), middle_frame.format(0, 40)]
    # full_three and cheap_first choose their searched first frames. seed_refit chooses its
    # .45 first-frame refit, so its searched middle joins the history instead.
    assert video_row["histories"] == {"baseline": middle_donors, "full_three": endpoint_donors,
                                      "cheap_first": endpoint_donors, "seed_refit": middle_donors}
    reused = video_row["scenes"][2]
    assert reused["baseline"]["reused_from"] == middle_donors[1]
    for method, donors in (("full_three", endpoint_donors), ("cheap_first", endpoint_donors),
                           ("seed_refit", middle_donors)):
        outcome = reused["methods"][method]
        # The nearer histogram goes first although scene 1's court is newer.
        assert outcome["known_courts_tried"] == donors[::-1]
        assert [(frame["route"], frame["reused_from"]) for frame in outcome["frames"]] == [
            ("history_reuse", donors[1])]
        assert outcome["frames_used"] == ["middle"] and outcome["established"] is None
    # Scene 2 searched nothing, and built one middle context per arm.
    calls_before, contexts_before = work_after_scene[1]
    assert work_after_scene[2][0] == calls_before
    assert [role_of(view_id) for view_id in detector.contexts[contexts_before:]] == ["middle"] * len(sampling.ARMS)
    assert video_row["scenes"][3]["status"] == "scene_too_short_for_feet"


def test_manifest_reads_ranges_and_midpoint_overrides(tmp_path) -> None:
    manifest = {"schema": sampling.MANIFEST_SCHEMA, "videos": [{
        "id": "v1", "video": "v1.mp4", "people": "poses",
        "scenes": [{"id": "s1", "start_frame": 0, "end_frame": 100},
                   {"id": "s2", "start_frame": 100, "end_frame": 201, "midpoint_frame": 149}],
        "later_scenes": [{"id": "s3", "start_frame": 300, "end_frame": 400}]}]}
    path = tmp_path / "manifest.json.gz"
    with gzip.open(path, "wt") as stream:
        json.dump(manifest, stream)
    video, = sampling.read_manifest(path)
    assert [(scene.scheduled_midpoint, scene.midpoint) for scene in video.scenes] == [(50, 50), (150, 149)]
    assert video.people == Path("poses") and [scene.scene_id for scene in video.later_scenes] == ["s3"]
    assert not video.full_video
    for bad_scene in ({"id": "s1", "start_frame": 5, "end_frame": 5},
                      {"id": "s1", "start_frame": 0, "end_frame": 10, "midpoint_frame": 10},
                      {"id": "a/b", "start_frame": 0, "end_frame": 10},
                      {"id": "s1", "start_frame": True, "end_frame": 10}):
        manifest["videos"][0]["scenes"] = [bad_scene]
        plain = tmp_path / "manifest.json"
        plain.write_text(json.dumps(manifest))
        with pytest.raises(ValueError):
            sampling.read_manifest(plain)


def test_manifest_reads_a_saved_cut_pass_as_a_full_video(tmp_path) -> None:
    cut_pass = {"video": "v.mp4", "frame_count": 300, "fps": 25.0, "native_size": [64, 48], "scene_seconds": 12.5,
                "scenes": [{"start_frame": 0, "end_frame": 120, "histogram": [.5, .5]},
                           {"start_frame": 120, "end_frame": 300, "histogram": None}]}
    (tmp_path / "cuts.json").write_text(json.dumps(cut_pass))
    entry = {"id": "v", "video": "v.mp4", "full_video_scenes": str(tmp_path / "cuts.json")}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema": sampling.MANIFEST_SCHEMA, "videos": [entry]}))
    video, = sampling.read_manifest(path)
    assert video.full_video and video.cut_pass_seconds == 12.5 and video.later_scenes == []
    assert [scene.scene_id for scene in video.scenes] == ["v_scene_0000", "v_scene_0001"]
    assert video.scenes[0].histogram is not None and video.scenes[0].histogram.tolist() == [.5, .5]
    assert video.scenes[1].histogram is None
    run.check_scenes(video, 300)
    with pytest.raises(ValueError, match="expected 301"):
        run.check_scenes(video, 301)
    for extra in ({"scenes": []}, {"later_scenes": []}):
        path.write_text(json.dumps({"schema": sampling.MANIFEST_SCHEMA, "videos": [{**entry, **extra}]}))
        with pytest.raises(ValueError, match="either"):
            sampling.read_manifest(path)


def test_overlays_are_one_pixel_red_dashes_labelled_by_scene_method_and_frame() -> None:
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    draw_outline(image, [[5., 5.], [58., 5.], [58., 40.], [5., 40.]])
    drawn = image.any(axis=2)
    assert np.all(image[drawn] == [0, 0, 255])
    assert drawn[5, 5:12].all() and not drawn[5, 12:17].any()  # a 6 px dash, then a gap
    frame = {"role": "first", "frame_index": 12, "route": "full_search", "status": "court",
             "corners_native_px": [[1, 2]] * 4}
    results = {"videos": [{"video": "v.mp4", "later_scenes": [], "scenes": [
        {"scene_id": "s0", "status": "scene_too_short_for_feet"},
        {"scene_id": "s1", "midpoint_frame": 50, "baseline": {"status": "no_court"},
         "methods": {"full_three": {"frames": [{**frame, "role": "middle", "status": "no_court"}, frame],
                                    "chosen_position": 1}}}]}]}
    assert [item[2] for item in overlay_items(results, set())] == ["s1__full_three__full_search_first_12__chosen"]
