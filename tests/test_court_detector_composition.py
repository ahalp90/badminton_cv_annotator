"""Scene composition in the detector: endpoint searches, the composite's middle-frame checks and fallbacks.

The measured cases reuse test_court_scene_compose's synthetic painted court, seen through
a pinhole camera with DeepLSD-like fragments and person boxes over single doubles lines.
The detector-flow cases replace the search and scoring with stand-ins.
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from court_detector import composition, detect, reuse, stripe_refit
from court_detector.detect import (
    MAX_HORIZON_TILT_DEG,
    CourtDetector,
    CourtFitError,
    CourtResult,
    Switches,
)
from court_detector.feet import FeetWindow
from court_detector.inputs import ViewInputs, same_frame_provenance
from tests.test_court_scene_compose import (
    JITTER_PX,
    LEFT_BOX,
    NATIVE_SIZE,
    RIGHT_BOX,
    camera_homography,
    frame_spec,
    render,
    shifted,
    true_corners,
)

FRAME_INDEX = {"middle": 50, "first": 12, "last": 88}
# Court metres of two standing players, one in each half.
PLAYERS_M = [(1.5, 3.0), (4.5, 10.0)]
# Near the top-left image corner: in the frame, but well beyond the court.
OFF_COURT_FEET_PX = [[[5.0, 5.0], [20.0, 5.0]]] * 5


@pytest.fixture(scope="module")
def detector() -> CourtDetector:
    return CourtDetector(Switches())


def players_feet(shift_px: np.ndarray) -> list[list]:
    """Five samples of both players' feet, in the native px of a frame with this camera shift."""
    feet = composition.carry(np.asarray(PLAYERS_M), shifted(camera_homography(), shift_px)).tolist()
    return [feet] * 5


def searched(detector: CourtDetector, spec: dict, paint: float | None,
             all_feet_px: list[list]) -> composition.SearchedFrame:
    """A frame as the detector prepares it, with the given court and final paint score."""
    role = spec["role"]
    view_id, index = f"v_{role}", FRAME_INDEX[role]
    view = ViewInputs(view_id, spec["image"], index, (0, 100), np.asarray(spec["lines"], dtype=np.float32),
                      np.asarray(spec["boxes"], dtype=float).reshape(-1, 4), same_frame_provenance(view_id, index))
    prepared = detector.prepare(view, all_feet_px)
    return composition.SearchedFrame(role, prepared.native_frame, prepared.context,
                                     np.asarray(spec["corners"], dtype=float), paint)


def compose(detector: CourtDetector, frames: list[composition.SearchedFrame],
            require_people: bool = True) -> tuple[composition.Composite | None, dict[str, Any]]:
    return composition.compose_scene(detector.live, frames, geometry_weight=detector.switches.geometry_weight,
                                     require_people=require_people, max_horizon_tilt_deg=MAX_HORIZON_TILT_DEG)


def described_paint(detector: CourtDetector, frame: composition.SearchedFrame, corners_native: np.ndarray) -> float:
    """The final stripe refit's paint measurement of these corners in this frame."""
    live = detector.live
    with live.prepared_measurements(live.verifier):
        described = stripe_refit.describe(corners_native / frame.native_per_working, frame.context, live.verifier,
                                          live.runtime, live.scoring.view_line_maps(frame.context))
    return described["paint_score"]


def middle_and_first(detector: CourtDetector, middle_paint: float, first_paint: float,
                     middle_feet: list[list]) -> list[composition.SearchedFrame]:
    """Each box hides one doubles line; each court is 4 px off at one corner, and the first frame's is turned."""
    middle_corners = true_corners(np.zeros(2)) + [[4.0, 0.0], [0, 0], [0, 0], [0, 0]]
    first_corners = true_corners(JITTER_PX) + [[0, 0], [0, 0], [-4.0, 0.0], [0, 0]]
    middle = frame_spec("middle", np.zeros(2), [LEFT_BOX], middle_corners)
    first = frame_spec("first", JITTER_PX, [RIGHT_BOX], np.roll(first_corners, composition.HALF_TURN_ROLL, axis=0))
    # Every frame is measured with the middle frame's feet, as the detector's endpoint searches are.
    return [searched(detector, middle, middle_paint, middle_feet), searched(detector, first, first_paint, middle_feet)]


def test_composite_takes_each_marking_from_its_best_frame_and_is_measured_in_the_middle(detector) -> None:
    frames = middle_and_first(detector, .9, .5, players_feet(np.zeros(2)))
    composite, record = compose(detector, frames)
    assert composite is not None, record["fallback_reason"]
    assert (record["reference"], record["used_frames"]) == ("middle", ["middle", "first"])
    assert record["half_turn_rolls"] == {"middle": 0, "first": composition.HALF_TURN_ROLL}
    donors = {marking["marking"]: marking["donor_role"] for marking in record["markings"]}
    assert (donors["left_doubles"], donors["right_doubles"]) == ("first", "middle")
    # Up to half a stripe of error, as in the experiment's own composite check.
    np.testing.assert_allclose(composite.corners_native_px, true_corners(np.zeros(2)), atol=2.0)
    # The stored paint is the middle frame's final-refit measurement, not a ranking score.
    assert composite.paint_score == described_paint(detector, frames[0], composite.corners_native_px)
    assert record["middle"]["measurement"]["historical"]["historical_fullcourt"]


def test_endpoint_reference_is_carried_into_the_middle_frame_in_its_own_corner_order(detector) -> None:
    first = frame_spec("first", JITTER_PX, [RIGHT_BOX], true_corners(JITTER_PX))
    # The middle frame's own search labelled the court from the other end.
    middle = frame_spec("middle", np.zeros(2), [LEFT_BOX], np.roll(true_corners(np.zeros(2)), 2, axis=0))
    feet = players_feet(np.zeros(2))
    frames = [searched(detector, middle, .5, feet), searched(detector, first, .9, feet)]
    composite, record = compose(detector, frames)
    assert composite is not None, record["fallback_reason"]
    assert record["reference"] == "first"
    assert record["half_turn_rolls"] == {"first": 0, "middle": composition.HALF_TURN_ROLL}
    np.testing.assert_allclose(record["fit"]["corners_reference_native_px"], true_corners(JITTER_PX), atol=2.0)
    np.testing.assert_allclose(composite.corners_native_px, np.roll(true_corners(np.zeros(2)), 2, axis=0), atol=2.0)
    assert composite.paint_score == described_paint(detector, frames[0], composite.corners_native_px)


@pytest.mark.parametrize("require_people", [True, False])
def test_required_feet_off_the_composite_keep_the_middle_court(detector, require_people: bool) -> None:
    composite, record = compose(detector, middle_and_first(detector, .9, .5, OFF_COURT_FEET_PX), require_people)
    if require_people:
        assert composite is None and record["fallback_reason"] == "middle_players_not_on_court"
    else:
        assert composite is not None and record["fallback_reason"] is None


def unrelated(role: str, shift_px: np.ndarray) -> dict:
    """The court's own fragments over an image that shares no texture with the others."""
    blank = render(camera_homography(), np.zeros(2), [(0, 0, *NATIVE_SIZE)], seed=11)
    return frame_spec(role, shift_px, [], true_corners(shift_px), image=blank, hidden_by=[])


@pytest.mark.parametrize("case", ["one_frame", "missing_paint", "unaligned_endpoint", "unaligned_middle", "failed_fit"])
def test_insufficient_or_unaligned_frames_and_failed_fits_keep_the_middle_court(detector, monkeypatch,
                                                                                 case: str) -> None:
    feet = players_feet(np.zeros(2))
    frames = middle_and_first(detector, .9, .5, feet)
    expected = {"one_frame": "too_few_accepted_frames", "missing_paint": "score_evidence_missing",
                "unaligned_endpoint": "too_few_aligned_frames", "unaligned_middle": "middle_not_aligned",
                "failed_fit": "fit_no_fit_corners"}[case]
    if case == "one_frame":
        frames = frames[:1]
    elif case == "missing_paint":
        frames[1] = composition.SearchedFrame("first", frames[1].native_frame, frames[1].context,
                                              frames[1].corners_native, None)
    elif case == "unaligned_endpoint":
        frames[1] = searched(detector, unrelated("first", JITTER_PX), .5, feet)
    elif case == "unaligned_middle":
        # The endpoints align with each other, so only the middle frame's warp is missing.
        frames = [searched(detector, unrelated("middle", np.zeros(2)), .5, feet),
                  searched(detector, frame_spec("first", JITTER_PX, [RIGHT_BOX], true_corners(JITTER_PX)), .9, feet),
                  searched(detector, frame_spec("last", JITTER_PX, [LEFT_BOX], true_corners(JITTER_PX)), .8, feet)]
    else:
        monkeypatch.setattr(composition.stripe_fitting, "refine",
                            lambda *args, **kwargs: {"status": "solver_failed", "corners_px": None})
    composite, record = compose(detector, frames)
    assert composite is None
    assert record["fallback_reason"] == expected


# The detector flow, with stand-ins for the context, search and scoring.

MIDDLE_CORNERS = np.array([[10., 10.], [50., 10.], [50., 40.], [10., 40.]])
COMPOSITE_CORNERS = MIDDLE_CORNERS + 1.0
FEET_ROWS = [[[4., 5.], [7., 8.]]]


def view(view_id: str, frame_index: int) -> ViewInputs:
    return ViewInputs(view_id, np.zeros((10, 20, 3), dtype=np.uint8), frame_index, (0, 100), np.empty((0, 4)),
                      np.empty((0, 4)), same_frame_provenance(view_id, frame_index))


def stub_detector(monkeypatch: pytest.MonkeyPatch, switches: Switches,
                  outcomes: dict[str, CourtResult | Exception]) -> tuple[CourtDetector, dict[str, list]]:
    """A detector that scores each view as outcomes says, and records its contexts and searches."""
    calls: dict[str, list] = {"prepared": [], "searched": []}

    def view_context(view_id, source, provenance, frame, label):
        calls["prepared"].append((view_id, source["all_feet_px"]))
        return SimpleNamespace(view_id=view_id, families=[object()])

    def search(context, source, frame, laps, artefacts):
        calls["searched"].append(context.view_id)
        return {}

    def score(view, context, populations, templates, frame, laps, artefacts):
        outcome = outcomes[view.view_id]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(detect.feet, "window_feet", lambda *args: FeetWindow([50], None, [50], FEET_ROWS))
    monkeypatch.setattr(detect.search, "seed_points", lambda family: [])
    detector = object.__new__(CourtDetector)
    detector.switches = switches
    detector.live = SimpleNamespace(
        verifier=SimpleNamespace(view_context=view_context), runtime={}, court_model=None,
        line_template_source=SimpleNamespace(generate=lambda *args, **kwargs: SimpleNamespace(entries=[], metadata={})),
        prepared_measurements=lambda verifier: nullcontext(),
    )
    detector.search = search
    detector.score_and_choose = score
    return detector, calls


def court(view_id: str, corners: np.ndarray = MIDDLE_CORNERS, paint: float = .8) -> CourtResult:
    return CourtResult(view_id, corners, None, f"{view_id}_key", None, paint)


@pytest.mark.parametrize(("route", "switches"), [
    ("searched", Switches(timing=True)),
    ("searched", Switches(timing=True, require_people=False, upright_camera=False)),
    ("reused", Switches(timing=True)),
    ("no_court", Switches(timing=True)),
    ("single_view", Switches(timing=True)),
])
def test_only_a_fresh_middle_court_searches_the_endpoints_with_the_middle_feet(monkeypatch, route: str,
                                                                              switches: Switches) -> None:
    middle = court("middle") if route != "no_court" else CourtResult("middle", None, "no_gated_court", None, None)
    outcomes = {"middle": middle, "first": court("first", paint=.9), "last": court("last", paint=.7)}
    detector, calls = stub_detector(monkeypatch, switches, outcomes)
    reused = reuse.ReusedCourt("earlier", MIDDLE_CORNERS, .6, .9, .01)
    monkeypatch.setattr(reuse, "try_reuse", lambda *args, **kwargs: reuse.ReuseAttempt(
        reused if route == "reused" else None, {"rejection": None}))
    composed = []

    def compose_scene(live, frames, **settings):
        composed.append((frames, settings))
        return composition.Composite(COMPOSITE_CORNERS, .6), {"fallback_reason": None, "reference": "first",
                                                              "used_frames": ["first", "middle"],
                                                              "middle": {"measurement": {"paint_score": .6}}}

    monkeypatch.setattr(composition, "compose_scene", compose_scene)
    requested = []

    def endpoint_views():
        requested.append(True)
        return [view("first", 12), view("last", 88)]

    result = detector.detect(view("middle", 50), object(), None, known_courts=[object()],
                             endpoint_views=None if route == "single_view" else endpoint_views)

    if route != "searched":
        assert requested == [] and composed == [] and result.composition is None
        assert calls["searched"] == ([] if route == "reused" else ["middle"])
        assert result.reused_from == ("earlier" if route == "reused" else None)
        # The video pool gets each court's middle frame, and nothing to donate without a composite.
        if route == "no_court":
            assert result.scene is None
        else:
            assert result.scene is not None and result.scene.used_frames == ()
            np.testing.assert_array_equal(result.scene.middle_corners_native_px, MIDDLE_CORNERS)
        return
    assert requested == [True]
    # Each frame gets its own context, all with the middle frame's feet.
    assert calls["prepared"] == [("middle", FEET_ROWS), ("first", FEET_ROWS), ("last", FEET_ROWS)]
    assert calls["searched"] == ["middle", "first", "last"]
    frames, settings = composed[0]
    assert [(frame.role, frame.paint_score) for frame in frames] == [("middle", .8), ("first", .9), ("last", .7)]
    assert settings == {"geometry_weight": switches.geometry_weight, "require_people": switches.require_people,
                        "max_horizon_tilt_deg": MAX_HORIZON_TILT_DEG if switches.upright_camera else None}
    assert (result.view_id, result.chosen_key, result.paint_score, result.reused_from) == ("middle", "composite", .6,
                                                                                            None)
    np.testing.assert_array_equal(result.corners_native_px, COMPOSITE_CORNERS)
    assert result.composition == {"court": "composite", "fallback_reason": None, "reference": "first",
                                  "used_frames": ["first", "middle"], "endpoints": {"first": "court", "last": "court"},
                                  "errors": {}, "middle_chosen_key": "middle_key"}
    scene = result.scene
    assert scene is not None and scene.composite_measurement == {"paint_score": .6}
    np.testing.assert_array_equal(scene.corners_native_px, COMPOSITE_CORNERS)
    np.testing.assert_array_equal(scene.middle_corners_native_px, MIDDLE_CORNERS)
    assert result.stage_seconds is not None
    assert list(result.stage_seconds)[-4:] == ["endpoint_inputs", "first_frame_search", "last_frame_search",
                                               "composition"]


@pytest.mark.parametrize("failure", ["endpoint_search", "composition"])
def test_failed_endpoint_or_composition_keeps_the_middle_court(monkeypatch, failure: str) -> None:
    outcomes: dict[str, CourtResult | Exception] = {"middle": court("middle"), "first": court("first"),
                                                    "last": court("last")}
    if failure == "endpoint_search":
        outcomes["first"] = ValueError("rank deficient")
    detector, _ = stub_detector(monkeypatch, Switches(), outcomes)
    composed = []

    def compose_scene(live, frames, **settings):
        composed.append([frame.role for frame in frames])
        if failure == "composition":
            raise ArithmeticError("singular warp")
        return None, {"fallback_reason": "too_few_aligned_frames", "used_frames": ["middle"]}

    monkeypatch.setattr(composition, "compose_scene", compose_scene)
    result = detector.detect(view("middle", 50), object(), None,
                             endpoint_views=lambda: [view("first", 12), view("last", 88)])
    np.testing.assert_array_equal(result.corners_native_px, MIDDLE_CORNERS)
    assert (result.chosen_key, result.paint_score) == ("middle_key", .8)
    summary = result.composition
    assert summary is not None and summary["court"] == "middle"
    if failure == "endpoint_search":
        assert composed == [["middle", "last"]]
        assert summary["endpoints"] == {"first": "detection_failed", "last": "court"}
        assert summary["errors"] == {"first": "ValueError('rank deficient')"}
        assert summary["fallback_reason"] == "too_few_aligned_frames"
    else:
        assert summary["fallback_reason"] == "composition_failed"
        assert summary["errors"] == {"composition": "ArithmeticError('singular warp')"}


def test_endpoint_input_errors_stop_the_scene_rather_than_fall_back(monkeypatch) -> None:
    detector, _ = stub_detector(monkeypatch, Switches(), {"middle": court("middle")})

    def endpoint_views():
        raise ValueError("people source did not return the requested frame 12")

    with pytest.raises(ValueError, match="requested frame 12") as caught:
        detector.detect(view("middle", 50), object(), None, endpoint_views=endpoint_views)
    assert not isinstance(caught.value, CourtFitError)
