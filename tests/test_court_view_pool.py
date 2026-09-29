"""Video-robust courts: view groups, donors carried between scenes, one pooled fit and where it applies.

The measured cases reuse test_court_scene_compose's synthetic painted court and compose
real scenes from it. The fast-robust cases give the pool fresh middle-frame courts
instead. The output cases replace the fit, member checks and scores with stand-ins.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from court_detector import composition, view_pool
from court_detector.detect import CourtDetector, SceneCourts, Switches
from court_detector.view_pool import CourtMode, Member, VideoPool, ViewGroup
from tests.test_court_detector_composition import (
    OFF_COURT_FEET_PX,
    compose,
    middle_and_first,
    players_feet,
    searched,
    unrelated,
)
from tests.test_court_scene_compose import (
    JITTER_PX,
    LEFT_BOX,
    RIGHT_BOX,
    frame_spec,
    true_corners,
)

# A second scene's camera, moved within reuse's same-camera limit of one reference px:
# one native px down is 0.8 reference px.
NUDGE_PX = np.array([0.0, 1.0])


@pytest.fixture(scope="module")
def detector() -> CourtDetector:
    return CourtDetector(Switches())


def composed(detector: CourtDetector, frames: list[composition.SearchedFrame]) -> SceneCourts:
    """A scene's finished courts, as the detector hands them to the pool. frames[0] is the middle frame."""
    composite, record = compose(detector, frames)
    assert composite is not None, record["fallback_reason"]
    middle = frames[0]
    middle_score = next(row for row in record["scores"] if row["role"] == composition.MIDDLE)
    return SceneCourts(middle.context, middle.native_frame, composite.corners_native_px, middle.corners_native,
                       middle_score, record["middle"]["measurement"], composite.used_frames)


def turned_scene(detector: CourtDetector) -> SceneCourts:
    """A nudged scene whose first frame is its reference and labels the court from the other end."""
    feet = players_feet(NUDGE_PX)
    middle = frame_spec("middle", NUDGE_PX, [LEFT_BOX], true_corners(NUDGE_PX))
    first = frame_spec("first", JITTER_PX + NUDGE_PX, [RIGHT_BOX],
                       np.roll(true_corners(JITTER_PX + NUDGE_PX), composition.HALF_TURN_ROLL, axis=0))
    return composed(detector, [searched(detector, middle, .5, feet), searched(detector, first, .9, feet)])


def middle_only_scene(detector: CourtDetector, spec: dict, all_feet_px: list[list]) -> SceneCourts:
    """A scene whose court came from its middle frame alone: reused, or searched without endpoints.

    Video-robust mode takes no donors from it; fast-robust mode takes its middle frame.
    """
    frame = searched(detector, spec, .8, all_feet_px)
    return SceneCourts(frame.context, frame.native_frame, frame.corners_native, frame.corners_native)


def row(view_id: str, scene: SceneCourts, reused_from: str | None = None) -> dict[str, Any]:
    return {"view_id": view_id, "corners_native_px": np.asarray(scene.corners_native_px).tolist(),
            "chosen_key": "reuse" if reused_from else "composite", "reused_from": reused_from}


@pytest.fixture(scope="module")
def scenes(detector: CourtDetector) -> dict[str, SceneCourts]:
    first_scene = composed(detector, middle_and_first(detector, .9, .5, players_feet(np.zeros(2))))
    unmoved = frame_spec("middle", np.zeros(2), [LEFT_BOX], true_corners(np.zeros(2)))
    return {"first": first_scene, "turned": turned_scene(detector),
            "feet_off_court": middle_only_scene(detector, unmoved, OFF_COURT_FEET_PX)}


def test_turned_donors_from_another_scene_reference_land_on_the_group_reference(detector, scenes) -> None:
    pool = VideoPool(detector.live, detector.switches)
    pool.add(row("first", scenes["first"]), scenes["first"])
    turned = scenes["turned"]
    record, to_group = composition.align(
        composition.SearchedFrame("middle", turned.native_frame, turned.context, turned.corners_native_px, None),
        pool.groups[0].reference,
    )
    assert to_group is not None and record["same_camera"], record
    reference = pool.groups[0].reference
    donors = view_pool.scene_donors(detector.live, turned.used_frames, to_group, reference)
    assert any(donor is not None and donor.frame.role == "first" for donor in donors)
    reference_court = true_corners(np.zeros(2)) / reference.native_per_working
    for donor in donors:
        assert donor is not None
        # Every donor's court, carried through its scene's reference, sits on the group
        # reference's court with the same corner order.
        carried = composition.carry(donor.corners_native / donor.frame.native_per_working, donor.to_reference)
        np.testing.assert_allclose(carried, reference_court, atol=4.0)


def test_one_pooled_court_reaches_each_member_in_its_own_order_and_feet(detector, scenes) -> None:
    pool = VideoPool(detector.live, detector.switches)
    rows = {name: row(name, scene, "first" if name == "feet_off_court" else None) for name, scene in scenes.items()}
    for name, scene in scenes.items():
        pool.add(rows[name], scene)
    (group,) = pool.groups
    # The reused court joins the group but donates nothing.
    assert group.donor_view_ids == ["first", "turned"]
    (summary,) = pool.apply()
    assert summary["reason"] is None and summary["fit"]["valid"], summary
    np.testing.assert_allclose(summary["fit"]["corners_reference_native_px"], true_corners(np.zeros(2)), atol=2.0)
    assert {marking["donor_view_id"] for marking in summary["markings"]} <= {"first", "turned"}
    # Every member whose own checks pass takes the pooled court, in its own pixels and corner order.
    assert summary["pooled_view_ids"] == ["first", "turned"]
    turned = rows["turned"]
    np.testing.assert_allclose(turned["corners_native_px"], true_corners(NUDGE_PX), atol=2.0)
    assert (turned["chosen_key"], turned["view_pool"]["court"]) == (view_pool.POOLED_KEY, "pooled")
    assert turned["scene_corners_native_px"] == np.asarray(scenes["turned"].corners_native_px).tolist()
    means = summary["mean_combined_scores"]
    assert all(means[court] is not None for court in view_pool.COMPARED_COURTS), means
    # The pooled court is scored in every member, but a member whose players' feet
    # are off it keeps its scene court.
    off_court = rows["feet_off_court"]
    assert off_court["view_pool"]["pooled_rejection"] == "players_not_on_court"
    assert off_court["view_pool"]["court"] == "scene" and "scene_corners_native_px" not in off_court
    assert off_court["view_pool"]["scores"]["pooled"]["combined_score"] is not None
    # The runner saves rows and summaries as strict JSON.
    json.dumps([summary, rows], allow_nan=False)


def test_a_single_donor_scene_does_not_pool(detector, scenes) -> None:
    pool = VideoPool(detector.live, detector.switches)
    rows = {name: row(name, scenes[name]) for name in ("first", "feet_off_court")}
    for name, member_row in rows.items():
        pool.add(member_row, scenes[name])
    (summary,) = pool.apply()
    assert (summary["member_view_ids"], summary["donor_view_ids"]) == (["first", "feet_off_court"], ["first"])
    assert (summary["pooled_view_ids"], summary["reason"]) == ([], "too_few_donor_scenes")
    assert all("view_pool" not in member_row for member_row in rows.values())


def test_two_composites_can_pool_when_one_scene_wins_every_marking(detector, scenes) -> None:
    # Two independently searched scenes can return identical observations. Exact
    # ties all favour the first, without making the second a reused court.
    pool = VideoPool(detector.live, detector.switches)
    for view_id in ("first_scene", "second_scene"):
        pool.add(row(view_id, scenes["first"]), scenes["first"])
    (group,) = pool.groups
    assert {donor.view_id for donor in group.donors if donor is not None} == {"first_scene"}
    (summary,) = pool.apply()
    assert summary["pooled_view_ids"] == ["first_scene", "second_scene"]


def test_large_group_warp_has_the_correct_direction_for_donors_and_members(detector, scenes) -> None:
    # Exercise the transform independently of the static-camera grouping gate.
    # A larger displacement makes a reversed warp unambiguous.
    translation = np.array([20., -15.])
    to_group = np.eye(3)
    to_group[:2, 2] = translation
    pool = VideoPool(detector.live, detector.switches)
    pool.add(row("first", scenes["first"]), scenes["first"])
    (group,) = pool.groups
    reference = group.reference
    expected_donor = true_corners(NUDGE_PX) / reference.native_per_working + translation
    donors = view_pool.scene_donors(detector.live, scenes["turned"].used_frames, to_group, reference)
    for donor in donors:
        assert donor is not None
        carried = composition.carry(donor.corners_native / donor.frame.native_per_working, donor.to_reference)
        np.testing.assert_allclose(carried, expected_donor, atol=4.0)

    scene = scenes["first"]
    member = Member(row("member", scene), scene.context, to_group, None, scene.corners_native_px,
                    scene.middle_corners_native_px, scene.middle_score, scene.composite_measurement)
    result = pool.member_outcome(group, member, true_corners(np.zeros(2)))
    expected_member = true_corners(np.zeros(2)) - translation * reference.native_per_working
    np.testing.assert_allclose(result["pooled_corners"], expected_member, atol=1e-6)


@pytest.mark.parametrize(("case", "expected_groups"), [
    ("unrelated_image", [["first"], ["other"]]),
    ("moved_camera", [["first", "other"]]),
])
def test_matching_hashes_need_usable_alignment_not_a_stationary_camera(
    detector, scenes, monkeypatch, case: str, expected_groups: list[list[str]],
) -> None:
    monkeypatch.setattr(view_pool, "image_hash", lambda native_frame: np.zeros((16, 16), dtype=bool))
    if case == "unrelated_image":
        other = middle_only_scene(detector, unrelated("middle", np.zeros(2)), players_feet(np.zeros(2)))
    else:
        moved = frame_spec("middle", JITTER_PX, [RIGHT_BOX], true_corners(JITTER_PX))
        other = middle_only_scene(detector, moved, players_feet(JITTER_PX))
    pool = VideoPool(detector.live, detector.switches)
    pool.add(row("first", scenes["first"]), scenes["first"])
    pool.add(row("other", other), other)
    assert [[member.row["view_id"] for member in group.members] for group in pool.groups] == expected_groups
    if case == "moved_camera":
        alignment = pool.groups[0].members[1].alignment
        assert alignment["usable"] and not alignment["same_camera"]
        assert alignment["max_corner_shift_refpx"] > 1.0


@pytest.fixture(scope="module")
def middle_scenes(detector: CourtDetector) -> dict[str, SceneCourts]:
    """Fresh middle-frame courts of one camera view, as fast-robust mode hands them to the pool.

    A person hides the left doubles line in the first two scenes and the right singles line
    in the third. The first court is 4 px off at one corner; the second is labelled from the
    other end.
    """
    specs = {
        "left_box": (np.zeros(2), LEFT_BOX, true_corners(np.zeros(2)) + [[4.0, 0.0], [0, 0], [0, 0], [0, 0]]),
        "turned": (NUDGE_PX, LEFT_BOX, np.roll(true_corners(NUDGE_PX), composition.HALF_TURN_ROLL, axis=0)),
        "right_box": (np.zeros(2), RIGHT_BOX, true_corners(np.zeros(2))),
        "fourth": (NUDGE_PX, LEFT_BOX, true_corners(NUDGE_PX)),
    }
    scenes = {}
    for name, (shift, box, corners) in specs.items():
        scenes[name] = middle_only_scene(detector, frame_spec("middle", shift, [box], corners), players_feet(shift))
    return scenes


def test_fast_robust_pools_fresh_middle_frames_and_scores_them_from_their_own_measurement(
        detector, middle_scenes, monkeypatch) -> None:
    # A member's own court is scored from the measurement its donation came from.
    monkeypatch.setattr(VideoPool, "measured_score", lambda *args: pytest.fail("measured a scene court again"))
    pool = VideoPool(detector.live, detector.switches, CourtMode.FAST_ROBUST)
    rows = {name: row(name, scene) for name, scene in middle_scenes.items()}
    for name, scene in middle_scenes.items():
        pool.add(rows[name], scene)
    (group,) = pool.groups
    assert group.donor_view_ids == list(middle_scenes)
    assert len(group.members) == 4
    # The boxes hide different lines, so the markings come from more than one scene.
    donors = {marking: donor.view_id for marking, donor in zip(view_pool.MARKINGS, group.donors, strict=True)
              if donor is not None}
    assert donors["left_doubles"] == "right_box" and donors["right_singles"] != "right_box"
    (summary,) = pool.apply()
    assert len(summary["scene_candidates"]) == 4
    assert summary["fit"]["valid"], summary
    np.testing.assert_allclose(summary["fit"]["corners_reference_native_px"], true_corners(np.zeros(2)), atol=2.0)
    means = summary["mean_combined_scores"]
    assert means["pooled"] is not None and means["scene"] == means["middle"]
    assert summary["chosen_court"] in ("pooled", "group_scene")
    for name, member_row in rows.items():
        record = member_row["view_pool"]
        assert record["scores"]["scene"] == record["scores"]["middle"]
        assert record["scores"]["scene"]["combined_score"] is not None, name
    json.dumps([summary, rows], allow_nan=False)


@pytest.mark.parametrize(("court_mode", "donor_view_ids"), [
    (CourtMode.FAST_ROBUST, ["left_box", "turned"]),
    (CourtMode.VIDEO_ROBUST, []),
])
def test_two_middle_frame_courts_do_not_pool_in_either_mode(detector, middle_scenes, court_mode: CourtMode,
                                                            donor_view_ids: list[str]) -> None:
    # Fast-robust counts each fresh middle frame as a donor but needs three scenes;
    # video-robust takes donors only from composites.
    pool = VideoPool(detector.live, detector.switches, court_mode)
    rows = {name: row(name, middle_scenes[name]) for name in ("left_box", "turned")}
    for name, member_row in rows.items():
        pool.add(member_row, middle_scenes[name])
    (summary,) = pool.apply()
    assert (summary["donor_view_ids"], summary["reason"]) == (donor_view_ids, "too_few_donor_scenes")
    assert all("view_pool" not in member_row for member_row in rows.values())


# Far enough off the painted sideline to lose its paint support
MISFIT_PX = 8.0


@pytest.fixture
def misfit_pool(monkeypatch) -> None:
    """The pooled fit's right sideline moved onto blank floor, as when two donors name one stripe differently."""
    real_fit = composition.fit_in_reference

    def fit_in_reference(live, reference, constraints) -> dict:
        fit = real_fit(live, reference, constraints)
        corners = np.asarray(fit["corners_native_px"], dtype=float)
        corners[[1, 2], 0] += MISFIT_PX
        return {**fit, "corners_native_px": corners.tolist()}

    monkeypatch.setattr(composition, "fit_in_reference", fit_in_reference)


def test_a_scene_court_that_beats_a_misfit_pool_replaces_it_in_every_member(detector, scenes, misfit_pool) -> None:
    pool = VideoPool(detector.live, detector.switches)
    rows = {name: row(name, scenes[name]) for name in ("first", "turned")}
    for name, member_row in rows.items():
        pool.add(member_row, scenes[name])
    (summary,) = pool.apply()
    candidates = summary["scene_candidates"]
    assert [candidate["rejection"] for candidate in candidates] == [None, None]
    best = max(candidates, key=lambda candidate: candidate["mean_combined_score"])
    # Both candidates are scored in the same two middle frames as the pool.
    assert best["mean_combined_score"] > summary["mean_combined_scores"]["pooled"]
    assert (summary["chosen_court"], summary["chosen_view_id"], summary["pooled_view_ids"]) == (
        "group_scene", best["view_id"], [])
    # Each member holds the winner in its own pixels and corner order; the turned scene
    # labels the court from the other end in its first frame only.
    for name, shift in (("first", np.zeros(2)), ("turned", NUDGE_PX)):
        member_row = rows[name]
        np.testing.assert_allclose(member_row["corners_native_px"], true_corners(shift), atol=2.0)
        record = member_row["view_pool"]
        assert (record["court"], record["group_scene_view_id"]) == ("group_scene", best["view_id"])
        assert record["group_scene_corners_native_px"] == member_row["corners_native_px"]
        # The pooled court stays on the row for comparison.
        assert record["pooled_corners_native_px"][1][0] > member_row["corners_native_px"][1][0] + MISFIT_PX / 2
        assert record["scores"]["group_scene"]["combined_score"] > record["scores"]["pooled"]["combined_score"]
    carried = rows["first" if best["view_id"] == "turned" else "turned"]
    assert carried["chosen_key"] == view_pool.GROUP_SCENE_KEY
    assert carried["scene_chosen_key"] == composition.COMPOSITE_KEY
    assert rows[best["view_id"]]["chosen_key"] == composition.COMPOSITE_KEY
    json.dumps([summary, rows], allow_nan=False)


def test_a_scene_court_must_pass_every_member_including_reused_ones(detector, scenes, misfit_pool) -> None:
    pool = VideoPool(detector.live, detector.switches)
    rows = {name: row(name, scene, "first" if name == "feet_off_court" else None) for name, scene in scenes.items()}
    for name, scene in scenes.items():
        pool.add(rows[name], scene)
    (summary,) = pool.apply()
    # The reused scene offers no candidate, and its players' feet rule out both others.
    assert summary["scene_candidates"] == [
        {"view_id": view_id, "mean_combined_score": None,
         "rejection": {"view_id": "feet_off_court", "reason": "players_not_on_court"}}
        for view_id in ("first", "turned")
    ]
    assert (summary["chosen_court"], summary["pooled_view_ids"]) == ("pooled", ["first", "turned"])
    assert rows["feet_off_court"]["view_pool"]["court"] == "scene"


# Choosing between the pooled court and the scene courts, with stand-ins for the fit,
# the member checks and the scores.

SCENE = np.array([[10., 10.], [50., 10.], [50., 40.], [10., 40.]])
SCENE_B = SCENE + 3.0
SCENE_C = SCENE - 3.0
MIDDLE = SCENE + 2.0
POOLED = SCENE + 1.0
# One native px per working px
CONTEXT = SimpleNamespace(native_size=(100, 100), size=(100, 100))


def score(value: float | None) -> dict:
    return {} if value is None else {"combined_score": value}


def stand_in_group(monkeypatch, pooled: dict[str, tuple[float | None, str | None]], own: dict[str, float],
                   carried: dict[tuple[str, str], tuple[float | None, str | None]], fit_valid: bool = True,
                   court_mode: CourtMode = CourtMode.VIDEO_ROBUST) -> tuple[VideoPool, dict]:
    """Donor scenes a and b in one camera view, and a third scene.

    In video-robust mode the third is scene r, reusing a's court. Fast-robust mode reuses
    nothing, so the third is donor c with its own court.

    :param pooled: each member's pooled combined score (None when missing) and rejection
    :param own: each donor's combined score for its own scene court
    :param carried: (member, source scene) to that scene court's combined score and rejection
    """
    monkeypatch.setattr(view_pool, "pooled_constraints", lambda donors: (SimpleNamespace(points=[0] * 9), []))
    fit_corners = POOLED.tolist() if fit_valid else None
    monkeypatch.setattr(composition, "fit_in_reference", lambda live, reference, constraints: {
        "status": "ok", "valid": fit_valid, "validity_reason": None if fit_valid else "no_fit_corners",
        "corners_native_px": fit_corners})

    def member_outcome(self, group, member, pooled_reference) -> dict:
        view_id = member.row["view_id"]
        pooled_corners = POOLED
        value, rejection = pooled[view_id]
        if pooled_reference is None:
            pooled_corners, value, rejection = None, None, view_pool.NO_POOLED_FIT
        return {"pooled_corners": pooled_corners, "pooled_rejection": rejection,
                "scores": {"pooled": score(value), "scene": score(own.get(view_id, .5)), "middle": score(.5)}}

    third = ("r", SCENE, "a") if court_mode == CourtMode.VIDEO_ROBUST else ("c", SCENE_C, None)
    scenes = (("a", SCENE, None), ("b", SCENE_B, None), third)
    checked = []

    def checked_score(self, member, corners_native) -> tuple[dict, str | None]:
        # r's court is a's, which comes first.
        source = next(view_id for view_id, corners, _ in scenes if np.allclose(corners_native, corners))
        checked.append((member.row["view_id"], source))
        # Each scene court is measured at most once in each other member.
        assert len(set(checked)) == len(checked) and checked[-1][0] != source
        value, rejection = carried[checked[-1]]
        return score(value), rejection

    monkeypatch.setattr(VideoPool, "member_outcome", member_outcome)
    monkeypatch.setattr(VideoPool, "checked_score", checked_score)
    rows, members = {}, []
    donor_view_ids = []
    for view_id, corners, reused_from in scenes:
        rows[view_id] = {"view_id": view_id, "corners_native_px": corners.tolist(),
                         "chosen_key": "reuse" if reused_from else "composite", "reused_from": reused_from}
        members.append(Member(rows[view_id], CONTEXT, np.eye(3), None, corners, MIDDLE, None, None))
        if reused_from is None:
            donor_view_ids.append(view_id)
    group = ViewGroup(SimpleNamespace(corners_native=SCENE, native_per_working=np.ones(2)), np.zeros(1), members,
                      [None], donor_view_ids)
    pool = VideoPool(None, Switches(), court_mode)
    pool.groups.append(group)
    return pool, rows


POOLED_EVERYWHERE = {"a": (.80, None), "b": (.80, None), "r": (.80, None)}


def carried_scores(a_in_b: float, a_in_r: float, b_in_a: float, b_in_r: float) -> dict:
    return {("b", "a"): (a_in_b, None), ("r", "a"): (a_in_r, None), ("a", "b"): (b_in_a, None),
            ("r", "b"): (b_in_r, None)}


def test_a_scene_court_that_beats_the_pool_on_the_same_members_holds_the_group(monkeypatch) -> None:
    carried = carried_scores(.90, .85, .60, .60)
    pool, rows = stand_in_group(monkeypatch, POOLED_EVERYWHERE, {"a": .95, "b": .95}, carried)
    (summary,) = pool.apply()
    assert [(candidate["view_id"], candidate["rejection"]) for candidate in summary["scene_candidates"]] == [
        ("a", None), ("b", None)]
    assert [candidate["mean_combined_score"] for candidate in summary["scene_candidates"]] == pytest.approx(
        [.90, (.60 + .95 + .60) / 3])
    assert (summary["chosen_court"], summary["chosen_view_id"], summary["pooled_view_ids"]) == ("group_scene", "a", [])
    a, b, r = rows["a"], rows["b"], rows["r"]
    # The winning scene keeps its own court; the others take it and keep theirs beside it.
    assert (a["corners_native_px"], a["chosen_key"], a["view_pool"]["court"]) == (SCENE.tolist(), "composite",
                                                                                  "group_scene")
    assert (b["corners_native_px"], b["chosen_key"]) == (SCENE.tolist(), view_pool.GROUP_SCENE_KEY)
    assert (b["scene_corners_native_px"], b["view_pool"]["group_scene_view_id"]) == (SCENE_B.tolist(), "a")
    assert (r["reused_from"], r["scene_reused_from"]) == (None, "a")
    assert r["view_pool"]["scores"]["group_scene"] == {"combined_score": .85}
    assert r["view_pool"]["scores"]["pooled"] == {"combined_score": .80}


@pytest.mark.parametrize("case", ["tie", "only_per_member_maxima_beat_the_pool"])
def test_the_pool_holds_unless_one_scene_court_beats_it_on_the_same_members(monkeypatch, case: str) -> None:
    if case == "tie":
        own, carried = {"a": .80, "b": .70}, carried_scores(.80, .80, .70, .70)
    else:
        # Each scene court is best in its own scene, but neither beats the pool in all three.
        own, carried = {"a": .95, "b": .95}, carried_scores(.60, .70, .60, .70)
    pool, rows = stand_in_group(monkeypatch, POOLED_EVERYWHERE, own, carried)
    (summary,) = pool.apply()
    assert (summary["chosen_court"], summary["chosen_view_id"]) == ("pooled", None)
    assert summary["pooled_view_ids"] == ["a", "b", "r"]
    assert all(member_row["chosen_key"] == view_pool.POOLED_KEY for member_row in rows.values())
    assert "group_scene" not in rows["a"]["view_pool"]["scores"]


def test_a_scene_court_with_a_failed_check_or_missing_score_cannot_win(monkeypatch) -> None:
    carried = carried_scores(.99, .99, .99, .99)
    carried[("r", "a")] = (.99, "players_not_on_court")
    carried[("r", "b")] = (None, None)
    pool, rows = stand_in_group(monkeypatch, POOLED_EVERYWHERE, {"a": .99, "b": .99}, carried)
    (summary,) = pool.apply()
    assert [candidate["rejection"] for candidate in summary["scene_candidates"]] == [
        {"view_id": "r", "reason": "players_not_on_court"}, {"view_id": "r", "reason": "missing_score"}]
    assert (summary["chosen_court"], summary["pooled_view_ids"]) == ("pooled", ["a", "b", "r"])
    assert all(member_row["corners_native_px"] == POOLED.tolist() for member_row in rows.values())


def test_a_fully_scored_scene_court_replaces_a_pool_with_a_missing_score(monkeypatch) -> None:
    pooled = {**POOLED_EVERYWHERE, "r": (None, None)}
    pool, rows = stand_in_group(monkeypatch, pooled, {"a": .50, "b": .50}, carried_scores(.50, .50, .50, .40))
    (summary,) = pool.apply()
    # Missing terms stay missing, not zero.
    assert summary["mean_combined_scores"]["pooled"] is None
    assert rows["r"]["view_pool"]["scores"]["pooled"] == {}
    assert (summary["chosen_court"], summary["chosen_view_id"]) == ("group_scene", "a")
    assert rows["b"]["corners_native_px"] == SCENE.tolist()


def test_failed_member_checks_or_an_invalid_fit_keep_scene_courts(monkeypatch) -> None:
    pooled = {**POOLED_EVERYWHERE, "r": (.99, "players_not_on_court")}
    pool, rows = stand_in_group(monkeypatch, pooled, {"a": .50, "b": .50}, carried_scores(.50, .50, .50, .50))
    (summary,) = pool.apply()
    assert summary["pooled_view_ids"] == ["a", "b"]
    reused = rows["r"]
    assert (reused["corners_native_px"], reused["chosen_key"], reused["reused_from"]) == (SCENE.tolist(), "reuse", "a")
    assert (reused["view_pool"]["court"], reused["view_pool"]["pooled_rejection"]) == ("scene", "players_not_on_court")

    pool, rows = stand_in_group(monkeypatch, POOLED_EVERYWHERE, {"a": .99, "b": .99},
                                carried_scores(.99, .99, .99, .99), fit_valid=False)
    (summary,) = pool.apply()
    assert (summary["reason"], summary["pooled_view_ids"], summary["chosen_court"]) == ("fit_no_fit_corners", [], None)
    assert rows["a"]["corners_native_px"] == SCENE.tolist() and rows["b"]["corners_native_px"] == SCENE_B.tolist()
    assert all("view_pool" not in member_row for member_row in rows.values())


def carried_everywhere(by_source: dict[str, float]) -> dict:
    """Each source scene's court with the same score and no rejection in every other member."""
    carried = {}
    for source, value in by_source.items():
        for member in by_source:
            if member != source:
                carried[(member, source)] = (value, None)
    return carried


@pytest.mark.parametrize(("pooled_score", "chosen_court"), [(.80, "group_scene"), (.81, "pooled")])
def test_fast_robust_pool_must_beat_the_best_complete_court_on_the_same_members(
        monkeypatch, pooled_score: float, chosen_court: str) -> None:
    pooled = {view_id: (pooled_score, None) for view_id in "abc"}
    # The pool fails c's own checks, so c keeps its scene court when the pool wins.
    pooled["c"] = (pooled_score, "players_not_on_court")
    # a's court scores .80 in every member; b's and c's score less.
    carried = carried_everywhere({"a": .80, "b": .60, "c": .60})
    pool, rows = stand_in_group(monkeypatch, pooled, {"a": .80, "b": .70, "c": .70}, carried,
                                court_mode=CourtMode.FAST_ROBUST)
    (summary,) = pool.apply()
    assert summary["donor_view_ids"] == ["a", "b", "c"]
    best = summary["scene_candidates"][0]
    assert (best["view_id"], best["rejection"], best["mean_combined_score"]) == ("a", None, pytest.approx(.80))
    assert summary["chosen_court"] == chosen_court
    if chosen_court == "group_scene":
        # An exact tie keeps the complete court, which every member now holds.
        assert best["mean_combined_score"] == summary["mean_combined_scores"]["pooled"]
        assert (summary["chosen_view_id"], summary["pooled_view_ids"]) == ("a", [])
        assert all(member_row["corners_native_px"] == SCENE.tolist() for member_row in rows.values())
        assert (rows["a"]["chosen_key"], rows["c"]["chosen_key"]) == ("composite", view_pool.GROUP_SCENE_KEY)
        return
    assert summary["pooled_view_ids"] == ["a", "b"]
    assert rows["b"]["corners_native_px"] == POOLED.tolist()
    assert (rows["c"]["corners_native_px"], rows["c"]["view_pool"]["court"]) == (SCENE_C.tolist(), "scene")


@pytest.mark.parametrize("candidates_pass", [True, False])
def test_fast_robust_can_share_a_complete_court_when_the_pooled_fit_fails(monkeypatch, candidates_pass: bool) -> None:
    carried = carried_everywhere({"a": .70, "b": .60, "c": .60})
    if not candidates_pass:
        carried = {pair: (.99, "camera_implausible") for pair in carried}
    pooled = {view_id: (.99, None) for view_id in "abc"}
    pool, rows = stand_in_group(monkeypatch, pooled, {"a": .90, "b": .90, "c": .90}, carried, fit_valid=False,
                                court_mode=CourtMode.FAST_ROBUST)
    (summary,) = pool.apply()
    assert summary["reason"] == "fit_no_fit_corners"
    assert summary["fit"]["corners_reference_native_px"] is None
    assert summary["mean_combined_scores"]["pooled"] is None
    json.dumps([summary, rows], allow_nan=False)
    if not candidates_pass:
        # No court holds the whole group, so every scene keeps its own.
        assert all(candidate["rejection"] is not None for candidate in summary["scene_candidates"])
        assert (summary["chosen_court"], summary["pooled_view_ids"]) == (None, [])
        assert [rows[view_id]["corners_native_px"] for view_id in "abc"] == [
            SCENE.tolist(), SCENE_B.tolist(), SCENE_C.tolist()]
        assert all("view_pool" not in member_row for member_row in rows.values())
        return
    assert (summary["chosen_court"], summary["chosen_view_id"], summary["pooled_view_ids"]) == ("group_scene", "a", [])
    assert all(member_row["corners_native_px"] == SCENE.tolist() for member_row in rows.values())
    record = rows["b"]["view_pool"]
    assert (record["court"], record["pooled_rejection"], record["pooled_corners_native_px"]) == (
        "group_scene", view_pool.NO_POOLED_FIT, None)
    assert record["scores"]["pooled"] == {}
