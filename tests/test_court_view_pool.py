"""Video-robust courts: view groups, donors carried between scenes, one pooled fit and where it applies.

The measured cases reuse test_court_scene_compose's synthetic painted court and compose
real scenes from it. The fast-robust cases give the pool fresh middle-frame courts
instead. The output cases replace the fit, member checks and scores with stand-ins.
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from court_detector import composition, view_pool
from court_detector.detect import CourtDetector, SceneCourts, Switches
from court_detector.view_pool import CourtMode, Member, Receiver, VideoPool, ViewGroup
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

    Its fresh middle-frame fit remains a donor in both robust video modes.
    """
    frame = searched(detector, spec, .8, all_feet_px)
    return SceneCourts(frame.context, frame.native_frame, frame.corners_native, frame.corners_native)


def row(view_id: str, scene: SceneCourts, reused_from: str | None = None) -> dict[str, Any]:
    return {"view_id": view_id, "corners_native_px": np.asarray(scene.corners_native_px).tolist(),
            "chosen_key": "reuse" if reused_from else "composite", "reused_from": reused_from}


def courtless_row(view_id: str) -> dict[str, Any]:
    """A scene whose net choice picked a court that then failed the refit's player check."""
    return {"view_id": view_id, "status": "no_court", "corners_native_px": None, "chosen_key": "net_pick",
            "no_court_reason": "refit_players_not_on_court", "reused_from": None}


def courtless(detector: CourtDetector, spec: dict, all_feet_px: list[list]) -> SimpleNamespace:
    """The prepared middle frame and context detect() hands on for a scene without a court."""
    frame = searched(detector, spec, None, all_feet_px)
    return SimpleNamespace(native_frame=frame.native_frame, context=frame.context)


def without_receivers(summary: dict[str, Any]) -> str:
    """A group summary's JSON without its receiver lists, to compare groups with and without receivers."""
    kept = {key: value for key, value in summary.items() if key not in ("receiver_view_ids", "received_view_ids")}
    return json.dumps(kept, sort_keys=True)


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


def test_one_pooled_court_reaches_each_member_even_when_players_leave(detector, scenes) -> None:
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
    assert summary["pooled_view_ids"] == ["first", "turned", "feet_off_court"]
    turned = rows["turned"]
    np.testing.assert_allclose(turned["corners_native_px"], true_corners(NUDGE_PX), atol=2.0)
    assert (turned["chosen_key"], turned["view_pool"]["court"]) == (view_pool.POOLED_KEY, "pooled")
    assert turned["scene_corners_native_px"] == np.asarray(scenes["turned"].corners_native_px).tolist()
    means = summary["mean_combined_scores"]
    assert all(means[court] is not None for court in view_pool.COMPARED_COURTS), means
    off_court = rows["feet_off_court"]
    assert off_court["view_pool"]["pooled_rejection"] is None
    assert off_court["view_pool"]["court"] == "pooled"
    assert off_court["view_pool"]["scores"]["pooled"]["combined_score"] is not None
    # The runner saves rows and summaries as strict JSON.
    json.dumps([summary, rows], allow_nan=False)


def test_a_single_donor_scene_does_not_pool(detector, scenes) -> None:
    pool = VideoPool(detector.live, detector.switches)
    rows = {name: row(name, scenes[name], reused_from="first" if name == "feet_off_court" else None)
            for name in ("first", "feet_off_court")}
    for name, member_row in rows.items():
        pool.add(member_row, scenes[name])
    # A courtless scene of the same view joins the group but adds no donor.
    receiver_row = courtless_row("receiver")
    nudged = frame_spec("middle", NUDGE_PX, [LEFT_BOX], true_corners(NUDGE_PX))
    pool.add_receiver(receiver_row, courtless(detector, nudged, players_feet(NUDGE_PX)))
    (summary,) = pool.apply()
    assert (summary["member_view_ids"], summary["donor_view_ids"]) == (["first", "feet_off_court"], ["first"])
    assert (summary["pooled_view_ids"], summary["reason"]) == ([], "too_few_donor_scenes")
    assert (summary["receiver_view_ids"], summary["received_view_ids"]) == (["receiver"], [])
    assert all("view_pool" not in member_row for member_row in rows.values())
    assert receiver_row == courtless_row("receiver")


@pytest.mark.parametrize("case", ["before_donors", "after_donors", "unrelated_view"])
def test_a_courtless_scene_takes_the_pooled_court_in_any_order_without_changing_it(
        detector, scenes, monkeypatch, case: str) -> None:
    def pooled(receiver: SimpleNamespace | None) -> tuple[dict, dict[str, dict], dict]:
        pool = VideoPool(detector.live, detector.switches)
        receiver_row = courtless_row("receiver")
        if receiver is not None and case == "before_donors":
            pool.add_receiver(receiver_row, receiver)
        rows = {name: row(name, scenes[name]) for name in ("first", "turned")}
        for name, member_row in rows.items():
            pool.add(member_row, scenes[name])
        if receiver is not None and case != "before_donors":
            pool.add_receiver(receiver_row, receiver)
        (summary,) = pool.apply()
        return summary, rows, receiver_row

    spec = frame_spec("middle", NUDGE_PX, [LEFT_BOX], true_corners(NUDGE_PX))
    if case == "unrelated_view":
        # Every hash matches, so the alignment alone keeps the unrelated image out.
        monkeypatch.setattr(view_pool, "image_hash", lambda native_frame: np.zeros((16, 16), dtype=bool))
        spec = unrelated("middle", NUDGE_PX)
    # Both players stand off the court, as when the scene's own fit failed the refit's player check.
    receiver = courtless(detector, spec, OFF_COURT_FEET_PX)
    baseline, baseline_rows, _ = pooled(None)
    summary, rows, receiver_row = pooled(receiver)
    # The receiver changes neither the donors, the fit, the scores, the choice nor any member's court.
    assert without_receivers(summary) == without_receivers(baseline)
    assert rows == baseline_rows
    assert summary["chosen_court"] == "pooled"
    if case == "unrelated_view":
        assert (summary["receiver_view_ids"], summary["received_view_ids"]) == ([], [])
        assert receiver_row == courtless_row("receiver")
        return
    assert (summary["receiver_view_ids"], summary["received_view_ids"]) == (["receiver"], ["receiver"])
    np.testing.assert_allclose(receiver_row["corners_native_px"], true_corners(NUDGE_PX), atol=2.0)
    assert (receiver_row["status"], receiver_row["no_court_reason"], receiver_row["chosen_key"]) == (
        "court", None, view_pool.POOLED_KEY)
    assert (receiver_row["scene_status"], receiver_row["scene_no_court_reason"]) == (
        "no_court", "refit_players_not_on_court")
    assert (receiver_row["scene_corners_native_px"], receiver_row["scene_chosen_key"]) == (None, "net_pick")
    record = receiver_row["view_pool"]
    assert (record["receiver"], record["court"], record["rejection"]) == (True, "pooled", None)
    assert record["alignment"]["usable"]
    # The player check that a required-people search applies would still reject this court here.
    received = np.asarray(receiver_row["corners_native_px"])
    _, rejection = composition.check_in_frame(detector.live, receiver.context, received, require_people=True,
                                              max_horizon_tilt_deg=None)
    assert rejection == "players_not_on_court"
    json.dumps([summary, receiver_row], allow_nan=False)


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
                    scene.middle_corners_native_px, scene.middle_score, scene.chosen_measurement)
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


@pytest.mark.parametrize("court_mode", [CourtMode.FAST_ROBUST, CourtMode.VIDEO_ROBUST])
def test_fresh_middle_frame_courts_remain_video_donors(detector, middle_scenes, court_mode: CourtMode) -> None:
    pool = VideoPool(detector.live, detector.switches, court_mode)
    for name in ("left_box", "turned"):
        pool.add(row(name, middle_scenes[name]), middle_scenes[name])
    (summary,) = pool.apply()
    assert summary["donor_view_ids"] == ["left_box", "turned"]
    if court_mode == CourtMode.FAST_ROBUST:
        assert summary["reason"] == "too_few_donor_scenes"
    else:
        assert summary["fit"]["valid"]
        assert summary["chosen_court"] in {"pooled", "group_scene"}


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


def test_off_court_players_do_not_disqualify_shared_scene_candidates(detector, scenes, misfit_pool) -> None:
    pool = VideoPool(detector.live, detector.switches)
    rows = {name: row(name, scene, "first" if name == "feet_off_court" else None) for name, scene in scenes.items()}
    for name, scene in scenes.items():
        pool.add(rows[name], scene)
    (summary,) = pool.apply()
    candidates = summary["scene_candidates"]
    assert [candidate["view_id"] for candidate in candidates] == ["first", "turned"]
    assert all(candidate["rejection"] is None for candidate in candidates)
    assert all(candidate["mean_combined_score"] is not None for candidate in candidates)
    assert all((candidate["measured_members"], candidate["transfer_rejections"]) == (3, []) for candidate in candidates)
    assert summary["chosen_court"] == "group_scene"
    assert rows["feet_off_court"]["view_pool"]["court"] == "group_scene"


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


def stand_in_group(monkeypatch, pooled: dict[str, tuple[float | None, str | None]], own: dict[str, float | None],
                   carried: dict[tuple[str, str], tuple[float | None, str | None]], fit_valid: bool = True,
                   court_mode: CourtMode = CourtMode.VIDEO_ROBUST,
                   receivers: dict[str, tuple[float | None, str | None]] | None = None) -> tuple[VideoPool, dict]:
    """Donor scenes a and b in one camera view, and a third scene.

    In video-robust mode the third is scene r, reusing a's court. Fast-robust mode reuses
    nothing, so the third is donor c with its own court. Scene a is the group's reference,
    where the pooled court is fitted.

    :param pooled: each member's pooled combined score (None when missing) and rejection
    :param own: each donor's combined score for its own scene court (None when missing)
    :param carried: (member, source scene) to that scene court's combined score and rejection
    :param receivers: each courtless receiver's combined score for the chosen court and its rejection
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
        if isinstance(member, Receiver):
            value, rejection = receivers[member.row["view_id"]]
            return score(value), rejection
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
    for view_id in receivers or {}:
        rows[view_id] = courtless_row(view_id)
        group.receivers.append(Receiver(rows[view_id], CONTEXT, np.eye(3), {}))
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


@pytest.mark.parametrize(("b_in_r", "reason", "measured_members"), [
    ((.90, "camera_implausible"), "camera_implausible", 3),
    ((None, "non_finite_or_non_convex_corners"), "non_finite_or_non_convex_corners", 2),
    ((None, None), view_pool.MISSING_SCORE, 2),
])
def test_a_failed_transfer_keeps_only_that_member_on_its_own_court(
        monkeypatch, b_in_r: tuple[float | None, str | None], reason: str, measured_members: int) -> None:
    # b's court beats the pool and a's court, but r cannot take it.
    carried = carried_scores(.60, .60, .90, .90)
    carried[("r", "b")] = b_in_r
    pool, rows = stand_in_group(monkeypatch, POOLED_EVERYWHERE, {"a": .60, "b": .95}, carried)
    if reason == "non_finite_or_non_convex_corners":
        original_carry = view_pool.carry_to_member
        original_check = VideoPool.checked_score

        def broken_transfer(corners: np.ndarray, member: Member) -> np.ndarray:
            if member.row["view_id"] == "r" and np.array_equal(corners, SCENE_B):
                return np.full((4, 2), np.nan)
            return original_carry(corners, member)

        def check_transfer(self: VideoPool, member: Member, corners: np.ndarray) -> tuple[dict, str | None]:
            if not np.isfinite(corners).all():
                return {}, reason
            return original_check(self, member, corners)

        monkeypatch.setattr(view_pool, "carry_to_member", broken_transfer)
        monkeypatch.setattr(VideoPool, "checked_score", check_transfer)
    (summary,) = pool.apply()
    # A score beside a failed check still counts; a missing score leaves the member out of the mean.
    mean = (.95 + .90 + .90) / 3 if measured_members == 3 else (.95 + .90) / 2
    assert summary["scene_candidates"][1] == {
        "view_id": "b", "mean_combined_score": pytest.approx(mean), "measured_members": measured_members,
        "rejection": None, "transfer_rejections": [{"view_id": "r", "reason": reason}]}
    assert (summary["chosen_court"], summary["chosen_view_id"]) == ("group_scene", "b")
    a, r = rows["a"], rows["r"]
    assert (a["corners_native_px"], a["chosen_key"]) == (SCENE_B.tolist(), view_pool.GROUP_SCENE_KEY)
    assert (a["view_pool"]["court"], a["view_pool"]["group_scene_rejection"]) == ("group_scene", None)
    # r's row keeps its reused court, with no replaced court beside it.
    assert (r["corners_native_px"], r["chosen_key"], r["reused_from"]) == (SCENE.tolist(), "reuse", "a")
    assert "scene_corners_native_px" not in r
    assert (r["view_pool"]["court"], r["view_pool"]["group_scene_rejection"]) == ("scene", reason)
    assert r["view_pool"]["group_scene_view_id"] == "b"
    if reason == "non_finite_or_non_convex_corners":
        assert r["view_pool"]["group_scene_corners_native_px"] is None
    json.dumps([summary, rows], allow_nan=False)


def test_a_scene_court_without_its_own_score_is_out_at_its_source(monkeypatch) -> None:
    # a's court scores best wherever it is carried, but has no score in its own frame.
    pool, rows = stand_in_group(monkeypatch, POOLED_EVERYWHERE, {"a": None, "b": .95},
                                carried_scores(.99, .99, .90, .90))
    (summary,) = pool.apply()
    a, b = summary["scene_candidates"]
    assert a["rejection"] == {"view_id": "a", "reason": view_pool.MISSING_SCORE}
    assert (b["rejection"], b["transfer_rejections"]) == (None, [])
    assert (summary["chosen_court"], summary["chosen_view_id"]) == ("group_scene", "b")
    assert all(member_row["corners_native_px"] == SCENE_B.tolist() for member_row in rows.values())


def test_a_pool_one_member_cannot_score_is_ranked_on_the_members_that_can(monkeypatch) -> None:
    pooled = {**POOLED_EVERYWHERE, "r": (None, None)}
    pool, rows = stand_in_group(monkeypatch, pooled, {"a": .50, "b": .50}, carried_scores(.50, .50, .50, .40))
    (summary,) = pool.apply()
    # Missing terms stay missing, not zero: r neither rules the pool out nor lowers its mean.
    assert summary["mean_combined_scores"]["pooled"] is None
    assert summary["pooled_candidate"] == {
        "view_id": None, "mean_combined_score": pytest.approx(.80), "measured_members": 2, "rejection": None,
        "transfer_rejections": [{"view_id": "r", "reason": view_pool.MISSING_SCORE}]}
    assert (summary["chosen_court"], summary["pooled_view_ids"]) == ("pooled", ["a", "b"])
    reused = rows["r"]
    assert reused["view_pool"]["scores"]["pooled"] == {}
    assert (reused["corners_native_px"], reused["view_pool"]["court"]) == (SCENE.tolist(), "scene")
    assert reused["view_pool"]["pooled_rejection"] == view_pool.MISSING_SCORE


@pytest.mark.parametrize("failed_score", [.99, None])
def test_a_member_that_rejects_the_pool_keeps_its_scene_court(monkeypatch, failed_score) -> None:
    pooled = {**POOLED_EVERYWHERE, "r": (failed_score, "camera_implausible")}
    pool, rows = stand_in_group(monkeypatch, pooled, {"a": .50, "b": .50}, carried_scores(.50, .50, .50, .50))
    (summary,) = pool.apply()
    assert (summary["chosen_court"], summary["reason"], summary["pooled_view_ids"]) == ("pooled", None, ["a", "b"])
    reused = rows["r"]
    assert (reused["corners_native_px"], reused["chosen_key"], reused["reused_from"]) == (SCENE.tolist(), "reuse", "a")
    assert (reused["view_pool"]["court"], reused["view_pool"]["pooled_rejection"]) == ("scene", "camera_implausible")


def test_a_pool_that_fails_the_reference_check_is_out_for_the_whole_group(monkeypatch) -> None:
    # The pool is fitted in a's frame, so a's check is a check of the fit itself. Its
    # mean would otherwise beat both scene courts.
    pooled = {**POOLED_EVERYWHERE, "a": (.99, "camera_implausible")}
    pool, rows = stand_in_group(monkeypatch, pooled, {"a": .50, "b": .50}, carried_scores(.50, .50, .40, .40))
    (summary,) = pool.apply()
    assert summary["reason"] == "reference_camera_implausible"
    assert summary["pooled_candidate"]["rejection"] == {"view_id": "a", "reason": "camera_implausible"}
    assert (summary["chosen_court"], summary["chosen_view_id"], summary["pooled_view_ids"]) == ("group_scene", "a", [])
    assert all(member_row["corners_native_px"] == SCENE.tolist() for member_row in rows.values())


def test_video_robust_shares_the_best_existing_court_when_the_pool_is_invalid(monkeypatch) -> None:
    pool, rows = stand_in_group(monkeypatch, POOLED_EVERYWHERE, {"a": .99, "b": .99},
                                carried_scores(.99, .99, .99, .99), fit_valid=False)
    (summary,) = pool.apply()
    assert (summary["reason"], summary["pooled_view_ids"], summary["chosen_court"]) == (
        "fit_no_fit_corners", [], "group_scene")
    assert summary["chosen_view_id"] == "a"
    assert all(member_row["corners_native_px"] == SCENE.tolist() for member_row in rows.values())


def test_a_retained_individual_can_win_video_sharing_and_replace_its_source_scene(monkeypatch) -> None:
    pool, rows = stand_in_group(monkeypatch, POOLED_EVERYWHERE, {"a": .60, "b": .60},
                                carried_scores(.60, .60, .60, .60))
    group = pool.groups[0]
    individual = SCENE + [1., 0.]
    measurement = {"paint_score": .99, "geometry_score": .99}
    group.members[0] = replace(group.members[0], individual_courts=(("first", individual, measurement),))
    original_check = pool.checked_score

    def checked_score(member, corners):
        if np.allclose(corners, individual):
            return score(.99), None
        return original_check(member, corners)

    monkeypatch.setattr(pool, "checked_score", checked_score)
    monkeypatch.setattr(pool, "net_reward", lambda *args: 0.)
    (summary,) = pool.apply()
    assert (summary["chosen_view_id"], summary["chosen_frame_role"]) == ("a", "first")
    assert all(member_row["corners_native_px"] == individual.tolist() for member_row in rows.values())
    assert rows["a"]["chosen_key"] == view_pool.GROUP_SCENE_KEY


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
    pooled["c"] = (pooled_score, "camera_implausible")
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


@pytest.mark.parametrize("transfers_pass", [True, False])
def test_fast_robust_can_share_a_complete_court_when_the_pooled_fit_fails(monkeypatch, transfers_pass: bool) -> None:
    carried = carried_everywhere({"a": .70, "b": .60, "c": .60})
    if not transfers_pass:
        carried = {pair: (value, "camera_implausible") for pair, (value, _) in carried.items()}
    pooled = {view_id: (.99, None) for view_id in "abc"}
    pool, rows = stand_in_group(monkeypatch, pooled, {"a": .90, "b": .90, "c": .90}, carried, fit_valid=False,
                                court_mode=CourtMode.FAST_ROBUST)
    (summary,) = pool.apply()
    assert summary["reason"] == "fit_no_fit_corners"
    assert summary["fit"]["corners_reference_native_px"] is None
    assert summary["mean_combined_scores"]["pooled"] is None and summary["pooled_candidate"] is None
    json.dumps([summary, rows], allow_nan=False)
    # a's court scores best in the same three frames, whether or not the others accept it.
    assert (summary["chosen_court"], summary["chosen_view_id"], summary["pooled_view_ids"]) == ("group_scene", "a", [])
    record = rows["b"]["view_pool"]
    assert (record["pooled_rejection"], record["pooled_corners_native_px"]) == (view_pool.NO_POOLED_FIT, None)
    assert record["scores"]["pooled"] == {}
    if not transfers_pass:
        # Every transfer fails, so each scene keeps its own court and records why.
        assert [rows[view_id]["corners_native_px"] for view_id in "abc"] == [
            SCENE.tolist(), SCENE_B.tolist(), SCENE_C.tolist()]
        assert (record["court"], record["group_scene_rejection"]) == ("scene", "camera_implausible")
        assert rows["b"]["chosen_key"] == "composite"
        return
    assert all(member_row["corners_native_px"] == SCENE.tolist() for member_row in rows.values())
    assert (record["court"], record["group_scene_rejection"]) == ("group_scene", None)


@pytest.mark.parametrize(("court_mode", "pooled", "own", "carried", "chosen_court", "chosen_corners"), [
    (CourtMode.VIDEO_ROBUST, POOLED_EVERYWHERE, {"a": .95, "b": .95}, carried_scores(.90, .85, .60, .60),
     "group_scene", SCENE),
    (CourtMode.FAST_ROBUST, {view_id: (.81, None) for view_id in "abc"}, {"a": .80, "b": .70, "c": .70},
     carried_everywhere({"a": .80, "b": .60, "c": .60}), "pooled", POOLED),
])
def test_courtless_receivers_take_only_the_chosen_court_and_change_nothing_else(
        monkeypatch, court_mode: CourtMode, pooled: dict, own: dict, carried: dict, chosen_court: str,
        chosen_corners: np.ndarray) -> None:
    # Receivers that would score far above every candidate, if they were scored.
    receivers = {"takes": (.99, None), "rejects": (.99, "camera_implausible"), "unscored": (None, None)}
    baseline_pool, baseline_rows = stand_in_group(monkeypatch, pooled, own, carried, court_mode=court_mode)
    (baseline,) = baseline_pool.apply()
    pool, rows = stand_in_group(monkeypatch, pooled, own, carried, court_mode=court_mode, receivers=receivers)
    (summary,) = pool.apply()
    assert without_receivers(summary) == without_receivers(baseline)
    assert all(rows[view_id] == member_row for view_id, member_row in baseline_rows.items())
    assert summary["chosen_court"] == chosen_court
    assert (summary["receiver_view_ids"], summary["received_view_ids"]) == (list(receivers), ["takes"])
    takes = rows["takes"]
    chosen_key = view_pool.POOLED_KEY if chosen_court == "pooled" else view_pool.GROUP_SCENE_KEY
    assert (takes["corners_native_px"], takes["chosen_key"], takes["status"]) == (
        chosen_corners.tolist(), chosen_key, "court")
    assert (takes["scene_status"], takes["scene_corners_native_px"]) == ("no_court", None)
    assert takes["view_pool"]["court"] == chosen_court
    # A failed check or a missing score leaves the scene without a court.
    for view_id, reason in (("rejects", "camera_implausible"), ("unscored", view_pool.MISSING_SCORE)):
        left = rows[view_id]
        assert {key: left[key] for key in courtless_row(view_id)} == courtless_row(view_id)
        assert "scene_status" not in left
        assert (left["view_pool"]["court"], left["view_pool"]["rejection"]) == (None, reason)
    json.dumps([summary, rows], allow_nan=False)
