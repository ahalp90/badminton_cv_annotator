"""Video-robust courts: view groups, donors carried between scenes, one pooled fit and where it applies.

The measured cases reuse test_court_scene_compose's synthetic painted court and compose
real scenes from it. The output cases replace the fit, member checks and scores with stand-ins.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from court_detector import composition, view_pool
from court_detector.detect import CourtDetector, SceneCourts, Switches
from court_detector.view_pool import Member, VideoPool, ViewGroup
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


def reused_scene(detector: CourtDetector, spec: dict, all_feet_px: list[list]) -> SceneCourts:
    """A scene whose court came without a composite, so it has nothing to donate."""
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
            "feet_off_court": reused_scene(detector, unmoved, OFF_COURT_FEET_PX)}


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


@pytest.mark.parametrize("case", ["unrelated_image", "moved_camera"])
def test_matching_hashes_still_need_the_same_camera_to_join(detector, scenes, monkeypatch, case: str) -> None:
    monkeypatch.setattr(view_pool, "image_hash", lambda native_frame: np.zeros((16, 16), dtype=bool))
    if case == "unrelated_image":
        other = reused_scene(detector, unrelated("middle", np.zeros(2)), players_feet(np.zeros(2)))
    else:
        moved = frame_spec("middle", JITTER_PX, [RIGHT_BOX], true_corners(JITTER_PX))
        other = reused_scene(detector, moved, players_feet(JITTER_PX))
    pool = VideoPool(detector.live, detector.switches)
    pool.add(row("first", scenes["first"]), scenes["first"])
    pool.add(row("other", other), other)
    assert [[member.row["view_id"] for member in group.members] for group in pool.groups] == [["first"], ["other"]]


# Applying the pooled court, with stand-ins for the fit and the member checks and scores.

SCENE = np.array([[10., 10.], [50., 10.], [50., 40.], [10., 40.]])
MIDDLE = SCENE + 2.0
POOLED = SCENE + 1.0


def scores(scene: float, pooled: float | None, middle: float) -> dict[str, dict]:
    return {"pooled": {} if pooled is None else {"combined_score": pooled}, "scene": {"combined_score": scene},
            "middle": {"combined_score": middle}}


def outcome(values: dict[str, dict], rejection: str | None = None) -> dict:
    return {"pooled_corners": POOLED, "pooled_rejection": rejection, "scores": values}


def stand_in_group(monkeypatch, outcomes: dict[str, dict], fit_valid: bool = True) -> tuple[VideoPool, dict]:
    """A composite scene and a reused scene in one group, checked and scored as outcomes says."""
    monkeypatch.setattr(view_pool, "pooled_constraints", lambda donors: (SimpleNamespace(points=[0] * 9), []))
    monkeypatch.setattr(composition, "fit_in_reference", lambda live, reference, constraints: {
        "status": "ok", "valid": fit_valid, "validity_reason": None if fit_valid else "no_fit_corners",
        "corners_native_px": POOLED.tolist()})
    monkeypatch.setattr(VideoPool, "member_outcome", lambda self, group, member, pooled: outcomes[member.row["view_id"]])
    rows = {"composite": {"view_id": "composite", "corners_native_px": SCENE.tolist(), "chosen_key": "composite",
                          "reused_from": None},
            "reused": {"view_id": "reused", "corners_native_px": SCENE.tolist(), "chosen_key": "reuse",
                       "reused_from": "composite"}}
    members = [Member(rows["composite"], None, np.eye(3), None, SCENE, MIDDLE, None, None),
               Member(rows["reused"], None, np.eye(3), None, SCENE, SCENE, None, None)]
    group = ViewGroup(SimpleNamespace(corners_native=SCENE), np.zeros(1), members, [None], ["composite", "other"])
    pool = VideoPool(None, Switches())
    pool.groups.append(group)
    return pool, rows


def test_a_valid_pool_is_emitted_even_when_it_scores_lower_or_a_score_is_missing(monkeypatch) -> None:
    # The pooled court scores below both other courts in one member, and has no score in the other.
    pool, rows = stand_in_group(monkeypatch, {"composite": outcome(scores(.90, .80, .95)),
                                              "reused": outcome(scores(.90, None, .90))})
    (summary,) = pool.apply()
    assert summary["mean_combined_scores"] == pytest.approx({"pooled": None, "scene": .90, "middle": .925})
    assert (summary["reason"], summary["pooled_view_ids"]) == (None, ["composite", "reused"])
    composite, reused = rows["composite"], rows["reused"]
    assert (composite["corners_native_px"], composite["chosen_key"]) == (POOLED.tolist(), view_pool.POOLED_KEY)
    assert (composite["scene_corners_native_px"], composite["scene_chosen_key"]) == (SCENE.tolist(), "composite")
    assert (reused["reused_from"], reused["scene_reused_from"]) == (None, "composite")
    # Missing terms are reported as missing, not as zero.
    assert reused["view_pool"]["scores"]["pooled"] == {}


def test_failed_member_checks_or_an_invalid_fit_keep_scene_courts(monkeypatch) -> None:
    pool, rows = stand_in_group(monkeypatch, {"composite": outcome(scores(.90, .95, .80)),
                                              "reused": outcome(scores(.90, .99, .80), "players_not_on_court")})
    (summary,) = pool.apply()
    assert summary["pooled_view_ids"] == ["composite"]
    reused = rows["reused"]
    assert (reused["corners_native_px"], reused["chosen_key"], reused["reused_from"]) == (SCENE.tolist(), "reuse",
                                                                                          "composite")
    assert (reused["view_pool"]["court"], reused["view_pool"]["pooled_rejection"]) == ("scene", "players_not_on_court")

    pool, rows = stand_in_group(monkeypatch, {"composite": outcome(scores(.90, .95, .80)),
                                              "reused": outcome(scores(.90, .95, .80))}, fit_valid=False)
    (summary,) = pool.apply()
    assert (summary["reason"], summary["pooled_view_ids"]) == ("fit_no_fit_corners", [])
    assert all(member_row["corners_native_px"] == SCENE.tolist() for member_row in rows.values())
    assert all("view_pool" not in member_row for member_row in rows.values())
