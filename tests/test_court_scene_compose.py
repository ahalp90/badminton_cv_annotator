"""Composite courts: warp direction, people-free alignment, turned courts, donated samples, fallbacks, fair comparison.

The synthetic frames show a painted court through a pinhole camera, with DeepLSD-like
fragments along the stripe centres and a "person" box over one doubles line.
"""

from __future__ import annotations

import copy
import csv
import dataclasses
import gzip
import json
import sys
from math import ceil, floor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from annotator import court_views
from court_detector import composition, reuse
from court_detector.detect import LiveModules, load_live_modules
from court_detector.line_observations import MARKINGS, distances_to_segments
from court_detector.paint_geometry import CENTRE_SEGMENTS_M, STRIPE_WIDTH_M
from experiments.annotator.court_scene_sampling import compose, rescore
from experiments.annotator.court_scene_sampling.sampling import METHODS, paint_evidence
from shared.court import HOMOGRAPHY_RESOLUTION
from shared.court_model import CORNER_COURT_M

NATIVE_SIZE = (1200, 900)  # working (960, 720); view images are 960x540, so y scales differently
CAMERA_M = (3.05, 20.5, -5.5)  # behind and above the near baseline; the court's z axis points into the floor
TARGET_M = (3.05, 6.0, 0.0)
FOCAL_PX = 1200.0
JITTER_PX = np.array([10.0, -6.0])  # the first frame's camera movement, beyond reuse's same-camera limit
LEFT_BOX = (310.0, 360.0, 400.0, 580.0)  # a person on the left doubles line, native px
RIGHT_BOX = (810.0, 354.0, 900.0, 574.0)  # a person on the right doubles line in the jittered frame
# Two players inside the court, then the same two elsewhere; enough to drop the unmasked correlation below 0.8.
FIRST_PEOPLE = [(380.0, 330.0, 500.0, 610.0), (640.0, 360.0, 760.0, 640.0)]
MIDDLE_PEOPLE = [(510.0, 340.0, 630.0, 620.0), (760.0, 450.0, 880.0, 730.0)]
# Past reuse's same-camera limit. From JITTER_PX, ECC does not converge once these players are cut out.
NUDGE_PX = np.array([6.0, -4.0])
# The same painted line after a 180 degree turn, by marking index.
TURNED_MARKING = {"left_doubles": "right_doubles", "left_singles": "right_singles", "centre": "centre",
                  "far_baseline": "near_baseline", "far_long_service": "near_long_service",
                  "far_short_service": "near_short_service"}
TURNED_MARKING |= {turned: name for name, turned in TURNED_MARKING.items()}
FRAGMENT_SPANS = ((0.0, 0.22), (0.26, 0.48), (0.52, 0.74), (0.78, 1.0))  # four pieces per stripe, with gaps


def camera_homography() -> np.ndarray:
    """Court metres to native px for an upright pinhole camera with a centred principal point."""
    camera, target = np.asarray(CAMERA_M), np.asarray(TARGET_M)
    forward = (target - camera) / np.linalg.norm(target - camera)
    right = np.cross(forward, (0.0, 0.0, -1.0))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack((right, down, forward))
    intrinsic = np.array([[FOCAL_PX, 0, NATIVE_SIZE[0] / 2], [0, FOCAL_PX, NATIVE_SIZE[1] / 2], [0, 0, 1]])
    return intrinsic @ np.column_stack((rotation[:, 0], rotation[:, 1], -rotation @ camera))


def shifted(homography: np.ndarray, shift_px: np.ndarray) -> np.ndarray:
    return np.array([[1, 0, shift_px[0]], [0, 1, shift_px[1]], [0, 0, 1]]) @ homography


def render(homography: np.ndarray, shift_px: np.ndarray, boxes: list, seed: int = 5) -> np.ndarray:
    """Smooth floor texture moved with the camera, painted stripes, and noise inside each person box."""
    rng = np.random.default_rng(seed)
    width, height = NATIVE_SIZE
    pad = 40
    noise = rng.integers(0, 255, (height + 2 * pad, width + 2 * pad), dtype=np.uint8)
    # Lower contrast or finer grain leaves ECC too little texture inside the court to converge.
    floor = cv2.normalize(cv2.GaussianBlur(noise, (0, 0), 6), None, 30, 170, cv2.NORM_MINMAX)
    shift_x, shift_y = np.rint(shift_px).astype(int)
    grey = floor[pad - shift_y:pad - shift_y + height, pad - shift_x:pad - shift_x + width].copy()
    for segment in CENTRE_SEGMENTS_M:
        direction = (segment[1] - segment[0]) / np.linalg.norm(segment[1] - segment[0])
        half_width = np.array([-direction[1], direction[0]]) * STRIPE_WIDTH_M / 2
        stripe = np.array([segment[0] - half_width, segment[1] - half_width,
                           segment[1] + half_width, segment[0] + half_width])
        # 4 fractional bits: OpenCV draws at 1/16 px.
        cv2.fillConvexPoly(grey, np.rint(composition.carry(stripe, homography) * 16).astype(np.int32), 235,
                           cv2.LINE_AA, 4)
    for x1, y1, x2, y2 in np.asarray(boxes, dtype=int):
        grey[y1:y2, x1:x2] = rng.integers(60, 200, (y2 - y1, x2 - x1))
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)


def inside(point: np.ndarray, boxes: list) -> bool:
    return any(x1 <= point[0] <= x2 and y1 <= point[1] <= y2 for x1, y1, x2, y2 in boxes)


def fragments(homography: np.ndarray, hidden_by: list, offsets_px: dict[int, float] | None = None) -> list:
    """Native-px fragments along each stripe centre, except where a box hides them.

    :param offsets_px: native px to move a stripe's fragments sideways, by stripe index.
    """
    offsets_px = offsets_px or {}
    pieces = []
    for index, segment in enumerate(CENTRE_SEGMENTS_M):
        ends = composition.carry(segment, homography)
        direction = (ends[1] - ends[0]) / np.linalg.norm(ends[1] - ends[0])
        sideways = np.array([-direction[1], direction[0]]) * offsets_px.get(index, 0.0)
        for start, end in FRAGMENT_SPANS:
            piece = ends[0] + np.array([[start], [end]]) * (ends[1] - ends[0]) + sideways
            if not inside(piece.mean(axis=0), hidden_by):
                pieces.append(piece.ravel().tolist())
    return pieces


def frame_spec(role: str, shift_px: np.ndarray, boxes: list, corners: np.ndarray, **options: Any) -> dict:
    homography = shifted(camera_homography(), shift_px)
    image = options.get("image")
    if image is None:
        image = render(homography, shift_px, boxes)
    lines = fragments(homography, options.get("hidden_by", boxes), options.get("offsets_px"))
    return {"role": role, "image": image, "lines": lines, "boxes": boxes, "corners": corners}


def true_corners(shift_px: np.ndarray) -> np.ndarray:
    return composition.carry(CORNER_COURT_M, shifted(camera_homography(), shift_px))


def write_gz(path: Path, value: Any) -> None:
    with gzip.open(path, "wt") as stream:
        json.dump(value, stream)


def write_run(tmp_path: Path, live: LiveModules, specs: list[dict]) -> dict[str, Path]:
    """Write a saved run whose accepted frames' evidence is measured as run.py measured it."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    lines, people, rows = {}, {}, []
    for spec in specs:
        view_id = f"v_{spec['role']}"
        cv2.imwrite(str(frames_dir / f"{view_id}.png"), spec["image"])
        lines[view_id] = {"native_size": list(NATIVE_SIZE), "segments_native_px": spec["lines"]}
        people[view_id] = {"native_size": list(NATIVE_SIZE), "boxes_native_px": [list(box) for box in spec["boxes"]]}
        row = {"role": spec["role"], "frame_index": {"middle": 20, "first": 10, "last": 30}[spec["role"]],
               "view_id": view_id, "route": "full_search", "status": "court",
               "corners_native_px": np.asarray(spec["corners"]).tolist()}
        frame = compose.load_frame(live, row, lines, people, frames_dir, list(NATIVE_SIZE))
        row["evidence"] = paint_evidence(live, frame.context, np.asarray(spec["corners"]))
        row["paint_score"] = row["evidence"]["q_paint10_span_weighted"]
        rows.append(row)
    outcome = {"status": "court", "frames": rows, "chosen_position": rescore.paint_position(rows)}
    methods = {method: {"status": "detection_failed"} for method in METHODS}
    methods[compose.METHOD] = outcome
    scene = {"scene_id": "s", "status": "analysed", "methods": methods}
    results = {"schema": rescore.SOURCE_SCHEMA, "finished": True,
               "videos": [{"video_id": "v", "native_size": list(NATIVE_SIZE), "scenes": [scene], "later_scenes": []}]}
    paths = {"results": tmp_path / "results.json.gz", "lines": tmp_path / "lines.json.gz",
             "people": tmp_path / "people.json.gz", "frames": frames_dir}
    write_gz(paths["results"], results)
    write_gz(paths["lines"], {"schema": rescore.LINES_SCHEMA, "frames": lines, "recovery_seconds": 3.0})
    write_gz(paths["people"], {"schema": compose.PEOPLE_SCHEMA, "frames": people, "recovery_seconds": 2.0})
    return paths


def compose_saved(live: LiveModules, paths: dict[str, Path], tmp_path: Path) -> dict:
    results = compose.read_json(paths["results"])
    lines = compose.read_json(paths["lines"])["frames"]
    people = compose.read_json(paths["people"])["frames"]
    video = results["videos"][0]
    return compose.compose_scene(live, video, video["scenes"][0], lines, people, paths["frames"], tmp_path / "out")


def loaded(live: LiveModules, paths: dict[str, Path], role: str) -> composition.SearchedFrame:
    results = compose.read_json(paths["results"])
    rows = results["videos"][0]["scenes"][0]["methods"][compose.METHOD]["frames"]
    row = next(row for row in rows if row["role"] == role)
    return compose.load_frame(live, row, compose.read_json(paths["lines"])["frames"],
                              compose.read_json(paths["people"])["frames"], paths["frames"], list(NATIVE_SIZE))


@pytest.fixture(scope="module")
def live() -> LiveModules:
    return load_live_modules()


def test_view_warp_becomes_the_same_warp_in_working_pixels() -> None:
    warp_view = np.array([[1.02, 0.01, 12.0], [-0.02, 0.98, -7.0], [1e-5, -2e-5, 1.0]])
    working_size = (960, 720)
    point_working = np.array([[300.0, 500.0]])
    # By hand: working px to view px, through the warp, and back.
    view_per_working = np.asarray((960, 540)) / np.asarray(working_size)
    expected = composition.carry(point_working * view_per_working, warp_view) / view_per_working
    moved = composition.carry(point_working, composition.to_reference_working(warp_view, working_size))
    np.testing.assert_allclose(moved, expected, atol=1e-9)


def test_alignment_carries_frame_pixels_onto_the_reference(tmp_path: Path, live: LiveModules) -> None:
    specs = [frame_spec("middle", np.zeros(2), [], true_corners(np.zeros(2))),
             frame_spec("first", JITTER_PX, [], true_corners(JITTER_PX))]
    paths = write_run(tmp_path, live, specs)
    reference, frame = loaded(live, paths, "middle"), loaded(live, paths, "first")
    record, to_reference = composition.align(frame, reference)
    assert record["usable"] and not record["same_camera"]
    assert to_reference is not None
    # The first frame's content sits JITTER_PX further on, so the warp must take it back.
    moved_native = composition.carry(true_corners(JITTER_PX) / frame.native_per_working, to_reference)
    np.testing.assert_allclose(moved_native * reference.native_per_working, true_corners(np.zeros(2)), atol=0.5)


def view_pixels_inside(box_native: tuple) -> tuple[slice, slice]:
    """A native box's whole pixels in a VIEW_RESOLUTION image, as row and column slices."""
    view_per_native = np.asarray(court_views.VIEW_RESOLUTION) / np.asarray(NATIVE_SIZE)
    x1, y1, x2, y2 = np.asarray(box_native) * np.tile(view_per_native, 2)
    return slice(ceil(y1), floor(y2)), slice(ceil(x1), floor(x2))


@pytest.mark.parametrize("camera_shift", [np.zeros(2), NUDGE_PX], ids=["still_camera", "moved_camera"])
def test_moving_people_are_cut_from_each_image_in_its_own_pixels(tmp_path: Path, live: LiveModules,
                                                                 monkeypatch: pytest.MonkeyPatch,
                                                                 camera_shift: np.ndarray) -> None:
    specs = [frame_spec("middle", np.zeros(2), MIDDLE_PEOPLE, true_corners(np.zeros(2))),
             frame_spec("first", camera_shift, FIRST_PEOPLE, true_corners(camera_shift))]
    paths = write_run(tmp_path, live, specs)
    reference, frame = loaded(live, paths, "middle"), loaded(live, paths, "first")
    # The premise: with the players inside the mask, reuse's alignment rejects this pair.
    corners_refpx = true_corners(camera_shift) * np.asarray(HOMOGRAPHY_RESOLUTION) / np.asarray(NATIVE_SIZE)
    unmasked = court_views.measure_view_alignment(reuse.view_image(frame.native_frame),
                                                  reuse.view_image(reference.native_frame), corners_refpx)
    assert unmasked is None or unmasked.correlation < court_views.MIN_ALIGNMENT_CORRELATION

    masks = {}
    real_ecc = cv2.findTransformECCWithMask

    def recording_ecc(template, sample, template_mask, input_mask, *rest):
        masks.update(template=template_mask.copy(), input=input_mask.copy())
        return real_ecc(template, sample, template_mask, input_mask, *rest)

    monkeypatch.setattr(cv2, "findTransformECCWithMask", recording_ecc)
    record, to_reference = composition.align(frame, reference)

    # The template is this frame and the input is the reference. Each mask loses only its
    # own image's players, placed at view scale, which differs from working scale in y.
    for people, own, other in ((FIRST_PEOPLE, "template", "input"), (MIDDLE_PEOPLE, "input", "template")):
        for box in people:
            rows, columns = view_pixels_inside(box)
            assert not masks[own][rows, columns].any() and masks[other][rows, columns].any()
    assert record["usable"] and record["same_camera"] == (not camera_shift.any())
    assert 0 < record["mask_kept_fraction"] < 1
    assert to_reference is not None
    moved = composition.carry(true_corners(camera_shift) / frame.native_per_working, to_reference)
    np.testing.assert_allclose(moved * reference.native_per_working, true_corners(np.zeros(2)), atol=0.5)


def test_alignment_with_no_pixels_left_is_unmeasurable(tmp_path: Path, live: LiveModules) -> None:
    specs = [frame_spec(role, np.zeros(2), [], true_corners(np.zeros(2))) for role in ("middle", "first")]
    paths = write_run(tmp_path, live, specs)
    reference, frame = loaded(live, paths, "middle"), loaded(live, paths, "first")
    assert composition.align(frame, reference)[0]["usable"]
    # One person box over the whole reference leaves no pixel valid in both masks.
    whole_image = np.array([[0.0, 0.0, *reference.context.size]])
    covered = dataclasses.replace(reference, context=dataclasses.replace(reference.context, mask_boxes=whole_image))
    record, to_reference = composition.align(frame, covered)
    assert record == {"mask_kept_fraction": 0.0, "usable": False, "skip_reason": "alignment_unmeasurable"}
    assert to_reference is None


def test_turned_court_is_reordered_and_names_the_same_painted_lines(tmp_path: Path, live: LiveModules) -> None:
    corners = true_corners(np.zeros(2))
    paths = write_run(tmp_path, live, [frame_spec("middle", np.zeros(2), [LEFT_BOX], corners)])
    frame = loaded(live, paths, "middle")
    turned = np.roll(corners, composition.HALF_TURN_ROLL, axis=0)
    working = corners / frame.native_per_working
    assert composition.half_turn_roll(turned / frame.native_per_working, working) == composition.HALF_TURN_ROLL
    assert composition.half_turn_roll(working, working) == 0
    as_saved = composition.measure(live, frame, corners)["markings"]
    after_turn = {marking["marking"]: marking for marking in composition.measure(live, frame, turned)["markings"]}
    # The box hides the left doubles line, so the turned court's right doubles shows the same drop.
    assert as_saved[0]["q_paint10"] < as_saved[4]["q_paint10"]
    for marking in as_saved:
        twin = after_turn[TURNED_MARKING.get(marking["marking"], marking["marking"])]
        assert twin["q_paint10"] == pytest.approx(marking["q_paint10"], abs=1e-6)


def test_donated_samples_are_observed_fragments_outside_person_boxes(tmp_path: Path, live: LiveModules) -> None:
    corners = true_corners(np.zeros(2))
    left_doubles = MARKINGS.index("left_doubles")
    # The left doubles fragments sit 1.25 native px (1 working px) off the stripe centre and
    # stay visible inside the box, so the box alone must drop their samples.
    spec = frame_spec("middle", np.zeros(2), [LEFT_BOX], corners, hidden_by=[], offsets_px={0: 1.25})
    frame = loaded(live, write_run(tmp_path, live, [spec]), "middle")
    evidence = composition.measure(live, frame, corners)
    used = composition.UsedFrame(frame, np.eye(3), corners, evidence)
    constraints, summary = composition.donated_constraints(used, [left_doubles])

    assert summary[left_doubles]["samples"] > 0 and summary[left_doubles]["occluded_samples"] > 0
    observed = frame.context.observations.samples.reshape(-1, 2)
    for point in constraints.points:
        assert np.isclose(observed, point, atol=1e-9).all(axis=1).any()
    box_working = np.asarray(LEFT_BOX) / np.tile(frame.native_per_working, 2)
    assert not any(inside(point, [box_working]) for point in constraints.points)
    stripe = composition.carry(CENTRE_SEGMENTS_M[0], composition.court_homography(frame, corners))
    np.testing.assert_allclose(distances_to_segments(constraints.points, stripe[None]), 1.0, atol=0.05)
    np.testing.assert_allclose(constraints.weights.sum(), summary[left_doubles]["weight"])


def score_row(label: str, mean: float | None) -> dict:
    role = None if label == compose.COMPOSITE else label.removeprefix("original_")
    return {"candidate": label, "role": role, "scores": [{}], "mean_combined_score": mean}


def test_exact_ties_keep_a_saved_court_in_role_order() -> None:
    evaluated = [score_row("original_last", 0.5), score_row("original_first", 0.5), score_row("composite", 0.5)]
    choice = compose.decide(evaluated, "original_last")
    assert choice == {"state": "compared", "common_frame_original": "original_first",
                      "composite_beats_originals": False, "final": "original_first"}
    evaluated[2]["mean_combined_score"] = 0.5 + 1e-12
    assert compose.decide(evaluated, "original_last")["final"] == "composite"


def test_a_saved_court_without_a_common_frame_mean_keeps_the_full_score_choice() -> None:
    evaluated = [score_row("original_middle", 0.4), score_row("original_last", None), score_row("composite", 0.9)]
    choice = compose.decide(evaluated, "original_last")
    assert choice == {"state": "original_scores_incomplete", "common_frame_original": None,
                      "composite_beats_originals": None, "final": "original_last"}


def test_failed_alignment_uses_full_score_instead_of_paint(tmp_path: Path, live: LiveModules,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    corners = true_corners(np.zeros(2)).tolist()
    frames = [{"role": role, "view_id": f"v_{role}", "corners_native_px": corners} for role in ("middle", "last")]
    outcome = {"frames": frames}
    methods = {method: {"status": "detection_failed"} for method in METHODS}
    methods[compose.METHOD] = outcome
    scene = {"scene_id": "s", "status": "analysed", "methods": methods}
    video = {"video_id": "v", "native_size": list(NATIVE_SIZE)}
    monkeypatch.setattr(rescore, "rescore_method", lambda *_: {
        "state": "rescored", "selected": {"paint": 0, "ranked": 1}, "ranked_order": [1, 0],
    })
    monkeypatch.setattr(compose, "prepare_frames", lambda *_: ([{"role": "last"}], [object()]))
    result = compose.compose_scene(live, video, scene, {}, {}, tmp_path, tmp_path / "out")
    assert result["state"] == "too_few_aligned_frames"
    assert result["choice"]["paint"] == "original_middle"
    assert result["choice"]["final"] == "original_last"
    assert result["choice"]["changed"] is True
    assert result["choice"]["final_view_id"] == "v_last"
    assert result["choice"]["final_corners_native_px"] == corners


def test_a_composite_without_a_common_frame_mean_cannot_win() -> None:
    evaluated = [score_row("original_middle", 0.4), score_row("original_last", 0.3), score_row("composite", None)]
    choice = compose.decide(evaluated, "original_last")
    assert choice["composite_beats_originals"] is None and choice["final"] == "original_middle"


@pytest.mark.parametrize("second", ["none", "unrelated_image"])
def test_too_few_aligned_frames_keep_the_saved_choice(tmp_path: Path, live: LiveModules, second: str) -> None:
    specs = [frame_spec("middle", np.zeros(2), [], true_corners(np.zeros(2)))]
    if second == "unrelated_image":
        blank = render(camera_homography(), np.zeros(2), [(0, 0, *NATIVE_SIZE)], seed=11)
        specs.append(frame_spec("first", JITTER_PX, [], true_corners(JITTER_PX), image=blank, hidden_by=[]))
    scene = compose_saved(live, write_run(tmp_path, live, specs), tmp_path)
    assert scene["state"] == "too_few_aligned_frames"
    assert scene["choice"]["final"] == scene["choice"]["paint"] and scene["choice"]["changed"] is False
    assert scene["frames"][0]["reproduction"] == {"max_score_difference": 0.0, "largest_at": None, "mismatches": []}
    if second == "unrelated_image":
        skipped = scene["frames"][1]
        assert skipped["half_turn_roll"] is None and not skipped["alignment"]["usable"]
        assert skipped["alignment"]["skip_reason"] in ("alignment_unmeasurable", "correlation_below_reuse_level")


def test_composite_takes_each_marking_from_the_frame_that_shows_it(tmp_path: Path, live: LiveModules,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    # Each saved court is 4 px off at one corner; the first frame's is also turned 180 degrees.
    middle_corners = true_corners(np.zeros(2)) + [[4.0, 0.0], [0, 0], [0, 0], [0, 0]]
    first_corners = true_corners(JITTER_PX) + [[0, 0], [0, 0], [-4.0, 0.0], [0, 0]]
    specs = [frame_spec("middle", np.zeros(2), [LEFT_BOX], middle_corners),
             frame_spec("first", JITTER_PX, [RIGHT_BOX], np.roll(first_corners, composition.HALF_TURN_ROLL, axis=0))]
    paths = write_run(tmp_path, live, specs)
    before = {name: path.read_bytes() for name, path in paths.items() if name != "frames"}
    output_dir = tmp_path / "composed"
    monkeypatch.setattr(sys, "argv", ["compose", "--results", str(paths["results"]), "--lines-cache",
                                      str(paths["lines"]), "--people-cache", str(paths["people"]),
                                      "--frames-dir", str(paths["frames"]), "--output-dir", str(output_dir)])
    assert compose.main() == 0
    assert {name: path.read_bytes() for name, path in paths.items() if name != "frames"} == before

    with gzip.open(output_dir / "results.json.gz", "rt") as stream:
        report = json.load(stream)
    scene = report["scenes"][0]
    assert scene["state"] == "composed"
    reference = scene["reference"]
    other = next(frame for frame in scene["frames"] if not frame["is_reference"])
    assert other["half_turn_roll"] == composition.HALF_TURN_ROLL
    # Each box hides a doubles line in one frame, so the other frame donates it. Marking
    # names follow the reference court, and the first frame's court is turned.
    donors = {marking["marking"]: marking["donor_role"] for marking in scene["markings"]}
    expected = {"left_doubles": "first", "right_doubles": "middle"}
    if reference == "first":
        expected = {"left_doubles": "middle", "right_doubles": "first"}
    assert {name: donors[name] for name in expected} == expected
    assert all(marking["samples"] > 0 for marking in scene["markings"])
    assert scene["fit"]["valid"], scene["fit"]["validity_reason"]
    truth = {"middle": true_corners(np.zeros(2)), "first": np.roll(true_corners(JITTER_PX), 2, axis=0)}[reference]
    # Stripe positions come from each saved court's own assignment. Against a court 4 px off,
    # some centre fragments read as stripe edges, which leaves up to half a stripe of error.
    np.testing.assert_allclose(scene["fit"]["corners_reference_native_px"], truth, atol=2.0)

    candidates = {row["candidate"]: row for row in scene["evaluation"]["candidates"]}
    assert set(candidates) == {"original_middle", "original_first", "composite"}
    for row in candidates.values():
        assert [score["frame_role"] for score in row["scores"]] == scene["evaluation"]["frames"]
        expected_mean = np.mean([score["combined_score"] for score in row["scores"]])
        assert row["mean_combined_score"] == pytest.approx(expected_mean)
    assert scene["choice"]["composite_beats_originals"] is True and scene["choice"]["final"] == "composite"
    assert scene["choice"]["changed"] is True
    assert sorted(scene["outlines"]) == sorted(f"s__{label}__on_{reference}.png" for label in candidates)
    assert all((output_dir / "outlines" / name).exists() for name in scene["outlines"])
    with gzip.open(output_dir / "comparison.csv.gz", "rt") as stream:
        row = next(csv.DictReader(stream))
    assert (row["state"], row["final"], row["half_turned_roles"]) == ("composed", "composite", other["role"])


def test_scene_record_does_not_change_the_saved_results(tmp_path: Path, live: LiveModules) -> None:
    specs = [frame_spec("middle", np.zeros(2), [LEFT_BOX], true_corners(np.zeros(2))),
             frame_spec("first", JITTER_PX, [RIGHT_BOX], true_corners(JITTER_PX))]
    paths = write_run(tmp_path, live, specs)
    results = compose.read_json(paths["results"])
    before = copy.deepcopy(results)
    video = results["videos"][0]
    compose.compose_scene(live, video, video["scenes"][0], compose.read_json(paths["lines"])["frames"],
                          compose.read_json(paths["people"])["frames"], paths["frames"], tmp_path / "out")
    assert results == before
