"""Reusing a court when the same camera view returns: alignment, refit and checks.

The real views are two broadcast scenes from one camera, shuttleset_03 scenes 16 and 17
(960x540 frames), and one amateur frame from another camera, gxBQ frame 5. Their courts
are the detector's saved results from the 26 September upright-camera check, not hand
marks.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np
import pytest

from annotator import court_views
from court_detector import measurements, reuse, stripe_refit
from court_detector.detect import (
    LiveModules,
    freeze_arrays,
    load_live_modules,
)
from scratch.court_det_fix.court_detector import frozen_cases

REPO = Path(__file__).resolve().parents[1]
COURT_ROOT = REPO / "scratch/court_det_fix"
SAVED_COURTS = COURT_ROOT / "court_detector/check_20260926_upright/upright/results"
EARLIER_SCENE = "shuttleset_03_scene_0016"
LATER_SCENE = "shuttleset_03_scene_0017"
OTHER_CAMERA_FRAME = COURT_ROOT / "frozen_views/frames/gx/images/gxBQ_window_00_frame_00000005.png"
UPRIGHT_LIMIT_DEG = 45.0  # detect.MAX_HORIZON_TILT_DEG
# (4, 2) native px: a synthetic court inside a 1920x1080 frame, in CORNER_COURT_M order.
SYNTHETIC_CORNERS = np.array([[640.0, 300.0], [1280.0, 300.0], [1560.0, 960.0], [360.0, 960.0]])


def saved_corners(case_id: str) -> np.ndarray:
    return np.asarray(json.loads((SAVED_COURTS / f"{case_id}.json").read_text())["corners_native_px"])


def textured_frame() -> np.ndarray:
    """A 1920x1080 BGR frame with smooth texture and a painted court outline."""
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 255, (1080, 1920), dtype=np.uint8)
    grey = cv2.GaussianBlur(noise, (0, 0), 6)
    grey = cv2.normalize(grey, None, 30, 200, cv2.NORM_MINMAX)
    cv2.polylines(grey, [np.rint(SYNTHETIC_CORNERS).astype(np.int32)], True, 250, 6)
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)


def court_homography(corners_working: np.ndarray) -> np.ndarray:
    return cv2.getPerspectiveTransform(reuse.CORNER_COURT_M.astype(np.float32),
                                       corners_working.astype(np.float32)).astype(float)


class RealViews(NamedTuple):
    live: LiveModules
    later: Any  # measurements.ViewContext for LATER_SCENE, frozen
    later_frame: np.ndarray  # (540, 960, 3) BGR
    earlier_frame: np.ndarray
    earlier_paint: float  # the earlier scene's saved court's final paint support
    later_paint: float


@pytest.fixture(scope="module")
def views() -> RealViews:
    live = load_live_modules()
    earlier = frozen_cases.prepare_view(COURT_ROOT, EARLIER_SCENE)
    later = frozen_cases.prepare_view(COURT_ROOT, LATER_SCENE)
    freeze_arrays(earlier)
    freeze_arrays(later)
    paints = []
    with live.prepared_measurements(live.verifier):
        for context, case_id in ((earlier, EARLIER_SCENE), (later, LATER_SCENE)):
            scale = np.asarray(context.native_size) / np.asarray(context.size)
            described = stripe_refit.describe(saved_corners(case_id) / scale, context, live.verifier, live.runtime,
                                              live.scoring.view_line_maps(context))
            paints.append(described["paint_score"])
    earlier_frame = cv2.imread(str(COURT_ROOT / earlier.frame_relative_path))
    later_frame = cv2.imread(str(COURT_ROOT / later.frame_relative_path))
    return RealViews(live, later, later_frame, earlier_frame, *paints)


@pytest.mark.parametrize("sample_size", [(1920, 1080), (1280, 720)])
def test_warp_carries_corners_from_the_known_view_into_the_new_views_native_pixels(sample_size) -> None:
    frame = textured_frame()
    shift_native = np.array([8.0, -4.0])
    translation = np.float32([[1, 0, shift_native[0]], [0, 1, shift_native[1]]])
    moved_frame = cv2.warpAffine(frame, translation, (1920, 1080))
    sample = cv2.resize(moved_frame, sample_size, interpolation=cv2.INTER_AREA)
    known = reuse.make_known_court("synthetic", frame, SYNTHETIC_CORNERS, 0.5)

    alignment = reuse.align_known_court(known, sample)

    assert alignment is not None
    # The warp maps the known view to the new one: content moved right and up.
    view_per_native = np.asarray(court_views.VIEW_RESOLUTION) / np.asarray((1920, 1080))
    np.testing.assert_allclose(alignment.warp[:2, 2], shift_native * view_per_native, atol=0.1)
    expected = (SYNTHETIC_CORNERS + shift_native) * np.asarray(sample_size) / np.asarray((1920, 1080))
    np.testing.assert_allclose(reuse.moved_corners_native(alignment, sample_size), expected, atol=0.3)
    # Eight native pixels is more than the same-camera test allows.
    assert not alignment.matches


def test_extracted_alignment_keeps_the_same_camera_result() -> None:
    frame = textured_frame()
    image = reuse.view_image(frame)
    corners_refpx = SYNTHETIC_CORNERS * np.asarray((1280, 720)) / np.asarray((1920, 1080))
    same = court_views.measure_view_alignment(image, image, corners_refpx)
    assert same is not None and same.matches
    np.testing.assert_allclose(same.moved_corners_refpx, corners_refpx, atol=1e-3)
    view = court_views.CourtView(np.zeros((1, 16, 16), dtype=bool), image)
    assert court_views._view_alignment(view, view, corners_refpx) is True

    blank = court_views.CourtView(view.hashes, np.zeros_like(image))
    assert court_views.measure_view_alignment(blank.image, image, corners_refpx) is None
    assert court_views._view_alignment(blank, view, corners_refpx) is False


def test_supplied_image_aligns_without_using_the_raw_frame() -> None:
    frame = textured_frame()
    image = reuse.view_image(frame)
    blank_frame = np.zeros_like(frame)
    known = reuse.make_known_court('median', blank_frame, SYNTHETIC_CORNERS, .5, alignment_image=image)
    assert known.image is image
    assert not image.flags.writeable
    alignment = reuse.align_known_court(known, blank_frame, image)
    assert alignment is not None and alignment.matches
    assert reuse.align_known_court(known, blank_frame) is None


def test_known_court_needs_positive_paint_and_is_read_only() -> None:
    frame = textured_frame()
    with pytest.raises(ValueError, match="positive paint"):
        reuse.make_known_court("synthetic", frame, SYNTHETIC_CORNERS, 0.0)
    known = reuse.make_known_court("synthetic", frame, SYNTHETIC_CORNERS, 0.5)
    assert known.native_size == (1920, 1080)
    assert known.image.shape == court_views.VIEW_RESOLUTION[::-1]
    assert not known.corners_native_px.flags.writeable and not known.image.flags.writeable


def test_equal_pixel_movement_counts_for_more_metres_at_the_far_end() -> None:
    working = saved_corners("gxBQ_window_00_frame_5") / 2  # 1920x1080 frame, 960x540 working image
    before = court_homography(working)
    far, near = working.copy(), working.copy()
    far[:2, 1] += 1.0  # CORNER_COURT_M's first two corners lie on the far baseline
    near[2:, 1] += 1.0

    far_shift, far_point = reuse.floor_shift_m(before, court_homography(far))
    near_shift, near_point = reuse.floor_shift_m(before, court_homography(near))

    assert reuse.floor_shift_m(before, before)[0] < 1e-9
    assert far_point[1] == 0.0 and near_point[1] == reuse.COURT_LENGTH_M
    # One working pixel downwards moves this low amateur camera's far baseline about half
    # a metre and its near baseline under a tenth of that.
    assert far_shift > 5 * near_shift
    assert far_shift > reuse.MAX_REFIT_SHIFT_M


def test_upright_filter_rejects_sideways_and_upside_down_courts() -> None:
    size = (960, 540)
    corners = saved_corners(LATER_SCENE)  # native 960x540, so already working pixels
    centre = np.asarray(size) / 2
    results = {}
    for degrees in (0, 90, 180):
        rotation = cv2.getRotationMatrix2D(tuple(centre), degrees, 1.0)
        rotated = cv2.transform(corners[None], rotation)[0]
        results[degrees] = reuse.upright_court(court_homography(rotated), rotated, size, UPRIGHT_LIMIT_DEG)
    assert results == {0: True, 90: False, 180: False}


def test_returning_camera_reuses_a_court_refitted_to_the_new_stripes(views: RealViews) -> None:
    known = reuse.make_known_court(EARLIER_SCENE, views.earlier_frame, saved_corners(EARLIER_SCENE),
                                   views.earlier_paint)

    court, record = reuse.try_reuse(known, views.later, views.later_frame, views.live,
                                    max_horizon_tilt_deg=UPRIGHT_LIMIT_DEG)

    assert court is not None, record
    assert record["rejection"] is None and record["alignment"]["shift_refpx"] < 1
    assert court.paint_ratio >= reuse.MIN_PAINT_RATIO and court.max_floor_shift_m <= reuse.MAX_REFIT_SHIFT_M
    # The later scene's own full search found a court within about a pixel of this one.
    np.testing.assert_allclose(court.corners_native_px, saved_corners(LATER_SCENE), atol=1.5)


def test_nearby_start_refits_back_onto_the_stripes(views: RealViews) -> None:
    offset_corners = saved_corners(LATER_SCENE) + [2.0, 0.0]
    known = reuse.make_known_court(LATER_SCENE, views.later_frame, offset_corners, views.later_paint)

    court, record = reuse.try_reuse(known, views.later, views.later_frame, views.live,
                                    max_horizon_tilt_deg=UPRIGHT_LIMIT_DEG)

    assert court is not None, record
    assert record["max_corner_shift_working_px"] > 1.5
    np.testing.assert_allclose(court.corners_native_px, saved_corners(LATER_SCENE), atol=1.0)


def test_another_cameras_court_falls_back_to_the_search(views: RealViews, monkeypatch) -> None:
    other_frame = cv2.imread(str(OTHER_CAMERA_FRAME))
    other_corners = saved_corners("gxBQ_window_00_frame_5")
    known = reuse.make_known_court("gxBQ_window_00_frame_5", other_frame, other_corners, 0.44)

    court, record = reuse.try_reuse(known, views.later, views.later_frame, views.live,
                                    max_horizon_tilt_deg=UPRIGHT_LIMIT_DEG)
    assert court is None and record["rejection"].startswith("alignment_")

    # A cutaway must not gain a court even if the image alignment wrongly matched.
    refpx = other_corners * np.asarray((1280, 720)) / np.asarray(known.native_size)
    forced = court_views.ViewAlignment(1.0, np.eye(3, dtype=np.float32), refpx, 0.0)
    monkeypatch.setattr(reuse, "align_known_court", lambda *_args: forced)
    court, record = reuse.try_reuse(known, views.later, views.later_frame, views.live,
                                    max_horizon_tilt_deg=UPRIGHT_LIMIT_DEG)
    assert court is None and not record["rejection"].startswith("alignment_")


def test_new_views_feet_outside_the_court_block_reuse(views: RealViews) -> None:
    source = views.later.source
    # Every standing foot moves to the frame's top-left corner, off the court.
    corner_feet = [[None if foot is None else [2.0, 2.0] for foot in frame] for frame in source["all_feet_px"]]
    context = measurements.view_context(LATER_SCENE, {**source, "all_feet_px": corner_feet},
                                        views.later.provenance, views.later_frame, views.later.frame_relative_path)
    known = reuse.make_known_court(LATER_SCENE, views.later_frame, saved_corners(LATER_SCENE), views.later_paint)

    court, record = reuse.try_reuse(known, context, views.later_frame, views.live,
                                    max_horizon_tilt_deg=UPRIGHT_LIMIT_DEG)

    assert court is None and record["rejection"] == "players_not_on_court"
    assert record["gates"]["player_fractions"] == [0.0, 0.0]


@pytest.mark.parametrize("require_people", [True, False])
def test_reuse_without_people_keeps_missing_measurements_explicit(views: RealViews, require_people: bool) -> None:
    source = {**views.later.source, "all_feet_px": [], "bbox_px": []}
    context = measurements.view_context(LATER_SCENE, source, views.later.provenance,
                                        views.later_frame, views.later.frame_relative_path)
    known = reuse.make_known_court(LATER_SCENE, views.later_frame, saved_corners(LATER_SCENE), views.later_paint)
    court, record = reuse.try_reuse(known, context, views.later_frame, views.live,
                                    max_horizon_tilt_deg=UPRIGHT_LIMIT_DEG, require_people=require_people)
    assert (court is None) is require_people, record
    assert record["rejection"] == ("players_not_on_court" if require_people else None)
    assert record["gates"]["player_fractions"] == [None, None]
    assert record["historical"] == {"historical_fullcourt": False, "historical_camera": True}


def test_mirrored_court_fails_the_geometry_check(views: RealViews) -> None:
    left_right_swapped = saved_corners(LATER_SCENE)[[1, 0, 3, 2]]
    known = reuse.make_known_court(LATER_SCENE, views.later_frame, left_right_swapped, views.later_paint)

    court, record = reuse.try_reuse(known, views.later, views.later_frame, views.live,
                                    max_horizon_tilt_deg=UPRIGHT_LIMIT_DEG)

    assert court is None and record["rejection"].startswith("moved_court_")


def test_camera_limit_applies_to_the_refitted_court(views: RealViews) -> None:
    known = reuse.make_known_court(LATER_SCENE, views.later_frame, saved_corners(LATER_SCENE), views.later_paint)

    # This court's horizon tilts about 0.4 degrees, so a zero limit rejects it.
    level_only, record = reuse.try_reuse(known, views.later, views.later_frame, views.live, max_horizon_tilt_deg=0.0)
    any_roll, _ = reuse.try_reuse(known, views.later, views.later_frame, views.live, max_horizon_tilt_deg=None)

    assert level_only is None and record["rejection"] == "camera_not_upright"
    assert any_roll is not None


def test_reuse_runs_without_scenedetect_or_torch() -> None:
    code = """
import sys
sys.modules["scenedetect"] = None
sys.modules["torch"] = None
import numpy as np
from court_detector import reuse
from tests.test_court_detector_reuse import SYNTHETIC_CORNERS, textured_frame
known = reuse.make_known_court("synthetic", textured_frame(), SYNTHETIC_CORNERS, 0.5)
alignment = reuse.align_known_court(known, textured_frame())
assert alignment is not None and alignment.matches
"""
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join((str(REPO), str(REPO / "src")))}
    completed = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=environment, capture_output=True,
                               text=True, timeout=300, check=False)
    assert completed.returncode == 0, completed.stderr
