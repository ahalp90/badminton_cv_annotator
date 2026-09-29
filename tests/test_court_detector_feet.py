"""Court foot sampling, pose classification and restoration of brief crouches."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from court_detector import feet
from court_detector.image_sources import CaseProvenance, ImageKind
from court_detector.inputs import PersonSample, ViewInputs

REPO = Path(__file__).resolve().parents[1]
FRESH_FEET = REPO / "scratch/court_det_fix/court_detector_optimisation_handover/claude_evidence/fresh_feet"


def evidence_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"fresh_feet_{name}", FRESH_FEET / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("fps", [25.0, 29.97002997002997, 30.0, 50.0, 59.94005994005994, 60.0000826168432])
@pytest.mark.parametrize("frame_count", [91, 230, 28142])
def test_window_matches_the_evidence_script(fps: float, frame_count: int) -> None:
    sample_frames = evidence_script("extract_window_people").sample_frames
    for anchor in (0, 1, 7, 15, 45, 150, frame_count - 46, frame_count - 10, frame_count - 1):
        try:
            expected = sample_frames(anchor, fps, frame_count)
        except ValueError:
            with pytest.raises(ValueError):
                feet.window_frames(anchor, fps, 0, frame_count)
            continue
        assert feet.window_frames(anchor, fps, 0, frame_count) == expected


# Expected frames come from the detector before scene ends became exclusive.
@pytest.mark.parametrize(("fps", "start_frame", "end_frame", "anchor", "expected"), [
    pytest.param(30.0, 7, 98, 52, list(range(7, 98, 3)), id="exactly-fits-91-frames"),
    pytest.param(30.0, 7, 100, 55, list(range(7, 98, 3)), id="spare-frames-after-window"),
    pytest.param(25.0, 7, 85, 52, [7, 10, 12, 14, 17, 20, 22, 24, 27, 30, 32, 34, 37, 40, 42, 44, 47, 50, 52, 54,
                                   57, 60, 62, 64, 67, 70, 72, 74, 77, 80, 82], id="shifted-back-from-end-25fps"),
    pytest.param(59.94005994005994, 0, 181, 0, list(range(0, 181, 6)), id="anchored-at-frame-0-fractional-fps"),
    pytest.param(29.97002997002997, 0, 28142, 28141, list(range(28051, 28142, 3)), id="anchored-at-final-frame"),
    pytest.param(60.0000826168432, 7, 189, 7, list(range(7, 188, 6)), id="fractional-step-fits-with-spare-frame"),
])
def test_window_keeps_reference_samples_at_scene_edges(
    fps: float, start_frame: int, end_frame: int, anchor: int, expected: list[int],
) -> None:
    assert feet.window_frames(anchor, fps, start_frame, end_frame) == expected


@pytest.mark.parametrize(("fps", "start_frame", "end_frame", "anchor"), [
    pytest.param(30.0, 7, 97, 52, id="one-frame-short-of-91"),
    pytest.param(59.94005994005994, 0, 180, 0, id="one-frame-short-at-frame-0"),
    # Rounding would put the 31st sample on frame 187, but the unrounded span ends just past it.
    pytest.param(60.0000826168432, 7, 188, 7, id="unrounded-span-ends-past-last-frame"),
    pytest.param(25.0, 7, 8, 7, id="one-frame-scene"),
])
def test_window_rejects_scenes_that_cannot_hold_it(fps: float, start_frame: int, end_frame: int, anchor: int) -> None:
    with pytest.raises(ValueError, match="does not fit scene"):
        feet.window_frames(anchor, fps, start_frame, end_frame)


def test_same_shot_run_matches_the_evidence_script() -> None:
    same_shot_samples = evidence_script("build_feet_variants").same_shot_samples
    rows = [json.loads(line) for line in (FRESH_FEET / "shot_check.jsonl").read_text().splitlines()]
    for row in rows:
        for anchor_position in range(len(row["differences"])):
            expected = same_shot_samples(row["differences"], anchor_position)
            assert feet.same_shot_run(row["differences"], anchor_position) == expected


def test_is_sitting_matches_bst_x() -> None:
    from bst_x.preparing_data.heuristics.base import SITTING_THRESHOLD, is_sitting

    random = np.random.default_rng(20260925)
    poses = random.uniform(0, 1000, size=(2000, 17, 2))
    degenerate = poses[:50].copy()
    degenerate[:, feet.SHOULDER_L] = degenerate[:, feet.HIP_L]
    degenerate[:, feet.SHOULDER_R] = degenerate[:, feet.HIP_R]
    # Knees 29, 30 and 31 px below hips on a 100 px torso: ratios -0.29, exactly -0.3, -0.31.
    at_threshold = np.stack([pose(120.0, 0.0) for _ in range(3)])
    at_threshold[:, [feet.KNEE_L, feet.KNEE_R], 1] = 200.0 + np.asarray([[29.0], [30.0], [31.0]])
    keypoints = np.concatenate((poses, degenerate, at_threshold))
    assert feet.SITTING_THRESHOLD == SITTING_THRESHOLD
    np.testing.assert_array_equal(feet.is_sitting(at_threshold), [True, False, False])
    np.testing.assert_array_equal(feet.is_sitting(keypoints), is_sitting(keypoints, SITTING_THRESHOLD))


def pose(foot_x: float, knee_offset_x: float) -> np.ndarray:
    """A COCO-17 pose with hips at y=200, shoulders 100 px above and knees offset from the hips."""
    keypoints = np.zeros((17, 2))
    keypoints[[feet.SHOULDER_L, feet.SHOULDER_R]] = (foot_x, 100.0)
    keypoints[[feet.HIP_L, feet.HIP_R]] = (foot_x, 200.0)
    keypoints[[feet.KNEE_L, feet.KNEE_R]] = (foot_x + knee_offset_x, 200.0 + (0.0 if knee_offset_x else 100.0))
    return keypoints


def test_standing_feet_drops_seated_people_and_marks_off_image_feet() -> None:
    boxes = np.asarray([[100.0, 0.0, 140.0, 400.0],  # standing, on image
                        [300.0, 0.0, 340.0, 400.0],  # seated
                        [900.0, 0.0, 940.0, 400.0]])  # standing, foot beyond the frame after scaling
    keypoints = np.stack((pose(120.0, 0.0), pose(320.0, 100.0), pose(920.0, 0.0)))
    samples = [PersonSample(0, boxes, keypoints), PersonSample(1, np.zeros((0, 4)), np.zeros((0, 17, 2)))]
    rows = feet.standing_feet(samples, np.asarray([0.5, 0.5]), (400, 300))
    assert rows == [[[60.0, 200.0], None], [None, None]]


@pytest.mark.parametrize(("sitting", "kept"), [
    ([False, True, False], [True, True, True]),
    ([False, True, True, False], [True, False, False, True]),
    ([True, True, False], [False, False, True]),
    ([True, True, True], [False, False, False]),
])
def test_standing_feet_restores_crouches_without_removing_standing_observations(
    sitting: list[bool], kept: list[bool],
) -> None:
    samples = []
    for frame_index, is_sitting in enumerate(sitting):
        foot_x = 120.0 + frame_index * 10
        boxes = np.asarray([[foot_x - 20, 0, foot_x + 20, 400], [700, 0, 740, 400]])
        keypoints = np.stack((pose(foot_x, 100.0 if is_sitting else 0.0), pose(720.0, 100.0)))
        # Detection order need not stay fixed between samples.
        if frame_index % 2:
            boxes, keypoints = boxes[::-1], keypoints[::-1]
        samples.append(PersonSample(frame_index, boxes, keypoints))
    rows = feet.standing_feet(samples, np.ones(2), (800, 500))
    expected = [[[120.0 + index * 10, 400.0], None] if keep else [None, None]
                for index, keep in enumerate(kept)]
    assert rows == expected


@pytest.mark.parametrize("missing_sample", [False, True])
def test_standing_track_does_not_restore_a_separate_seated_person(missing_sample: bool) -> None:
    standing_box = np.asarray([[100, 0, 140, 400]])
    samples = [PersonSample(index, standing_box, pose(120.0, 0.0)[None]) for index in range(3)]
    if missing_sample:
        samples.append(PersonSample(3, np.empty((0, 4)), np.empty((0, 17, 2))))
        seated_x = 120.0
    else:
        seated_x = 720.0  # More than one body height from the previous foot.
    samples.append(PersonSample(4, np.asarray([[seated_x - 20, 0, seated_x + 20, 400]]),
                                pose(seated_x, 100.0)[None]))
    rows = feet.standing_feet(samples, np.ones(2), (800, 500))
    assert rows[:3] == [[[120.0, 400.0], None]] * 3
    assert rows[3:] == [[None, None]] * (2 if missing_sample else 1)


def test_grey_differences_are_rounded_to_a_tenth() -> None:
    anchor = np.full((36, 64, 3), 100, dtype=np.uint8)
    brighter = np.full((36, 64, 3), 103, dtype=np.uint8)
    assert feet.grey_differences(anchor, [anchor, brighter]) == [0.0, 3.0]


class ShiftedPeople:
    def samples(self, frame_indices: list[int]) -> list[PersonSample]:
        return [PersonSample(frame_index + 1, np.zeros((0, 4)), np.zeros((0, 17, 2))) for frame_index in frame_indices]


class BlankFrames:
    fps = 10.0
    size = (64, 36)

    def read(self, frame_indices: list[int]) -> list[np.ndarray]:
        return [np.zeros((36, 64, 3), dtype=np.uint8) for _ in frame_indices]


def test_window_feet_rejects_people_from_other_frames() -> None:
    view = ViewInputs("view", np.zeros((36, 64, 3), dtype=np.uint8), 20, (0, 101), np.zeros((0, 4)),
                      np.zeros((0, 4)), CaseProvenance("view", ImageKind.SOURCE_FRAME, (20,), 20))
    with pytest.raises(ValueError, match="people source returned frames"):
        feet.window_feet(view, ShiftedPeople(), BlankFrames(), enforce_scene_consistency=True)


def test_absent_people_need_no_frame_window_or_fake_foot_measurements() -> None:
    view = ViewInputs("view", np.zeros((36, 64, 3), dtype=np.uint8), 0, (0, 1), np.zeros((0, 4)),
                      np.zeros((0, 4)), CaseProvenance("view", ImageKind.SOURCE_FRAME, (0,), 0))
    window = feet.window_feet(view, None, BlankFrames(), True, allow_short_window=True)
    assert window == feet.FeetWindow([], None, [], [])


def test_optional_short_window_uses_available_anchor_people() -> None:
    class EmptyPeople:
        def samples(self, frame_indices: list[int]) -> list[PersonSample]:
            return [PersonSample(index, np.empty((0, 4)), np.empty((0, 17, 2))) for index in frame_indices]

    view = ViewInputs("view", np.zeros((36, 64, 3), dtype=np.uint8), 0, (0, 1), np.zeros((0, 4)),
                      np.zeros((0, 4)), CaseProvenance("view", ImageKind.SOURCE_FRAME, (0,), 0))
    with pytest.raises(ValueError, match="does not fit scene"):
        feet.window_feet(view, EmptyPeople(), BlankFrames(), True)
    window = feet.window_feet(view, EmptyPeople(), BlankFrames(), True, allow_short_window=True)
    assert window.kept_frames == [0]
    assert window.all_feet_px == [[None, None]]


@pytest.mark.parametrize(("counts", "possible"), [
    ([], False), ([0], False), ([2, 0, 2], False), ([1, 2], True),
    ([1, 1, 2], False), ([2, 2, 1], True), ([2], True),
])
def test_required_people_counts_are_necessary_not_a_court_position_check(counts: list[int], possible: bool) -> None:
    rows = [[[1.0, 2.0]] * count + [None] * (2 - count) for count in counts]
    assert feet.can_satisfy_player_requirement(rows) is possible
