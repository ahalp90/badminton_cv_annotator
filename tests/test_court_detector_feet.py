"""The court detector's feet rules agree with the fresh-feet evidence scripts and BST-X."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from experiments.annotator.independent_court.case_provenance import (
    CaseProvenance,
    ImageKind,
)
from scratch.court_det_fix.court_detector import feet
from scratch.court_det_fix.court_detector.inputs import PersonSample, ViewInputs

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
                feet.window_frames(anchor, fps, 0, frame_count - 1)
            continue
        assert feet.window_frames(anchor, fps, 0, frame_count - 1) == expected


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
    view = ViewInputs("view", np.zeros((36, 64, 3), dtype=np.uint8), 20, (0, 100), np.zeros((0, 4)),
                      np.zeros((0, 4)), CaseProvenance("view", ImageKind.SOURCE_FRAME, (20,), 20))
    with pytest.raises(ValueError, match="people source returned frames"):
        feet.window_feet(view, ShiftedPeople(), BlankFrames(), enforce_scene_consistency=True)
