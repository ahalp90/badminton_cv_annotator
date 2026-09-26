"""The video caller keeps frame, line, pose and scene coordinates together."""

from collections.abc import Sequence

import numpy as np
import pytest

from scratch.court_det_fix.court_detector.detect import CourtResult
from scratch.court_det_fix.court_detector.inputs import PersonSample, ViewInputs
from scratch.court_det_fix.court_detector.run_video import scene_courts, validate_scenes
from scratch.court_det_fix.court_detector.scene_sources import SceneInfo


class Frames:
    fps = 25.0
    size = (64, 48)

    def read(self, indices: Sequence[int]) -> list[np.ndarray]:
        return [np.full((48, 64, 3), index, dtype=np.uint8) for index in indices]


class People:
    def samples(self, indices: Sequence[int]) -> list[PersonSample]:
        return [PersonSample(index, np.array([[index, 2, 30, 40]]), np.zeros((1, 17, 2))) for index in indices]


class Lines:
    def __init__(self) -> None:
        self.indices: list[int] = []

    def segments(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        assert np.all(frame == frame_index)
        self.indices.append(frame_index)
        return np.array([[1., 2., 30., 40.]])


class Detector:
    def __init__(self) -> None:
        self.views: list[ViewInputs] = []

    def detect(self, view, people, frames) -> CourtResult:
        self.views.append(view)
        return CourtResult(view.view_id, None, 'no_gated_court', None, {'feet': .1})


def test_scenes_keep_native_inputs_together_and_report_short_scenes() -> None:
    detector, lines = Detector(), Lines()
    scenes = [SceneInfo(0, 99), SceneInfo(100, 109), SceneInfo(110, 209)]
    rows = list(scene_courts(detector, Frames(), People(), lines, scenes, video_id='clip'))  # type: ignore[arg-type]
    assert [row['status'] for row in rows] == ['no_court', 'scene_too_short_for_feet', 'no_court']
    assert lines.indices == [49, 159]
    assert [view.scene_frames for view in detector.views] == [(0, 99), (110, 209)]
    for view in detector.views:
        assert view.person_boxes_px[0, 0] == view.frame_index
        np.testing.assert_array_equal(view.segments_px, [[1., 2., 30., 40.]])
    assert rows[0]['no_court_reason'] == 'no_gated_court'
    assert 'no_court_reason' not in rows[1]


@pytest.mark.parametrize('scenes', [[], [SceneInfo(1, 9)], [SceneInfo(0, 8)],
                                    [SceneInfo(0, 4), SceneInfo(6, 9)], [SceneInfo(0, 5), SceneInfo(5, 9)]])
def test_incomplete_or_overlapping_external_scenes_fail(scenes: list[SceneInfo]) -> None:
    with pytest.raises(ValueError):
        validate_scenes(scenes, 10)


def test_valid_scene_partition() -> None:
    validate_scenes([SceneInfo(0, 4), SceneInfo(5, 9)], 10)
