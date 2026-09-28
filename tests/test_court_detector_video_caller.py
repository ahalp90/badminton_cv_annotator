"""The video caller keeps frame, line, pose and scene coordinates together."""

import gzip
import json
from collections.abc import Sequence

import numpy as np
import pytest

from scratch.court_det_fix.court_detector import run_video
from scratch.court_det_fix.court_detector.detect import (
    CourtDetector,
    CourtResult,
    Switches,
)
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
    def __init__(self, switches: Switches | None = None) -> None:
        self.switches = switches or Switches()
        self.views: list[ViewInputs] = []
        self.people_sources = []

    def detect(self, view, people, frames) -> CourtResult:
        self.views.append(view)
        self.people_sources.append(people)
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


def test_optional_people_analyse_short_scene_without_pose_source() -> None:
    detector, lines = Detector(Switches(require_people=False)), Lines()
    rows = list(scene_courts(detector, Frames(), None, lines, [SceneInfo(0, 9)], video_id='clip'))  # type: ignore[arg-type]
    assert [row['status'] for row in rows] == ['no_court']
    assert lines.indices == [4]
    assert detector.people_sources == [None]
    assert detector.views[0].scene_frames == (0, 9)
    assert detector.views[0].person_boxes_px.shape == (0, 4)


def test_optional_people_uses_supplied_pose_source() -> None:
    detector = Detector(Switches(require_people=False))
    rows = list(scene_courts(detector, Frames(), People(), Lines(), [SceneInfo(0, 9)],  # type: ignore[arg-type]
                            video_id='clip'))
    assert rows[0]['status'] == 'no_court'
    np.testing.assert_array_equal(detector.views[0].person_boxes_px, [[4, 2, 30, 40]])


@pytest.mark.parametrize('with_people', [False, True])
def test_empty_line_extract_returns_no_court_and_continues_video(with_people: bool) -> None:
    class EmptyThenSingleLine(Lines):
        def segments(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
            segments = super().segments(frame, frame_index)
            return segments[:0] if frame_index == 4 else segments

    detector = CourtDetector(Switches(require_people=False))
    lines = EmptyThenSingleLine()
    rows = list(scene_courts(detector, Frames(), People() if with_people else None, lines,
                            [SceneInfo(0, 9), SceneInfo(10, 19)], video_id='clip'))
    assert lines.indices == [4, 14]
    assert [row['status'] for row in rows] == ['no_court', 'no_court']
    assert [row['no_court_reason'] for row in rows] == ['no_gated_court', 'no_gated_court']


def test_required_people_rejects_missing_source() -> None:
    with pytest.raises(ValueError, match='people source is required'):
        list(scene_courts(Detector(), Frames(), None, Lines(), [SceneInfo(0, 99)],  # type: ignore[arg-type]
                          video_id='clip'))


@pytest.mark.parametrize(('flag', 'expected_status', 'live_setup_count'), [
    ([], 'scene_too_short_for_feet', 1),
    (['--no-require-people'], 'no_court', 0),
])
def test_cli_defaults_to_live_people_but_optional_mode_skips_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path, flag: list[str], expected_status: str, live_setup_count: int,
) -> None:
    class VideoFileFrames(Frames):
        frame_count = 10

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    detectors = []
    live_setups = []

    def make_detector(switches):
        detector = Detector(switches)
        detectors.append(detector)
        return detector

    def make_live_people(frames, device):
        live_setups.append((frames, device))
        return People()

    output = tmp_path / 'result.json.gz'
    monkeypatch.setattr(run_video, 'VideoFrames', lambda path: VideoFileFrames())
    monkeypatch.setattr(run_video, 'SavedLines', lambda records: Lines())
    monkeypatch.setattr(run_video, 'read_json', lambda path: {})
    monkeypatch.setattr(run_video, 'RtmlibPeople', make_live_people)
    monkeypatch.setattr(run_video, 'CourtDetector', make_detector)
    monkeypatch.setattr(run_video.os, 'sched_setaffinity', lambda *_: None)
    monkeypatch.setattr('sys.argv', ['run_video', '--video', 'input.mp4', '--output', str(output),
                                    '--saved-lines', 'lines.json.gz', *flag])

    assert run_video.main() == 0
    with gzip.open(output, 'rt') as stream:
        result = json.load(stream)
    assert len(live_setups) == live_setup_count
    assert result['require_people'] is (not flag)
    assert result['scenes'][0]['status'] == expected_status
    assert detectors[0].switches.require_people is (not flag)
    if flag:
        assert detectors[0].people_sources == [None]
        assert detectors[0].views[0].person_boxes_px.shape == (0, 4)


@pytest.mark.parametrize('scenes', [[], [SceneInfo(1, 9)], [SceneInfo(0, 8)],
                                    [SceneInfo(0, 4), SceneInfo(6, 9)], [SceneInfo(0, 5), SceneInfo(5, 9)]])
def test_incomplete_or_overlapping_external_scenes_fail(scenes: list[SceneInfo]) -> None:
    with pytest.raises(ValueError):
        validate_scenes(scenes, 10)


def test_valid_scene_partition() -> None:
    validate_scenes([SceneInfo(0, 4), SceneInfo(5, 9)], 10)


def test_reuse_keeps_searched_templates_and_orders_by_optional_histograms() -> None:
    class ReusingDetector:
        def __init__(self) -> None:
            self.switches = Switches()
            self.attempts: list[list[str]] = []

        def detect(self, view, people, frames, *, known_courts=()) -> CourtResult:
            self.attempts.append([known.view_id for known in known_courts])
            source = known_courts[0].view_id if len(self.attempts) == 3 else None
            corners = np.array([[10., 10.], [50., 10.], [50., 40.], [10., 40.]])
            return CourtResult(view.view_id, corners, None, 'reuse' if source else 'searched', None, .9, source)

    detector = ReusingDetector()
    scenes = [SceneInfo(0, 79, np.array([1., 0.])), SceneInfo(80, 159, np.array([0., 1.])),
              SceneInfo(160, 239, np.array([.9, .1]))]
    rows = list(scene_courts(detector, Frames(), People(), Lines(), scenes,  # type: ignore[arg-type]
                            video_id='clip', reuse_courts=True))
    first, second = rows[0]['view_id'], rows[1]['view_id']
    assert detector.attempts == [[], [first], [first, second]]
    assert rows[2]['reused_from'] == first


def test_reused_court_does_not_become_a_template() -> None:
    class ReusingDetector:
        def __init__(self) -> None:
            self.switches = Switches()
            self.attempts: list[list[str]] = []

        def detect(self, view, people, frames, *, known_courts=()) -> CourtResult:
            self.attempts.append([known.view_id for known in known_courts])
            source = known_courts[0].view_id if known_courts else None
            return CourtResult(view.view_id, np.zeros((4, 2)), None, 'court', None, .9, source)

    detector = ReusingDetector()
    scenes = [SceneInfo(0, 79), SceneInfo(80, 159), SceneInfo(160, 239)]
    rows = list(scene_courts(detector, Frames(), People(), Lines(), scenes,  # type: ignore[arg-type]
                            video_id='clip', reuse_courts=True))
    first = rows[0]['view_id']
    assert detector.attempts == [[], [first], [first]]


@pytest.mark.parametrize('scene_length', [10, 80])
def test_reuse_shares_median_image_and_keeps_live_anchor_inputs(scene_length: int) -> None:
    class MovingFrames(Frames):
        def __init__(self) -> None:
            self.requested = []

        def read(self, indices: Sequence[int]) -> list[np.ndarray]:
            self.requested.extend(indices)
            decoded = []
            for index in indices:
                frame = np.full((48, 64, 3), 40, dtype=np.uint8)
                if index % scene_length == (scene_length - 1) // 2:
                    frame[:, 20:40] = 240
                decoded.append(frame)
            return decoded

    class AnchorLines:
        def segments(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
            assert np.all(frame[:, 20:40] == 240)
            return np.empty((0, 4))

    class MedianDetector:
        switches = Switches(require_people=False)

        def __init__(self) -> None:
            self.images = []

        def detect(self, view, people, frames, *, known_courts=()) -> CourtResult:
            assert np.all(view.frame[:, 20:40] == 240)
            assert np.all(view.alignment_image == 40)
            assert view.alignment_image.shape == (540, 960)
            assert view.alignment_image.dtype == np.uint8
            assert not view.alignment_image.flags.writeable
            if known_courts:
                assert known_courts[0].image is self.images[0]
            self.images.append(view.alignment_image)
            return CourtResult(view.view_id, np.zeros((4, 2)), None, 'court', None, .9)

    frames, detector = MovingFrames(), MedianDetector()
    scenes = [SceneInfo(0, scene_length - 1), SceneInfo(scene_length, 2 * scene_length - 1)]
    rows = list(scene_courts(detector, frames, None, AnchorLines(), scenes,  # type: ignore[arg-type]
                            video_id='clip', reuse_courts=True))
    assert [row['status'] for row in rows] == ['court', 'court']
    assert len(frames.requested) == 6
    for scene, requested in zip(scenes, (frames.requested[:3], frames.requested[3:]), strict=True):
        assert len(set(requested)) == 3
        assert all(scene.first_frame <= index <= scene.last_frame for index in requested)
