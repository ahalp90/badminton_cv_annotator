"""The video caller keeps frame, line, pose and scene coordinates together."""

import gzip
import json
import lzma
import sys
import types
from collections.abc import Sequence
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Self

import numpy as np
import pytest

from court_detector import feet, run_video
from court_detector.detect import (
    CourtDetector,
    CourtResult,
    PreparedView,
    SceneCourts,
    Switches,
)
from court_detector.inputs import PersonSample, ViewInputs, same_frame_provenance
from court_detector.run_video import scene_courts, validate_scenes
from court_detector.scene_sources import SceneInfo
from court_detector.video_inputs import PoseArrays
from court_detector.view_pool import CourtMode


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
    live = None

    def __init__(self, switches: Switches | None = None) -> None:
        self.switches = switches or Switches()
        self.views: list[ViewInputs] = []
        self.people_sources = []
        self.endpoint_views = []  # each scene's lazy endpoint builder, or None
        self.events: list[str] = []  # 'open' and 'close' for the worker pool's with block, and 'detect'

    def __enter__(self) -> Self:
        self.events.append('open')
        return self

    def __exit__(self, *_exception_info: object) -> None:
        self.events.append('close')

    def detect(self, view, people, frames, *, known_courts=(), endpoint_views=None) -> CourtResult:
        self.events.append('detect')
        self.views.append(view)
        self.people_sources.append(people)
        self.endpoint_views.append(endpoint_views)
        return CourtResult(view.view_id, None, 'no_gated_court', None, {'feet': .1})


def test_scenes_keep_native_inputs_together_and_report_short_scenes() -> None:
    detector, lines = Detector(), Lines()
    scenes = [SceneInfo(0, 100), SceneInfo(100, 110), SceneInfo(110, 210)]
    rows = list(scene_courts(detector, Frames(), People(), lines, scenes, video_id='clip'))  # type: ignore[arg-type]
    assert [row['status'] for row in rows] == ['no_court', 'scene_too_short_for_feet', 'no_court']
    assert [(row['start_frame'], row['end_frame'], row['frame_index']) for row in rows] == [
        (0, 100, 50), (100, 110, 105), (110, 210, 160)]
    # Each analysed scene offers its endpoints, but their lines wait until the detector asks.
    assert lines.indices == [50, 160]
    assert all(callable(endpoint_views) for endpoint_views in detector.endpoint_views)
    assert [view.scene_frames for view in detector.views] == [(0, 100), (110, 210)]
    for view in detector.views:
        assert view.person_boxes_px[0, 0] == view.frame_index
        np.testing.assert_array_equal(view.segments_px, [[1., 2., 30., 40.]])
    assert rows[0]['no_court_reason'] == 'no_gated_court'
    assert 'no_court_reason' not in rows[1]


def test_optional_people_analyse_short_scene_without_pose_source() -> None:
    detector, lines = Detector(Switches(require_people=False)), Lines()
    rows = list(scene_courts(detector, Frames(), None, lines, [SceneInfo(0, 10)],  # type: ignore[arg-type]
                            video_id='clip'))
    assert [row['status'] for row in rows] == ['no_court']
    assert lines.indices == [5]
    # Too short for the foot window, so there are no endpoint frames to compose from.
    assert detector.endpoint_views == [None]
    assert detector.people_sources == [None]
    assert detector.views[0].scene_frames == (0, 10)
    assert detector.views[0].person_boxes_px.shape == (0, 4)


def test_optional_people_uses_supplied_pose_source() -> None:
    detector = Detector(Switches(require_people=False))
    rows = list(scene_courts(detector, Frames(), People(), Lines(), [SceneInfo(0, 10)],  # type: ignore[arg-type]
                            video_id='clip'))
    assert rows[0]['status'] == 'no_court'
    np.testing.assert_array_equal(detector.views[0].person_boxes_px, [[5, 2, 30, 40]])


@pytest.mark.parametrize('with_people', [False, True])
def test_empty_line_extract_returns_no_court_and_continues_video(with_people: bool) -> None:
    class EmptyThenSingleLine(Lines):
        def segments(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
            segments = super().segments(frame, frame_index)
            return segments[:0] if frame_index == 5 else segments

    detector = CourtDetector(Switches(require_people=False))
    lines = EmptyThenSingleLine()
    rows = list(scene_courts(detector, Frames(), People() if with_people else None, lines,
                            [SceneInfo(0, 10), SceneInfo(10, 20)], video_id='clip'))
    assert lines.indices == [5, 15]
    assert [row['status'] for row in rows] == ['no_court', 'no_court']
    assert [row['no_court_reason'] for row in rows] == ['no_gated_court', 'no_gated_court']


def test_required_people_rejects_missing_source() -> None:
    with pytest.raises(ValueError, match='people source is required'):
        list(scene_courts(Detector(), Frames(), None, Lines(), [SceneInfo(0, 100)],  # type: ignore[arg-type]
                          video_id='clip'))


def write_json_gz(path: Path, value: object) -> Path:
    with gzip.open(path, 'wt') as stream:
        json.dump(value, stream)
    return path


def read_json_gz(path: Path) -> dict:
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def write_poses(directory: Path, frame_count: int) -> Path:
    """One standing person per frame, saved as the pose stage saves them."""
    directory.mkdir()
    arrays = {'bboxes': np.tile([10., 5., 30., 45.], (frame_count, 1, 1)),
              'kps': np.zeros((frame_count, 1, 17, 2)), 'ndet': np.ones(frame_count, dtype=np.int64)}
    for name, array in arrays.items():
        with lzma.open(directory / f'pose_{name}.npy.xz', 'wb', format=lzma.FORMAT_XZ) as stream:
            np.save(stream, array)
    return directory


class VideoFileFrames(Frames):
    """A 100-frame video file; long enough for the foot window."""

    frame_count = 100

    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        pass


class LiveTools:
    """Stand-ins for DeepLSD, RTMLib and the detector that count how often each loads."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, detector_type: type[Detector] = Detector) -> None:
        self.line_loads: list[tuple[Path, Path, str]] = []
        self.extractor_loads: list[str] = []
        self.people_setups: list[tuple[VideoFileFrames, object]] = []
        self.detectors: list[Detector] = []

        def load_lines(source: Path, weights: Path, *, device: str) -> Lines:
            self.line_loads.append((source, weights, device))
            return Lines()

        def make_detector(switches: Switches) -> Detector:
            self.detectors.append(detector_type(switches))
            return self.detectors[-1]

        def make_live_people(frames: VideoFileFrames, extractor: object) -> People:
            self.people_setups.append((frames, extractor))
            return People()

        loads = self.extractor_loads

        class Extractor:
            def __init__(self, device: str) -> None:
                loads.append(device)

        # The live branch imports the shared extractor, which needs RTMLib; stand in for it.
        rtmlib_pose = types.ModuleType('shared.rtmlib_pose')
        rtmlib_pose.RtmlibPoseExtractor = Extractor  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, 'shared.rtmlib_pose', rtmlib_pose)
        monkeypatch.setattr(run_video, 'VideoFrames', VideoFileFrames)
        monkeypatch.setattr(run_video, 'DeepLSDLines', load_lines)
        monkeypatch.setattr(run_video, 'SavedLines', lambda records: Lines())
        monkeypatch.setattr(run_video, 'RtmlibPeople', make_live_people)
        monkeypatch.setattr(run_video, 'CourtDetector', make_detector)
        monkeypatch.setattr(run_video.os, 'sched_setaffinity', lambda *_: None)


@pytest.mark.parametrize(('flag', 'people_source'), [
    pytest.param([], People, id='live-rtmlib'),
    pytest.param(['--no-require-people'], type(None), id='optional-without-people'),
    pytest.param(['--no-require-people', '--full'], type(None), id='optional-full-search'),
    pytest.param(['--people', 'POSES'], PoseArrays, id='saved-poses'),
])
def test_cli_uses_live_optional_or_saved_people(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, flag: list[str], people_source: type,
) -> None:
    tools = LiveTools(monkeypatch)
    if 'POSES' in flag:
        flag = ['--people', str(write_poses(tmp_path / 'poses', VideoFileFrames.frame_count))]
    lines = write_json_gz(tmp_path / 'lines.json.gz', {})
    output = tmp_path / 'result.json.gz'
    monkeypatch.setattr('sys.argv', ['run_video', '--video', 'input.mp4', '--output', str(output),
                                    '--saved-lines', str(lines), *flag])

    assert run_video.main() == 0
    result = read_json_gz(output)
    assert result['schema'] == run_video.VIDEO_RESULT_SCHEMA
    assert tools.extractor_loads == (['cuda'] if people_source is People else [])
    assert result['require_people'] is ('--no-require-people' not in flag)
    assert tools.detectors[0].switches.full_no_people_search is ('--full' in flag)
    assert result['saved_people'] is (people_source is PoseArrays)
    assert (result['court_mode'], result['reuse_courts']) == (CourtMode.VIDEO_ROBUST, False)
    # Without scene options the whole video is one scene.
    assert [(row['start_frame'], row['end_frame'], row['frame_index']) for row in result['scenes']] == [(0, 100, 50)]
    assert result['scenes'][0]['status'] == 'no_court'
    # One pool serves every scene.
    assert tools.detectors[0].events == ['open', 'detect', 'close']
    assert isinstance(tools.detectors[0].people_sources[0], people_source)


class CourtDetectorStandIn(Detector):
    """Finds a court in the first scene only and hands the video pool each scene's state."""

    live = None

    def detect(self, view, people, frames, *, known_courts=(), endpoint_views=None) -> CourtResult:
        self.events.append('detect')
        self.endpoint_views.append(endpoint_views)
        if len(self.endpoint_views) > 1:
            prepared = PreparedView(view, {}, view.frame, None)
            return CourtResult(view.view_id, None, 'refit_players_not_on_court', 'searched', None, prepared=prepared)
        corners = np.array([[50., 40.], [10., 40.], [10., 10.], [50., 10.]])
        scene = SceneCourts(None, view.frame, corners, corners)
        return CourtResult(view.view_id, corners, None, 'searched', None, .9, scene=scene)


@pytest.mark.parametrize('mode', [None, *CourtMode])
def test_court_mode_reaches_the_video_and_pooled_rows_print_only_when_final(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str], mode: CourtMode | None,
) -> None:
    tools = LiveTools(monkeypatch, CourtDetectorStandIn)
    # Four-second scenes fit the feet window, so the endpoint frames are on offer.
    monkeypatch.setattr(VideoFileFrames, 'frame_count', 200)
    pools = []

    class Pool:
        def __init__(self, live: object, switches: Switches, court_mode: CourtMode) -> None:
            self.court_mode = court_mode
            self.rows: list[dict] = []
            self.receivers: list[dict] = []
            pools.append(self)

        def add(self, row: dict, scene: SceneCourts) -> None:
            # Nothing prints before every scene has joined the pool.
            assert capsys.readouterr().out == ''
            assert np.mean(np.asarray(row['corners_native_px'])[:2, 1]) == 40.
            self.rows.append(row)

        def add_receiver(self, row: dict, prepared: PreparedView) -> None:
            assert capsys.readouterr().out == ''
            assert prepared.view.view_id == row['view_id']
            self.receivers.append(row)

        def apply(self) -> list[dict]:
            self.rows[0]['scene_corners_native_px'] = self.rows[0]['corners_native_px']
            self.rows[0]['corners_native_px'] = [[60., 45.], [5., 45.], [15., 12.], [45., 12.]]
            return [{'reference_view_id': self.rows[0]['view_id']}]

    monkeypatch.setattr(run_video, 'VideoPool', Pool)
    scenes = write_json_gz(tmp_path / 'scenes.json.gz', [[0, 100], [100, 200]])
    output = tmp_path / 'result.json.gz'
    monkeypatch.setattr('sys.argv', ['run_video', '--video', 'input.mp4', '--output', str(output), '--people',
                                     str(write_poses(tmp_path / 'poses', VideoFileFrames.frame_count)),
                                     '--saved-lines', str(write_json_gz(tmp_path / 'lines.json.gz', {})),
                                     '--scenes', str(scenes), *([] if mode is None else ['--court-mode', str(mode)])])
    mode = CourtMode.VIDEO_ROBUST if mode is None else mode

    assert run_video.main() == 0
    result = read_json_gz(output)
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert result['court_mode'] == mode
    # Only fast-robust leaves every scene without endpoint frames.
    offered = [endpoint_views is not None for endpoint_views in tools.detectors[0].endpoint_views]
    assert offered == [mode != CourtMode.FAST_ROBUST] * 2
    # stdout and the saved result agree, pooled rows included.
    assert printed == result['scenes']
    if mode == CourtMode.SCENE_ROBUST:
        assert pools == [] and 'view_groups' not in result
        assert printed[0]['corners_native_px'] == [[10., 10.], [50., 10.], [50., 40.], [10., 40.]]
        return
    view_ids = [row['view_id'] for row in result['scenes']]
    assert pools[0].court_mode == mode
    # The courtless scene waits in the pool as a receiver.
    assert [row['view_id'] for row in pools[0].rows] == view_ids[:1]
    assert [row['view_id'] for row in pools[0].receivers] == view_ids[1:]
    assert result['view_groups'] == [{'reference_view_id': view_ids[0]}]
    assert printed[0]['corners_native_px'] == [[15., 12.], [45., 12.], [60., 45.], [5., 45.]]
    assert printed[0]['scene_corners_native_px'] == [[50., 40.], [10., 40.], [10., 10.], [50., 10.]]


def test_fast_robust_rejects_court_reuse_before_any_model_loads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    tools = LiveTools(monkeypatch)
    monkeypatch.setattr('sys.argv', ['run_video', '--video', 'input.mp4', '--output', str(tmp_path / 'out.json.gz'),
                                     '--saved-lines', 'lines.json.gz', '--court-mode', 'fast-robust',
                                     '--reuse-courts'])
    with pytest.raises(SystemExit) as exited:
        run_video.main()
    assert exited.value.code == 2 and 'cannot combine with --reuse-courts' in capsys.readouterr().err
    assert tools.detectors == []
    # Callers that skip the command line get the same refusal.
    with pytest.raises(ValueError, match='cannot reuse courts'):
        run_video.detect_video(Path('input.mp4'), types.SimpleNamespace(),  # type: ignore[arg-type]
                               video_id='clip', people_dir=None, scene_source=None, reuse_courts=True,
                               court_mode=CourtMode.FAST_ROBUST)


@pytest.mark.parametrize('scene_consistency', [False, True])
def test_middle_frame_only_scenes_skip_endpoint_work_but_keep_the_foot_window(scene_consistency: bool) -> None:
    class RecordingFrames(Frames):
        def __init__(self) -> None:
            self.requests: list[list[int]] = []

        def read(self, indices: Sequence[int]) -> list[np.ndarray]:
            self.requests.append(list(indices))
            return super().read(indices)

    class FootWindowDetector(Detector):
        def __init__(self) -> None:
            super().__init__(Switches(enforce_scene_consistency=scene_consistency))
            self.feet_frames: list[list[int]] = []

        def detect(self, view, people, frames, *, known_courts=(), endpoint_views=None) -> CourtResult:
            # The detector gathers its feet from these same inputs.
            gathered = feet.window_feet(view, people, frames, self.switches.enforce_scene_consistency)
            self.feet_frames.append(gathered.kept_frames)
            return super().detect(view, people, frames, known_courts=known_courts, endpoint_views=endpoint_views)

    frames, lines, detector = RecordingFrames(), Lines(), FootWindowDetector()
    list(scene_courts(detector, frames, People(), lines, [SceneInfo(0, 100)],  # type: ignore[arg-type]
                      video_id='clip', compose_scenes=False))
    window = feet.window_frames(50, Frames.fps, 0, 100)
    assert detector.endpoint_views == [None]
    assert lines.indices == [50]
    if scene_consistency:
        # The shot check reads the whole window. These test frames change brightness
        # every frame, so it keeps only the frames near the anchor.
        assert frames.requests == [window, window]
        assert 50 in detector.feet_frames[0] and len(detector.feet_frames[0]) > 1
    else:
        # Only the middle frame is decoded; the feet still come from the whole window.
        assert frames.requests == [[50]]
        assert detector.feet_frames == [window]


@pytest.mark.parametrize(('saved_scenes', 'expected_rows'), [
    ([[0, 5], [5, 10]], [(0, 5, 2), (5, 10, 7)]),
    ([[0, 1], [1, 10]], [(0, 1, 0), (1, 10, 5)]),
    ([[0, 4], [5, 9]], None),  # old inclusive bounds leave frames 4 and 9 uncovered
])
def test_cli_scene_file_uses_exclusive_ends(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, saved_scenes: list[list[int]],
    expected_rows: list[tuple[int, int, int]] | None,
) -> None:
    LiveTools(monkeypatch)
    monkeypatch.setattr(VideoFileFrames, 'frame_count', 10)
    lines = write_json_gz(tmp_path / 'lines.json.gz', {})
    scenes = write_json_gz(tmp_path / 'scenes.json.gz', saved_scenes)
    output = tmp_path / 'result.json.gz'
    monkeypatch.setattr('sys.argv', ['run_video', '--video', 'input.mp4', '--output', str(output),
                                    '--saved-lines', str(lines), '--scenes', str(scenes), '--no-require-people'])

    if expected_rows is None:
        with pytest.raises(ValueError, match='must start at frame'):
            run_video.main()
        return
    assert run_video.main() == 0
    rows = read_json_gz(output)['scenes']
    assert [(row['start_frame'], row['end_frame'], row['frame_index']) for row in rows] == expected_rows


@pytest.mark.parametrize('scenes', [
    pytest.param([], id='no-scenes'),
    pytest.param([SceneInfo(1, 10)], id='starts-after-frame-0'),
    pytest.param([SceneInfo(-1, 10)], id='starts-before-frame-0'),
    pytest.param([SceneInfo(0, 9)], id='misses-final-frame'),
    pytest.param([SceneInfo(0, 11)], id='ends-past-frame-count'),
    pytest.param([SceneInfo(0, 4), SceneInfo(5, 10)], id='gap'),
    pytest.param([SceneInfo(0, 6), SceneInfo(5, 10)], id='overlap'),
    pytest.param([SceneInfo(0, 5), SceneInfo(5, 5), SceneInfo(5, 10)], id='empty-scene'),
    # An old inclusive partition, (0, 4) and (5, 9), leaves frames 4 and 9 uncovered.
    pytest.param([SceneInfo(0, 4), SceneInfo(5, 9)], id='old-inclusive-bounds'),
])
def test_incomplete_or_overlapping_external_scenes_fail(scenes: list[SceneInfo]) -> None:
    with pytest.raises(ValueError):
        validate_scenes(scenes, 10)


@pytest.mark.parametrize('scenes', [
    pytest.param([SceneInfo(0, 10)], id='whole-video'),
    pytest.param([SceneInfo(0, 5), SceneInfo(5, 10)], id='cut-at-frame-5'),
    pytest.param([SceneInfo(0, 1), SceneInfo(1, 9), SceneInfo(9, 10)], id='one-frame-scenes-at-both-ends'),
])
def test_valid_scene_partition(scenes: list[SceneInfo]) -> None:
    validate_scenes(scenes, 10)


@pytest.mark.parametrize(('start_frame', 'end_frame', 'middle_frame'), [(0, 10, 5), (0, 11, 5), (7, 8, 7), (5, 7, 6)])
def test_scene_middle_is_the_upper_middle_frame(start_frame: int, end_frame: int, middle_frame: int) -> None:
    assert SceneInfo(start_frame, end_frame).middle_frame == middle_frame


def test_cut_frame_starts_the_following_scene_only() -> None:
    class ThirtyFpsFrames(Frames):
        fps = 30.0

        def __init__(self) -> None:
            self.requests: list[list[int]] = []

        def read(self, indices: Sequence[int]) -> list[np.ndarray]:
            self.requests.append(list(indices))
            return super().read(indices)

    # At 30 fps the 31 samples span exactly 91 frames. Frame 91 is the cut, and the
    # 90-frame final scene is one frame too short.
    frames, lines, detector = ThirtyFpsFrames(), Lines(), Detector()
    scenes = [SceneInfo(0, 91), SceneInfo(91, 182), SceneInfo(182, 272)]
    rows = list(scene_courts(detector, frames, People(), lines, scenes,  # type: ignore[arg-type]
                            video_id='clip', reuse_courts=True))
    assert [row['status'] for row in rows] == ['no_court', 'no_court', 'scene_too_short_for_feet']
    assert frames.requests == [list(range(0, 91, 3)), list(range(91, 182, 3))]
    assert lines.indices == [45, 136]
    assert [view.scene_frames for view in detector.views] == [(0, 91), (91, 182)]


@pytest.mark.parametrize('people', [People(), None], ids=['with-people', 'optional-without-people'])
def test_endpoint_views_bring_their_own_inputs_and_a_composite_is_stored(people: People | None) -> None:
    composite = np.array([[11., 10.], [51., 10.], [50., 41.], [10., 40.]])

    class ComposingDetector(Detector):
        def __init__(self) -> None:
            super().__init__(Switches(require_people=people is not None))
            self.known: list[list] = []
            self.endpoints: list[list[ViewInputs]] = []

        def detect(self, view, people, frames, *, known_courts=(), endpoint_views=None) -> CourtResult:
            self.known.append(list(known_courts))
            self.endpoints.append(endpoint_views())
            return CourtResult(view.view_id, composite, None, 'composite', None, .7, composition={'court': 'composite'})

    detector, lines = ComposingDetector(), Lines()
    scenes = [SceneInfo(0, 100), SceneInfo(100, 200)]
    rows = list(scene_courts(detector, Frames(), people, lines, scenes,  # type: ignore[arg-type]
                            video_id='clip', reuse_courts=True))
    window = feet.window_frames(50, Frames.fps, 0, 100)
    views = detector.endpoints[0]
    assert [view.view_id for view in views] == [f'clip_scene_0000_frame_{window[0]}', f'clip_scene_0000_frame_{window[-1]}']
    assert lines.indices[:3] == [50, window[0], window[-1]]
    for view in views:
        assert np.all(view.frame == view.frame_index)
        assert view.scene_frames == (0, 100)
        assert view.provenance == same_frame_provenance(view.view_id, view.frame_index)
        assert view.alignment_image is None
        if people is None:
            assert view.person_boxes_px.shape == (0, 4)
        else:
            assert view.person_boxes_px[0, 0] == view.frame_index
    assert (rows[0]['chosen_key'], rows[0]['composition']) == ('composite', {'court': 'composite'})
    # The next scene tries the composite, in the middle frame's pixels with its paint support.
    stored, = detector.known[1]
    assert (stored.view_id, stored.paint_score) == (rows[0]['view_id'], .7)
    np.testing.assert_array_equal(stored.corners_native_px, composite)


def test_reuse_keeps_searched_templates_and_orders_by_optional_histograms() -> None:
    class ReusingDetector:
        def __init__(self) -> None:
            self.switches = Switches()
            self.attempts: list[list[str]] = []

        def detect(self, view, people, frames, *, known_courts=(), endpoint_views=None) -> CourtResult:
            self.attempts.append([known.view_id for known in known_courts])
            source = known_courts[0].view_id if len(self.attempts) == 3 else None
            corners = np.array([[10., 10.], [50., 10.], [50., 40.], [10., 40.]])
            return CourtResult(view.view_id, corners, None, 'reuse' if source else 'searched', None, .9, source)

    detector = ReusingDetector()
    scenes = [SceneInfo(0, 80, np.array([1., 0.])), SceneInfo(80, 160, np.array([0., 1.])),
              SceneInfo(160, 240, np.array([.9, .1]))]
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

        def detect(self, view, people, frames, *, known_courts=(), endpoint_views=None) -> CourtResult:
            self.attempts.append([known.view_id for known in known_courts])
            source = known_courts[0].view_id if known_courts else None
            return CourtResult(view.view_id, np.zeros((4, 2)), None, 'court', None, .9, source)

    detector = ReusingDetector()
    scenes = [SceneInfo(0, 80), SceneInfo(80, 160), SceneInfo(160, 240)]
    rows = list(scene_courts(detector, Frames(), People(), Lines(), scenes,  # type: ignore[arg-type]
                            video_id='clip', reuse_courts=True))
    first = rows[0]['view_id']
    assert detector.attempts == [[], [first], [first]]


@pytest.mark.parametrize(('scene_length', 'expected_samples'), [(10, [0, 5, 9]), (80, [2, 40, 78])])
def test_reuse_shares_median_image_and_keeps_live_anchor_inputs(scene_length: int, expected_samples: list[int]) -> None:
    class MovingFrames(Frames):
        def __init__(self) -> None:
            self.requested = []

        def read(self, indices: Sequence[int]) -> list[np.ndarray]:
            self.requested.extend(indices)
            decoded = []
            for index in indices:
                frame = np.full((48, 64, 3), 40, dtype=np.uint8)
                if index % scene_length == scene_length // 2:
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

        def detect(self, view, people, frames, *, known_courts=(), endpoint_views=None) -> CourtResult:
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
    scenes = [SceneInfo(0, scene_length), SceneInfo(scene_length, 2 * scene_length)]
    rows = list(scene_courts(detector, frames, None, AnchorLines(), scenes,  # type: ignore[arg-type]
                            video_id='clip', reuse_courts=True))
    assert [row['status'] for row in rows] == ['court', 'court']
    assert len(frames.requested) == 6
    for scene, requested in zip(scenes, (frames.requested[:3], frames.requested[3:]), strict=True):
        assert requested == [scene.start_frame + index for index in expected_samples]


@pytest.mark.parametrize('scene_consistency', [False, True])
def test_reuse_decodes_in_order_and_preloads_the_feet_window_when_needed(scene_consistency: bool) -> None:
    class RecordingFrames(Frames):
        def __init__(self) -> None:
            self.requests = []

        def read(self, indices: Sequence[int]) -> list[np.ndarray]:
            self.requests.append(list(indices))
            return super().read(indices)

    class ReusingDetector(Detector):
        def detect(self, view, people, frames, *, known_courts=(), endpoint_views=None) -> CourtResult:
            assert np.all(view.alignment_image == 40)
            return super().detect(view, people, frames)

    frames = RecordingFrames()
    detector = ReusingDetector(Switches(enforce_scene_consistency=scene_consistency))
    list(scene_courts(detector, frames, People(), Lines(), [SceneInfo(0, 80)],  # type: ignore[arg-type]
                     video_id='clip', reuse_courts=True))
    assert len(frames.requests) == 1
    requested = frames.requests[0]
    assert requested == sorted(requested)
    assert (requested[0], requested[-1]) == (2, 78)
    assert 40 in requested
    assert len(requested) == (31 if scene_consistency else 3)


def test_failed_scene_is_recorded_and_never_becomes_a_template() -> None:
    class FailingDetector(Detector):
        def __init__(self) -> None:
            super().__init__()
            self.attempts: list[list[str]] = []

        def detect(self, view, people, frames, *, known_courts=(), endpoint_views=None) -> CourtResult:
            self.attempts.append([known.view_id for known in known_courts])
            if len(self.attempts) == 2:
                raise run_video.CourtFitError('original fit returned no corners')
            return CourtResult(view.view_id, np.zeros((4, 2)), None, 'searched', None, .9)

    detector = FailingDetector()
    scenes = [SceneInfo(0, 80), SceneInfo(80, 160), SceneInfo(160, 240)]
    rows = list(scene_courts(detector, Frames(), People(), Lines(), scenes,  # type: ignore[arg-type]
                            video_id='clip', reuse_courts=True))
    assert [row['status'] for row in rows] == ['court', 'detection_failed', 'court']
    assert rows[1]['corners_native_px'] is None
    assert rows[1]['error'] == "CourtFitError('original fit returned no corners')"
    assert 'Traceback' in rows[1]['traceback']
    # Had the second scene succeeded, the third would try it first.
    assert detector.attempts == [[], [rows[0]['view_id']], [rows[0]['view_id']]]


def test_input_and_worker_failures_stop_the_video() -> None:
    class WrongFramePeople(People):
        def samples(self, indices: Sequence[int]) -> list[PersonSample]:
            return super().samples([index + 1 for index in indices])

    class BrokenPoolDetector(Detector):
        def detect(self, view, people, frames, *, known_courts=(), endpoint_views=None) -> CourtResult:
            raise BrokenProcessPool('a worker died')

    scenes = [SceneInfo(0, 100), SceneInfo(100, 200)]
    with pytest.raises(ValueError, match='did not return the requested frame'):
        list(scene_courts(Detector(), Frames(), WrongFramePeople(), Lines(), scenes,  # type: ignore[arg-type]
                          video_id='clip'))
    with pytest.raises(BrokenProcessPool):
        list(scene_courts(BrokenPoolDetector(), Frames(), People(), Lines(), scenes,  # type: ignore[arg-type]
                          video_id='clip'))


def run_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, entries: list[dict], *flags: str) -> int:
    manifest = write_json_gz(tmp_path / 'manifest.json.gz', entries)
    monkeypatch.setattr('sys.argv', ['run_video', '--manifest', str(manifest), '--output-dir', str(tmp_path / 'out'),
                                    '--deeplsd-source', 'deeplsd', '--deeplsd-weights', 'weights.tar', *flags])
    return run_video.main()


def test_batch_loads_models_once_and_keeps_each_video_separate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class CourtFinder(Detector):
        def __init__(self, switches: Switches) -> None:
            super().__init__(switches)
            self.attempts: dict[str, list[str]] = {}

        def detect(self, view, people, frames, *, known_courts=(), endpoint_views=None) -> CourtResult:
            super().detect(view, people, frames)
            self.attempts[view.view_id] = [known.view_id for known in known_courts]
            return CourtResult(view.view_id, np.zeros((4, 2)), None, 'searched', None, .9)

    tools = LiveTools(monkeypatch, CourtFinder)
    entries = [
        {'id': 'first', 'video': 'first.mp4', 'scenes': str(write_json_gz(tmp_path / 'two.json.gz', [[0, 80], [80, 100]]))},
        # Scenes that miss frames 50-99 are this video's own input fault.
        {'id': 'missing_frames', 'video': 'second.mp4', 'scenes': str(write_json_gz(tmp_path / 'short.json.gz', [[0, 50]]))},
        {'id': 'saved', 'video': 'third.mp4', 'people': str(write_poses(tmp_path / 'poses', 100))},
    ]

    assert run_manifest(monkeypatch, tmp_path, entries, '--reuse-courts') == 1
    summary = read_json_gz(tmp_path / 'out' / 'summary.json.gz')
    assert summary['schema'] == run_video.BATCH_SUMMARY_SCHEMA and summary['finished']
    assert [video['status'] for video in summary['videos']] == ['complete', 'failed', 'complete']
    assert 'Scenes end at frame 50' in summary['videos'][1]['error']
    assert summary['videos'][0]['scene_statuses'] == {'court': 1, 'scene_too_short_for_feet': 1}
    assert not (tmp_path / 'out' / 'videos' / 'missing_frames.json.gz').exists()
    saved = read_json_gz(tmp_path / 'out' / 'videos' / 'saved.json.gz')
    assert saved['schema'] == run_video.VIDEO_RESULT_SCHEMA and saved['saved_people']
    # DeepLSD, RTMLib and the detector load once for the batch.
    assert len(tools.line_loads) == len(tools.extractor_loads) == len(tools.detectors) == 1
    assert tools.detectors[0].events == ['open', 'detect', 'detect', 'close']
    # Live people get one cache per video around the shared extractor.
    (first_frames, first_extractor), (second_frames, second_extractor) = tools.people_setups
    assert first_frames is not second_frames and first_extractor is second_extractor
    # The first video's court is never offered to another video.
    assert tools.detectors[0].attempts == {'first_scene_0000_frame_40': [], 'saved_scene_0000_frame_50': []}


@pytest.mark.parametrize(('error', 'statuses', 'events'), [
    pytest.param(BrokenProcessPool('a worker died'), ['failed', 'complete'],
                 ['open', 'detect', 'close', 'open', 'detect', 'close'], id='pool-replaced'),
    pytest.param(RuntimeError('CUDA error'), ['failed', 'not_run'], ['open', 'detect', 'close'], id='batch-stops'),
])
def test_batch_replaces_a_broken_pool_but_stops_after_unknown_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error: Exception, statuses: list[str], events: list[str],
) -> None:
    class FailsFirstVideo(Detector):
        def detect(self, view, people, frames, *, known_courts=(), endpoint_views=None) -> CourtResult:
            result = super().detect(view, people, frames)
            if len(self.views) == 1:
                raise error
            return result

    tools = LiveTools(monkeypatch, FailsFirstVideo)
    entries = [{'id': 'first', 'video': 'first.mp4'}, {'id': 'second', 'video': 'second.mp4'}]

    assert run_manifest(monkeypatch, tmp_path, entries) == 1
    summary = read_json_gz(tmp_path / 'out' / 'summary.json.gz')
    assert [video['status'] for video in summary['videos']] == statuses
    assert summary['finished'] is (statuses[-1] != 'not_run')
    assert tools.detectors[0].events == events


@pytest.mark.parametrize('entries', [
    pytest.param([], id='no-videos'),
    pytest.param([{'id': '', 'video': 'a.mp4'}], id='empty-id'),
    pytest.param([{'id': 'a', 'video': 'a.mp4', 'pose': 'poses'}], id='unknown-key'),
    pytest.param([{'id': 'a/b', 'video': 'a.mp4'}], id='path-like-id'),
    pytest.param([{'id': 'a', 'video': 'a.mp4'}, {'id': 'a', 'video': 'b.mp4'}], id='duplicate-id'),
])
def test_malformed_manifest_fails_before_any_model_loads(tmp_path: Path, entries: list[dict]) -> None:
    with pytest.raises(ValueError):
        run_video.read_manifest(write_json_gz(tmp_path / 'manifest.json.gz', entries))


def test_wrong_foot_window_frames_are_not_recorded_as_a_fit_failure() -> None:
    class WrongWindowPeople(People):
        def samples(self, indices: Sequence[int]) -> list[PersonSample]:
            requested = indices if len(indices) == 1 else [index + 1 for index in indices]
            return super().samples(requested)

    detector = CourtDetector(Switches(enforce_scene_consistency=False))
    with pytest.raises(ValueError, match='[Ff]rame|[Pp]eople'):
        list(scene_courts(detector, Frames(), WrongWindowPeople(), Lines(),
                          [SceneInfo(0, 100)], video_id='clip'))


@pytest.mark.parametrize('fault', ['truncated_scenes', 'null_end', 'flat_ranges', 'corrupt_poses', 'truncated_poses'])
def test_bad_compressed_inputs_fail_only_their_video(monkeypatch, tmp_path, fault) -> None:
    LiveTools(monkeypatch)
    bad = {'id': 'bad', 'video': 'bad.mp4'}
    if fault in ('truncated_scenes', 'null_end', 'flat_ranges'):
        payload = [[0, None]] if fault == 'null_end' else [0, 100] if fault == 'flat_ranges' else [[0, 100]]
        scenes = write_json_gz(tmp_path / 'scenes.json.gz', payload)
        if fault == 'truncated_scenes':
            scenes.write_bytes(scenes.read_bytes()[:-4])
        bad['scenes'] = str(scenes)
    else:
        poses = write_poses(tmp_path / 'poses', 100)
        boxes = poses / 'pose_bboxes.npy.xz'
        contents = boxes.read_bytes()
        boxes.write_bytes(b'not an xz stream' if fault == 'corrupt_poses' else contents[:len(contents) // 2])
        bad['people'] = str(poses)
    assert run_manifest(monkeypatch, tmp_path, [bad, {'id': 'good', 'video': 'good.mp4'}]) == 1
    summary = read_json_gz(tmp_path / 'out' / 'summary.json.gz')
    assert [video['status'] for video in summary['videos']] == ['failed', 'complete']
    assert summary['finished']


def test_pose_prerun_uses_the_existing_sharded_producer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from annotator import video_metadata
    from dataset_builder import pose_sharding

    metadata = object()
    monkeypatch.setattr(video_metadata, 'probe_video_metadata', lambda video: metadata)
    calls = []
    monkeypatch.setattr(pose_sharding, 'extract_sharded_rtmlib_pose_stage', lambda **kwargs: calls.append(kwargs))
    settings = run_video.PosePrerun(tmp_path, 'pose-python', 'cuda')
    output = run_video.prerun_people(Path('clip.mp4'), 'clip', settings)
    assert output == tmp_path / 'clip'
    assert calls == [{'metadata': metadata, 'output_dir': output, 'interpreter': 'pose-python',
                      'shards': 8, 'n_max': 10, 'device': 'cuda'}]
    with pytest.raises(FileExistsError):
        run_video.prerun_people(Path('other.mp4'), 'clip', settings)
    assert len(calls) == 1


def test_batch_preruns_only_missing_people(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tools = LiveTools(monkeypatch)
    saved = write_poses(tmp_path / 'existing', VideoFileFrames.frame_count)
    calls = []

    def prerun(video, video_id, settings):
        calls.append((video, video_id, settings))
        return write_poses(settings.output_dir / video_id, VideoFileFrames.frame_count)

    monkeypatch.setattr(run_video, 'prerun_people', prerun)
    root = tmp_path / 'pose-output'
    root.mkdir()
    entries = [{'id': 'saved', 'video': 'first.mp4', 'people': str(saved)},
               {'id': 'fresh', 'video': 'second.mp4'}]
    assert run_manifest(monkeypatch, tmp_path, entries, '--pose-prerun', str(root),
                        '--pose-python', 'pose-python') == 0
    assert calls == [(Path('second.mp4'), 'fresh', run_video.PosePrerun(root, 'pose-python', 'cuda'))]
    assert tools.extractor_loads == []
    for video_id in ('saved', 'fresh'):
        result = read_json_gz(tmp_path / 'out' / 'videos' / f'{video_id}.json.gz')
        assert result['saved_people'] is True
        assert [row['status'] for row in result['scenes']] == ['no_court']
    assert read_json_gz(tmp_path / 'out/videos/saved.json.gz')['pose_prerun_seconds'] == 0


def test_single_video_prerun_failure_stops_before_court_detection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    tools = LiveTools(monkeypatch)

    def fail(*args):
        raise RuntimeError('pose worker failed')

    monkeypatch.setattr(run_video, 'prerun_people', fail)
    output = tmp_path / 'result.json.gz'
    monkeypatch.setattr('sys.argv', ['run_video', '--video', 'clip.mp4', '--output', str(output),
                                    '--deeplsd-source', 'deeplsd', '--deeplsd-weights', 'weights.tar',
                                    '--pose-prerun', str(tmp_path / 'poses')])
    with pytest.raises(RuntimeError, match='pose worker failed'):
        run_video.main()
    assert tools.extractor_loads == []
    assert not output.exists()


@pytest.mark.parametrize('failing_step', ['probe', 'extract'])
def test_prerun_process_failures_are_video_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failing_step: str,
) -> None:
    from annotator import video_metadata
    from dataset_builder import pose_sharding

    def fail(*args, **kwargs):
        raise RuntimeError('child exited with status 1')

    monkeypatch.setattr(video_metadata, 'probe_video_metadata', fail if failing_step == 'probe' else lambda _: object())
    monkeypatch.setattr(pose_sharding, 'extract_sharded_rtmlib_pose_stage', fail)
    with pytest.raises(run_video.PosePrerunError, match='clip: pose prerun failed') as caught:
        run_video.prerun_people(Path('clip.mp4'), 'clip', run_video.PosePrerun(tmp_path, 'python', 'cuda'))
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_batch_continues_after_an_isolated_pose_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tools = LiveTools(monkeypatch)
    poses = write_poses(tmp_path / 'good-poses', VideoFileFrames.frame_count)

    def prerun(video, video_id, settings):
        if video_id == 'bad':
            raise run_video.PosePrerunError('bad pose child')
        return poses

    monkeypatch.setattr(run_video, 'prerun_people', prerun)
    entries = [{'id': 'bad', 'video': 'bad.mp4'}, {'id': 'good', 'video': 'good.mp4'}]
    assert run_manifest(monkeypatch, tmp_path, entries, '--pose-prerun', str(tmp_path / 'poses')) == 1
    summary = read_json_gz(tmp_path / 'out/summary.json.gz')
    assert summary['finished'] is True
    assert [row['status'] for row in summary['videos']] == ['failed', 'complete']
    assert 'bad pose child' in summary['videos'][0]['error']
    assert tools.extractor_loads == []
