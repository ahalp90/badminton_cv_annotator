"""Still-image court detection: live lines, optional people, and no scene or video inputs."""

import gzip
import json
import sys
import types
from pathlib import Path
from typing import NamedTuple, Self

import cv2
import numpy as np
import pytest

from court_detector import run_image
from court_detector.detect import CourtDetector, CourtResult, Switches
from court_detector.inputs import ViewInputs
from court_detector.run_image import ImageTools, detect_image

HEIGHT, WIDTH = 48, 64
CORNERS = [[-5.0, 40.0], [70.0, 40.0], [50.0, 10.0], [12.0, 10.0]]


def ramp_image() -> np.ndarray:
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[:, :, 2] = np.arange(WIDTH, dtype=np.uint8)  # a left-to-right ramp shows a flipped or resized image
    return image


class Detections(NamedTuple):
    bboxes: np.ndarray
    keypoints: np.ndarray


class Extractor:
    def __init__(self, device: str = 'cpu') -> None:
        self.device = device
        self.frames: list[np.ndarray] = []

    def detect_frame(self, frame: np.ndarray) -> Detections:
        self.frames.append(frame)
        return Detections(np.array([[20., 5., 30., 40.]]), np.zeros((1, 17, 2)))


class Lines:
    def __init__(self, segments: np.ndarray | None = None) -> None:
        self.segments_px = np.array([[1., 2., 30., 40.]], dtype=np.float32) if segments is None else segments
        self.frames: list[np.ndarray] = []

    def segments(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        self.frames.append(frame)
        return self.segments_px


class Detector:
    def __init__(self, switches: Switches, corners: list[list[float]] | None = CORNERS) -> None:
        self.switches = switches
        self.corners = corners
        self.calls: list[tuple] = []
        self.events: list[str] = []

    def __enter__(self) -> Self:
        self.events.append('open')
        return self

    def __exit__(self, *_exception_info: object) -> None:
        self.events.append('close')

    def detect(self, view: ViewInputs, people, frames, *, known_courts=()) -> CourtResult:
        self.calls.append((view, people, frames))
        if people is not None:
            people.samples([view.frame_index])  # the real detector asks again for the feet window
        if self.corners is None:
            return CourtResult(view.view_id, None, 'no_gated_court', None, {'feet': .1})
        return CourtResult(view.view_id, np.array(self.corners), None, 'all_lines:0', {'feet': .1})


def test_lines_see_the_native_image_and_corners_stay_native() -> None:
    image, lines = ramp_image(), Lines()
    detector = Detector(Switches(require_people=False))
    tools = ImageTools(lines, detector, None, 2.0)  # type: ignore[arg-type]

    result = detect_image(image, tools, image_id='hall', source='hall.jpg')

    assert len(lines.frames) == 1 and lines.frames[0] is image
    view, people, frames = detector.calls[0]
    assert view.frame is image and view.frame_index == 0 and view.scene_frames == (0, 1)
    assert view.person_boxes_px.shape == (0, 4) and people is None
    assert view.provenance.has_same_image_boxes
    assert frames.size == (WIDTH, HEIGHT) and frames.read([0])[0] is image
    assert result['schema'] == 'court-detector-image/1'
    assert (result['image_id'], result['image'], result['native_size']) == ('hall', 'hall.jpg', (WIDTH, HEIGHT))
    assert result['status'] == 'court' and result['corners_native_px'] == CORNERS
    assert result['no_court_reason'] is None and result['with_people'] is False
    assert result['people_seconds'] is None and result['tools_seconds'] == 2.0
    assert 'chosen_key' not in result


def test_people_run_once_on_the_image_and_mask_their_boxes() -> None:
    image, extractor = ramp_image(), Extractor()
    detector = Detector(Switches(require_people=False), corners=None)
    tools = ImageTools(Lines(), detector, extractor, 0.0)  # type: ignore[arg-type]

    result = detect_image(image, tools, image_id='hall')

    assert len(extractor.frames) == 1 and extractor.frames[0] is image
    view, people, _ = detector.calls[0]
    np.testing.assert_array_equal(view.person_boxes_px, [[20., 5., 30., 40.]])
    assert people is not None
    assert result['status'] == 'no_court' and result['corners_native_px'] is None
    assert result['no_court_reason'] == 'no_gated_court'
    assert result['with_people'] is True and result['people_seconds'] >= 0
    assert result['image'] is None


@pytest.mark.parametrize('with_people', [False, True])
def test_real_detector_returns_no_court_for_an_image_without_lines(with_people: bool) -> None:
    extractor = Extractor() if with_people else None
    tools = ImageTools(Lines(np.empty((0, 4), dtype=np.float32)), CourtDetector(Switches(require_people=False)),
                       extractor, 0.0)  # type: ignore[arg-type]

    result = detect_image(ramp_image(), tools, image_id='blank')

    assert result['status'] == 'no_court' and result['no_court_reason'] == 'no_gated_court'
    if extractor is not None:
        assert len(extractor.frames) == 1  # the feet window reuses the cached pass


def test_image_tools_reject_the_video_player_requirement() -> None:
    with pytest.raises(ValueError, match='require_people=False'):
        ImageTools(Lines(), Detector(Switches()), None, 0.0)  # type: ignore[arg-type]


@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> types.SimpleNamespace:
    """Run main() with fake models; record what each loader received."""
    state = types.SimpleNamespace(lines=[], detectors=[], extractors=[], image=tmp_path / 'hall.png',
                                  output=tmp_path / 'out' / 'hall.json.gz')
    cv2.imwrite(str(state.image), ramp_image())

    def load_lines(source: Path, weights: Path, *, device: str) -> Lines:
        state.lines.append((source, weights, device, Lines()))
        return state.lines[-1][3]

    def make_detector(switches: Switches) -> Detector:
        state.detectors.append(Detector(switches))
        return state.detectors[-1]

    def make_extractor(device: str) -> Extractor:
        state.extractors.append(Extractor(device))
        return state.extractors[-1]

    monkeypatch.setattr(run_image, 'DeepLSDLines', load_lines)
    monkeypatch.setattr(run_image, 'CourtDetector', make_detector)
    monkeypatch.setitem(sys.modules, 'shared.rtmlib_pose', types.SimpleNamespace(RtmlibPoseExtractor=make_extractor))
    # A None entry makes any import of the package fail, so these prove neither is used.
    monkeypatch.setitem(sys.modules, 'scenedetect', None)
    monkeypatch.setattr(cv2, 'VideoCapture', None)

    def run(*options: str) -> int:
        monkeypatch.setattr('sys.argv', ['run_image', '--image', str(state.image), '--output', str(state.output),
                                         '--deeplsd-source', 'deeplsd', '--deeplsd-weights', 'weights.tar',
                                         '--device', 'cpu', *options])
        return run_image.main()

    state.run = run
    return state


def test_cli_runs_deeplsd_on_the_image_without_rtmlib_by_default(cli: types.SimpleNamespace) -> None:
    assert cli.run() == 0

    assert cli.extractors == []
    [(source, weights, device, lines)] = cli.lines
    assert (source, weights, device) == (Path('deeplsd'), Path('weights.tar'), 'cpu')
    np.testing.assert_array_equal(lines.frames[0], ramp_image())  # PNG is lossless
    [detector] = cli.detectors
    assert detector.switches.require_people is False and detector.switches.workers == 8
    assert detector.events == ['open', 'close']
    with gzip.open(cli.output, 'rt') as stream:
        saved = json.load(stream)
    assert saved['image'] == 'hall.png' and saved['image_id'] == 'hall'
    assert saved['native_size'] == [WIDTH, HEIGHT] and saved['corners_native_px'] == CORNERS
    assert saved['with_people'] is False


def test_cli_with_people_loads_rtmlib_once_and_passes_options(cli: types.SimpleNamespace) -> None:
    assert cli.run('--with-people', '--workers', '2') == 0

    [extractor] = cli.extractors
    assert extractor.device == 'cpu' and len(extractor.frames) == 1
    [detector] = cli.detectors
    assert detector.switches.workers == 2 and detector.switches.template_device == 'cpu'
    with gzip.open(cli.output, 'rt') as stream:
        assert json.load(stream)['with_people'] is True


@pytest.mark.parametrize('contents', [None, b'not an image'])
def test_unreadable_image_fails_before_models_load(cli: types.SimpleNamespace, contents: bytes | None) -> None:
    if contents is None:
        cli.image.unlink()
    else:
        cli.image.write_bytes(contents)

    with pytest.raises(OSError, match='Cannot read image'):
        cli.run('--with-people')

    assert cli.lines == [] and cli.detectors == [] and cli.extractors == []
    assert not cli.output.exists()

