"""Detect a court in one still image with live DeepLSD lines and, on request, RTMLib people.

An image has no scenes or neighbouring frames, so this mode needs neither a video
nor PySceneDetect. People are optional evidence here; the player checks that
video mode requires over a three-second window cannot apply to one image.
README.md owns the options and the output format.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

# Process workers inherit these settings. Set them before importing NumPy.
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['BLIS_NUM_THREADS'] = '1'

import cv2
import numpy as np

from .detect import CourtDetector, Switches
from .geometry import normalise_output_corners
from .inputs import ViewInputs, same_frame_provenance
from .line_sources import DeepLSDLines, LineSource
from .measurements import write_json_gz
from .template_arrays import TEMPLATE_DEVICES
from .video_inputs import RtmlibPeople

if TYPE_CHECKING:
    from shared.rtmlib_pose import RtmlibPoseExtractor

IMAGE_RESULT_SCHEMA = 'court-detector-image/1'
# The image is the only frame of a one-frame scene [0, 1).
IMAGE_FRAME_INDEX = 0
# An image has no frame rate. Any positive rate makes a one-frame scene too short
# for the three-second foot window, so the detector samples this image alone.
IMAGE_FPS = 1.0


class StillFrame:
    """One image served through the detector's frame-reader interface."""

    fps = IMAGE_FPS

    def __init__(self, image: np.ndarray) -> None:
        height, width = image.shape[:2]
        self.image = image
        self.size = (width, height)

    def read(self, frame_indices: Sequence[int]) -> list[np.ndarray]:
        if any(index != IMAGE_FRAME_INDEX for index in frame_indices):
            raise IndexError(f'A still image has only frame {IMAGE_FRAME_INDEX}, not {list(frame_indices)}')
        return [self.image for _ in frame_indices]


@dataclass(frozen=True)
class ImageTools:
    """Models loaded once and reused for every image.

    The detector must run with `require_people=False`: one image cannot supply
    the player window that the required checks sample.
    """

    lines: LineSource
    detector: CourtDetector
    pose_extractor: RtmlibPoseExtractor | None  # None unless people were requested
    load_seconds: float

    def __post_init__(self) -> None:
        if self.detector.switches.require_people:
            raise ValueError('Image mode needs Switches(require_people=False); one image has no player window')


def read_image(path: Path) -> np.ndarray:
    """(height, width, 3) BGR uint8 image at its native size; an unreadable file raises OSError."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f'Cannot read image: {path}')
    image.flags.writeable = False
    return image


def load_image_tools(switches: Switches, deeplsd_source: Path, deeplsd_weights: Path, *, device: str = 'cuda',
                     with_people: bool = False) -> ImageTools:
    """Load DeepLSD, the detector and, only when `with_people` is set, RTMLib.

    :param device: torch or ONNX device for DeepLSD and RTMLib.
    """
    started = perf_counter()
    lines = DeepLSDLines(deeplsd_source, deeplsd_weights, device=device)
    pose_extractor = None
    if with_people:
        from shared.rtmlib_pose import RtmlibPoseExtractor

        pose_extractor = RtmlibPoseExtractor(device=device)
    return ImageTools(lines, CourtDetector(switches), pose_extractor, perf_counter() - started)


def detect_image(image: np.ndarray, tools: ImageTools, *, image_id: str, source: str | None = None) -> dict[str, Any]:
    """Find the court in one BGR image; return the `IMAGE_RESULT_SCHEMA` result.

    Open `tools.detector` as a context manager around one or many calls, so every
    image shares one worker pool. A search or fit failure raises CourtFitError.

    :param image: (height, width, 3) BGR uint8 image at native size, as `read_image` returns.
    :param image_id: names the view in the result and in any detector artefacts.
    :param source: the image's file name for the result; None when the pixels come from elsewhere.
    """
    frame = StillFrame(image)
    started = perf_counter()
    segments = tools.lines.segments(image, IMAGE_FRAME_INDEX)
    line_seconds = perf_counter() - started

    people = None
    people_seconds = None
    boxes = np.empty((0, 4), dtype=float)
    if tools.pose_extractor is not None:
        started = perf_counter()
        # RtmlibPeople caches by frame, so the detector's own request reuses this one pass.
        people = RtmlibPeople(frame, tools.pose_extractor)
        boxes = people.samples([IMAGE_FRAME_INDEX])[0].boxes_px
        people_seconds = perf_counter() - started

    started = perf_counter()
    view = ViewInputs(image_id, image, IMAGE_FRAME_INDEX, (IMAGE_FRAME_INDEX, IMAGE_FRAME_INDEX + 1), segments,
                      boxes, same_frame_provenance(image_id, IMAGE_FRAME_INDEX))
    result = tools.detector.detect(view, people, frame)
    corners = None if result.corners_native_px is None else normalise_output_corners(result.corners_native_px)
    return {'schema': IMAGE_RESULT_SCHEMA, 'image_id': image_id, 'image': source, 'native_size': frame.size,
            'status': 'court' if corners is not None else 'no_court',
            'corners_native_px': None if corners is None else corners.tolist(),
            'no_court_reason': result.no_court_reason, 'with_people': tools.pose_extractor is not None,
            'tools_seconds': tools.load_seconds, 'line_seconds': line_seconds, 'people_seconds': people_seconds,
            'detection_seconds': perf_counter() - started, 'stage_seconds': result.stage_seconds}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True, help='image file that OpenCV can read, such as JPG or PNG')
    parser.add_argument('--output', type=Path, required=True, help='output .json.gz file')
    parser.add_argument('--deeplsd-source', type=Path, required=True)
    parser.add_argument('--deeplsd-weights', type=Path, required=True)
    parser.add_argument('--device', default='cuda', help='device for DeepLSD and, with --with-people, RTMLib')
    parser.add_argument('--with-people', action='store_true',
                        help='run RTMLib to mask people from paint measurements; '
                             'with --full, also break exact search ties')
    parser.add_argument('--template-device', choices=TEMPLATE_DEVICES, default='cpu',
                        help='device for line-template scoring; cuda needs CuPy and a GPU (default: cpu)')
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=8)
    search_options = parser.add_mutually_exclusive_group()
    search_options.add_argument('--fast', action='store_true', help='search line templates alone (default)')
    search_options.add_argument('--full', action='store_true', help='also search all lines and painted lines')
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    # Check the image before spending time on model loading.
    image = read_image(args.image)
    switches = Switches(workers=args.workers, timing=True, require_people=False,
                        template_device=args.template_device, full_no_people_search=args.full)
    tools = load_image_tools(switches, args.deeplsd_source, args.deeplsd_weights, device=args.device,
                             with_people=args.with_people)
    with tools.detector:
        result = detect_image(image, tools, image_id=args.image.stem, source=args.image.name)
    write_json_gz(args.output, result)
    print(json.dumps(result), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
