"""What the court detector reads: one view's frame and lines, and the video and people around it."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple, Protocol

import numpy as np

from experiments.annotator.independent_court.case_provenance import (
    CaseProvenance,
    ImageKind,
)


class PersonSample(NamedTuple):
    """The people found in one video frame."""

    frame_index: int
    boxes_px: np.ndarray  # (people, 4) x1, y1, x2, y2 in FrameReader pixels
    keypoints_px: np.ndarray  # (people, 17, 2) COCO-17 x, y in FrameReader pixels


class PeopleSource(Protocol):
    def samples(self, frame_indices: Sequence[int]) -> list[PersonSample]:
        """One sample per requested frame, in the order asked."""
        ...


class FrameReader(Protocol):
    fps: float
    size: tuple[int, int]  # (width, height) of the decoded frames

    def read(self, frame_indices: Sequence[int]) -> list[np.ndarray]:
        """BGR frames, in the order asked."""
        ...


@dataclass(frozen=True)
class ViewInputs:
    """One analysed image and what is already known about it."""

    view_id: str
    frame: np.ndarray  # (height, width, 3) BGR uint8 at native size: the image the court is found in
    frame_index: int  # the video frame it shows; for a composite, its middle frame
    scene_frames: tuple[int, int]  # first and last frame of the scene, inclusive
    segments_px: np.ndarray  # (fragments, 4) DeepLSD x1, y1, x2, y2 in native pixels
    person_boxes_px: np.ndarray  # (boxes, 4) in native pixels; masked out only when provenance allows
    provenance: CaseProvenance


def same_frame_provenance(view_id: str, frame_index: int) -> CaseProvenance:
    """Provenance for a live view: the image and its person boxes come from the same video frame."""
    return CaseProvenance(view_id, ImageKind.SOURCE_FRAME, (frame_index,), frame_index)
