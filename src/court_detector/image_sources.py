"""Track which video frames supplied each image and set of person boxes.

The detector hides person boxes from paint measurements only when they come from
the measured image itself. A caller states where the image and boxes came from in
a CaseProvenance; the detector never infers it from a view's name."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ImageKind(StrEnum):
    """Kind of pixels used for the measured image."""

    SOURCE_FRAME = "source_frame"
    COMPOSITE = "composite"


class BoxRelation(StrEnum):
    """Relationship between selected person boxes and the measured image."""

    SAME_IMAGE = "same_image"
    NEARBY_SOURCE_FRAME = "nearby_source_frame"
    COMPOSITE = "composite"


@dataclass(frozen=True, slots=True)
class CaseProvenance:
    """Validated immutable provenance for one view."""

    case_id: str
    image_kind: ImageKind
    image_frame_indices: tuple[int, ...]
    box_frame_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id:
            raise ValueError("case_id must be a non-empty string")
        if not isinstance(self.image_kind, ImageKind):
            raise TypeError("image_kind must be an ImageKind")
        if not isinstance(self.image_frame_indices, tuple) or not self.image_frame_indices:
            raise ValueError("image_frame_indices must be a non-empty tuple")
        if any(isinstance(frame, bool) or not isinstance(frame, int) or frame < 0 for frame in self.image_frame_indices):
            raise ValueError("image_frame_indices must contain non-negative integers")
        if self.image_kind is ImageKind.SOURCE_FRAME and len(self.image_frame_indices) != 1:
            raise ValueError("source_frame requires exactly one image frame")
        if len(set(self.image_frame_indices)) != len(self.image_frame_indices):
            raise ValueError("image_frame_indices must not contain duplicates")
        if isinstance(self.box_frame_index, bool) or not isinstance(self.box_frame_index, int) or self.box_frame_index < 0:
            raise ValueError("box_frame_index must be a non-negative integer")

    @property
    def box_relation(self) -> BoxRelation:
        """Return the relation derived from image kind and frame identity."""

        if self.image_kind is ImageKind.COMPOSITE:
            return BoxRelation.COMPOSITE
        if self.box_frame_index == self.image_frame_indices[0]:
            return BoxRelation.SAME_IMAGE
        return BoxRelation.NEARBY_SOURCE_FRAME

    @property
    def has_same_image_boxes(self) -> bool:
        """Return whether person boxes describe the measured image exactly."""

        return self.box_relation is BoxRelation.SAME_IMAGE

    @property
    def unavailable_reason(self) -> str | None:
        """Return a plain reason when same-image person boxes are unavailable."""

        if self.has_same_image_boxes:
            return None
        if self.box_relation is BoxRelation.COMPOSITE:
            return "same-image person boxes unavailable: measured image is a composite"
        return (
            "same-image person boxes unavailable: boxes come from nearby source frame "
            f"{self.box_frame_index}, while the image is source frame {self.image_frame_indices[0]}"
        )


def require_same_image_boxes(case: CaseProvenance) -> CaseProvenance:
    """Require same-image person boxes and return the validated case."""

    if not isinstance(case, CaseProvenance):
        raise TypeError("case must be a CaseProvenance")
    if not case.has_same_image_boxes:
        raise ValueError(case.unavailable_reason or "same-image person boxes unavailable")
    return case


__all__ = [
    "BoxRelation",
    "CaseProvenance",
    "ImageKind",
    "require_same_image_boxes",
]
