"""Match recurring static camera views before sharing their court calibration."""

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from shared.court import HOMOGRAPHY_RESOLUTION

VIEW_RESOLUTION = (960, 540)
HASH_SIZE = 16
HASH_LOWPASS = 2
MAX_HASH_DISTANCE = 0.30
MIN_ALIGNMENT_CORRELATION = 0.8
MAX_ALIGNMENT_SHIFT_REFPX = 1.0
MIN_SHARED_SCENES = 3
ALIGNMENT_MASK_SCALE = 1.08
ALIGNMENT_ITERATIONS = 80
ALIGNMENT_EPSILON = 1e-5
ALIGNMENT_BLUR_SIZE = 5


@dataclass(frozen=True)
class CourtView:
    hashes: np.ndarray  # One perceptual hash per selected scene sample.
    image: np.ndarray  # Median greyscale image at VIEW_RESOLUTION.


def describe_court_view(frames: Sequence[np.ndarray]) -> CourtView:
    """Summarise the first, middle and last already-decoded scene samples."""
    from scenedetect import HashDetector

    indices = sorted({0, len(frames) // 2, len(frames) - 1})
    images = [cv2.resize(frames[index], VIEW_RESOLUTION, interpolation=cv2.INTER_AREA) for index in indices]
    hashes = np.array([HashDetector.hash_frame(image, HASH_SIZE, HASH_LOWPASS) for image in images])
    greyscale = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
    return CourtView(hashes, np.median(greyscale, axis=0).astype(np.uint8))


@dataclass(frozen=True)
class ViewAlignment:
    """How a sample view's image lines up with a template view's, inside the template court."""

    correlation: float  # ECC correlation coefficient
    warp: np.ndarray  # (3, 3) float32 homography from template to sample VIEW_RESOLUTION pixels
    moved_corners_refpx: np.ndarray  # (4, 2) the template court's corners carried into the sample
    shift_refpx: float  # the largest of those four corner movements

    @property
    def matches(self) -> bool:
        """Whether the images correlate and the camera has not measurably moved."""
        return bool(self.correlation >= MIN_ALIGNMENT_CORRELATION and self.shift_refpx <= MAX_ALIGNMENT_SHIFT_REFPX)


def measure_view_alignment(
    template_image: np.ndarray, sample_image: np.ndarray, corners_refpx: np.ndarray,
) -> ViewAlignment | None:
    """ECC-align two greyscale VIEW_RESOLUTION images inside the template court.

    :param corners_refpx: The template court's corners in HOMOGRAPHY_RESOLUTION pixels.
    :return: The alignment, or None when ECC cannot measure one.
    """
    scale = np.asarray(VIEW_RESOLUTION) / np.asarray(HOMOGRAPHY_RESOLUTION)
    corners = corners_refpx * scale
    centre = corners.mean(axis=0)
    mask = np.zeros(sample_image.shape, np.uint8)
    cv2.fillConvexPoly(mask, np.rint(centre + ALIGNMENT_MASK_SCALE * (corners - centre)).astype(np.int32), 255)
    try:
        correlation, warp = cv2.findTransformECC(
            template_image, sample_image, np.eye(3, dtype=np.float32), cv2.MOTION_HOMOGRAPHY,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, ALIGNMENT_ITERATIONS, ALIGNMENT_EPSILON),
            mask, ALIGNMENT_BLUR_SIZE,
        )
    except cv2.error:
        return None
    moved = cv2.perspectiveTransform(corners[None].astype(np.float32), warp)[0]
    shift = np.linalg.norm((moved - corners) / scale, axis=1).max()
    return ViewAlignment(correlation, warp, moved / scale, shift)


def _view_alignment(template: CourtView, sample: CourtView, corners_refpx: np.ndarray) -> bool:
    alignment = measure_view_alignment(template.image, sample.image, corners_refpx)
    # An unmeasurable alignment provides no evidence for sharing calibration.
    return alignment is not None and alignment.matches


def matching_view_groups(
    views: Sequence[CourtView | None], corners_refpx: Sequence[np.ndarray | None], scene_valid: Sequence[bool],
) -> list[list[int]]:
    """Find image-supported groups among initially accepted, static scene courts.

    Every member must hash-match and align to one fixed representative. Requiring
    all pairs to hash-match can split a static view when players or overlays change.
    Alignment to the representative excludes zoom/crop changes and neighbour chains.
    Missing image evidence leaves a scene's existing calibration in place.
    """
    indices = [index for index, (view, valid) in enumerate(zip(views, scene_valid)) if view is not None and valid]
    distances = np.zeros((len(indices), len(indices)))
    for first, scene_index in enumerate(indices):
        hashes = views[scene_index].hashes
        for second in range(first + 1, len(indices)):
            other_hashes = views[indices[second]].hashes
            distance = np.median(np.mean(hashes[:, None] != other_hashes[None, :], axis=(2, 3)))
            distances[first, second] = distances[second, first] = distance
    remaining = list(range(len(indices)))
    representatives = remaining.copy()
    matched = []
    while len(remaining) >= MIN_SHARED_SCENES and representatives:
        medoid = representatives[int(np.argmin(distances[np.ix_(representatives, remaining)].sum(axis=1)))]
        representatives.remove(medoid)
        template_index = indices[medoid]
        template = views[template_index]
        corners = corners_refpx[template_index]
        members = [position for position in remaining
                   if distances[medoid, position] <= MAX_HASH_DISTANCE
                   and _view_alignment(template, views[indices[position]], corners)]
        if len(members) >= MIN_SHARED_SCENES:
            matched.append([indices[position] for position in members])
            remaining = [position for position in remaining if position not in members]
            representatives = [position for position in representatives if position not in members]
    return matched
