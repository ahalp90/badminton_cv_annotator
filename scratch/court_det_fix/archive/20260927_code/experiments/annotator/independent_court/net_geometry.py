"""Experimental metric projection of a badminton net from court corners.

This helper assumes square pixels, a principal point at the image centre and a
pinhole camera. It searches a fixed focal-length grid from 0.4 to 4.0 image
widths and chooses the candidate whose recovered ground-plane axes are closest
to orthogonal and equal length. Those are experimental approximations for this
diagnostic; they are not a court-geometry acceptance claim.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from .detector import CORNER_COURT_M

FOCAL_MIN_WIDTHS = 0.4
FOCAL_MAX_WIDTHS = 4.0
FOCAL_GRID_SIZE = 200
COURT_WIDTH_M = float(np.max(CORNER_COURT_M[:, 0]))
COURT_LENGTH_M = float(np.max(CORNER_COURT_M[:, 1]))
NET_Y_M = COURT_LENGTH_M / 2.0
NET_POST_HEIGHT_M = 1.55
NET_CENTRE_HEIGHT_M = 1.524


@dataclass(frozen=True)
class NetProjection:
    """Projected net geometry and the camera-search diagnostics."""

    segments_px: np.ndarray  # (4, 2, 2): top halves, then left and right posts
    camera_error: float
    focal_widths: float


def _validate_corners(corners_px: np.ndarray) -> np.ndarray:
    corners = np.asarray(corners_px, dtype=np.float64)
    if corners.shape != (4, 2):
        raise ValueError(f"corners_px must have shape (4, 2), got {corners.shape}")
    if not np.isfinite(corners).all():
        raise ValueError("corners_px must contain only finite values")

    edges = np.roll(corners, -1, axis=0) - corners
    turns = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    # Image y increases downwards, so the TL, TR, BR, BL order is clockwise
    # when its signed turns are positive.
    if not np.all(turns > 1e-10):
        raise ValueError("corners_px must be a non-degenerate clockwise convex quadrilateral")
    return corners


def _validate_frame_size(frame_size: Sequence[float]) -> tuple[float, float]:
    size = np.asarray(frame_size, dtype=np.float64)
    if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError("frame_size must contain two finite positive dimensions")
    return float(size[0]), float(size[1])


def _camera_from_corners(
    corners_px: np.ndarray, frame_size: tuple[float, float]
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    width, height = frame_size
    try:
        homography = cv2.getPerspectiveTransform(
            CORNER_COURT_M.astype(np.float32), corners_px.astype(np.float32)
        ).astype(np.float64)
    except cv2.error as error:
        raise ValueError("could not recover a court homography") from error
    if not np.isfinite(homography).all() or abs(np.linalg.det(homography)) <= 1e-12:
        raise ValueError("court homography is singular or non-finite")

    focals = np.geomspace(
        FOCAL_MIN_WIDTHS * width,
        FOCAL_MAX_WIDTHS * width,
        FOCAL_GRID_SIZE,
    )
    axes = np.broadcast_to(homography[:, :2], (len(focals), 3, 2)).copy()
    principal = np.array([width / 2.0, height / 2.0])
    axes[:, :2] -= principal[None, :, None] * axes[:, 2:3]
    axes[:, :2] /= focals[:, None, None]
    norms = np.linalg.norm(axes, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = (axes[:, :, 0] * axes[:, :, 1]).sum(axis=1) / np.prod(norms, axis=1)
        ratio = np.log(norms[:, 0] / norms[:, 1])
        errors = np.hypot(cosine, ratio)
    valid = np.isfinite(errors) & np.all(np.isfinite(norms), axis=1) & np.all(norms > 0, axis=1)
    if not np.any(valid):
        raise ValueError("court homography has no valid positive focal estimate")
    best = int(np.nanargmin(np.where(valid, errors, np.nan)))

    focal = float(focals[best])
    intrinsic = np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    pose = np.linalg.inv(intrinsic) @ homography
    axis_lengths = np.linalg.norm(pose[:, :2], axis=0)
    pose /= axis_lengths.mean()
    vertical = np.cross(pose[:, 0], pose[:, 1])
    vertical /= np.linalg.norm(vertical)
    return float(errors[best]), focal / width, pose, intrinsic, vertical


def _project_camera_points(points_camera: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    depths = points_camera[:, 2]
    if not np.isfinite(points_camera).all() or np.any(depths <= 1e-10):
        raise ValueError("net geometry contains a non-positive or non-finite camera depth")
    pixels_h = points_camera @ intrinsic.T
    pixels = pixels_h[:, :2] / depths[:, None]
    if not np.isfinite(pixels).all():
        raise ValueError("net geometry projects to non-finite image coordinates")
    return pixels


def project_net(corners_px: Sequence[Sequence[float]], frame_size: Sequence[float]) -> NetProjection:
    """Project the net posts and top halves from a detected court quadrilateral.

    The court corners use TL, TR, BR, BL order in image coordinates. The
    returned segments are the two top halves followed by the left and right
    posts, all in native image pixels. The world-up direction is the negative
    of the recovered ground-plane normal for this clockwise image ordering.

    :param corners_px: Four finite court corners in clockwise image order.
    :param frame_size: Image dimensions as ``(width, height)``.
    :return: Net segments and the focal-search diagnostics.
    :raises ValueError: If the quadrilateral, camera estimate or depths are invalid.
    """
    corners = _validate_corners(np.asarray(corners_px, dtype=np.float64))
    size = _validate_frame_size(frame_size)
    camera_error, focal_widths, pose, intrinsic, vertical = _camera_from_corners(corners, size)

    corner_world = np.column_stack((CORNER_COURT_M.astype(np.float64), np.ones(4)))
    corner_camera = corner_world @ pose.T
    if np.any(corner_camera[:, 2] <= 1e-10) or not np.isfinite(corner_camera).all():
        raise ValueError("court corners have invalid positive camera depths")

    ground_world = np.array(
        [[0.0, NET_Y_M, 1.0], [COURT_WIDTH_M / 2.0, NET_Y_M, 1.0], [COURT_WIDTH_M, NET_Y_M, 1.0]],
        dtype=np.float64,
    )
    ground_camera = ground_world @ pose.T
    heights = np.array([NET_POST_HEIGHT_M, NET_CENTRE_HEIGHT_M, NET_POST_HEIGHT_M])
    top_camera = ground_camera - heights[:, None] * vertical
    ground_px = _project_camera_points(ground_camera, intrinsic)
    top_px = _project_camera_points(top_camera, intrinsic)
    segments = np.array(
        [
            [top_px[0], top_px[1]],
            [top_px[1], top_px[2]],
            [ground_px[0], top_px[0]],
            [ground_px[2], top_px[2]],
        ],
        dtype=np.float64,
    )
    return NetProjection(segments, camera_error, focal_widths)
