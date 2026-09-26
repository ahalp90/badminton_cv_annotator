"""Line-averaged paint: pass or fail each painted line of a court on its average contrast along its length.

W5's paint score passes or fails each sample on its own. On faint far lines few samples pass, on the right
court and on a court slipped by one line alike. This module averages each line's grey levels along its
whole length instead: at the line, and at a point on the floor either side of it. The line passes if its
average is brighter than both sides' averages by at least PASS_BAR_GREY.

A court's line-averaged paint score is W5's span-weighted paint score with that per-line pass: each line
counts its fragment support (W5's q_geom) if it passes and 0 if not. Lines are weighted by their image
length within each direction, and the weaker direction counts. A line W5 sees but this module cannot
measure, such as a sliver at the frame's edge, counts as unpainted; W5 leaves such a line out.
check_20260926_line_paint/README.md has the measurements behind the constants.

Research scripts may import this, so it imports only numpy, OpenCV and the experiments package.
"""

from __future__ import annotations

import cv2
import numpy as np

from experiments.annotator.independent_court.assignment import MARKING_INTERVALS
from experiments.annotator.independent_court.paint_geometry import CENTRE_SEGMENTS_M

PAINT_CRITERION = "q_paint10_span_weighted"
STEP_M = 0.01  # one sample per 1 cm of floor along the line
# Each side point sits this share of the gap to the nearest parallel painted line away, so it stays on bare
# floor; capped so that a line with no close neighbour is compared with nearby floor.
SIDE_SHARE_OF_GAP = 0.4
MAX_SIDE_M = 0.30
PASS_BAR_GREY = 1.0  # check_20260926_line_paint/line_bar.py: best separates true lines from bare floor
LENGTHWISE_MARKINGS = range(5)
TRANSVERSE_MARKINGS = range(5, 11)


def side_distance_m(line_m: np.ndarray) -> float:
    """How far each side point sits from a court line, on the floor."""
    (start_x, start_y), _ = line_m
    lengthwise = line_m[0, 0] == line_m[1, 0]
    axis = 0 if lengthwise else 1
    parallel = {segment[0, axis] for segment in CENTRE_SEGMENTS_M if (segment[0, 0] == segment[1, 0]) == lengthwise}
    position = start_x if lengthwise else start_y
    gap = min(abs(other - position) for other in parallel if other != position)
    return min(MAX_SIDE_M, SIDE_SHARE_OF_GAP * gap)


SIDE_DISTANCES_M = [side_distance_m(segment) for segment in CENTRE_SEGMENTS_M]


def line_samples(grey: np.ndarray, homography_native: np.ndarray, line_m: np.ndarray, side_m: float,
                 boxes_native: np.ndarray) -> np.ndarray:
    """Grey levels at each usable 1 cm sample along a floor line: columns are one side, the line, the other side.

    A sample is usable when its three points are in front of the camera, inside the frame and outside
    every person box.
    """
    start_m, end_m = line_m
    length_m = np.linalg.norm(end_m - start_m)
    along = (end_m - start_m) / length_m
    normal = np.array([-along[1], along[0]])
    centres_m = start_m + np.arange(0, length_m, STEP_M)[:, None] * along
    points_m = centres_m[:, None, :] + np.array([-side_m, 0.0, side_m])[None, :, None] * normal
    homogeneous = np.concatenate([points_m, np.ones(points_m.shape[:-1] + (1,))], axis=-1) @ homography_native.T
    in_front = homogeneous[..., 2] > 0
    points_px = homogeneous[..., :2] / np.where(in_front, homogeneous[..., 2], 1.0)[..., None]
    height, width = grey.shape
    in_frame = ((points_px >= 0) & (points_px < [width, height])).all(axis=-1)
    x, y = points_px[..., 0], points_px[..., 1]
    in_box = np.zeros(in_frame.shape, dtype=bool)
    for x1, y1, x2, y2 in boxes_native:
        in_box |= (x >= x1) & (x <= x2) & (y >= y1) & (y <= y2)
    usable_points = in_front & in_frame & ~in_box
    usable = usable_points.all(axis=1)
    points_px = points_px[usable].astype(np.float32)
    if not len(points_px):
        return np.empty((0, 3), dtype=np.float32)
    return cv2.remap(grey, points_px[..., 0], points_px[..., 1], cv2.INTER_LINEAR)


def line_contrast(samples: np.ndarray) -> float | None:
    """How much brighter the line's average is than the darker-contrast side's average; None with no samples."""
    if not len(samples):
        return None
    one_side, line, other_side = samples.mean(axis=0)
    return float(min(line - one_side, line - other_side))


def court_paint(evidence: dict, homography_working: np.ndarray, grey_native: np.ndarray,
                native_per_working: np.ndarray, boxes_native: np.ndarray) -> float:
    """A court's line-averaged paint score, from W5's evidence for that court."""
    homography_native = np.diag([*native_per_working, 1.0]) @ homography_working
    line_scores: list[float | None] = []
    for marking, segments in enumerate(MARKING_INTERVALS):
        support = evidence["markings"][marking]["q_geom"]
        samples = np.concatenate([line_samples(grey_native, homography_native, CENTRE_SEGMENTS_M[segment],
                                               SIDE_DISTANCES_M[segment], boxes_native) for segment in segments])
        contrast = line_contrast(samples)
        passes = contrast is not None and contrast >= PASS_BAR_GREY
        line_scores.append(None if support is None else support * passes)
    direction_scores = []
    for markings in (LENGTHWISE_MARKINGS, TRANSVERSE_MARKINGS):
        # W5's span weighting: each line's image length, in working px. A court W5 scored has a visible
        # line in each direction, so neither list is empty.
        weighted = [(line_scores[marking], evidence["markings"][marking]["projected_visible_span_px"])
                    for marking in markings]
        weighted = [(score, span) for score, span in weighted if score is not None and span > 0]
        direction_scores.append(sum(score * span for score, span in weighted) / sum(span for _, span in weighted))
    return min(direction_scores)


def rescore_rows(rows: list[dict], record: dict, native_frame: np.ndarray, context) -> list[dict]:
    """The net choice's rows with W5's paint score replaced by the line-averaged one.

    A view whose W5 ranking fell back to the geometry score keeps its rows as they are.
    """
    if record["rankings"]["C"]["r2_criterion"] != PAINT_CRITERION:
        return rows
    candidates = {item["origin_key"]: item for item in record["parents"] + record["valid_children"]}
    grey = cv2.cvtColor(native_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    native_per_working = np.asarray(context.native_size, dtype=float) / np.asarray(context.size, dtype=float)
    boxes_native = np.asarray(context.mask_boxes, dtype=float).reshape(-1, 4) * np.tile(native_per_working, 2)
    rescored = []
    for row in rows:
        candidate = candidates[row["origin_key"]]
        paint = court_paint(candidate["evidence"], np.asarray(candidate["homography_working"], dtype=float), grey,
                            native_per_working, boxes_native)
        rescored.append({**row, "w5_paint_score": row["paint_score"], "paint_score": paint})
    return rescored
