"""Match canonical court offsets along two supplied homogeneous perspective directions."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from vp_pruning import angular_residuals, normalisation

from experiments.annotator.independent_court import assignment, detector


def corner_errors(corners: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Compare physical courts while allowing the canonical 180-degree relabelling."""
    direct = np.linalg.norm(corners - reference, axis=-1).max(axis=-1)
    rotated = np.linalg.norm(corners - reference[[2, 3, 0, 1]], axis=-1).max(axis=-1)
    return np.minimum(direct, rotated)


@dataclass(frozen=True)
class Settings:
    angle_deg: float = 1.5
    max_groups: int = 128
    keep_axes: int = 64
    minimum_matches: int = 3
    batch: int = 256
    maximum_condition: float = 1e8


@dataclass(frozen=True)
class AxisMatches:
    parameters: np.ndarray  # One affine scale/offset pair per enumerated interpretation.
    scores: np.ndarray
    matches: np.ndarray  # Predicted coordinate -> original observed group ID, or -1.
    anchors: np.ndarray  # Two original group IDs and two template coordinate indexes.
    supported: np.ndarray
    player_compatible: np.ndarray
    distinct: np.ndarray
    retained: np.ndarray
    diagnostics: dict


def basis_for(points: np.ndarray, size: tuple[int, int], settings: Settings) -> tuple[np.ndarray | None, dict]:
    """Choose a finite chart without converting infinite VPs to Euclidean points."""
    normaliser = normalisation(size)
    directions = np.linalg.solve(normaliser, points.T)
    directions /= np.linalg.norm(directions, axis=0)
    grid = np.array([(horizontal, vertical, 1.) for vertical in (-.25, 0., .25)
                     for horizontal in (-.4, 0., .4)])
    determinants = grid @ np.cross(directions[:, 0], directions[:, 1])
    origin = int(np.argmax(np.abs(determinants)))
    normalised = np.column_stack((directions, grid[origin]))
    condition = float(np.linalg.cond(normalised))
    details = {'origin_grid_index': origin, 'condition': condition if np.isfinite(condition) else None}
    if not np.isfinite(condition) or condition > settings.maximum_condition:
        return None, {**details, 'status': 'degenerate_basis'}
    return normaliser @ normalised, {**details, 'status': 'valid'}


def group_lines(observations: assignment.Observations) -> tuple[np.ndarray, np.ndarray]:
    """Use existing group leaders while preserving their raw-fragment identities."""
    endpoints = np.asarray([observations.segments[members[0]] for members in observations.groups]).reshape(-1, 2, 2)
    homogeneous = np.concatenate((endpoints, np.ones((*endpoints.shape[:-1], 1))), axis=2)
    lines = np.cross(homogeneous[:, 0], homogeneous[:, 1])
    lines /= np.linalg.norm(lines[:, :2], axis=1)[:, None]
    return endpoints, lines


def offsets(
    basis: np.ndarray, axis: int, observations: assignment.Observations,
    size: tuple[int, int], settings: Settings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    endpoints, lines = group_lines(observations)
    normaliser = normalisation(size)
    direction = np.linalg.solve(normaliser, basis[:, 1 - axis])
    angles = angular_residuals(lines @ normaliser, direction[None])[0]
    inverse = np.linalg.inv(basis)
    homogeneous = np.concatenate((endpoints, np.ones((*endpoints.shape[:-1], 1))), axis=2)
    rectified = homogeneous @ inverse.T
    finite_chart = (np.abs(rectified[..., 2]) > 1e-10).all(axis=1)
    same_side = rectified[:, 0, 2] * rectified[:, 1, 2] > 0
    compatible = angles <= settings.angle_deg
    eligible = np.flatnonzero(compatible & finite_chart & same_side)
    order = np.lexsort((eligible, -observations.group_lengths[eligible]))
    ids = eligible[order[:settings.max_groups]]
    midpoint = rectified[ids].mean(axis=1)
    values = midpoint[:, axis] / midpoint[:, 2]
    ordered = np.argsort(values, kind='stable')
    ids, values = ids[ordered], values[ordered]
    details = {'groups': len(lines), 'angular_compatible': int(compatible.sum()),
               'horizon_excluded_group_ids': np.flatnonzero(compatible & ~(finite_chart & same_side)).tolist(),
               'group_cap_excluded_ids': eligible[order[settings.max_groups:]].tolist(),
               'retained_group_ids': ids.tolist(), 'orientation_residual_deg': angles[ids].tolist(),
               'offsets': values.tolist()}
    return ids, values, endpoints[ids], details


def score_axes(
    parameters: np.ndarray, coordinates: np.ndarray, endpoints: np.ndarray, basis: np.ndarray, axis: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measure image-space endpoint residuals and forbid reuse of one group by two offsets."""
    inverse = np.linalg.inv(basis)
    predicted = parameters[:, :1] * coordinates + parameters[:, 1:]
    lines = inverse[axis][None, None] - predicted[..., None] * inverse[2]
    norms = np.linalg.norm(lines[..., :2], axis=2)
    distances = np.abs(np.einsum('hmd,ged->hmge', lines[..., :2], endpoints) + lines[..., 2, None, None])
    distances = distances.max(axis=3) / np.maximum(norms[..., None], 1e-15)
    nearest = distances.argmin(axis=2)
    residual = np.take_along_axis(distances, nearest[..., None], axis=2)[..., 0]
    response = np.exp(-.5 * np.square(residual / assignment.DISTANCE_SIGMA_PX))
    same_group = nearest[:, :, None] == nearest[:, None, :]
    stronger = response[:, None, :] > response[:, :, None]
    tied_before = ((response[:, None, :] == response[:, :, None])
                  & (np.arange(len(coordinates))[None, None, :] < np.arange(len(coordinates))[None, :, None]))
    exclusive = ~np.any(same_group & (stronger | tied_before), axis=2)
    supported = exclusive & (residual <= assignment.SUPPORT_DISTANCE_PX)
    response *= exclusive
    return response.mean(axis=1), np.where(supported, nearest, -1), supported.sum(axis=1)


def rectify_feet(basis: np.ndarray, feet_px: np.ndarray) -> np.ndarray:
    """Feet in the basis chart: (frames, foot samples, 2), one column per court axis. NaN feet stay NaN."""
    homogeneous_feet = np.concatenate((feet_px, np.ones((*feet_px.shape[:-1], 1))), axis=2)
    mapped = homogeneous_feet @ np.linalg.inv(basis).T
    with np.errstate(divide='ignore', invalid='ignore'):
        return mapped[..., :2] / mapped[..., 2:]


def band_masks(parameters: np.ndarray, rectified_feet: np.ndarray, extent: float) -> tuple[np.ndarray, np.ndarray]:
    """Foot samples inside each axis hypothesis's court band, and those inside its first half.

    The band is zone_net.player_fractions' court test along this one axis.

    :param rectified_feet: (frames, foot samples) basis-chart coordinate along this axis.
    :return: two boolean arrays of shape (axis hypotheses, frames, foot samples).
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        position = (rectified_feet[None] - parameters[:, None, None, 1]) / parameters[:, None, None, 0] / extent
    inside = np.isfinite(position) & (position >= -.15) & (position <= 1.15)
    return inside, inside & (position < .5)


def necessary_players(parameters: np.ndarray, rectified_feet: np.ndarray, axis: int, extent: float) -> np.ndarray:
    """Apply necessary parts of the existing joint player rule before an axis cap."""
    inside, far = band_masks(parameters, rectified_feet, extent)
    one = inside.any(axis=2).all(axis=1)
    if axis == 0:
        return one
    near = inside & ~far
    return one & ((far.any(axis=2) & near.any(axis=2)).mean(axis=1) >= .5)


def joint_player_fractions(
    basis: np.ndarray, horizontal: AxisMatches, vertical: AxisMatches, feet_px: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """zone_net.player_fractions for every combined court, in combine() order (horizontal-major).

    A combined court maps court x through its horizontal hypothesis alone and court y through
    its vertical one, so a foot is inside the court when it is inside both bands. The
    180-degree relabelling in canonicalise swaps the two halves, which the both-halves test
    ignores. The two versions round differently, so they could differ for a foot within
    rounding of a band edge or exactly on the halfway line.
    """
    rectified = rectify_feet(basis, feet_px)
    inside_x, _ = band_masks(horizontal.parameters[horizontal.retained], rectified[..., 0],
                             float(detector.X_COORDS.max()))
    inside_y, far_y = band_masks(vertical.parameters[vertical.retained], rectified[..., 1],
                                 float(detector.Y_COORDS.max()))
    # Per frame, one matrix product counts the foot samples inside both bands for every
    # (horizontal, vertical) pair. Counts are small whole numbers, so float32 is exact.
    by_frame_x = inside_x.transpose(1, 0, 2).astype(np.float32)  # (frames, horizontal, foot samples)
    anyone = (by_frame_x @ inside_y.transpose(1, 2, 0).astype(np.float32)) > 0  # (frames, horizontal, vertical)
    far = (by_frame_x @ far_y.transpose(1, 2, 0).astype(np.float32)) > 0
    near = (by_frame_x @ (inside_y & ~far_y).transpose(1, 2, 0).astype(np.float32)) > 0
    return anyone.mean(axis=0).reshape(-1), (far & near).mean(axis=0).reshape(-1)


def match_axis(
    basis: np.ndarray, axis: int, coordinates: np.ndarray, observations: assignment.Observations,
    size: tuple[int, int], settings: Settings, feet_px: np.ndarray | None = None,
) -> AxisMatches:
    ids, values, endpoints, details = offsets(basis, axis, observations, size, settings)
    observed_pairs = np.asarray(list(combinations(range(len(ids)), 2)), dtype=int).reshape(-1, 2)
    template_pairs = np.asarray(list(combinations(range(len(coordinates)), 2)), dtype=int).reshape(-1, 2)
    template_pairs = np.concatenate((template_pairs, template_pairs[:, ::-1]))
    observed = np.repeat(observed_pairs, len(template_pairs), axis=0)
    template = np.tile(template_pairs, (len(observed_pairs), 1))
    scale = (values[observed[:, 1]] - values[observed[:, 0]]) / (
        coordinates[template[:, 1]] - coordinates[template[:, 0]])
    shift = values[observed[:, 0]] - scale * coordinates[template[:, 0]]
    nonzero = np.abs(scale) > 1e-12
    parameters = np.column_stack((scale, shift))[nonzero]
    anchors = np.column_stack((ids[observed], template))[nonzero]
    scores = np.full(len(parameters), np.nan)
    matches = np.full((len(parameters), len(coordinates)), -1, dtype=int)
    supported = np.zeros(len(parameters), dtype=int)
    player_compatible = np.ones(len(parameters), dtype=bool)
    if feet_px is not None:
        rectified_feet = rectify_feet(basis, feet_px)[..., axis]
        for start in range(0, len(parameters), settings.batch):
            stop = start + settings.batch
            player_compatible[start:stop] = necessary_players(parameters[start:stop], rectified_feet, axis,
                                                              float(coordinates.max()))
    # Only player-compatible hypotheses can be retained, so only they are scored. The others
    # keep NaN scores, no matches and zero support.
    to_score = np.flatnonzero(player_compatible)
    for start in range(0, len(to_score), settings.batch):
        rows = to_score[start:start + settings.batch]
        batch_scores, group_indexes, counts = score_axes(parameters[rows], coordinates, endpoints, basis, axis)
        scores[rows] = batch_scores
        matches[rows] = np.where(group_indexes >= 0, ids[np.maximum(group_indexes, 0)], -1)
        supported[rows] = counts
    pattern = supported >= settings.minimum_matches
    eligible = np.flatnonzero(pattern & player_compatible)
    ordered = eligible[np.argsort(-scores[eligible], kind='stable')]
    seen = set()
    distinct = []
    for index in ordered:
        identity = tuple(matches[index])
        if identity in seen:
            continue
        seen.add(identity)
        distinct.append(int(index))
    retained = np.asarray(distinct[:settings.keep_axes], dtype=int)
    details.update({'pair_anchors': len(observed_pairs), 'enumerated': len(scale),
                    # Unmeasured under player pruning: player-incompatible hypotheses are never scored.
                    'zero_scale_excluded': int((~nonzero).sum()),
                    'pattern_supported': int(pattern.sum()) if feet_px is None else None,
                    'necessary_player_pruning': feet_px is not None, 'scored': len(to_score),
                    'pattern_and_players': len(eligible),
                    'distinct_assignments': len(distinct), 'axis_cap_excluded': max(0, len(distinct) - len(retained))})
    return AxisMatches(parameters, scores, matches, anchors, supported, player_compatible,
                       np.asarray(distinct, dtype=int), retained, details)


def combine(basis: np.ndarray, horizontal: AxisMatches, vertical: AxisMatches) -> tuple[np.ndarray, np.ndarray]:
    """Compose every retained axis pair; direction roles and signs remain explicit."""
    pairs = np.array([(first, second) for first in horizontal.retained for second in vertical.retained], dtype=int)
    pairs = pairs.reshape(-1, 2)
    maps = np.tile(np.eye(3), (len(pairs), 1, 1))
    maps[:, 0, 0], maps[:, 0, 2] = horizontal.parameters[pairs[:, 0]].T
    maps[:, 1, 1], maps[:, 1, 2] = vertical.parameters[pairs[:, 1]].T
    homographies = basis @ maps
    # The projective matrix scale is arbitrary; test depth consistently at court centre.
    centres = homographies @ np.append(detector.CORNER_COURT_M.mean(axis=0), 1)
    homographies *= np.where(centres[:, 2] < 0, -1., 1.)[:, None, None]
    return homographies, pairs
