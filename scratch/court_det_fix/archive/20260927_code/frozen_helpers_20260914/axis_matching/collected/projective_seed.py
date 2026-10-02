"""Match canonical court offsets along two supplied homogeneous perspective directions."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from vp_pruning import angular_residuals, normalisation

from experiments.annotator.independent_court import assignment, detector


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


def necessary_players(parameters: np.ndarray, rectified_feet: np.ndarray, axis: int, extent: float) -> np.ndarray:
    """Apply necessary parts of the existing joint player rule before an axis cap."""
    with np.errstate(divide='ignore', invalid='ignore'):
        position = (rectified_feet[None] - parameters[:, None, None, 1]) / parameters[:, None, None, 0] / extent
    inside = np.isfinite(position) & (position >= -.15) & (position <= 1.15)
    one = inside.any(axis=2).all(axis=1)
    if axis == 0:
        return one
    far = inside & (position < .5)
    near = inside & (position >= .5)
    return one & ((far.any(axis=2) & near.any(axis=2)).mean(axis=1) >= .5)


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
    scores = np.empty(len(parameters))
    matches = np.full((len(parameters), len(coordinates)), -1, dtype=int)
    supported = np.zeros(len(parameters), dtype=int)
    player_compatible = np.ones(len(parameters), dtype=bool)
    rectified_feet = None
    if feet_px is not None:
        homogeneous_feet = np.concatenate((feet_px, np.ones((*feet_px.shape[:-1], 1))), axis=2)
        mapped = homogeneous_feet @ np.linalg.inv(basis).T
        with np.errstate(divide='ignore', invalid='ignore'):
            rectified_feet = mapped[..., axis] / mapped[..., 2]
    for start in range(0, len(parameters), settings.batch):
        stop = start + settings.batch
        batch_scores, group_indexes, counts = score_axes(parameters[start:stop], coordinates, endpoints, basis, axis)
        scores[start:stop] = batch_scores
        matches[start:stop] = np.where(group_indexes >= 0, ids[np.maximum(group_indexes, 0)], -1)
        supported[start:stop] = counts
        if rectified_feet is not None:
            player_compatible[start:stop] = necessary_players(parameters[start:stop], rectified_feet, axis,
                                                              float(coordinates.max()))
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
                    'zero_scale_excluded': int((~nonzero).sum()), 'pattern_supported': int(pattern.sum()),
                    'necessary_player_pruning': feet_px is not None, 'pattern_and_players': len(eligible),
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
