"""Match each pair of line directions to possible court layouts."""


from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import geometry as detector
from . import line_observations as assignment
from .candidate_geometry import continuous_support, geometry
from .line_matching import (
    AxisMatches,
    Settings,
    basis_for,
    combine,
    joint_player_fractions,
    match_axis,
)

FAR_HORIZON_DIAGONALS = 10.


def finite_scores(
    homographies: np.ndarray, observations: assignment.Observations,
    axes: tuple[AxisMatches, AxisMatches], size: tuple[int, int],
) -> np.ndarray:
    """Score canonically oriented courts using families from the current directions."""
    families = []
    for matched in axes:
        groups = matched.diagnostics['retained_group_ids']
        members = np.concatenate([observations.groups[index] for index in groups])
        families.append(observations.segments[members].reshape(-1, 4))
    maps = detector._distance_maps((families[0], families[1]), size)
    scores = np.empty(len(homographies))
    for start in range(0, len(homographies), 256):
        scores[start:start + 256] = continuous_support(homographies[start:start + 256], maps, size)
    return scores


def canonicalise(homographies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    corners, _ = detector.project(homographies, detector.CORNER_COURT_M)
    rotate = corners[:, :2, 1].mean(axis=1) > corners[:, 2:, 1].mean(axis=1)
    symmetry = np.array([[-1., 0., detector.CORNER_COURT_M[:, 0].max()],
                         [0., -1., detector.CORNER_COURT_M[:, 1].max()], [0., 0., 1.]])
    homographies = homographies.copy()
    homographies[rotate] = homographies[rotate] @ symmetry
    return homographies, rotate


def pack_axis(matches: AxisMatches) -> dict:
    entries = []
    for index in matches.retained:
        entries.append({'axis_id': int(index), 'parameters': matches.parameters[index].tolist(),
                        'score': float(matches.scores[index]), 'matched_groups': matches.matches[index].tolist(),
                        'anchors': matches.anchors[index].tolist(), 'support_count': int(matches.supported[index])})
    return {'diagnostics': matches.diagnostics, 'entries': entries}


@dataclass
class RoleProposals:
    record: dict
    basis: np.ndarray | None
    axes: tuple[AxisMatches, AxisMatches] | None
    candidates: list[detector.Candidate]
    # One row per candidate, in candidates order, for detail(): the two axis hypothesis IDs,
    # whether canonicalise turned the court 180 degrees, the mean axis score and the homography.
    axis_ids: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=int))
    rotated: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    axis_scores: np.ndarray = field(default_factory=lambda: np.empty(0))
    homographies: np.ndarray = field(default_factory=lambda: np.empty((0, 3, 3)))
    # pregate copy: every combined court in transforms order (working px, float32), the geometry
    # mask, the player mask and the two player fractions. Empty when the basis fails.
    combined_corners: np.ndarray = field(default_factory=lambda: np.empty((0, 4, 2), dtype=np.float32))
    valid: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    usable: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    player_any: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    player_both_halves: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))

    def detail(self, position: int) -> dict:
        """One candidate's provenance, built on demand because most candidates never reach a shortlist."""
        first, second = self.axis_ids[position]
        return {'axis_ids': [int(first), int(second)], 'rotated_180': bool(self.rotated[position]),
                'axis_score': float(self.axis_scores[position]),
                'homography_working': self.homographies[position].tolist()}


def horizon(points: np.ndarray, size: tuple[int, int]) -> np.ndarray | None:
    """The line through the pair's two vanishing points, or None when it is too far away to test.

    Every court the pair builds shares this horizon. Working pixels keep the frame's aspect
    ratio, so its tilt and sides match the native frame's.
    """
    line = np.cross(points[0], points[1])
    width, height = size
    normal_length = np.hypot(line[0], line[1])
    if normal_length == 0:
        return None
    centre_distance = abs(line @ [width / 2, height / 2, 1.]) / normal_length
    return None if centre_distance > FAR_HORIZON_DIAGONALS * np.hypot(width, height) else line


def below_horizon(points: np.ndarray, corners: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Courts whose four corners lie below the pair's horizon, as the floor does for an upright camera.

    A court above it needs an upside-down camera. Callers skip pairs with a steep horizon
    first, so "below" is well defined. Every court passes when the horizon is too far away.

    :param corners: (courts, 4, 2) working px.
    """
    line = horizon(points, size)
    if line is None:
        return np.ones(len(corners), dtype=bool)
    # Image y grows downwards, so the side the y coefficient points to is below the horizon.
    side = np.sign(line[1]) * (corners @ line[:2] + line[2])
    return (side > 0).all(axis=1)


def propose_role(
    points: np.ndarray, observations: assignment.Observations, feet: np.ndarray,
    size: tuple[int, int], settings: Settings,
    player_pruning: bool = True, combined_ranking: str = 'finite', upright_only: bool = False,
) -> RoleProposals:
    """Generate one ordered direction role without reference geometry or labels.

    :param upright_only: also count courts above the pair's horizon as invalid geometry.
    """
    basis, details = basis_for(points, size, settings)
    record = {'basis_status': details}
    if basis is None:
        return RoleProposals(record, None, None, [])
    axis_feet = feet if player_pruning else None
    horizontal = match_axis(basis, 0, detector.X_COORDS, observations, size, settings, axis_feet)
    vertical = match_axis(basis, 1, detector.Y_COORDS, observations, size, settings, axis_feet)
    transforms, axis_pairs = combine(basis, horizontal, vertical)
    transforms, rotated = canonicalise(transforms)
    valid, corners = geometry(transforms, size)
    if upright_only:
        valid = valid & below_horizon(points, corners, size)
    one, two = joint_player_fractions(basis, horizontal, vertical, feet)
    usable = valid & (one == 1) & (two >= .5)
    record.update({'basis_working': basis.tolist(), 'axes': [pack_axis(horizontal), pack_axis(vertical)],
                   'combined': len(transforms), 'geometry_valid': int(valid.sum()),
                   'geometry_players': int(usable.sum())})
    usable_ids = np.flatnonzero(usable)
    axis_ids = axis_pairs[usable_ids]
    axis_scores = (horizontal.scores[axis_ids[:, 0]] + vertical.scores[axis_ids[:, 1]]) / 2
    finite = finite_scores(transforms[usable], observations, (horizontal, vertical), size) if (
        combined_ranking == 'finite' and len(usable_ids)) else None
    shortlist_scores = axis_scores if finite is None else finite
    candidates = [detector.Candidate(corners[index], float(score), (0., 0.), (0, 0))
                  for index, score in zip(usable_ids, shortlist_scores, strict=True)]
    return RoleProposals(record, basis, (horizontal, vertical), candidates,
                         axis_ids, rotated[usable_ids], axis_scores, transforms[usable_ids],
                         np.asarray(corners, dtype=np.float32).reshape(-1, 4, 2),
                         np.asarray(valid, dtype=bool), np.asarray(usable, dtype=bool),
                         np.asarray(one, dtype=np.float32), np.asarray(two, dtype=np.float32))
