"""Check and rank courts found by matching line directions."""


from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np

from . import geometry as detector
from . import line_observations as assignment
from . import proposals as run_given
from .candidate_geometry import retain
from .court_checks import gate_evidence
from .line_matching import Settings
from .prepare_lines import prepare
from .proposals import propose_role

__all__ = ['CAMERA_ERROR_LIMIT', 'CAMERA_ROUNDING_MARGIN', 'KEEP_COURTS', 'Settings', 'camera_direction_bound', 'detector', 'evaluate_pool', 'horizon_tilt_deg', 'prepare', 'propose_role', 'retain', 'select_pool']


KEEP_COURTS = 256


CAMERA_ERROR_LIMIT = .1


CAMERA_ROUNDING_MARGIN = 1e-6


def camera_direction_bound(points: np.ndarray, size: tuple[int, int]) -> float:
    """Lower-bound the archived camera error without choosing court scale or position."""
    width, height = size
    focals = np.geomspace(.4 * width, 4 * width, 200)
    axes = np.broadcast_to(points, (len(focals), 2, 3)).copy()
    axes[:, :, :2] -= np.array([width / 2, height / 2]) * axes[:, :, 2:]
    axes[:, :, :2] /= focals[:, None, None]
    norms = np.linalg.norm(axes, axis=2)
    cosine = np.sum(axes[:, 0] * axes[:, 1], axis=1) / np.prod(norms, axis=1)
    return float(np.abs(cosine).min())


def horizon_tilt_deg(points: np.ndarray, size: tuple[int, int]) -> float | None:
    """How far the pair's horizon tilts from level, 0 to 90 degrees: the roll a camera would need.

    None when the horizon is too far away to measure (run_given.horizon).
    """
    line = run_given.horizon(points, size)
    if line is None:
        return None
    return float(np.degrees(np.arctan2(abs(line[0]), abs(line[1]))))


def select_pool(candidates: list[detector.Candidate]) -> list[detector.Candidate]:
    settings = replace(detector.DEFAULT_SETTINGS, keep_candidates=KEEP_COURTS, distinct_corner_distance=2.)
    return retain(candidates, settings)


def evaluate_pool(
    source: dict, shortlist: list[dict], observations: assignment.Observations,
    size: tuple[int, int], segments: np.ndarray, families: tuple, zone: object,
) -> list[dict]:
    """Measure the unchanged camera, floor and player gates for saved courts.

    The archived research pool also scored each court's stripes and paint profile for
    its own rankings; nothing in the detector reads them.
    """
    scale = np.array([source['dimensions']['width'], source['dimensions']['height']]) / size
    maps = detector._distance_maps(detector._wide_line_families(segments), size)
    entries = []
    started = perf_counter()
    for position, details in enumerate(shortlist):
        if len(shortlist) > 256 and position % 512 == 0:
            print(source['id'], 'full evidence', position, 'of', len(shortlist),
                  'seconds', perf_counter() - started, flush=True)
        corners = np.asarray(details['corners_px'])
        entry = {**details, 'corners_px': corners.tolist(), 'shortlist_score': details['shortlist_score']}
        entry['gates'] = gate_evidence(corners, source, scale, size, families, maps, zone)
        entries.append(entry)
    return entries
