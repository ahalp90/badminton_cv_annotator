"""Shared local paths, loaders and the frozen population for the line-identity experiments.

Everything here resolves inside the repository checkout, so the committed scripts carry no
private paths. Remote work goes through run_remote.sh and the gitignored paths.local.sh.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
CHECKS = REPO / 'scratch/court_det_fix/worklog/checks/independent'
# The frozen helper sources the direction experiment imports (the packet's local tree L).
HELPERS = CHECKS / 'player_guided/20260914'
DIRECTION_AGREEMENT = REPO / 'scratch/court_det_fix/direction_agreement'
DIRECTION_RUN = DIRECTION_AGREEMENT / 'runs/direction_agreement_20260915_144900'
SAVED_ESTIMATORS = HELPERS / 'vp_pruning/coverage/results'
LEGACY_ZONE = CHECKS / 'player_guided/20260908'

PACKS = {
    'gx': CHECKS / 'player_guided/20260909/gx_extension/inputs.json.gz',
    'amateur': CHECKS / 'player_guided/20260908/marking_refit/marking_inputs.json.gz',
    'broadcast': CHECKS / 'player_guided/20260909/broadcast_extension/inputs.json.gz',
}
# The nine frozen views in the direction experiment's order, each with its pack and short label.
CASES: tuple[tuple[str, str, str], ...] = (
    ('gxBQ_window_00_frame_0', 'gx', 'GX0'),
    ('gxBQ_window_00_frame_5', 'gx', 'GX5'),
    ('am2_window_00_frame_150', 'amateur', 'Am2-150'),
    ('am2_window_01_frame_28019', 'amateur', 'Am2-28019'),
    ('am3_window_00_frame_0', 'amateur', 'Am3-0'),
    ('shuttleset_03_scene_0017', 'broadcast', 'SS03-17'),
    ('shuttleset_03_scene_0019', 'broadcast', 'SS03-19'),
    ('shuttleset_03_scene_0016', 'broadcast', 'SS03-16'),
    ('shuttleset_21_scene_0020', 'broadcast', 'SS21-20'),
)
CASE_IDS = tuple(case_id for case_id, _, _ in CASES)
LABELS = {case_id: label for case_id, _, label in CASES}
PACK_OF = {case_id: pack for case_id, pack, _ in CASES}
FLOAT_ATOL = 1e-12


def add_helper_paths() -> None:
    """Put the repository and the frozen helper modules on sys.path, this folder first."""
    for path in (
        REPO, REPO / 'src', HELPERS / 'vp_pruning', HELPERS / 'marking_diagnosis',
        HELPERS / 'axis_matching', HELPERS / 'automatic_axes', HELPERS / 'automatic_axes/svd_fixed',
        DIRECTION_AGREEMENT, Path(__file__).resolve().parent,
    ):
        sys.path.insert(0, str(path))


def read(path: Path) -> Any:
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt') as stream:
        json.dump(value, stream, allow_nan=False)


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_source(case_id: str) -> dict:
    """The case's entry in its frozen input pack."""
    pack = read(PACKS[PACK_OF[case_id]])
    for source in pack['cases']:
        if source['id'] == case_id:
            return source
    raise KeyError(case_id)


def frame_path(source: dict) -> Path:
    """The native frame the pack's fragments were detected on."""
    if source['id'].startswith('gxBQ'):
        return CHECKS / 'player_guided/20260909/gx_extension/people' / source['image']
    if source['id'].startswith('shuttleset'):
        return CHECKS / 'inputs/original' / source['image']
    video = source['id'].split('_')[0]
    frame = int(source['id'].rsplit('_', 1)[1])
    return CHECKS / 'inputs/amateur' / video / f'frame_{frame:08d}.png'


def load_estimator(case_id: str) -> dict:
    """The saved baseline direction record (coverage selection) the matcher ran from."""
    saved = read(SAVED_ESTIMATORS / f'{case_id}.json.gz')
    assert saved['case_id'] == case_id
    return saved


def load_direction_record(stage: str, case_id: str) -> dict:
    """One direction-experiment record: e0 membership, e2 frozen arms, e3 control fits."""
    record = read(DIRECTION_RUN / stage / f'{case_id}.json.gz')
    assert record['case_id'] == case_id
    return record


def control_corners(case_id: str) -> tuple[np.ndarray, dict]:
    """The frozen control in working pixels, read from the direction experiment's fit record."""
    control = load_direction_record('e3', case_id)['control']
    return np.asarray(control['corners_working_px'], dtype=float), control


def arm_points(case_id: str, arm: str) -> np.ndarray:
    """The 16 selected directions of one arm in working homogeneous coordinates."""
    if arm == 'B':
        return np.asarray(load_estimator(case_id)['estimator']['points_working'], dtype=float)
    return np.asarray(load_direction_record('e2', case_id)['arms'][arm]['points_working'], dtype=float)


def feet_working(source: dict, size: tuple[int, int]) -> np.ndarray:
    """Player feet per pose sample in working pixels, NaN where a foot is missing (as the matcher builds them)."""
    scale = np.asarray([source['dimensions']['width'], source['dimensions']['height']], dtype=float) / size
    return np.asarray([[[np.nan, np.nan] if foot is None else foot for foot in frame]
                       for frame in source['all_feet_px']], dtype=float) / scale


def corner_errors(corners: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Maximum corner distance allowing the 180-degree relabelling (copy of projective_seed.corner_errors)."""
    direct = np.linalg.norm(corners - reference, axis=-1).max(axis=-1)
    rotated = np.linalg.norm(corners - reference[[2, 3, 0, 1]], axis=-1).max(axis=-1)
    return np.minimum(direct, rotated)
