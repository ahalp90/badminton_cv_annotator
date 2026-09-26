"""Shared local paths, loaders and the frozen population for the line-identity experiments.

Paths are resolved from this module's experiment root, so the committed scripts carry no private
paths. Remote work goes through run_remote.sh and the gitignored paths.local.sh.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from experiments.annotator.independent_court.case_provenance import CaseProvenance


def _resolve_roots(line_identity: Path) -> tuple[Path, Path]:
    """Return the project root and experiment root for either supported checkout layout."""
    court_det_fix = line_identity.resolve().parent
    for project_root in (court_det_fix, *court_det_fix.parents):
        if (project_root / 'src').is_dir():
            return project_root, court_det_fix
    raise RuntimeError(f'Could not locate project root above {court_det_fix}; expected src/')


LINE_IDENTITY = Path(__file__).resolve().parent
REPO, COURT_DET_FIX = _resolve_roots(LINE_IDENTITY)
# The frozen helper scripts the experiments import: the tracked copy of the player_guided/20260914
# tree the direction experiment calls L (code only; see its README.md).
HELPERS = COURT_DET_FIX / 'frozen_helpers_20260914'
# Frozen view evidence: source packs, native frames and baseline direction/matcher records.
FROZEN_VIEWS = COURT_DET_FIX / 'frozen_views'
DIRECTION_AGREEMENT = COURT_DET_FIX / 'direction_agreement'
DIRECTION_RUN = DIRECTION_AGREEMENT / 'runs/direction_agreement_20260915_144900'
SAVED_ESTIMATORS = FROZEN_VIEWS / 'baseline_directions'
BASELINE_GENERATION = FROZEN_VIEWS / 'baseline_generation'
LEGACY_ZONE = HELPERS / 'legacy'

PACKS = {
    'gx': FROZEN_VIEWS / 'packs/gx_extension_inputs.json.gz',
    'amateur': FROZEN_VIEWS / 'packs/marking_refit_inputs.json.gz',
    'broadcast': FROZEN_VIEWS / 'packs/broadcast_extension_inputs.json.gz',
}
# The nine regression views in the direction experiment's order, each with its pack and short label.
REGRESSION_CASES: tuple[tuple[str, str, str], ...] = (
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
UNUSED_CASES: tuple[tuple[str, str, str], ...] = (
    ('gxBQ_window_00_frame_689', 'gx', 'gxBQ_window_00_frame_689'),
    ('gxBQ_window_01_frame_5111', 'gx', 'gxBQ_window_01_frame_5111'),
    ('gxBQ_window_02_frame_5766', 'gx', 'gxBQ_window_02_frame_5766'),
    ('gxBQ_window_03_frame_77876', 'gx', 'gxBQ_window_03_frame_77876'),
    ('gxBQ_window_04_frame_86088', 'gx', 'gxBQ_window_04_frame_86088'),
    ('yellow_short_frame_14', 'amateur', 'yellow_short_frame_14'),
    ('letterboxed_short_frame_45', 'amateur', 'letterboxed_short_frame_45'),
    ('centre_short_frame_36', 'amateur', 'centre_short_frame_36'),
    ('am1_window_00_frame_54', 'amateur', 'am1_window_00_frame_54'),
    ('am3_window_01_frame_10514', 'amateur', 'am3_window_01_frame_10514'),
    ('am4_window_00_frame_0', 'amateur', 'am4_window_00_frame_0'),
    ('am4_window_01_frame_13782', 'amateur', 'am4_window_01_frame_13782'),
    ('shuttleset_03_scene_0029', 'broadcast', 'shuttleset_03_scene_0029'),
    ('shuttleset_03_scene_0034', 'broadcast', 'shuttleset_03_scene_0034'),
    ('shuttleset_03_scene_0038', 'broadcast', 'shuttleset_03_scene_0038'),
    ('shuttleset_21_scene_0000', 'broadcast', 'shuttleset_21_scene_0000'),
    ('shuttleset_21_scene_0010', 'broadcast', 'shuttleset_21_scene_0010'),
    ('shuttleset_21_scene_0039', 'broadcast', 'shuttleset_21_scene_0039'),
)
ALL_CASES: tuple[tuple[str, str, str], ...] = REGRESSION_CASES + UNUSED_CASES
REGRESSION_CASE_IDS = tuple(case_id for case_id, _, _ in REGRESSION_CASES)
UNUSED_CASE_IDS = tuple(case_id for case_id, _, _ in UNUSED_CASES)
ALL_CASE_IDS = tuple(case_id for case_id, _, _ in ALL_CASES)
CASES: tuple[tuple[str, str, str], ...] = REGRESSION_CASES
CASE_IDS = REGRESSION_CASE_IDS
LABELS = {case_id: label for case_id, _, label in ALL_CASES}
PACK_OF = {case_id: pack for case_id, pack, _ in ALL_CASES}
FLOAT_ATOL = 1e-12


def add_helper_paths() -> None:
    """Put the repository and the frozen helper modules on sys.path, this folder first."""
    for path in (
        REPO, REPO / 'src', HELPERS / 'vp_pruning', HELPERS / 'marking_diagnosis',
        HELPERS / 'axis_matching', HELPERS / 'automatic_axes', HELPERS / 'automatic_axes/svd_fixed',
        DIRECTION_AGREEMENT, LINE_IDENTITY,
    ):
        sys.path.insert(0, str(path))


def read(path: Path) -> Any:
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def write(path: Path, value: Any) -> None:
    """Write gzip JSON with a fixed header time and an atomic replace, so identical content hashes identically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_bytes(gzip.compress(json.dumps(value, allow_nan=False).encode(), mtime=0))
    os.replace(temporary, path)


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


@lru_cache(maxsize=3)
def _load_pack_provenance(pack_path: Path) -> Mapping[str, CaseProvenance]:
    """Load one frozen pack's validated provenance mapping once per process."""
    from experiments.annotator.independent_court.case_provenance import (
        load_frozen_case_provenance,
    )

    return load_frozen_case_provenance(pack_path)


def case_provenance(case_id: str) -> CaseProvenance:
    """Return the validated image/box provenance for one frozen line case."""
    pack_name = PACK_OF[case_id]
    return _load_pack_provenance(PACKS[pack_name])[case_id]


def require_same_image_boxes(case: CaseProvenance) -> CaseProvenance:
    """Apply the central guard without importing the contract before helper paths are ready."""
    from experiments.annotator.independent_court.case_provenance import (
        require_same_image_boxes as require,
    )

    return require(case)


def frame_path(source: dict) -> Path:
    """The native frame the pack's fragments were detected on (frozen_views/frames keeps the packs' layout)."""
    if source['id'].startswith('gxBQ'):
        return FROZEN_VIEWS / 'frames/gx' / source['image']
    if source['id'].startswith('shuttleset'):
        return FROZEN_VIEWS / 'frames/original' / source['image']
    video = source['id'].split('_')[0]
    frame = int(source['id'].rsplit('_', 1)[1])
    provenance = case_provenance(source['id'])
    if provenance.image_kind.value != 'source_frame' or provenance.image_frame_indices != (frame,):
        raise ValueError(
            f"{source['id']}: amateur frame path uses frame {frame}, but provenance identifies "
            f"{provenance.image_kind.value} frames {provenance.image_frame_indices}"
        )
    return FROZEN_VIEWS / 'frames/amateur' / video / f'frame_{frame:08d}.png'


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
