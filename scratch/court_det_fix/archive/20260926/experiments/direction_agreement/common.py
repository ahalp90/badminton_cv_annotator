"""Shared paths, record I/O and frozen-control loading for the direction-agreement experiments.

Every path is relative to the experiment root: the remote root on the compute host, or
the local copy of the same tree when rendering saved results.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import lzma
import os
from pathlib import Path
from typing import Any

import numpy as np

# Frozen population in the packet's order, each with its input pack.
CASES: tuple[tuple[str, str], ...] = (
    ('gxBQ_window_00_frame_0', 'gx_extension/inputs.json.gz'),
    ('gxBQ_window_00_frame_5', 'gx_extension/inputs.json.gz'),
    ('am2_window_00_frame_150', 'gx_seed_20260913/marking_inputs.json.gz'),
    ('am2_window_01_frame_28019', 'gx_seed_20260913/marking_inputs.json.gz'),
    ('am3_window_00_frame_0', 'gx_seed_20260913/marking_inputs.json.gz'),
    ('shuttleset_03_scene_0017', 'broadcast_extension/inputs.json.gz'),
    ('shuttleset_03_scene_0019', 'broadcast_extension/inputs.json.gz'),
    ('shuttleset_03_scene_0016', 'broadcast_extension/inputs.json.gz'),
    ('shuttleset_21_scene_0020', 'broadcast_extension/inputs.json.gz'),
)
CASE_IDS = tuple(case_id for case_id, _ in CASES)
INPUT_PACKS = tuple(dict.fromkeys(pack for _, pack in CASES))

SAVED_ESTIMATORS = Path('vp_pruning_20260914/coverage/results')
GIVEN_FINITE = Path('axis_matching_20260914/given_finite')
AUTOMATIC = Path('automatic_axes_20260914')
BASELINE_STAGES = {
    'results': AUTOMATIC / 'results',
    'camera_first': AUTOMATIC / 'camera_first',
    'all_camera': AUTOMATIC / 'all_camera',
}
SVD_RECORDS = AUTOMATIC / 'svd_fixed/records'
LEGACY = Path('smoke/legacy')
# Three cases use the exact control corners of their earlier bank diagnoses.
BANK_CONTROLS = {
    'gxBQ_window_00_frame_0': AUTOMATIC / 'gx0_control/bank_diagnosis.json.gz',
    'gxBQ_window_00_frame_5': AUTOMATIC / 'gx5_bank_diagnosis.json.gz',
    'am2_window_01_frame_28019': AUTOMATIC / 'am2_far_bank_diagnosis.json.gz',
}
EXPERIMENT_DIR = Path('direction_agreement')
ARMS = ('B', 'M', 'R', 'MR')
MATCHER_ARMS = ('M', 'R')
STAGES = ('results', 'camera_first', 'all_camera')


def read(path: Path) -> Any:
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def write(path: Path, value: Any) -> None:
    """Write gzip JSON atomically so a partial file never passes for a complete record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_bytes(gzip.compress(json.dumps(value, allow_nan=False).encode(), mtime=0))
    os.replace(temporary, path)


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    """Write a gzip CSV atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerows(rows)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_bytes(gzip.compress(buffer.getvalue().encode(), mtime=0))
    os.replace(temporary, path)


def save_array(path: Path, array: np.ndarray) -> None:
    """Store a NumPy array as .npy.xz atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with lzma.open(temporary, 'wb', format=lzma.FORMAT_XZ, preset=9) as stream:
        np.save(stream, array)
    os.replace(temporary, path)


def load_array(path: Path) -> np.ndarray:
    with lzma.open(path, 'rb') as stream:
        return np.load(stream)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def run_dir(root: Path, run: str) -> Path:
    return root / EXPERIMENT_DIR / 'runs' / run


def code_md5(root: Path) -> dict[str, str]:
    """MD5 of every experiment module, so an output can prove which code produced it."""
    directory = root / EXPERIMENT_DIR
    return {path.name: md5(path) for path in sorted(directory.glob('*.py'))}


def load_sources(root: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """Return every case source and manual reference from the frozen input packs."""
    sources: dict[str, dict] = {}
    references: dict[str, dict] = {}
    for pack_name in INPUT_PACKS:
        pack = read(root / pack_name)
        sources.update({source['id']: source for source in pack['cases']})
        references.update(pack['references'])
    return sources, references


def native_size(source: dict) -> tuple[int, int]:
    return source['dimensions']['width'], source['dimensions']['height']


def frozen_control(case_id: str, root: Path, native: tuple[int, int], working: tuple[int, int]) -> dict:
    """Load the case's frozen control in working and native pixels with its exact provenance.

    The three bank-diagnosis cases already carry their control in working pixels. The
    other six divide the given record's native control corners by the native/working
    scale, as diagnose_directions.py does. The approval status is whatever the saved
    source string says; nothing here upgrades a manual reference to an approved court.
    """
    scale = np.asarray(native, dtype=float) / np.asarray(working, dtype=float)
    if case_id in BANK_CONTROLS:
        record_path = root / BANK_CONTROLS[case_id]
        record = read(record_path)
        corners_working = np.asarray(record['control_corners_working_px'], dtype=float)
        source = record['control_source']
    else:
        record_path = root / GIVEN_FINITE / f'{case_id}.json.gz'
        record = read(record_path)
        corners_working = np.asarray(record['control_corners_px'], dtype=float) / scale
        source = record['given_direction_source']
    assert record['case_id'] == case_id, (record['case_id'], case_id)
    return {
        'case_id': case_id,
        'control_source': source,
        'visually_approved': source.startswith('visually_approved'),
        'record': str(record_path.relative_to(root)),
        'record_md5': md5(record_path),
        'corners_working_px': corners_working.tolist(),
        'corners_native_px': (corners_working * scale).tolist(),
        'native_size': list(native),
        'working_size': list(working),
    }
