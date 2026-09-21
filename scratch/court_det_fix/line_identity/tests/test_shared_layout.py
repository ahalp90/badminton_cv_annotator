"""Path-resolution checks for the local and Carmack line-identity layouts."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import shared


def _make_project_markers(project_root: Path) -> None:
    (project_root / 'src').mkdir(parents=True)


def test_resolve_roots_for_full_checkout_layout(tmp_path: Path):
    project_root = tmp_path / 'checkout'
    court_det_fix = project_root / 'scratch' / 'court_det_fix'
    line_identity = court_det_fix / 'line_identity'
    _make_project_markers(project_root)

    repo, resolved_court_det_fix = shared._resolve_roots(line_identity)

    assert repo == project_root
    assert resolved_court_det_fix == court_det_fix


def test_resolve_roots_for_carmack_staging_layout(tmp_path: Path):
    experiment_root = tmp_path / 'paint_geometry_20260909'
    line_identity = experiment_root / 'line_identity'
    _make_project_markers(experiment_root)

    repo, resolved_court_det_fix = shared._resolve_roots(line_identity)

    assert repo == experiment_root
    assert resolved_court_det_fix == experiment_root


def test_add_helper_paths_includes_project_and_frozen_helpers(monkeypatch, tmp_path: Path):
    project_root = tmp_path / 'checkout'
    helpers = project_root / 'frozen_helpers_20260914'
    direction_agreement = project_root / 'direction_agreement'
    line_identity = project_root / 'line_identity'
    helper_paths = (
        project_root,
        project_root / 'src',
        helpers / 'vp_pruning',
        helpers / 'marking_diagnosis',
        helpers / 'axis_matching',
        helpers / 'automatic_axes',
        helpers / 'automatic_axes' / 'svd_fixed',
        direction_agreement,
        line_identity,
    )
    monkeypatch.setattr(shared, 'REPO', project_root)
    monkeypatch.setattr(shared, 'HELPERS', helpers)
    monkeypatch.setattr(shared, 'DIRECTION_AGREEMENT', direction_agreement)
    monkeypatch.setattr(shared, 'LINE_IDENTITY', line_identity)
    monkeypatch.setattr(sys, 'path', [])

    shared.add_helper_paths()

    assert {str(path) for path in helper_paths} <= set(sys.path)
