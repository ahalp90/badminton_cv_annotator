"""Synthetic checks at the automatic-direction pruning boundary."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from run_automatic import camera_direction_bound

from experiments.annotator.independent_court import detector


@pytest.mark.parametrize('homography', [
    np.array([[60., 0., 100.], [0., 25., 50.], [0., 0., 1.]]),
    np.array([[70., 12., 120.], [3., 40., 40.], [.01, .04, 1.]]),
    np.array([[10., 45., 70.], [55., 3., 50.], [.01, .02, 1.]]),
    np.array([[75., 8., -35.], [2., 65., 35.], [.005, .12, 1.]]),
])
def test_direction_bound_is_necessary_for_archived_camera(homography: np.ndarray) -> None:
    path = Path(__file__).resolve().parents[2] / '20260908/camera_diagnostic.py'
    spec = importlib.util.spec_from_file_location('archived_camera', path)
    assert spec is not None and spec.loader is not None
    archived = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(archived)
    corners, _ = detector.project(homography[None], detector.CORNER_COURT_M)
    error, *_ = archived.camera(corners[0], (960, 540))
    points = homography[:, :2].T
    bound = camera_direction_bound(points, (960, 540))
    assert bound <= error + 1e-6
    np.testing.assert_allclose(camera_direction_bound(points[::-1], (960, 540)), bound, atol=1e-12)
    np.testing.assert_allclose(camera_direction_bound(points * np.array([-3., .02])[:, None], (960, 540)),
                               bound, atol=1e-12)


@pytest.mark.parametrize('native_size', [(1920, 1080), (1366, 768)])
@pytest.mark.parametrize('directions', [
    np.array([[500., 100., 1.], [100., -200., 1.]]),
    np.array([[1., 0., 0.], [0., 1., 0.]]),
])
def test_generation_checks_bound_in_native_coordinates(
    monkeypatch: pytest.MonkeyPatch, native_size: tuple[int, int], directions: np.ndarray,
) -> None:
    from types import SimpleNamespace

    import run_automatic

    size = (960, 540)
    calls = []

    def record_bound(points: np.ndarray, actual_size: tuple[int, int]) -> float:
        calls.append((points.copy(), actual_size))
        return 1.

    monkeypatch.setattr(run_automatic, 'prepare', lambda source: (np.empty((0, 4)), (), size))
    monkeypatch.setattr(run_automatic, 'evaluate_pool', lambda *args: [])
    monkeypatch.setattr(run_automatic, 'camera_direction_bound', record_bound)
    # Empty groups are valid here: rejecting both pairs must bypass the matcher.
    monkeypatch.setattr(run_automatic.assignment, 'prepare_observations',
                        lambda segments, size: SimpleNamespace(fragment_ids=np.array([]), groups=[]))
    source = {'id': 'synthetic', 'dimensions': {'width': native_size[0], 'height': native_size[1]},
              'all_feet_px': [[[0., 0.], [1., 1.]]]}
    saved = {'working_size': list(size), 'settings': {'pencil_selection': 'coverage'},
             'estimator': {'points_working': directions.tolist()}}
    result = run_automatic.generate(source, saved, None, Path('.'))
    scale = np.append(np.asarray(native_size) / size, 1.)
    assert len(calls) == 2
    for (points, actual_size), order in zip(calls, ([0, 1], [1, 0]), strict=True):
        np.testing.assert_array_equal(points, directions[order] * scale)
        assert actual_size == native_size
    assert result['camera_bound_coordinate_space'] == 'native'
    assert result['entries'] == []
