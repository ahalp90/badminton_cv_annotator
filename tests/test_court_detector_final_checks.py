"""Final court checks and the exported corner convention."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from annotator.court_evidence import detected_court_info
from court_detector import detect
from court_detector.geometry import normalise_output_corners


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_output_order_preserves_pixels_and_the_consumers_near_far_coordinates(dtype) -> None:
    canonical = np.array([[12., 10.], [50., 10.], [70., 40.], [-5., 40.]], dtype=dtype)
    for half_turn in (0, 2):
        raw = np.roll(canonical, half_turn, axis=0)
        before = raw.copy()
        corners = normalise_output_corners(raw)
        np.testing.assert_array_equal(corners, canonical)
        np.testing.assert_array_equal(raw, before)
        assert corners.dtype == raw.dtype
        court = detected_court_info(corners)
        baseline_centres = np.array([corners[:2].mean(axis=0), corners[2:].mean(axis=0)], dtype=np.float32)
        normalised = cv2.perspectiveTransform(baseline_centres[None], court['H'])[0]
        np.testing.assert_allclose(normalised, [[.5, 0.], [.5, 1.]], atol=1e-6)
    # Equal baseline heights preserve the supplied order, including a quarter-turn.
    sideways = np.array([[10., 10.], [10., 40.], [20., 40.], [20., 10.]], dtype=dtype)
    np.testing.assert_array_equal(normalise_output_corners(sideways), sideways)


@pytest.mark.parametrize(('require_people', 'camera', 'players', 'reason'), [
    (True, False, False, 'refit_camera_implausible'),
    (False, False, False, 'refit_camera_implausible'),
    (True, True, False, 'refit_players_not_on_court'),
    (False, True, False, None),
    (True, True, True, None),
])
def test_final_refit_checks_the_camera_and_required_players(monkeypatch, require_people, camera, players, reason) -> None:
    corners = np.array([[10., 10.], [20., 10.], [30., 30.], [0., 30.]])
    corrected = {
        'valid': True, 'corners_native_px': corners.tolist(),
        'measurement': {'historical': {'historical_camera': camera, 'historical_fullcourt': players},
                        'paint_score': 0.8},
    }
    monkeypatch.setattr(detect.net_choice, 'net_rows', lambda *args, **kwargs: [])
    monkeypatch.setattr(detect.net_choice, 'choose', lambda *args, **kwargs: ('chosen', []))
    monkeypatch.setattr(detect.stripe_refit, 'refit_chosen', lambda *args, **kwargs: {'corrected': corrected})
    live = SimpleNamespace(verifier=None, runtime={})
    result = detect.choose_court(
        'view', {}, None, np.zeros((40, 40, 3), dtype=np.uint8), np.empty(0), live,
        detect.Switches(self_checks=False, require_people=require_people), detect.Laps(), {},
    )
    assert result.no_court_reason == reason
    assert result.chosen_key == 'chosen'
    if reason is None:
        np.testing.assert_array_equal(result.corners_native_px, corners)
    else:
        assert result.corners_native_px is None
