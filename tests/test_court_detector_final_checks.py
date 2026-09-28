"""Final stripe corrections must still satisfy the candidate admission rules."""
from types import SimpleNamespace

import numpy as np
import pytest

from court_detector import detect


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
