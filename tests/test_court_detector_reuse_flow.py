"""Reuse and fallback share the prepared inputs and report the route taken."""
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from scratch.court_det_fix.court_detector import detect, reuse
from scratch.court_det_fix.court_detector.feet import FeetWindow
from scratch.court_det_fix.court_detector.inputs import (
    ViewInputs,
    same_frame_provenance,
)


@pytest.mark.parametrize('accepted', [False, True])
def test_reuse_success_skips_search_and_rejection_keeps_prepared_context(monkeypatch, accepted) -> None:
    context = SimpleNamespace(families=[object()])
    prepared, searched, saved = [], [], []

    def prepare(*args):
        prepared.append(args)
        return context

    monkeypatch.setattr(detect.feet, 'window_feet', lambda *args: FeetWindow([50], None, [50], [[[4., 5.]]]))
    monkeypatch.setattr(detect.search, 'seed_points', lambda family: [])
    detector = object.__new__(detect.CourtDetector)
    detector.switches = detect.Switches(timing=True)
    detector.live = SimpleNamespace(
        verifier=SimpleNamespace(view_context=prepare), runtime={}, court_model=None,
        line_template_source=SimpleNamespace(generate=lambda *args, **kwargs: SimpleNamespace(entries=[], metadata={})),
        prepared_measurements=lambda verifier: nullcontext(),
    )

    def search(actual_context, source, frame, laps, artefacts):
        searched.append(actual_context)
        return {}

    def score(view, actual_context, populations, templates, frame, laps, artefacts):
        assert actual_context is context
        saved.append(artefacts)
        return detect.CourtResult(view.view_id, None, 'no_gated_court', None, None)

    detector.search = search
    detector.score_and_choose = score
    corners = np.ones((4, 2))
    court = reuse.ReusedCourt('earlier', corners, .7, .95, .01) if accepted else None

    def try_reuse(known, actual_context, frame, live, **kwargs):
        assert actual_context is context
        return reuse.ReuseAttempt(court, {'rejection': None if accepted else 'alignment_mismatch'})

    monkeypatch.setattr(reuse, 'try_reuse', try_reuse)
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    view = ViewInputs('later', frame, 50, (0, 99), np.empty((0, 4)), np.empty((0, 4)),
                      same_frame_provenance('later', 50))
    result = detector.detect(view, object(), None, known_courts=[object()])
    assert len(prepared) == 1
    assert searched == ([] if accepted else [context])
    assert result.reused_from == ('earlier' if accepted else None)
    assert result.paint_score == (.7 if accepted else None)
    assert 'reuse' in result.stage_seconds
    if not accepted:
        assert saved[0]['reuse'] == [{'rejection': 'alignment_mismatch'}]
