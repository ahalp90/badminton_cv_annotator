"""Reuse and fallback share the prepared inputs and report the route taken."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from court_detector import detect, reuse
from court_detector.feet import FeetWindow
from court_detector.inputs import (
    ViewInputs,
    same_frame_provenance,
)


@pytest.mark.parametrize(('require_people', 'full', 'direction_search'), [
    (False, False, False), (False, True, True), (True, False, True), (True, True, True),
])
def test_frame_search_breadth_keeps_templates_and_the_player_policy(monkeypatch, require_people, full,
                                                                  direction_search) -> None:
    detector = object.__new__(detect.CourtDetector)
    detector.switches = detect.Switches(require_people=require_people, full_no_people_search=full)
    searched, scored = [], []
    template = {'origin_key': 'template'}
    detector.live = SimpleNamespace(
        runtime={}, court_model=None, verifier=None, prepared_measurements=lambda verifier: nullcontext(),
        line_template_source=SimpleNamespace(generate=lambda *args, **kwargs:
                                              SimpleNamespace(entries=[template], metadata={})),
    )
    monkeypatch.setattr(detect.search, 'seed_points', lambda family: [])
    detector.search = lambda *args: searched.append(True) or {'all_lines': ['all'], 'painted_lines': ['painted']}

    def choose(view, context, populations, templates, frame, laps, artefacts):
        scored.append((populations, templates))
        return detect.CourtResult('frame', None, 'no_gated_court', None, None)

    detector.score_and_choose = choose
    prepared = SimpleNamespace(context=SimpleNamespace(families=[object()]), view=None, source={}, native_frame=None)
    detector.search_and_choose(prepared, detect.Laps(), {})
    assert bool(searched) is direction_search
    populations, templates = scored[0]
    assert templates == [template]
    assert populations == ({'all_lines': ['all'], 'painted_lines': ['painted']} if direction_search
                           else {'all_lines': [], 'painted_lines': []})


@pytest.mark.parametrize('accepted', [False, True])
@pytest.mark.parametrize(('switches', 'observations'), [
    (detect.Switches(timing=True), [[4., 5.], [7., 8.]]),
    (detect.Switches(timing=True, require_people=False), [None, None]),
    (detect.Switches(timing=True, artefacts_dir=Path('unused')), [None, None]),
])
def test_reuse_success_skips_search_and_rejection_keeps_prepared_context(monkeypatch, accepted,
                                                                      switches, observations) -> None:
    context = SimpleNamespace(families=[object()])
    prepared, searched, saved = [], [], []

    def prepare(*args):
        prepared.append(args)
        return context

    monkeypatch.setattr(detect.feet, 'window_feet',
                        lambda *args: FeetWindow([50], None, [50], [observations]))
    monkeypatch.setattr(detect.search, 'seed_points', lambda family: [])
    detector = object.__new__(detect.CourtDetector)
    detector.switches = switches
    detector.live = SimpleNamespace(
        verifier=SimpleNamespace(view_context=prepare, write_json_gz=lambda *args: None), runtime={}, court_model=None,
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
        assert kwargs['alignment_image'] is alignment_image
        return reuse.ReuseAttempt(court, {'rejection': None if accepted else 'alignment_mismatch'})

    monkeypatch.setattr(reuse, 'try_reuse', try_reuse)
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    alignment_image = np.full((540, 960), 100, dtype=np.uint8)
    view = ViewInputs('later', frame, 50, (0, 100), np.empty((0, 4)), np.empty((0, 4)),
                      same_frame_provenance('later', 50), alignment_image)
    result = detector.detect(view, object(), None, known_courts=[object()])
    assert len(prepared) == 1
    assert searched == ([context] if not accepted and switches.require_people else [])
    assert result.reused_from == ('earlier' if accepted else None)
    assert result.paint_score == (.7 if accepted else None)
    # A view left without a court hands on its prepared inputs, for a view pool to share a court with.
    assert (result.prepared is None) is accepted
    assert 'reuse' in result.stage_seconds
    if not accepted:
        assert saved[0]['reuse'] == [{'rejection': 'alignment_mismatch'}]


def test_insufficient_player_counts_skip_searches_and_keep_a_prepared_receiver(monkeypatch) -> None:
    monkeypatch.setattr(detect.feet, 'window_feet',
                        lambda *args: FeetWindow([50], None, [50], [[None, None]]))
    detector = object.__new__(detect.CourtDetector)
    detector.switches = detect.Switches(timing=True)
    detector.live = SimpleNamespace(verifier=SimpleNamespace(view_context=lambda *args: SimpleNamespace()))
    searched, endpoints = [], []

    def search_and_choose(prepared, laps, artefacts):
        searched.append(prepared.source['all_feet_px'])
        laps.lap('search')
        return detect.CourtResult(prepared.view.view_id, None, 'no_gated_court', None, None)

    detector.search_and_choose = search_and_choose
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    view = ViewInputs('empty', frame, 50, (0, 100), np.empty((0, 4)), np.empty((0, 4)),
                      same_frame_provenance('empty', 50))
    result = detector.detect(view, object(), None, endpoint_views=lambda: endpoints.append(True) or [])
    assert searched == endpoints == []
    assert result.no_court_reason == 'no_gated_court'
    assert result.corners_native_px is None
    assert result.prepared is not None
    assert result.prepared.view is view
    assert result.prepared.source['all_feet_px'] == [[None, None]]
    assert list(result.stage_seconds) == ['feet', 'context']
