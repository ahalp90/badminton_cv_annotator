"""Smoke coverage for the public annotator video composition."""
import numpy as np
import pandas as pd
import pytest

import annotator.run_video as run_video_module
import annotator.rally_segmentation as stage8_seg
from annotator.calibration.gt_scoring import write_geometric_verdicts_csv
from annotator.point_winner import GeometricVerdictRow, Half, Landing, LandingFilterOptions, Verdict
from annotator.fps_constants import scale_for_fps
from annotator.rally_segmentation import CourtBox, ServeStartClose, ServeStartMode, StickyResult
from annotator.run_video import AnnotatorResult, build_serve_options, run_video, scoring_filter
from annotator.types import ContactCandidate, ServeStartConfig


def test_run_video_no_play_returns_empty_result():
    video_id = 1
    resolution = (1920.0, 1080.0)
    track = np.zeros((300, 3), dtype=np.float64)
    bboxes = np.zeros((300, 1, 4), dtype=np.float32)
    scores = np.zeros((300, 1), dtype=np.float32)
    kps = np.zeros((300, 1, 17, 2), dtype=np.float32)
    ndet = np.zeros(300, dtype=np.int64)
    dead = np.zeros(300, dtype=bool)
    court_info = {
        "H": np.eye(3),
        "border_L": 0.0,
        "border_R": 1280.0,
        "border_U": 0.0,
        "border_D": 720.0,
    }
    homo_df = pd.DataFrame(
        {
            "upleft_x": [0.0], "upright_x": [1280.0],
            "downleft_x": [0.0], "downright_x": [1280.0],
            "upleft_y": [0.0], "upright_y": [0.0],
            "downleft_y": [720.0], "downright_y": [720.0],
        },
        index=[video_id],
    )
    gate_resolution_table = pd.DataFrame(
        {"width": [1920.0], "height": [1080.0]}, index=[str(video_id)],
    )

    result = run_video(
        track, bboxes, scores, kps, ndet,
        fps=25.0,
        landing_options=LandingFilterOptions(7, 0.004, 5, 7, 0.75),
        court_box=CourtBox(
            x_range=(635.0, 1316.0), y_range=(254.0, 1030.0),
            height_band=(84.0, 336.0), mid_band=(642.0, 642.0),
        ),
        net_band=(664.6, 703.7),
        resolution=resolution,
        video_id=video_id,
        court_info=court_info,
        homo_df=homo_df,
        gate_court_info={str(video_id): court_info},
        gate_resolution_table=gate_resolution_table,
        dead_mask=dead,
        **_default_scene_inputs(len(track)),
    )

    assert result == AnnotatorResult([], [], [], {}, [], [], [], [], {}, {}, {}, {}, [])


def test_scoring_filter_keeps_only_unfailed_unsuppressed_rows():
    rows = [
        ContactCandidate(0, 1, None, True, False),
        ContactCandidate(0, 2, None, True, True),
        ContactCandidate(0, 3, None, False, False),
        ContactCandidate(0, 4, None, None, None),
    ]

    assert scoring_filter(rows) == [rows[0], rows[3]]


def test_build_serve_options_wires_sticky_setup() -> None:
    n_frames = 40
    sticky = StickyResult(
        distances=np.full(n_frames, np.nan), picks=np.full((n_frames, 2), -1),
        standing_count=np.full(n_frames, 2),
        ankle_pos=np.full((n_frames, 2, 2), (0.2, 0.3)),
        bbox_height=np.full((n_frames, 2), 100.0),
        distances_per_slot=np.full((n_frames, 2), 0.1), analysed=np.ones(n_frames, dtype=bool),
    )
    resolution = (1920.0, 1080.0)
    options = build_serve_options(
        ServeStartConfig(0.75, ServeStartMode.TRIM, stillness_threshold_bh=0.2),
        sticky, scale_for_fps(30.0), resolution,
    )
    assert options.dist is None
    assert options.setup is not None
    assert options.stillness_threshold_bh == 0.2
    assert options.lookback_frames == 25
    assert options.stillness_window_frames == 15
    assert options.setup.top_height[0] == pytest.approx(100.0 / 1080.0)

    with pytest.raises(ValueError, match='serve_start.close is unsupported with BACK_FILL'):
        build_serve_options(
            ServeStartConfig(0.1, ServeStartMode.TRIM, close=ServeStartClose.BURST),
            sticky, scale_for_fps(30.0), resolution,
        )


def test_run_video_injected_spans_bypass_natural_span_finding(monkeypatch):
    inputs = _synthetic_inputs()
    injected = [(10, 20)]
    monkeypatch.setattr(
        stage8_seg, 'find_rally_spans',
        lambda *args, **kwargs: pytest.fail('natural span finding was not bypassed'),
    )

    result = run_video(**inputs, **_default_scene_inputs(len(inputs['track'])), spans=injected)

    assert result.spans == injected


@pytest.mark.parametrize('kwargs', [{}, {'spans': [(10, 20)]}, {'spans': [(10, 20)], 'contacts': {0: [14]}}])
def test_run_video_requires_scene_inputs_for_every_sticky_consumer(kwargs):
    with pytest.raises(ValueError, match='^scene-gated sticky needs homography_rows and court_present$'):
        run_video(**_synthetic_inputs(), **kwargs)


def test_run_video_hands_tracker_segments_output_to_sticky_builder(monkeypatch):
    """The sticky builder must receive exactly the list tracker_segments produced.

    The handoff is internal to run_video, so a wrong list (rally spans, say) would
    only show up as plausible end-number movement. The spy records the argument and
    calls the real builder through, so the real path still runs.
    """
    inputs = _synthetic_inputs()
    n_frames = len(inputs['track'])
    court_present = np.ones(n_frames, dtype=bool)
    court_present[40:60] = False
    scene_row = _default_scene_inputs(n_frames)['homography_rows'][0]
    homography_rows = [
        {**scene_row, 'start_frame': '0', 'end_frame': '150'},
        {**scene_row, 'start_frame': '150', 'end_frame': str(n_frames)},
    ]
    expected = stage8_seg.tracker_segments(homography_rows, court_present, n_frames)
    # Dropout splits the first scene row: the fixture must stay non-trivial.
    assert expected == [(0, 40), (60, 150), (150, 300)]

    real_builder = stage8_seg.build_sticky_result
    received = []

    def spy(track, segments, *args, **kwargs):
        received.append(segments)
        return real_builder(track, segments, *args, **kwargs)

    monkeypatch.setattr(stage8_seg, 'build_sticky_result', spy)

    run_video(**inputs, court_present=court_present, homography_rows=homography_rows)

    assert received == [expected]


def test_run_video_rejects_serve_start_with_injected_spans() -> None:
    inputs = _synthetic_inputs()
    with pytest.raises(ValueError, match='serve_start cannot be combined with injected spans'):
        run_video(
            **inputs, **_default_scene_inputs(len(inputs['track'])), spans=[(10, 20)],
            serve_start=ServeStartConfig(0.5, ServeStartMode.TRIM),
        )


def test_run_video_injected_contacts_are_unmeasured_and_scored():
    inputs = _synthetic_inputs()
    spans = [(10, 20)]
    frames = [14, 16]
    expected = [ContactCandidate(0, frame, None, None, None) for frame in frames]

    result = run_video(**inputs, **_default_scene_inputs(len(inputs['track'])), spans=spans, contacts={0: frames})

    assert result.contacts == expected
    assert result.filtered_contacts == expected
    assert result.filtered_by_rally == {0: frames}


def test_run_video_injected_contacts_without_mask_completes(monkeypatch):
    inputs = _synthetic_inputs()
    del inputs['dead_mask']
    monkeypatch.setattr(
        run_video_module.point_winner, 'attribute_half',
        lambda *args, **kwargs: Half.TOP,
    )

    result = run_video(
        **inputs, **_default_scene_inputs(len(inputs['track'])), spans=[(10, 20)], contacts={0: [14]},
    )

    assert result.striker_halves == [Half.TOP]
    assert 0 in result.verdict_rows
    assert 0 in result.geometric_verdict_rows


def test_run_video_geometric_diagnostic_has_nullable_agreement(monkeypatch):
    inputs = _synthetic_inputs()
    monkeypatch.setattr(
        run_video_module.point_winner, 'attribute_half',
        lambda *args, **kwargs: Half.TOP,
    )
    monkeypatch.setattr(
        run_video_module.point_winner, 'pick_landing',
        lambda *args, **kwargs: None,
    )

    result = run_video(
        **inputs, **_default_scene_inputs(len(inputs['track'])), spans=[(10, 20)], contacts={0: [14]},
    )

    diagnostic = result.geometric_verdict_rows[0]
    assert diagnostic.geometric_verdict is None
    assert diagnostic.geometric_winner is None
    assert diagnostic.agreement is None


def test_run_video_geometric_diagnostic_records_a_resolved_winner(monkeypatch):
    inputs = _synthetic_inputs()
    monkeypatch.setattr(
        run_video_module.point_winner, 'attribute_half',
        lambda *args, **kwargs: Half.TOP,
    )
    monkeypatch.setattr(
        run_video_module.point_winner, 'pick_landing',
        lambda *args, **kwargs: Landing(15, (0.5, 0.75), Half.BOT, False, False, False),
    )

    result = run_video(
        **inputs, **_default_scene_inputs(len(inputs['track'])), spans=[(10, 20)], contacts={0: [14]},
    )

    diagnostic = result.geometric_verdict_rows[0]
    assert diagnostic.geometric_verdict.value == 'won'
    assert diagnostic.geometric_winner is Half.TOP
    assert diagnostic.agreement is True


def test_run_video_has_no_geometric_diagnostic_without_resolved_striker(monkeypatch):
    inputs = _synthetic_inputs()
    monkeypatch.setattr(
        run_video_module.point_winner, 'attribute_half',
        lambda *args, **kwargs: None,
    )

    result = run_video(
        **inputs, **_default_scene_inputs(len(inputs['track'])), spans=[(10, 20)], contacts={0: [14]},
    )

    assert result.verdict_rows == {}
    assert result.geometric_verdict_rows == {}


def test_run_video_injected_contacts_build_shared_sticky_once(monkeypatch):
    inputs = _synthetic_inputs()
    real_build_sticky_result = stage8_seg.build_sticky_result
    build_calls = 0

    def count_builds(*args, **kwargs):
        nonlocal build_calls
        build_calls += 1
        return real_build_sticky_result(*args, **kwargs)

    def fail_if_called(*args, **kwargs):
        raise AssertionError('dead-mask builder must be bypassed')

    monkeypatch.setattr(
        stage8_seg, 'build_sticky_result', count_builds,
    )
    monkeypatch.setattr(
        run_video_module, 'build_dead_mask', fail_if_called,
    )

    result = run_video(
        **inputs, **_default_scene_inputs(len(inputs['track'])), spans=[(10, 20)], contacts={0: [14]},
    )

    assert result.spans == [(10, 20)]
    assert result.contacts == [ContactCandidate(0, 14, None, None, None)]
    assert build_calls == 1


def _synthetic_inputs():
    video_id = 1
    n_frames = 300
    resolution = (1920.0, 1080.0)
    court_info = {
        'H': np.eye(3), 'border_L': 0.0, 'border_R': 1280.0,
        'border_U': 0.0, 'border_D': 720.0,
    }
    return {
        'track': np.zeros((n_frames, 3), dtype=np.float64),
        'bboxes': np.zeros((n_frames, 1, 4), dtype=np.float32),
        'scores': np.zeros((n_frames, 1), dtype=np.float32),
        'kps': np.zeros((n_frames, 1, 17, 2), dtype=np.float32),
        'ndet': np.zeros(n_frames, dtype=np.int64),
        'fps': 25.0,
        'landing_options': LandingFilterOptions(7, 0.004, 5, 7, 0.75),
        'court_box': CourtBox(
            x_range=(635.0, 1316.0), y_range=(254.0, 1030.0),
            height_band=(84.0, 336.0), mid_band=(642.0, 642.0),
        ),
        'net_band': (664.6, 703.7), 'resolution': resolution,
        'video_id': video_id, 'court_info': court_info,
        'homo_df': pd.DataFrame({
            'upleft_x': [0.0], 'upright_x': [1280.0],
            'downleft_x': [0.0], 'downright_x': [1280.0],
            'upleft_y': [0.0], 'upright_y': [0.0],
            'downleft_y': [720.0], 'downright_y': [720.0],
        }, index=[video_id]),
        'gate_court_info': {str(video_id): court_info},
        'gate_resolution_table': pd.DataFrame(
            {'width': [1920.0], 'height': [1080.0]}, index=[str(video_id)],
        ),
        'dead_mask': np.zeros(n_frames, dtype=bool),
    }


def _default_scene_inputs(n_frames: int):
    return {
        'court_present': np.ones(n_frames, dtype=bool),
        'homography_rows': [{
            'start_frame': '0', 'end_frame': str(n_frames),
            'upleft_x': 0.0, 'upright_x': 1280.0,
            'downleft_x': 0.0, 'downright_x': 1280.0,
            'upleft_y': 0.0, 'upright_y': 0.0,
            'downleft_y': 720.0, 'downright_y': 720.0,
        }],
    }


def test_write_geometric_verdicts_csv_serialises_nulls_blank(tmp_path) -> None:
    rows = [
        GeometricVerdictRow(0, Verdict.WON, Half.TOP, True),
        GeometricVerdictRow(2, None, None, None),
    ]
    path = tmp_path / 'pilot_geometric_verdicts.csv'
    write_geometric_verdicts_csv(rows, path)
    assert path.read_text(encoding='utf-8').splitlines() == [
        'rally_id,geometric_verdict,geometric_winner,agreement',
        '0,won,Top,True',
        '2,,,',
    ]
