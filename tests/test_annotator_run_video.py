"""Smoke coverage for the public annotator video composition."""
import numpy as np
import pandas as pd
import pytest

import annotator.run_video as run_video_module
import annotator.rally_segmentation as stage8_seg
from annotator.point_winner import LandingFilterOptions
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
    )

    assert result == AnnotatorResult([], [], [], {}, [], [], [], [], {}, {}, {}, [])


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

    result = run_video(**inputs, spans=injected)

    assert result.spans == injected


def test_run_video_rejects_serve_start_with_injected_spans() -> None:
    inputs = _synthetic_inputs()
    with pytest.raises(ValueError, match='serve_start cannot be combined with injected spans'):
        run_video(
            **inputs, spans=[(10, 20)], serve_start=ServeStartConfig(0.5, ServeStartMode.TRIM),
        )


def test_run_video_injected_contacts_are_unmeasured_and_scored():
    inputs = _synthetic_inputs()
    spans = [(10, 20)]
    frames = [14, 16]
    expected = [ContactCandidate(0, frame, None, None, None) for frame in frames]

    result = run_video(**inputs, spans=spans, contacts={0: frames})

    assert result.contacts == expected
    assert result.filtered_contacts == expected
    assert result.filtered_by_rally == {0: frames}


def test_run_video_injected_contacts_skip_discarded_evidence_builders(monkeypatch):
    inputs = _synthetic_inputs()

    def fail_if_called(*args, **kwargs):
        raise AssertionError('discarded evidence builder must be bypassed')

    monkeypatch.setattr(
        stage8_seg, 'build_sticky_result', fail_if_called,
    )
    monkeypatch.setattr(
        run_video_module, 'build_dead_mask', fail_if_called,
    )

    result = run_video(**inputs, spans=[(10, 20)], contacts={0: [14]})

    assert result.spans == [(10, 20)]
    assert result.contacts == [ContactCandidate(0, 14, None, None, None)]


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
