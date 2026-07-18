"""Smoke coverage for the public annotator video composition."""
import numpy as np
import pandas as pd

from annotator.run_video import AnnotatorResult, run_video
from scraper.config import SHIPPED_THRESHOLDS
from annotator.point_winner import LandingFilterOptions
from annotator.rally_segmentation import CourtBox, scale_thresholds


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
        track, bboxes, scores, kps, ndet, dead,
        fps=25.0,
        thresholds=scale_thresholds(SHIPPED_THRESHOLDS, 25.0),
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
    )

    assert result == AnnotatorResult([], [], [], {}, [], [], [], [], {}, {}, {}, [])
