from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from annotator.video_metadata import VideoMetadata
from dataset_builder.features import InterpolationType, PlayerFeatureInputs


def test_summary_counts_finite_provenance_and_metric_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from measure_player_court_features import _summarise_player_inputs

    positions = np.array(
        [
            [[0.0, 0.0], [0.5, 0.75]],
            [[0.1, 0.0], [0.5, 0.75]],
            [[1.1, 0.0], [0.5, 0.75]],
            [[0.2, 0.0], [0.5, 0.75]],
        ],
        dtype=float,
    )
    provenance = np.zeros((4, 2), dtype=np.int8)
    provenance[2, 0] = InterpolationType.LINEAR
    inputs = PlayerFeatureInputs(
        posture=np.zeros((4, 2), dtype=float),
        court_positions=positions,
        posture_interpolation=np.zeros((4, 2), dtype=np.int8),
        position_interpolation=provenance,
        tracker_segments=((0, 2), (2, 4)),
    )
    metadata = VideoMetadata(Path("/video.mp4"), Fraction(30), 4, 1280, 720)

    summary = _summarise_player_inputs(
        metadata,
        {"video_id": "sset_01", "metadata": "metadata", "court": "court", "pose": "pose", "shuttle_track": "track"},
        inputs,
    )

    assert summary["fps_fraction"] == "30/1"
    assert summary["segment_count"] == 2
    assert summary["positions"] == {
        "finite_count": 8,
        "denominator": 8,
        "finite_coverage": 1.0,
        "observed_finite_count": 7,
        "linear_finite_count": 1,
        "out_of_court_finite_count": 1,
    }
    assert summary["speed_mps"]["unit"] == "m/s"
    assert summary["speed_mps"]["finite_count"] == 3
    assert summary["speed_mps"]["denominator"] == 8
    assert summary["speed_mps"]["maximum_sample"] == {"frame": 1, "slot": 0}
    assert summary["half_court_centre_distance_m"]["unit"] == "m"
    assert summary["half_court_centre_distance_m"]["finite_count"] == 8
