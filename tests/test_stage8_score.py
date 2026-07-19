"""Stage-8 scoring CLI helper tests."""
import numpy as np
import pandas as pd
import pytest

from scripts.stage8_score import _parse_proximity, _parse_tolerances, _spans_from_df


def test_spans_from_df_requires_contiguous_rally_ids():
    good = pd.DataFrame({
        'video_id': ['v', 'v'], 'rally_id': [1, 0],
        'start_frame': [30, 8], 'end_frame': [50, 20],
    })
    assert _spans_from_df(good) == [(8, 20), (30, 50)]

    gapped = pd.DataFrame({
        'video_id': ['v', 'v'], 'rally_id': [0, 2],
        'start_frame': [8, 30], 'end_frame': [20, 50],
    })
    with pytest.raises(ValueError, match='contiguous'):
        _spans_from_df(gapped)


def test_parse_proximity_true_false_blank():
    assert _parse_proximity('True') is True
    assert _parse_proximity('False') is False
    assert _parse_proximity(np.nan) is None
    assert _parse_proximity(float('nan')) is None


def test_parse_tolerances_list():
    assert _parse_tolerances('1,2,5,10') == [1, 2, 5, 10]
    assert _parse_tolerances('3') == [3]
    assert _parse_tolerances('1, 2') == [1, 2]
