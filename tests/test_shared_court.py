"""Tests for shared court helpers."""

import pandas as pd
import pytest

from shared.court import build_all_court_info, load_all_court_info


def _write_homography(path):
    pd.DataFrame([
        {
            'id': 1,
            'homography_matrix': '[[1, 0, 0], [0, 1, 0], [0, 0, 1]]',
            'upleft_x': 0,
            'upright_x': 10,
            'downleft_x': 0,
            'downright_x': 10,
            'upleft_y': 0,
            'upright_y': 0,
            'downleft_y': 20,
            'downright_y': 20,
        },
        {
            'id': 2,
            'homography_matrix': '[[1, 0, 0], [0, 1, 0], [0, 0, 1]]',
            'upleft_x': 1,
            'upright_x': 11,
            'downleft_x': 1,
            'downright_x': 11,
            'upleft_y': 2,
            'upright_y': 2,
            'downleft_y': 22,
            'downright_y': 22,
        },
    ]).to_csv(path / 'homography.csv', index=False)


def test_build_all_court_info_uses_resolution_index(tmp_path):
    _write_homography(tmp_path)
    resolution = pd.DataFrame({'width': [1280], 'height': [720]}, index=[2])

    result = build_all_court_info(tmp_path, resolution)

    assert list(result) == [2]
    assert result[2]['border_L'] == pytest.approx(1)
    assert result[2]['border_R'] == pytest.approx(11)
    assert result[2]['border_U'] == pytest.approx(2)
    assert result[2]['border_D'] == pytest.approx(22)


def test_build_all_court_info_fails_on_missing_homography(tmp_path):
    _write_homography(tmp_path)
    resolution = pd.DataFrame({'width': [1280], 'height': [720]}, index=[3])

    with pytest.raises(KeyError):
        build_all_court_info(tmp_path, resolution)


def test_load_all_court_info_uses_every_homography_row(tmp_path):
    _write_homography(tmp_path)

    result = load_all_court_info(tmp_path / 'homography.csv')

    assert list(result) == [1, 2]
