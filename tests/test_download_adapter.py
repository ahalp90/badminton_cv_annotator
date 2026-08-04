"""Tests for the ShuttleSet adapter to the scraper downloader."""

import csv
from pathlib import Path

import pandas as pd

from src.bst_x.pipeline import download_adapter


def test_adapter_maps_match_rows_filters_exclusions_and_enables_video_only(
    tmp_path: Path,
    monkeypatch,
) -> None:
    match_csv = tmp_path / 'match.csv'
    pd.DataFrame(
        {
            'id': [1, 2, 3],
            'url': ['u1', 'u2', 'u3'],
            'video': ['Match A', 'Match B', 'Match C'],
        }
    ).to_csv(match_csv, index=False)
    output_dir = tmp_path / 'videos'
    observed: dict[str, object] = {}

    def fake_download_all_videos(**kwargs: object) -> list:
        candidates_path = kwargs['candidates_path']
        assert isinstance(candidates_path, Path)
        with candidates_path.open(newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            observed['header'] = reader.fieldnames
            observed['rows'] = list(reader)
        observed['kwargs'] = kwargs
        return []

    monkeypatch.setattr(download_adapter, 'download_all_videos', fake_download_all_videos)
    outcomes = download_adapter.download_shuttleset_videos(
        match_csv_path=match_csv,
        output_dir=output_dir,
        excluded=frozenset({2}),
        max_workers=3,
    )

    assert outcomes == []
    assert observed['header'] == download_adapter.scraper_config.CANDIDATES_COLUMNS
    rows = observed['rows']
    assert isinstance(rows, list)
    assert [(row['video_id'], row['url'], row['title'], row['keep']) for row in rows] == [
        ('1', 'u1', 'Match A', 'True'),
        ('3', 'u3', 'Match C', 'True'),
    ]
    kwargs = observed['kwargs']
    assert isinstance(kwargs, dict)
    assert kwargs['output_dir'] == output_dir
    assert kwargs['max_workers'] == 3
    assert kwargs['dataset'] == 'shuttleset'
    assert kwargs['video_only'] is True
