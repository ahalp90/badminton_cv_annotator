"""Tests for the ShuttleSet/keep-list downloader (download_videos.py).

subprocess.run is monkeypatched so no yt-dlp process spawns; the tests read back
the captured argv to check the built tasks, the keep-list filter, the D22
throttle flags and that a string video_id reaches the output filename.
"""
from pathlib import Path

import pandas as pd

from src.bst_x.pipeline import download_videos as dv


class _FakeCompleted:
    """Stand-in for subprocess.CompletedProcess with a success return code."""

    returncode = 0
    stderr = ''
    stdout = ''


def _make_fake_run(captured):
    """Return a subprocess.run stub that records each argv into ``captured``."""
    def fake_run(argv, **_kwargs):
        captured.append(list(argv))
        return _FakeCompleted()

    return fake_run


def _stem_of(argv, out_dir):
    """Recover the '{id} {name}' filename stem from a captured yt-dlp argv."""
    template = argv[argv.index('--output') + 1]
    assert Path(template).parent == out_dir
    return Path(template).name.removesuffix('.%(ext)s')


def test_default_shuttleset_shape_builds_expected_tasks(tmp_path, monkeypatch):
    """match.csv columns id,url,video build today's tasks; excluded ids are dropped."""
    monkeypatch.setattr(dv, '_check_ytdlp', lambda: None)
    captured = []
    monkeypatch.setattr(dv.subprocess, 'run', _make_fake_run(captured))

    csv_path = tmp_path / 'match.csv'
    pd.DataFrame({
        'id': [1, 2, 3],
        'url': ['u1', 'u2', 'u3'],
        'video': ['Match A', 'Match B', 'Match C'],
    }).to_csv(csv_path, index=False)
    out_dir = tmp_path / 'out'

    dv.download_all_videos(
        match_csv_path=csv_path, output_dir=out_dir, excluded=frozenset({2}), max_workers=1,
    )

    assert sorted(_stem_of(argv, out_dir) for argv in captured) == ['1 Match A', '3 Match C']
    assert sorted(argv[-1] for argv in captured) == ['u1', 'u3']


def test_keep_list_shape_filters_kept_rows(tmp_path, monkeypatch):
    """keep_col filters to keep=='True' for both a bool-dtype and a string-with-blanks column."""
    monkeypatch.setattr(dv, '_check_ytdlp', lambda: None)
    captured = []
    monkeypatch.setattr(dv.subprocess, 'run', _make_fake_run(captured))

    # All-bool keep column: pandas reads True/False into bool dtype.
    bool_csv = tmp_path / 'keep_bool.csv'
    pd.DataFrame({
        'video_id': ['aaa', 'bbb', 'ccc'],
        'url': ['ua', 'ub', 'uc'],
        'title': ['Vid A', 'Vid B', 'Vid C'],
        'keep': [True, False, True],
    }).to_csv(bool_csv, index=False)
    out_bool = tmp_path / 'out_bool'
    dv.download_all_videos(
        match_csv_path=bool_csv, output_dir=out_bool, excluded=frozenset(), max_workers=1,
        id_col='video_id', url_col='url', name_col='title', keep_col='keep',
    )
    assert sorted(_stem_of(argv, out_bool) for argv in captured) == ['aaa Vid A', 'ccc Vid C']

    # String keep column with a blank cell: stays object dtype; only 'True' passes.
    captured.clear()
    str_csv = tmp_path / 'keep_str.csv'
    str_csv.write_text(
        'video_id,url,title,keep\n'
        'ddd,ud,Vid D,True\n'
        'eee,ue,Vid E,False\n'
        'fff,uf,Vid F,\n'
        'ggg,ug,Vid G,True\n'
    )
    out_str = tmp_path / 'out_str'
    dv.download_all_videos(
        match_csv_path=str_csv, output_dir=out_str, excluded=frozenset(), max_workers=1,
        id_col='video_id', url_col='url', name_col='title', keep_col='keep',
    )
    assert sorted(_stem_of(argv, out_str) for argv in captured) == ['ddd Vid D', 'ggg Vid G']


def test_ytdlp_argv_carries_d22_throttle_flags(tmp_path, monkeypatch):
    """The yt-dlp argv carries all five D22 throttle flags with their values."""
    monkeypatch.setattr(dv, '_check_ytdlp', lambda: None)
    captured = []
    monkeypatch.setattr(dv.subprocess, 'run', _make_fake_run(captured))

    csv_path = tmp_path / 'match.csv'
    pd.DataFrame({'id': [7], 'url': ['u7'], 'video': ['Solo']}).to_csv(csv_path, index=False)
    dv.download_all_videos(
        match_csv_path=csv_path, output_dir=tmp_path / 'o', excluded=frozenset(), max_workers=1,
    )

    argv = captured[0]
    for flag, value in (
        ('--sleep-interval', '5'),
        ('--max-sleep-interval', '15'),
        ('--sleep-requests', '10'),
        ('--limit-rate', '2M'),
        ('--concurrent-fragments', '1'),
    ):
        assert flag in argv, f'missing {flag}'
        assert argv[argv.index(flag) + 1] == value


def test_string_video_id_flows_to_output_filename(tmp_path, monkeypatch):
    """A YouTube string id passes through unchanged into the output filename pattern."""
    monkeypatch.setattr(dv, '_check_ytdlp', lambda: None)
    captured = []
    monkeypatch.setattr(dv.subprocess, 'run', _make_fake_run(captured))

    csv_path = tmp_path / 'keep.csv'
    pd.DataFrame({
        'video_id': ['dQw4w9WgXcQ'],
        'url': ['https://youtu.be/dQw4w9WgXcQ'],
        'title': ['Never Gonna'],
        'keep': ['True'],
    }).to_csv(csv_path, index=False)
    out_dir = tmp_path / 'out'

    dv.download_all_videos(
        match_csv_path=csv_path, output_dir=out_dir, excluded=frozenset(), max_workers=1,
        id_col='video_id', url_col='url', name_col='title', keep_col='keep',
    )

    template = captured[0][captured[0].index('--output') + 1]
    assert Path(template).name == 'dQw4w9WgXcQ Never Gonna.%(ext)s'
