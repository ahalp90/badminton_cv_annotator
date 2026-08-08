"""Tests for the annotator's windowed doubles verdict.

Pins the fraction-only rule: the strict-greater boundary at
``DOUBLES_SPAN_FRACTION``, the half-open span slicing, and the CLI round-trip.
The boundary test drives its arrays off the config constant so it tracks a
tuned value rather than a hard-coded default.
"""
from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

from annotator import config
from annotator.doubles_flag import doubles_flag, read_whole_video_flags
from annotator.run_video import AnnotatorResult


# -- Span-fraction boundary: strict greater-than -----------------------------


def test_span_fraction_just_over_vs_exactly_half():
    """Exactly ``DOUBLES_SPAN_FRACTION`` of frames does not fire; one more frame does."""
    frac = config.DOUBLES_SPAN_FRACTION
    n = 1000
    n_at = int(frac * n)  # 500 at the 0.5 default; exact for any k/1000 fraction

    at_threshold = np.zeros(n, dtype=bool)
    at_threshold[:n_at] = True
    assert at_threshold.mean() == frac  # construction really sits on the boundary
    assert doubles_flag(at_threshold) is False  # strict >: equal does not fire

    just_over = np.zeros(n, dtype=bool)
    just_over[:n_at + 1] = True
    assert doubles_flag(just_over) is True


# -- All-False input ---------------------------------------------------------


def test_all_false_never_flags():
    """No over-count anywhere: never doubles, whole array or any span."""
    clean = np.zeros(500, dtype=bool)
    assert doubles_flag(clean) is False
    assert doubles_flag(clean, span=(10, 100)) is False


# -- Span slicing ------------------------------------------------------------


def test_span_slicing_restricts_to_window():
    """The verdict reads only the half-open ``[start, end)`` window."""
    n = 60
    block_start = 5
    block_end = 36  # 31 True frames: over half of the whole 60-frame array
    mask = np.zeros(n, dtype=bool)
    mask[block_start:block_end] = True

    assert doubles_flag(mask) is True  # whole array sees the block

    # Span starting just past the block excludes it entirely -> clean.
    assert doubles_flag(mask, span=(block_end, n)) is False
    # Span covering exactly the block is all True -> flagged.
    assert doubles_flag(mask, span=(block_start, block_end)) is True
    # Half-open upper bound: a span ending exactly at the block's start excludes it.
    assert doubles_flag(mask, span=(0, block_start)) is False


# -- CLI round-trip ----------------------------------------------------------


def test_cli_whole_video_and_spans(tmp_path, monkeypatch):
    """main() sweeps <video_id>_overcount.npy into doubles_flags.csv, per-video and per-span."""
    from annotator import doubles_flag as df_mod

    overcount_dir = tmp_path / "overcount"
    overcount_dir.mkdir()

    # vid_a over-counts on 60 of 100 frames; vid_b is clean throughout.
    a = np.zeros(100, dtype=bool)
    a[:60] = True
    np.save(overcount_dir / "vid_a_overcount.npy", a)
    np.save(overcount_dir / "vid_b_overcount.npy", np.zeros(100, dtype=bool))

    out_csv = tmp_path / "doubles_flags.csv"
    monkeypatch.setattr(df_mod, "SCRAPE_DIR", tmp_path)
    monkeypatch.setattr(df_mod, "DOUBLES_FLAGS_CSV", out_csv)

    # Whole-video branch: one row per file, rally_id blank, bools as 'True'/'False'.
    assert df_mod.main(["--overcount-dir", str(overcount_dir)]) == 0
    by_id = {row["video_id"]: row for row in csv.DictReader(out_csv.open())}
    assert by_id["vid_a"]["doubles_flag"] == "True"
    assert by_id["vid_a"]["rally_id"] == ""
    assert by_id["vid_b"]["doubles_flag"] == "False"

    # Spans branch: the clean tail reads False, the over-count block reads True.
    spans_csv = tmp_path / "rally_spans.csv"
    with spans_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_id", "rally_id", "start_frame", "end_frame"])
        writer.writerow(["vid_a", "r1", 60, 100])
        writer.writerow(["vid_a", "r2", 0, 60])

    assert df_mod.main([
        "--overcount-dir", str(overcount_dir), "--rally-spans", str(spans_csv),
    ]) == 0
    verdict = {
        (row["video_id"], row["rally_id"]): row["doubles_flag"]
        for row in csv.DictReader(out_csv.open())
    }
    assert verdict[("vid_a", "r1")] == "False"
    assert verdict[("vid_a", "r2")] == "True"


def test_read_whole_video_flags_ignores_rally_rows(tmp_path, write_doubles_flags):
    flags_csv = tmp_path / 'doubles_flags.csv'
    write_doubles_flags(flags_csv, [
        ('vid_a', '', 'True'),
        ('vid_a', 'rally-1', 'False'),
        ('vid_b', '', 'False'),
    ])

    assert read_whole_video_flags(flags_csv) == {'vid_a': True, 'vid_b': False}


def test_read_whole_video_flags_rejects_duplicate_whole_video(tmp_path, write_doubles_flags):
    flags_csv = tmp_path / 'doubles_flags.csv'
    write_doubles_flags(flags_csv, [('vid_a', '', 'True'), ('vid_a', '', 'False')])

    with pytest.raises(ValueError, match='duplicate whole-video'):
        read_whole_video_flags(flags_csv)


@pytest.mark.parametrize('value', ['true', '1', ''])
def test_read_whole_video_flags_rejects_non_literal_booleans(
    tmp_path, value, write_doubles_flags,
):
    flags_csv = tmp_path / 'doubles_flags.csv'
    write_doubles_flags(flags_csv, [('vid_a', '', value)])

    with pytest.raises(ValueError, match='vid_a'):
        read_whole_video_flags(flags_csv)


def _run_segmentation_cli(tmp_path, monkeypatch, *, doubles_csv: Path | None, processed: list[str]) -> None:
    shuttle_dir = tmp_path / 'tracks'
    shuttle_dir.mkdir()
    # Distinct lengths so `processed` identifies WHICH video ran, not just how many.
    for video_id, n_frames in (('vid_a', 4), ('vid_b', 5), ('vid_c', 6)):
        np.save(shuttle_dir / f'{video_id}.npy', np.zeros((n_frames, 3)))

    import annotator.run_video as run_video_module
    from annotator import rally_segmentation as segmentation

    def fake_run_video(track, *args, **kwargs):
        processed.append(str(len(track)))
        return AnnotatorResult([], [], [], {}, [], [], [], [], {}, {}, {}, {}, [])

    monkeypatch.setattr(
        run_video_module, 'run_video', fake_run_video,
    )
    argv = [
        'rally_segmentation', '--shuttle-dir', str(shuttle_dir), '--fps', '30',
        '--rally-spans-csv', str(tmp_path / 'spans.csv'),
        '--contact-frames-csv', str(tmp_path / 'contacts.csv'),
    ]
    if doubles_csv is not None:
        argv.extend(['--doubles-csv', str(doubles_csv)])
    monkeypatch.setattr(sys, 'argv', argv)
    segmentation.main()


def test_runner_excludes_doubles_and_missing_rows(
    tmp_path, monkeypatch, caplog, write_doubles_flags,
):
    flags_csv = tmp_path / 'doubles_flags.csv'
    write_doubles_flags(flags_csv, [('vid_a', '', 'True'), ('vid_b', '', 'False')])
    processed: list[str] = []

    with caplog.at_level(logging.WARNING):
        _run_segmentation_cli(tmp_path, monkeypatch, doubles_csv=flags_csv, processed=processed)

    assert processed == ['5']  # exactly one segment_video call reached the fake
    # The batch report names the videos directly, so the outcome check no
    # longer leans on the track-length proxy alone.
    report_text = (tmp_path / 'spans_batch_report.txt').read_text(encoding='utf-8')
    assert '- vid_b: processed; 0 rallies; 0 contacts' in report_text
    assert '- vid_a: excluded; flagged doubles' in report_text
    assert '- vid_c: excluded; no doubles row; not assuming singles' in report_text
    assert 'excluding vid_a: flagged doubles' in caplog.text
    assert 'excluding vid_c: no doubles row; not assuming singles' in caplog.text


def test_runner_without_doubles_csv_processes_all(tmp_path, monkeypatch):
    processed: list[str] = []

    _run_segmentation_cli(tmp_path, monkeypatch, doubles_csv=None, processed=processed)

    assert processed == ['4', '5', '6']


def test_runner_raises_when_doubles_filter_excludes_every_video(
    tmp_path, monkeypatch, write_doubles_flags,
):
    # Per-rally rows only (this module's own CLI output form): no video has a
    # whole-video row, so all are excluded and the batch must refuse to run.
    flags_csv = tmp_path / 'doubles_flags.csv'
    write_doubles_flags(flags_csv, [('vid_a', '0', 'False'), ('vid_b', '1', 'True')])
    processed: list[str] = []

    with pytest.raises(ValueError, match='excluded every video'):
        _run_segmentation_cli(tmp_path, monkeypatch, doubles_csv=flags_csv, processed=processed)

    assert processed == []
