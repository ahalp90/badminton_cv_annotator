"""Tests for the windowed doubles verdict (``scraper.doubles_flag``, spec s8).

Pins the fraction-only rule: the strict-greater boundary at
``DOUBLES_SPAN_FRACTION``, the half-open span slicing, and the CLI round-trip.
The boundary test drives its arrays off the config constant so it tracks a
tuned value rather than a hard-coded default.
"""
from __future__ import annotations

import csv

import numpy as np

from annotator import config
from annotator.doubles_flag import doubles_flag


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
