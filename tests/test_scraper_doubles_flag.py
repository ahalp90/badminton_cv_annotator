"""Tests for the windowed doubles verdict (``scraper.doubles_flag``, spec s8).

Pins the two-legged rule (span fraction OR consecutive run), the strict-greater
fraction boundary, the run-length boundary at ``DOUBLES_MIN_CONSECUTIVE``, and the
half-open span slicing. The boundary tests drive their arrays off the config
constants so they track a tuned value rather than a hard-coded default, keeping the
runs short enough that only the leg under test can fire.
"""
from __future__ import annotations

import csv

import numpy as np

from scraper import config
from scraper.doubles_flag import doubles_flag


def _spread_true(n: int, k: int) -> np.ndarray:
    """Bool array of length ``n`` with exactly ``k`` True, spread as evenly as possible.

    ``floor(i * n / k)`` for i in [0, k) gives k strictly-increasing indices (step
    >= 1 since n >= k), so runs stay length 1-2. That isolates the fraction leg: the
    consecutive-run leg can never fire on a short-run mask.
    """
    mask = np.zeros(n, dtype=bool)
    if k > 0:
        mask[(np.arange(k) * n) // k] = True
    return mask


# -- Span-fraction boundary: strict greater-than -----------------------------


def test_span_fraction_just_over_vs_exactly_half():
    """Exactly ``DOUBLES_SPAN_FRACTION`` of frames does not fire; one more frame does."""
    frac = config.DOUBLES_SPAN_FRACTION
    n = 1000
    n_at = int(frac * n)  # 500 at the 0.5 default; exact for any k/1000 fraction

    at_threshold = _spread_true(n, n_at)
    assert at_threshold.mean() == frac  # construction really sits on the boundary
    assert doubles_flag(at_threshold) is False  # strict >: equal does not fire

    just_over = _spread_true(n, n_at + 1)
    assert doubles_flag(just_over) is True


# -- Consecutive-run boundary ------------------------------------------------


def test_consecutive_run_boundary():
    """A run of exactly ``DOUBLES_MIN_CONSECUTIVE`` fires; one frame shorter does not."""
    consec = config.DOUBLES_MIN_CONSECUTIVE
    # Long array so the fraction leg stays silent (run/length ~0.1 < the fraction):
    # this isolates the run leg as the only thing that can fire.
    n = consec * 10

    at_threshold = np.zeros(n, dtype=bool)
    at_threshold[:consec] = True
    assert doubles_flag(at_threshold) is True

    one_fewer = np.zeros(n, dtype=bool)
    one_fewer[: consec - 1] = True
    assert doubles_flag(one_fewer) is False


# -- All-False input ---------------------------------------------------------


def test_all_false_never_flags():
    """No over-count anywhere: never doubles, whole array or any span."""
    clean = np.zeros(500, dtype=bool)
    assert doubles_flag(clean) is False
    assert doubles_flag(clean, span=(10, 100)) is False


# -- Span slicing ------------------------------------------------------------


def test_span_slicing_restricts_to_window():
    """The verdict reads only the half-open ``[start, end)`` window."""
    consec = config.DOUBLES_MIN_CONSECUTIVE
    n = consec * 10
    run_start = 5
    mask = np.zeros(n, dtype=bool)
    mask[run_start:run_start + consec] = True  # a doubles run near the start

    assert doubles_flag(mask) is True  # whole array sees the run

    # Span starting just past the run excludes it entirely -> clean.
    assert doubles_flag(mask, span=(run_start + consec, n)) is False
    # Span containing the whole run -> flagged.
    assert doubles_flag(mask, span=(0, run_start + consec)) is True
    # Half-open upper bound: a span ending exactly at the run's start excludes it.
    assert doubles_flag(mask, span=(0, run_start)) is False


# -- CLI round-trip ----------------------------------------------------------


def test_cli_whole_video_and_spans(tmp_path, monkeypatch):
    """main() sweeps <video_id>_overcount.npy into doubles_flags.csv, per-video and per-span."""
    from scraper import doubles_flag as df_mod

    consec = config.DOUBLES_MIN_CONSECUTIVE
    overcount_dir = tmp_path / "overcount"
    overcount_dir.mkdir()

    # vid_a carries a clear doubles run; vid_b is clean throughout.
    a = np.zeros(consec * 10, dtype=bool)
    a[3:3 + consec] = True
    np.save(overcount_dir / "vid_a_overcount.npy", a)
    np.save(overcount_dir / "vid_b_overcount.npy", np.zeros(consec * 10, dtype=bool))

    out_csv = tmp_path / "doubles_flags.csv"
    monkeypatch.setattr(df_mod, "SCRAPE_DIR", tmp_path)
    monkeypatch.setattr(df_mod, "DOUBLES_FLAGS_CSV", out_csv)

    # Whole-video branch: one row per file, rally_id blank, bools as 'True'/'False'.
    assert df_mod.main(["--overcount-dir", str(overcount_dir)]) == 0
    by_id = {row["video_id"]: row for row in csv.DictReader(out_csv.open())}
    assert by_id["vid_a"]["doubles_flag"] == "True"
    assert by_id["vid_a"]["rally_id"] == ""
    assert by_id["vid_b"]["doubles_flag"] == "False"

    # Spans branch: a span after the run reads clean, one containing it reads doubles.
    spans_csv = tmp_path / "rally_spans.csv"
    with spans_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_id", "rally_id", "start_frame", "end_frame"])
        writer.writerow(["vid_a", "r1", 3 + consec, consec * 10])
        writer.writerow(["vid_a", "r2", 0, 3 + consec])

    assert df_mod.main([
        "--overcount-dir", str(overcount_dir), "--rally-spans", str(spans_csv),
    ]) == 0
    verdict = {
        (row["video_id"], row["rally_id"]): row["doubles_flag"]
        for row in csv.DictReader(out_csv.open())
    }
    assert verdict[("vid_a", "r1")] == "False"
    assert verdict[("vid_a", "r2")] == "True"
