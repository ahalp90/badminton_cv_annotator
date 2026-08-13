"""Focused tests for compressed annotator artifact I/O."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from annotator.artifact_io import (
    artifacts_are_byte_equal,
    atomic_gzip_text_writer,
    load_npy,
    open_text_artifact,
    read_json_object,
    save_npy_xz,
    write_json_object,
)


def test_json_gzip_is_deterministic_and_plain_json_remains_readable(tmp_path: Path) -> None:
    first = tmp_path / "first.json.gz"
    second = tmp_path / "second.json.gz"
    payload = {"schema_version": 1, "values": [1, 2, 3]}

    write_json_object(first, payload)
    write_json_object(second, payload)

    assert first.read_bytes() == second.read_bytes()
    assert read_json_object(first) == payload

    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps(payload), encoding="utf-8")
    assert read_json_object(tmp_path / "legacy.json.gz") == payload


def test_csv_gzip_round_trip_and_plain_fallback(tmp_path: Path) -> None:
    compressed = tmp_path / "rows.csv.gz"
    with atomic_gzip_text_writer(compressed, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "value"], lineterminator="\n")
        writer.writeheader()
        writer.writerow({"id": 1, "value": "one"})

    with open_text_artifact(compressed, newline="") as handle:
        assert list(csv.DictReader(handle)) == [{"id": "1", "value": "one"}]

    legacy = tmp_path / "legacy.csv"
    legacy.write_text("id,value\n2,two\n", encoding="utf-8")
    with open_text_artifact(tmp_path / "legacy.csv.gz", newline="") as handle:
        assert list(csv.DictReader(handle)) == [{"id": "2", "value": "two"}]


def test_npy_xz_round_trip_and_plain_fallback(tmp_path: Path) -> None:
    values = np.array([True, False, True], dtype=np.bool_)
    compressed = tmp_path / "values.npy.xz"
    save_npy_xz(compressed, values)
    np.testing.assert_array_equal(load_npy(compressed), values)

    legacy = tmp_path / "legacy.npy"
    np.save(legacy, values, allow_pickle=False)
    np.testing.assert_array_equal(load_npy(tmp_path / "legacy.npy.xz"), values)


def test_vote_byte_comparison_resolves_legacy_plain_fallbacks(tmp_path: Path) -> None:
    values = np.array([True, False, True], dtype=np.bool_)
    first = tmp_path / "first.npy"
    second = tmp_path / "second.npy"
    np.save(first, values, allow_pickle=False)
    np.save(second, values, allow_pickle=False)

    assert artifacts_are_byte_equal(tmp_path / "first.npy.xz", tmp_path / "second.npy.xz")
