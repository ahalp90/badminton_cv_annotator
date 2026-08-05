"""Tests for the manual broadcast-timeline interval contract."""

from pathlib import Path

import pytest

from annotator.broadcast_timeline_labels import (
    LABEL_CSV_HEADER,
    LabelInterval,
    SceneTruth,
    VideoMetadata,
    interval_index_at,
    make_interval,
    read_label_csv,
    replace_interval,
    validate_intervals,
    validate_partition,
    write_label_csv,
)


METADATA = VideoMetadata("sset_01", 25.0, 100)


def _partition() -> list[LabelInterval]:
    return [
        make_interval(METADATA, 0, 20, SceneTruth.LIVE),
        make_interval(METADATA, 20, 30, SceneTruth.REPLAY, "slow-motion repeat"),
        make_interval(METADATA, 30, 100, SceneTruth.LIVE_NON_STANDARD),
    ]


@pytest.mark.parametrize(
    "metadata",
    [
        ("", 25.0, 100),
        ("sset_01", 0.0, 100),
        ("sset_01", float("nan"), 100),
        ("sset_01", 25.0, 0),
        ("sset_01", 25.0, True),
    ],
)
def test_video_metadata_rejects_invalid_values(metadata: tuple[object, object, object]) -> None:
    with pytest.raises(ValueError):
        VideoMetadata(*metadata)  # type: ignore[arg-type]


def test_partition_accepts_contiguous_half_open_intervals() -> None:
    intervals = _partition()

    assert validate_partition(intervals) == METADATA
    assert interval_index_at(intervals, 0) == 0
    assert interval_index_at(intervals, 19) == 0
    assert interval_index_at(intervals, 20) == 1
    assert interval_index_at(intervals, 99) == 2
    assert interval_index_at(intervals, 100) is None


def test_validation_rejects_overlap_gap_bounds_and_metadata_drift() -> None:
    with pytest.raises(ValueError, match="previous interval ends"):
        validate_intervals([
            make_interval(METADATA, 0, 20, SceneTruth.LIVE),
            make_interval(METADATA, 19, 30, SceneTruth.REPLAY),
        ])

    with pytest.raises(ValueError, match=r"gap \[20, 21\)"):
        validate_partition([
            make_interval(METADATA, 0, 20, SceneTruth.LIVE),
            make_interval(METADATA, 21, 100, SceneTruth.REPLAY),
        ])

    with pytest.raises(ValueError, match="outside"):
        validate_intervals([make_interval(METADATA, 0, 101, SceneTruth.LIVE)])

    other_metadata = VideoMetadata("sset_15", 25.0, 100)
    with pytest.raises(ValueError, match="does not match"):
        validate_intervals([
            make_interval(METADATA, 0, 20, SceneTruth.LIVE),
            make_interval(other_metadata, 20, 100, SceneTruth.LIVE),
        ])


def test_partial_coverage_is_valid_until_partition_freeze() -> None:
    intervals = [make_interval(METADATA, 20, 30, SceneTruth.CUTAWAY)]

    assert validate_intervals(intervals) == METADATA
    with pytest.raises(ValueError, match="partition starts"):
        validate_partition(intervals)


def test_relabel_preserves_interval_identity_and_note() -> None:
    intervals = _partition()

    updated = replace_interval(intervals, 1, truth=SceneTruth.CUTAWAY)

    assert updated[1] == make_interval(METADATA, 20, 30, SceneTruth.CUTAWAY, "slow-motion repeat")
    assert intervals[1].truth is SceneTruth.REPLAY


@pytest.mark.parametrize("suffix", [".csv", ".csv.gz"])
def test_csv_round_trip_preserves_schema_and_values(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"labels{suffix}"
    intervals = _partition()

    write_label_csv(path, intervals, METADATA)

    assert read_label_csv(path) == intervals
    assert not list(tmp_path.glob(".*.tmp.csv*"))


@pytest.mark.parametrize("suffix", [".csv", ".csv.gz"])
def test_csv_round_trip_preserves_fractional_fps(tmp_path: Path, suffix: str) -> None:
    metadata = VideoMetadata("fractional", 30000 / 1001, 10)
    interval = make_interval(metadata, 0, 10, SceneTruth.LIVE)
    path = tmp_path / f"labels{suffix}"

    write_label_csv(path, [interval], metadata)

    assert read_label_csv(path) == [interval]


def test_failed_temporary_validation_preserves_existing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import annotator.broadcast_timeline_labels as labels_module

    path = tmp_path / "labels.csv"
    write_label_csv(path, _partition(), METADATA)
    original = path.read_bytes()
    real_reader = labels_module.read_label_csv

    def corrupt_temporary_read(candidate: Path) -> list[LabelInterval]:
        if candidate != path:
            return []
        return real_reader(candidate)

    monkeypatch.setattr(labels_module, "read_label_csv", corrupt_temporary_read)
    with pytest.raises(RuntimeError, match="round trip changed"):
        write_label_csv(path, _partition(), METADATA)

    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp.csv*"))


def test_gzip_output_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv.gz"
    intervals = _partition()

    write_label_csv(path, intervals, METADATA)
    first = path.read_bytes()
    write_label_csv(path, intervals, METADATA)

    assert path.read_bytes() == first


def test_empty_csv_keeps_the_exact_header(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"

    write_label_csv(path, [], METADATA)

    assert path.read_text(encoding="utf-8").splitlines() == [",".join(LABEL_CSV_HEADER)]
    assert read_label_csv(path) == []


def test_reader_rejects_foreign_header_and_unknown_truth(tmp_path: Path) -> None:
    foreign = tmp_path / "foreign.csv"
    foreign.write_text("start,end\n0,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="has columns"):
        read_label_csv(foreign)

    bad_truth = tmp_path / "bad_truth.csv"
    bad_truth.write_text(
        ",".join(LABEL_CSV_HEADER) + "\n" + "sset_01,25,100,0,100,unsure,\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid truth"):
        read_label_csv(bad_truth)
