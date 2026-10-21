"""Optional review-policy sidecar behaviour and CLI contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dataset_builder import review_policy
from dataset_builder.cli import main
from dataset_builder.records import load_rally_records
from dataset_builder.review_policy import review_decision, validate_review_thresholds
from dataset_builder.vision import load_json_gz
from tests.test_dataset_builder_records import _annotation, _assemble, _manifest, _write


def _write_run(tmp_path: Path) -> Path:
    manifest = _manifest()
    annotation = _annotation()._replace(rally_confidence=(0.2, None))
    projection = _assemble(tmp_path, manifest=manifest, annotation=annotation)
    run_dir = tmp_path / "run"
    return _write(run_dir, manifest, [projection]).records


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (None, "review"),
        (0.0, "filter"),
        (0.299, "filter"),
        (0.3, "review"),
        (0.5, "review"),
        (0.7, "accept"),
        (1.0, "accept"),
    ],
)
def test_review_decision_threshold_boundaries(score: float | None, expected: str) -> None:
    assert review_decision(score, 0.3, 0.7) == expected


def test_equal_thresholds_are_supported() -> None:
    validate_review_thresholds(0.5, 0.5)
    assert review_decision(0.499, 0.5, 0.5) == "filter"
    assert review_decision(0.5, 0.5, 0.5) == "accept"


@pytest.mark.parametrize(
    ("low", "high"),
    [(-0.1, 0.7), (0.3, 1.1), (0.8, 0.2), (float("nan"), 0.7), (0.3, float("inf"))],
)
def test_invalid_review_thresholds_fail(low: float, high: float) -> None:
    with pytest.raises(ValueError):
        validate_review_thresholds(low, high)


def test_cli_writes_one_keyed_decision_and_preserves_source_bytes(tmp_path: Path) -> None:
    source_path = _write_run(tmp_path)
    source_before = source_path.read_bytes()
    original_records = load_rally_records(source_path)
    output_path = tmp_path / "decisions.json.gz"

    status = main([
        "review-policy",
        "--run-dir", str(source_path.parent),
        "--output", str(output_path),
        "--low-filter-threshold", "0.3",
        "--high-accept-threshold", "0.7",
    ])

    assert status == 0
    assert source_path.read_bytes() == source_before
    assert [record["contacts"]["review_score"] for record in load_rally_records(source_path)] == [0.2, None]
    payload = load_json_gz(output_path)
    decisions = payload["decisions"]
    assert isinstance(decisions, list)
    assert [row["key"] for row in decisions] == [record["key"] for record in original_records]
    assert [row["decision"] for row in decisions] == ["filter", "review"]
    assert payload["source"] == {
        "collection_filename": source_path.name,
        "collection_sha256": hashlib.sha256(source_before).hexdigest(),
    }
    assert str(tmp_path) not in json.dumps(payload)


def test_source_replacement_during_loading_does_not_write_stale_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = _write_run(tmp_path)
    output_path = tmp_path / "decisions.json.gz"

    def load_then_replace(path: Path) -> list[dict[str, object]]:
        records = load_rally_records(path)
        path.write_bytes(b"concurrent replacement")
        return records

    monkeypatch.setattr(review_policy, "load_rally_records", load_then_replace)
    with pytest.raises(ValueError, match="source records changed"):
        review_policy.write_review_policy(
            source_path.parent, output_path, low_filter_threshold=0.3, high_accept_threshold=0.7,
        )
    assert not output_path.exists()


def test_invalid_thresholds_and_existing_output_fail_without_replacement(tmp_path: Path) -> None:
    source_path = _write_run(tmp_path)
    output_path = tmp_path / "decisions.json.gz"

    assert main([
        "review-policy",
        "--run-dir", str(source_path.parent),
        "--output", str(output_path),
        "--low-filter-threshold", "0.8",
        "--high-accept-threshold", "0.2",
    ]) == 1
    assert not output_path.exists()

    output_path.write_bytes(b"keep existing output")
    assert main([
        "review-policy",
        "--run-dir", str(source_path.parent),
        "--output", str(output_path),
        "--low-filter-threshold", "0.3",
        "--high-accept-threshold", "0.7",
    ]) == 1
    assert output_path.read_bytes() == b"keep existing output"


def test_cli_rejects_source_records_as_output(tmp_path: Path) -> None:
    source_path = _write_run(tmp_path)
    source_before = source_path.read_bytes()

    assert main([
        "review-policy",
        "--run-dir", str(source_path.parent),
        "--output", str(source_path),
        "--low-filter-threshold", "0.3",
        "--high-accept-threshold", "0.7",
    ]) == 1
    assert source_path.read_bytes() == source_before
