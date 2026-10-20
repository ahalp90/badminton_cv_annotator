"""Write optional accept, review, or filter decisions for a rally-record run."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from dataset_builder.records import RALLY_RECORDS_FILENAME, load_rally_records
from dataset_builder.vision import save_json_gz

REVIEW_POLICY_SCHEMA = "rally-review-policy/0.1"


def review_decision(score: float | None, low_filter_threshold: float, high_accept_threshold: float) -> str:
    """Classify a persisted score after the threshold pair has been validated."""
    if score is None:
        return "review"
    if score < low_filter_threshold:
        return "filter"
    if score >= high_accept_threshold:
        return "accept"
    return "review"


def validate_review_thresholds(low_filter_threshold: float, high_accept_threshold: float) -> None:
    """Require finite thresholds with 0 <= low <= high <= 1."""
    if not math.isfinite(low_filter_threshold) or not math.isfinite(high_accept_threshold):
        raise ValueError("review-policy thresholds must be finite numbers")
    if not 0 <= low_filter_threshold <= high_accept_threshold <= 1:
        raise ValueError("review-policy thresholds must satisfy 0 <= low <= high <= 1")


def write_review_policy(
    run_dir: Path,
    output: Path,
    *,
    low_filter_threshold: float,
    high_accept_threshold: float,
) -> Path:
    """Write a separate, provenance-linked decision sidecar for one validated run."""
    validate_review_thresholds(low_filter_threshold, high_accept_threshold)
    run_path = Path(run_dir)
    source_path = run_path / RALLY_RECORDS_FILENAME
    output_path = Path(output)

    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"review-policy output already exists: {output_path}")

    source_checksum = _sha256_file(source_path)
    records = load_rally_records(source_path)
    decisions: list[dict[str, object]] = []
    for record in records:
        contacts = cast(Mapping[str, object], record["contacts"])
        key = cast(Mapping[str, object], record["key"])
        decisions.append({
            "key": dict(key),
            "decision": review_decision(
                cast(float | None, contacts["review_score"]),
                low_filter_threshold,
                high_accept_threshold,
            ),
        })

    # A replay can replace the records while this command reads them.
    if _sha256_file(source_path) != source_checksum:
        raise ValueError("source records changed during policy evaluation; retry with a stable run")
    payload: dict[str, object] = {
        "schema": REVIEW_POLICY_SCHEMA,
        "source": {
            "collection_filename": RALLY_RECORDS_FILENAME,
            "collection_sha256": source_checksum,
        },
        "thresholds": {
            "low_filter": low_filter_threshold,
            "high_accept": high_accept_threshold,
        },
        "decisions": decisions,
    }
    return save_json_gz(output_path, payload)


def _sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()
