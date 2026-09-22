"""Check source restriction and per-view measurement reuse boundaries."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from compare import selections
from measurement import prepared_measurements


def test_merged_source_membership_and_player_rejection() -> None:
    def candidate(key: str, sources: list[str], passes: bool) -> dict:
        return {"origin_key": key, "source_memberships": sources,
                "historical": {"historical_fullcourt": passes}}

    record = {
        "parents": [candidate("G0:best", ["G0"], True), candidate("G0:merged", ["G0", "G1"], False)],
        "valid_children": [candidate("line_template:child", ["line_template"], True)],
        "rankings": {"C": {"provisional_rank": ["G0:best", "G0:merged", "line_template:child"]}},
    }
    result = selections(record)
    assert result["full"]["gated"] == "G0:best"
    assert result["g1_templates"]["ungated"] == "G0:merged"
    assert result["g1_templates"]["gated"] == "line_template:child"
    assert result["g1_templates"]["rejected_by_player_gate"] == ["G0:merged"]
    record["parents"] = [record["parents"][0]]
    record["valid_children"] = []
    record["rankings"]["C"]["provisional_rank"] = ["G0:best"]
    assert selections(record)["g1_templates"]["gated"] is None


def test_greyscale_cache_changes_view_and_restores_after_failure() -> None:
    verifier = ModuleType("measurement_test")
    original_sample = object()
    original_junctions = object()
    verifier.grayscale_sample = original_sample
    verifier.raw_junctions = original_junctions
    first = np.arange(72, dtype=np.uint8).reshape(4, 6, 3)
    second = 255 - first
    points = np.array([[-1.25, 0], [2.4, 1.7], [9, 6]], dtype=np.float32)
    with (
        pytest.raises(RuntimeError, match="test interruption"),
        prepared_measurements(verifier) as counts,
    ):
        for frame in (first, first, second):
            expected = cv2.remap(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32),
                                 points[:, 0], points[:, 1], cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE).reshape(-1)
            np.testing.assert_array_equal(verifier.grayscale_sample(frame, points), expected)
        assert counts == {"greyscale_conversions": 2, "sampling_calls": 3}
        raise RuntimeError("test interruption")
    assert verifier.grayscale_sample is original_sample
    assert verifier.raw_junctions is original_junctions
