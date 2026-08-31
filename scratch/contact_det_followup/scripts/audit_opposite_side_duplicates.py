"""Count near-simultaneous events attributed to opposite players."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import pairwise

from scratch.contact_det.scripts.score_contact_rallies import FixedEvent
from scratch.contact_det_followup.scripts.prediction_io import (
    REPO_ROOT,
    load_development_predictions,
    load_frozen_test_predictions,
    read_json,
)

OUTPUT_PATH = (
    REPO_ROOT
    / "scratch/contact_det_followup/results/opposite_side_duplicate_audit.json"
)


def _repository_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def count_opposite_side_pairs(
    events_by_fixture: Mapping[str, Sequence[FixedEvent]],
) -> dict[str, object]:
    """Count adjacent opposite-side events no more than two frames apart."""
    frame_gap_counts: Counter[int] = Counter()
    affected_videos: set[str] = set()
    for fixture, events in events_by_fixture.items():
        for first, second in pairwise(events):
            frame_gap = second.frame - first.frame
            if frame_gap > 2:
                continue
            if first.predicted_side is None or second.predicted_side is None:
                continue
            if first.predicted_side == second.predicted_side:
                continue
            frame_gap_counts[frame_gap] += 1
            affected_videos.add(fixture)

    return {
        "pair_count": sum(frame_gap_counts.values()),
        "pair_count_by_frame_gap": {
            str(frame_gap): frame_gap_counts[frame_gap]
            for frame_gap in (0, 1, 2)
        },
        "affected_videos": len(affected_videos),
    }


def run_audit() -> dict[str, object]:
    """Count candidate duplicate pairs without reading labels."""
    development = load_development_predictions()
    test = load_frozen_test_predictions()
    development_counts = count_opposite_side_pairs(development.events_by_fixture)
    test_counts = count_opposite_side_pairs(test.events_by_fixture)
    pair_count = int(development_counts["pair_count"]) + int(
        test_counts["pair_count"]
    )
    development_inputs: list[dict[str, object]] = []
    for path in development.paths:
        source = read_json(path)
        development_inputs.append(
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "schema": source["schema"],
                "source_commit": source["source_commit"],
            }
        )
    return {
        "schema": "contact-detector-opposite-side-duplicate-audit/1",
        "run_id": "opposite-side-pairs-within-two-frames",
        "repository_commit": _repository_commit(),
        "labels_read": False,
        "inputs": {
            "development": development_inputs,
            "frozen_test": {
                "path": str(test.path.relative_to(REPO_ROOT)),
                "schema": test.payload["schema"],
                "source_commit": test.source_commit,
            },
        },
        "rule": "Adjacent events have different known player sides and are zero to two frames apart.",
        "development": development_counts,
        "frozen_test": test_counts,
        "decision": "stop" if pair_count == 0 else "inspect_pairs",
        "decision_reason": (
            "No qualifying pairs exist in either saved prediction set."
            if pair_count == 0
            else "Qualifying pairs exist and need the label-guided best-case check."
        ),
    }


def main() -> None:
    """Write the compact duplicate audit."""
    payload = run_audit()
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
