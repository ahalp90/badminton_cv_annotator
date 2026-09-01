"""Save label-free side decisions for the 40 development videos."""

from __future__ import annotations

import gzip
import json
import subprocess
from dataclasses import asdict

from scratch.contact_det_followup.scripts.prediction_io import (
    REPO_ROOT,
    load_development_predictions,
)
from scratch.contact_det_followup.scripts.side_rules import simple_alternation_decisions

OUTPUT_PATH = REPO_ROOT / "scratch/contact_det_followup/results/simple_side_decisions_development.json.gz"


def _repository_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> None:
    """Write the compact development side-decision record."""
    predictions = load_development_predictions()
    decisions = simple_alternation_decisions(predictions.spans)
    changed_contacts = sum(
        before != after
        for decision in decisions
        for before, after in zip(decision.sides_before, decision.sides_after, strict=True)
    )
    payload = {
        "schema": "contact-detector-development-side-decisions/1",
        "status": "complete",
        "run_id": "simple-alternation-vote-development",
        "repository_commit": _repository_commit(),
        "labels_read": False,
        "source_predictions": [str(path.relative_to(REPO_ROOT)) for path in predictions.paths],
        "sections_seen": len(predictions.spans),
        "sections_changed": len(decisions),
        "contacts_changed": changed_contacts,
        "decisions": [asdict(decision) for decision in decisions],
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUTPUT_PATH, "wt", encoding="utf-8", compresslevel=9) as handle:
        json.dump(payload, handle, separators=(",", ":"), sort_keys=True)
    print(json.dumps({name: payload[name] for name in ("sections_seen", "sections_changed", "contacts_changed")}, indent=2))


if __name__ == "__main__":
    main()
