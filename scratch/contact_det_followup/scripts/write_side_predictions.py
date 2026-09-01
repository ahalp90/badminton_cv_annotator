"""Save label-free whole-rally player-side decisions."""

from __future__ import annotations

import gzip
import json
import subprocess
from dataclasses import asdict

from scratch.contact_det_followup.scripts.prediction_io import (
    REPO_ROOT,
    load_frozen_test_predictions,
)
from scratch.contact_det_followup.scripts.side_rules import simple_alternation_decisions

OUTPUT_PATH = REPO_ROOT / "scratch/contact_det_followup/results/simple_side_decisions.json.gz"
CONFIG_PATH = REPO_ROOT / "scratch/contact_det_followup/configs/side_rule.json"


def _repository_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def build_decision_payload() -> dict[str, object]:
    """Choose whole-rally sides without reading labels."""
    predictions = load_frozen_test_predictions()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if (
        config.get("schema") != "contact-detector-side-rule/1"
        or config.get("rule") != "simple_alternation_vote"
    ):
        raise ValueError("Player-side config has another schema or rule")
    minimum_vote_gap = int(config["minimum_vote_gap"])
    decisions = tuple(
        decision
        for decision in simple_alternation_decisions(predictions.spans)
        if decision.score_gap >= minimum_vote_gap
    )
    changed_contacts = sum(
        before != after
        for decision in decisions
        for before, after in zip(decision.sides_before, decision.sides_after, strict=True)
    )
    return {
        "schema": "contact-detector-side-decisions/1",
        "status": "complete",
        "run_id": "simple-alternation-vote",
        "repository_commit": _repository_commit(),
        "prediction_source_commit": predictions.source_commit,
        "labels_read": False,
        "rule": config["rule"],
        "rule_description": "Choose the alternating Top/Bot pattern that matches more independent side guesses; keep ties unchanged.",
        "minimum_vote_gap": minimum_vote_gap,
        "config": str(CONFIG_PATH.relative_to(REPO_ROOT)),
        "source_predictions": str(predictions.path.relative_to(REPO_ROOT)),
        "sections_seen": len(predictions.spans),
        "sections_changed": len(decisions),
        "contacts_changed": changed_contacts,
        "decisions": [asdict(decision) for decision in decisions],
    }


def main() -> None:
    """Write the compact side-decision record."""
    payload = build_decision_payload()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUTPUT_PATH, "wt", encoding="utf-8", compresslevel=9) as handle:
        json.dump(payload, handle, separators=(",", ":"), sort_keys=True)
    print(
        json.dumps(
            {
                "sections_seen": payload["sections_seen"],
                "sections_changed": payload["sections_changed"],
                "contacts_changed": payload["contacts_changed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
