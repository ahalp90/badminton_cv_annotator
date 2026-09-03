"""End-to-end CLI path, exercised through --dry-run so no model is needed."""
import json
from pathlib import Path

import pytest

from feedback_eval.score_cli import main

STUB = Path(__file__).resolve().parent.parent / "experiments" / "feedback_eval" / "stub"


def _run_stub(tmp_path, *extra):
    out = tmp_path / "scores.json"
    exit_code = main(
        [
            "--references", str(STUB / "references.jsonl"),
            "--predictions", str(STUB / "predictions_version_a_stub.jsonl"),
            "--model-version", "A-stub",
            "--dry-run",
            "--out", str(out),
            *extra,
        ]
    )
    assert exit_code == 0
    return json.loads(out.read_text(encoding="utf-8"))


def test_cli_scores_the_stub_set_end_to_end(tmp_path):
    payload = _run_stub(tmp_path)
    assert payload["aggregates"]["n_clips"] == 8
    assert payload["aggregates"]["n_players"] == 3
    assert payload["aggregates"]["n_empty_predictions"] == 1
    assert len(payload["clips"]) == 8


def test_cli_marks_dry_run_output_so_it_is_never_reported_as_a_result(tmp_path):
    assert _run_stub(tmp_path)["dry_run"] is True


def test_stub_faithful_predictions_outscore_the_generic_ones(tmp_path):
    """The harness has to separate on-target feedback from plausible filler.

    Fake scorer, so these are token-overlap numbers, not BERTScore -- this pins
    the wiring and the stub set's discriminating power, not model quality.
    """
    by_clip = {clip["clip_id"]: clip["f1"] for clip in _run_stub(tmp_path)["clips"]}
    faithful = ["stub_0001", "stub_0002", "stub_0006", "stub_0008"]
    filler = ["stub_0003", "stub_0004", "stub_0005", "stub_0007"]
    assert min(by_clip[clip] for clip in faithful) > max(by_clip[clip] for clip in filler)


def test_cli_reports_the_scorer_config_it_used(tmp_path):
    payload = _run_stub(tmp_path, "--model-type", "roberta-large", "--rescale-with-baseline")
    assert payload["scorer"]["model_type"] == "roberta-large"
    assert payload["scorer"]["rescale_with_baseline"] is True


def test_cli_refuses_mismatched_reference_and_prediction_files(tmp_path):
    predictions = tmp_path / "short.jsonl"
    predictions.write_text(
        json.dumps({"clip_id": "stub_0001", "feedback": "only one clip"}) + "\n",
        encoding="utf-8",
    )
    from feedback_eval.records import RecordError

    with pytest.raises(RecordError, match="clip sets differ"):
        main(
            [
                "--references", str(STUB / "references.jsonl"),
                "--predictions", str(predictions),
                "--model-version", "A-stub",
                "--dry-run",
            ]
        )


# --- split-aware scoring ----------------------------------------------------


def _stub_split(tmp_path):
    """Split the stub set by player, then score only the test side."""
    from feedback_eval.records import load_references
    from feedback_eval.splits import build_split, save_split

    references = load_references(STUB / "references.jsonl")
    split = build_split(references, test_players=("stub_player_c",))
    return save_split(tmp_path / "split.json", split), split


def test_cli_scores_only_the_named_side_of_a_split(tmp_path):
    path, split = _stub_split(tmp_path)
    payload = _run_stub(tmp_path, "--split", str(path))
    assert payload["aggregates"]["n_clips"] == len(split.test_clips)
    assert {clip["clip_id"] for clip in payload["clips"]} == set(split.test_clips)


def test_cli_records_which_split_produced_the_run(tmp_path):
    """A score without its split is not a reportable result."""
    path, _ = _stub_split(tmp_path)
    provenance = _run_stub(tmp_path, "--split", str(path))["split"]
    assert provenance["side"] == "test"
    assert provenance["players"] == ["stub_player_c"]


def test_cli_can_score_the_train_side_for_comparison(tmp_path):
    path, split = _stub_split(tmp_path)
    payload = _run_stub(tmp_path, "--split", str(path), "--split-side", "train")
    assert payload["aggregates"]["n_clips"] == len(split.train_clips)
    assert payload["split"]["side"] == "train"


def test_cli_marks_a_split_free_run_as_having_no_split(tmp_path):
    assert _run_stub(tmp_path)["split"] is None


def test_cli_refuses_a_split_naming_clips_the_references_lack(tmp_path):
    from feedback_eval.splits import SplitError

    path = tmp_path / "split.json"
    path.write_text(
        json.dumps(
            {
                "schema": "feedback-eval-split/1",
                "seed": None,
                "train": {"players": ["stub_player_a"], "clips": ["stub_0001"]},
                "test": {"players": ["ghost"], "clips": ["clip_that_does_not_exist"]},
                "discarded_clips": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SplitError, match="absent from the records"):
        main(
            [
                "--references", str(STUB / "references.jsonl"),
                "--predictions", str(STUB / "predictions_version_a_stub.jsonl"),
                "--model-version", "A-stub",
                "--dry-run",
                "--split", str(path),
            ]
        )
