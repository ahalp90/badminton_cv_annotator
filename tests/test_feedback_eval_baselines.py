"""The non-model baselines that calibrate the BERTScore band.

What matters here is that each baseline is what it claims to be. A `constant`
baseline that quietly varied by clip, or a `random_template` that happened to
draw the clip's own correction, would flatter the model version it is supposed
to bracket -- and the failure would be invisible in the resulting numbers.
"""
import json
from pathlib import Path

import pytest

from feedback_eval.baselines import (
    BASELINES,
    CONSTANT_FEEDBACK,
    BaselineError,
    generate,
    write_predictions,
)
from feedback_eval.contracts import ReferenceRecord
from feedback_eval.records import load_predictions

REPO = Path(__file__).resolve().parent.parent
SHUTTLESET_LIBRARY = (
    REPO / "data" / "feedback_eval" / "templates" / "badminton_singles_shuttleset_v1.jsonl"
)

CORRECTIONS = ("Get the racket up sooner.", "Take the shuttle in front.", "Recover to base.")


def _references(n=5):
    return tuple(
        ReferenceRecord(
            clip_id=f"c{i}",
            player_ids=("alice", "bob"),
            references=(f"reference one for {i}", f"reference two for {i}"),
            source="template",
        )
        for i in range(n)
    )


def test_every_named_baseline_generates(tmp_path):
    for baseline in BASELINES:
        rows = generate(_references(), baseline, corrections=CORRECTIONS, seed=1)
        assert len(rows) == 5


def test_a_baseline_covers_every_clip_in_the_reference_set():
    """A baseline that skipped clips would not be comparable with a model run."""
    references = _references(7)
    rows = generate(references, "constant")
    assert [clip_id for clip_id, _ in rows] == [r.clip_id for r in references]


# --- what each baseline actually is -----------------------------------------


def test_the_empty_baseline_returns_nothing_so_it_scores_a_hard_zero():
    rows = generate(_references(), "empty")
    assert {feedback for _, feedback in rows} == {""}


def test_the_constant_baseline_is_identical_on_every_clip():
    """The whole point: whatever it scores, it earned without reading the rally."""
    rows = generate(_references(), "constant")
    assert {feedback for _, feedback in rows} == {CONSTANT_FEEDBACK}


def test_the_constant_feedback_names_no_specific_fault():
    """If it described a fault it would be right on some clips by accident."""
    lowered = CONSTANT_FEEDBACK.lower()
    assert not any(word in lowered for word in ("smash", "serve", "net shot", "lift", "clear"))


def test_the_random_template_baseline_draws_real_coaching_text():
    rows = generate(_references(20), "random_template", corrections=CORRECTIONS, seed=3)
    assert {feedback for _, feedback in rows} <= set(CORRECTIONS)


def test_the_random_template_baseline_varies_across_clips():
    rows = generate(_references(40), "random_template", corrections=CORRECTIONS, seed=3)
    assert len({feedback for _, feedback in rows}) > 1


def test_the_oracle_baseline_copies_the_clips_own_reference():
    references = _references(3)
    rows = generate(references, "oracle")
    assert [feedback for _, feedback in rows] == [r.references[0] for r in references]


# --- reproducibility --------------------------------------------------------


def test_the_same_seed_reproduces_the_same_random_baseline():
    """A baseline that cannot be regenerated from its arguments is not a baseline."""
    first = generate(_references(30), "random_template", corrections=CORRECTIONS, seed=11)
    second = generate(_references(30), "random_template", corrections=CORRECTIONS, seed=11)
    assert first == second


def test_a_different_seed_changes_the_random_baseline():
    references = _references(60)
    first = generate(references, "random_template", corrections=CORRECTIONS, seed=1)
    second = generate(references, "random_template", corrections=CORRECTIONS, seed=2)
    assert first != second


# --- guards -----------------------------------------------------------------


def test_an_unknown_baseline_is_refused():
    with pytest.raises(BaselineError, match="unknown baseline"):
        generate(_references(), "wishful")


def test_the_random_template_baseline_refuses_an_empty_library():
    with pytest.raises(BaselineError, match="needs a template library"):
        generate(_references(), "random_template", corrections=())


# --- output -----------------------------------------------------------------


def test_written_predictions_load_back_through_the_harness(tmp_path):
    rows = generate(_references(4), "constant")
    path = write_predictions(tmp_path / "p.jsonl", rows)
    loaded = load_predictions(path)
    assert [p.clip_id for p in loaded] == ["c0", "c1", "c2", "c3"]
    assert {p.feedback for p in loaded} == {CONSTANT_FEEDBACK}


def test_empty_predictions_survive_the_round_trip_as_empty(tmp_path):
    """scoring.score_run keys its hard zero off a falsy feedback string."""
    path = write_predictions(tmp_path / "p.jsonl", generate(_references(2), "empty"))
    assert all(not prediction.feedback for prediction in load_predictions(path))


def test_written_predictions_are_one_json_object_per_line(tmp_path):
    path = write_predictions(tmp_path / "p.jsonl", generate(_references(3), "constant"))
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert all(set(json.loads(line)) == {"clip_id", "feedback"} for line in lines)
