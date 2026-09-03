"""Input validation for the feedback-evaluation harness.

Every case here is a way a bad input could still produce a plausible-looking
mean, which is the failure that matters: a number that looks comparable but was
computed over different clips or a different scorer.
"""
import json

import pytest

from feedback_eval.records import RecordError, align, load_predictions, load_references


def _write(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _reference_row(clip_id="clip_1", **overrides):
    row = {
        "clip_id": clip_id,
        "player_ids": ["player_1"],
        "references": ["prepare the racket earlier on the backhand"],
        "source": "template",
    }
    row.update(overrides)
    return row


def test_load_references_reads_every_field(tmp_path):
    path = _write(
        tmp_path / "references.jsonl",
        [_reference_row("clip_1"), _reference_row("clip_2", source="expert")],
    )
    references = load_references(path)
    assert [reference.clip_id for reference in references] == ["clip_1", "clip_2"]
    assert references[0].references == ("prepare the racket earlier on the backhand",)
    assert references[1].source == "expert"


def test_load_references_keeps_every_phrasing(tmp_path):
    path = _write(
        tmp_path / "references.jsonl",
        [_reference_row(references=["get the racket up earlier", "prepare the racket sooner"])],
    )
    assert len(load_references(path)[0].references) == 2


def test_load_references_rejects_a_duplicate_clip_id(tmp_path):
    path = _write(tmp_path / "references.jsonl", [_reference_row("clip_1"), _reference_row("clip_1")])
    with pytest.raises(RecordError, match="duplicate clip_id"):
        load_references(path)


def test_load_references_rejects_an_unknown_source(tmp_path):
    path = _write(tmp_path / "references.jsonl", [_reference_row(source="vibes")])
    with pytest.raises(RecordError, match="source 'vibes' is not one of"):
        load_references(path)


def test_load_references_rejects_an_empty_reference_list(tmp_path):
    path = _write(tmp_path / "references.jsonl", [_reference_row(references=[])])
    with pytest.raises(RecordError, match="non-empty list"):
        load_references(path)


def test_load_references_rejects_a_clip_with_no_players(tmp_path):
    path = _write(tmp_path / "references.jsonl", [_reference_row(player_ids=[])])
    with pytest.raises(RecordError, match="non-empty list"):
        load_references(path)


def test_load_references_keeps_both_players_of_a_singles_rally(tmp_path):
    """A rally involves two players and the split needs to see both."""
    path = _write(tmp_path / "references.jsonl", [_reference_row(player_ids=["alice", "bob"])])
    assert load_references(path)[0].player_ids == ("alice", "bob")


def test_load_references_collapses_a_repeated_player(tmp_path):
    """A player named twice would otherwise count twice in their own mean."""
    path = _write(tmp_path / "references.jsonl", [_reference_row(player_ids=["alice", "alice"])])
    assert load_references(path)[0].player_ids == ("alice",)


def test_load_references_rejects_a_blank_reference(tmp_path):
    path = _write(tmp_path / "references.jsonl", [_reference_row(references=["   "])])
    with pytest.raises(RecordError, match=r"references\[0\]"):
        load_references(path)


def test_load_references_names_the_line_for_invalid_json(tmp_path):
    path = tmp_path / "references.jsonl"
    path.write_text(json.dumps(_reference_row()) + "\nnot json\n", encoding="utf-8")
    with pytest.raises(RecordError, match=":2: invalid JSON"):
        load_references(path)


def test_load_references_rejects_an_empty_file(tmp_path):
    path = tmp_path / "references.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(RecordError, match="no records"):
        load_references(path)


def test_load_predictions_allows_blank_feedback(tmp_path):
    """A model that returns nothing is a real result, not a malformed file."""
    path = _write(tmp_path / "predictions.jsonl", [{"clip_id": "clip_1", "feedback": ""}])
    assert load_predictions(path)[0].feedback == ""


def test_load_predictions_rejects_a_missing_feedback_field(tmp_path):
    path = _write(tmp_path / "predictions.jsonl", [{"clip_id": "clip_1"}])
    with pytest.raises(RecordError, match="'feedback' must be a string"):
        load_predictions(path)


def test_align_pairs_in_reference_order(tmp_path):
    references = load_references(
        _write(tmp_path / "r.jsonl", [_reference_row("clip_1"), _reference_row("clip_2")])
    )
    predictions = load_predictions(
        _write(
            tmp_path / "p.jsonl",
            [{"clip_id": "clip_2", "feedback": "b"}, {"clip_id": "clip_1", "feedback": "a"}],
        )
    )
    pairs = align(references, predictions)
    assert [(reference.clip_id, prediction.feedback) for reference, prediction in pairs] == [
        ("clip_1", "a"),
        ("clip_2", "b"),
    ]


def test_align_rejects_a_missing_prediction(tmp_path):
    """Scoring a subset would still yield a mean -- over a different denominator."""
    references = load_references(
        _write(tmp_path / "r.jsonl", [_reference_row("clip_1"), _reference_row("clip_2")])
    )
    predictions = load_predictions(
        _write(tmp_path / "p.jsonl", [{"clip_id": "clip_1", "feedback": "a"}])
    )
    with pytest.raises(RecordError, match="clip sets differ"):
        align(references, predictions)


def test_align_rejects_an_unexpected_prediction(tmp_path):
    references = load_references(_write(tmp_path / "r.jsonl", [_reference_row("clip_1")]))
    predictions = load_predictions(
        _write(
            tmp_path / "p.jsonl",
            [{"clip_id": "clip_1", "feedback": "a"}, {"clip_id": "clip_9", "feedback": "b"}],
        )
    )
    with pytest.raises(RecordError, match="clip sets differ"):
        align(references, predictions)
