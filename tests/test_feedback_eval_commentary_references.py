"""Building the reference set from the COSC595 commentary lane.

The failure that matters here is a quiet one: a reference set that dropped a
third of its rallies still scores, and the mean it produces looks fine. Every
drop is counted and every disagreement between the three input files is an
error.
"""
import json

import pytest

from feedback_eval.commentary_references import (
    CommentaryReferenceError,
    build_references,
    clip_id_for,
    load_player_map,
    references_from_chunk,
    write_references,
)
from feedback_eval.records import load_references

PAIRS_HEADER = "video_id,rally_id,rally_start,rally_end,chunk_id,commentary_start,commentary_end"


def _pairs(tmp_path, rows):
    path = tmp_path / "rally_commentary_pairs.csv"
    lines = [PAIRS_HEADER]
    for video_id, rally_id, chunk_id in rows:
        lines.append(f"{video_id},{rally_id},0,100,{chunk_id},1.0,2.0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _chunks(tmp_path, video_id="vid1", chunks=None):
    directory = tmp_path / "chunks"
    directory.mkdir(exist_ok=True)
    if chunks is None:
        chunks = [_chunk("k1")]
    (directory / f"{video_id}.json").write_text(json.dumps(chunks), encoding="utf-8")
    return directory


def _chunk(chunk_id, **overrides):
    chunk = {
        "chunk_id": chunk_id,
        "start": 1.0,
        "end": 2.0,
        "text": "and thats a lovely drop shot there",
        "text_clean": "That is a lovely drop shot.",
        "alt_phrasings": ["A beautiful drop shot.", "Lovely drop shot played there."],
        "bert_f1": 0.91,
        "clean_pass": True,
    }
    chunk.update(overrides)
    return chunk


def _players(tmp_path, rows=(("vid1", 0, "axelsen"),)):
    path = tmp_path / "players.csv"
    lines = ["video_id,rally_id,player_id"]
    lines += [f"{video_id},{rally_id},{player_id}" for video_id, rally_id, player_id in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_builds_a_record_from_a_paired_rally(tmp_path):
    records, report = build_references(
        _pairs(tmp_path, [("vid1", 0, "k1")]),
        _chunks(tmp_path),
        load_player_map(_players(tmp_path)),
    )
    assert report.kept == 1
    assert records[0].clip_id == "vid1_r0"
    assert records[0].player_ids == ("axelsen",)


def test_records_are_written_as_commentary_never_expert(tmp_path):
    """Commentary is descriptive; filing it as expert would overstate the claim."""
    records, _ = build_references(
        _pairs(tmp_path, [("vid1", 0, "k1")]),
        _chunks(tmp_path),
        load_player_map(_players(tmp_path)),
    )
    assert records[0].source == "commentary"


def test_cleaned_text_leads_and_alternate_phrasings_follow(tmp_path):
    records, _ = build_references(
        _pairs(tmp_path, [("vid1", 0, "k1")]),
        _chunks(tmp_path),
        load_player_map(_players(tmp_path)),
    )
    assert records[0].references == (
        "That is a lovely drop shot.",
        "A beautiful drop shot.",
        "Lovely drop shot played there.",
    )


def test_references_from_chunk_drops_duplicate_phrasings():
    chunk = _chunk("k1", text_clean="Same text.", alt_phrasings=["Same text.", "Different."])
    assert references_from_chunk(chunk) == ("Same text.", "Different.")


def test_references_from_chunk_ignores_blank_phrasings():
    chunk = _chunk("k1", alt_phrasings=["   ", "Real phrasing."])
    assert references_from_chunk(chunk) == ("That is a lovely drop shot.", "Real phrasing.")


def test_unpaired_rallies_are_counted_not_silently_lost(tmp_path):
    """The pairing stage keeps replay-masked rallies unpaired by design."""
    _, report = build_references(
        _pairs(tmp_path, [("vid1", 0, "k1"), ("vid1", 1, "")]),
        _chunks(tmp_path),
        load_player_map(_players(tmp_path, [("vid1", 0, "axelsen"), ("vid1", 1, "axelsen")])),
    )
    assert (report.kept, report.unpaired, report.dropped) == (1, 1, 1)


def test_chunks_that_failed_the_cleaning_gate_are_dropped(tmp_path):
    _, report = build_references(
        _pairs(tmp_path, [("vid1", 0, "k1")]),
        _chunks(tmp_path, chunks=[_chunk("k1", clean_pass=False, bert_f1=0.42)]),
        load_player_map(_players(tmp_path)),
    )
    assert (report.kept, report.not_clean_pass) == (0, 1)


def test_failed_clean_chunks_can_be_kept_explicitly(tmp_path):
    records, report = build_references(
        _pairs(tmp_path, [("vid1", 0, "k1")]),
        _chunks(tmp_path, chunks=[_chunk("k1", clean_pass=False)]),
        load_player_map(_players(tmp_path)),
        require_clean_pass=False,
    )
    assert (report.kept, len(records)) == (1, 1)


def test_a_rally_naming_a_missing_chunk_is_counted(tmp_path):
    _, report = build_references(
        _pairs(tmp_path, [("vid1", 0, "absent")]),
        _chunks(tmp_path),
        load_player_map(_players(tmp_path)),
    )
    assert (report.kept, report.chunk_missing) == (0, 1)


def test_a_rally_with_no_player_is_counted(tmp_path):
    _, report = build_references(
        _pairs(tmp_path, [("vid1", 0, "k1"), ("vid1", 1, "k2")]),
        _chunks(tmp_path, chunks=[_chunk("k1"), _chunk("k2")]),
        load_player_map(_players(tmp_path)),
    )
    assert (report.kept, report.no_player) == (1, 1)


def test_a_chunk_with_no_usable_text_is_counted(tmp_path):
    _, report = build_references(
        _pairs(tmp_path, [("vid1", 0, "k1")]),
        _chunks(tmp_path, chunks=[_chunk("k1", text_clean="  ", alt_phrasings=[])]),
        load_player_map(_players(tmp_path)),
    )
    assert (report.kept, report.blank_text) == (0, 1)


def test_player_map_keeps_both_players_of_a_singles_rally(tmp_path):
    """Two rows for one rally is the normal case, not a conflict."""
    path = _players(tmp_path, [("vid1", 0, "axelsen"), ("vid1", 0, "momota")])
    assert load_player_map(path) == {("vid1", 0): ("axelsen", "momota")}


def test_player_map_collapses_a_repeated_identical_row(tmp_path):
    path = _players(tmp_path, [("vid1", 0, "axelsen"), ("vid1", 0, "axelsen")])
    assert load_player_map(path) == {("vid1", 0): ("axelsen",)}


def test_both_rally_players_reach_the_reference_record(tmp_path):
    records, _ = build_references(
        _pairs(tmp_path, [("vid1", 0, "k1")]),
        _chunks(tmp_path),
        load_player_map(_players(tmp_path, [("vid1", 0, "axelsen"), ("vid1", 0, "momota")])),
    )
    assert records[0].player_ids == ("axelsen", "momota")


def test_player_map_rejects_a_missing_column(tmp_path):
    path = tmp_path / "players.csv"
    path.write_text("video_id,rally_id\nvid1,0\n", encoding="utf-8")
    with pytest.raises(CommentaryReferenceError, match="missing column"):
        load_player_map(path)


def test_player_map_rejects_a_non_integer_rally_id(tmp_path):
    path = tmp_path / "players.csv"
    path.write_text("video_id,rally_id,player_id\nvid1,first,axelsen\n", encoding="utf-8")
    with pytest.raises(CommentaryReferenceError, match="not an integer"):
        load_player_map(path)


def test_run_id_prefixes_the_clip_id():
    """rally_id is a list position, so it is only stable within one run."""
    assert clip_id_for("vid1", 3) == "vid1_r3"
    assert clip_id_for("vid1", 3, run_id="run7") == "run7_vid1_r3"


def test_video_ids_are_never_numerically_coerced(tmp_path):
    """The rally dataset contract: '0012' and '12' are different videos."""
    records, _ = build_references(
        _pairs(tmp_path, [("0012", 0, "k1")]),
        _chunks(tmp_path, video_id="0012"),
        load_player_map(_players(tmp_path, [("0012", 0, "axelsen")])),
    )
    assert records[0].clip_id == "0012_r0"


def test_written_references_load_back_through_the_harness(tmp_path):
    """The adapter's output has to satisfy the reader the scorer actually uses."""
    records, _ = build_references(
        _pairs(tmp_path, [("vid1", 0, "k1"), ("vid1", 1, "k2")]),
        _chunks(tmp_path, chunks=[_chunk("k1"), _chunk("k2")]),
        load_player_map(_players(tmp_path, [("vid1", 0, "axelsen"), ("vid1", 1, "momota")])),
    )
    path = write_references(tmp_path / "references.jsonl", records)
    reloaded = load_references(path)
    assert [record.clip_id for record in reloaded] == ["vid1_r0", "vid1_r1"]
    assert {p for record in reloaded for p in record.player_ids} == {"axelsen", "momota"}
    assert reloaded[0].source == "commentary"


def test_refuses_to_write_an_empty_reference_set(tmp_path):
    with pytest.raises(CommentaryReferenceError, match="empty reference set"):
        write_references(tmp_path / "references.jsonl", [])


def test_reports_a_missing_pairs_file(tmp_path):
    with pytest.raises(CommentaryReferenceError, match="no such file"):
        build_references(tmp_path / "absent.csv", tmp_path, {})
