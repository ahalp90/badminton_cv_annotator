"""The coaching-template library and the reference set built from it.

Provenance is the thing under test as much as the join is: a library that is
mostly team-drafted supports a weaker claim than one grounded in coaching
material, and the failure mode is that the difference goes unrecorded.
"""
import json
from pathlib import Path

import pytest

from feedback_eval.records import load_references
from feedback_eval.template_references import (
    TemplateReferenceError,
    build_references,
    load_clip_faults,
    load_clip_players,
)
from feedback_eval.templates import (
    Provenance,
    TemplateError,
    library_summary,
    load_templates,
)

SEED_LIBRARY = (
    Path(__file__).resolve().parent.parent
    / "data" / "feedback_eval" / "templates" / "badminton_singles_v1.jsonl"
)


def _template_row(template_id="late_racket_preparation", **overrides):
    row = {
        "template_id": template_id,
        "skill_area": "overhead_backhand",
        "fault": "Racket preparation begins late.",
        "corrections": ["Start the backswing on the opponent's contact.", "Get the racket up sooner."],
        "provenance": {"kind": "drafted", "detail": "COSC320 team, 3 Sep 2026"},
    }
    row.update(overrides)
    return row


def _write_library(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _write_csv(path, header, rows):
    lines = [",".join(header)] + [",".join(row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --- the shipped seed library ----------------------------------------------


def test_the_seed_library_loads():
    templates = load_templates(SEED_LIBRARY)
    assert len(templates) == 16


def test_every_seed_template_offers_several_phrasings():
    """One phrasing systematically under-scores correct feedback worded differently."""
    for template in load_templates(SEED_LIBRARY).values():
        assert len(template.corrections) >= 3, template.template_id


def test_every_seed_template_declares_its_provenance():
    for template in load_templates(SEED_LIBRARY).values():
        assert template.provenance.kind in {"drafted", "transcript", "publication"}
        assert template.provenance.detail.strip()


def test_the_seed_library_is_honest_that_it_is_undrafted_from_sources():
    """Shipped as team-drafted. If this ever fails, the claim changed -- check why."""
    kinds = {t.provenance.kind for t in load_templates(SEED_LIBRARY).values()}
    assert kinds == {"drafted"}


def test_library_summary_counts_every_provenance_kind():
    summary = library_summary(load_templates(SEED_LIBRARY))
    assert "drafted        16" in summary
    assert "transcript      0" in summary


# --- library validation -----------------------------------------------------


def test_load_templates_rejects_a_duplicate_id(tmp_path):
    path = _write_library(tmp_path / "t.jsonl", [_template_row(), _template_row()])
    with pytest.raises(TemplateError, match="duplicate template_id"):
        load_templates(path)


def test_load_templates_rejects_an_unknown_provenance_kind(tmp_path):
    path = _write_library(
        tmp_path / "t.jsonl",
        [_template_row(provenance={"kind": "vibes", "detail": "somewhere"})],
    )
    with pytest.raises(TemplateError, match="provenance kind 'vibes'"):
        load_templates(path)


def test_load_templates_rejects_provenance_with_no_detail(tmp_path):
    """'From a book' with no citation is not provenance."""
    path = _write_library(
        tmp_path / "t.jsonl",
        [_template_row(provenance={"kind": "publication", "detail": "  "})],
    )
    with pytest.raises(TemplateError, match="needs a detail"):
        load_templates(path)


def test_load_templates_rejects_an_empty_correction_list(tmp_path):
    path = _write_library(tmp_path / "t.jsonl", [_template_row(corrections=[])])
    with pytest.raises(TemplateError, match="non-empty list"):
        load_templates(path)


def test_load_templates_collapses_duplicate_corrections(tmp_path):
    path = _write_library(tmp_path / "t.jsonl", [_template_row(corrections=["Same.", "Same."])])
    assert load_templates(path)["late_racket_preparation"].corrections == ("Same.",)


def test_provenance_rejects_a_blank_detail_directly():
    with pytest.raises(TemplateError, match="needs a detail"):
        Provenance(kind="drafted", detail="")


# --- building references from templates -------------------------------------


def _inputs(tmp_path, faults, players):
    return (
        load_templates(SEED_LIBRARY),
        load_clip_faults(_write_csv(tmp_path / "f.csv", ["clip_id", "template_id"], faults)),
        load_clip_players(_write_csv(tmp_path / "p.csv", ["clip_id", "player_id"], players)),
    )


def test_a_clip_takes_its_templates_corrections_as_references(tmp_path):
    templates, faults, players = _inputs(
        tmp_path, [["c1", "late_racket_preparation"]], [["c1", "axelsen"], ["c1", "momota"]]
    )
    records, report = build_references(templates, faults, players)
    assert report.kept == 1
    assert records[0].references == templates["late_racket_preparation"].corrections
    assert records[0].player_ids == ("axelsen", "momota")


def test_records_are_written_as_template_never_expert(tmp_path):
    """A template says what a coach says about a fault, not about this player."""
    records, _ = build_references(*_inputs(
        tmp_path, [["c1", "late_split_step"]], [["c1", "axelsen"]]
    ))
    assert records[0].source == "template"


def test_a_clip_with_two_faults_unions_both_templates(tmp_path):
    """Feedback naming either real fault on the clip is not wrong."""
    templates, faults, players = _inputs(
        tmp_path,
        [["c1", "late_split_step"], ["c1", "no_base_recovery_after_net"]],
        [["c1", "axelsen"]],
    )
    records, _ = build_references(templates, faults, players)
    expected = len(templates["late_split_step"].corrections) + len(
        templates["no_base_recovery_after_net"].corrections
    )
    assert len(records[0].references) == expected


def test_a_clip_with_no_player_is_skipped_and_counted(tmp_path):
    _, report = build_references(*_inputs(
        tmp_path,
        [["c1", "late_split_step"], ["c2", "late_split_step"]],
        [["c1", "axelsen"]],
    ))
    assert (report.kept, report.no_player) == (1, 1)


def test_an_unknown_template_id_is_a_hard_error(tmp_path):
    """A typo would otherwise silently drop that clip's only reference."""
    with pytest.raises(TemplateReferenceError, match="not in the library"):
        build_references(*_inputs(
            tmp_path, [["c1", "no_such_fault"]], [["c1", "axelsen"]]
        ))


def test_the_report_records_which_provenance_the_references_came_from(tmp_path):
    _, report = build_references(*_inputs(
        tmp_path, [["c1", "late_split_step"]], [["c1", "axelsen"]]
    ))
    assert report.provenance_mix == (("drafted", 1),)


def test_clip_faults_rejects_a_missing_column(tmp_path):
    path = _write_csv(tmp_path / "f.csv", ["clip_id"], [["c1"]])
    with pytest.raises(TemplateReferenceError, match="missing column"):
        load_clip_faults(path)


def test_written_template_references_load_back_through_the_harness(tmp_path):
    from feedback_eval.commentary_references import write_references

    records, _ = build_references(*_inputs(
        tmp_path,
        [["c1", "late_split_step"], ["c2", "low_serve_sits_up"]],
        [["c1", "axelsen"], ["c2", "momota"]],
    ))
    path = write_references(tmp_path / "references.jsonl", records)
    reloaded = load_references(path)
    assert [record.clip_id for record in reloaded] == ["c1", "c2"]
    assert reloaded[0].source == "template"
