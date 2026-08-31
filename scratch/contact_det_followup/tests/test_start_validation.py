"""Focused checks for the frozen first-contact validation boundary."""

from types import SimpleNamespace

import pytest

from scratch.contact_det.scripts.score_contact_rallies import FixedEvent, FixedSpan
from scratch.contact_det_followup.scripts import score_start_validation as scorer
from scratch.contact_det_followup.scripts import (
    write_start_validation_predictions as writer,
)
from scratch.contact_det_followup.scripts.score_start_model import ActionRow
from scratch.contact_det_full_ds_fit.scripts.rally_start_model import CandidateRow


def _action_rows() -> tuple[ActionRow, ...]:
    candidate = CandidateRow(
        fixture="sset_18",
        group="V",
        fps=30.0,
        span_id=0,
        section_start_frame=100,
        section_end_frame=200,
        prefix_start_frame=80,
        fixed_contact_frame=120,
        frame=90,
        contact_score=0.8,
        fixed_contact_score=0.95,
        kept=False,
        predicted_side="Top",
        fixed_predicted_side="Bot",
        features=(0.8, 0.95, 30.0, -10.0, 100.0, 0.0, 1.0, 1.0, 0.0),
    )
    return (
        ActionRow(candidate, "add", (*candidate.features, 0.0)),
        ActionRow(candidate, "replace", (*candidate.features, 1.0)),
    )


def _frozen_payload(rows: tuple[ActionRow, ...]) -> dict[str, object]:
    return {
        "schema": "contact-detector-start-action-validation-predictions/1",
        "status": "complete",
        "model_id": "shallow_hgb",
        "cutoff": 0.9,
        "feature_names": [
            "candidate_contact_score",
            "fixed_contact_score",
            "frames_before_fixed_at_30_fps",
            "candidate_from_section_start_at_30_fps",
            "section_length_at_30_fps",
            "candidate_already_kept",
            "candidate_side_known",
            "fixed_side_known",
            "candidate_and_fixed_side_match",
            "action_is_replace",
        ],
        "source_files": {},
        "validation_videos": ["sset_18"],
        "scores": [
            {
                "fixture": row.identity[0],
                "span_id": row.identity[1],
                "frame": row.identity[2],
                "action": row.identity[3],
                "score": score,
            }
            for row, score in zip(rows, (0.95, 0.91), strict=True)
        ],
        "selected_action_identities": [list(rows[0].identity)],
        "validation_labels_read": False,
    }


def test_prediction_module_has_no_v_label_loader_or_v_label_path() -> None:
    assert not hasattr(writer, "load_human_labels")
    assert not hasattr(writer, "V_LABEL_PATH")
    assert not hasattr(writer, "VALIDATION_LABEL_PATH")


def test_saved_selection_is_rebuilt_and_checked_before_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _action_rows()
    payload = _frozen_payload(rows)
    baseline_span = FixedSpan(
        "sset_18",
        0,
        100,
        200,
        (
            FixedEvent("sset_18", 120, 0.95, "Bot"),
            FixedEvent("sset_18", 150, 0.95, "Top"),
        ),
    )
    pack = SimpleNamespace(
        group_by_fixture={"sset_18": "V"},
        spans=(baseline_span,),
        events_by_fixture={"sset_18": baseline_span.events},
    )
    monkeypatch.setattr(scorer, "read_json", lambda _path: payload)
    monkeypatch.setattr(scorer, "_verify_source_files", lambda _payload: None)
    monkeypatch.setattr(
        scorer._prediction_writer, "_validate_development_choice", lambda: None
    )
    monkeypatch.setattr(
        scorer._prediction_writer,
        "load_rally_start_model_config",
        lambda _path: SimpleNamespace(selection_cutoffs=(0.9,)),
    )
    monkeypatch.setattr(
        scorer._prediction_writer,
        "_load_validation_videos",
        lambda: ({"fixture": "sset_18"},),
    )
    monkeypatch.setattr(scorer, "load_development_predictions", lambda: pack)
    monkeypatch.setattr(
        scorer._start_model,
        "build_candidate_rows",
        lambda _videos, *, default_group: (rows[0].candidate,),
    )
    monkeypatch.setattr(
        scorer._start_model,
        "build_action_rows",
        lambda _rows: rows,
    )

    _payload, spans, events, action_rows, scores, selections = (
        scorer._rebuild_frozen_choice()
    )

    assert spans == (baseline_span,)
    assert next(iter(events["sset_18"])).frame == 120
    assert action_rows == rows
    assert set(scores) == {row.identity for row in rows}
    assert selections[("sset_18", 0)].identity == rows[0].identity

    payload["selected_action_identities"] = [list(rows[1].identity)]
    with pytest.raises(ValueError, match="do not reproduce"):
        scorer._rebuild_frozen_choice()
