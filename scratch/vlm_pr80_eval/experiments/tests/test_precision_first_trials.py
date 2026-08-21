from __future__ import annotations

from experiments.precision_first_trials import (
    RULE_LADDER,
    choose_rung,
    retained_span_ids,
)


def _record(span_id: int, **overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "span_id": span_id,
        "automatic_record_complete": True,
        "court_present_fraction": 0.9,
        "court_keep_fraction": 0.9,
        "track_visible_fraction": 0.9,
        "outcome_corroborated": True,
    }
    record.update(overrides)
    return record


def test_rule_ladder_is_monotone() -> None:
    records = [
        _record(0),
        _record(1, outcome_corroborated=False),
        _record(2, track_visible_fraction=0.79),
        _record(3, court_keep_fraction=0.79),
        _record(4, automatic_record_complete=False),
    ]

    retained = [set(retained_span_ids(records, rung)) for rung in RULE_LADDER]

    assert retained == [
        {0, 1, 2, 3},
        {0, 1, 2},
        {0, 1},
        {0},
    ]


def _summary(retained: int, correct: int) -> dict[str, object]:
    return {
        "retained_records": retained,
        "correct_complete_records": correct,
    }


def test_choose_rung_maximises_zero_error_development_coverage() -> None:
    scored = {
        "dev-a": {
            "automatic-completeness": _summary(4, 3),
            "scene-support": _summary(3, 3),
            "track-support": _summary(2, 2),
            "outcome-corroboration": _summary(1, 1),
        },
        "dev-b": {
            "automatic-completeness": _summary(4, 4),
            "scene-support": _summary(2, 2),
            "track-support": _summary(1, 1),
            "outcome-corroboration": _summary(1, 1),
        },
    }

    assert choose_rung(scored, ["dev-a", "dev-b"]) == "scene-support"


def test_choose_rung_prefers_stricter_tie() -> None:
    scored = {
        "dev-a": {
            rung: _summary(1, 1)
            for rung in RULE_LADDER
        }
    }

    assert choose_rung(scored, ["dev-a"]) == "outcome-corroboration"


def test_choose_rung_returns_none_when_no_nonempty_rung_has_zero_errors() -> None:
    scored = {
        "dev-a": {
            rung: _summary(1, 0)
            for rung in RULE_LADDER
        }
    }

    assert choose_rung(scored, ["dev-a"]) is None
