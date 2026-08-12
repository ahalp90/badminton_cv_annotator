"""Tests for rally-start event truth and failure-safe persistence."""

from __future__ import annotations

import csv
from dataclasses import replace
import gzip
from pathlib import Path

import pytest

import annotator.rally_start_events as events
from annotator.broadcast_timeline_labels import VideoMetadata


METADATA = VideoMetadata("sset_01", 25.0, 1_000)


def _target(
    rally: int = 1,
    *,
    review_start: int = 100,
    review_end: int = 200,
) -> events.RallyStartTarget:
    return events.RallyStartTarget(
        key=events.RallyStartKey("sset_01", "set1", rally),
        metadata=METADATA,
        gt_first_frame=150 + rally,
        gt_first_type_en="short service",
        gt_first_flaw=False,
        timeline_truth="live",
        timeline_interval_start=120,
        timeline_interval_end=300,
        preceding_truth="replay",
        live_transition_frame=120,
        review_start_frame=review_start,
        review_end_frame=review_end,
        pilot_stratum="full-audit-only",
        note=f"set1 rally {rally}",
    )


def _reviewed(
    target: events.RallyStartTarget,
    visibility: events.ServeVisibility,
) -> events.RallyStartDecision:
    values: dict[str, object] = {
        "key": target.key,
        "review_status": events.ReviewStatus.REVIEWED,
        "serve_visibility": visibility,
        "confidence": (
            events.Confidence.UNCERTAIN
            if visibility is events.ServeVisibility.UNCERTAIN
            else events.Confidence.CERTAIN
        ),
        "review_note": f"reviewed {visibility.value}",
    }
    if visibility is events.ServeVisibility.VISIBLE:
        values["visible_serve_frame"] = 151
    elif visibility is events.ServeVisibility.BROADCAST_OMITTED:
        values["broadcast_return_frame"] = 140
        values["first_visible_rally_frame"] = 145
    return events.RallyStartDecision(**values)  # type: ignore[arg-type]


def _target_row(target: events.RallyStartTarget) -> dict[str, str]:
    row = {column: "" for column in events.TARGET_COLUMNS}
    row.update({
        "video_id": target.key.video_id,
        "fps": str(target.metadata.fps),
        "frame_count": str(target.metadata.frame_count),
        "set_id": target.key.set_id,
        "rally": str(target.key.rally),
        "gt_first_frame": str(target.gt_first_frame),
        "gt_first_ball_round": "1",
        "gt_first_type_raw": "1",
        "gt_first_type_en": target.gt_first_type_en,
        "gt_first_flaw": str(target.gt_first_flaw).lower(),
        "gt_first_server": "1",
        "committed_status": "serve_missed_later_strokes_matched",
        "later_strokes_matched": "2",
        "timeline_truth": target.timeline_truth,
        "timeline_interval_start": str(target.timeline_interval_start),
        "timeline_interval_end": str(target.timeline_interval_end),
        "preceding_truth": target.preceding_truth,
        "live_transition_frame": str(target.live_transition_frame),
        "frames_from_live_transition": str(
            target.gt_first_frame - target.live_transition_frame
        ),
        "review_start_frame": str(target.review_start_frame),
        "review_end_frame": str(target.review_end_frame),
        "pilot_stratum": target.pilot_stratum,
        "note": target.note,
        "review_status": "pending",
    })
    return row


def _write_rows(
    path: Path,
    columns: tuple[str, ...],
    rows: list[dict[str, str]],
) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("visibility", list(events.ServeVisibility))
def test_four_reviewed_states_accept_only_their_valid_markers(
    visibility: events.ServeVisibility,
) -> None:
    target = _target()
    decision = _reviewed(target, visibility)

    events.validate_decision(decision, target)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda target: replace(
                events.RallyStartDecision.pending(target.key),
                review_note="stray",
            ),
            "pending rows require blank",
        ),
        (
            lambda target: replace(
                _reviewed(target, events.ServeVisibility.VISIBLE),
                visible_serve_frame=None,
            ),
            "visible requires only",
        ),
        (
            lambda target: replace(
                _reviewed(target, events.ServeVisibility.BROADCAST_OMITTED),
                broadcast_return_frame=146,
            ),
            "ordered return",
        ),
        (
            lambda target: replace(
                _reviewed(target, events.ServeVisibility.OFF_FRAME),
                visible_serve_frame=151,
            ),
            "off-frame leaves all",
        ),
        (
            lambda target: replace(
                _reviewed(target, events.ServeVisibility.UNCERTAIN),
                confidence=events.Confidence.CERTAIN,
            ),
            "uncertain requires confidence=uncertain",
        ),
        (
            lambda target: replace(
                _reviewed(target, events.ServeVisibility.VISIBLE),
                review_note="  ",
            ),
            "require review_note",
        ),
        (
            lambda target: replace(
                _reviewed(target, events.ServeVisibility.VISIBLE),
                visible_serve_frame=target.review_end_frame,
            ),
            "outside the review window",
        ),
    ],
)
def test_decision_validation_rejects_one_field_contract_breaks(
    mutator: object,
    message: str,
) -> None:
    target = _target()
    decision = mutator(target)  # type: ignore[operator]

    with pytest.raises(ValueError, match=message):
        events.validate_decision(decision, target)


def test_target_reader_requires_exact_pending_rows_and_one_video(tmp_path: Path) -> None:
    first = _target(1)
    second = _target(2)
    path = tmp_path / "targets.csv.gz"
    _write_rows(path, events.TARGET_COLUMNS, [_target_row(first), _target_row(second)])

    assert events.read_target_csv(path) == [first, second]

    reviewed = _target_row(first)
    reviewed.update(events.decision_to_row(_reviewed(first, events.ServeVisibility.VISIBLE)))
    _write_rows(path, events.TARGET_COLUMNS, [reviewed])
    with pytest.raises(ValueError, match="full targets must contain pending"):
        events.read_target_csv(path)

    inconsistent = _target_row(first)
    inconsistent["frames_from_live_transition"] = "999"
    _write_rows(path, events.TARGET_COLUMNS, [inconsistent])
    with pytest.raises(ValueError, match="frames_from_live_transition 999"):
        events.read_target_csv(path)

    other = _target_row(second)
    other["video_id"] = "sset_15"
    _write_rows(path, events.TARGET_COLUMNS, [_target_row(first), other])
    with pytest.raises(ValueError, match="target metadata differs"):
        events.read_target_csv(path)


@pytest.mark.parametrize("width_change", (-1, 1), ids=("missing", "extra"))
def test_decision_reader_rejects_malformed_width(
    tmp_path: Path,
    width_change: int,
) -> None:
    target = _target()
    path = tmp_path / "decisions.csv.gz"
    row = events.decision_to_row(events.RallyStartDecision.pending(target.key))
    values = [row[column] for column in events.DECISION_COLUMNS]
    values = values[:-1] if width_change < 0 else [*values, "unexpected"]
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(events.DECISION_COLUMNS)
        writer.writerow(values)

    with pytest.raises(ValueError, match="row width does not match header"):
        events.read_decision_csv(path)


def test_decisions_require_unique_exact_keys_and_return_target_order(tmp_path: Path) -> None:
    targets = [_target(1), _target(2)]
    second = events.RallyStartDecision.pending(targets[1].key)
    first = events.RallyStartDecision.pending(targets[0].key)
    path = tmp_path / "decisions.csv.gz"
    _write_rows(
        path,
        events.DECISION_COLUMNS,
        [events.decision_to_row(second), events.decision_to_row(first)],
    )

    assert events.read_decision_csv(path, targets) == [first, second]

    _write_rows(
        path,
        events.DECISION_COLUMNS,
        [events.decision_to_row(first), events.decision_to_row(first)],
    )
    with pytest.raises(ValueError, match="duplicate decision key"):
        events.read_decision_csv(path, targets)

    _write_rows(path, events.DECISION_COLUMNS, [events.decision_to_row(first)])
    with pytest.raises(ValueError, match="decision key mismatch"):
        events.read_decision_csv(path, targets)


def test_seed_preserves_reviewed_rows_and_fills_remaining_keys() -> None:
    targets = [_target(1), _target(2), _target(3)]
    reviewed = _reviewed(targets[1], events.ServeVisibility.OFF_FRAME)

    seed = events.build_decision_seed(targets, [reviewed])

    assert [decision.key for decision in seed] == [target.key for target in targets]
    assert seed[1] == reviewed
    assert [decision.review_status for decision in seed] == [
        events.ReviewStatus.PENDING,
        events.ReviewStatus.REVIEWED,
        events.ReviewStatus.PENDING,
    ]


def test_atomic_writer_is_deterministic_and_preserves_complex_notes(tmp_path: Path) -> None:
    target = _target()
    decision = replace(
        _reviewed(target, events.ServeVisibility.VISIBLE),
        review_note='contact, then "rally"\nsecond line ✓',
    )
    path = tmp_path / "decisions.csv.gz"

    events.write_decision_csv(path, [decision], [target])
    first = path.read_bytes()
    events.write_decision_csv(path, [decision], [target])

    assert path.read_bytes() == first
    assert events.read_decision_csv(path, [target]) == [decision]
    assert not list(tmp_path.glob(".*.tmp.csv.gz"))


def test_initialization_never_replaces_an_existing_decision_table(tmp_path: Path) -> None:
    target = _target()
    pending = events.RallyStartDecision.pending(target.key)
    reviewed = _reviewed(target, events.ServeVisibility.VISIBLE)
    path = tmp_path / "decisions.csv.gz"
    events.initialize_decision_csv(path, [pending], [target])
    original = path.read_bytes()

    with pytest.raises(FileExistsError):
        events.initialize_decision_csv(path, [reviewed], [target])

    assert path.read_bytes() == original
    assert events.read_decision_csv(path, [target]) == [pending]
    assert not list(tmp_path.glob(".*.tmp.csv.gz"))


def test_writer_rejects_a_canonical_timeline_filename(tmp_path: Path) -> None:
    target = _target()
    path = tmp_path / "sset_01_broadcast_timeline_labels.csv.gz"

    with pytest.raises(ValueError, match="protected canonical timeline name"):
        events.write_decision_csv(
            path,
            [events.RallyStartDecision.pending(target.key)],
            [target],
        )

    assert not path.exists()


def test_failed_candidate_reload_preserves_destination_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target()
    pending = events.RallyStartDecision.pending(target.key)
    reviewed = _reviewed(target, events.ServeVisibility.VISIBLE)
    path = tmp_path / "decisions.csv.gz"
    events.write_decision_csv(path, [pending], [target])
    original = path.read_bytes()
    real_reader = events.read_decision_csv

    def corrupt_temporary_read(
        candidate: Path,
        targets: list[events.RallyStartTarget] | tuple[events.RallyStartTarget, ...] | None = None,
    ) -> list[events.RallyStartDecision]:
        if candidate != path:
            return []
        return real_reader(candidate, targets)

    monkeypatch.setattr(events, "read_decision_csv", corrupt_temporary_read)
    with pytest.raises(RuntimeError, match="round trip changed"):
        events.write_decision_csv(path, [reviewed], [target])

    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp.csv.gz"))


def test_session_resumes_by_key_and_ignores_overlapping_window_position() -> None:
    targets = [
        _target(1, review_start=100, review_end=200),
        _target(2, review_start=110, review_end=210),
        _target(3, review_start=115, review_end=215),
    ]
    decisions = [
        _reviewed(targets[0], events.ServeVisibility.VISIBLE),
        events.RallyStartDecision.pending(targets[1].key),
        _reviewed(targets[2], events.ServeVisibility.OFF_FRAME),
    ]

    session = events.RallyStartAuditSession(targets, decisions)

    assert session.row_index == 1
    assert session.current_target.key == targets[1].key
    assert session.reviewed_count == 2
    assert session.pending_count == 1
    assert session.move_row(1) == 2
    assert session.current_target.key == targets[2].key


def test_session_marker_capture_checks_state_window_and_order() -> None:
    target = _target()
    session = events.RallyStartAuditSession(
        [target],
        [events.RallyStartDecision.pending(target.key)],
    )

    with pytest.raises(ValueError, match="select visible"):
        session.capture_visible_serve(151)
    session.select_visibility(events.ServeVisibility.VISIBLE)
    with pytest.raises(ValueError, match="outside review window"):
        session.capture_visible_serve(target.review_end_frame)
    session.capture_visible_serve(151)
    session.set_note("contact is visible")
    session.validate_draft()

    session.select_visibility(events.ServeVisibility.BROADCAST_OMITTED)
    session.capture_broadcast_return(160)
    session.capture_first_visible_rally(159)
    session.set_note("rally returns after the serve")
    with pytest.raises(ValueError, match="ordered return"):
        session.validate_draft()


def test_failed_session_save_preserves_all_live_state(tmp_path: Path) -> None:
    targets = [_target(1), _target(2)]
    decisions = [events.RallyStartDecision.pending(target.key) for target in targets]
    session = events.RallyStartAuditSession(targets, decisions)
    session.select_visibility(events.ServeVisibility.VISIBLE)
    session.capture_visible_serve(151)
    session.set_note("visible contact")
    before = (
        session.decisions,
        session.row_index,
        session.draft,
        session.decision_path,
    )

    def fail_writer(
        _path: Path,
        _decisions: list[events.RallyStartDecision],
        _targets: tuple[events.RallyStartTarget, ...],
    ) -> None:
        raise OSError("simulated write failure")

    with pytest.raises(OSError, match="simulated write failure"):
        session.save(tmp_path / "decisions.csv.gz", fail_writer)

    assert (
        session.decisions,
        session.row_index,
        session.draft,
        session.decision_path,
    ) == before


def test_session_save_and_one_step_undo_are_atomic(tmp_path: Path) -> None:
    targets = [_target(1), _target(2)]
    pending = [events.RallyStartDecision.pending(target.key) for target in targets]
    path = tmp_path / "decisions.csv.gz"
    events.initialize_decision_csv(path, pending, targets)
    session = events.RallyStartAuditSession(targets, pending)
    session.select_visibility(events.ServeVisibility.VISIBLE)
    session.capture_visible_serve(151)
    session.set_note("visible contact")

    saved = session.save(path)

    assert saved.review_status is events.ReviewStatus.REVIEWED
    assert session.row_index == 1
    assert session.reviewed_count == 1
    assert events.read_decision_csv(path, targets) == list(session.decisions)

    before_failed_undo = (session.decisions, session.row_index, path.read_bytes())

    def fail_undo_writer(
        _path: Path,
        _decisions: list[events.RallyStartDecision],
        _targets: tuple[events.RallyStartTarget, ...],
    ) -> None:
        raise OSError("simulated undo failure")

    with pytest.raises(OSError, match="simulated undo failure"):
        session.undo(path, fail_undo_writer)
    assert (session.decisions, session.row_index, path.read_bytes()) == before_failed_undo

    before_wrong_path = (session.decisions, session.row_index, path.read_bytes())
    with pytest.raises(ValueError, match="session decision path"):
        session.undo(tmp_path / "other.csv.gz")
    assert (session.decisions, session.row_index, path.read_bytes()) == before_wrong_path

    assert session.undo(path) is events.UndoResult.SAVED_ROW_RESTORED
    assert session.row_index == 0
    assert session.decisions == tuple(pending)
    assert events.read_decision_csv(path, targets) == pending
    assert session.undo(path) is events.UndoResult.NOTHING_TO_UNDO


def test_saving_the_final_pending_row_keeps_focus_on_that_row(tmp_path: Path) -> None:
    target = _target()
    pending = events.RallyStartDecision.pending(target.key)
    path = tmp_path / "decisions.csv.gz"
    events.initialize_decision_csv(path, [pending], [target])
    session = events.RallyStartAuditSession([target], [pending], path)
    session.select_visibility(events.ServeVisibility.OFF_FRAME)
    session.set_note("contact occurs below the camera frame")

    session.save(path)

    assert session.pending_count == 0
    assert session.row_index == 0
    assert session.current_decision.review_status is events.ReviewStatus.REVIEWED


def test_undo_clears_unsaved_draft_before_considering_saved_history(tmp_path: Path) -> None:
    target = _target()
    pending = events.RallyStartDecision.pending(target.key)
    session = events.RallyStartAuditSession([target], [pending])
    session.select_visibility(events.ServeVisibility.OFF_FRAME)

    def unexpected_writer(
        _path: Path,
        _decisions: list[events.RallyStartDecision],
        _targets: tuple[events.RallyStartTarget, ...],
    ) -> None:
        raise AssertionError("draft undo must not write")

    assert session.undo(tmp_path / "unused.csv.gz", unexpected_writer) is events.UndoResult.DRAFT_CLEARED
    assert session.draft == pending
    assert session.undo(tmp_path / "unused.csv.gz", unexpected_writer) is events.UndoResult.NOTHING_TO_UNDO
