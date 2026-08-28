from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from scratch.contact_det.scripts.score_contact_rallies import (
    FixedEvent,
    FixedSpan,
    RallyReference,
)
from scratch.contact_det_full_ds_fit.scripts import (
    score_shuttleset22_test as scorer,
)
from scratch.contact_det_full_ds_fit.scripts.prepare_shuttleset22_predictions import (
    ARTIFACT_IDENTITY_SHA256,
    COMBINED_SCHEMA,
    EXPECTED_FPS,
    INPAINT_RUN_STATE_SHA256,
    MODEL_RESULT_SHA256,
    MODEL_SHA256,
    NEARBY_DISTANCE_AT_30_FPS,
    PREDICTION_OUTPUT_FILENAMES,
    PREDICTION_SCHEMA,
    RUN_STATE_SCHEMA,
    SCORE_CUTOFF,
    SETTING_RESULT_SHA256,
    SOURCE_MANIFEST_SHA256,
    VIDEO_RESULT_SCHEMA,
    SourceSpec,
)
from scratch.contact_det_full_ds_fit.scripts.score_shuttleset22_test import (
    CleanLabels,
    HumanContact,
    HumanRally,
    VerifiedPredictions,
    _rally_category,
    _tree_digest,
    annotation_corpus_sha256,
    load_annotation_rallies,
    load_clean_labels,
    player_side_metrics,
    score_shuttleset22_test,
    timing_metrics,
    validate_frozen_predictions,
    whole_rally_metrics,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_json_gz(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True).encode()
    with (
        path.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped,
    ):
        zipped.write(encoded)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prediction_payload() -> dict[str, object]:
    return {
        "schema": PREDICTION_SCHEMA,
        "video_id": 8,
        "fixture": "8",
        "fps": EXPECTED_FPS,
        "frame_count": 100,
        "spans": [{"span_id": 0, "start_frame": 5, "end_frame": 40}],
        "contacts": [
            {
                "frame": 10,
                "contact_score": 0.95,
                "predicted_side": "Top",
                "span_id": 0,
            }
        ],
    }


def _prediction_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(scorer, "VIDEO_IDS", (8,))
    root = tmp_path / "predictions"
    directory = root / "videos" / "ss22_08"
    prediction = _prediction_payload()
    _write_json_gz(directory / "predictions.json.gz", prediction)
    output_hashes: dict[str, str] = {}
    for relative_path in PREDICTION_OUTPUT_FILENAMES:
        path = directory / relative_path
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(relative_path.encode())
        output_hashes[relative_path] = _sha256(path)
    result = {
        "schema": VIDEO_RESULT_SCHEMA,
        "status": "complete",
        "video_id": 8,
        "fixture": "8",
        "source_commit": "abcdef0",
        "labels_read": False,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "prepared_artifact_identity_sha256": ARTIFACT_IDENTITY_SHA256,
        "inpaint_run_state_sha256": INPAINT_RUN_STATE_SHA256,
        "model_sha256": MODEL_SHA256,
        "model_result_sha256": MODEL_RESULT_SHA256,
        "setting_result_sha256": SETTING_RESULT_SHA256,
        "model_input_fields": ["field"],
        "score_cutoff": SCORE_CUTOFF,
        "nearby_distance_at_30_fps": NEARBY_DISTANCE_AT_30_FPS,
        "frame_count": 100,
        "feature_summary": {},
        "candidate_row_count": 1,
        "kept_contact_count": 1,
        "input_files": [],
        "output_hashes": output_hashes,
    }
    _write_json(directory / "result.json", result)
    combined = {
        "schema": COMBINED_SCHEMA,
        "status": "complete",
        "source_commit": "abcdef0",
        "labels_read": False,
        "video_ids": [8],
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "prepared_artifact_identity_sha256": ARTIFACT_IDENTITY_SHA256,
        "inpaint_run_state_sha256": INPAINT_RUN_STATE_SHA256,
        "model_sha256": MODEL_SHA256,
        "model_result_sha256": MODEL_RESULT_SHA256,
        "setting_result_sha256": SETTING_RESULT_SHA256,
        "score_cutoff": SCORE_CUTOFF,
        "nearby_distance_at_30_fps": NEARBY_DISTANCE_AT_30_FPS,
        "videos": [prediction],
    }
    combined_path = root / "combined_predictions.json.gz"
    _write_json_gz(combined_path, combined)
    combined_hash = _sha256(combined_path)
    monkeypatch.setattr(scorer, "COMBINED_PREDICTIONS_SHA256", combined_hash)
    _write_json(
        root / "run_state.json",
        {
            "schema": RUN_STATE_SCHEMA,
            "status": "complete",
            "expected_video_ids": [8],
            "completed_video_ids": [8],
            "completed_count": 1,
            "combined_prediction_sha256": combined_hash,
        },
    )
    monkeypatch.setattr(
        scorer,
        "load_source_specs",
        lambda _path: (SourceSpec(8, "video_08"),),
    )
    return root


def _rally(
    frames: tuple[int, ...], sides: tuple[str | None, ...] | None = None
) -> HumanRally:
    if sides is None:
        sides = tuple("Top" for _ in frames)
    return HumanRally(
        "set1",
        1,
        tuple(
            HumanContact(frame, side) for frame, side in zip(frames, sides, strict=True)
        ),
        tuple(range(1, len(frames) + 1)),
        tuple(None for _ in frames),
    )


def _labels(rallies: tuple[HumanRally, ...]) -> CleanLabels:
    return CleanLabels(
        {"8": rallies},
        {"8": {field: 0 for field in scorer.POPULATION_FIELDS}},
        "manifest",
        "tree",
    )


def test_frozen_prediction_check_reloads_the_child_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _prediction_tree(tmp_path, monkeypatch)
    source_manifest = tmp_path / "sources.toml"
    source_manifest.write_text("fixed", encoding="utf-8")

    verified = validate_frozen_predictions(root, source_manifest)

    assert verified.source_commit == "abcdef0"
    assert verified.videos[0]["contacts"][0]["frame"] == 10

    child = root / "videos" / "ss22_08" / "predictions.json.gz"
    child.write_bytes(b"changed")
    with pytest.raises(ValueError, match="predictions.json.gz hash differs"):
        validate_frozen_predictions(root, source_manifest)


def test_changed_combined_file_stops_before_source_or_label_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _prediction_tree(tmp_path, monkeypatch)
    combined = root / "combined_predictions.json.gz"
    combined.write_bytes(b"changed")
    touched = False

    def load_sources(_path: Path) -> tuple[SourceSpec, ...]:
        nonlocal touched
        touched = True
        return ()

    monkeypatch.setattr(scorer, "load_source_specs", load_sources)

    with pytest.raises(ValueError, match="combined prediction SHA-256 differs"):
        validate_frozen_predictions(root, tmp_path / "sources.toml")

    assert touched is False


def test_failed_prediction_check_keeps_labels_unopened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_predictions(_root: Path, _manifest: Path) -> VerifiedPredictions:
        raise ValueError("predictions differ")

    labels_touched = False

    def load_labels(*_args: object) -> CleanLabels:
        nonlocal labels_touched
        labels_touched = True
        raise AssertionError("label loader must not run")

    monkeypatch.setattr(scorer, "validate_frozen_predictions", reject_predictions)
    output = tmp_path / "result.json"

    with pytest.raises(ValueError, match="predictions differ"):
        score_shuttleset22_test(
            tmp_path / "predictions",
            tmp_path / "sources.toml",
            tmp_path / "annotations",
            tmp_path / "clean.json.gz",
            output,
            "abcdef0",
            label_loader=load_labels,
        )

    assert labels_touched is False
    failed = json.loads(output.read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["labels_read_started"] is False
    assert failed["error_type"] == "ValueError"


def test_wrapper_records_label_access_and_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scorer, "VIDEO_IDS", (8,))
    combined = tmp_path / "combined_predictions.json.gz"
    combined.write_bytes(b"frozen")
    verified = VerifiedPredictions(
        tmp_path,
        combined,
        "abcdef0",
        (_prediction_payload(),),
        (SourceSpec(8, "video_08"),),
    )
    monkeypatch.setattr(
        scorer,
        "validate_frozen_predictions",
        lambda _root, _manifest: verified,
    )
    monkeypatch.setattr(scorer, "score_predictions", lambda _verified, _labels: {})
    clean_output = tmp_path / "clean.json.gz"
    source_manifest = tmp_path / "sources.toml"
    source_manifest.write_text("fixed", encoding="utf-8")

    def load_labels(
        _annotation_root: Path,
        _sources: object,
        _frame_counts: object,
        output: Path,
    ) -> CleanLabels:
        _write_json_gz(output, {"status": "complete"})
        return _labels((_rally((10,), ("Top",)),))

    output = score_shuttleset22_test(
        tmp_path,
        source_manifest,
        tmp_path / "annotations",
        clean_output,
        tmp_path / "result.json",
        "abcdef0",
        label_loader=load_labels,
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "complete"
    assert result["labels_read_started"] is True
    assert (
        result["label_read_started_ns"]
        > result["inputs"]["combined_prediction_mtime_ns"]
    )
    assert result["inputs"]["clean_label_sha256"] == _sha256(clean_output)


def test_failure_after_label_access_keeps_the_true_boundary_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    combined = tmp_path / "combined_predictions.json.gz"
    combined.write_bytes(b"frozen")
    verified = VerifiedPredictions(
        tmp_path,
        combined,
        "abcdef0",
        (_prediction_payload(),),
        (SourceSpec(8, "video_08"),),
    )
    monkeypatch.setattr(
        scorer,
        "validate_frozen_predictions",
        lambda _root, _manifest: verified,
    )

    def fail_labels(*_args: object) -> CleanLabels:
        raise ValueError("bad labels")

    output = tmp_path / "result.json"
    with pytest.raises(ValueError, match="bad labels"):
        score_shuttleset22_test(
            tmp_path,
            tmp_path / "sources.toml",
            tmp_path / "annotations",
            tmp_path / "clean.json.gz",
            output,
            "abcdef0",
            label_loader=fail_labels,
        )

    failed = json.loads(output.read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["labels_read_started"] is True
    assert failed["error_type"] == "ValueError"
    assert "label_read_started_at_utc" in failed


def test_annotation_cleaning_rejects_whole_bad_rallies_and_orders_rows(
    tmp_path: Path,
) -> None:
    table = tmp_path / "set1.csv"
    table.write_text(
        "rally,frame_num,flaw,ball_round,player_location_y,opponent_location_y,type\n"
        "1,20,,2,700,300,return\n"
        "1,10,,1,200,800,serve\n"
        "2,30,,1,200,800,serve\n"
        "2,150,,2,700,300,return\n"
        "3,40,bad,1,200,800,serve\n"
        "4,50,,1,,800,serve\n"
        "4,60,,2,500,500,return\n"
        "5,70,,1,200,800,serve\n"
        "5,70,,2,700,300,return\n",
        encoding="utf-8",
    )

    rallies, population = load_annotation_rallies(tmp_path, 100)

    assert [rally.stroke_frames for rally in rallies] == [(10, 20), (50, 60)]
    assert [contact.side for contact in rallies[0].contacts] == ["Top", "Bot"]
    assert [contact.side for contact in rallies[1].contacts] == [None, None]
    assert population == {
        "source_contact_rows": 9,
        "usable_contact_rows": 4,
        "excluded_flaw_rows": 1,
        "excluded_invalid_frame_rows": 1,
        "usable_rallies": 2,
        "excluded_incomplete_rallies": 2,
        "excluded_incomplete_rally_rows": 3,
        "excluded_non_monotonic_rallies": 1,
        "excluded_non_monotonic_rally_rows": 2,
    }


def test_clean_loader_authenticates_all_58_but_reads_only_fixed_videos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    annotation_root = tmp_path / "ShuttleSet22"
    set_root = annotation_root / "set"
    set_root.mkdir(parents=True)
    match_rows = ["id,video"]
    for video_id in range(1, 59):
        video_name = f"video_{video_id:02d}"
        match_rows.append(f"{video_id},{video_name}")
        directory = set_root / video_name
        directory.mkdir()
        table = directory / "set1.csv"
        if video_id == 8:
            table.write_text(
                "rally,frame_num,flaw,ball_round,player_location_y,opponent_location_y,type\n"
                "1,10,,1,200,800,serve\n",
                encoding="utf-8",
            )
        else:
            table.write_text("not,a,label\n", encoding="utf-8")
    (set_root / "match.csv").write_text("\n".join(match_rows) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        scorer, "ANNOTATION_CORPUS_SHA256", annotation_corpus_sha256(annotation_root)
    )
    monkeypatch.setattr(scorer, "ANNOTATION_TREE_SHA256", _tree_digest(annotation_root))
    monkeypatch.setattr(scorer, "VIDEO_IDS", (8,))
    monkeypatch.setattr(scorer, "EXPECTED_SOURCE_ROWS", 1)
    monkeypatch.setattr(scorer, "EXPECTED_USABLE_ROWS", 1)
    monkeypatch.setattr(scorer, "EXPECTED_USABLE_RALLIES", 1)
    clean_output = tmp_path / "clean.json.gz"

    labels = load_clean_labels(
        annotation_root,
        (SourceSpec(8, "video_08"),),
        {8: 100},
        clean_output,
    )

    assert labels.rallies_by_fixture["8"][0].stroke_frames == (10,)
    with gzip.open(clean_output, "rt", encoding="utf-8") as source:
        saved = json.load(source)
    assert saved["video_ids"] == [8]
    assert [video["video_id"] for video in saved["videos"]] == [8]


@pytest.mark.parametrize(
    ("expected", "human_sides", "events", "tolerance", "category"),
    [
        ((10, 20), ("Top", "Top"), ((10, "Top"), (20, "Top")), 1, "fully_correct"),
        ((10, 20), ("Top", "Top"), ((10, "Top"),), 1, "missing_contacts_only"),
        (
            (10, 20),
            ("Top", "Top"),
            ((10, "Top"), (20, "Top"), (30, "Top")),
            1,
            "extra_contacts_only",
        ),
        (
            (10, 20),
            ("Top", "Top"),
            ((10, "Top"), (40, "Top")),
            1,
            "timing_mismatch_equal_counts",
        ),
        (
            (10, 20, 30),
            ("Top", "Top", "Top"),
            ((10, "Top"), (50, "Top")),
            1,
            "missing_and_extra_contacts",
        ),
        (
            (10, 20),
            ("Top", "Top"),
            ((10, "Top"), (20, None)),
            1,
            "predicted_side_unanswered",
        ),
        (
            (10, 20),
            ("Top", "Top"),
            ((10, "Top"), (20, "Bot")),
            1,
            "wrong_predicted_side",
        ),
        (
            (10, 20),
            ("Top", None),
            ((10, "Top"), (20, "Top")),
            1,
            "human_side_unassessable",
        ),
    ],
)
def test_whole_rally_outcomes_are_exclusive(
    expected: tuple[int, ...],
    human_sides: tuple[str | None, ...],
    events: tuple[tuple[int, str | None], ...],
    tolerance: int,
    category: str,
) -> None:
    fixed_events = tuple(FixedEvent("8", frame, 0.95, side) for frame, side in events)
    span = FixedSpan("8", 0, 0, 100, fixed_events)
    reference = RallyReference("8", 0, "set1:1", expected)
    human_contacts = tuple(
        HumanContact(frame, side)
        for frame, side in zip(expected, human_sides, strict=True)
    )

    assert _rally_category(span, reference, human_contacts, tolerance) == category


def test_timing_and_player_side_metrics_keep_unknown_human_sides_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(scorer, "VIDEO_IDS", (8,))
    labels = _labels((_rally((10, 20, 30), ("Top", None, "Bot")),))
    events = {
        "8": (
            FixedEvent("8", 11, 0.95, "Top"),
            FixedEvent("8", 20, 0.96, "Bot"),
            FixedEvent("8", 32, 0.97, None),
        )
    }

    timing = timing_metrics(labels, events)["2"]["total"]
    sides = player_side_metrics(labels, events)["2"]["total"]

    assert timing["matched_contacts"] == 3
    assert timing["first_contact_recall"] == 1.0
    assert timing["later_contact_recall"] == 1.0
    assert timing["signed_frame_error"] == {
        "count": 3,
        "mean": 1.0,
        "median": 1.0,
        "minimum": 0,
        "maximum": 2,
    }
    assert sides == {
        "timing_matched_labels": 3,
        "known_human_sides": 2,
        "human_side_coverage": 2 / 3,
        "predicted_side_answers": 1,
        "prediction_coverage": 0.5,
        "correct_player_sides": 1,
        "accuracy_when_both_answered": 1.0,
    }


def test_confidence_curve_reports_retained_and_assessable_denominators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(scorer, "VIDEO_IDS", (8,))
    labels = _labels((_rally((10,), ("Top",)),))
    mapped = FixedSpan("8", 0, 0, 20, (FixedEvent("8", 10, 0.95, "Top"),))
    unmapped = FixedSpan("8", 1, 30, 40, (FixedEvent("8", 35, 0.96, "Top"),))
    events = {"8": mapped.events + unmapped.events}

    result = whole_rally_metrics(labels, (mapped, unmapped), events)
    curve = result["confidence_curve_at_five_frames"][0]

    assert curve["sections_retained"] == 2
    assert curve["human_side_assessable_sections"] == 1
    assert curve["fully_correct_sections"] == 1
    assert curve["fully_correct_share_of_retained"] == 0.5
    assert curve["fully_correct_accuracy_when_assessable"] == 1.0
