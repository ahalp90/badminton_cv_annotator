from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scratch.contact_det_full_ds_fit.scripts import run_rally_start_model as runner


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(gzip.decompress(path.read_bytes()))


def test_label_loader_uses_only_the_allowed_video_rows(tmp_path: Path) -> None:
    labels_path = tmp_path / "shots_master.csv"
    with labels_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["player_side", "frame_num", "rally", "set_id", "vid"])
        writer.writerow(["Top", "10", "1", "set1", "1"])
        writer.writerow(["Bottom", "20", "1", "set1", "1"])
        writer.writerow(["an ignored value", "bad", "bad", "", "2"])
    videos = [SimpleNamespace(video_id=1, fixture="sset_01")]

    labels = runner.load_human_labels(labels_path, videos)

    assert labels.rallies["sset_01"][0].frames == (10, 20)
    assert labels.rallies["sset_01"][0].rally_id == "set1:1"
    assert labels.target_sides == {
        ("sset_01", 10): "Top",
        ("sset_01", 20): "Bot",
    }


def test_training_video_order_follows_the_fixed_groups() -> None:
    videos = tuple(
        SimpleNamespace(fixture=fixture)
        for fixture in ("sset_01", "sset_02", "sset_03", "sset_04")
    )
    split = SimpleNamespace(training_videos=videos)
    config = SimpleNamespace(training_groups=("A", "B"))
    groups = {
        "sset_01": "A",
        "sset_02": "B",
        "sset_03": "A",
        "sset_04": "B",
    }

    assert runner._training_names_in_group_order(split, config, groups) == [
        "sset_01",
        "sset_03",
        "sset_02",
        "sset_04",
    ]


def test_label_hash_is_checked_between_timing_and_side_reads(tmp_path: Path) -> None:
    labels_path = tmp_path / "shots_master.csv"
    labels_path.write_text("original\n", encoding="utf-8")
    expected_hash = runner._sha256(labels_path)

    def change_after_timing(_path: Path, _videos: object) -> runner.TimingLabelSet:
        labels_path.write_text("changed\n", encoding="utf-8")
        return runner.TimingLabelSet({}, frozenset())

    with pytest.raises(ValueError, match="changed after timing"):
        runner._load_checked_human_labels(
            labels_path,
            [],
            expected_hash,
            change_after_timing,
            lambda *_args: {},
        )

    labels_path.write_text("original\n", encoding="utf-8")
    expected_hash = runner._sha256(labels_path)

    def change_after_side(
        _path: Path,
        _videos: object,
        _identities: frozenset[tuple[str, int]],
    ) -> dict[tuple[str, int], str]:
        labels_path.write_text("changed again\n", encoding="utf-8")
        return {}

    with pytest.raises(ValueError, match="changed after player sides"):
        runner._load_checked_human_labels(
            labels_path,
            [],
            expected_hash,
            lambda *_args: runner.TimingLabelSet({}, frozenset()),
            change_after_side,
        )


def test_saved_candidate_rows_reproduce_the_action_counts(tmp_path: Path) -> None:
    detail_path = tmp_path / "candidate_details.json.gz"
    runner._write_json(
        detail_path,
        {
            "candidates": [
                {
                    "fixture": "sset_01",
                    "span_id": 2,
                    "frame": 100,
                    "correct_action": True,
                    "candidate_already_kept": False,
                },
                {
                    "fixture": "sset_01",
                    "span_id": 3,
                    "frame": 200,
                    "correct_action": False,
                    "candidate_already_kept": True,
                },
            ]
        },
    )
    metrics = {
        "selected_candidate_identities": [
            ["sset_01", 2, 100],
            ["sset_01", 3, 200],
        ],
        "selected_actions": 2,
        "correct_additions": 1,
        "newly_added_contacts": 1,
        "recovered_sections": 1,
    }

    runner._verify_action_counts(detail_path, [{"metrics": metrics}])

    metrics["correct_additions"] = 2
    with pytest.raises(ValueError, match="does not reproduce"):
        runner._verify_action_counts(detail_path, [{"metrics": metrics}])


def _prepare_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    training_passed: bool,
) -> tuple[list[str], Path]:
    training_video = SimpleNamespace(fixture="sset_01", video_id=1, fps=30.0)
    validation_video = SimpleNamespace(fixture="sset_18", video_id=18, fps=30.0)
    split = SimpleNamespace(
        training_videos=(training_video,),
        validation_videos=(validation_video,),
    )
    model_spec = SimpleNamespace(model_id="logistic_regression")
    config = SimpleNamespace(
        training_groups=("A",),
        models=(model_spec,),
        selection_cutoffs=(0.5,),
        training_gate=object(),
        validation_gate=object(),
    )
    training_saved = {
        "group": "A",
        "video": {"fixture": "sset_01"},
        "model_training_videos": [],
    }
    validation_saved = {"fixture": "sset_18"}
    monkeypatch.setattr(runner, "load_rally_start_model_config", lambda _path: config)
    monkeypatch.setattr(runner, "load_development_split", lambda _path: split)
    monkeypatch.setattr(runner, "verify_accepted_development_split", lambda _split: None)
    monkeypatch.setattr(
        runner,
        "_checked_groups",
        lambda *_args: {"sset_01": "A"},
    )
    checked_inputs = iter(
        [({}, [training_saved]), ({}, [validation_saved])]
    )
    monkeypatch.setattr(runner, "_checked_input", lambda *_args, **_kwargs: next(checked_inputs))
    monkeypatch.setattr(
        runner,
        "build_candidate_rows",
        lambda _videos, *, default_group: (f"row-{default_group}",),
    )
    monkeypatch.setattr(runner, "assign_candidate_targets", lambda *_args, **_kwargs: "targets")
    monkeypatch.setattr(runner, "_target_rows", lambda *_args: [])
    monkeypatch.setattr(runner, "_verify_action_counts", lambda *_args: None)
    monkeypatch.setattr(
        runner,
        "held_out_candidate_scores",
        lambda *_args: {"logistic_regression": {("sset_01", 0, 10): 0.8}},
    )
    monkeypatch.setattr(runner, "_checked_apply", lambda *_args, **_kwargs: "streams")
    monkeypatch.setattr(runner, "select_candidates", lambda *_args: {})
    monkeypatch.setattr(
        runner,
        "score_candidate_choice",
        lambda *_args: {"choice": "metrics"},
    )
    gate_results = iter([training_passed, True])
    monkeypatch.setattr(runner, "passes_result_gate", lambda *_args, **_kwargs: next(gate_results))
    monkeypatch.setattr(runner, "model_choice_key", lambda *_args: (1, 0, 1.0, 1))
    monkeypatch.setattr(runner, "fit_final_candidate_model", lambda *_args: "model")
    monkeypatch.setattr(
        runner,
        "predict_candidate_scores",
        lambda *_args: {("sset_18", 0, 20): 0.7},
    )

    paths = [tmp_path / f"input-{index}.json" for index in range(8)]
    for path in paths:
        path.write_text("{}\n", encoding="utf-8")
    output_dir = tmp_path / "result"
    label_calls: list[str] = []

    def load_timing(_path: Path, videos: object) -> runner.TimingLabelSet:
        fixture = next(iter(videos)).fixture
        if fixture == "sset_18":
            score_path = output_dir / runner.SCORE_FILENAME
            assert score_path.exists()
            assert _read_json(score_path)["status"] == "complete"
        label_calls.append(f"{fixture}:timing")
        return runner.TimingLabelSet({}, frozenset())

    def load_sides(
        _path: Path,
        videos: object,
        _identities: frozenset[tuple[str, int]],
    ) -> dict[tuple[str, int], str]:
        fixture = next(iter(videos)).fixture
        label_calls.append(f"{fixture}:side")
        return {}

    result_path = runner.run_rally_start_model(
        *paths,
        output_dir,
        "abc1234",
        timing_label_loader=load_timing,
        side_label_loader=load_sides,
    )
    return label_calls, result_path


def test_validation_labels_are_read_after_validation_scores_are_saved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, result_path = _prepare_run(tmp_path, monkeypatch, training_passed=True)

    result = _read_json(result_path)
    assert calls == [
        "sset_01:timing",
        "sset_01:side",
        "sset_18:timing",
        "sset_18:side",
    ]
    assert result["outcome"] == "passed_validation"
    assert result["validation_labels_read"] is True
    saved_text = gzip.decompress(result_path.read_bytes()).decode("utf-8")
    assert str(tmp_path) not in saved_text


def test_failed_training_gate_does_not_read_validation_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, result_path = _prepare_run(tmp_path, monkeypatch, training_passed=False)

    result = _read_json(result_path)
    assert calls == ["sset_01:timing", "sset_01:side"]
    assert result["outcome"] == "stopped_at_training_gate"
    assert result["validation_labels_read"] is False
    assert not (result_path.parent / runner.SCORE_FILENAME).exists()


def test_equal_choices_prefer_the_higher_cutoff() -> None:
    metrics = {
        "fully_correct_at_10_frames": {
            "0.0": {"alternative_count": 12},
        },
        "newly_added_contacts": 10,
        "correct_addition_rate": 0.9,
    }
    key = runner.model_choice_key("logistic_regression", metrics)
    choices = [
        (key, 0.5, "logistic_regression"),
        (key, 0.9, "logistic_regression"),
    ]

    assert max(choices)[1] == 0.9
