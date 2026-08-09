"""Focused tests for the issue-32 rally-start audit guide."""

from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
from pathlib import Path

import pytest

import annotator.rally_start_events as events
from annotator.manual_broadcast_timeline_annotator import read_guides


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPO_ROOT
    / "docs/scraper_pipeline/serve_prepend_lookback/build_rally_start_audit_guide.py"
)
SPEC = importlib.util.spec_from_file_location("build_rally_start_audit_guide", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
guide = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guide)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ball_round_one_wins_a_tied_frame_regardless_of_row_order() -> None:
    tied_rows = [
        {"ball_round": "2.0", "frame_num": "59395.0", "flaw": ""},
        {"ball_round": "1.0", "frame_num": "59395.0", "flaw": "1.0"},
    ]

    selected = guide.select_ball_round_one(tied_rows, "synthetic rally")

    assert selected["ball_round"] == "1.0"
    assert selected["flaw"] == "1.0"


@pytest.mark.parametrize(
    "rows",
    (
        [{"ball_round": "2.0"}],
        [{"ball_round": "1.0"}, {"ball_round": "1"}],
    ),
)
def test_ball_round_one_requires_exactly_one_row(rows: list[dict[str, str]]) -> None:
    with pytest.raises(ValueError, match="expected one ball_round=1 row"):
        guide.select_ball_round_one(rows, "bad rally")


def test_complete_source_join_reproduces_target_and_quality_counts() -> None:
    rows, input_paths = guide.build_audit_rows(REPO_ROOT)

    by_video = {
        video_id: [row for row in rows if row["video_id"] == video_id]
        for video_id in guide.EXPECTED_TARGET_COUNTS
    }
    assert {name: len(video_rows) for name, video_rows in by_video.items()} == {
        "sset_01": 63,
        "sset_15": 39,
        "sset_21": 34,
    }
    assert {
        name: sum(bool(row["gt_first_flaw"]) for row in video_rows)
        for name, video_rows in by_video.items()
    } == {"sset_01": 2, "sset_15": 0, "sset_21": 24}
    assert {
        name: sum(row["gt_first_type_en"] == "unknown" for row in video_rows)
        for name, video_rows in by_video.items()
    } == {"sset_01": 1, "sset_15": 0, "sset_21": 24}
    assert len(input_paths) == 17
    assert all(
        row["review_start_frame"] <= row["live_transition_frame"] <= row["gt_first_frame"]
        for row in rows
    )

    keyed = {(row["video_id"], row["set_id"], row["rally"]): row for row in rows}
    first_tie = keyed[("sset_01", "set2", 15)]
    second_tie = keyed[("sset_01", "set2", 26)]
    assert (first_tie["gt_first_ball_round"], first_tie["gt_first_frame"]) == (1, 59395)
    assert (second_tie["gt_first_ball_round"], second_tie["gt_first_frame"]) == (1, 71547)
    assert first_tie["gt_first_flaw"] is True
    assert first_tie["gt_first_type_en"] == "unknown"
    assert second_tie["gt_first_flaw"] is True


def test_review_decisions_cover_the_pilot_and_separate_off_frame() -> None:
    rows, _input_paths = guide.build_audit_rows(REPO_ROOT)
    decisions_path = REPO_ROOT / guide.REVIEW_DECISIONS_RELATIVE_PATH
    decisions = guide.read_review_decisions(decisions_path)

    reviewed = guide.apply_review_decisions(rows, decisions, str(decisions_path))

    assert len(reviewed) == 32
    counts = {
        video_id: {
            visibility: sum(
                row["video_id"] == video_id
                and row["serve_visibility"] == visibility
                for row in reviewed
            )
            for visibility in guide.SERVE_VISIBILITIES
        }
        for video_id in guide.EXPECTED_TARGET_COUNTS
    }
    assert counts == {
        "sset_01": {
            "visible": 0,
            "broadcast-omitted": 2,
            "off-frame": 2,
            "uncertain": 0,
        },
        "sset_15": {
            "visible": 2,
            "broadcast-omitted": 0,
            "off-frame": 0,
            "uncertain": 0,
        },
        "sset_21": {
            "visible": 17,
            "broadcast-omitted": 2,
            "off-frame": 6,
            "uncertain": 1,
        },
    }
    keyed = {(row["video_id"], row["set_id"], row["rally"]): row for row in reviewed}
    assert keyed[("sset_21", "set1", 29)]["visible_serve_frame"] == "40979"
    assert keyed[("sset_21", "set2", 32)]["serve_visibility"] == "uncertain"


def test_off_frame_decisions_reject_contact_markers() -> None:
    rows, _input_paths = guide.build_audit_rows(REPO_ROOT)
    decisions = guide.read_review_decisions(
        REPO_ROOT / guide.REVIEW_DECISIONS_RELATIVE_PATH
    )
    invalid = [dict(row) for row in decisions]
    off_frame = next(row for row in invalid if row["serve_visibility"] == "off-frame")
    off_frame["visible_serve_frame"] = "29118"

    with pytest.raises(ValueError, match="off-frame leaves all frame markers blank"):
        guide.apply_review_decisions(rows, invalid, "invalid decisions")


@pytest.mark.parametrize("width_change", (-1, 1), ids=("missing-field", "extra-field"))
def test_review_decisions_reject_malformed_row_width(
    tmp_path: Path,
    width_change: int,
) -> None:
    decisions = guide.read_review_decisions(
        REPO_ROOT / guide.REVIEW_DECISIONS_RELATIVE_PATH
    )
    malformed_path = tmp_path / f"malformed-{width_change}.csv.gz"
    with gzip.open(malformed_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(guide.DECISION_COLUMNS)
        for row_number, row in enumerate(decisions):
            values = [row[column] for column in guide.DECISION_COLUMNS]
            if row_number == 0:
                values = values[:-1] if width_change < 0 else [*values, "unexpected"]
            writer.writerow(values)

    with pytest.raises(ValueError, match="row width does not match header"):
        guide.read_review_decisions(malformed_path)


def test_output_package_is_deterministic_and_viewer_compatible(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    timeline_paths = sorted(
        (REPO_ROOT / "docs/scraper_pipeline/broadcast_nonstandard_camera_id/data").glob(
            "sset_*_broadcast_timeline_labels.csv.gz"
        )
    )
    timeline_before = {path: _sha256(path) for path in timeline_paths}

    first_summary = guide.write_audit_package(REPO_ROOT, first)
    second_summary = guide.write_audit_package(REPO_ROOT, second)

    assert first_summary == second_summary
    first_files = sorted(path.name for path in first.iterdir())
    second_files = sorted(path.name for path in second.iterdir())
    repository_files = sorted(path.name for path in guide.DEFAULT_OUTPUT_DIR.iterdir())
    assert first_files == second_files
    assert first_files == repository_files
    assert all((first / name).read_bytes() == (second / name).read_bytes() for name in first_files)
    assert all(
        (first / name).read_bytes() == (guide.DEFAULT_OUTPUT_DIR / name).read_bytes()
        for name in first_files
    )
    assert first_summary["counts"]["pooled_targets"] == 136
    assert first_summary["counts"]["pooled_quality_audit"] == 26
    assert first_summary["counts"]["pooled_transition_controls"] == 6
    assert first_summary["counts"]["pooled_pilot_rows"] == 32
    assert first_summary["counts"]["pooled_reviewed_pilot_rows"] == 32
    assert first_summary["counts"]["pooled_full_audit_seed_rows"] == 136
    assert first_summary["counts"]["pooled_pending_full_audit_rows"] == 104
    assert first_summary["counts"]["pooled_reviewed_visibility"] == {
        "visible": 19,
        "broadcast-omitted": 4,
        "off-frame": 8,
        "uncertain": 1,
    }

    expected_pilot_rows = {"sset_01": 4, "sset_15": 2, "sset_21": 26}
    expected_pending_rows = {"sset_01": 59, "sset_15": 37, "sset_21": 8}
    frame_counts = {"sset_01": 154393, "sset_15": 149487, "sset_21": 100349}
    generated_seed: list[events.RallyStartDecision] = []
    for video_id, expected in expected_pilot_rows.items():
        pilot_path = first / f"{video_id}_rally_start_pilot.csv.gz"
        proposals = read_guides(
            pilot_path,
            frame_count=frame_counts[video_id],
            start_column="review_start_frame",
            end_column="review_end_frame",
            label_column="pilot_stratum",
        )
        assert len(proposals) == expected
        with gzip.open(pilot_path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            pilot_rows = list(reader)
        assert tuple(reader.fieldnames or ()) == guide.TARGET_COLUMNS
        starts = [int(row["review_start_frame"]) for row in pilot_rows]
        assert starts == sorted(starts)
        assert all(
            int(row["review_start_frame"])
            <= int(row["live_transition_frame"])
            <= int(row["gt_first_frame"])
            for row in pilot_rows
        )
        reviewed_path = first / f"{video_id}_rally_start_reviewed.csv.gz"
        with gzip.open(reviewed_path, "rt", encoding="utf-8", newline="") as handle:
            reviewed_rows = list(csv.DictReader(handle))
        assert len(reviewed_rows) == expected
        assert all(row["review_status"] == "reviewed" for row in reviewed_rows)
        targets = events.read_target_csv(
            first / f"{video_id}_rally_start_targets.csv.gz"
        )
        seed = events.read_decision_csv(
            first / f"{video_id}_rally_start_decision_seed.csv.gz",
            targets,
        )
        assert len(seed) == len(targets)
        assert sum(
            decision.review_status is events.ReviewStatus.PENDING
            for decision in seed
        ) == expected_pending_rows[video_id]
        generated_seed.extend(seed)

    primary_decisions = events.read_decision_csv(
        REPO_ROOT / guide.REVIEW_DECISIONS_RELATIVE_PATH
    )
    seed_by_key = {decision.key: decision for decision in generated_seed}
    assert all(seed_by_key[decision.key] == decision for decision in primary_decisions)

    assert {path: _sha256(path) for path in timeline_paths} == timeline_before
