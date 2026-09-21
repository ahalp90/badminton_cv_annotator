"""Focused contract tests for the directional W5 comparison packet."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from compare_directional_runs import (
    ARM_COLOURS,
    ARM_FLOORS,
    ARM_IDS,
    CONTACT_CASES,
    CONTACT_HEADER_HEIGHT,
    EXPECTED_CASES,
    PANEL_SIZE,
    VISIBILITY_COLUMNS,
    ComparisonError,
    compare_runs,
    safe_name,
)
from PIL import Image


def _admission(floor: tuple[int, int]) -> dict:
    before = 300
    after = 260
    rejected = 40
    new_count = 2
    removed_count = 2
    return {
        "min_visible_lengthwise": floor[0],
        "min_visible_cross_court": floor[1],
        "visibility_columns": dict(VISIBILITY_COLUMNS),
        "hypotheses_before": before,
        "hypotheses_after": after,
        "hypotheses_rejected": rejected,
        "floor_zero_scanned_for_proposal_cap": 256,
        "floor_zero_selected_count": 256,
        "floor_zero_proposal_ids": [f"proposal-{index}" for index in range(256)],
        "scanned_for_proposal_cap": 258,
        "removed_from_floor_zero_count": removed_count,
        "refilled_proposal_count": new_count,
        "newly_admitted_indices": [260, 261],
        "newly_admitted_proposal_ids": ["new-260", "new-261"],
        "removed_from_floor_zero_indices": [254, 255],
        "removed_from_floor_zero_proposal_ids": ["proposal-254", "proposal-255"],
    }


def _line_source(floor: tuple[int, int]) -> tuple[dict, dict]:
    admission = _admission(floor)
    proposal_ids = [f"proposal-{index}" for index in range(254)]
    proposal_ids.extend(("new-260", "new-261"))
    source = {
        "name": "line_template",
        "status": "generated",
        "settings": {
            "min_visible_lengthwise": floor[0],
            "min_visible_cross_court": floor[1],
            "visibility_columns": dict(VISIBILITY_COLUMNS),
        },
        "contamination_check": {
            "references_loaded": False,
            "prior_controls_loaded": False,
        },
        "generation": {
            "visibility_admission": admission,
            "combined_admission_hypotheses": 260,
        },
        "caps": {
            "256": {
                "count": 256,
                "proposal_ids": proposal_ids,
            }
        },
        "proposal_count": 256,
    }
    return source, admission


def _write_run(root: Path, arm_id: str, floor: tuple[int, int]) -> Path:
    run_dir = root / f"run_{arm_id}"
    run_dir.mkdir()
    source_by_case = {}
    admission_by_case = {}
    for case_id in EXPECTED_CASES:
        source, admission = _line_source(floor)
        source_by_case[case_id] = source
        admission_by_case[case_id] = admission

    checks = {case_id: {"match": True, "fields": []} for case_id in EXPECTED_CASES}
    manifest = {
        "schema": "w5-manifest/2",
        "run_id": run_dir.name,
        "cases": list(EXPECTED_CASES),
        "requested_cases": list(EXPECTED_CASES),
        "min_visible_lengthwise": floor[0],
        "min_visible_cross_court": floor[1],
        "visibility_columns": dict(VISIBILITY_COLUMNS),
        "stopped_views": [],
        "global_parameters": {
            "working_size": [960, 540],
            "camera_limit": 0.1,
            "scorer": "frozen",
            "refit_max_evaluations": 100,
            "workers": 10,
            "min_visible_lengthwise": floor[0],
            "min_visible_cross_court": floor[1],
            "visibility_columns": dict(VISIBILITY_COLUMNS),
        },
        "source_stage": {
            "automatic_path_free_of_reference_fields": True,
            "preflight_path_free_of_reference_fields": True,
            "full_run_contamination_checks": checks,
        },
        "imported_helper_paths": {"helper": "frozen/helper.py"},
        "imported_helper_hashes": {
            "imported": {"helper": {"path": "frozen/helper.py", "sha256": "same"}}
        },
        "line_template_sources": source_by_case,
        "line_template_admission": {
            "min_visible_lengthwise": floor[0],
            "min_visible_cross_court": floor[1],
            "visibility_columns": dict(VISIBILITY_COLUMNS),
            "cases": admission_by_case,
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    rankings = {"schema": "w5-rankings/1", "cases": {}}
    reviews = {}
    for case_id in EXPECTED_CASES:
        origin_key = f"G0:{case_id}:winner"
        rankings["cases"][case_id] = {
            "C": {"selected_origin_key": origin_key, "status": "provisional_for_review"}
        }
        reviews[case_id] = {
            "selected": {"C": origin_key},
            "candidates": {
                origin_key: {
                    "origin_key": origin_key,
                    "candidate_id": f"{case_id}:winner",
                    "source": "G0",
                }
            },
        }
    (run_dir / "rankings.json").write_text(json.dumps(rankings, indent=2) + "\n")
    (run_dir / "review_candidates.json").write_text(
        json.dumps(reviews, indent=2) + "\n"
    )

    fields = [
        "case_id",
        "view_status",
        "G0_count",
        "G1_count",
        "line_template_count",
        "min_visible_lengthwise",
        "min_visible_cross_court",
    ]
    with (run_dir / "per_view.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for case_id in EXPECTED_CASES:
            writer.writerow(
                {
                    "case_id": case_id,
                    "view_status": "completed",
                    "G0_count": 256,
                    "G1_count": 256,
                    "line_template_count": 256,
                    "min_visible_lengthwise": floor[0],
                    "min_visible_cross_court": floor[1],
                }
            )

    gallery = run_dir / "gallery"
    gallery.mkdir()
    for case_id in (
        "gxBQ_window_00_frame_689",
        "gxBQ_window_00_frame_5",
        "am2_window_01_frame_28019",
    ):
        origin_key = f"G0:{case_id}:winner"
        image = Image.new("RGB", (32, 20), (140, 160, 180))
        image.save(gallery / f"{safe_name(f'{case_id}__{origin_key}')}__crop.png")
    for case_id in set(EXPECTED_CASES) - {
        "gxBQ_window_00_frame_689",
        "gxBQ_window_00_frame_5",
        "am2_window_01_frame_28019",
    }:
        origin_key = f"G0:{case_id}:winner"
        Image.new("RGB", (32, 20), (140, 160, 180)).save(
            gallery / f"{safe_name(f'{case_id}__{origin_key}')}__crop.png"
        )
    return run_dir


@pytest.fixture
def run_set(tmp_path: Path) -> dict[str, Path]:
    return {
        arm_id: _write_run(tmp_path, arm_id, floor)
        for arm_id, floor in zip(ARM_IDS, ARM_FLOORS, strict=True)
    }


def _make_empty_case(run_set: dict[str, Path], case_id: str) -> None:
    for run_dir in run_set.values():
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        source = manifest["line_template_sources"][case_id]
        source["status"] = "empty"
        source.pop("caps")
        source["proposal_count"] = 0
        source["generation"].pop("combined_admission_hypotheses")
        admission = source["generation"]["visibility_admission"]
        for field in (
            "hypotheses_before",
            "hypotheses_after",
            "hypotheses_rejected",
            "floor_zero_scanned_for_proposal_cap",
            "floor_zero_selected_count",
            "scanned_for_proposal_cap",
            "removed_from_floor_zero_count",
            "refilled_proposal_count",
        ):
            admission[field] = 0
        for field in (
            "floor_zero_proposal_ids",
            "newly_admitted_indices",
            "newly_admitted_proposal_ids",
            "removed_from_floor_zero_indices",
            "removed_from_floor_zero_proposal_ids",
        ):
            admission[field] = []
        manifest["line_template_admission"]["cases"][case_id] = admission
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

        rows = []
        with (run_dir / "per_view.csv").open(newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = reader.fieldnames
            for row in reader:
                if row["case_id"] == case_id:
                    row["line_template_count"] = "0"
                rows.append(row)
        with (run_dir / "per_view.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def test_comparison_writes_validated_json_markdown_and_contact_sheets(
    run_set: dict[str, Path], tmp_path: Path
) -> None:
    output_dir = tmp_path / "comparison"
    comparison = compare_runs(run_set, output_dir)

    assert comparison["schema"] == "w5-directional-comparison/1"
    assert output_dir.is_dir()
    assert (output_dir / "comparison.json").is_file()
    assert (output_dir / "comparison.md").is_file()
    expected_names = {
        f"{safe_name(case_id)}__directional_arms.png" for case_id in CONTACT_CASES
    }
    actual_names = {path.name for path in (output_dir / "contact_sheets").glob("*.png")}
    assert actual_names == expected_names
    assert comparison["contact_sheet_cases"] == list(CONTACT_CASES)
    markdown_cases = [
        line.split("`")[1]
        for line in (output_dir / "comparison.md").read_text().splitlines()
        if line.startswith("- `")
    ]
    assert markdown_cases == list(CONTACT_CASES)
    first_sheet = Image.open(
        output_dir
        / "contact_sheets"
        / f"{safe_name(CONTACT_CASES[0])}__directional_arms.png"
    )
    assert first_sheet.size == (
        PANEL_SIZE[0] * len(ARM_IDS),
        PANEL_SIZE[1] + CONTACT_HEADER_HEIGHT,
    )
    for index, arm_id in enumerate(ARM_IDS):
        header_x = index * PANEL_SIZE[0] + PANEL_SIZE[0] // 2
        assert (
            first_sheet.getpixel((header_x, CONTACT_HEADER_HEIGHT // 2))
            == ARM_COLOURS[arm_id]
        )
    arm_record = comparison["cases"][0]["arms"]["3_3"]
    assert arm_record["floor_zero_scanned_for_proposal_cap"] == 256
    assert arm_record["floor_zero_selected_count"] == 256
    assert arm_record["floor_zero_proposal_ids"] == [
        f"proposal-{index}" for index in range(256)
    ]
    assert arm_record["scanned_for_proposal_cap"] == 258
    assert arm_record["caps"]["256"]["proposal_ids"][-2:] == ["new-260", "new-261"]
    assert "six lengthwise" in (output_dir / "comparison.md").read_text()


def test_empty_line_template_metadata_is_accepted(
    run_set: dict[str, Path], tmp_path: Path
) -> None:
    case_id = EXPECTED_CASES[0]
    _make_empty_case(run_set, case_id)

    comparison = compare_runs(run_set, tmp_path / "comparison")

    empty_arm = next(row for row in comparison["cases"] if row["case_id"] == case_id)
    for arm_id in ARM_IDS:
        record = empty_arm["arms"][arm_id]
        assert record["proposal_count"] == 0
        assert record["cap_256_count"] == 0
        assert record["combined_admission_hypotheses"] == 0
        assert record["floor_zero_proposal_ids"] == []
        assert record["caps"]["256"]["proposal_ids"] == []


def test_mid_write_failure_removes_partial_output_and_temp_directory(
    run_set: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "comparison"
    original_save = Image.Image.save
    save_calls = 0

    def fail_after_two_saves(
        self: Image.Image, *args: object, **kwargs: object
    ) -> None:
        nonlocal save_calls
        save_calls += 1
        if save_calls == 3:
            raise RuntimeError("synthetic contact-sheet write failure")
        original_save(self, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", fail_after_two_saves)
    with pytest.raises(RuntimeError, match="synthetic contact-sheet write failure"):
        compare_runs(run_set, output_dir)

    assert save_calls == 3
    assert not output_dir.exists()
    assert not list(tmp_path.glob(f".{output_dir.name}.*"))


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("schema", "manifest schema"),
        ("order", "exact ordered 27-case"),
        ("floor", "visibility floor"),
        ("stopped", "stopped views"),
        ("contamination", "contamination check"),
        ("refill", "refill list lengths"),
        ("wrong_refill_id", "refill IDs"),
        ("missing_floor_zero_ledger", "visibility_admission missing"),
        ("wrong_floor_zero_ledger", "arm's floor-zero ledger"),
        ("unstable_floor_zero_ledger", "differs from the reference arm"),
        ("stable_source", "stable line-template source"),
        ("missing_per_view_floor", "per_view is missing"),
        ("join", "source-qualified"),
        ("bogus_source", "source or candidate-ID"),
        ("mismatched_candidate_id", "source or candidate-ID"),
        ("crop", "crop"),
    ],
)
def test_comparison_rejects_unsafe_packets(
    run_set: dict[str, Path], tmp_path: Path, mutation: str, message: str
) -> None:
    target = run_set["3_3"]
    if mutation == "schema":
        manifest = json.loads((target / "manifest.json").read_text())
        manifest["schema"] = "w5-manifest/1"
        (target / "manifest.json").write_text(json.dumps(manifest))
    elif mutation == "order":
        manifest = json.loads((target / "manifest.json").read_text())
        manifest["cases"] = list(reversed(manifest["cases"]))
        (target / "manifest.json").write_text(json.dumps(manifest))
    elif mutation == "floor":
        manifest = json.loads((target / "manifest.json").read_text())
        manifest["min_visible_lengthwise"] = 4
        (target / "manifest.json").write_text(json.dumps(manifest))
    elif mutation == "stopped":
        manifest = json.loads((target / "manifest.json").read_text())
        manifest["stopped_views"] = [{"case_id": EXPECTED_CASES[0], "reason": "test"}]
        (target / "manifest.json").write_text(json.dumps(manifest))
    elif mutation == "contamination":
        manifest = json.loads((target / "manifest.json").read_text())
        manifest["source_stage"]["full_run_contamination_checks"][EXPECTED_CASES[0]][
            "match"
        ] = False
        (target / "manifest.json").write_text(json.dumps(manifest))
    elif mutation == "refill":
        manifest = json.loads((target / "manifest.json").read_text())
        case_id = EXPECTED_CASES[0]
        admission = manifest["line_template_sources"][case_id]["generation"][
            "visibility_admission"
        ]
        admission["refilled_proposal_count"] = 1
        manifest["line_template_admission"]["cases"][case_id] = admission
        (target / "manifest.json").write_text(json.dumps(manifest))
    elif mutation == "wrong_refill_id":
        manifest = json.loads((target / "manifest.json").read_text())
        case_id = EXPECTED_CASES[0]
        cap_ids = manifest["line_template_sources"][case_id]["caps"]["256"][
            "proposal_ids"
        ]
        cap_ids[0] = "new-wrong-id"
        (target / "manifest.json").write_text(json.dumps(manifest))
    elif mutation == "missing_floor_zero_ledger":
        manifest = json.loads((target / "manifest.json").read_text())
        case_id = EXPECTED_CASES[0]
        source_admission = manifest["line_template_sources"][case_id]["generation"][
            "visibility_admission"
        ]
        source_admission.pop("floor_zero_proposal_ids")
        (target / "manifest.json").write_text(json.dumps(manifest))
    elif mutation == "wrong_floor_zero_ledger":
        manifest = json.loads((target / "manifest.json").read_text())
        case_id = EXPECTED_CASES[0]
        source_admission = manifest["line_template_sources"][case_id]["generation"][
            "visibility_admission"
        ]
        source_admission["floor_zero_proposal_ids"][0] = "proposal-not-in-cap"
        manifest["line_template_admission"]["cases"][case_id][
            "floor_zero_proposal_ids"
        ][0] = "proposal-not-in-cap"
        (target / "manifest.json").write_text(json.dumps(manifest))
    elif mutation == "unstable_floor_zero_ledger":
        target = run_set["4_3"]
        manifest = json.loads((target / "manifest.json").read_text())
        case_id = EXPECTED_CASES[0]
        source_admission = manifest["line_template_sources"][case_id]["generation"][
            "visibility_admission"
        ]
        source_admission["floor_zero_proposal_ids"][:2] = [
            "proposal-1",
            "proposal-0",
        ]
        manifest["line_template_admission"]["cases"][case_id][
            "floor_zero_proposal_ids"
        ][:2] = ["proposal-1", "proposal-0"]
        (target / "manifest.json").write_text(json.dumps(manifest))
    elif mutation == "stable_source":
        manifest = json.loads((target / "manifest.json").read_text())
        case_id = EXPECTED_CASES[0]
        manifest["line_template_sources"][case_id]["settings"]["template_count"] = 151
        (target / "manifest.json").write_text(json.dumps(manifest))
    elif mutation == "missing_per_view_floor":
        rows = []
        with (target / "per_view.csv").open(newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = [
                field
                for field in (reader.fieldnames or [])
                if field != "min_visible_lengthwise"
            ]
            for row in reader:
                row.pop("min_visible_lengthwise", None)
                rows.append(row)
        with (target / "per_view.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    elif mutation == "join":
        rankings = json.loads((target / "rankings.json").read_text())
        reviews = json.loads((target / "review_candidates.json").read_text())
        case_id = EXPECTED_CASES[0]
        rankings["cases"][case_id]["C"]["selected_origin_key"] = "winner"
        reviews[case_id]["selected"]["C"] = "winner"
        (target / "rankings.json").write_text(json.dumps(rankings))
        (target / "review_candidates.json").write_text(json.dumps(reviews))
    elif mutation == "bogus_source":
        reviews = json.loads((target / "review_candidates.json").read_text())
        case_id = EXPECTED_CASES[0]
        origin_key = reviews[case_id]["selected"]["C"]
        reviews[case_id]["candidates"][origin_key]["source"] = "bogus"
        (target / "review_candidates.json").write_text(json.dumps(reviews))
    elif mutation == "mismatched_candidate_id":
        reviews = json.loads((target / "review_candidates.json").read_text())
        case_id = EXPECTED_CASES[0]
        origin_key = reviews[case_id]["selected"]["C"]
        reviews[case_id]["candidates"][origin_key]["candidate_id"] = "different"
        (target / "review_candidates.json").write_text(json.dumps(reviews))
    elif mutation == "crop":
        crop = next((target / "gallery").glob("*gxBQ_window_00_frame_689*__crop.png"))
        crop.unlink()

    output_dir = tmp_path / "comparison"
    with pytest.raises(ComparisonError, match=message):
        compare_runs(run_set, output_dir)
    assert not output_dir.exists()
