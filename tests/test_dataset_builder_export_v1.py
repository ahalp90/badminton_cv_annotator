"""End-to-end export of the frozen v1 dataset from one completed run directory."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest

from dataset_builder import vision
from dataset_builder.cli import main
from dataset_builder.export_v1 import ExportInputs, export_dataset_v1
from dataset_builder.manifest import artifact_integrity
from dataset_builder.models import RunManifest, StageOutcome
from dataset_builder.records import assemble_rally_records
from dataset_builder.schema_v1 import (
    DATASET_MANIFEST_FILENAME,
    DATASET_SCHEMA,
    FEATURE_DISPOSITIONS,
    PLAYER_RALLIES,
    PLAYER_SIGNALS,
    PLAYER_SIGNALS_DIRECTORY,
    PLAYERS,
    PRIMITIVE_ARTIFACTS,
    RALLIES,
    SOURCE_CONTACTS,
    TABLES,
    read_table,
)
from tests.test_dataset_builder_records import (
    CODE_VERSION,
    FPS,
    VIDEO_ID,
    _annotation,
    _chunks,
    _metadata,
    _pairing,
    _provenance,
    _source_reference,
    _stage,
    _write,
)
from tests.test_dataset_builder_vision import _court_vision, _pose_arrays


RUN_ID = "run-18"
SOURCE_DATASET = "scraped-professional"
FRAME_COUNT = 100
RESOLUTION = (100.0, 50.0)
ANNOTATION_DIRECTORY = "set/fixture_match"
# The two curated people of the fixture match, as configs/players.csv spells them.
WINNER_ID = "kento_momota"  # player A; downcourt 1 starts A on the top court in set 1
LOSER_ID = "chou_tien_chen"
# The frozen artifact names the export copies out of each per-video run stage.
STAGE_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "shuttle": ("shuttle_track", "shuttle_guard_codes"),
    "pose": ("pose_kps", "pose_bboxes", "pose_scores", "pose_kp_scores", "pose_ndet"),
    "court": ("court_evidence", "court_keep_vote", "court_present"),
    "annotation": ("raw_replay_mask", "definitive_exclusion_mask"),
}
SIGNAL_SHAPES = {
    "posture": (FRAME_COUNT, 2),
    "court_position": (FRAME_COUNT, 2, 2),
    "posture_interpolation": (FRAME_COUNT, 2),
    "position_interpolation": (FRAME_COUNT, 2),
}
# rally, ball_round, frame_num, type, flaw, player, the two round scores, and one
# field v1 deliberately drops.
SET_CSV = """rally,ball_round,frame_num,type,flaw,player,getpoint_player,roundscore_A,roundscore_B
1,1,5,發短球,,A,,1,0
1,2,12,挑球,,B,,1,0
1,3,20,殺球,,A,A,1,0
2,1,61,發長球,,B,,2,0
2,2,70,切球,,A,A,2,0
"""
# Only rally 1, so the single side phase is [5, 21) and misses the (60, 100) span.
NARROW_SET_CSV = """rally,ball_round,frame_num,type,flaw,player,getpoint_player,roundscore_A,roundscore_B
1,1,5,發短球,,A,,1,0
1,2,12,挑球,,B,,1,0
1,3,20,殺球,,A,A,1,0
"""
MATCH_CSV = """id,video,winner,loser,downcourt
1,fixture_match,Kento MOMOTA,CHOU Tien Chen,1
"""
FIXED_SOURCES_TOML = f"""schema = "dataset-builder-fixed-sources/1"
dataset = "ShuttleSet"

[[videos]]
video_id = "{VIDEO_ID}"
source_id = "1"
source_url = "https://example.test/watch?v={VIDEO_ID}"
source_basename = "{VIDEO_ID}.mp4"
source_available = true
source_md5 = "{'a' * 32}"
fps = "25/1"
frame_count = {FRAME_COUNT}
eligible = true

[videos.ground_truth]
match_id = "1"
annotation_directory = "{ANNOTATION_DIRECTORY}"
"""


def assert_rally_ids_match_player_rows(
    rallies: pd.DataFrame, player_rallies: pd.DataFrame
) -> None:
    """Each rally's two id columns must equal its own player_rallies rows, nulls included."""
    sides: dict[tuple[str, str, str, int], dict[str, object]] = {}
    for row in player_rallies.itertuples():
        key = (row.source_dataset, row.video_id, row.rally_origin, int(row.rally_id))
        sides.setdefault(key, {})[row.court_side] = row.player_id
    assert len(sides) == len(rallies)
    for row in rallies.itertuples():
        key = (row.source_dataset, row.video_id, row.rally_origin, int(row.rally_id))
        for column, side in (("top_player_id", "top"), ("bottom_player_id", "bottom")):
            expected = sides[key][side]
            actual = getattr(row, column)
            if pd.isna(expected):
                assert pd.isna(actual), (key, column)
            else:
                assert actual == expected, (key, column)


class RunFixture(NamedTuple):
    """Paths of one completed run and the optional ShuttleSet join inputs."""

    run_dir: Path
    ground_truth_root: Path
    fixed_sources: Path


def _write_primitives(run_dir: Path) -> dict[str, Path]:
    """Write the twelve frame-aligned artifacts a completed run leaves behind."""
    stages = run_dir / "stages"
    shuttle_dir = stages / "shuttle" / VIDEO_ID
    files = {
        "shuttle_track": vision.save_npy_xz(
            shuttle_dir / vision.TRACK_FILENAME, np.zeros((FRAME_COUNT, 3), dtype=np.float64)
        ),
        "shuttle_guard_codes": vision.save_npy_xz(
            shuttle_dir / "shuttle_guard_codes.npy.xz", np.zeros(FRAME_COUNT, dtype=np.int8)
        ),
    }
    pose = vision.save_pose_arrays(
        stages / "pose" / VIDEO_ID, _pose_arrays(FRAME_COUNT), FRAME_COUNT
    )
    files.update(pose.as_mapping())
    court = vision.persist_court_vision(
        stages / "court" / VIDEO_ID,
        video_id=VIDEO_ID,
        court=_court_vision(VIDEO_ID, FRAME_COUNT),
        frame_count=FRAME_COUNT,
        resolution=RESOLUTION,
    )
    files.update(court.as_mapping())
    for name in STAGE_ARTIFACTS["annotation"]:
        files[name] = vision.save_npy_xz(
            stages / "annotation" / VIDEO_ID / f"{name}.npy.xz",
            np.zeros(FRAME_COUNT, dtype=bool),
        )
    return files


def _run_manifest(run_dir: Path, files: dict[str, Path], *, skip: str = "") -> RunManifest:
    stages = tuple(
        _stage(
            f"{base}:{VIDEO_ID}",
            configuration={"phase": base},
            outputs=tuple(
                artifact_integrity(name, files[name], relative_to=run_dir) for name in names
            ),
            marker=str(index + 1),
        )
        for index, (base, names) in enumerate(STAGE_ARTIFACTS.items())
        if base != skip
    )
    return RunManifest(run_id=RUN_ID, created_at_utc="2026-09-02T00:00:00Z", stages=stages)


def _build_run(tmp_path: Path, *, skip_stage: str = "") -> RunFixture:
    """Build one completed run directory plus its ShuttleSet ground truth."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    metadata = _metadata(tmp_path)
    manifest = _run_manifest(run_dir, _write_primitives(run_dir), skip=skip_stage)
    projection = assemble_rally_records(
        manifest=manifest,
        source_dataset=SOURCE_DATASET,
        video_id=VIDEO_ID,
        source_reference=_source_reference(),
        metadata=metadata,
        annotation=_annotation(),
        annotation_fps=FPS,
        annotation_frame_count=FRAME_COUNT,
        pairing=_pairing(metadata),
        chunks=_chunks(),
        commentary_outcome=StageOutcome.PROCESSED,
        commentary_reason=None,
        commentary_missing_reasons={1: "no_time_window_pair"},
        commentary_provenance=_provenance(),
        mask_stage_name=f"annotation:{VIDEO_ID}",
    )
    _write(run_dir, manifest, [projection])

    ground_truth_root = tmp_path / "ground_truth"
    annotations = ground_truth_root / ANNOTATION_DIRECTORY
    annotations.mkdir(parents=True)
    (annotations / "set1.csv").write_text(SET_CSV, encoding="utf-8")
    (annotations.parent / "match.csv").write_text(MATCH_CSV, encoding="utf-8")
    fixed_sources = tmp_path / "fixed_sources.toml"
    fixed_sources.write_text(FIXED_SOURCES_TOML, encoding="utf-8")
    return RunFixture(run_dir, ground_truth_root, fixed_sources)


def test_export_writes_every_table_and_manifest(tmp_path: Path) -> None:
    fixture = _build_run(tmp_path)
    output_dir = tmp_path / "export"

    manifest = export_dataset_v1(
        ExportInputs(
            run_dir=fixture.run_dir,
            output_dir=output_dir,
            fixed_sources_manifest=fixture.fixed_sources,
            ground_truth_root=fixture.ground_truth_root,
        )
    )
    tables = {table.name: read_table(output_dir, table) for table in TABLES}

    rallies = tables[RALLIES.name]
    annotator = rallies[rallies["rally_origin"] == "annotator"]
    source = rallies[rallies["rally_origin"] == "source_contacts"]
    assert list(zip(annotator["start_frame"], annotator["end_frame"])) == [(0, 50), (60, 100)]
    assert annotator["source_set"].isna().all()
    assert annotator["source_rally"].isna().all()
    assert list(zip(source["start_frame"], source["end_frame"])) == [(5, 21), (61, 71)]
    assert source["source_set"].tolist() == [1, 1]
    assert source["source_rally"].tolist() == [1, 2]
    assert (rallies["fps"] == 25.0).all()
    assert (rallies["frame_count"] == FRAME_COUNT).all()
    assert rallies["duration_seconds"].tolist() == [2.0, 1.6, 0.64, 0.4]
    # 25 fps: 50 frames of lead-in and 75 of tail, clamped to [0, frame_count).
    assert list(zip(rallies["clip_start_frame"], rallies["clip_end_frame"])) == [
        (0, 100), (10, 100), (0, 96), (11, 100),
    ]

    players = tables[PLAYER_RALLIES.name]
    assert len(players) == 2 * len(rallies)
    for _, rally_players in players.groupby(["rally_origin", "rally_id"]):
        assert rally_players["court_side"].tolist() == ["bottom", "top"]
    # downcourt = 1 and set 1 put the match winner on the top court.
    assert set(zip(players["court_side"], players["player_id"])) == {
        ("top", WINNER_ID), ("bottom", LOSER_ID),
    }
    assert_rally_ids_match_player_rows(rallies, players)
    people = tables[PLAYERS.name]
    assert people["player_id"].tolist() == [LOSER_ID, WINNER_ID]
    assert people["player_name"].tolist() == ["CHOU Tien Chen", "Kento MOMOTA"]
    assert people["sex"].tolist() == ["male", "male"]
    # The fixture's pose is all-NaN with no detections, so no frame has a posture value.
    assert players["posture_mad"].isna().all()
    assert (players["posture_frames_valid"] == 0).all()
    assert (players["position_frames_valid"] == 0).all()

    contacts = tables[SOURCE_CONTACTS.name]
    assert contacts["source_row"].tolist() == [0, 1, 2, 3, 4]
    assert contacts["frame_num"].tolist() == [5, 12, 20, 61, 70]
    assert contacts["rally_id"].tolist() == [0, 0, 0, 1, 1]
    assert contacts["player_id"].tolist() == [
        WINNER_ID, LOSER_ID, WINNER_ID, LOSER_ID, WINNER_ID,
    ]
    assert contacts["contact_type_en"].tolist() == [
        "short_service", "lob", "smash", "long_service", "drop",
    ]
    assert not contacts["flaw_marked"].any()

    artifacts = tables[PRIMITIVE_ARTIFACTS.name]
    run_artifacts = {name for names in STAGE_ARTIFACTS.values() for name in names}
    signal_names = {signal.name for signal in PLAYER_SIGNALS}
    assert set(artifacts.loc[artifacts["location"] == "input_dir", "artifact"]) == run_artifacts
    assert set(artifacts.loc[artifacts["location"] == "export_dir", "artifact"]) == signal_names
    for row in artifacts.itertuples():
        root = fixture.run_dir if row.location == "input_dir" else output_dir
        stored = artifact_integrity(row.artifact, root / row.relative_path, relative_to=root)
        assert (stored.md5, stored.size_bytes) == (row.md5, row.size_bytes)

    signals = output_dir / PLAYER_SIGNALS_DIRECTORY / VIDEO_ID
    for signal in PLAYER_SIGNALS:
        assert vision.load_npy_xz(signals / signal.filename).shape == SIGNAL_SHAPES[signal.name]

    assert manifest == vision.load_json_gz(output_dir / DATASET_MANIFEST_FILENAME)
    assert manifest["schema"] == DATASET_SCHEMA == "rally-dataset/1.0"
    assert manifest["run_id"] == RUN_ID
    assert manifest["code_version"] == CODE_VERSION
    assert manifest["tables"] == {
        table.name: {
            "filename": table.filename,
            "rows": len(tables[table.name]),
            "md5": artifact_integrity(
                table.name, output_dir / table.filename, relative_to=output_dir
            ).md5,
            "size_bytes": (output_dir / table.filename).stat().st_size,
        }
        for table in TABLES
    }
    assert manifest["videos"] == [
        {
            **manifest["videos"][0],
            "video_id": VIDEO_ID,
            "source_dataset": SOURCE_DATASET,
            "fps": "25/1",
            "frame_count": FRAME_COUNT,
            "annotator_rallies": 2,
            "source_rallies": 2,
        }
    ]
    assert manifest["players_table"]["name"] == "players"
    assert manifest["videos"][0]["match_players"] == {
        "player_a": WINNER_ID,
        "player_b": LOSER_ID,
        "first_a_is_top": True,
    }
    assert [entry["feature"] for entry in manifest["dispositions"]] == [
        disposition.feature for disposition in FEATURE_DISPOSITIONS
    ]
    assert tables["transcript_segments"].empty and tables["commentary_chunks"].empty

    repeat_dir = tmp_path / "export-again"
    export_dataset_v1(
        ExportInputs(
            run_dir=fixture.run_dir,
            output_dir=repeat_dir,
            fixed_sources_manifest=fixture.fixed_sources,
            ground_truth_root=fixture.ground_truth_root,
        )
    )
    for table in TABLES:
        assert (repeat_dir / table.filename).read_bytes() == (
            output_dir / table.filename
        ).read_bytes()


def test_annotator_spans_take_ids_only_from_one_overlapping_side_phase(tmp_path: Path) -> None:
    """Span (0, 50) meets the one side phase; span (60, 100) meets none, so it is null."""
    fixture = _build_run(tmp_path)
    (fixture.ground_truth_root / ANNOTATION_DIRECTORY / "set1.csv").write_text(
        NARROW_SET_CSV, encoding="utf-8"
    )
    output_dir = tmp_path / "export"

    export_dataset_v1(
        ExportInputs(
            run_dir=fixture.run_dir,
            output_dir=output_dir,
            fixed_sources_manifest=fixture.fixed_sources,
            ground_truth_root=fixture.ground_truth_root,
        )
    )

    players = read_table(output_dir, PLAYER_RALLIES)
    assert_rally_ids_match_player_rows(read_table(output_dir, RALLIES), players)
    annotator = players[players["rally_origin"] == "annotator"]
    inside = annotator[annotator["rally_id"] == 0]
    outside = annotator[annotator["rally_id"] == 1]
    assert set(zip(inside["court_side"], inside["player_id"])) == {
        ("top", WINNER_ID), ("bottom", LOSER_ID),
    }
    assert outside["player_id"].isna().all()


def test_export_via_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fixture = _build_run(tmp_path)
    joined = tmp_path / "joined"
    bare = tmp_path / "bare"

    assert main([
        "export-v1",
        "--run-dir", str(fixture.run_dir),
        "--output-dir", str(joined),
        "--fixed-sources", str(fixture.fixed_sources),
        "--ground-truth-root", str(fixture.ground_truth_root),
    ]) == 0
    assert DATASET_SCHEMA in capsys.readouterr().out

    assert main([
        "export-v1", "--run-dir", str(fixture.run_dir), "--output-dir", str(bare),
    ]) == 0
    rallies = read_table(bare, RALLIES)
    assert set(rallies["rally_origin"]) == {"annotator"}
    assert read_table(bare, SOURCE_CONTACTS).empty
    # No annotations means no match table, so no side phase resolves a person.
    assert read_table(bare, PLAYERS).empty
    assert read_table(bare, PLAYER_RALLIES)["player_id"].isna().all()
    assert rallies["top_player_id"].isna().all()
    assert rallies["bottom_player_id"].isna().all()


def test_export_rejects_missing_primitive_stage(tmp_path: Path) -> None:
    fixture = _build_run(tmp_path, skip_stage="pose")

    with pytest.raises(ValueError, match="no processed pose stage"):
        export_dataset_v1(ExportInputs(run_dir=fixture.run_dir, output_dir=tmp_path / "export"))

    with pytest.raises(ValueError, match="must be given together"):
        ExportInputs(
            run_dir=fixture.run_dir,
            output_dir=tmp_path / "export",
            fixed_sources_manifest=Path("x"),
        )


def test_export_video_id_filter_selects_known_videos_only(tmp_path: Path) -> None:
    fixture = _build_run(tmp_path)
    with pytest.raises(ValueError, match="no rally records for video_ids"):
        export_dataset_v1(
            ExportInputs(
                run_dir=fixture.run_dir,
                output_dir=tmp_path / "export-unknown",
                video_ids=("nope",),
            )
        )
    with pytest.raises(ValueError, match="without repeats"):
        ExportInputs(run_dir=fixture.run_dir, output_dir=tmp_path, video_ids=("0012", "0012"))
    manifest = export_dataset_v1(
        ExportInputs(run_dir=fixture.run_dir, output_dir=tmp_path / "export", video_ids=("0012",))
    )
    assert [video["video_id"] for video in manifest["videos"]] == ["0012"]
