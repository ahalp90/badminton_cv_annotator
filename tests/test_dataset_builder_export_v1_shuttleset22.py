"""End-to-end export of the frozen v1 dataset from ShuttleSet22 artifacts.

ShuttleSet22 has no dataset-builder run, so every rally comes from the human
contacts and the export reads the issue #106 ``extracted-simple`` layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

from dataset_builder import vision
from dataset_builder import export_v1_shuttleset22 as ss22_export
from dataset_builder.cli import main
from dataset_builder.export_v1_shuttleset22 import (
    INPAINTED_GUARD_CODES_FILENAME,
    INPAINTED_TRACK_FILENAME,
    ShuttleSet22ExportInputs,
    export_shuttleset22_v1,
    metadata_from_receipt,
)
from dataset_builder.manifest import artifact_integrity
from dataset_builder.schema_v1 import (
    DATASET_SCHEMA,
    PLAYER_RALLIES,
    PLAYER_SIGNALS,
    PLAYERS,
    PRIMITIVE_ARTIFACTS,
    RALLIES,
    SOURCE_CONTACTS,
    TABLES,
    read_table,
)
from shuttleset22 import COURT_RECEIPT_SCHEMA, Source, SourceKind
from tests.test_dataset_builder_export_v1 import (
    FRAME_COUNT,
    LOSER_ID,
    RESOLUTION,
    SET_CSV,
    WINNER_ID,
    assert_rally_ids_match_player_rows,
)
from tests.test_dataset_builder_vision import _court_vision, _pose_arrays


RUN_ID = "issue106-test"
SOURCE_DATASET = "ShuttleSet22"
MATCH_ID = 3
VIDEO = "Fixture_Match_2022_Finals"
VIDEO_ID = "ss22_03"
COURT_CODE_ID = "abc123"
EXTRACTED_DIRECTORY = "extracted-simple"
ANNOTATIONS_DIRECTORY = "annotations"
COURT_RECEIPT_FILENAME = "court_receipt.json.gz"
SET_FILENAME = "set1.csv"
MATCH_FILENAME = "match.csv"
# The two clean rallies of the shared fixture, plus a third whose only contact is
# flaw-marked. Issue #138: a flaw-marked row no longer drops its rally, so all
# three are usable; rallies.flaw_marked marks the third one.
FLAW_MARKED_CSV = SET_CSV + "3,1,80,殺球,1,A,A,3,0\n"
# ShuttleSet22's 2022 rows write downcourt as a float, so the parser must accept "1.0".
MATCH_CSV = f"""id,video,winner,loser,downcourt
{MATCH_ID},{VIDEO},Kento MOMOTA,CHOU Tien Chen,1.0
"""
RECEIPT_METADATA: dict[str, object] = {
    "fps_numerator": 30,
    "fps_denominator": 1,
    "frame_count": FRAME_COUNT,
    "width": int(RESOLUTION[0]),
    "height": int(RESOLUTION[1]),
}
SOURCES_HEADER = """schema = "shuttleset22-sources/1"
expected_fps = "30/1"

"""
DOWNLOAD_ENTRY = f"""[[videos]]
id = {MATCH_ID}
video = "{VIDEO}"
url = "https://example.test/watch?v=fixture22"
youtube_id = "fixture22"
source_kind = "download"
"""
OVERLAP_ENTRY = """[[videos]]
id = 1
video = "overlap_match"
url = "https://example.test/watch?v=overlap"
youtube_id = "overlap"
source_kind = "shuttleset_overlap"
overlap_shuttleset_id = 23
"""


class DataFixture(NamedTuple):
    """One ShuttleSet22 data root and the sources manifest that selects it."""

    data_root: Path
    sources: Path
    input_artifacts: dict[str, Path]


def _receipt(**overrides: object) -> dict[str, object]:
    """The completed court receipt the issue #106 court stage leaves behind."""
    return {
        "schema": COURT_RECEIPT_SCHEMA,
        "match_id": MATCH_ID,
        "video": VIDEO,
        "completed": True,
        "code_id": COURT_CODE_ID,
        "model": {"name": "courtkeynet_weights", "md5": "0" * 32, "size_bytes": 1},
        "metadata": dict(RECEIPT_METADATA),
        **overrides,
    }


def _sources_toml(path: Path, entries: dict[int, str]) -> Path:
    """Write a full 1..58 manifest; every unlisted ID is an unresolved placeholder."""
    placeholder = (
        '[[videos]]\nid = {match_id}\nvideo = "placeholder_{match_id:02d}"\n'
        'source_kind = "unresolved"\nunresolved_reason = "fixture placeholder"\n'
    )
    blocks = [
        entries.get(match_id, placeholder.format(match_id=match_id))
        for match_id in range(1, 59)
    ]
    path.write_text(SOURCES_HEADER + "\n".join(blocks), encoding="utf-8")
    return path


def _build_data_root(tmp_path: Path) -> DataFixture:
    """Write the nine primitives, the court receipt, and the set CSV for one match."""
    data_root = tmp_path / "data"
    extracted = data_root / EXTRACTED_DIRECTORY / f"{MATCH_ID:02d} {VIDEO}"
    extracted.mkdir(parents=True)
    court_video_id = str(MATCH_ID)
    artifacts = {
        "shuttle_track": vision.save_npy_xz(
            extracted / vision.TRACK_FILENAME, np.zeros((FRAME_COUNT, 3), dtype=np.float64)
        )
    }
    pose = vision.save_pose_arrays(extracted, _pose_arrays(FRAME_COUNT), FRAME_COUNT)
    artifacts.update(pose.as_mapping())
    court = vision.persist_court_vision(
        extracted,
        video_id=court_video_id,
        court=_court_vision(court_video_id, FRAME_COUNT),
        frame_count=FRAME_COUNT,
        resolution=RESOLUTION,
    )
    artifacts.update(court.as_mapping())
    vision.save_json_gz(extracted / COURT_RECEIPT_FILENAME, _receipt())

    annotations = data_root / ANNOTATIONS_DIRECTORY / "set" / VIDEO
    annotations.mkdir(parents=True)
    (annotations / SET_FILENAME).write_text(FLAW_MARKED_CSV, encoding="utf-8")
    (annotations.parent / MATCH_FILENAME).write_text(MATCH_CSV, encoding="utf-8")
    sources = _sources_toml(tmp_path / "sources.toml", {MATCH_ID: DOWNLOAD_ENTRY})
    return DataFixture(data_root.resolve(), sources, artifacts)


def test_shuttleset22_export_writes_source_rallies(tmp_path: Path) -> None:
    fixture = _build_data_root(tmp_path)
    output_dir = tmp_path / "export"

    manifest = export_shuttleset22_v1(
        ShuttleSet22ExportInputs(
            data_root=fixture.data_root,
            output_dir=output_dir,
            run_id=RUN_ID,
            sources=fixture.sources,
        )
    )
    tables = {table.name: read_table(output_dir, table) for table in TABLES}

    rallies = tables[RALLIES.name]
    assert len(rallies) == 3
    assert set(rallies["rally_origin"]) == {"source_contacts"}
    assert set(rallies["video_id"]) == {VIDEO_ID}
    assert set(rallies["run_id"]) == {RUN_ID}
    assert set(rallies["source_dataset"]) == {SOURCE_DATASET}
    assert (rallies["fps"] == 30.0).all()
    assert rallies["source_set"].tolist() == [1, 1, 1]
    assert rallies["source_rally"].tolist() == [1, 2, 3]
    assert list(zip(rallies["start_frame"], rallies["end_frame"])) == [
        (5, 21), (61, 71), (80, 81),
    ]
    # Issue #138: shots_per_rally is wired through build_video_tables, shared with
    # the ShuttleSet export, so it is exact here too: 3 contacts, then 2, then 1.
    assert rallies["shots_per_rally"].tolist() == [3, 2, 1]
    # Rolled up from source_contacts.flaw_marked: only the third rally's contact
    # carries the flag.
    assert rallies["flaw_marked"].tolist() == [False, False, True]
    # 30 fps: 60 frames of lead-in and 90 of tail, both clamped by this short fixture.
    assert list(zip(rallies["clip_start_frame"], rallies["clip_end_frame"])) == [
        (0, 100), (1, 100), (20, 100),
    ]

    player_rallies = tables[PLAYER_RALLIES.name]
    assert len(player_rallies) == 6
    # downcourt = 1.0 and set 1 put the match winner on the top court.
    assert set(zip(player_rallies["court_side"], player_rallies["player_id"])) == {
        ("top", WINNER_ID), ("bottom", LOSER_ID),
    }
    assert_rally_ids_match_player_rows(rallies, player_rallies)
    people = tables[PLAYERS.name]
    assert people["player_id"].tolist() == [LOSER_ID, WINNER_ID]
    assert people["sex"].tolist() == ["male", "male"]

    contacts = tables[SOURCE_CONTACTS.name]
    assert contacts["frame_num"].tolist() == [5, 12, 20, 61, 70, 80]
    assert contacts["rally_id"].tolist() == [0, 0, 0, 1, 1, 2]
    assert contacts["flaw_marked"].tolist() == [False] * 5 + [True]
    assert contacts["player_id"].tolist() == [
        WINNER_ID, LOSER_ID, WINNER_ID, LOSER_ID, WINNER_ID, WINNER_ID,
    ]

    artifacts = tables[PRIMITIVE_ARTIFACTS.name]
    inputs = artifacts[artifacts["location"] == "input_dir"]
    exports = artifacts[artifacts["location"] == "export_dir"]
    assert len(inputs) == len(fixture.input_artifacts) == 9
    assert set(inputs["artifact"]) == set(fixture.input_artifacts)
    assert len(exports) == len(PLAYER_SIGNALS) == 4
    assert set(exports["artifact"]) == {signal.name for signal in PLAYER_SIGNALS}
    assert inputs["relative_path"].str.startswith(f"{EXTRACTED_DIRECTORY}/").all()
    # Without --inpainted-root, no inpainted sidecar rows are written at all.
    assert not artifacts["artifact"].str.endswith("_inpainted").any()
    for row in artifacts.itertuples():
        root = fixture.data_root if row.location == "input_dir" else output_dir
        stored = artifact_integrity(row.artifact, root / row.relative_path, relative_to=root)
        assert (stored.md5, stored.size_bytes) == (row.md5, row.size_bytes)

    assert manifest["run_id"] == RUN_ID
    assert manifest["source_dataset"] == SOURCE_DATASET
    assert manifest["input_root"] == str(fixture.data_root)
    assert manifest["code_version"] is None
    assert manifest["input_manifest_sha256"] is None
    assert manifest["sources_manifest"] == artifact_integrity(
        "shuttleset22_sources", fixture.sources
    ).to_dict()
    assert str(manifest["ground_truth_root"]).endswith(ANNOTATIONS_DIRECTORY)
    # Without --inpainted-root, the manifest records the gap explicitly rather
    # than omitting the key.
    assert manifest["inpainted_root"] is None
    assert manifest["players_table"]["name"] == "players"
    assert manifest["videos"] == [
        {
            **manifest["videos"][0],
            "video_id": VIDEO_ID,
            "match_id": MATCH_ID,
            "video": VIDEO,
            "court_code_id": COURT_CODE_ID,
            "fps": "30/1",
            "annotator_rallies": 0,
            "source_rallies": 3,
            "match_players": {
                "player_a": WINNER_ID,
                "player_b": LOSER_ID,
                "first_a_is_top": True,
            },
        }
    ]
    assert [entry["path"] for entry in manifest["videos"][0]["source_annotation_files"]] == [
        f"set/{VIDEO}/{SET_FILENAME}"
    ]


def _build_inpainted_root(tmp_path: Path) -> Path:
    """Write the two corrected sidecars for MATCH_ID under a second root."""
    inpainted_root = tmp_path / "inpainted"
    video_dir = inpainted_root / f"{MATCH_ID:02d} {VIDEO}"
    video_dir.mkdir(parents=True)
    vision.save_npy_xz(
        video_dir / INPAINTED_TRACK_FILENAME, np.ones((FRAME_COUNT, 3), dtype=np.float64)
    )
    vision.save_npy_xz(
        video_dir / INPAINTED_GUARD_CODES_FILENAME, np.zeros(FRAME_COUNT, dtype=np.int8)
    )
    return inpainted_root


def test_shuttleset22_export_with_inpainted_root_adds_corrected_sidecars(
    tmp_path: Path,
) -> None:
    fixture = _build_data_root(tmp_path)
    inpainted_root = _build_inpainted_root(tmp_path)
    output_dir = tmp_path / "export"

    manifest = export_shuttleset22_v1(
        ShuttleSet22ExportInputs(
            data_root=fixture.data_root,
            output_dir=output_dir,
            run_id=RUN_ID,
            sources=fixture.sources,
            inpainted_root=inpainted_root,
        )
    )
    assert manifest["inpainted_root"] == str(inpainted_root)
    artifacts = read_table(output_dir, PRIMITIVE_ARTIFACTS)
    inpainted = artifacts[artifacts["location"] == "inpainted_root"]

    assert set(inpainted["artifact"]) == {
        "shuttle_track_inpainted", "shuttle_guard_codes_inpainted",
    }
    assert set(inpainted["video_id"]) == {VIDEO_ID}
    # relative_path is relative to inpainted_root, not data_root, matching the
    # distinct "inpainted_root" location.
    expected_prefix = f"{MATCH_ID:02d} {VIDEO}/"
    assert inpainted["relative_path"].str.startswith(expected_prefix).all()
    for row in inpainted.itertuples():
        stored = artifact_integrity(
            row.artifact, inpainted_root / row.relative_path, relative_to=inpainted_root
        )
        assert (stored.md5, stored.size_bytes) == (row.md5, row.size_bytes)
    # The plain shuttle_track input artifact is still present alongside it.
    assert "shuttle_track" in set(artifacts["artifact"])


def test_shuttleset22_export_with_inpainted_root_feeds_inpainted_track_to_player_signals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The corrected track, not extracted-simple's, must drive player-signal derivation."""
    fixture = _build_data_root(tmp_path)
    inpainted_root = _build_inpainted_root(tmp_path)
    seen_track_paths: list[Path] = []
    real_derive_player_inputs = ss22_export.derive_player_inputs

    def spy_derive_player_inputs(track_path, *args, **kwargs):
        seen_track_paths.append(Path(track_path))
        return real_derive_player_inputs(track_path, *args, **kwargs)

    monkeypatch.setattr(ss22_export, "derive_player_inputs", spy_derive_player_inputs)

    export_shuttleset22_v1(
        ShuttleSet22ExportInputs(
            data_root=fixture.data_root,
            output_dir=tmp_path / "export",
            run_id=RUN_ID,
            sources=fixture.sources,
            inpainted_root=inpainted_root,
        )
    )

    assert seen_track_paths == [
        (inpainted_root / f"{MATCH_ID:02d} {VIDEO}" / INPAINTED_TRACK_FILENAME).resolve()
    ]


def test_shuttleset22_export_with_inpainted_root_missing_directory_names_video(
    tmp_path: Path,
) -> None:
    fixture = _build_data_root(tmp_path)
    empty_inpainted_root = tmp_path / "inpainted-empty"
    empty_inpainted_root.mkdir()

    with pytest.raises(FileNotFoundError, match=f"{MATCH_ID:02d} {VIDEO}.*directory not found"):
        export_shuttleset22_v1(
            ShuttleSet22ExportInputs(
                data_root=fixture.data_root,
                output_dir=tmp_path / "export",
                run_id=RUN_ID,
                sources=fixture.sources,
                inpainted_root=empty_inpainted_root,
            )
        )


def test_shuttleset22_export_with_inpainted_root_missing_file_names_video(
    tmp_path: Path,
) -> None:
    fixture = _build_data_root(tmp_path)
    inpainted_root = tmp_path / "inpainted"
    video_dir = inpainted_root / f"{MATCH_ID:02d} {VIDEO}"
    video_dir.mkdir(parents=True)
    vision.save_npy_xz(
        video_dir / INPAINTED_TRACK_FILENAME, np.ones((FRAME_COUNT, 3), dtype=np.float64)
    )
    # shuttle_guard_codes_inpainted.npy.xz is left missing.

    with pytest.raises(FileNotFoundError, match=f"{MATCH_ID:02d} {VIDEO}.*artifact not found"):
        export_shuttleset22_v1(
            ShuttleSet22ExportInputs(
                data_root=fixture.data_root,
                output_dir=tmp_path / "export",
                run_id=RUN_ID,
                sources=fixture.sources,
                inpainted_root=inpainted_root,
            )
        )


def test_shuttleset22_export_rejects_non_download_source(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    sources = _sources_toml(tmp_path / "sources.toml", {1: OVERLAP_ENTRY})

    with pytest.raises(ValueError, match="not a ShuttleSet22 download"):
        export_shuttleset22_v1(
            ShuttleSet22ExportInputs(
                data_root=data_root,
                output_dir=tmp_path / "export",
                run_id=RUN_ID,
                sources=sources,
                match_ids=(1,),
            )
        )


def test_metadata_from_receipt_rejects_identity_drift() -> None:
    source = Source(MATCH_ID, VIDEO, SourceKind.DOWNLOAD, url="https://example.test")
    data_root = Path("/data")
    without_frames = {
        name: value for name, value in RECEIPT_METADATA.items() if name != "frame_count"
    }

    assert metadata_from_receipt(_receipt(), data_root, source).frame_count == FRAME_COUNT
    with pytest.raises(ValueError, match="identity differs"):
        metadata_from_receipt(_receipt(match_id=MATCH_ID + 1), data_root, source)
    with pytest.raises(ValueError, match="incomplete"):
        metadata_from_receipt(_receipt(completed=False), data_root, source)
    with pytest.raises(ValueError, match="frame_count must be a positive integer"):
        metadata_from_receipt(_receipt(metadata=without_frames), data_root, source)


def test_shuttleset22_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fixture = _build_data_root(tmp_path)

    assert main([
        "export-v1-shuttleset22",
        "--data-root", str(fixture.data_root),
        "--output-dir", str(tmp_path / "export"),
        "--run-id", "x",
        "--sources", str(fixture.sources),
    ]) == 0
    assert DATASET_SCHEMA in capsys.readouterr().out


def test_shuttleset22_cli_with_inpainted_root(tmp_path: Path) -> None:
    fixture = _build_data_root(tmp_path)
    inpainted_root = _build_inpainted_root(tmp_path)
    output_dir = tmp_path / "export"

    assert main([
        "export-v1-shuttleset22",
        "--data-root", str(fixture.data_root),
        "--output-dir", str(output_dir),
        "--run-id", "x",
        "--sources", str(fixture.sources),
        "--inpainted-root", str(inpainted_root),
    ]) == 0
    artifacts = read_table(output_dir, PRIMITIVE_ARTIFACTS)
    assert set(artifacts.loc[artifacts["location"] == "inpainted_root", "artifact"]) == {
        "shuttle_track_inpainted", "shuttle_guard_codes_inpainted",
    }
