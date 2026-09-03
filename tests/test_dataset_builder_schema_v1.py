"""Freeze test for the v1 dataset schema (issue #18).

Every literal in this file IS the freeze: a table's column set or order, a
column's type, nullability, or reliability class, the schema version string,
or an issue #104 keep/cut decision cannot change without a human consciously
editing a literal here.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dataset_builder.schema_v1 import (
    DATASET_SCHEMA,
    FEATURE_DISPOSITIONS,
    PLAYER_SIGNALS,
    PRIMITIVE_ARTIFACT_NOTES,
    RALLIES,
    RELIABILITY_MEANINGS,
    SOURCE_CONTACTS,
    TABLES,
    Disposition,
    ReliabilityClass,
    frozen_column_names,
    read_table,
    validate_table,
    write_table,
)


_EXPECTED_COLUMN_NAMES: dict[str, tuple[str, ...]] = {
    "rallies": (
        "run_id", "source_dataset", "video_id", "rally_origin", "rally_id",
        "fps", "frame_count", "start_frame", "end_frame", "duration_frames",
        "start_seconds", "end_seconds", "duration_seconds", "clip_start_frame",
        "clip_end_frame", "source_set", "source_rally", "top_player_id", "bottom_player_id",
    ),
    "player_rallies": (
        "run_id", "source_dataset", "video_id", "rally_origin", "rally_id",
        "court_side", "player_id", "posture_frames_valid", "posture_frames_linear",
        "posture_mad", "position_frames_valid", "position_frames_linear",
    ),
    "players": ("player_id", "player_name", "sex"),
    "source_contacts": (
        "source_dataset", "video_id", "source_set", "source_row", "source_rally",
        "ball_round", "player_id", "frame_num", "contact_type", "contact_type_en",
        "flaw_marked", "rally_id",
    ),
    "primitive_artifacts": (
        "source_dataset", "video_id", "artifact", "location", "relative_path",
        "md5", "size_bytes", "reliability", "note",
    ),
    "transcript_segments": (
        "source_dataset", "video_id", "segment_index", "timestamp_precision",
        "start_seconds", "end_seconds", "text",
    ),
    "commentary_chunks": (
        "source_dataset", "video_id", "chunk_id", "timestamp_precision",
        "start_seconds", "end_seconds", "text", "text_clean", "bert_f1", "clean_pass",
    ),
}

# (name, type.value, nullable, reliability.value) per column, in frozen order.
_EXPECTED_COLUMN_SPECS: dict[str, tuple[tuple[str, str, bool, str], ...]] = {
    "rallies": (
        ("run_id", "string", False, "observed"),
        ("source_dataset", "string", False, "observed"),
        ("video_id", "string", False, "observed"),
        ("rally_origin", "string", False, "observed"),
        ("rally_id", "int64", False, "observed"),
        ("fps", "float64", False, "observed"),
        ("frame_count", "int64", False, "observed"),
        ("start_frame", "int64", False, "by_rally_origin"),
        ("end_frame", "int64", False, "by_rally_origin"),
        ("duration_frames", "int64", False, "by_rally_origin"),
        ("start_seconds", "float64", False, "by_rally_origin"),
        ("end_seconds", "float64", False, "by_rally_origin"),
        ("duration_seconds", "float64", False, "by_rally_origin"),
        ("clip_start_frame", "int64", False, "by_rally_origin"),
        ("clip_end_frame", "int64", False, "by_rally_origin"),
        ("source_set", "int64", True, "source_annotation"),
        ("source_rally", "int64", True, "source_annotation"),
        ("top_player_id", "string", True, "derived"),
        ("bottom_player_id", "string", True, "derived"),
    ),
    "player_rallies": (
        ("run_id", "string", False, "observed"),
        ("source_dataset", "string", False, "observed"),
        ("video_id", "string", False, "observed"),
        ("rally_origin", "string", False, "observed"),
        ("rally_id", "int64", False, "observed"),
        ("court_side", "string", False, "derived"),
        ("player_id", "string", True, "derived"),
        ("posture_frames_valid", "int64", False, "derived"),
        ("posture_frames_linear", "int64", False, "derived"),
        ("posture_mad", "float64", True, "derived"),
        ("position_frames_valid", "int64", False, "derived"),
        ("position_frames_linear", "int64", False, "derived"),
    ),
    "players": (
        ("player_id", "string", False, "curated"),
        ("player_name", "string", False, "curated"),
        ("sex", "string", False, "curated"),
    ),
    "source_contacts": (
        ("source_dataset", "string", False, "observed"),
        ("video_id", "string", False, "observed"),
        ("source_set", "int64", False, "source_annotation"),
        ("source_row", "int64", False, "observed"),
        ("source_rally", "int64", True, "source_annotation"),
        ("ball_round", "int64", True, "source_annotation"),
        ("player_id", "string", True, "source_annotation"),
        ("frame_num", "int64", True, "source_annotation"),
        ("contact_type", "string", True, "source_annotation"),
        ("contact_type_en", "string", True, "derived"),
        ("flaw_marked", "bool", False, "source_annotation"),
        ("rally_id", "int64", True, "derived"),
    ),
    "primitive_artifacts": (
        ("source_dataset", "string", False, "observed"),
        ("video_id", "string", False, "observed"),
        ("artifact", "string", False, "observed"),
        ("location", "string", False, "observed"),
        ("relative_path", "string", False, "observed"),
        ("md5", "string", False, "observed"),
        ("size_bytes", "int64", False, "observed"),
        ("reliability", "string", False, "observed"),
        ("note", "string", False, "observed"),
    ),
    "transcript_segments": (
        ("source_dataset", "string", False, "observed"),
        ("video_id", "string", False, "observed"),
        ("segment_index", "int64", False, "observed"),
        ("timestamp_precision", "string", False, "observed"),
        ("start_seconds", "float64", False, "observed"),
        ("end_seconds", "float64", False, "observed"),
        ("text", "string", False, "observed"),
    ),
    "commentary_chunks": (
        ("source_dataset", "string", False, "observed"),
        ("video_id", "string", False, "observed"),
        ("chunk_id", "string", False, "observed"),
        ("timestamp_precision", "string", False, "observed"),
        ("start_seconds", "float64", False, "observed"),
        ("end_seconds", "float64", False, "observed"),
        ("text", "string", False, "observed"),
        ("text_clean", "string", False, "derived"),
        ("bert_f1", "float64", True, "derived"),
        ("clean_pass", "bool", True, "derived"),
    ),
}

_EXPECTED_KEYS: dict[str, tuple[str, ...]] = {
    "rallies": ("run_id", "source_dataset", "video_id", "rally_origin", "rally_id"),
    "player_rallies": (
        "run_id", "source_dataset", "video_id", "rally_origin", "rally_id", "court_side",
    ),
    "players": ("player_id",),
    "source_contacts": ("source_dataset", "video_id", "source_set", "source_row"),
    "primitive_artifacts": ("source_dataset", "video_id", "artifact"),
    "transcript_segments": ("source_dataset", "video_id", "segment_index"),
    "commentary_chunks": ("source_dataset", "video_id", "chunk_id"),
}


def test_frozen_schema_surface():
    assert DATASET_SCHEMA == "rally-dataset/1.0"
    assert frozen_column_names() == _EXPECTED_COLUMN_NAMES

    for table in TABLES:
        actual_specs = tuple(
            (column.name, column.type.value, column.nullable, column.reliability.value)
            for column in table.columns
        )
        assert actual_specs == _EXPECTED_COLUMN_SPECS[table.name], table.name
        assert table.key == _EXPECTED_KEYS[table.name], table.name


def test_key_columns_are_not_nullable_and_exist():
    for table in TABLES:
        for key_name in table.key:
            column = table.column(key_name)  # raises KeyError if the key isn't a column
            assert column.nullable is False, f"{table.name}.{key_name}"


_TABLES_BY_NAME = {table.name: table for table in TABLES}
_PLAYER_SIGNAL_NAMES = {signal.name for signal in PLAYER_SIGNALS}

_EXPECTED_KEEP_FEATURES = {
    "Rally timestamps: start, end, FPS, frame ranges",
    "Posture variability (MAD)",
    "Linear-interpolation provenance",
    "ShuttleSet source fields: contact type, round, set",
    "Raw pose, court, and shuttle primitives",
    "Commentary raw captions, normalised transcripts, cleaned text",
    "Rally duration from final contact plus offset",
    "Player identity and sex",
}

_EXPECTED_CUT_FEATURES = {
    "Shots per rally",
    "Away-from-centre recovery",
    "Movement inefficiency",
    "Rally-to-commentary association",
}

# Hypothetical column names for cut/unresolved features that must never sneak into the
# frozen surface tested above.
_FORBIDDEN_COLUMN_NAMES = (
    "shots_per_rally", "recovery", "movement_inefficiency",
    "rally_duration_base30", "serve_speed", "degradation",
)


def _disposition_ref_resolves(ref: str) -> bool:
    """True if ``ref`` is a real table name, a real ``table.column``, or a real
    ``player_signals.<signal name>``."""
    table_name, separator, column_name = ref.partition(".")
    if not separator:
        return table_name in _TABLES_BY_NAME
    if table_name == "player_signals":
        return column_name in _PLAYER_SIGNAL_NAMES
    table = _TABLES_BY_NAME.get(table_name)
    return table is not None and column_name in table.column_names()


def test_disposition_registry_covers_issue_104_decisions():
    features_by_disposition: dict[Disposition, set[str]] = {}
    for entry in FEATURE_DISPOSITIONS:
        features_by_disposition.setdefault(entry.disposition, set()).add(entry.feature)

    assert features_by_disposition[Disposition.KEEP] == _EXPECTED_KEEP_FEATURES
    assert features_by_disposition[Disposition.CUT] == _EXPECTED_CUT_FEATURES

    for entry in FEATURE_DISPOSITIONS:
        if entry.disposition is Disposition.KEEP:
            assert len(entry.columns) >= 1, entry.feature
            for ref in entry.columns:
                assert _disposition_ref_resolves(ref), f"{entry.feature}: {ref!r}"
        else:
            assert entry.columns == (), entry.feature

    frozen_names = {column.name for table in TABLES for column in table.columns}
    assert frozen_names.isdisjoint(_FORBIDDEN_COLUMN_NAMES)


def _valid_rallies_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": ["run1", "run1"],
            "source_dataset": ["shuttleset", "shuttleset"],
            "video_id": ["0012", "0013"],
            "rally_origin": ["source_contacts", "source_contacts"],
            "rally_id": [0, 1],
            "fps": [25.0, 25.0],
            "frame_count": [100, 100],
            "start_frame": [0, 10],
            "end_frame": [50, 60],
            "duration_frames": [50, 50],
            "start_seconds": [0.0, 0.4],
            "end_seconds": [2.0, 2.4],
            "duration_seconds": [2.0, 2.0],
            "clip_start_frame": [0, 0],
            "clip_end_frame": [100, 100],
            "source_set": pd.array([1, 1], dtype="Int64"),
            "source_rally": pd.array([1, 2], dtype="Int64"),
            "top_player_id": pd.array(["kento_momota"] * 2, dtype="string"),
            "bottom_player_id": pd.array(["chou_tien_chen"] * 2, dtype="string"),
        }
    )


def test_write_and_read_round_trip_preserves_types(tmp_path):
    precise_seconds = 100 / 3  # 33.333333333333336: stresses exact float round-trip
    frame = pd.DataFrame(
        {
            "run_id": ["run1", "run1"],
            "source_dataset": ["shuttleset", "shuttleset"],
            "video_id": ["0012", "12"],
            "rally_origin": ["source_contacts", "source_contacts"],
            "rally_id": [0, 0],
            "fps": [29.97002997002997, 29.97002997002997],
            "frame_count": [1000, 1000],
            "start_frame": [0, 50],
            "end_frame": [50, 90],
            "duration_frames": [50, 40],
            "start_seconds": [0.0, 2.0],
            "end_seconds": [2.0, precise_seconds],
            "duration_seconds": [2.0, precise_seconds],
            "clip_start_frame": [0, 0],
            "clip_end_frame": [140, 180],
            "source_set": pd.array([2, None], dtype="Int64"),
            "source_rally": pd.array([1, None], dtype="Int64"),
            "top_player_id": pd.array(["kento_momota", None], dtype="string"),
            "bottom_player_id": pd.array(["chou_tien_chen", None], dtype="string"),
        }
    )

    destination_a = write_table(tmp_path / "a", RALLIES, frame)
    result = read_table(tmp_path / "a", RALLIES)

    # video_id kept as an exact string, not coerced to an int: "0012" != "12".
    assert result["video_id"].tolist() == ["0012", "12"]
    assert str(result["video_id"].dtype) == "string"

    # Rows sorted by key: video_id "0012" < "12" lexically.
    assert result.index.tolist() == [0, 1]

    assert str(result["source_set"].dtype) == "Int64"
    assert result["source_set"].isna().tolist() == [False, True]
    assert result.loc[result["video_id"] == "0012", "source_set"].item() == 2

    assert result["end_seconds"].tolist() == [2.0, precise_seconds]
    assert result["duration_seconds"].iloc[1] == precise_seconds

    # Determinism: two writes of the same validated data produce identical bytes.
    destination_b = write_table(tmp_path / "b", RALLIES, frame)
    assert destination_a.read_bytes() == destination_b.read_bytes()

    # Extra check on SOURCE_CONTACTS: a literal text "NA" and a boolean column both
    # round-trip correctly. read_table only treats the empty field as null
    # (keep_default_na=False), so the text "NA" must survive as the string "NA".
    contacts = pd.DataFrame(
        {
            "source_dataset": ["shuttleset", "shuttleset"],
            "video_id": ["0012", "0012"],
            "source_set": [1, 1],
            "source_row": [0, 1],
            "source_rally": pd.array([1, 1], dtype="Int64"),
            "ball_round": pd.array([1, 2], dtype="Int64"),
            "player_id": pd.array(["kento_momota", None], dtype="string"),
            "frame_num": pd.array([10, 20], dtype="Int64"),
            "contact_type": ["NA", "smash"],
            "contact_type_en": ["NA", "smash"],
            "flaw_marked": [True, False],
            "rally_id": pd.array([0, 0], dtype="Int64"),
        }
    )
    write_table(tmp_path / "c", SOURCE_CONTACTS, contacts)
    contacts_result = read_table(tmp_path / "c", SOURCE_CONTACTS)
    assert contacts_result["contact_type"].tolist() == ["NA", "smash"]
    assert contacts_result["flaw_marked"].tolist() == [True, False]
    # A nullable string foreign key survives as the id and as a real null.
    assert contacts_result["player_id"].tolist()[0] == "kento_momota"
    assert contacts_result["player_id"].isna().tolist() == [False, True]
    assert str(contacts_result["flaw_marked"].dtype) == "boolean"


def test_validate_table_rejects_contract_violations():
    base = _valid_rallies_frame()

    missing = base.drop(columns=["fps"])
    with pytest.raises(ValueError, match=r"missing="):
        validate_table(RALLIES, missing)

    extra = base.copy()
    extra["bogus_extra"] = [1, 2]
    with pytest.raises(ValueError, match=r"extra="):
        validate_table(RALLIES, extra)

    null_in_non_nullable = base.copy()
    null_in_non_nullable.loc[0, "fps"] = None
    with pytest.raises(ValueError, match=r"fps must not contain nulls"):
        validate_table(RALLIES, null_in_non_nullable)

    empty_string = base.copy()
    empty_string.loc[0, "video_id"] = ""
    with pytest.raises(ValueError, match=r"video_id must not contain empty strings"):
        validate_table(RALLIES, empty_string)

    duplicate_key = base.copy()
    duplicate_key["video_id"] = ["0012", "0012"]
    duplicate_key["rally_id"] = [0, 0]
    with pytest.raises(ValueError, match=r"duplicate keys"):
        validate_table(RALLIES, duplicate_key)


def test_reliability_meanings_cover_every_class():
    assert set(RELIABILITY_MEANINGS) == set(ReliabilityClass)

    for table in TABLES:
        for column in table.columns:
            assert column.reliability in RELIABILITY_MEANINGS, f"{table.name}.{column.name}"

    for signal in PLAYER_SIGNALS:
        assert signal.reliability in RELIABILITY_MEANINGS, signal.name

    for note in PRIMITIVE_ARTIFACT_NOTES:
        assert note.reliability in RELIABILITY_MEANINGS, note.artifact
