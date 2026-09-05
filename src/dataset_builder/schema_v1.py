"""Frozen version 1 dataset schema: tables, columns, types, and reliability.

This module is the single source of truth for the v1 export. Changing a
table, column, type, or nullability changes the frozen schema and must bump
``DATASET_SCHEMA`` and update ``tests/test_dataset_builder_schema_v1.py``.

Decisions come from issue #22 (formulas), issue #104 (keep, cut, unresolved),
issue #18 (this freeze), Ari's review of PR #135 (player identity and sex),
and issue #138 (rally-dataset/1.1: shots per rally, recovery, and movement
inefficiency, once the dataset moved onto human ShuttleSet contacts; and
``rallies.flaw_marked``, once a flaw-marked rally stopped being dropped;
rally-dataset/1.2: player degradation trends and their tanh temperature; and
rally-dataset/1.3: the commentary-to-rally link, aligned times from issue
#136).
See ``docs/dataset_v1_schema.md``.
"""

from __future__ import annotations

from enum import StrEnum
import gzip
import io
import os
from pathlib import Path
from typing import NamedTuple
from uuid import uuid4

import pandas as pd


DATASET_SCHEMA = "rally-dataset/1.3"
SCHEMA_FROZEN_ON = "2026-09-04"
DATASET_MANIFEST_FILENAME = "dataset_manifest.json.gz"
PLAYER_SIGNALS_DIRECTORY = "player_signals"


class ReliabilityClass(StrEnum):
    """How much a consumer may trust a column, by the origin of its values."""

    OBSERVED = "observed"
    SOURCE_ANNOTATION = "source_annotation"
    PREDICTED = "predicted"
    DERIVED = "derived"
    CURATED = "curated"
    BY_RALLY_ORIGIN = "by_rally_origin"


RELIABILITY_MEANINGS: dict[ReliabilityClass, str] = {
    ReliabilityClass.OBSERVED: (
        "Exact source identity, timing, or file metadata after validation."
    ),
    ReliabilityClass.SOURCE_ANNOTATION: (
        "Human ShuttleSet or ShuttleSet22 label carried verbatim. Not an annotator prediction."
    ),
    ReliabilityClass.PREDICTED: (
        "Production annotator heuristic output. Measured weak on ShuttleSet: 66% of "
        "human rallies covered, strict contact F1 49%."
    ),
    ReliabilityClass.DERIVED: (
        "Verified computation over frame-aligned primitives. Not validated against "
        "independent ground truth."
    ),
    ReliabilityClass.CURATED: (
        "Hand-maintained metadata table in the repository, checked against the ShuttleSet "
        "match tables at export. Not a prediction and not inferred from video."
    ),
    ReliabilityClass.BY_RALLY_ORIGIN: (
        "predicted when rally_origin is annotator; source_annotation when rally_origin "
        "is source_contacts."
    ),
}


class RallyOrigin(StrEnum):
    """Which stage defined a rally row's boundaries."""

    ANNOTATOR = "annotator"
    SOURCE_CONTACTS = "source_contacts"


class CommentaryRelation(StrEnum):
    """How a commentary chunk relates to a rally it links to (issue #138)."""

    INSIDE = "inside"
    POST_RALLY = "post_rally"


class ColumnType(StrEnum):
    """Logical column types and their pandas storage dtypes."""

    STRING = "string"
    INTEGER = "int64"
    FLOAT = "float64"
    BOOLEAN = "bool"


_PANDAS_DTYPES: dict[ColumnType, str] = {
    ColumnType.STRING: "string",
    ColumnType.INTEGER: "Int64",
    ColumnType.FLOAT: "float64",
    ColumnType.BOOLEAN: "boolean",
}


class ColumnSpec(NamedTuple):
    """One frozen column."""

    name: str
    type: ColumnType
    nullable: bool
    reliability: ReliabilityClass
    description: str


class TableSpec(NamedTuple):
    """One frozen table stored as a gzip-compressed CSV file."""

    name: str
    filename: str
    key: tuple[str, ...]
    columns: tuple[ColumnSpec, ...]
    description: str

    def column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    def column(self, name: str) -> ColumnSpec:
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(f"{self.name} has no column {name!r}")

    def pandas_dtypes(self) -> dict[str, str]:
        return {column.name: _PANDAS_DTYPES[column.type] for column in self.columns}


class SignalSpec(NamedTuple):
    """One frame-aligned per-video array in the player-signal bundle."""

    name: str
    filename: str
    shape: str
    dtype: str
    reliability: ReliabilityClass
    description: str


class ArtifactNote(NamedTuple):
    """Reliability note for one raw primitive artifact in the bundle."""

    artifact: str
    reliability: ReliabilityClass
    note: str


class Disposition(StrEnum):
    """Issue #104 decision for one trial feature."""

    KEEP = "keep"
    CUT = "cut"
    UNRESOLVED = "unresolved"
    NOT_MEASURED = "not_measured"
    OUT_OF_SCOPE = "out_of_scope"


class FeatureDisposition(NamedTuple):
    """Where a trial feature ended up, and why."""

    feature: str
    disposition: Disposition
    columns: tuple[str, ...]
    reason: str


def _identity_columns() -> tuple[ColumnSpec, ...]:
    return (
        ColumnSpec(
            "run_id", ColumnType.STRING, False, ReliabilityClass.OBSERVED,
            "Immutable dataset-builder run that produced the row.",
        ),
        ColumnSpec(
            "source_dataset", ColumnType.STRING, False, ReliabilityClass.OBSERVED,
            "Dataset label that namespaces video identifiers, for example ShuttleSet.",
        ),
        ColumnSpec(
            "video_id", ColumnType.STRING, False, ReliabilityClass.OBSERVED,
            "Exact string video identifier. Never coerce it to a number: 0012 and 12 differ.",
        ),
    )


def _rally_key_columns() -> tuple[ColumnSpec, ...]:
    return _identity_columns() + (
        ColumnSpec(
            "rally_origin", ColumnType.STRING, False, ReliabilityClass.OBSERVED,
            "annotator: a predicted span from the production annotator. source_contacts: "
            "the half-open span from the first to one past the last usable human contact "
            "of one ShuttleSet rally.",
        ),
        ColumnSpec(
            "rally_id", ColumnType.INTEGER, False, ReliabilityClass.OBSERVED,
            "Zero-based list position within one (run_id, source_dataset, video_id, "
            "rally_origin) group. Not stable across runs or origins.",
        ),
    )


RALLIES = TableSpec(
    name="rallies",
    filename="rallies.csv.gz",
    key=("run_id", "source_dataset", "video_id", "rally_origin", "rally_id"),
    columns=_rally_key_columns() + (
        ColumnSpec(
            "fps", ColumnType.FLOAT, False, ReliabilityClass.OBSERVED,
            "Probed constant frame rate of the decoded source. Stored per row so time "
            "normalisation is explicit: base-30 frames = frames * 30 / fps.",
        ),
        ColumnSpec(
            "frame_count", ColumnType.INTEGER, False, ReliabilityClass.OBSERVED,
            "Decoded frame total of the source video. Every frame-aligned array has this "
            "many rows.",
        ),
        ColumnSpec(
            "start_frame", ColumnType.INTEGER, False, ReliabilityClass.BY_RALLY_ORIGIN,
            "First frame of the rally, zero-based on the whole source video, included.",
        ),
        ColumnSpec(
            "end_frame", ColumnType.INTEGER, False, ReliabilityClass.BY_RALLY_ORIGIN,
            "One past the last frame of the rally. Intervals are half-open.",
        ),
        ColumnSpec(
            "duration_frames", ColumnType.INTEGER, False, ReliabilityClass.BY_RALLY_ORIGIN,
            "end_frame - start_frame at the source frame rate.",
        ),
        ColumnSpec(
            "start_seconds", ColumnType.FLOAT, False, ReliabilityClass.BY_RALLY_ORIGIN,
            "start_frame / fps on the source-video timeline.",
        ),
        ColumnSpec(
            "end_seconds", ColumnType.FLOAT, False, ReliabilityClass.BY_RALLY_ORIGIN,
            "end_frame / fps on the source-video timeline.",
        ),
        ColumnSpec(
            "duration_seconds", ColumnType.FLOAT, False, ReliabilityClass.BY_RALLY_ORIGIN,
            "duration_frames / fps. This is the span length, not the issue #22 rally "
            "duration from the final contact plus an offset.",
        ),
        ColumnSpec(
            "clip_start_frame", ColumnType.INTEGER, False, ReliabilityClass.BY_RALLY_ORIGIN,
            "start_frame minus 2 s of lead-in at the source frame rate, clamped at 0. "
            "Issue #32 context for the serve setup.",
        ),
        ColumnSpec(
            "clip_end_frame", ColumnType.INTEGER, False, ReliabilityClass.BY_RALLY_ORIGIN,
            "end_frame plus 3 s of tail at the source frame rate, clamped at frame_count; "
            "one past the last clip frame. On source_contacts rows the issue #22 rally "
            "duration from the final contact plus offset is clip_end_frame - start_frame.",
        ),
        ColumnSpec(
            "source_set", ColumnType.INTEGER, True, ReliabilityClass.SOURCE_ANNOTATION,
            "ShuttleSet set number for source_contacts rows. Null for annotator rows.",
        ),
        ColumnSpec(
            "source_rally", ColumnType.INTEGER, True, ReliabilityClass.SOURCE_ANNOTATION,
            "ShuttleSet rally number within its set for source_contacts rows. Null for "
            "annotator rows.",
        ),
        ColumnSpec(
            "top_player_id", ColumnType.STRING, True, ReliabilityClass.DERIVED,
            "players.player_id of the person on the top court during this rally; same "
            "derivation and null cases as player_rallies.player_id.",
        ),
        ColumnSpec(
            "bottom_player_id", ColumnType.STRING, True, ReliabilityClass.DERIVED,
            "players.player_id of the person on the bottom court during this rally; same "
            "derivation and null cases as player_rallies.player_id.",
        ),
        ColumnSpec(
            "shots_per_rally", ColumnType.INTEGER, True, ReliabilityClass.DERIVED,
            "Count of human contact rows in this rally, exact by construction. Null on "
            "annotator rows, which have no contact rows.",
        ),
        ColumnSpec(
            "flaw_marked", ColumnType.BOOLEAN, False, ReliabilityClass.SOURCE_ANNOTATION,
            "True when any human contact row in this rally carries the ShuttleSet flaw "
            "flag. No upstream definition of this flag exists; for most flagged rows its "
            "meaning is unknown. On roughly 320 of the 40-video corpus's 1,314 flagged "
            "rows the contact frame number is demonstrably wrong (see "
            "docs/dataset_v1_schema.md, Source contacts), which corrupts every "
            "frame-anchored value derived from that rally. Stroke sequence, counts and "
            "hitters stay sound regardless. Filter on this column before trusting a "
            "frame-anchored value. False on annotator-origin rows.",
        ),
    ),
    description=(
        "One row per rally. Annotator rows come from the production rally records. "
        "source_contacts rows come from usable human ShuttleSet contact rows. The two "
        "origins are never joined to each other here; the benchmark showed that join is "
        "unsafe."
    ),
)


PLAYER_RALLIES = TableSpec(
    name="player_rallies",
    filename="player_rallies.csv.gz",
    key=("run_id", "source_dataset", "video_id", "rally_origin", "rally_id", "court_side"),
    columns=_rally_key_columns() + (
        ColumnSpec(
            "court_side", ColumnType.STRING, False, ReliabilityClass.DERIVED,
            "top or bottom: the court half the sticky-player picker assigned. A row belongs "
            "to a side within one rally, not to a person.",
        ),
        ColumnSpec(
            "player_id", ColumnType.STRING, True, ReliabilityClass.DERIVED,
            "players.player_id of the person on court_side in this rally. source_contacts "
            "rows: exact, from the match table's downcourt flag, the set number, and the "
            "set-3 change of ends when a score first reaches 11. annotator rows: the one "
            "side phase whose human-contact frame envelope overlaps the span; null when no "
            "phase or more than one overlaps, or when the video has no source annotations.",
        ),
        ColumnSpec(
            "posture_frames_valid", ColumnType.INTEGER, False, ReliabilityClass.DERIVED,
            "Frames in the rally with a finite posture value after bounded linear "
            "interpolation.",
        ),
        ColumnSpec(
            "posture_frames_linear", ColumnType.INTEGER, False, ReliabilityClass.DERIVED,
            "Of posture_frames_valid, frames filled by linear interpolation between "
            "observed frames inside one court scene.",
        ),
        ColumnSpec(
            "posture_mad", ColumnType.FLOAT, True, ReliabilityClass.DERIVED,
            "Posture variability: median absolute deviation over the rally of the "
            "per-frame posture |mean eye y - mean ankle y| / hip width. Unitless. Null "
            "when no frame has a finite value. A derived signal, not validated "
            "biomechanics.",
        ),
        ColumnSpec(
            "position_frames_valid", ColumnType.INTEGER, False, ReliabilityClass.DERIVED,
            "Frames with a finite court-normalised mean-ankle position after bounded "
            "linear interpolation.",
        ),
        ColumnSpec(
            "position_frames_linear", ColumnType.INTEGER, False, ReliabilityClass.DERIVED,
            "Of position_frames_valid, frames filled by linear interpolation.",
        ),
        ColumnSpec(
            "recovery_distance_median", ColumnType.FLOAT, True, ReliabilityClass.DERIVED,
            "Median of this side's source_contacts.recovery_distance values over the "
            "rally, i.e. the contacts where this side was not striking. Unitless: "
            "normalised doubles-court Euclidean distance. Null when no contact in the "
            "rally has a recovery_distance for this side.",
        ),
        ColumnSpec(
            "movement_inefficiency_median", ColumnType.FLOAT, True, ReliabilityClass.DERIVED,
            "Median of this side's source_contacts.movement_inefficiency_top or "
            "_bottom values over the rally. Unitless: normalised doubles-court "
            "Euclidean distance. Null when no interval in the rally has a value for "
            "this side.",
        ),
    ),
    description=(
        "One row per rally and court side with the kept issue #22 features. Cut and "
        "unresolved features are absent by decision, not by omission."
    ),
)


PLAYER_TRENDS = TableSpec(
    name="player_trends",
    filename="player_trends.csv.gz",
    key=("run_id", "source_dataset", "video_id", "player_id", "scope", "scope_id", "feature"),
    columns=(
        *_identity_columns(),
        ColumnSpec(
            "player_id", ColumnType.STRING, False, ReliabilityClass.DERIVED,
            "players.player_id of the person this trend is fit for.",
        ),
        ColumnSpec(
            "scope", ColumnType.STRING, False, ReliabilityClass.DERIVED,
            "set: trend across one player's rallies within one ShuttleSet set, ordered "
            "by source_rally. match: trend across that player's sets in the video, one "
            "point per set (the median of the feature over the player's rallies in that "
            "set), ordered by source_set.",
        ),
        ColumnSpec(
            "scope_id", ColumnType.INTEGER, False, ReliabilityClass.DERIVED,
            "ShuttleSet set number for scope=set. Fixed sentinel 0 for scope=match; no "
            "ShuttleSet set is ever numbered 0.",
        ),
        ColumnSpec(
            "feature", ColumnType.STRING, False, ReliabilityClass.DERIVED,
            "Feature this trend was fit over: a player_rallies float column (for example "
            "posture_mad) or a named rally-level column of rallies (duration_seconds, "
            "and shots_per_rally once that column exists).",
        ),
        ColumnSpec(
            "n_points", ColumnType.INTEGER, False, ReliabilityClass.DERIVED,
            "Values that fed the fit: rallies for scope=set (at least 3), sets for "
            "scope=match (at least 2). A fit with fewer points is not written.",
        ),
        ColumnSpec(
            "slope", ColumnType.FLOAT, False, ReliabilityClass.DERIVED,
            "Ordinary least squares slope of the feature value against its position: "
            "the rally's source_rally number for scope=set, or the set's source_set "
            "number for scope=match.",
        ),
        ColumnSpec(
            "slope_tanh", ColumnType.FLOAT, False, ReliabilityClass.DERIVED,
            "tanh(slope / temperature): the slope compressed to (-1, 1) so trends of "
            "differently scaled features are comparable.",
        ),
        ColumnSpec(
            "temperature", ColumnType.FLOAT, False, ReliabilityClass.DERIVED,
            "Tanh scaling constant used for slope_tanh, stored so the scaling reverses: "
            "slope = temperature * arctanh(slope_tanh).",
        ),
    ),
    description=(
        "One row per player, scope, and trended feature: an ordinary least squares "
        "trend over that player's source_contacts rallies or sets, plus its "
        "tanh-normalised slope. Annotator rallies are excluded because their player "
        "identity is a guess, not a label."
    ),
)


PLAYERS = TableSpec(
    name="players",
    filename="players.csv.gz",
    key=("player_id",),
    columns=(
        ColumnSpec(
            "player_id", ColumnType.STRING, False, ReliabilityClass.CURATED,
            "Stable lowercase identifier shared by ShuttleSet and ShuttleSet22, from "
            "configs/players.csv.",
        ),
        ColumnSpec(
            "player_name", ColumnType.STRING, False, ReliabilityClass.CURATED,
            "Display name, spelled as the ShuttleSet match tables spell it.",
        ),
        ColumnSpec(
            "sex", ColumnType.STRING, False, ReliabilityClass.CURATED,
            "female or male: the BWF singles draw the player competes in. Both players of "
            "a singles match share the value; the exporter refuses a match where they "
            "differ.",
        ),
    ),
    description=(
        "One row per person referenced by this export. player_rallies.player_id and "
        "source_contacts.player_id are foreign keys to player_id."
    ),
)


SOURCE_CONTACTS = TableSpec(
    name="source_contacts",
    filename="source_contacts.csv.gz",
    key=("source_dataset", "video_id", "source_set", "source_row"),
    columns=_identity_columns()[1:] + (
        ColumnSpec(
            "source_set", ColumnType.INTEGER, False, ReliabilityClass.SOURCE_ANNOTATION,
            "Set number parsed from the ShuttleSet set CSV filename.",
        ),
        ColumnSpec(
            "source_row", ColumnType.INTEGER, False, ReliabilityClass.OBSERVED,
            "Zero-based row position within that set CSV. Keeps duplicate source rows "
            "distinct.",
        ),
        ColumnSpec(
            "source_rally", ColumnType.INTEGER, True, ReliabilityClass.SOURCE_ANNOTATION,
            "ShuttleSet rally number within the set.",
        ),
        ColumnSpec(
            "ball_round", ColumnType.INTEGER, True, ReliabilityClass.SOURCE_ANNOTATION,
            "ShuttleSet shot number within the rally.",
        ),
        ColumnSpec(
            "player_id", ColumnType.STRING, True, ReliabilityClass.SOURCE_ANNOTATION,
            "players.player_id of the hitter: the source player letter resolved through "
            "the match table, where A is the match winner. Null when the letter is not A "
            "or B.",
        ),
        ColumnSpec(
            "frame_num", ColumnType.INTEGER, True, ReliabilityClass.SOURCE_ANNOTATION,
            "Human contact frame on the source-video timeline. Null when the source "
            "field is empty or not a number.",
        ),
        ColumnSpec(
            "contact_type", ColumnType.STRING, True, ReliabilityClass.SOURCE_ANNOTATION,
            "Verbatim ShuttleSet stroke-type label for the contact.",
        ),
        ColumnSpec(
            "contact_type_en", ColumnType.STRING, True, ReliabilityClass.DERIVED,
            "English name for contact_type from the shared classifier taxonomy. Null when "
            "the label has no mapping.",
        ),
        ColumnSpec(
            "flaw_marked", ColumnType.BOOLEAN, False, ReliabilityClass.SOURCE_ANNOTATION,
            "True when the ShuttleSet flaw field is non-empty for this row.",
        ),
        ColumnSpec(
            "rally_id", ColumnType.INTEGER, True, ReliabilityClass.DERIVED,
            "rally_id of the source_contacts row in rallies that this contact belongs to. "
            "Null when its rally was unusable: an invalid frame, or contacts out of order. "
            "A flaw-marked row does not null this; see rallies.flaw_marked.",
        ),
        ColumnSpec(
            "recovery_distance", ColumnType.FLOAT, True, ReliabilityClass.DERIVED,
            "The measured player is this rally's other player, never the hitter: "
            "rallies.top_player_id when this row's player_id is bottom_player_id, and "
            "the reverse. recovery_distance is that player's mean distance from their "
            "own half-centre over the +/- 5 base-30-frame window around this contact, "
            "clipped to the rally. Unitless: normalised doubles-court Euclidean "
            "distance. Null when the hitter is not one of this rally's two players, or "
            "when no frame in the window has a finite position.",
        ),
        ColumnSpec(
            "recovery_frames_valid", ColumnType.INTEGER, False, ReliabilityClass.DERIVED,
            "How many frames of the recovery_distance window, for that same other "
            "player, had a finite position. Zero when the hitter could not be matched "
            "to a rally side, so no window could be built. Provenance, matching how "
            "the table records player_rallies.posture_frames_valid.",
        ),
        ColumnSpec(
            "movement_inefficiency_top", ColumnType.FLOAT, True, ReliabilityClass.DERIVED,
            "Top player's path length minus straight-line displacement from this "
            "contact to the next contact in the rally. Unitless: normalised "
            "doubles-court Euclidean distance. Null on a rally's last contact, and "
            "null when a position in the interval is not finite.",
        ),
        ColumnSpec(
            "movement_inefficiency_bottom", ColumnType.FLOAT, True, ReliabilityClass.DERIVED,
            "Bottom player's path length minus straight-line displacement from this "
            "contact to the next contact in the rally. Same units and null cases as "
            "movement_inefficiency_top.",
        ),
    ),
    description=(
        "Human ShuttleSet contact rows, restricted to the kept source fields: contact "
        "type, rally and shot numbers, set, and frame. All other ShuttleSet columns are "
        "excluded from v1. Rows are source-scoped and never annotator predictions. "
        "recovery_distance, recovery_frames_valid, movement_inefficiency_top, and "
        "movement_inefficiency_bottom are derived from this table's own contact order "
        "and the player-signal positions, not copied from ShuttleSet."
    ),
)


PRIMITIVE_ARTIFACTS = TableSpec(
    name="primitive_artifacts",
    filename="primitive_artifacts.csv.gz",
    key=("source_dataset", "video_id", "artifact"),
    columns=_identity_columns()[1:] + (
        ColumnSpec(
            "artifact", ColumnType.STRING, False, ReliabilityClass.OBSERVED,
            "Canonical artifact name; see PRIMITIVE_ARTIFACT_NOTES.",
        ),
        ColumnSpec(
            "location", ColumnType.STRING, False, ReliabilityClass.OBSERVED,
            "input_dir, export_dir, or inpainted_root: the root that relative_path is "
            "relative to. The dataset manifest records input_root and inpainted_root by "
            "name; export_dir is implicit, since the manifest file itself lives there.",
        ),
        ColumnSpec(
            "relative_path", ColumnType.STRING, False, ReliabilityClass.OBSERVED,
            "POSIX path of the file under location.",
        ),
        ColumnSpec(
            "md5", ColumnType.STRING, False, ReliabilityClass.OBSERVED,
            "MD5 of the stored file, matching the run manifest convention.",
        ),
        ColumnSpec(
            "size_bytes", ColumnType.INTEGER, False, ReliabilityClass.OBSERVED,
            "Stored file size.",
        ),
        ColumnSpec(
            "reliability", ColumnType.STRING, False, ReliabilityClass.OBSERVED,
            "Reliability class of the artifact's content.",
        ),
        ColumnSpec(
            "note", ColumnType.STRING, False, ReliabilityClass.OBSERVED,
            "Reliability note for the artifact.",
        ),
    ),
    description=(
        "Manifest of the raw primitive bundle: frame-aligned shuttle, pose, court, and "
        "mask artifacts from the run, plus the derived player-signal arrays. Files are "
        "referenced, not copied."
    ),
)


def _commentary_columns() -> tuple[ColumnSpec, ...]:
    return _identity_columns()[1:] + (
        ColumnSpec(
            "timestamp_precision", ColumnType.STRING, False, ReliabilityClass.OBSERVED,
            "caption: automatic caption segment timing. whisperx_coarse: segment-level "
            "WhisperX timing, not word-level. whisperx_aligned: chunk timing replaced "
            "by forced-aligned word boundaries (issue #136); commentary_chunks only. "
            "None of the three were checked against rally boundaries; that check is "
            "the unmeasured part of commentary_rally_links.",
        ),
        ColumnSpec(
            "start_seconds", ColumnType.FLOAT, False, ReliabilityClass.OBSERVED,
            "Segment start on the source-video timeline.",
        ),
        ColumnSpec(
            "end_seconds", ColumnType.FLOAT, False, ReliabilityClass.OBSERVED,
            "Segment end on the source-video timeline.",
        ),
    )


TRANSCRIPT_SEGMENTS = TableSpec(
    name="transcript_segments",
    filename="transcript_segments.csv.gz",
    key=("source_dataset", "video_id", "segment_index"),
    columns=_commentary_columns()[:2] + (
        ColumnSpec(
            "segment_index", ColumnType.INTEGER, False, ReliabilityClass.OBSERVED,
            "Zero-based segment position in the normalised transcript.",
        ),
    ) + _commentary_columns()[2:] + (
        ColumnSpec(
            "text", ColumnType.STRING, False, ReliabilityClass.OBSERVED,
            "Normalised transcript text. May contain transcription errors.",
        ),
    ),
    description=(
        "Auxiliary component: normalised commentary transcript segments tied to the "
        "video, not to rallies. Rally association is cut from v1."
    ),
)


COMMENTARY_CHUNKS = TableSpec(
    name="commentary_chunks",
    filename="commentary_chunks.csv.gz",
    key=("source_dataset", "video_id", "chunk_id"),
    columns=_commentary_columns()[:2] + (
        ColumnSpec(
            "chunk_id", ColumnType.STRING, False, ReliabilityClass.OBSERVED,
            "Cleaning-stage chunk identifier, unique within the video.",
        ),
    ) + _commentary_columns()[2:] + (
        ColumnSpec(
            "text", ColumnType.STRING, False, ReliabilityClass.OBSERVED,
            "Raw chunk text before cleaning.",
        ),
        ColumnSpec(
            "text_clean", ColumnType.STRING, False, ReliabilityClass.DERIVED,
            "Generated cleaned text. Not a human judgement of relevance or accuracy.",
        ),
        ColumnSpec(
            "bert_f1", ColumnType.FLOAT, True, ReliabilityClass.DERIVED,
            "BERTScore F1 between text and text_clean. A cleaning diagnostic, not a truth "
            "probability.",
        ),
        ColumnSpec(
            "clean_pass", ColumnType.BOOLEAN, True, ReliabilityClass.DERIVED,
            "Whether the chunk passed the cleaning contract.",
        ),
    ),
    description=(
        "Auxiliary component: relevance-triaged commentary chunks with raw and cleaned "
        "text, tied to the video. Sentiment, concept, and player link are unresolved and "
        "absent. commentary_rally_links carries the separate, unresolved rally association."
    ),
)


COMMENTARY_RALLY_LINKS = TableSpec(
    name="commentary_rally_links",
    filename="commentary_rally_links.csv.gz",
    key=("run_id", "source_dataset", "video_id", "chunk_id", "rally_origin", "rally_id"),
    columns=(
        *_identity_columns(),
        ColumnSpec(
            "chunk_id", ColumnType.STRING, False, ReliabilityClass.DERIVED,
            "commentary_chunks.chunk_id of the linked chunk.",
        ),
        ColumnSpec(
            "rally_origin", ColumnType.STRING, False, ReliabilityClass.DERIVED,
            "Always source_contacts: only ShuttleSet human-contact rallies are linked "
            "in v1. Kept as a column so the key joins directly to rallies.",
        ),
        ColumnSpec(
            "rally_id", ColumnType.INTEGER, False, ReliabilityClass.DERIVED,
            "rallies.rally_id of the linked rally.",
        ),
        ColumnSpec(
            "relation", ColumnType.STRING, False, ReliabilityClass.DERIVED,
            "inside: the chunk starts inside this rally's span. post_rally: this rally "
            "ended within LAG_SECONDS before the chunk started.",
        ),
        ColumnSpec(
            "lag_seconds", ColumnType.FLOAT, False, ReliabilityClass.DERIVED,
            "0.0 for relation inside; otherwise the chunk's start_seconds minus this "
            "rally's end_seconds, the gap the pairing rule allowed.",
        ),
        ColumnSpec(
            "ambiguous", ColumnType.BOOLEAN, False, ReliabilityClass.DERIVED,
            "True when this chunk links to more than one rally under the pairing rule; "
            "every one of its link rows carries the same value.",
        ),
        ColumnSpec(
            "starts_on_masked_frame", ColumnType.BOOLEAN, True, ReliabilityClass.DERIVED,
            "True when the chunk's start lands on a frame the optional replay mask "
            "marks excluded. Null when the export was not given a replay mask root; "
            "the export never drops a masked-start chunk on this basis.",
        ),
    ),
    description=(
        "One row per commentary chunk linked to one source_contacts rally under the "
        "issue #138 pairing rule. A chunk with more than one candidate rally gets one "
        "row per rally, all marked ambiguous. Coverage is measured; accuracy is not: "
        "nobody has labelled a sample to check that a linked chunk actually discusses "
        "its rally."
    ),
)


TABLES: tuple[TableSpec, ...] = (
    RALLIES,
    PLAYER_RALLIES,
    PLAYER_TRENDS,
    PLAYERS,
    SOURCE_CONTACTS,
    PRIMITIVE_ARTIFACTS,
    TRANSCRIPT_SEGMENTS,
    COMMENTARY_CHUNKS,
    COMMENTARY_RALLY_LINKS,
)


PLAYER_SIGNALS: tuple[SignalSpec, ...] = (
    SignalSpec(
        "posture", "posture.npy.xz", "(frame_count, 2)", "float64",
        ReliabilityClass.DERIVED,
        "Per-frame posture |mean eye y - mean ankle y| / hip width for the top and "
        "bottom sticky players, after bounded linear interpolation. NaN where unavailable.",
    ),
    SignalSpec(
        "court_position", "court_position.npy.xz", "(frame_count, 2, 2)", "float64",
        ReliabilityClass.DERIVED,
        "Per-frame mean-ankle position of the top and bottom players projected into "
        "doubles-court unit coordinates, not clipped to [0, 1]. NaN where unavailable.",
    ),
    SignalSpec(
        "posture_interpolation", "posture_interpolation.npy.xz", "(frame_count, 2)",
        "int8", ReliabilityClass.DERIVED,
        "interpolation_type per posture frame: 0 observed, 1 linear, 2 backward "
        "extrapolated (never emitted in v1).",
    ),
    SignalSpec(
        "position_interpolation", "position_interpolation.npy.xz", "(frame_count, 2)",
        "int8", ReliabilityClass.DERIVED,
        "interpolation_type per court_position frame, same codes as "
        "posture_interpolation.",
    ),
)


PRIMITIVE_ARTIFACT_NOTES: tuple[ArtifactNote, ...] = (
    ArtifactNote(
        "shuttle_track", ReliabilityClass.PREDICTED,
        "(frame_count, 3) TrackNet x, y normalised by resolution, and visibility. Median "
        "court error 0.459 units at human contacts. Do not describe as accurate.",
    ),
    ArtifactNote(
        "shuttle_guard_codes", ReliabilityClass.PREDICTED,
        "(frame_count,) inpaint hallucination guard grades. Mask rejected grades before "
        "using shuttle positions.",
    ),
    ArtifactNote(
        "shuttle_track_inpainted", ReliabilityClass.PREDICTED,
        "(frame_count, 3) TrackNet x, y normalised by resolution, and visibility, from a "
        "later InpaintNet pass over the ShuttleSet22 extract. The base ShuttleSet22 "
        "extract was run with InpaintNet off, so this replaces shuttle_track with a "
        "higher-visibility track. Do not describe as accurate.",
    ),
    ArtifactNote(
        "shuttle_guard_codes_inpainted", ReliabilityClass.PREDICTED,
        "(frame_count,) inpaint hallucination guard grades for shuttle_track_inpainted. "
        "Mask rejected grades before using shuttle positions.",
    ),
    ArtifactNote(
        "pose_kps", ReliabilityClass.PREDICTED,
        "(frame_count, slots, 17, 2) RTMLib keypoints per detection slot. Slots are not "
        "player identities.",
    ),
    ArtifactNote(
        "pose_bboxes", ReliabilityClass.PREDICTED,
        "(frame_count, slots, 4) detection boxes. NaN in inactive slots.",
    ),
    ArtifactNote(
        "pose_scores", ReliabilityClass.PREDICTED,
        "(frame_count, slots) detection scores. NaN in inactive slots.",
    ),
    ArtifactNote(
        "pose_kp_scores", ReliabilityClass.PREDICTED,
        "(frame_count, slots, 17) keypoint scores.",
    ),
    ArtifactNote(
        "pose_ndet", ReliabilityClass.PREDICTED,
        "(frame_count,) active detection count per frame.",
    ),
    ArtifactNote(
        "court_evidence", ReliabilityClass.PREDICTED,
        "Scene homography rows and gate inputs. Median corner error 4.34 px on ShuttleSet.",
    ),
    ArtifactNote(
        "court_keep_vote", ReliabilityClass.PREDICTED,
        "(frame_count,) CourtKeyNet keep vote mask.",
    ),
    ArtifactNote(
        "court_present", ReliabilityClass.PREDICTED,
        "(frame_count,) court-present mask that bounds every interpolation segment.",
    ),
    ArtifactNote(
        "raw_replay_mask", ReliabilityClass.PREDICTED,
        "(frame_count,) raw replay mask from the annotation stage.",
    ),
    ArtifactNote(
        "definitive_exclusion_mask", ReliabilityClass.PREDICTED,
        "(frame_count,) definitive exclusion mask from the annotation stage.",
    ),
) + tuple(
    ArtifactNote(signal.name, signal.reliability, f"{signal.shape} {signal.dtype}. {signal.description}")
    for signal in PLAYER_SIGNALS
)


FEATURE_DISPOSITIONS: tuple[FeatureDisposition, ...] = (
    FeatureDisposition(
        "Rally timestamps: start, end, FPS, frame ranges", Disposition.KEEP,
        (
            "rallies.fps", "rallies.start_frame", "rallies.end_frame",
            "rallies.duration_frames", "rallies.start_seconds", "rallies.end_seconds",
            "rallies.duration_seconds",
        ),
        "Exact conversion over a complete population. Reliability follows rally_origin.",
    ),
    FeatureDisposition(
        "Posture variability (MAD)", Disposition.KEEP,
        ("player_rallies.posture_mad",),
        "Formula complete, 99.57% coverage, stable leave-one-video-out medians. No "
        "independent posture ground truth, so labelled derived.",
    ),
    FeatureDisposition(
        "Linear-interpolation provenance", Disposition.KEEP,
        (
            "player_rallies.posture_frames_linear", "player_rallies.position_frames_linear",
            "player_signals.posture_interpolation", "player_signals.position_interpolation",
        ),
        "Gaps are filled only between observations inside one court scene, and every "
        "filled frame is marked.",
    ),
    FeatureDisposition(
        "ShuttleSet source fields: contact type, round, set", Disposition.KEEP,
        (
            "source_contacts.source_set", "source_contacts.source_rally",
            "source_contacts.ball_round", "source_contacts.contact_type",
        ),
        "Direct human-source fields, kept source-scoped and never presented as predictions.",
    ),
    FeatureDisposition(
        "Raw pose, court, and shuttle primitives", Disposition.KEEP,
        ("primitive_artifacts",),
        "Kept as a separate referenced bundle with masks and reliability notes.",
    ),
    FeatureDisposition(
        "Commentary raw captions, normalised transcripts, cleaned text", Disposition.KEEP,
        ("transcript_segments", "commentary_chunks"),
        "Auxiliary component tied to the video with segment timestamps and a precision "
        "class. Not rally labels.",
    ),
    FeatureDisposition(
        "Rally duration from final contact plus offset", Disposition.KEEP,
        ("rallies.clip_start_frame", "rallies.clip_end_frame"),
        "Issue #32 fixed the offsets: 2 s before the first contact and 3 s after the last, "
        "clamped to the video. Exact on source_contacts rows; predicted spans on annotator "
        "rows.",
    ),
    FeatureDisposition(
        "Player identity and sex", Disposition.KEEP,
        (
            "players.player_id", "players.sex", "rallies.top_player_id",
            "rallies.bottom_player_id", "player_rallies.player_id",
            "source_contacts.player_id",
        ),
        "Curated per-player table joined through the ShuttleSet match tables. Court sides "
        "map to people by the downcourt flag, the set number, and the set-3 change of ends.",
    ),
    FeatureDisposition(
        "Raw degradation slope", Disposition.KEEP,
        ("player_trends.slope", "player_trends.n_points"),
        "Issue #104 could not fit a trend without a retained feature set and stable "
        "player identity across rallies. Both now exist: player_rallies keeps float "
        "features and source_contacts rallies carry an exact player_id.",
    ),
    FeatureDisposition(
        "Tanh-normalised degradation", Disposition.KEEP,
        ("player_trends.slope_tanh", "player_trends.temperature"),
        "Issue #22 left the tanh scaling temperature undefined. Issue #138 asked to "
        "sweep it if that was cheap, and otherwise pick a magic number like 2. The "
        "sweep was skipped, so the feature's owner used that named fallback, 2.0; the "
        "raw slope is kept alongside it so the scaling reverses.",
    ),
    FeatureDisposition(
        "Shots per rally", Disposition.KEEP,
        ("rallies.shots_per_rally",),
        "Issue #104 measured this against predicted contacts, exact on only 298 of 3,287 "
        "rallies. It is now the count of human ShuttleSet contact rows in the rally, exact "
        "by construction, so the weak input that cut it is gone.",
    ),
    FeatureDisposition(
        "Away-from-centre recovery", Disposition.KEEP,
        (
            "source_contacts.recovery_distance", "source_contacts.recovery_frames_valid",
            "player_rallies.recovery_distance_median",
        ),
        "Cut because predicted contact and server attribution were too weak for a "
        "player-specific window. Human ShuttleSet contacts fix the contact frame, and the "
        "hitter resolved against the match table's own side assignment fixes the "
        "non-striking player, so both weak inputs are gone.",
    ),
    FeatureDisposition(
        "Movement inefficiency", Disposition.KEEP,
        (
            "source_contacts.movement_inefficiency_top",
            "source_contacts.movement_inefficiency_bottom",
            "player_rallies.movement_inefficiency_median",
        ),
        "Cut because production intervals used predicted contacts that missed or added "
        "events. Human ShuttleSet contacts fix each interval's start and end exactly.",
    ),
    FeatureDisposition(
        "Serve speed proxy", Disposition.UNRESOLVED, (),
        "Return, static, and viewport endpoints are undefined and shuttle error is large.",
    ),
    FeatureDisposition(
        "Backward extrapolation", Disposition.UNRESOLVED, (),
        "No defined scene boundary, range, or provenance policy.",
    ),
    FeatureDisposition(
        "Rally-to-commentary association", Disposition.UNRESOLVED, (),
        "Issue #138's lag rule fixes coverage on aligned times with zero ambiguity at "
        "10 s, but accuracy is unmeasured: nobody has labelled a sample to check the "
        "pairs are right.",
    ),
    FeatureDisposition(
        "Commentary sentiment, concept, and player link", Disposition.UNRESOLVED, (),
        "Supported schemas emit no semantic fields and no labelled population exists.",
    ),
    FeatureDisposition(
        "Out-of-position posture states", Disposition.NOT_MEASURED, (),
        "The three states need pose-term definitions.",
    ),
    FeatureDisposition(
        "Rest time, work density, effective playing time", Disposition.NOT_MEASURED, (),
        "Work density and cutaway handling need definitions.",
    ),
    FeatureDisposition(
        "Smash shuttle speed", Disposition.NOT_MEASURED, (),
        "Needs smash classification, which production does not have.",
    ),
    FeatureDisposition(
        "Match duration", Disposition.NOT_MEASURED, (),
        "Depends on complete-rally contacts.",
    ),
    FeatureDisposition(
        "Shot frequency within rally", Disposition.NOT_MEASURED, (),
        "Depends on complete-rally contacts.",
    ),
    FeatureDisposition(
        "Aggression markers", Disposition.NOT_MEASURED, (),
        "Depends on complete-rally contacts and shot classification.",
    ),
    FeatureDisposition(
        "Rally-length distribution by outcome and landing zone", Disposition.NOT_MEASURED, (),
        "Depends on complete-rally contacts, outcome, and landing.",
    ),
    FeatureDisposition(
        "Stroke duration", Disposition.NOT_MEASURED, (),
        "Needs a motion-onset definition.",
    ),
    FeatureDisposition(
        "Court coverage near the shuttle", Disposition.NOT_MEASURED, (),
        "Needs a relative measure and event anchor.",
    ),
    FeatureDisposition(
        "Split-step stance geometry", Disposition.NOT_MEASURED, (),
        "Needs a stance measure and event detector.",
    ),
    FeatureDisposition(
        "Net-game share, clear share, backhand proportion, forced-to-unforced error ratio, "
        "shot-outcome success by type, footwork-to-shot coupling, hit height, "
        "shot-selection deception",
        Disposition.OUT_OF_SCOPE, (),
        "Outside the trial. No gate planned.",
    ),
)


def frozen_column_names() -> dict[str, tuple[str, ...]]:
    """Return every table's ordered column names, the frozen surface."""
    return {table.name: table.column_names() for table in TABLES}


def validate_table(table: TableSpec, frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy in frozen column order with frozen dtypes, or raise."""
    expected = table.column_names()
    if set(frame.columns) != set(expected):
        missing = sorted(set(expected) - set(frame.columns))
        extra = sorted(set(frame.columns) - set(expected))
        raise ValueError(f"{table.name} columns differ: missing={missing}, extra={extra}")
    result = pd.DataFrame(index=frame.index)
    for column in table.columns:
        values = frame[column.name].astype(_PANDAS_DTYPES[column.type])
        null = values.isna()
        if null.any() and not column.nullable:
            raise ValueError(f"{table.name}.{column.name} must not contain nulls")
        if column.type is ColumnType.STRING and bool((values[~null] == "").any()):
            raise ValueError(f"{table.name}.{column.name} must not contain empty strings")
        result[column.name] = values
    if result.duplicated(subset=list(table.key)).any():
        raise ValueError(f"{table.name} contains duplicate keys {table.key}")
    return result.sort_values(list(table.key), kind="stable").reset_index(drop=True)


def write_table(directory: Path, table: TableSpec, frame: pd.DataFrame) -> Path:
    """Validate and store one table as deterministic gzip-compressed CSV."""
    validated = validate_table(table, frame)
    buffer = io.StringIO()
    validated.to_csv(buffer, index=False, lineterminator="\n")
    destination = Path(directory) / table.filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(
            gzip.compress(buffer.getvalue().encode("utf-8"), compresslevel=9, mtime=0)
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def read_table(directory: Path, table: TableSpec) -> pd.DataFrame:
    """Load one table with frozen dtypes, treating only empty fields as null."""
    frame = pd.read_csv(
        Path(directory) / table.filename,
        dtype=table.pandas_dtypes(),
        keep_default_na=False,
        na_values=[""],
    )
    return validate_table(table, frame)
