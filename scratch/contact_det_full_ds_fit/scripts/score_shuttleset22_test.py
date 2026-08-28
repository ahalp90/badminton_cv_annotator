"""Score the frozen ShuttleSet22 predictions after their label-free check."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from scratch.contact_det.scripts.score_contact_rallies import (
    DEFAULT_CONFIDENCE_REQUIREMENTS,
    FixedEvent,
    FixedSpan,
    RallyReference,
    fixed_spans_from_evidence,
    normalise_rallies,
    unassigned_events,
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
    VIDEO_IDS,
    VIDEO_RESULT_SCHEMA,
    SourceSpec,
    load_source_specs,
)
from scratch.contact_det_full_ds_fit.scripts.score_contact_baseline import (
    _match_contacts,
)

RESULT_SCHEMA = "shuttleset22-contact-test-result/1"
CLEAN_LABEL_SCHEMA = "shuttleset22-clean-contact-labels/1"
COMBINED_PREDICTIONS_SHA256 = (
    "6199ab99fe2746f83b7f90cc2e2c02301acbd5f90dcf02c989af65ca6be5bd04"
)
ANNOTATION_CORPUS_SHA256 = (
    "2c0208d13d13a4b72a9005ec16e92c442bfe5f223e0f9c499ea5a36f4339052c"
)
ANNOTATION_TREE_SHA256 = (
    "55f832221646229b8b65dea31e24e8d02e0876fd6d0799cb0f6eff12583e1485"
)
EXPECTED_SOURCE_ROWS = 43_159
EXPECTED_USABLE_ROWS = 38_218
EXPECTED_USABLE_RALLIES = 3_422
TIMING_TOLERANCES = (1, 2, 5, 10)
WHOLE_RALLY_TOLERANCES = (5, 10)
SECTION_MAPPINGS = (
    "no_labelled_rally",
    "one_labelled_rally",
    "several_labelled_rallies",
)
RALLY_OUTCOMES = (
    "missing_contacts_only",
    "extra_contacts_only",
    "missing_and_extra_contacts",
    "timing_mismatch_equal_counts",
    "predicted_side_unanswered",
    "wrong_predicted_side",
    "human_side_unassessable",
    "fully_correct",
)
SOURCE_COMMIT = re.compile(r"^[0-9a-f]{7,40}$")

COMBINED_FIELDS = {
    "schema",
    "status",
    "source_commit",
    "labels_read",
    "video_ids",
    "source_manifest_sha256",
    "prepared_artifact_identity_sha256",
    "inpaint_run_state_sha256",
    "model_sha256",
    "model_result_sha256",
    "setting_result_sha256",
    "score_cutoff",
    "nearby_distance_at_30_fps",
    "videos",
}
PREDICTION_FIELDS = {
    "schema",
    "video_id",
    "fixture",
    "fps",
    "frame_count",
    "spans",
    "contacts",
}
SPAN_FIELDS = {"span_id", "start_frame", "end_frame"}
CONTACT_FIELDS = {"frame", "contact_score", "predicted_side", "span_id"}
POPULATION_FIELDS = {
    "source_contact_rows",
    "usable_contact_rows",
    "excluded_flaw_rows",
    "excluded_invalid_frame_rows",
    "usable_rallies",
    "excluded_incomplete_rallies",
    "excluded_incomplete_rally_rows",
    "excluded_non_monotonic_rallies",
    "excluded_non_monotonic_rally_rows",
}

TableReader = Callable[..., Any]
LabelLoader = Callable[
    [Path, Sequence[SourceSpec], Mapping[int, int], Path], "CleanLabels"
]


@dataclass(frozen=True)
class HumanContact:
    """One accepted human contact and its implicit player side."""

    frame: int
    side: str | None


@dataclass(frozen=True)
class HumanRally:
    """One accepted ShuttleSet22 rally."""

    set_id: str
    rally: int
    contacts: tuple[HumanContact, ...]
    ball_rounds: tuple[int, ...]
    contact_types: tuple[str | None, ...]

    @property
    def stroke_frames(self) -> tuple[int, ...]:
        return tuple(contact.frame for contact in self.contacts)


@dataclass(frozen=True)
class CleanLabels:
    """Checked clean labels and their saved population record."""

    rallies_by_fixture: Mapping[str, tuple[HumanRally, ...]]
    population_by_fixture: Mapping[str, Mapping[str, int]]
    annotation_corpus_sha256: str
    annotation_tree_sha256: str


@dataclass(frozen=True)
class VerifiedPredictions:
    """The exact frozen prediction file, checked without label access."""

    root: Path
    combined_path: Path
    source_commit: str
    videos: tuple[Mapping[str, Any], ...]
    sources: tuple[SourceSpec, ...]


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be an object with string keys")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a list")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer of at least {minimum}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    source = Path(root).resolve(strict=True)
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(b"\0")
        digest.update(_sha256(path).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def annotation_corpus_sha256(annotation_root: Path) -> str:
    """Hash the official match table and every set table using the pinned rule."""
    root = Path(annotation_root).resolve(strict=True)
    set_root = root / "set"
    paths = (set_root / "match.csv", *sorted(set_root.glob("*/set*.csv")))
    if not paths[0].is_file() or len(paths) == 1:
        raise FileNotFoundError("ShuttleSet22 annotation tables are incomplete")
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise FileNotFoundError(
            "ShuttleSet22 annotations must be regular non-symlink files"
        )

    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    source_path = Path(path)
    if source_path.name.endswith(".gz"):
        with gzip.open(source_path, "rt", encoding="utf-8") as source:
            value = json.load(source)
    else:
        value = json.loads(source_path.read_text(encoding="utf-8"))
    return dict(_mapping(value, label))


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.partial")
    encoded = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if destination.name.endswith(".gz"):
        with (
            temporary.open("wb") as raw,
            gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0
            ) as zipped,
        ):
            zipped.write(encoded)
    else:
        temporary.write_bytes(encoded)
    os.replace(temporary, destination)


def _utc_time(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=UTC).isoformat()


def _normalise_side(value: object, label: str) -> str | None:
    if value is None:
        return None
    if value == "Top":
        return "Top"
    if value in {"Bot", "Bottom"}:
        return "Bot"
    raise ValueError(f"{label}: player side differs")


def _validate_prediction_payload(
    value: object, expected_video_id: int
) -> Mapping[str, Any]:
    payload = _mapping(value, f"video {expected_video_id}: prediction")
    if set(payload) != PREDICTION_FIELDS:
        raise ValueError(f"video {expected_video_id}: prediction fields differ")
    frame_count = _integer(
        payload["frame_count"], f"video {expected_video_id}: frame count", minimum=1
    )
    if (
        payload["schema"] != PREDICTION_SCHEMA
        or payload["video_id"] != expected_video_id
        or payload["fixture"] != str(expected_video_id)
        or payload["fps"] != EXPECTED_FPS
    ):
        raise ValueError(f"video {expected_video_id}: prediction identity differs")

    spans = _sequence(payload["spans"], f"video {expected_video_id}: spans")
    checked_spans: list[tuple[int, int, int]] = []
    previous_end = -1
    for expected_span_id, raw_span in enumerate(spans):
        span = _mapping(raw_span, f"video {expected_video_id}: span")
        if set(span) != SPAN_FIELDS:
            raise ValueError(f"video {expected_video_id}: span fields differ")
        span_id = _integer(span["span_id"], f"video {expected_video_id}: span ID")
        start = _integer(span["start_frame"], f"video {expected_video_id}: span start")
        end = _integer(
            span["end_frame"], f"video {expected_video_id}: span end", minimum=1
        )
        if (
            span_id != expected_span_id
            or end <= start
            or start < previous_end
            or end > frame_count
        ):
            raise ValueError(f"video {expected_video_id}: span bounds differ")
        checked_spans.append((span_id, start, end))
        previous_end = end

    contacts = _sequence(payload["contacts"], f"video {expected_video_id}: contacts")
    previous_frame = -1
    for raw_contact in contacts:
        contact = _mapping(raw_contact, f"video {expected_video_id}: contact")
        if set(contact) != CONTACT_FIELDS:
            raise ValueError(f"video {expected_video_id}: contact fields differ")
        frame = _integer(contact["frame"], f"video {expected_video_id}: contact frame")
        score = contact["contact_score"]
        if (
            frame <= previous_frame
            or frame >= frame_count
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not SCORE_CUTOFF <= float(score) <= 1.0
        ):
            raise ValueError(f"video {expected_video_id}: contact values differ")
        previous_frame = frame
        _normalise_side(contact["predicted_side"], f"video {expected_video_id}/{frame}")
        containing = [
            span_id for span_id, start, end in checked_spans if start <= frame < end
        ]
        expected_span_id = containing[0] if containing else None
        if len(containing) > 1 or contact["span_id"] != expected_span_id:
            raise ValueError(f"video {expected_video_id}/{frame}: span ID differs")
    return payload


def validate_frozen_predictions(
    prediction_root: Path,
    source_manifest: Path,
) -> VerifiedPredictions:
    """Check the exact combined file and all children without touching labels."""
    root = Path(prediction_root)
    combined_path = root / "combined_predictions.json.gz"
    combined_hash = _sha256(combined_path)
    if combined_hash != COMBINED_PREDICTIONS_SHA256:
        raise ValueError("combined prediction SHA-256 differs")
    sources = load_source_specs(source_manifest)
    if tuple(source.video_id for source in sources) != VIDEO_IDS:
        raise ValueError("source manifest test video order differs")

    run_state = _read_json(root / "run_state.json", "prediction run state")
    expected_run_state = {
        "schema": RUN_STATE_SCHEMA,
        "status": "complete",
        "expected_video_ids": list(VIDEO_IDS),
        "completed_video_ids": list(VIDEO_IDS),
        "completed_count": len(VIDEO_IDS),
        "combined_prediction_sha256": combined_hash,
    }
    if run_state != expected_run_state:
        raise ValueError("prediction run state differs")

    combined = _read_json(combined_path, "combined predictions")
    if set(combined) != COMBINED_FIELDS:
        raise ValueError("combined prediction fields differ")
    expected_identity = {
        "schema": COMBINED_SCHEMA,
        "status": "complete",
        "labels_read": False,
        "video_ids": list(VIDEO_IDS),
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "prepared_artifact_identity_sha256": ARTIFACT_IDENTITY_SHA256,
        "inpaint_run_state_sha256": INPAINT_RUN_STATE_SHA256,
        "model_sha256": MODEL_SHA256,
        "model_result_sha256": MODEL_RESULT_SHA256,
        "setting_result_sha256": SETTING_RESULT_SHA256,
        "score_cutoff": SCORE_CUTOFF,
        "nearby_distance_at_30_fps": NEARBY_DISTANCE_AT_30_FPS,
    }
    if any(combined.get(name) != value for name, value in expected_identity.items()):
        raise ValueError("combined prediction identity differs")
    source_commit = combined.get("source_commit")
    if (
        not isinstance(source_commit, str)
        or SOURCE_COMMIT.fullmatch(source_commit) is None
    ):
        raise ValueError("combined prediction source commit differs")

    video_directories = root / "videos"
    expected_directories = [f"ss22_{video_id:02d}" for video_id in VIDEO_IDS]
    actual_directories = sorted(
        path.name for path in video_directories.iterdir() if path.is_dir()
    )
    if actual_directories != sorted(expected_directories):
        raise ValueError("prediction video directories differ")
    if list(video_directories.glob("*.working")):
        raise ValueError("unfinished prediction directories remain")

    raw_videos = _sequence(combined["videos"], "combined prediction videos")
    if len(raw_videos) != len(VIDEO_IDS):
        raise ValueError("combined prediction video count differs")
    checked_videos: list[Mapping[str, Any]] = []
    for video_id, raw_video in zip(VIDEO_IDS, raw_videos, strict=True):
        combined_video = _validate_prediction_payload(raw_video, video_id)
        directory = video_directories / f"ss22_{video_id:02d}"
        result = _read_json(directory / "result.json", "video prediction result")
        expected_result = {
            "schema": VIDEO_RESULT_SCHEMA,
            "status": "complete",
            "video_id": video_id,
            "fixture": str(video_id),
            "source_commit": source_commit,
            "labels_read": False,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "prepared_artifact_identity_sha256": ARTIFACT_IDENTITY_SHA256,
            "inpaint_run_state_sha256": INPAINT_RUN_STATE_SHA256,
            "model_sha256": MODEL_SHA256,
            "model_result_sha256": MODEL_RESULT_SHA256,
            "setting_result_sha256": SETTING_RESULT_SHA256,
            "score_cutoff": SCORE_CUTOFF,
            "nearby_distance_at_30_fps": NEARBY_DISTANCE_AT_30_FPS,
            "frame_count": combined_video["frame_count"],
            "kept_contact_count": len(combined_video["contacts"]),
        }
        if any(result.get(name) != value for name, value in expected_result.items()):
            raise ValueError(f"video {video_id}: saved prediction result differs")
        output_hashes = _mapping(
            result.get("output_hashes"), f"video {video_id}: output hashes"
        )
        if set(output_hashes) != set(PREDICTION_OUTPUT_FILENAMES):
            raise ValueError(f"video {video_id}: output hash list differs")
        for relative_path, expected_hash in output_hashes.items():
            if (
                not isinstance(expected_hash, str)
                or _sha256(directory / relative_path) != expected_hash
            ):
                raise ValueError(f"video {video_id}: {relative_path} hash differs")
        child = _read_json(directory / "predictions.json.gz", "child prediction")
        if child != combined_video:
            raise ValueError(f"video {video_id}: combined and child predictions differ")
        checked_videos.append(combined_video)
    return VerifiedPredictions(
        root,
        combined_path,
        source_commit,
        tuple(checked_videos),
        sources,
    )


def _player_slot(row: Any, table_module: Any) -> str | None:
    player_y = table_module.to_numeric(
        table_module.Series([row["player_location_y"]]), errors="coerce"
    ).iloc[0]
    opponent_y = table_module.to_numeric(
        table_module.Series([row["opponent_location_y"]]), errors="coerce"
    ).iloc[0]
    if (
        not np.isfinite(player_y)
        or not np.isfinite(opponent_y)
        or player_y == opponent_y
    ):
        return None
    return "Top" if player_y < opponent_y else "Bot"


def load_annotation_rallies(
    set_directory: Path,
    frame_count: int,
    *,
    table_reader: TableReader | None = None,
) -> tuple[tuple[HumanRally, ...], dict[str, int]]:
    """Apply the preserved whole-rally ShuttleSet22 cleaning rule."""
    if table_reader is None:
        import pandas as pd

        table_reader = pd.read_csv
    else:
        import pandas as pd

    tables = []
    for path in sorted(Path(set_directory).glob("set*.csv")):
        table = table_reader(path)
        table["set_id"] = path.stem
        tables.append(table)
    if not tables:
        raise ValueError(f"no ShuttleSet22 set tables under {set_directory}")
    contacts = pd.concat(tables, ignore_index=True)
    frames = pd.to_numeric(contacts["frame_num"], errors="coerce")
    invalid_frame = frames.isna() | (frames < 0) | (frames >= frame_count)
    flaw_marked = contacts["flaw"].notna()
    rallies: list[HumanRally] = []
    incomplete = 0
    incomplete_rows = 0
    non_monotonic = 0
    non_monotonic_rows = 0
    for (set_id, rally_number), group in contacts.groupby(
        ["set_id", "rally"], sort=True
    ):
        group_invalid = invalid_frame.loc[group.index]
        group_flaw = flaw_marked.loc[group.index]
        if bool((group_invalid | group_flaw).any()):
            incomplete += 1
            incomplete_rows += len(group)
            continue
        group = group.copy()
        group["frame_num"] = frames.loc[group.index].astype(int)
        ordered = group.sort_values(["ball_round", "frame_num"])
        contact_frames = [int(value) for value in ordered["frame_num"]]
        if not contact_frames or any(
            right <= left for left, right in pairwise(contact_frames)
        ):
            non_monotonic += 1
            non_monotonic_rows += len(group)
            continue
        human_contacts = tuple(
            HumanContact(int(row["frame_num"]), _player_slot(row, pd))
            for _, row in ordered.iterrows()
        )
        rallies.append(
            HumanRally(
                set_id=str(set_id),
                rally=int(rally_number),
                contacts=human_contacts,
                ball_rounds=tuple(int(value) for value in ordered["ball_round"]),
                contact_types=tuple(
                    None if pd.isna(value) else str(value) for value in ordered["type"]
                ),
            )
        )
    population = {
        "source_contact_rows": len(contacts),
        "usable_contact_rows": sum(len(rally.contacts) for rally in rallies),
        "excluded_flaw_rows": int(flaw_marked.sum()),
        "excluded_invalid_frame_rows": int((invalid_frame & ~flaw_marked).sum()),
        "usable_rallies": len(rallies),
        "excluded_incomplete_rallies": incomplete,
        "excluded_incomplete_rally_rows": incomplete_rows,
        "excluded_non_monotonic_rallies": non_monotonic,
        "excluded_non_monotonic_rally_rows": non_monotonic_rows,
    }
    return tuple(rallies), population


def _sum_population(
    population_by_fixture: Mapping[str, Mapping[str, int]],
) -> dict[str, int]:
    if any(
        set(population) != POPULATION_FIELDS
        for population in population_by_fixture.values()
    ):
        raise ValueError("label population fields differ")
    return {
        field: sum(population[field] for population in population_by_fixture.values())
        for field in sorted(POPULATION_FIELDS)
    }


def _clean_label_payload(
    labels: CleanLabels,
    sources: Sequence[SourceSpec],
) -> dict[str, object]:
    videos: list[dict[str, object]] = []
    for source in sources:
        fixture = str(source.video_id)
        videos.append(
            {
                "video_id": source.video_id,
                "fixture": fixture,
                "population": dict(labels.population_by_fixture[fixture]),
                "rallies": [
                    {
                        "set_id": rally.set_id,
                        "rally": rally.rally,
                        "ball_rounds": list(rally.ball_rounds),
                        "contact_types": list(rally.contact_types),
                        "contacts": [asdict(contact) for contact in rally.contacts],
                    }
                    for rally in labels.rallies_by_fixture[fixture]
                ],
            }
        )
    return {
        "schema": CLEAN_LABEL_SCHEMA,
        "status": "complete",
        "video_ids": list(VIDEO_IDS),
        "annotation_corpus_sha256": labels.annotation_corpus_sha256,
        "annotation_tree_sha256": labels.annotation_tree_sha256,
        "population": _sum_population(labels.population_by_fixture),
        "videos": videos,
    }


def load_clean_labels(
    annotation_root: Path,
    sources: Sequence[SourceSpec],
    frame_counts: Mapping[int, int],
    clean_output: Path,
    *,
    table_reader: TableReader | None = None,
) -> CleanLabels:
    """Check label identities, clean every table, then save the clean rows."""
    corpus_hash = annotation_corpus_sha256(annotation_root)
    if corpus_hash != ANNOTATION_CORPUS_SHA256:
        raise ValueError("official annotation corpus SHA-256 differs")
    tree_hash = _tree_digest(annotation_root)
    if tree_hash != ANNOTATION_TREE_SHA256:
        raise ValueError("annotation tree SHA-256 differs")
    set_root = Path(annotation_root) / "set"
    if table_reader is None:
        import pandas as pd

        table_reader = pd.read_csv
    match_table = table_reader(set_root / "match.csv", usecols=["id", "video"])
    if set(match_table.columns) != {"id", "video"} or len(match_table) != 58:
        raise ValueError("official ShuttleSet22 match table differs")
    match_ids = [int(value) for value in match_table["id"]]
    match_names = [str(value) for value in match_table["video"]]
    if match_ids != list(range(1, 59)) or len(set(match_names)) != 58:
        raise ValueError("official ShuttleSet22 match identities differ")
    actual_names = sorted(path.name for path in set_root.iterdir() if path.is_dir())
    if actual_names != sorted(match_names):
        raise ValueError("official ShuttleSet22 annotation directories differ")
    official_names = dict(zip(match_ids, match_names, strict=True))
    if any(official_names[source.video_id] != source.video_name for source in sources):
        raise ValueError("fixed test videos differ from the official annotations")

    rallies_by_fixture: dict[str, tuple[HumanRally, ...]] = {}
    population_by_fixture: dict[str, Mapping[str, int]] = {}
    for source in sources:
        rallies, population = load_annotation_rallies(
            set_root / source.video_name,
            frame_counts[source.video_id],
            table_reader=table_reader,
        )
        fixture = str(source.video_id)
        rallies_by_fixture[fixture] = rallies
        population_by_fixture[fixture] = population
    labels = CleanLabels(
        rallies_by_fixture,
        population_by_fixture,
        corpus_hash,
        tree_hash,
    )
    payload = _clean_label_payload(labels, sources)
    _write_json(clean_output, payload)
    population = _mapping(payload["population"], "clean-label population")
    if (
        population["source_contact_rows"] != EXPECTED_SOURCE_ROWS
        or population["usable_contact_rows"] != EXPECTED_USABLE_ROWS
        or population["usable_rallies"] != EXPECTED_USABLE_RALLIES
    ):
        raise ValueError("clean ShuttleSet22 population differs from the fixed recount")
    return labels


def _events_by_fixture(
    verified: VerifiedPredictions,
) -> dict[str, tuple[FixedEvent, ...]]:
    events: dict[str, tuple[FixedEvent, ...]] = {}
    for video in verified.videos:
        fixture = str(video["fixture"])
        events[fixture] = tuple(
            FixedEvent(
                fixture,
                int(contact["frame"]),
                float(contact["contact_score"]),
                _normalise_side(
                    contact["predicted_side"], f"{fixture}: predicted side"
                ),
            )
            for contact in video["contacts"]
        )
    return events


def _fixed_spans(
    verified: VerifiedPredictions,
    events_by_fixture: Mapping[str, Sequence[FixedEvent]],
) -> tuple[FixedSpan, ...]:
    evidence = {
        "fixtures": [
            {"fixture": video["fixture"], "spans": video["spans"]}
            for video in verified.videos
        ]
    }
    return fixed_spans_from_evidence(evidence, events_by_fixture)


def _contact_rows(
    rallies: Sequence[HumanRally],
) -> tuple[tuple[HumanContact, ...], frozenset[int]]:
    contacts: list[HumanContact] = []
    first_indices: set[int] = set()
    for rally in rallies:
        first_indices.add(len(contacts))
        contacts.extend(rally.contacts)
    return tuple(contacts), frozenset(first_indices)


def _error_summary(
    offsets: Sequence[int], *, absolute: bool
) -> dict[str, int | float | None]:
    values = np.asarray(offsets, dtype=np.int32)
    if absolute:
        values = np.abs(values)
    if not len(values):
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "minimum": int(np.min(values)),
        "maximum": int(np.max(values)),
    }


def _timing_counts(
    rallies: Sequence[HumanRally],
    events: Sequence[FixedEvent],
    tolerance: int,
) -> tuple[dict[str, object], list[tuple[int, int, int]], tuple[HumanContact, ...]]:
    contacts, first_indices = _contact_rows(rallies)
    expected = np.asarray([contact.frame for contact in contacts], dtype=np.int32)
    predicted = np.asarray([event.frame for event in events], dtype=np.int32)
    matches = _match_contacts(expected, predicted, tolerance)
    first_matched = sum(index in first_indices for index, _, _ in matches)
    first_count = len(first_indices)
    later_count = len(contacts) - first_count
    later_matched = len(matches) - first_matched
    precision = len(matches) / len(events) if events else 0.0
    recall = len(matches) / len(contacts) if contacts else 0.0
    f1 = (
        0.0
        if precision + recall == 0
        else 2.0 * precision * recall / (precision + recall)
    )
    offsets = [offset for _, _, offset in matches]
    result: dict[str, object] = {
        "labelled_contacts": len(contacts),
        "predicted_contacts": len(events),
        "matched_contacts": len(matches),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "first_contacts": first_count,
        "matched_first_contacts": first_matched,
        "first_contact_recall": first_matched / first_count if first_count else 0.0,
        "later_contacts": later_count,
        "matched_later_contacts": later_matched,
        "later_contact_recall": later_matched / later_count if later_count else 0.0,
        "signed_frame_error": _error_summary(offsets, absolute=False),
        "absolute_frame_error": _error_summary(offsets, absolute=True),
    }
    return result, matches, contacts


def _sum_timing_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    labelled = sum(int(row["labelled_contacts"]) for row in rows)
    predicted = sum(int(row["predicted_contacts"]) for row in rows)
    matched = sum(int(row["matched_contacts"]) for row in rows)
    first = sum(int(row["first_contacts"]) for row in rows)
    first_matched = sum(int(row["matched_first_contacts"]) for row in rows)
    later = sum(int(row["later_contacts"]) for row in rows)
    later_matched = sum(int(row["matched_later_contacts"]) for row in rows)
    signed_offsets: list[int] = []
    for row in rows:
        signed_offsets.extend(int(value) for value in row["_offsets"])
    precision = matched / predicted if predicted else 0.0
    recall = matched / labelled if labelled else 0.0
    return {
        "labelled_contacts": labelled,
        "predicted_contacts": predicted,
        "matched_contacts": matched,
        "precision": precision,
        "recall": recall,
        "f1": 0.0
        if precision + recall == 0
        else 2.0 * precision * recall / (precision + recall),
        "first_contacts": first,
        "matched_first_contacts": first_matched,
        "first_contact_recall": first_matched / first if first else 0.0,
        "later_contacts": later,
        "matched_later_contacts": later_matched,
        "later_contact_recall": later_matched / later if later else 0.0,
        "signed_frame_error": _error_summary(signed_offsets, absolute=False),
        "absolute_frame_error": _error_summary(signed_offsets, absolute=True),
    }


def timing_metrics(
    labels: CleanLabels,
    events_by_fixture: Mapping[str, Sequence[FixedEvent]],
) -> dict[str, object]:
    """Score one-to-one contact timing at every fixed tolerance."""
    output: dict[str, object] = {}
    for tolerance in TIMING_TOLERANCES:
        per_video: list[dict[str, object]] = []
        for video_id in VIDEO_IDS:
            fixture = str(video_id)
            counts, matches, _ = _timing_counts(
                labels.rallies_by_fixture[fixture],
                events_by_fixture[fixture],
                tolerance,
            )
            row = {"video_id": video_id, "fixture": fixture, **counts}
            row["_offsets"] = [offset for _, _, offset in matches]
            per_video.append(row)
        total = _sum_timing_rows(per_video)
        for row in per_video:
            row.pop("_offsets")
        output[str(tolerance)] = {
            "tolerance_frames": tolerance,
            "total": total,
            "by_video": per_video,
        }
    return output


def player_side_metrics(
    labels: CleanLabels,
    events_by_fixture: Mapping[str, Sequence[FixedEvent]],
) -> dict[str, object]:
    """Score player side only where timing matched a known human side."""
    output: dict[str, object] = {}
    for tolerance in TIMING_TOLERANCES:
        per_video: list[dict[str, object]] = []
        for video_id in VIDEO_IDS:
            fixture = str(video_id)
            _, matches, contacts = _timing_counts(
                labels.rallies_by_fixture[fixture],
                events_by_fixture[fixture],
                tolerance,
            )
            known = answered = correct = 0
            for contact_index, event_index, _ in matches:
                human_side = contacts[contact_index].side
                if human_side is None:
                    continue
                known += 1
                predicted_side = events_by_fixture[fixture][event_index].predicted_side
                if predicted_side is None:
                    continue
                answered += 1
                correct += predicted_side == human_side
            per_video.append(
                {
                    "video_id": video_id,
                    "fixture": fixture,
                    "timing_matched_labels": len(matches),
                    "known_human_sides": known,
                    "human_side_coverage": known / len(matches) if matches else None,
                    "predicted_side_answers": answered,
                    "prediction_coverage": answered / known if known else None,
                    "correct_player_sides": correct,
                    "accuracy_when_both_answered": correct / answered
                    if answered
                    else None,
                }
            )
        matched = sum(int(row["timing_matched_labels"]) for row in per_video)
        known = sum(int(row["known_human_sides"]) for row in per_video)
        answered = sum(int(row["predicted_side_answers"]) for row in per_video)
        correct = sum(int(row["correct_player_sides"]) for row in per_video)
        output[str(tolerance)] = {
            "tolerance_frames": tolerance,
            "total": {
                "timing_matched_labels": matched,
                "known_human_sides": known,
                "human_side_coverage": known / matched if matched else None,
                "predicted_side_answers": answered,
                "prediction_coverage": answered / known if known else None,
                "correct_player_sides": correct,
                "accuracy_when_both_answered": correct / answered if answered else None,
            },
            "by_video": per_video,
        }
    return output


def _rally_category(
    span: FixedSpan,
    rally: RallyReference,
    human_contacts: Sequence[HumanContact],
    tolerance: int,
) -> str:
    event_frames = np.asarray([event.frame for event in span.events], dtype=np.int32)
    expected_frames = np.asarray(rally.frames, dtype=np.int32)
    matches = _match_contacts(expected_frames, event_frames, tolerance)
    if len(expected_frames) == len(event_frames) and len(matches) < len(
        expected_frames
    ):
        return "timing_mismatch_equal_counts"
    missing = len(matches) < len(expected_frames)
    extra = len(matches) < len(event_frames)
    if missing and extra:
        return "missing_and_extra_contacts"
    if missing:
        return "missing_contacts_only"
    if extra:
        return "extra_contacts_only"
    if any(contact.side is None for contact in human_contacts):
        return "human_side_unassessable"
    if any(
        span.events[event_index].predicted_side is None for _, event_index, _ in matches
    ):
        return "predicted_side_unanswered"
    if any(
        span.events[event_index].predicted_side != human_contacts[contact_index].side
        for contact_index, event_index, _ in matches
    ):
        return "wrong_predicted_side"
    return "fully_correct"


def _rally_lookup(
    labels: CleanLabels,
) -> dict[tuple[str, str], HumanRally]:
    return {
        (fixture, f"{rally.set_id}:{rally.rally}"): rally
        for fixture, rallies in labels.rallies_by_fixture.items()
        for rally in rallies
    }


def whole_rally_metrics(
    labels: CleanLabels,
    spans: Sequence[FixedSpan],
    events_by_fixture: Mapping[str, Sequence[FixedEvent]],
) -> dict[str, object]:
    """Map sections to rallies and report the fixed exclusive outcomes."""
    references = normalise_rallies(labels.rallies_by_fixture)
    lookup = _rally_lookup(labels)
    mapping_rows: list[dict[str, object]] = []
    candidates_by_span: dict[tuple[str, int], tuple[RallyReference, ...]] = {}
    for span in spans:
        candidates = tuple(
            rally
            for rally in references[span.fixture]
            if any(span.start_frame <= frame < span.end_frame for frame in rally.frames)
        )
        candidates_by_span[(span.fixture, span.span_id)] = candidates
        mapping_rows.append(
            {
                "fixture": span.fixture,
                "span_id": span.span_id,
                "start_frame": span.start_frame,
                "end_frame": span.end_frame,
                "labelled_rally_count": len(candidates),
                "mapping": (
                    "no_labelled_rally"
                    if not candidates
                    else "one_labelled_rally"
                    if len(candidates) == 1
                    else "several_labelled_rallies"
                ),
                "rally_ids": [rally.rally_id for rally in candidates],
            }
        )
    counted_mappings = Counter(str(row["mapping"]) for row in mapping_rows)
    mapping_counts = {
        mapping: counted_mappings[mapping] for mapping in SECTION_MAPPINGS
    }

    outcomes: dict[str, object] = {}
    for tolerance in WHOLE_RALLY_TOLERANCES:
        details: list[dict[str, object]] = []
        for span in spans:
            candidates = candidates_by_span[(span.fixture, span.span_id)]
            if len(candidates) != 1:
                continue
            rally = candidates[0]
            human_rally = lookup[(span.fixture, rally.rally_id)]
            category = _rally_category(
                span,
                rally,
                human_rally.contacts,
                tolerance,
            )
            details.append(
                {
                    "fixture": span.fixture,
                    "span_id": span.span_id,
                    "rally_id": rally.rally_id,
                    "labelled_contacts": len(rally.frames),
                    "predicted_contacts": len(span.events),
                    "outcome": category,
                }
            )
        counted_outcomes = Counter(str(row["outcome"]) for row in details)
        counts = {outcome: counted_outcomes[outcome] for outcome in RALLY_OUTCOMES}
        assessable = len(details) - counts["human_side_unassessable"]
        per_video = []
        for video_id in VIDEO_IDS:
            fixture = str(video_id)
            video_rows = [row for row in details if row["fixture"] == fixture]
            video_counts = Counter(str(row["outcome"]) for row in video_rows)
            per_video.append(
                {
                    "video_id": video_id,
                    "fixture": fixture,
                    "mapped_sections": len(video_rows),
                    "outcome_counts": {
                        outcome: video_counts[outcome] for outcome in RALLY_OUTCOMES
                    },
                }
            )
        outcomes[str(tolerance)] = {
            "tolerance_frames": tolerance,
            "mapped_sections": len(details),
            "outcome_counts": counts,
            "human_side_assessable_sections": assessable,
            "human_side_unassessable_sections": counts["human_side_unassessable"],
            "human_side_unassessable_share": (
                counts["human_side_unassessable"] / len(details) if details else None
            ),
            "fully_correct_sections": counts["fully_correct"],
            "fully_correct_accuracy_when_assessable": (
                counts["fully_correct"] / assessable if assessable else None
            ),
            "by_video": per_video,
            "sections": details,
        }

    unassigned = unassigned_events(spans, events_by_fixture)
    curve: list[dict[str, int | float | None]] = []
    details_at_five = _mapping(outcomes["5"], "five-frame rally results")["sections"]
    category_by_span = {
        (str(row["fixture"]), int(row["span_id"])): str(row["outcome"])
        for row in details_at_five
    }
    for requirement in DEFAULT_CONFIDENCE_REQUIREMENTS:
        retained = [
            span
            for span in spans
            if span.events
            and min(event.timing_score for event in span.events) >= requirement
        ]
        assessable = []
        for span in retained:
            category = category_by_span.get((span.fixture, span.span_id))
            if category is not None and category != "human_side_unassessable":
                assessable.append(span)
        fully_correct = sum(
            category_by_span.get((span.fixture, span.span_id)) == "fully_correct"
            for span in retained
        )
        curve.append(
            {
                "confidence_requirement": requirement,
                "sections_retained": len(retained),
                "human_side_assessable_sections": len(assessable),
                "fully_correct_sections": fully_correct,
                "fully_correct_share_of_retained": (
                    fully_correct / len(retained) if retained else None
                ),
                "fully_correct_accuracy_when_assessable": (
                    fully_correct / len(assessable) if assessable else None
                ),
            }
        )
    return {
        "section_mapping_counts": mapping_counts,
        "section_mappings": mapping_rows,
        "unassigned_contacts": [asdict(event) for event in unassigned],
        "by_tolerance": outcomes,
        "confidence_curve_at_five_frames": curve,
    }


def score_predictions(
    verified: VerifiedPredictions,
    labels: CleanLabels,
) -> dict[str, object]:
    """Calculate every fixed timing, player-side and whole-rally measure."""
    events = _events_by_fixture(verified)
    spans = _fixed_spans(verified, events)
    return {
        "contact_timing": timing_metrics(labels, events),
        "player_side": player_side_metrics(labels, events),
        "whole_rallies": whole_rally_metrics(labels, spans, events),
    }


def score_shuttleset22_test(
    prediction_root: Path,
    source_manifest: Path,
    annotation_root: Path,
    clean_label_output: Path,
    output_path: Path,
    source_commit: str,
    *,
    label_loader: LabelLoader | None = None,
) -> Path:
    """Validate predictions, read labels once, and save the fixed test result."""
    destination = Path(output_path)
    state: dict[str, object] = {
        "schema": RESULT_SCHEMA,
        "status": "running",
        "source_commit": source_commit,
        "labels_read_started": False,
    }
    _write_json(destination, state)
    try:
        if SOURCE_COMMIT.fullmatch(source_commit) is None:
            raise ValueError("source commit must be a short or full Git commit")
        verified = validate_frozen_predictions(prediction_root, source_manifest)
        frame_counts = {
            int(video["video_id"]): int(video["frame_count"])
            for video in verified.videos
        }
        prediction_mtime_ns = verified.combined_path.stat().st_mtime_ns
        label_read_started_ns = time.time_ns()
        if prediction_mtime_ns >= label_read_started_ns:
            raise ValueError("combined prediction does not predate label reading")
        state["labels_read_started"] = True
        state["label_read_started_at_utc"] = _utc_time(label_read_started_ns)
        state["label_read_started_ns"] = label_read_started_ns
        state["combined_prediction_mtime_utc"] = _utc_time(prediction_mtime_ns)
        state["combined_prediction_mtime_ns"] = prediction_mtime_ns
        _write_json(destination, state)
        if label_loader is None:
            labels = load_clean_labels(
                annotation_root,
                verified.sources,
                frame_counts,
                clean_label_output,
            )
        else:
            labels = label_loader(
                Path(annotation_root),
                verified.sources,
                frame_counts,
                Path(clean_label_output),
            )
        result = {
            "schema": RESULT_SCHEMA,
            "status": "complete",
            "source_commit": source_commit,
            "labels_read_started": True,
            "label_read_started_at_utc": state["label_read_started_at_utc"],
            "label_read_started_ns": label_read_started_ns,
            "prediction_source_commit": verified.source_commit,
            "video_ids": list(VIDEO_IDS),
            "inputs": {
                "combined_prediction_file": verified.combined_path.name,
                "combined_prediction_sha256": _sha256(verified.combined_path),
                "combined_prediction_mtime_utc": state["combined_prediction_mtime_utc"],
                "combined_prediction_mtime_ns": prediction_mtime_ns,
                "source_manifest_file": Path(source_manifest).name,
                "source_manifest_sha256": _sha256(source_manifest),
                "annotation_corpus_sha256": labels.annotation_corpus_sha256,
                "annotation_tree_sha256": labels.annotation_tree_sha256,
                "clean_label_file": Path(clean_label_output).name,
                "clean_label_sha256": _sha256(clean_label_output),
            },
            "label_population": _sum_population(labels.population_by_fixture),
            **score_predictions(verified, labels),
        }
        _write_json(destination, result)
        return destination
    except Exception as error:
        failed = {
            **state,
            "status": "failed",
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        _write_json(destination, failed)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--clean-label-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    score_shuttleset22_test(
        args.prediction_root,
        args.source_manifest,
        args.annotation_root,
        args.clean_label_output,
        args.output,
        args.source_commit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
