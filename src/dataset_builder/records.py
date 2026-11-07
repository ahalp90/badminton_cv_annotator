"""Validation-only projection of annotator primitives into rally records."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from annotator.annotate_video import AnnotatorResult
from annotator.rally_results.landing_and_winner import (
    Landing,
    RallyOutcome,
    WinnerSource,
)
from annotator.types import ContactCandidate, ContactEvent, CourtHalf
from dataset_builder._record_validation import (
    RALLY_RECORD_COLLECTION_SCHEMA,
    RALLY_RECORD_PROJECTION_SCHEMA,
    RALLY_RECORD_SCHEMA,
    RALLY_RECORDS_FILENAME,
    reject_manifest_output_cycle,
    validate_live_manifest_extension,
    validate_paired_commentary_provenance,
    validate_record_collection,
)
from dataset_builder.manifest import (
    MANIFEST_FILENAME,
    load_run_manifest,
    redact_configuration,
    run_manifest_sha256,
    write_run_manifest,
)
from dataset_builder.models import RunManifest, StageOutcome
from dataset_builder.vision import load_json_gz, save_json_gz
from scraper.commentary_pairing import CanonicalPairing
from shared.progress import progress_iter
from shared.video_metadata import VideoMetadata


@dataclass(frozen=True)
class SourceReference:
    """Stable acquisition provenance for one exact source video."""

    video_id: str
    basename: str
    title: str
    url: str
    commentary_eligible: bool

    def __post_init__(self) -> None:
        for name in ("video_id", "basename", "title", "url"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"source reference {name} must be a non-empty string")
        if not isinstance(self.commentary_eligible, bool):
            raise TypeError("source reference commentary_eligible must be boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "video_id": self.video_id,
            "basename": self.basename,
            "title": self.title,
            "url": self.url,
            "commentary_eligible": self.commentary_eligible,
        }


@dataclass(frozen=True)
class RallyRecordArtifacts:
    """The compact record collection and its referenced run manifest."""

    records: Path
    run_manifest: Path


@dataclass(frozen=True)
class RallyRecordProjection:
    """Validated source provenance and rally rows for one video."""

    input_manifest_sha256: str
    source: dict[str, object]
    records: tuple[dict[str, object], ...]


def write_rally_record_projection(
    path: Path,
    manifest: RunManifest,
    projection: RallyRecordProjection,
) -> Path:
    """Persist one validated per-video projection and its manifest binding."""
    _validate_projection(manifest, projection)
    return save_json_gz(
        path,
        {
            "schema": RALLY_RECORD_PROJECTION_SCHEMA,
            "input_manifest_sha256": projection.input_manifest_sha256,
            "source": projection.source,
            "records": list(projection.records),
        },
    )


def load_rally_record_projection(
    path: Path,
    manifest: RunManifest,
    *,
    video_id: str,
) -> RallyRecordProjection:
    """Load and validate one exact per-video projection."""
    video_id = _string(video_id, "primitive projection video_id")
    payload = _mapping(load_json_gz(path), "primitive projection")
    expected_fields = {"schema", "input_manifest_sha256", "source", "records"}
    if set(payload) != expected_fields:
        raise ValueError("primitive projection fields differ")
    if payload["schema"] != RALLY_RECORD_PROJECTION_SCHEMA:
        raise ValueError(f"unsupported primitive projection schema: {payload['schema']!r}")
    source = dict(_mapping(payload["source"], "primitive projection source"))
    if source.get("video_id") != video_id:
        raise ValueError("primitive projection video_id differs from the expected video")
    raw_records = payload["records"]
    if not isinstance(raw_records, list):
        raise TypeError("primitive projection records must be a list")
    projection = RallyRecordProjection(
        _string(
            payload["input_manifest_sha256"],
            "primitive projection input_manifest_sha256",
        ),
        source,
        tuple(dict(_mapping(record, "primitive projection record")) for record in raw_records),
    )
    _validate_projection(manifest, projection)
    return projection


def assemble_rally_records(
    *,
    manifest: RunManifest,
    source_dataset: str,
    video_id: str,
    source_reference: SourceReference,
    metadata: VideoMetadata,
    annotation: AnnotatorResult,
    annotation_fps: Fraction,
    annotation_frame_count: int,
    pairing: CanonicalPairing | None,
    chunks: Sequence[Mapping[str, object]],
    commentary_outcome: StageOutcome,
    commentary_reason: str | None,
    commentary_missing_reasons: Mapping[int, str],
    commentary_provenance: Mapping[str, object],
    mask_stage_name: str,
) -> RallyRecordProjection:
    """Validate and join existing producer values without reinterpreting them."""
    _validate_identity(source_dataset, video_id, source_reference, mask_stage_name)
    if not isinstance(metadata, VideoMetadata):
        raise TypeError("metadata must be canonical VideoMetadata")
    if not isinstance(annotation, AnnotatorResult):
        raise TypeError("annotation must be AnnotatorResult")
    _validate_source_metadata(source_reference, metadata)
    _validate_timing(metadata, annotation_fps, annotation_frame_count, pairing, video_id)
    spans = _validate_spans([(rally.sequence.start_frame, rally.sequence.end_frame)
                             for rally in annotation.rallies], metadata.frame_count)
    raw_contacts, accepted_contacts, failures = _contact_payloads(annotation, spans)
    outcome_payloads = _outcome_payloads(annotation, len(spans), metadata.frame_count)
    commentary_payloads = _commentary_payloads(
        video_id=video_id,
        spans=spans,
        pairing=pairing,
        chunks=chunks,
        outcome=commentary_outcome,
        stage_reason=commentary_reason,
        missing_reasons=commentary_missing_reasons,
        provenance=commentary_provenance,
        commentary_eligible=source_reference.commentary_eligible,
    )
    source_payload = {
        "source_dataset": source_dataset,
        "video_id": video_id,
        "source_reference": source_reference.to_dict(),
        "video_metadata": metadata.to_dict(),
        "mask_stage": mask_stage_name,
    }

    records: list[dict[str, object]] = []
    for rally_id, (start_frame, end_frame) in enumerate(progress_iter(spans, "projecting rallies", unit="rallies")):
        duration_frames = end_frame - start_frame
        records.append({
            "schema": RALLY_RECORD_SCHEMA,
            "key": {
                "run_id": manifest.run_id,
                "source_dataset": source_dataset,
                "video_id": video_id,
                "rally_id": rally_id,
            },
            "rally": {
                "rally_id": rally_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "duration_frames": duration_frames,
                "duration_seconds": float(Fraction(duration_frames, 1) / metadata.fps),
            },
            "contacts": {
                "raw_candidates": raw_contacts[rally_id],
                "accepted": accepted_contacts[rally_id],
                "stroke_count": len(annotation.rallies[rally_id].sequence.events),
                "hit_height_failures": failures[rally_id],
                "review_score": annotation.rally_confidence[rally_id] if annotation.rally_confidence else None,
            },
            "outcomes": outcome_payloads[rally_id],
            "commentary": deepcopy(commentary_payloads[rally_id]),
        })
    validate_record_collection(manifest, [source_payload], records)
    return RallyRecordProjection(run_manifest_sha256(manifest), source_payload, tuple(records))


def write_rally_records(
    run_dir: Path,
    manifest: RunManifest,
    projections: Sequence[RallyRecordProjection],
    *,
    code_version: str,
    assembly_configuration: Mapping[str, object],
    projection_manifest: RunManifest | None = None,
) -> RallyRecordArtifacts:
    """Persist validated projections and the immutable run-manifest snapshot."""
    if not isinstance(manifest, RunManifest):
        raise TypeError("manifest must be RunManifest")
    if projection_manifest is None:
        projection_manifest = manifest
    if not isinstance(projection_manifest, RunManifest):
        raise TypeError("projection_manifest must be RunManifest")
    validate_live_manifest_extension(projection_manifest, manifest)
    _validate_manifest_code_version(manifest, code_version)
    run_dir = Path(run_dir)
    records_path = run_dir / RALLY_RECORDS_FILENAME
    reject_manifest_output_cycle(manifest, run_dir, records_path)
    input_manifest_sha256 = run_manifest_sha256(manifest)
    projection_manifest_sha256 = run_manifest_sha256(projection_manifest)
    sources: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    for projection in progress_iter(projections, "assembling videos", unit="videos"):
        if not isinstance(projection, RallyRecordProjection):
            raise TypeError("projections must contain RallyRecordProjection values")
        if projection.input_manifest_sha256 != projection_manifest_sha256:
            raise ValueError("projection input-manifest digest differs from the supplied manifest")
        sources.append(dict(projection.source))
        records.extend(dict(record) for record in projection.records)
    validate_record_collection(projection_manifest, sources, records)
    manifest_path = write_run_manifest(run_dir, manifest)
    records_path = save_json_gz(
        records_path,
        {
            "schema": RALLY_RECORD_COLLECTION_SCHEMA,
            "run_id": manifest.run_id,
            "run_manifest": MANIFEST_FILENAME,
            "input_manifest_sha256": input_manifest_sha256,
            "input_manifest": manifest.to_dict(),
            "code_version": code_version,
            "assembly_configuration": redact_configuration(assembly_configuration),
            "sources": sources,
            "records": records,
        },
    )
    return RallyRecordArtifacts(records=records_path, run_manifest=manifest_path)


def _validate_projection(
    manifest: RunManifest,
    projection: RallyRecordProjection,
) -> None:
    if not isinstance(manifest, RunManifest):
        raise TypeError("manifest must be RunManifest")
    if not isinstance(projection, RallyRecordProjection):
        raise TypeError("projection must be RallyRecordProjection")
    if projection.input_manifest_sha256 != run_manifest_sha256(manifest):
        raise ValueError("projection input-manifest digest differs from the supplied manifest")
    validate_record_collection(manifest, [projection.source], projection.records)


def load_rally_records(path: Path) -> list[dict[str, object]]:
    """Load and structurally validate one compressed rally-record collection."""
    payload = load_json_gz(path)
    expected_fields = {
        "schema", "run_id", "run_manifest", "input_manifest_sha256",
        "input_manifest", "code_version", "assembly_configuration", "sources", "records",
    }
    if set(payload) != expected_fields:
        raise ValueError("rally record collection fields differ")
    if payload["schema"] != RALLY_RECORD_COLLECTION_SCHEMA:
        raise ValueError(f"unsupported rally record collection schema: {payload['schema']!r}")
    run_id = _string(payload["run_id"], "record collection run_id")
    if payload["run_manifest"] != MANIFEST_FILENAME:
        raise ValueError("rally record collection must reference run_manifest.json.gz")
    input_manifest = RunManifest.from_dict(
        dict(_mapping(payload["input_manifest"], "input-manifest snapshot")),
    )
    input_digest = _string(
        payload["input_manifest_sha256"], "record collection input_manifest_sha256",
    )
    if run_manifest_sha256(input_manifest) != input_digest:
        raise ValueError("record collection input-manifest digest differs from its snapshot")
    if input_manifest.run_id != run_id:
        raise ValueError("record collection run_id differs from its input-manifest snapshot")
    code_version = _string(payload["code_version"], "record collection code_version")
    _validate_manifest_code_version(input_manifest, code_version)
    assembly_configuration = _mapping(
        payload["assembly_configuration"], "record assembly configuration",
    )
    if redact_configuration(assembly_configuration) != assembly_configuration:
        raise ValueError("record assembly configuration contains unredacted secrets")
    raw_sources = payload["sources"]
    if not isinstance(raw_sources, list):
        raise TypeError("rally record collection sources must be a list")
    sources = [dict(_mapping(source, "record source")) for source in raw_sources]
    raw_records = payload["records"]
    if not isinstance(raw_records, list):
        raise TypeError("rally record collection records must be a list")
    records = [dict(_mapping(record, "rally record")) for record in raw_records]
    validate_record_collection(input_manifest, sources, records)
    live_manifest = load_run_manifest(Path(path).parent)
    validate_live_manifest_extension(input_manifest, live_manifest)
    _validate_manifest_code_version(live_manifest, code_version)
    return records


def _validate_identity(
    source_dataset: str,
    video_id: str,
    source_reference: SourceReference,
    mask_stage_name: str,
) -> None:
    for name, value in (
        ("source_dataset", source_dataset),
        ("video_id", video_id),
        ("mask_stage_name", mask_stage_name),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    if not isinstance(source_reference, SourceReference):
        raise TypeError("source_reference must be SourceReference")
    if source_reference.video_id != video_id:
        raise ValueError("source reference video_id does not match the exact record video_id")


def _validate_manifest_code_version(manifest: RunManifest, code_version: str) -> None:
    code_version = _string(code_version, "record collection code_version")
    manifest_versions = {stage.fingerprint.source_commit for stage in manifest.stages}
    if manifest_versions and manifest_versions != {code_version}:
        raise ValueError(
            f"run manifest code versions {sorted(manifest_versions)!r} conflict with {code_version!r}"
        )


def _validate_timing(
    metadata: VideoMetadata,
    annotation_fps: Fraction,
    annotation_frame_count: int,
    pairing: CanonicalPairing | None,
    video_id: str,
) -> None:
    if not isinstance(annotation_fps, Fraction) or annotation_fps <= 0:
        raise ValueError("annotation_fps must be a positive exact Fraction")
    if annotation_fps != metadata.fps:
        raise ValueError(
            f"annotation fps {annotation_fps} conflicts with canonical fps {metadata.fps}"
        )
    if isinstance(annotation_frame_count, bool) or not isinstance(annotation_frame_count, int):
        raise TypeError("annotation_frame_count must be an integer")
    if annotation_frame_count != metadata.frame_count:
        raise ValueError(
            f"annotation frame_count {annotation_frame_count} conflicts with canonical "
            f"frame_count {metadata.frame_count}"
        )
    if pairing is None:
        return
    if not isinstance(pairing, CanonicalPairing):
        raise TypeError("pairing must be CanonicalPairing or None")
    if pairing.video_id != video_id:
        raise ValueError("pairing video_id does not match the exact record video_id")
    if pairing.metadata != metadata:
        raise ValueError("pairing metadata conflicts with canonical video metadata")


def _validate_source_metadata(
    source_reference: SourceReference,
    metadata: VideoMetadata,
) -> None:
    if source_reference.basename != metadata.source_path.name:
        raise ValueError("source reference basename conflicts with canonical video metadata")


def _validate_spans(
    raw_spans: object,
    frame_count: int,
) -> list[tuple[int, int]]:
    if not isinstance(raw_spans, list):
        raise TypeError("annotation spans must be a list")
    spans: list[tuple[int, int]] = []
    previous_end = 0
    for rally_id, span in enumerate(raw_spans):
        if not isinstance(span, tuple) or len(span) != 2:
            raise ValueError(f"rally {rally_id} span must be a two-integer tuple")
        start_frame, end_frame = span
        if any(isinstance(value, bool) or not isinstance(value, int) for value in span):
            raise ValueError(f"rally {rally_id} span bounds must be integers")
        if not 0 <= start_frame < end_frame <= frame_count:
            raise ValueError(
                f"rally {rally_id} span [{start_frame}, {end_frame}) is outside "
                f"frame_count {frame_count}"
            )
        if rally_id and start_frame < previous_end:
            raise ValueError("rally spans must be ordered and non-overlapping")
        spans.append((start_frame, end_frame))
        previous_end = end_frame
    return spans


def _contact_payloads(
    annotation: AnnotatorResult,
    spans: Sequence[tuple[int, int]],
) -> tuple[list[list[dict[str, object]]], list[list[dict[str, object]]], list[list[dict[str, object]]]]:
    raw_by_rally: list[list[dict[str, object]]] = [[] for _ in spans]
    for contact in annotation.heuristic_candidates:
        _validate_contact(contact, spans, "raw contact")
        raw_by_rally[contact.rally_id].append(_raw_contact_payload(contact))
    _require_ascending_contact_frames(raw_by_rally, "raw contacts")

    accepted_by_rally: list[list[dict[str, object]]] = [[] for _ in spans]
    for rally_id, rally in enumerate(annotation.rallies):
        start, end = spans[rally_id]
        for stroke_idx, event in enumerate(rally.sequence.events):
            if not isinstance(event, ContactEvent) or isinstance(event.frame, bool) or not isinstance(event.frame, int):
                raise TypeError("accepted contacts must be ContactEvent values with integer frames")
            if not start <= event.frame < end:
                raise ValueError(f"accepted contact frame {event.frame} is outside rally {rally_id}")
            if event.side is not None and not isinstance(event.side, CourtHalf):
                raise ValueError("accepted contact side must be CourtHalf or null")
            if not annotation.heuristic_candidates:
                raw_by_rally[rally_id].append({
                    "contact_frame": event.frame, "wrist_near": None, "temporal_proximity_suppressed": None,
                })
            accepted_by_rally[rally_id].append({
                "stroke_idx": stroke_idx, "contact_frame": event.frame, "player_half": _enum_value(event.side),
            })
    failures = _attach_hit_heights(annotation, accepted_by_rally)
    return raw_by_rally, accepted_by_rally, failures


def _validate_contact(
    contact: object,
    spans: Sequence[tuple[int, int]],
    name: str,
) -> None:
    if not isinstance(contact, ContactCandidate):
        raise TypeError(f"{name} must be ContactCandidate")
    rally_id = contact.rally_id
    frame = contact.contact_frame
    if isinstance(rally_id, bool) or not isinstance(rally_id, int) or not 0 <= rally_id < len(spans):
        raise ValueError(f"{name} rally_id is outside the detected rallies: {rally_id!r}")
    if isinstance(frame, bool) or not isinstance(frame, int):
        raise TypeError(f"{name} frame must be an integer")
    start, end = spans[rally_id]
    if not start <= frame < end:
        raise ValueError(f"{name} frame {frame} is outside rally {rally_id} span [{start}, {end})")
    for field_name in ("wrist_near", "temporal_proximity_suppressed"):
        value = getattr(contact, field_name)
        if value is not None and not isinstance(value, bool):
            raise ValueError(f"{name} {field_name} must be boolean or null")


def _raw_contact_payload(contact: ContactCandidate) -> dict[str, object]:
    return {
        "contact_frame": contact.contact_frame,
        "wrist_near": contact.wrist_near,
        "temporal_proximity_suppressed": contact.temporal_proximity_suppressed,
    }


def _require_ascending_contact_frames(rows: Sequence[Sequence[Mapping[str, object]]], name: str) -> None:
    for rally_id, rally_rows in enumerate(rows):
        frames = [row["contact_frame"] for row in rally_rows]
        if frames != sorted(frames):
            raise ValueError(f"{name} must be in ascending frame order for rally {rally_id}")


def _attach_hit_heights(
    annotation: AnnotatorResult,
    accepted: list[list[dict[str, object]]],
) -> list[list[dict[str, object]]]:
    accepted_frames = {int(row["contact_frame"]) for rows in accepted for row in rows}
    for frame, code in annotation.hit_heights.items():
        if isinstance(frame, bool) or not isinstance(frame, int) or frame not in accepted_frames:
            raise ValueError(f"hit height has no accepted contact at frame {frame!r}")
        if isinstance(code, bool) or not isinstance(code, int) or code not in (1, 2):
            raise ValueError(f"hit height at frame {frame} must use code 1 or 2")
    for frame, reason in annotation.unavailable_hit_height_reasons.items():
        if isinstance(frame, bool) or not isinstance(frame, int) or frame not in accepted_frames:
            raise ValueError(f"hit-height failure has no accepted contact at frame {frame!r}")
        if not isinstance(reason, str) or not reason:
            raise ValueError("hit-height failure reason must be non-empty")
    failures: list[list[dict[str, object]]] = [[] for _ in accepted]
    for rally_id, rows in enumerate(accepted):
        for row in rows:
            frame = int(row["contact_frame"])
            if (frame in annotation.hit_heights) == (frame in annotation.unavailable_hit_height_reasons):
                raise ValueError(f"accepted contact {frame} must have exactly one hit height or failure")
            row["hit_height_code"] = annotation.hit_heights.get(frame)
            if frame in annotation.unavailable_hit_height_reasons:
                failures[rally_id].append({
                    "stroke_idx": row["stroke_idx"], "contact_frame": frame,
                    "reason": annotation.unavailable_hit_height_reasons[frame],
                })
    return failures


def _outcome_payloads(
    annotation: AnnotatorResult,
    rally_count: int,
    frame_count: int,
) -> list[dict[str, object]]:
    resolved = {rally_id for rally_id, rally in enumerate(annotation.rallies) if rally.final_half is not None}
    _validate_outcome_maps(annotation, resolved, frame_count)
    return [_outcome_payload(annotation, rally_id) for rally_id in range(rally_count)]


def _validate_outcome_maps(
    annotation: AnnotatorResult,
    resolved: set[int],
    frame_count: int,
) -> None:
    if set(annotation.outcomes) != resolved:
        raise ValueError("outcome keys must exactly match rallies with resolved strikers")
    for rally_id in resolved:
        outcome = annotation.outcomes[rally_id]
        if not isinstance(outcome, RallyOutcome):
            raise TypeError("rally outcomes must be RallyOutcome values")
        for winner in (outcome.winner, outcome.conservative_landing_winner):
            if winner is not None and not isinstance(winner, CourtHalf):
                raise ValueError("outcome winner must be CourtHalf or null")
        if outcome.winner_source is not None and not isinstance(outcome.winner_source, WinnerSource):
            raise ValueError("outcome source must be WinnerSource or null")
        if outcome.landing_margin_m is not None:
            _finite_number(outcome.landing_margin_m, "outcome landing margin")
        for flag in (outcome.within_line_margin, outcome.within_net_margin, outcome.search_shortened_by_exclusion):
            if not isinstance(flag, bool):
                raise TypeError("outcome ambiguity and exclusion flags must be boolean")
        _validate_landing(outcome.landing, frame_count)


def _validate_landing(landing: object, frame_count: int) -> None:
    if landing is None:
        return
    if not isinstance(landing, Landing):
        raise TypeError("landing values must be Landing or null")
    if (
        isinstance(landing.frame, bool)
        or not isinstance(landing.frame, int)
        or not 0 <= landing.frame < frame_count
    ):
        raise ValueError("landing frame must be an integer inside canonical frame bounds")
    if not isinstance(landing.court_xy, tuple) or len(landing.court_xy) != 2:
        raise ValueError("landing normalized court position must contain two coordinates")
    for coordinate in landing.court_xy:
        _finite_number(coordinate, "landing normalized court coordinate")
    if not isinstance(landing.half, CourtHalf):
        raise TypeError("landing half must be CourtHalf")
    if not isinstance(landing.at_border, bool) or not isinstance(landing.net_ender, bool):
        raise TypeError("landing quality flags must be boolean")


def _outcome_payload(annotation: AnnotatorResult, rally_id: int) -> dict[str, object]:
    outcome = annotation.outcomes.get(rally_id)
    landing = None if outcome is None else outcome.landing
    rally = annotation.rallies[rally_id]
    next_server = annotation.rallies[rally_id + 1].first_half if rally_id + 1 < len(annotation.rallies) else None
    winner = None if outcome is None else outcome.winner
    conservative_winner = None if outcome is None else outcome.conservative_landing_winner
    return {
        "striker_half": _enum_value(annotation.rallies[rally_id].final_half),
        "server_prediction": _enum_value(rally.first_half),
        "next_server": _enum_value(next_server),
        "verdict": {
            "value": None if winner is None else ("won" if winner == rally.final_half else "lost"),
            "source": None if outcome is None else _enum_value(outcome.winner_source),
            "landing_margin_m": None if outcome is None else outcome.landing_margin_m,
            "within_line_margin": None if outcome is None else outcome.within_line_margin,
            "within_net_margin": None if outcome is None else outcome.within_net_margin,
        },
        "landing": None if landing is None else {
            "frame": landing.frame,
            "normalized_court_position": list(landing.court_xy),
            "court_half": landing.half.value,
            "at_image_border": landing.at_border,
            "net_ender": landing.net_ender,
        },
        "geometric_verdict": {
            "value": None if conservative_winner is None else ("won" if conservative_winner == rally.final_half else "lost"),
            "winner": _enum_value(conservative_winner),
            "agreement": None if winner is None or conservative_winner is None else winner == conservative_winner,
            "window_closed_by_mask": (
                None if outcome is None else outcome.search_shortened_by_exclusion
            ),
        },
    }


def _commentary_payloads(
    *,
    video_id: str,
    spans: Sequence[tuple[int, int]],
    pairing: CanonicalPairing | None,
    chunks: Sequence[Mapping[str, object]],
    outcome: StageOutcome,
    stage_reason: str | None,
    missing_reasons: Mapping[int, str],
    provenance: Mapping[str, object],
    commentary_eligible: bool,
) -> list[dict[str, object]]:
    if not isinstance(outcome, StageOutcome):
        raise TypeError("commentary_outcome must be StageOutcome")
    if outcome in {
        StageOutcome.SKIPPED,
        StageOutcome.EXCLUDED,
        StageOutcome.FAILED,
        StageOutcome.UNAVAILABLE,
    } and not stage_reason:
        raise ValueError(f"commentary outcome {outcome.value!r} requires a reason")
    if stage_reason is not None and (not isinstance(stage_reason, str) or not stage_reason):
        raise ValueError("commentary_reason must be non-empty when present")
    if pairing is None and outcome in {StageOutcome.PROCESSED, StageOutcome.SKIPPED}:
        raise ValueError(
            f"commentary outcome {outcome.value!r} requires canonical pairing evidence"
        )
    normalized_provenance = redact_configuration(provenance)
    reason_map = _missing_reason_map(missing_reasons, len(spans))
    pair_rows = () if pairing is None else pairing.rows
    pair_index = _pair_index(pair_rows, video_id, spans)
    if pairing is not None and len(pair_index) != len(spans):
        raise ValueError("processed canonical pairing must contain one row per rally")
    if not commentary_eligible and any(
        row["chunk_id"] not in (None, "")
        for row in pair_index.values()
    ):
        raise ValueError("commentary-ineligible source cannot contain a paired chunk")
    chunks_by_id = _chunk_index(chunks)
    payloads: list[dict[str, object]] = []
    for rally_id in range(len(spans)):
        pair = pair_index.get(rally_id)
        payloads.append(_commentary_payload(
            rally_id=rally_id,
            pair=pair,
            chunks_by_id=chunks_by_id,
            outcome=outcome,
            stage_reason=stage_reason,
            missing_reason=reason_map.get(rally_id),
            provenance=normalized_provenance,
        ))
    extra_reasons = set(reason_map) - {
        rally_id
        for rally_id, payload in enumerate(payloads)
        if payload["chunk_id"] is None
    }
    if extra_reasons:
        raise ValueError(f"paired rallies cannot have missing commentary reasons: {extra_reasons}")
    return payloads


def _pair_index(
    rows: Sequence[Mapping[str, object]],
    video_id: str,
    spans: Sequence[tuple[int, int]],
) -> dict[int, Mapping[str, object]]:
    index: dict[int, Mapping[str, object]] = {}
    for raw_row in rows:
        row = _mapping(raw_row, "commentary pair row")
        required = {
            "video_id", "rally_id", "rally_start", "rally_end",
            "chunk_id", "commentary_start", "commentary_end",
        }
        if not required.issubset(row):
            raise ValueError("commentary pair row is missing required fields")
        if row["video_id"] != video_id:
            raise ValueError("commentary pair video_id differs from the exact record video_id")
        rally_id = _integer_value(row["rally_id"], "pair rally_id")
        if not 0 <= rally_id < len(spans):
            raise ValueError(f"pair rally_id is outside detected rallies: {rally_id}")
        if rally_id in index:
            raise ValueError(f"commentary pair composite key is duplicated: {(video_id, rally_id)}")
        start = _integer_value(row["rally_start"], "pair rally_start")
        end = _integer_value(row["rally_end"], "pair rally_end")
        if (start, end) != spans[rally_id]:
            raise ValueError("commentary pair span differs from the annotator span")
        index[rally_id] = row
    return index


def _chunk_index(chunks: Sequence[Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
    index: dict[str, Mapping[str, object]] = {}
    for raw_chunk in chunks:
        chunk = _mapping(raw_chunk, "commentary chunk")
        chunk_id = _string(chunk.get("chunk_id"), "commentary chunk_id")
        if chunk_id in index:
            raise ValueError(f"commentary chunk_id is duplicated: {chunk_id!r}")
        index[chunk_id] = chunk
    return index


def _commentary_payload(
    *,
    rally_id: int,
    pair: Mapping[str, object] | None,
    chunks_by_id: Mapping[str, Mapping[str, object]],
    outcome: StageOutcome,
    stage_reason: str | None,
    missing_reason: str | None,
    provenance: Mapping[str, object],
) -> dict[str, object]:
    chunk_id_value = None if pair is None else pair["chunk_id"]
    if chunk_id_value in (None, ""):
        if pair is not None and any(
            pair[name] not in (None, "")
            for name in ("commentary_start", "commentary_end")
        ):
            raise ValueError("unpaired commentary row must have blank commentary times")
        reason = missing_reason or stage_reason
        if reason is None:
            raise ValueError(f"rally {rally_id} has no commentary and no missing reason")
        return {
            "stage_outcome": outcome.value,
            "stage_reason": stage_reason,
            "missing_reason": reason,
            "chunk_id": None,
            "start_seconds": None,
            "end_seconds": None,
            "raw_text": None,
            "cleaned_text": None,
            "alternatives": None,
            "cleaning_diagnostics": {"bert_f1": None, "clean_pass": None},
            "provenance": provenance,
        }
    chunk_id = _string(chunk_id_value, "paired chunk_id")
    if outcome in {StageOutcome.EXCLUDED, StageOutcome.FAILED, StageOutcome.UNAVAILABLE}:
        raise ValueError(f"commentary outcome {outcome.value!r} cannot contain a paired chunk")
    if chunk_id not in chunks_by_id:
        raise ValueError(f"paired chunk {chunk_id!r} is missing from the selected sidecar")
    chunk = chunks_by_id[chunk_id]
    start = _finite_number(pair["commentary_start"], "pair commentary_start")
    end = _finite_number(pair["commentary_end"], "pair commentary_end")
    chunk_start = _finite_number(chunk.get("start"), "chunk start")
    chunk_end = _finite_number(chunk.get("end"), "chunk end")
    if start != chunk_start or end != chunk_end or not 0 <= start <= end:
        raise ValueError("paired commentary times differ from the selected chunk sidecar")
    raw_text = _text(chunk.get("text"), "chunk raw text")
    cleaned_text = _optional_text(chunk.get("text_clean"), "chunk cleaned text")
    alternatives = _alternatives(chunk.get("alt_phrasings"))
    bert_f1 = _optional_number(chunk.get("bert_f1"), "chunk bert_f1")
    clean_pass = chunk.get("clean_pass")
    if clean_pass is not None and not isinstance(clean_pass, bool):
        raise ValueError("chunk clean_pass must be boolean or null")
    validate_paired_commentary_provenance(provenance)
    return {
        "stage_outcome": outcome.value,
        "stage_reason": stage_reason,
        "missing_reason": None,
        "chunk_id": chunk_id,
        "start_seconds": start,
        "end_seconds": end,
        "raw_text": raw_text,
        "cleaned_text": cleaned_text,
        "alternatives": alternatives,
        "cleaning_diagnostics": {"bert_f1": bert_f1, "clean_pass": clean_pass},
        "provenance": provenance,
    }


def _missing_reason_map(reasons: Mapping[int, str], rally_count: int) -> dict[int, str]:
    normalized: dict[int, str] = {}
    for rally_id, reason in reasons.items():
        if isinstance(rally_id, bool) or not isinstance(rally_id, int) or not 0 <= rally_id < rally_count:
            raise ValueError(f"commentary missing-reason rally_id is invalid: {rally_id!r}")
        normalized[rally_id] = _string(reason, f"commentary missing reason {rally_id}")
    return normalized


def _mapping(payload: object, name: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping) or any(not isinstance(key, str) for key in payload):
        raise ValueError(f"{name} must be an object with string keys")
    return payload


def _integer_value(payload: object, name: str) -> int:
    if isinstance(payload, bool) or not isinstance(payload, int):
        raise TypeError(f"{name} must be an integer")
    return payload


def _finite_number(payload: object, name: str) -> float:
    if isinstance(payload, bool):
        raise TypeError(f"{name} must be a finite number")
    try:
        value = float(payload)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _optional_number(payload: object, name: str) -> float | None:
    return None if payload is None else _finite_number(payload, name)


def _string(payload: object, name: str) -> str:
    if not isinstance(payload, str) or not payload:
        raise ValueError(f"{name} must be a non-empty string")
    return payload


def _text(payload: object, name: str) -> str:
    if not isinstance(payload, str):
        raise TypeError(f"{name} must be a string")
    return payload


def _optional_text(payload: object, name: str) -> str | None:
    return None if payload is None else _text(payload, name)


def _alternatives(payload: object) -> list[str] | None:
    if payload is None:
        return None
    if not isinstance(payload, list) or any(not isinstance(item, str) for item in payload):
        raise ValueError("chunk alt_phrasings must be a list of strings or null")
    return list(payload)


def _enum_value(payload: object) -> str | None:
    return None if payload is None else str(payload.value)
