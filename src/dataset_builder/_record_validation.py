"""Deep structural validation for persisted ``rally-record/0.1`` rows."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from fractions import Fraction
import math
from pathlib import Path
import re

from annotator.video_metadata import VideoMetadata
from dataset_builder.manifest import MANIFEST_FILENAME
from dataset_builder.models import ArtifactIntegrity, RunManifest, StageOutcome


RALLY_RECORD_SCHEMA = "rally-record/0.1"
RALLY_RECORDS_FILENAME = "rally_records.json.gz"
RAW_REPLAY_ARTIFACT = "raw_replay_mask"
DEFINITIVE_EXCLUSION_ARTIFACT = "definitive_exclusion_mask"

_HALVES = frozenset({"Top", "Bot"})
_VERDICTS = frozenset({"won", "lost"})
_VERDICT_SOURCES = frozenset({"next_server", "landing_geometry", "net_rule"})
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def validate_record_collection(
    run_id: str,
    records: Sequence[Mapping[str, object]],
) -> None:
    """Validate every required persisted field and composite record key."""
    collection_run_id = _string(run_id, "record collection run_id")
    seen: set[tuple[str, str, str, int]] = set()
    rally_ids: defaultdict[tuple[str, str], set[int]] = defaultdict(set)
    rally_spans: defaultdict[tuple[str, str], dict[int, tuple[int, int]]] = defaultdict(dict)
    outcome_links: defaultdict[
        tuple[str, str],
        dict[int, tuple[str | None, str | None]],
    ] = defaultdict(dict)
    run_snapshot: Mapping[str, object] | None = None
    source_snapshots: dict[tuple[str, str], Mapping[str, object]] = {}
    for raw_record in records:
        record = _object(raw_record, "rally record")
        _exact_fields(
            record,
            {
                "schema", "key", "run", "source", "rally", "contacts",
                "outcomes", "artifacts", "commentary",
            },
            "rally record",
        )
        if record["schema"] != RALLY_RECORD_SCHEMA:
            raise ValueError("rally record schema differs from rally-record/0.1")
        key = _record_key(record["key"], collection_run_id)
        if key in seen:
            raise ValueError(f"rally record composite key is duplicated: {key}")
        seen.add(key)
        _, source_dataset, video_id, rally_id = key
        source_key = (source_dataset, video_id)
        rally_ids[source_key].add(rally_id)
        run = _run(record["run"], collection_run_id)
        raw_run = _object(record["run"], "record run provenance")
        if run_snapshot is None:
            run_snapshot = raw_run
        elif raw_run != run_snapshot:
            raise ValueError("record run provenance differs within its collection")
        metadata, commentary_eligible = _source(
            record["source"], source_dataset, video_id,
        )
        raw_source = _object(record["source"], "record source provenance")
        if source_key in source_snapshots and raw_source != source_snapshots[source_key]:
            raise ValueError("record source provenance differs within one source video")
        source_snapshots[source_key] = raw_source
        rally = _rally(record["rally"], rally_id, metadata)
        rally_spans[source_key][rally_id] = rally
        stroke_count = _contacts(record["contacts"], rally)
        outcome_links[source_key][rally_id] = _outcomes(
            record["outcomes"], metadata.frame_count, stroke_count,
        )
        _record_artifacts(record["artifacts"], run)
        _commentary(record["commentary"], commentary_eligible)
    for source_key, ids in rally_ids.items():
        if ids != set(range(len(ids))):
            raise ValueError(f"rally ids must be contiguous from zero for source {source_key}")
        ordered_spans = [rally_spans[source_key][rally_id] for rally_id in range(len(ids))]
        for previous, current in zip(ordered_spans, ordered_spans[1:]):
            if current[0] < previous[1]:
                raise ValueError(f"rally spans overlap or are unordered for source {source_key}")
        for rally_id in range(len(ids)):
            expected_next = (
                outcome_links[source_key][rally_id + 1][0]
                if rally_id + 1 < len(ids)
                else None
            )
            if outcome_links[source_key][rally_id][1] != expected_next:
                raise ValueError(f"next_server conflicts with the following rally for {source_key}")


def validate_paired_commentary_provenance(payload: Mapping[str, object]) -> None:
    """Require the transcript and cleaning provenance promised by the schema."""
    provenance = _object(payload, "paired commentary provenance")
    for name in ("transcript", "cleaning"):
        if name not in provenance:
            raise ValueError(f"paired commentary provenance is missing {name}")
        section = _object(provenance[name], f"commentary provenance {name}")
        if not {"method", "configuration"}.issubset(section):
            raise ValueError(
                f"commentary provenance {name} requires method and configuration"
            )
        _string(section["method"], f"commentary provenance {name} method")
        _json_object(
            section["configuration"],
            f"commentary provenance {name} configuration",
        )


def validate_live_manifest_extension(
    input_manifest: RunManifest,
    live_manifest: RunManifest,
) -> None:
    """Require the live manifest to preserve and extend the immutable input snapshot."""
    same_identity = (
        live_manifest.schema == input_manifest.schema
        and live_manifest.run_id == input_manifest.run_id
        and live_manifest.created_at_utc == input_manifest.created_at_utc
    )
    prefix = live_manifest.stages[:len(input_manifest.stages)]
    if (
        not same_identity
        or len(live_manifest.stages) < len(input_manifest.stages)
        or prefix != input_manifest.stages
    ):
        raise ValueError("live run manifest does not extend the record input-manifest snapshot")


def reject_manifest_output_cycle(
    manifest: RunManifest,
    run_dir: Path,
    destination: Path,
) -> None:
    """Reject manifest artifacts that resolve to the record output being rebuilt."""
    root = Path(run_dir).resolve(strict=False)
    target = Path(destination).resolve(strict=False)
    for stage in manifest.stages:
        artifacts = (
            *stage.fingerprint.inputs,
            *stage.fingerprint.model_weights,
            *stage.outputs,
        )
        for artifact in artifacts:
            stored = Path(artifact.path)
            candidate = stored if stored.is_absolute() else root / stored
            if candidate.resolve(strict=False) == target:
                raise ValueError(
                    "input manifest already references the rally-record output"
                )


def _record_key(payload: object, run_id: str) -> tuple[str, str, str, int]:
    key = _object(payload, "rally record key")
    _exact_fields(key, {"run_id", "source_dataset", "video_id", "rally_id"}, "record key")
    key_run_id = _string(key["run_id"], "record key run_id")
    source_dataset = _string(key["source_dataset"], "record key source_dataset")
    video_id = _string(key["video_id"], "record key video_id")
    rally_id = _integer(key["rally_id"], "record key rally_id")
    if key_run_id != run_id or rally_id < 0:
        raise ValueError("rally record key conflicts with its collection")
    return key_run_id, source_dataset, video_id, rally_id


def _run(payload: object, run_id: str) -> dict[str, object]:
    run = _object(payload, "record run provenance")
    _exact_fields(
        run,
        {
            "run_id", "created_at_utc", "run_manifest", "input_manifest_sha256", "code_version",
            "configuration", "integrity", "stage_outcomes",
        },
        "record run provenance",
    )
    if _string(run["run_id"], "record run_id") != run_id:
        raise ValueError("record run provenance differs from its composite key")
    _string(run["created_at_utc"], "record created_at_utc")
    if run["run_manifest"] != MANIFEST_FILENAME:
        raise ValueError("record run provenance must reference run_manifest.json.gz")
    manifest_digest = _string(
        run["input_manifest_sha256"], "record input_manifest_sha256",
    )
    if not _SHA256_PATTERN.fullmatch(manifest_digest):
        raise ValueError("record input_manifest_sha256 must be lowercase SHA-256")
    _string(run["code_version"], "record code_version")
    configuration = _run_configuration(run["configuration"])
    integrity = _run_integrity(run["integrity"])
    outcomes = _stage_outcomes(run["stage_outcomes"])
    if set(configuration["stages"]) != set(integrity) or set(integrity) != set(outcomes):
        raise ValueError("record run stage configuration, integrity, and outcomes differ")
    return {
        "configuration": configuration,
        "integrity": integrity,
        "stage_outcomes": outcomes,
    }


def _run_configuration(payload: object) -> dict[str, Mapping[str, object]]:
    configuration = _object(payload, "record run configuration")
    _exact_fields(configuration, {"assembly", "stages"}, "record run configuration")
    assembly = _json_object(configuration["assembly"], "assembly configuration")
    stages = _object(configuration["stages"], "stage configurations")
    normalized_stages: dict[str, Mapping[str, object]] = {}
    for stage_name, raw_stage in stages.items():
        name = _string(stage_name, "stage configuration name")
        normalized_stages[name] = _json_object(raw_stage, f"stage {name} configuration")
    return {"assembly": assembly, "stages": normalized_stages}


def _run_integrity(payload: object) -> dict[str, dict[str, object]]:
    stages = _object(payload, "record run integrity")
    normalized: dict[str, dict[str, object]] = {}
    for stage_name, raw_stage in stages.items():
        name = _string(stage_name, "stage integrity name")
        stage = _object(raw_stage, f"stage {name} integrity")
        _exact_fields(
            stage,
            {"fingerprint", "inputs", "model_weights", "outputs"},
            f"stage {name} integrity",
        )
        fingerprint = _string(stage["fingerprint"], f"stage {name} fingerprint")
        if not _SHA256_PATTERN.fullmatch(fingerprint):
            raise ValueError(f"stage {name} fingerprint must be lowercase SHA-256")
        normalized[name] = {
            "fingerprint": fingerprint,
            "inputs": _artifact_list(stage["inputs"], f"stage {name} inputs"),
            "model_weights": _artifact_list(
                stage["model_weights"], f"stage {name} model weights",
            ),
            "outputs": _artifact_list(stage["outputs"], f"stage {name} outputs"),
        }
    return normalized


def _stage_outcomes(payload: object) -> dict[str, StageOutcome]:
    stages = _object(payload, "record stage outcomes")
    normalized: dict[str, StageOutcome] = {}
    for stage_name, raw_stage in stages.items():
        name = _string(stage_name, "stage outcome name")
        stage = _object(raw_stage, f"stage {name} outcome")
        _exact_fields(stage, {"outcome", "reason"}, f"stage {name} outcome")
        outcome = _stage_outcome(stage["outcome"], f"stage {name} outcome")
        reason = _optional_string(stage["reason"], f"stage {name} reason")
        if outcome is not StageOutcome.PROCESSED and reason is None:
            raise ValueError(f"stage {name} outcome {outcome.value!r} requires a reason")
        normalized[name] = outcome
    return normalized


def _source(
    payload: object,
    source_dataset: str,
    video_id: str,
) -> tuple[VideoMetadata, bool]:
    source = _object(payload, "record source provenance")
    _exact_fields(
        source,
        {"source_dataset", "video_id", "source_reference", "video_metadata"},
        "record source provenance",
    )
    if source["source_dataset"] != source_dataset or source["video_id"] != video_id:
        raise ValueError("record source provenance differs from its composite key")
    reference = _object(source["source_reference"], "source reference")
    _exact_fields(
        reference,
        {"video_id", "basename", "title", "url", "commentary_eligible"},
        "source reference",
    )
    if _string(reference["video_id"], "source reference video_id") != video_id:
        raise ValueError("source reference video_id differs from its composite key")
    basename = _string(reference["basename"], "source reference basename")
    _string(reference["title"], "source reference title")
    _string(reference["url"], "source reference url")
    commentary_eligible = _boolean(
        reference["commentary_eligible"], "source commentary eligibility",
    )
    metadata = VideoMetadata.from_dict(dict(_object(source["video_metadata"], "video metadata")))
    if metadata.source_path.name != basename or metadata.source_path.stem != video_id:
        raise ValueError("source reference conflicts with canonical video metadata")
    return metadata, commentary_eligible


def _rally(payload: object, rally_id: int, metadata: VideoMetadata) -> tuple[int, int]:
    rally = _object(payload, "rally fields")
    _exact_fields(
        rally,
        {
            "rally_id", "start_frame", "end_frame", "duration_frames",
            "duration_seconds",
        },
        "rally fields",
    )
    if _integer(rally["rally_id"], "rally id") != rally_id:
        raise ValueError("rally payload id differs from its composite key")
    start = _integer(rally["start_frame"], "rally start_frame")
    end = _integer(rally["end_frame"], "rally end_frame")
    if not 0 <= start < end <= metadata.frame_count:
        raise ValueError("persisted rally span is outside canonical frame bounds")
    duration_frames = _integer(rally["duration_frames"], "rally duration_frames")
    if duration_frames != end - start:
        raise ValueError("rally duration_frames differs from its half-open span")
    duration_seconds = _number(rally["duration_seconds"], "rally duration_seconds")
    expected_seconds = float(Fraction(duration_frames, 1) / metadata.fps)
    if duration_seconds != expected_seconds:
        raise ValueError("rally duration_seconds conflicts with canonical fps")
    return start, end


def _contacts(payload: object, rally: tuple[int, int]) -> int:
    contacts = _object(payload, "record contacts")
    _exact_fields(
        contacts,
        {"raw_candidates", "accepted", "stroke_count", "hit_height_failures"},
        "record contacts",
    )
    raw_frames = _raw_contacts(contacts["raw_candidates"], rally)
    accepted = _accepted_contacts(contacts["accepted"], rally, raw_frames)
    stroke_count = _integer(contacts["stroke_count"], "record stroke_count")
    if stroke_count != len(accepted):
        raise ValueError("record stroke_count differs from accepted contacts")
    failed = _hit_height_failures(contacts["hit_height_failures"], accepted)
    for stroke_idx, row in enumerate(accepted):
        has_height = row["hit_height_code"] is not None
        if has_height == (stroke_idx in failed):
            raise ValueError("accepted contact must have exactly one hit height or failure")
    return stroke_count


def _raw_contacts(payload: object, rally: tuple[int, int]) -> set[int]:
    rows = _list(payload, "raw contact candidates")
    frames: list[int] = []
    for raw_row in rows:
        row = _object(raw_row, "raw contact candidate")
        _exact_fields(
            row,
            {"contact_frame", "proximity_ok", "wrist_near", "suppressed"},
            "raw contact candidate",
        )
        frame = _frame_in_rally(row["contact_frame"], rally, "raw contact frame")
        frames.append(frame)
        for name in ("proximity_ok", "wrist_near", "suppressed"):
            _optional_boolean(row[name], f"raw contact {name}")
    if frames != sorted(frames) or len(frames) != len(set(frames)):
        raise ValueError("raw contact frames must be unique and ascending")
    return set(frames)


def _accepted_contacts(
    payload: object,
    rally: tuple[int, int],
    raw_frames: set[int],
) -> list[Mapping[str, object]]:
    rows = _list(payload, "accepted contacts")
    normalized: list[Mapping[str, object]] = []
    frames: list[int] = []
    for expected_stroke_idx, raw_row in enumerate(rows):
        row = _object(raw_row, "accepted contact")
        _exact_fields(
            row, {"stroke_idx", "contact_frame", "hit_height_code"}, "accepted contact",
        )
        if _integer(row["stroke_idx"], "accepted stroke_idx") != expected_stroke_idx:
            raise ValueError("accepted stroke_idx must be contiguous from zero")
        frame = _frame_in_rally(row["contact_frame"], rally, "accepted contact frame")
        if frame not in raw_frames:
            raise ValueError("accepted contact has no matching raw candidate")
        code = row["hit_height_code"]
        if code is not None and _integer(code, "accepted hit_height_code") not in (1, 2):
            raise ValueError("accepted hit_height_code must be 1, 2, or null")
        frames.append(frame)
        normalized.append(row)
    if frames != sorted(frames) or len(frames) != len(set(frames)):
        raise ValueError("accepted contact frames must be unique and ascending")
    return normalized


def _hit_height_failures(
    payload: object,
    accepted: Sequence[Mapping[str, object]],
) -> set[int]:
    rows = _list(payload, "hit-height failures")
    failed: set[int] = set()
    for raw_row in rows:
        row = _object(raw_row, "hit-height failure")
        _exact_fields(
            row, {"stroke_idx", "contact_frame", "reason"}, "hit-height failure",
        )
        stroke_idx = _integer(row["stroke_idx"], "hit-height failure stroke_idx")
        if not 0 <= stroke_idx < len(accepted) or stroke_idx in failed:
            raise ValueError("hit-height failure stroke_idx is invalid or duplicated")
        frame = _integer(row["contact_frame"], "hit-height failure contact_frame")
        if frame != accepted[stroke_idx]["contact_frame"]:
            raise ValueError("hit-height failure frame differs from accepted contact")
        _string(row["reason"], "hit-height failure reason")
        failed.add(stroke_idx)
    return failed


def _outcomes(
    payload: object,
    frame_count: int,
    stroke_count: int,
) -> tuple[str | None, str | None]:
    outcomes = _object(payload, "record outcomes")
    _exact_fields(
        outcomes,
        {"striker_half", "server_prediction", "next_server", "verdict", "landing", "geometric_verdict"},
        "record outcomes",
    )
    striker = _optional_enum(outcomes["striker_half"], _HALVES, "striker_half")
    server = _optional_enum(outcomes["server_prediction"], _HALVES, "server_prediction")
    next_server = _optional_enum(outcomes["next_server"], _HALVES, "next_server")
    if striker is None and server is not None:
        raise ValueError("server prediction for an unresolved striker must be null")
    if striker is not None:
        if stroke_count <= 0:
            raise ValueError("resolved striker must have an accepted contact")
        expected = striker if (stroke_count - 1) % 2 == 0 else _other_half(striker)
        if server != expected:
            raise ValueError("server prediction conflicts with striker parity")
    _verdict(outcomes["verdict"], resolved=striker is not None)
    _landing(outcomes["landing"], frame_count, resolved=striker is not None)
    _geometric_verdict(outcomes["geometric_verdict"], resolved=striker is not None)
    return server, next_server


def _verdict(payload: object, *, resolved: bool) -> None:
    verdict = _object(payload, "record verdict")
    _exact_fields(
        verdict,
        {"value", "source", "landing_margin_m", "within_line_margin", "within_net_margin"},
        "record verdict",
    )
    _optional_enum(verdict["value"], _VERDICTS, "verdict value")
    _optional_enum(verdict["source"], _VERDICT_SOURCES, "verdict source")
    _optional_number(verdict["landing_margin_m"], "verdict landing_margin_m")
    line_flag = _optional_boolean(verdict["within_line_margin"], "verdict line-margin flag")
    net_flag = _optional_boolean(verdict["within_net_margin"], "verdict net-margin flag")
    if resolved and (line_flag is None or net_flag is None):
        raise ValueError("resolved verdict margin flags must be boolean")
    if not resolved and any(value is not None for value in verdict.values()):
        raise ValueError("unresolved striker cannot contain verdict primitives")


def _landing(payload: object, frame_count: int, *, resolved: bool) -> None:
    if payload is None:
        return
    if not resolved:
        raise ValueError("unresolved striker cannot contain a landing")
    landing = _object(payload, "record landing")
    _exact_fields(
        landing,
        {"frame", "normalized_court_position", "court_half", "at_image_border", "net_ender"},
        "record landing",
    )
    frame = _integer(landing["frame"], "landing frame")
    if not 0 <= frame < frame_count:
        raise ValueError("landing frame is outside canonical frame bounds")
    position = _list(landing["normalized_court_position"], "landing court position")
    if len(position) != 2:
        raise ValueError("landing court position must contain two coordinates")
    for coordinate in position:
        _number(coordinate, "landing court coordinate")
    _optional_enum(landing["court_half"], _HALVES, "landing court_half", nullable=False)
    _boolean(landing["at_image_border"], "landing image-border flag")
    _boolean(landing["net_ender"], "landing net-ender flag")


def _geometric_verdict(payload: object, *, resolved: bool) -> None:
    geometric = _object(payload, "record geometric verdict")
    _exact_fields(
        geometric,
        {"value", "winner", "agreement", "window_closed_by_mask"},
        "record geometric verdict",
    )
    _optional_enum(geometric["value"], _VERDICTS, "geometric verdict value")
    _optional_enum(geometric["winner"], _HALVES, "geometric verdict winner")
    _optional_boolean(geometric["agreement"], "geometric verdict agreement")
    mask_flag = _optional_boolean(
        geometric["window_closed_by_mask"], "geometric verdict mask flag",
    )
    if resolved and mask_flag is None:
        raise ValueError("resolved geometric verdict mask flag must be boolean")
    if not resolved and any(value is not None for value in geometric.values()):
        raise ValueError("unresolved striker cannot contain geometric verdict primitives")


def _record_artifacts(payload: object, run: Mapping[str, object]) -> None:
    artifacts = _object(payload, "record artifacts")
    _exact_fields(artifacts, {"by_stage", "masks"}, "record artifacts")
    integrity = _object(run["integrity"], "record run integrity")
    by_stage = _object(artifacts["by_stage"], "record artifacts by stage")
    expected_by_stage = {
        stage_name: stage["outputs"]
        for stage_name, stage in integrity.items()
        if stage["outputs"]
    }
    normalized_by_stage = {
        _string(stage_name, "artifact stage name"): _artifact_list(
            rows, f"stage {stage_name} record artifacts",
        )
        for stage_name, rows in by_stage.items()
    }
    if normalized_by_stage != expected_by_stage:
        raise ValueError("record stage artifact references differ from run integrity")
    _mask_artifacts(artifacts["masks"], run)


def _mask_artifacts(payload: object, run: Mapping[str, object]) -> None:
    masks = _object(payload, "record mask artifacts")
    _exact_fields(
        masks,
        {"stage", "stage_configuration", RAW_REPLAY_ARTIFACT, DEFINITIVE_EXCLUSION_ARTIFACT},
        "record mask artifacts",
    )
    stage_name = _string(masks["stage"], "mask stage name")
    configuration = _object(run["configuration"], "record run configuration")
    stage_configurations = _object(configuration["stages"], "stage configurations")
    if stage_name not in stage_configurations:
        raise ValueError("mask stage is absent from record run configuration")
    mask_configuration = _json_object(masks["stage_configuration"], "mask stage configuration")
    if mask_configuration != stage_configurations[stage_name]:
        raise ValueError("mask stage configuration differs from run configuration")
    integrity = _object(run["integrity"], "record run integrity")
    outputs = _object(integrity[stage_name], "mask stage integrity")["outputs"]
    stage_outcomes = _object(run["stage_outcomes"], "record stage outcomes")
    if stage_outcomes.get(stage_name) not in {StageOutcome.PROCESSED, StageOutcome.SKIPPED}:
        raise ValueError("record mask stage must have a reusable successful outcome")
    outputs_by_name = {artifact["name"]: artifact for artifact in outputs}
    for artifact_name in (RAW_REPLAY_ARTIFACT, DEFINITIVE_EXCLUSION_ARTIFACT):
        artifact = ArtifactIntegrity.from_dict(masks[artifact_name]).to_dict()
        if artifact["name"] != artifact_name or outputs_by_name.get(artifact_name) != artifact:
            raise ValueError(f"record mask artifact {artifact_name!r} differs from run integrity")


def _commentary(payload: object, commentary_eligible: bool) -> None:
    commentary = _object(payload, "record commentary")
    _exact_fields(
        commentary,
        {
            "stage_outcome", "stage_reason", "missing_reason", "chunk_id",
            "start_seconds", "end_seconds", "raw_text", "cleaned_text",
            "alternatives", "cleaning_diagnostics", "provenance",
        },
        "record commentary",
    )
    outcome = _stage_outcome(commentary["stage_outcome"], "commentary stage_outcome")
    stage_reason = _optional_string(commentary["stage_reason"], "commentary stage_reason")
    if outcome is not StageOutcome.PROCESSED and stage_reason is None:
        raise ValueError(f"commentary outcome {outcome.value!r} requires a reason")
    chunk_id = _optional_string(commentary["chunk_id"], "commentary chunk_id")
    diagnostics = _commentary_diagnostics(commentary["cleaning_diagnostics"])
    provenance = _json_object(commentary["provenance"], "commentary provenance")
    if chunk_id is None:
        _missing_commentary(commentary, diagnostics)
        return
    if not commentary_eligible:
        raise ValueError("commentary-ineligible source cannot contain a paired chunk")
    if outcome in {StageOutcome.EXCLUDED, StageOutcome.FAILED, StageOutcome.UNAVAILABLE}:
        raise ValueError(f"commentary outcome {outcome.value!r} cannot contain a paired chunk")
    if commentary["missing_reason"] is not None:
        raise ValueError("paired commentary cannot contain a missing reason")
    start = _number(commentary["start_seconds"], "commentary start_seconds")
    end = _number(commentary["end_seconds"], "commentary end_seconds")
    if not 0 <= start <= end:
        raise ValueError("paired commentary times are invalid")
    _text(commentary["raw_text"], "commentary raw_text")
    _optional_text(commentary["cleaned_text"], "commentary cleaned_text")
    _alternatives(commentary["alternatives"])
    validate_paired_commentary_provenance(provenance)


def _missing_commentary(
    commentary: Mapping[str, object],
    diagnostics: tuple[float | None, bool | None],
) -> None:
    _string(commentary["missing_reason"], "commentary missing_reason")
    required_null = (
        "start_seconds", "end_seconds", "raw_text", "cleaned_text", "alternatives",
    )
    if any(commentary[name] is not None for name in required_null) or diagnostics != (None, None):
        raise ValueError("missing commentary evidence fields must be null")


def _commentary_diagnostics(payload: object) -> tuple[float | None, bool | None]:
    diagnostics = _object(payload, "commentary cleaning diagnostics")
    _exact_fields(diagnostics, {"bert_f1", "clean_pass"}, "commentary cleaning diagnostics")
    return (
        _optional_number(diagnostics["bert_f1"], "commentary bert_f1"),
        _optional_boolean(diagnostics["clean_pass"], "commentary clean_pass"),
    )


def _artifact_list(payload: object, name: str) -> list[dict[str, object]]:
    rows = _list(payload, name)
    artifacts = [ArtifactIntegrity.from_dict(row).to_dict() for row in rows]
    names = [artifact["name"] for artifact in artifacts]
    if len(names) != len(set(names)):
        raise ValueError(f"{name} contains duplicate artifact names")
    return artifacts


def _json_object(payload: object, name: str) -> Mapping[str, object]:
    value = _object(payload, name)
    _json_value(value, name)
    return value


def _json_value(payload: object, name: str) -> None:
    if payload is None or isinstance(payload, (bool, int, str)):
        return
    if isinstance(payload, float):
        if not math.isfinite(payload):
            raise ValueError(f"{name} contains a non-finite number")
        return
    if isinstance(payload, list):
        for index, item in enumerate(payload):
            _json_value(item, f"{name}[{index}]")
        return
    if isinstance(payload, Mapping) and all(isinstance(key, str) for key in payload):
        for key, item in payload.items():
            _json_value(item, f"{name}.{key}")
        return
    raise ValueError(f"{name} must contain only JSON-compatible values")


def _object(payload: object, name: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping) or any(not isinstance(key, str) for key in payload):
        raise ValueError(f"{name} must be an object with string keys")
    return payload


def _list(payload: object, name: str) -> list[object]:
    if not isinstance(payload, list):
        raise ValueError(f"{name} must be a list")
    return payload


def _exact_fields(payload: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(payload) != expected:
        raise ValueError(f"{name} fields differ")


def _frame_in_rally(payload: object, rally: tuple[int, int], name: str) -> int:
    frame = _integer(payload, name)
    if not rally[0] <= frame < rally[1]:
        raise ValueError(f"{name} is outside its half-open rally span")
    return frame


def _integer(payload: object, name: str) -> int:
    if isinstance(payload, bool) or not isinstance(payload, int):
        raise ValueError(f"{name} must be an integer")
    return payload


def _number(payload: object, name: str) -> float:
    if isinstance(payload, bool) or not isinstance(payload, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(payload)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _optional_number(payload: object, name: str) -> float | None:
    return None if payload is None else _number(payload, name)


def _string(payload: object, name: str) -> str:
    if not isinstance(payload, str) or not payload:
        raise ValueError(f"{name} must be a non-empty string")
    return payload


def _optional_string(payload: object, name: str) -> str | None:
    return None if payload is None else _string(payload, name)


def _text(payload: object, name: str) -> str:
    if not isinstance(payload, str):
        raise ValueError(f"{name} must be a string")
    return payload


def _optional_text(payload: object, name: str) -> str | None:
    return None if payload is None else _text(payload, name)


def _boolean(payload: object, name: str) -> bool:
    if not isinstance(payload, bool):
        raise ValueError(f"{name} must be boolean")
    return payload


def _optional_boolean(payload: object, name: str) -> bool | None:
    return None if payload is None else _boolean(payload, name)


def _optional_enum(
    payload: object,
    choices: frozenset[str],
    name: str,
    *,
    nullable: bool = True,
) -> str | None:
    if payload is None and nullable:
        return None
    if not isinstance(payload, str) or payload not in choices:
        raise ValueError(f"{name} has an unsupported value")
    return payload


def _other_half(half: str) -> str:
    return "Bot" if half == "Top" else "Top"


def _stage_outcome(payload: object, name: str) -> StageOutcome:
    value = _string(payload, name)
    try:
        return StageOutcome(value)
    except ValueError as error:
        raise ValueError(f"{name} has an unsupported value") from error


def _alternatives(payload: object) -> list[str] | None:
    if payload is None:
        return None
    values = _list(payload, "commentary alternatives")
    if any(not isinstance(value, str) for value in values):
        raise ValueError("commentary alternatives must be strings")
    return values
