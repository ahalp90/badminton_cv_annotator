"""Primitive rally-record projection, validation, and persistence contracts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from annotator.point_winner import (
    GeometricVerdictRow,
    Half,
    Landing,
    Verdict,
    VerdictRow,
    VerdictSource,
)
from annotator.run_video import AnnotatorResult
from annotator.types import ContactCandidate
from annotator.video_metadata import VideoMetadata
from dataset_builder.manifest import artifact_integrity, run_manifest_sha256, write_run_manifest
from dataset_builder.models import (
    ArtifactIntegrity,
    InterpreterIdentity,
    RunManifest,
    SemanticValidation,
    StageFingerprint,
    StageOutcome,
    StageRecord,
)
from dataset_builder.records import (
    RALLY_RECORD_COLLECTION_SCHEMA,
    RALLY_RECORD_PROJECTION_SCHEMA,
    RALLY_RECORD_SCHEMA,
    RALLY_RECORDS_FILENAME,
    RallyRecordArtifacts,
    RallyRecordProjection,
    SourceReference,
    assemble_rally_records,
    load_rally_record_projection,
    load_rally_records,
    write_rally_record_projection,
    write_rally_records,
)
from dataset_builder.vision import load_json_gz, save_json_gz
from scraper.commentary_pairing import CanonicalPairing, pair_video_with_metadata


CODE_VERSION = "a" * 40
FPS = Fraction(25, 1)
VIDEO_ID = "0012"


def _artifact(name: str, path: str, marker: str) -> ArtifactIntegrity:
    return ArtifactIntegrity(name=name, path=path, md5=marker * 32, size_bytes=10)


def _stage(
    name: str,
    *,
    configuration: dict[str, object],
    outputs: tuple[ArtifactIntegrity, ...],
    dependencies: tuple[str, ...] = (),
    marker: str,
) -> StageRecord:
    fingerprint = StageFingerprint(
        digest=marker * 64,
        source_commit=CODE_VERSION,
        contract_version=f"{name}/0.1",
        configuration_sha256="f" * 64,
        interpreter=InterpreterIdentity("/usr/bin/python", "Python 3.12"),
        model_weights=(),
        inputs=(),
    )
    return StageRecord(
        name=name,
        outcome=StageOutcome.PROCESSED,
        fingerprint=fingerprint,
        dependencies=dependencies,
        command=("python", name),
        configuration=configuration,
        outputs=outputs,
        counts=(),
        elapsed_seconds=1.0,
        semantic_validation=(),
    )


def _manifest() -> RunManifest:
    vision = _stage(
        "vision",
        configuration={"device": "cuda"},
        outputs=(_artifact("track", "vision/shuttle_track.npy.xz", "1"),),
        marker="1",
    )
    annotation = _stage(
        "annotation",
        configuration={"dead_mask_mode": "replay"},
        outputs=(
            _artifact("annotator_result", "annotation/annotator_result.json.gz", "2"),
            _artifact("raw_replay_mask", "annotation/raw_replay_mask.npy.xz", "3"),
            _artifact(
                "definitive_exclusion_mask",
                "annotation/definitive_exclusion_mask.npy.xz",
                "4",
            ),
        ),
        dependencies=("vision",),
        marker="2",
    )
    commentary = _stage(
        "commentary",
        configuration={"window_seconds": 8},
        outputs=(_artifact("pairs", "commentary/pairs.csv.gz", "5"),),
        dependencies=("annotation",),
        marker="3",
    )
    return RunManifest(
        run_id="run-15",
        created_at_utc="2026-08-09T00:00:00Z",
        stages=(vision, annotation, commentary),
    )


def _metadata(
    tmp_path: Path,
    *,
    fps: Fraction = FPS,
    frame_count: int = 100,
) -> VideoMetadata:
    source = (tmp_path / f"{VIDEO_ID}.mp4").resolve()
    source.write_bytes(b"video")
    return VideoMetadata(
        source_path=source,
        fps=fps,
        frame_count=frame_count,
        width=100,
        height=50,
    )


def _annotation() -> AnnotatorResult:
    rejected = ContactCandidate(0, 10, False, False, None)
    first = ContactCandidate(0, 20, True, True, False)
    second = ContactCandidate(0, 30, True, True, False)
    unresolved = ContactCandidate(1, 70, None, None, False)
    return AnnotatorResult(
        spans=[(0, 50), (60, 100)],
        contacts=[rejected, first, second, unresolved],
        filtered_contacts=[first, second, unresolved],
        filtered_by_rally={0: [20, 30], 1: [70]},
        striker_halves=[Half.TOP, None],
        n_strokes_list=[2, 1],
        next_servers=[None, None],
        fitted_first_all=[Half.BOT, None],
        verdict_rows={
            0: VerdictRow(
                0,
                Half.TOP,
                Verdict.LOST,
                VerdictSource.NET_RULE,
                -0.2,
                False,
                True,
            ),
        },
        landings={0: Landing(45, (0.4, 0.8), Half.BOT, False, True)},
        geometric_verdict_rows={
            0: GeometricVerdictRow(0, Verdict.LOST, Half.BOT, True, True),
        },
        hit_height_by_frame={20: 1, 70: 2},
        hit_height_failures=[(0, 1, 30, "shuttle not visible")],
    )


def _pairing(metadata: VideoMetadata) -> CanonicalPairing:
    return CanonicalPairing(
        VIDEO_ID,
        metadata,
        (
            {
                "video_id": VIDEO_ID,
                "rally_id": 0,
                "rally_start": 0,
                "rally_end": 50,
                "chunk_id": "c0",
                "commentary_start": 2.2,
                "commentary_end": 3.0,
            },
            {
                "video_id": VIDEO_ID,
                "rally_id": 1,
                "rally_start": 60,
                "rally_end": 100,
                "chunk_id": "",
                "commentary_start": "",
                "commentary_end": "",
            },
        ),
    )


def _chunks() -> list[dict[str, object]]:
    return [{
        "chunk_id": "c0",
        "start": 2.2,
        "end": 3.0,
        "text": "raw call",
        "text_clean": "clean call",
        "alt_phrasings": ["alternate one", "alternate two"],
        "bert_f1": 0.9,
        "clean_pass": True,
    }]


def _source_reference() -> SourceReference:
    return SourceReference(
        video_id=VIDEO_ID,
        basename=f"{VIDEO_ID}.mp4",
        title="Professional singles final",
        url="https://example.test/watch?v=0012",
        commentary_eligible=True,
    )


def _provenance() -> dict[str, object]:
    return {
        "transcript": {"method": "captions", "configuration": {"language": "en"}},
        "cleaning": {"method": "gemini", "configuration": {"model": "clean-model"}},
        "pairing": {"method": "first_chunk_after_rally", "configuration": {"window_s": 8}},
    }


def _assemble(
    tmp_path: Path,
    *,
    manifest: RunManifest | None = None,
    annotation: AnnotatorResult | None = None,
    metadata: VideoMetadata | None = None,
    source_reference: SourceReference | None = None,
    annotation_fps: Fraction = FPS,
    annotation_frame_count: int = 100,
    pairing: CanonicalPairing | None = None,
    include_pairing: bool = True,
    chunks: list[dict[str, object]] | None = None,
    commentary_outcome: StageOutcome = StageOutcome.PROCESSED,
    commentary_reason: str | None = None,
    missing_reasons: dict[int, str] | None = None,
    commentary_provenance: dict[str, object] | None = None,
) -> RallyRecordProjection:
    canonical = _metadata(tmp_path) if metadata is None else metadata
    selected_manifest = _manifest() if manifest is None else manifest
    selected_pairing = _pairing(canonical) if pairing is None else pairing
    if not include_pairing:
        selected_pairing = None
    return assemble_rally_records(
        manifest=selected_manifest,
        source_dataset="scraped-professional",
        video_id=VIDEO_ID,
        source_reference=_source_reference() if source_reference is None else source_reference,
        metadata=canonical,
        annotation=_annotation() if annotation is None else annotation,
        annotation_fps=annotation_fps,
        annotation_frame_count=annotation_frame_count,
        pairing=selected_pairing,
        chunks=_chunks() if chunks is None else chunks,
        commentary_outcome=commentary_outcome,
        commentary_reason=commentary_reason,
        commentary_missing_reasons={1: "no_time_window_pair"} if missing_reasons is None else missing_reasons,
        commentary_provenance=(
            _provenance() if commentary_provenance is None else commentary_provenance
        ),
        mask_stage_name="annotation",
    )


def _write(
    run_dir: Path,
    manifest: RunManifest,
    projections: list[RallyRecordProjection],
    *,
    assembly_configuration: dict[str, object] | None = None,
) -> RallyRecordArtifacts:
    return write_rally_records(
        run_dir,
        manifest,
        projections,
        code_version=CODE_VERSION,
        assembly_configuration=(
            {"record_mode": "primitive"}
            if assembly_configuration is None
            else assembly_configuration
        ),
    )


def test_exact_record_fixture_covers_every_mapped_primitive(tmp_path: Path) -> None:
    metadata = _metadata(tmp_path)
    projection = _assemble(tmp_path, metadata=metadata)
    records = list(projection.records)
    expected_source = {
        "source_dataset": "scraped-professional",
        "video_id": VIDEO_ID,
        "source_reference": _source_reference().to_dict(),
        "video_metadata": metadata.to_dict(),
        "mask_stage": "annotation",
    }
    shared = {"schema": RALLY_RECORD_SCHEMA}
    expected = [
        {
            **shared,
            "key": {
                "run_id": "run-15",
                "source_dataset": "scraped-professional",
                "video_id": VIDEO_ID,
                "rally_id": 0,
            },
            "rally": {
                "rally_id": 0,
                "start_frame": 0,
                "end_frame": 50,
                "duration_frames": 50,
                "duration_seconds": 2.0,
            },
            "contacts": {
                "raw_candidates": [
                    {
                        "contact_frame": 10,
                        "proximity_ok": False,
                        "wrist_near": False,
                        "suppressed": None,
                    },
                    {
                        "contact_frame": 20,
                        "proximity_ok": True,
                        "wrist_near": True,
                        "suppressed": False,
                    },
                    {
                        "contact_frame": 30,
                        "proximity_ok": True,
                        "wrist_near": True,
                        "suppressed": False,
                    },
                ],
                "accepted": [
                    {"stroke_idx": 0, "contact_frame": 20, "hit_height_code": 1},
                    {"stroke_idx": 1, "contact_frame": 30, "hit_height_code": None},
                ],
                "stroke_count": 2,
                "hit_height_failures": [{
                    "stroke_idx": 1,
                    "contact_frame": 30,
                    "reason": "shuttle not visible",
                }],
            },
            "outcomes": {
                "striker_half": "Top",
                "server_prediction": "Bot",
                "next_server": None,
                "verdict": {
                    "value": "lost",
                    "source": "net_rule",
                    "landing_margin_m": -0.2,
                    "within_line_margin": False,
                    "within_net_margin": True,
                },
                "landing": {
                    "frame": 45,
                    "normalized_court_position": [0.4, 0.8],
                    "court_half": "Bot",
                    "at_image_border": False,
                    "net_ender": True,
                },
                "geometric_verdict": {
                    "value": "lost",
                    "winner": "Bot",
                    "agreement": True,
                    "window_closed_by_mask": True,
                },
            },
            "commentary": {
                "stage_outcome": "processed",
                "stage_reason": None,
                "missing_reason": None,
                "chunk_id": "c0",
                "start_seconds": 2.2,
                "end_seconds": 3.0,
                "raw_text": "raw call",
                "cleaned_text": "clean call",
                "alternatives": ["alternate one", "alternate two"],
                "cleaning_diagnostics": {"bert_f1": 0.9, "clean_pass": True},
                "provenance": _provenance(),
            },
        },
        {
            **shared,
            "key": {
                "run_id": "run-15",
                "source_dataset": "scraped-professional",
                "video_id": VIDEO_ID,
                "rally_id": 1,
            },
            "rally": {
                "rally_id": 1,
                "start_frame": 60,
                "end_frame": 100,
                "duration_frames": 40,
                "duration_seconds": 1.6,
            },
            "contacts": {
                "raw_candidates": [{
                    "contact_frame": 70,
                    "proximity_ok": None,
                    "wrist_near": None,
                    "suppressed": False,
                }],
                "accepted": [
                    {"stroke_idx": 0, "contact_frame": 70, "hit_height_code": 2},
                ],
                "stroke_count": 1,
                "hit_height_failures": [],
            },
            "outcomes": {
                "striker_half": None,
                "server_prediction": None,
                "next_server": None,
                "verdict": {
                    "value": None,
                    "source": None,
                    "landing_margin_m": None,
                    "within_line_margin": None,
                    "within_net_margin": None,
                },
                "landing": None,
                "geometric_verdict": {
                    "value": None,
                    "winner": None,
                    "agreement": None,
                    "window_closed_by_mask": None,
                },
            },
            "commentary": {
                "stage_outcome": "processed",
                "stage_reason": None,
                "missing_reason": "no_time_window_pair",
                "chunk_id": None,
                "start_seconds": None,
                "end_seconds": None,
                "raw_text": None,
                "cleaned_text": None,
                "alternatives": None,
                "cleaning_diagnostics": {"bert_f1": None, "clean_pass": None},
                "provenance": _provenance(),
            },
        },
    ]

    assert projection.source == expected_source
    assert records == expected
    assert records[0]["key"]["video_id"] == VIDEO_ID
    assert isinstance(records[0]["key"]["video_id"], str)


@pytest.mark.parametrize(
    ("conflict", "match"),
    [
        ("annotation_fps", "annotation fps"),
        ("annotation_frame_count", "annotation frame_count"),
        ("pairing_fps", "pairing metadata"),
        ("pairing_frame_count", "pairing metadata"),
    ],
)
def test_conflicting_timing_stops_assembly(
    tmp_path: Path,
    conflict: str,
    match: str,
) -> None:
    metadata = _metadata(tmp_path)
    annotation_fps = Fraction(30, 1) if conflict == "annotation_fps" else FPS
    annotation_frame_count = 99 if conflict == "annotation_frame_count" else 100
    pairing_metadata = metadata
    if conflict == "pairing_fps":
        pairing_metadata = _metadata(tmp_path, fps=Fraction(30, 1))
    elif conflict == "pairing_frame_count":
        pairing_metadata = _metadata(tmp_path, frame_count=101)

    with pytest.raises(ValueError, match=match):
        _assemble(
            tmp_path,
            metadata=metadata,
            annotation_fps=annotation_fps,
            annotation_frame_count=annotation_frame_count,
            pairing=_pairing(pairing_metadata),
        )


@pytest.mark.parametrize(
    "spans",
    [
        [(0, 0), (60, 100)],
        [(-1, 50), (60, 100)],
        [(0, 101), (60, 100)],
        [(0, 50), (40, 100)],
    ],
)
def test_invalid_half_open_spans_stop_assembly(
    tmp_path: Path,
    spans: list[tuple[int, int]],
) -> None:
    annotation = _annotation()._replace(spans=spans)

    with pytest.raises(ValueError, match="span|overlapping"):
        _assemble(tmp_path, annotation=annotation)


def test_duplicate_raw_contact_composite_key_stops_assembly(tmp_path: Path) -> None:
    annotation = _annotation()
    duplicate = annotation._replace(contacts=[*annotation.contacts, annotation.contacts[0]])

    with pytest.raises(ValueError, match="raw contact composite key is duplicated"):
        _assemble(tmp_path, annotation=duplicate)


def test_duplicate_pair_composite_key_stops_assembly(tmp_path: Path) -> None:
    metadata = _metadata(tmp_path)
    pairing = _pairing(metadata)
    duplicate = CanonicalPairing(
        VIDEO_ID,
        metadata,
        (*pairing.rows, dict(pairing.rows[0])),
    )

    with pytest.raises(ValueError, match="pair composite key is duplicated"):
        _assemble(tmp_path, metadata=metadata, pairing=duplicate)


def test_filtered_contacts_and_stroke_counts_must_agree(tmp_path: Path) -> None:
    annotation = _annotation()
    mismatched_frames = annotation._replace(filtered_by_rally={0: [20], 1: [70]})
    with pytest.raises(ValueError, match="do not agree exactly"):
        _assemble(tmp_path, annotation=mismatched_frames)

    mismatched_count = annotation._replace(n_strokes_list=[1, 1])
    with pytest.raises(ValueError, match="accepted-contact count"):
        _assemble(tmp_path, annotation=mismatched_count)


def test_outcome_primitives_must_match_their_typed_producer_contracts(tmp_path: Path) -> None:
    annotation = _annotation()
    verdict = annotation.verdict_rows[0]
    invalid_verdict = annotation._replace(
        verdict_rows={0: verdict._replace(margin_m=float("nan"))},
    )
    with pytest.raises(ValueError, match="finite number"):
        _assemble(tmp_path, annotation=invalid_verdict)

    landing = annotation.landings[0]
    invalid_landing = annotation._replace(
        landings={0: landing._replace(frame=100)},
    )
    with pytest.raises(ValueError, match="canonical frame bounds"):
        _assemble(tmp_path, annotation=invalid_landing)

    geometric = annotation.geometric_verdict_rows[0]
    invalid_geometric = annotation._replace(
        geometric_verdict_rows={0: geometric._replace(agreement=1)},
    )
    with pytest.raises(ValueError, match="agreement must be boolean"):
        _assemble(tmp_path, annotation=invalid_geometric)


def test_server_prediction_must_match_striker_resolution_and_parity(tmp_path: Path) -> None:
    annotation = _annotation()
    unresolved_server = annotation._replace(fitted_first_all=[Half.BOT, Half.TOP])
    with pytest.raises(ValueError, match="unresolved rally 1 must be null"):
        _assemble(tmp_path, annotation=unresolved_server)

    wrong_parity = annotation._replace(fitted_first_all=[Half.TOP, None])
    with pytest.raises(ValueError, match="conflicts with striker parity"):
        _assemble(tmp_path, annotation=wrong_parity)

    wrong_next_server = annotation._replace(next_servers=[Half.TOP, None])
    with pytest.raises(ValueError, match="following rallies' server predictions"):
        _assemble(tmp_path, annotation=wrong_next_server)


def test_source_reference_must_match_canonical_metadata_basename(tmp_path: Path) -> None:
    source_reference = SourceReference(
        video_id=VIDEO_ID,
        basename="different.mp4",
        title="Professional singles final",
        url="https://example.test/watch?v=0012",
        commentary_eligible=True,
    )

    with pytest.raises(ValueError, match="basename conflicts"):
        _assemble(tmp_path, source_reference=source_reference)


def test_source_basename_is_provenance_not_record_identity(tmp_path: Path) -> None:
    source = (tmp_path / f"{VIDEO_ID} Match Name.mp4").resolve()
    source.write_bytes(b"video")
    metadata = VideoMetadata(source, FPS, 100, 100, 50)
    source_reference = SourceReference(
        video_id=VIDEO_ID,
        basename=source.name,
        title="Professional singles final",
        url="https://example.test/watch?v=0012",
        commentary_eligible=True,
    )

    projection = _assemble(
        tmp_path,
        metadata=metadata,
        source_reference=source_reference,
    )

    key = projection.records[0]["key"]
    source_payload = projection.source
    assert isinstance(key, dict)
    assert isinstance(source_payload, dict)
    reference_payload = source_payload["source_reference"]
    assert isinstance(reference_payload, dict)
    assert key["video_id"] == VIDEO_ID
    assert reference_payload["basename"] == source.name


def test_persisted_free_form_provenance_is_redacted(tmp_path: Path) -> None:
    provenance = _provenance()
    provenance["transcript_api_token"] = "secret-value"
    projection = _assemble(tmp_path, commentary_provenance=provenance)
    artifacts = _write(
        tmp_path / "redacted",
        _manifest(),
        [projection],
        assembly_configuration={"service_password": "secret-value"},
    )
    payload = load_json_gz(artifacts.records)
    commentary = payload["records"][0]["commentary"]

    assert payload["assembly_configuration"] == {"service_password": "<redacted>"}
    assert commentary["provenance"]["transcript_api_token"] == "<redacted>"
    assert "secret-value" not in str(payload)
    assert load_rally_records(artifacts.records) == list(projection.records)

    payload["assembly_configuration"] = {"service_password": "secret-value"}
    tampered = save_json_gz(tmp_path / "redacted" / "unredacted.json.gz", payload)
    with pytest.raises(ValueError, match="unredacted secrets"):
        load_rally_records(tampered)

    malformed_records = deepcopy(projection.records)
    malformed_records[0]["commentary"]["provenance"]["api_token"] = "secret-value"
    malformed = replace(projection, records=tuple(malformed_records))
    with pytest.raises(ValueError, match="commentary provenance contains unredacted secrets"):
        _write(tmp_path / "unredacted-projection", _manifest(), [malformed])

    payload = load_json_gz(artifacts.records)
    payload["records"][0]["commentary"]["provenance"]["api_token"] = "secret-value"
    tampered = save_json_gz(tmp_path / "redacted" / "unredacted-commentary.json.gz", payload)
    with pytest.raises(ValueError, match="commentary provenance contains unredacted secrets"):
        load_rally_records(tampered)


def test_paired_commentary_requires_transcript_and_cleaning_provenance(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing transcript"):
        _assemble(tmp_path, commentary_provenance={})


def test_assembled_records_do_not_share_mutable_provenance(tmp_path: Path) -> None:
    records = _assemble(tmp_path).records
    first_commentary = records[0]["commentary"]
    second_commentary = records[1]["commentary"]
    assert isinstance(first_commentary, dict)
    assert isinstance(second_commentary, dict)

    first_provenance = first_commentary["provenance"]
    second_provenance = second_commentary["provenance"]
    assert isinstance(first_provenance, dict)
    assert isinstance(second_provenance, dict)
    first_provenance["transcript"] = "mutated"

    assert second_provenance["transcript"] == _provenance()["transcript"]


def test_commentary_ineligible_source_cannot_contain_a_pair(tmp_path: Path) -> None:
    source_reference = SourceReference(
        video_id=VIDEO_ID,
        basename=f"{VIDEO_ID}.mp4",
        title="Professional singles final",
        url="https://example.test/watch?v=0012",
        commentary_eligible=False,
    )

    with pytest.raises(ValueError, match="ineligible source"):
        _assemble(tmp_path, source_reference=source_reference)


def test_unavailable_commentary_keeps_every_rally_with_null_values(tmp_path: Path) -> None:
    records = _assemble(
        tmp_path,
        include_pairing=False,
        chunks=[],
        commentary_outcome=StageOutcome.UNAVAILABLE,
        commentary_reason="unavailable_transcript",
        missing_reasons={},
    ).records

    assert len(records) == len(_annotation().spans)
    for record in records:
        commentary = record["commentary"]
        assert commentary["stage_outcome"] == "unavailable"
        assert commentary["missing_reason"] == "unavailable_transcript"
        assert commentary["chunk_id"] is None
        assert commentary["raw_text"] is None


@pytest.mark.parametrize("outcome", [StageOutcome.PROCESSED, StageOutcome.SKIPPED])
def test_successful_commentary_requires_canonical_pairing(
    tmp_path: Path,
    outcome: StageOutcome,
) -> None:
    reason = "reused canonical pairing" if outcome is StageOutcome.SKIPPED else None
    with pytest.raises(ValueError, match="requires canonical pairing evidence"):
        _assemble(
            tmp_path,
            include_pairing=False,
            commentary_outcome=outcome,
            commentary_reason=reason,
        )


def test_record_persistence_round_trips_and_writes_manifest(tmp_path: Path) -> None:
    projection = _assemble(tmp_path)
    run_dir = tmp_path / "run"

    artifacts = _write(run_dir, _manifest(), [projection])
    payload = load_json_gz(artifacts.records)

    assert artifacts.records == run_dir / RALLY_RECORDS_FILENAME
    assert artifacts.run_manifest == run_dir / "run_manifest.json.gz"
    assert payload["schema"] == RALLY_RECORD_COLLECTION_SCHEMA
    assert payload["sources"] == [projection.source]
    assert all(record["schema"] == RALLY_RECORD_SCHEMA for record in payload["records"])
    assert all(
        set(record) == {"schema", "key", "rally", "contacts", "outcomes", "commentary"}
        for record in payload["records"]
    )
    assert load_rally_records(artifacts.records) == list(projection.records)

    payload["schema"] = "rally-record-collection/0.1"
    unsupported = save_json_gz(run_dir / "unsupported.json.gz", payload)
    with pytest.raises(ValueError, match="unsupported rally record collection schema"):
        load_rally_records(unsupported)

    payload = load_json_gz(artifacts.records)
    payload["records"][0]["schema"] = "rally-record/0.1"
    unsupported_row = save_json_gz(run_dir / "unsupported-row.json.gz", payload)
    with pytest.raises(ValueError, match="rally record schema differs"):
        load_rally_records(unsupported_row)


def test_projection_persistence_round_trips_source_and_minimal_rows(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    projection = _assemble(tmp_path, manifest=manifest)
    path = tmp_path / "primitive_projection.json.gz"

    assert write_rally_record_projection(path, manifest, projection) == path
    payload = load_json_gz(path)

    assert payload == {
        "schema": RALLY_RECORD_PROJECTION_SCHEMA,
        "input_manifest_sha256": run_manifest_sha256(manifest),
        "source": projection.source,
        "records": list(projection.records),
    }
    assert load_rally_record_projection(
        path,
        manifest,
        video_id=VIDEO_ID,
    ) == projection


def test_projection_loader_rejects_corrupt_record_content(tmp_path: Path) -> None:
    manifest = _manifest()
    projection = _assemble(tmp_path, manifest=manifest)
    path = write_rally_record_projection(
        tmp_path / "primitive_projection.json.gz",
        manifest,
        projection,
    )
    payload = load_json_gz(path)
    del payload["records"][0]["contacts"]["accepted"]
    save_json_gz(path, payload)

    with pytest.raises(ValueError, match="record contacts fields differ"):
        load_rally_record_projection(path, manifest, video_id=VIDEO_ID)


def test_projection_loader_rejects_a_different_manifest_snapshot(tmp_path: Path) -> None:
    manifest = _manifest()
    projection = _assemble(tmp_path, manifest=manifest)
    path = write_rally_record_projection(
        tmp_path / "primitive_projection.json.gz",
        manifest,
        projection,
    )
    changed_stage = replace(manifest.stages[0], command=("python", "changed"))
    changed = replace(manifest, stages=(changed_stage, *manifest.stages[1:]))

    with pytest.raises(ValueError, match="projection input-manifest digest differs"):
        load_rally_record_projection(path, changed, video_id=VIDEO_ID)


def test_duplicate_record_composite_key_is_rejected_before_write(tmp_path: Path) -> None:
    projection = _assemble(tmp_path)
    duplicate = replace(
        projection,
        records=(projection.records[0], projection.records[0]),
    )

    with pytest.raises(ValueError, match="composite key is duplicated"):
        _write(tmp_path / "run", _manifest(), [duplicate])


def test_duplicate_source_is_rejected_before_write(tmp_path: Path) -> None:
    projection = _assemble(tmp_path)

    with pytest.raises(ValueError, match="record source is duplicated"):
        _write(tmp_path / "duplicate-source", _manifest(), [projection, projection])


def test_loader_rejects_rows_outside_collection_source_order(tmp_path: Path) -> None:
    first = _assemble(tmp_path)
    second_source = deepcopy(first.source)
    second_source["source_dataset"] = "second-dataset"
    second_records = deepcopy(first.records)
    for record in second_records:
        record["key"]["source_dataset"] = "second-dataset"
    second = RallyRecordProjection(
        first.input_manifest_sha256,
        second_source,
        tuple(second_records),
    )
    run_dir = tmp_path / "source-order"
    artifacts = _write(run_dir, _manifest(), [first, second])
    payload = load_json_gz(artifacts.records)
    payload["records"] = [*payload["records"][2:], *payload["records"][:2]]
    tampered = save_json_gz(run_dir / "source-order-tampered.json.gz", payload)

    with pytest.raises(ValueError, match="source order"):
        load_rally_records(tampered)


def test_persistence_rejects_a_missing_nested_record_primitive(tmp_path: Path) -> None:
    projection = _assemble(tmp_path)
    malformed = deepcopy(projection.records)
    del malformed[0]["contacts"]["accepted"]
    malformed_projection = replace(projection, records=tuple(malformed))

    with pytest.raises(ValueError, match="record contacts fields differ"):
        _write(tmp_path / "rejected", _manifest(), [malformed_projection])

    artifacts = _write(tmp_path / "valid", _manifest(), [projection])
    payload = load_json_gz(artifacts.records)
    payload["records"][0]["contacts"].pop("accepted")
    tampered = save_json_gz(tmp_path / "tampered.json.gz", payload)
    with pytest.raises(ValueError, match="record contacts fields differ"):
        load_rally_records(tampered)


def test_persistence_rejects_span_and_outcome_link_drift(tmp_path: Path) -> None:
    projection = _assemble(tmp_path)
    overlapping = deepcopy(projection.records)
    overlapping[1]["rally"].update({
        "start_frame": 40,
        "duration_frames": 60,
        "duration_seconds": 2.4,
    })
    with pytest.raises(ValueError, match="overlap or are unordered"):
        _write(
            tmp_path / "span-drift",
            _manifest(),
            [replace(projection, records=tuple(overlapping))],
        )

    wrong_next_server = deepcopy(projection.records)
    wrong_next_server[0]["outcomes"]["next_server"] = "Top"
    with pytest.raises(ValueError, match="next_server conflicts with the following rally"):
        _write(
            tmp_path / "next-server-drift",
            _manifest(),
            [replace(projection, records=tuple(wrong_next_server))],
        )


def test_writer_rejects_a_different_same_id_manifest_before_publication(tmp_path: Path) -> None:
    projection = _assemble(tmp_path)
    original = _manifest()
    changed_stage = replace(original.stages[0], command=("python", "different"))
    different = RunManifest(
        run_id=original.run_id,
        created_at_utc=original.created_at_utc,
        stages=(changed_stage, *original.stages[1:]),
    )
    run_dir = tmp_path / "mismatched-manifest"

    with pytest.raises(ValueError, match="projection input-manifest digest differs"):
        _write(run_dir, different, [projection])

    assert not run_dir.exists()


def test_writer_validates_rows_against_the_exact_projection_manifest(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    projection_manifest = replace(manifest, stages=manifest.stages[:1])
    forged = replace(
        _assemble(tmp_path, manifest=manifest),
        input_manifest_sha256=run_manifest_sha256(projection_manifest),
    )
    run_dir = tmp_path / "missing-projection-mask"

    with pytest.raises(ValueError, match="mask stage is absent"):
        write_rally_records(
            run_dir,
            manifest,
            [forged],
            code_version=CODE_VERSION,
            assembly_configuration={"record_mode": "primitive"},
            projection_manifest=projection_manifest,
        )

    assert not run_dir.exists()


@pytest.mark.parametrize(
    "field",
    ["command", "dependencies", "counts", "elapsed_seconds", "semantic_validation", "fingerprint"],
)
def test_manifest_digest_covers_every_stage_identity_group(tmp_path: Path, field: str) -> None:
    projection = _assemble(tmp_path)
    original = _manifest()
    stages = list(original.stages)
    stage_index = 2 if field == "dependencies" else 0
    stage = stages[stage_index]
    if field == "command":
        changed = replace(stage, command=("python", "different"))
    elif field == "dependencies":
        changed = replace(stage, dependencies=("vision", "annotation"))
    elif field == "counts":
        changed = replace(stage, counts=(("videos", 1),))
    elif field == "elapsed_seconds":
        changed = replace(stage, elapsed_seconds=2.0)
    elif field == "semantic_validation":
        changed = replace(
            stage,
            semantic_validation=(SemanticValidation("schema", True),),
        )
    else:
        changed = replace(
            stage,
            fingerprint=replace(stage.fingerprint, contract_version="different/0.1"),
        )
    stages[stage_index] = changed
    different = RunManifest(
        run_id=original.run_id,
        created_at_utc=original.created_at_utc,
        stages=tuple(stages),
    )
    run_dir = tmp_path / field
    artifacts = _write(run_dir, original, [projection])
    payload = load_json_gz(artifacts.records)
    payload["input_manifest"] = different.to_dict()
    tampered = save_json_gz(run_dir / f"tampered-{field}.json.gz", payload)

    assert run_manifest_sha256(original) != run_manifest_sha256(different)
    with pytest.raises(ValueError, match="digest differs from its snapshot"):
        load_rally_records(tampered)


def test_loader_rejects_live_manifest_drift_from_the_input_snapshot(tmp_path: Path) -> None:
    projection = _assemble(tmp_path)
    original = _manifest()
    run_dir = tmp_path / "load-manifest-drift"
    artifacts = _write(run_dir, original, [projection])
    changed_stage = replace(original.stages[0], command=("python", "different"))
    different = RunManifest(
        run_id=original.run_id,
        created_at_utc=original.created_at_utc,
        stages=(changed_stage, *original.stages[1:]),
    )
    write_run_manifest(run_dir, different)

    with pytest.raises(ValueError, match="live run manifest does not extend"):
        load_rally_records(artifacts.records)


def test_live_manifest_can_append_the_assembly_output_without_a_hash_cycle(tmp_path: Path) -> None:
    projection = _assemble(tmp_path)
    input_manifest = _manifest()
    run_dir = tmp_path / "assembly-extension"
    artifacts = _write(run_dir, input_manifest, [projection])
    assembly = _stage(
        "assembly",
        configuration={"record_mode": "primitive"},
        outputs=(artifact_integrity("rally_records", artifacts.records, relative_to=run_dir),),
        dependencies=("commentary",),
        marker="6",
    )
    live_manifest = RunManifest(
        run_id=input_manifest.run_id,
        created_at_utc=input_manifest.created_at_utc,
        stages=(*input_manifest.stages, assembly),
    )
    write_run_manifest(run_dir, live_manifest)

    assert load_rally_records(artifacts.records) == list(projection.records)
    reassembled = _assemble(tmp_path, manifest=live_manifest)
    with pytest.raises(ValueError, match="already references the rally-record output"):
        _write(run_dir, live_manifest, [reassembled])
    assert load_rally_records(artifacts.records) == list(projection.records)


def test_loader_rejects_live_manifest_extension_from_a_different_code_version(
    tmp_path: Path,
) -> None:
    projection = _assemble(tmp_path)
    input_manifest = _manifest()
    run_dir = tmp_path / "assembly-version-drift"
    artifacts = _write(run_dir, input_manifest, [projection])
    assembly = _stage(
        "assembly",
        configuration={"record_mode": "primitive"},
        outputs=(artifact_integrity("rally_records", artifacts.records, relative_to=run_dir),),
        dependencies=("commentary",),
        marker="6",
    )
    assembly = replace(
        assembly,
        fingerprint=replace(assembly.fingerprint, source_commit="b" * 40),
    )
    live_manifest = replace(input_manifest, stages=(*input_manifest.stages, assembly))
    write_run_manifest(run_dir, live_manifest)

    with pytest.raises(ValueError, match="run manifest code versions"):
        load_rally_records(artifacts.records)


def test_empty_collection_detects_live_manifest_drift(tmp_path: Path) -> None:
    original = _manifest()
    run_dir = tmp_path / "empty-manifest-drift"
    artifacts = _write(run_dir, original, [])
    changed_stage = replace(original.stages[0], command=("python", "different"))
    different = RunManifest(
        run_id=original.run_id,
        created_at_utc=original.created_at_utc,
        stages=(changed_stage, *original.stages[1:]),
    )
    write_run_manifest(run_dir, different)

    with pytest.raises(ValueError, match="live run manifest does not extend"):
        load_rally_records(artifacts.records)


def test_persistence_rejects_masks_from_a_failed_stage(tmp_path: Path) -> None:
    projection = _assemble(tmp_path)
    manifest = _manifest()
    failed_annotation = replace(
        manifest.stages[1],
        outcome=StageOutcome.FAILED,
        reason="synthetic failure",
    )
    failed_manifest = replace(
        manifest,
        stages=(manifest.stages[0], failed_annotation, manifest.stages[2]),
    )
    failed_projection = replace(
        projection,
        input_manifest_sha256=run_manifest_sha256(failed_manifest),
    )

    with pytest.raises(ValueError, match="mask stage must have a reusable"):
        _write(tmp_path / "failed-mask", failed_manifest, [failed_projection])

    run_dir = tmp_path / "valid-mask"
    artifacts = _write(run_dir, manifest, [projection])
    payload = load_json_gz(artifacts.records)
    payload["input_manifest"] = failed_manifest.to_dict()
    payload["input_manifest_sha256"] = run_manifest_sha256(failed_manifest)
    tampered = save_json_gz(run_dir / "failed-mask.json.gz", payload)
    with pytest.raises(ValueError, match="mask stage must have a reusable"):
        load_rally_records(tampered)


def test_canonical_pairing_consumes_exact_metadata_and_frame_count(tmp_path: Path) -> None:
    metadata = _metadata(tmp_path)
    mask = np.zeros(metadata.frame_count, dtype=bool)
    chunks = [{"chunk_id": "c0", "start": 2.2, "end": 3.0, "text": "raw"}]

    pairing = pair_video_with_metadata(
        VIDEO_ID,
        [(0, 0, 50)],
        chunks,
        mask,
        metadata,
    )

    assert pairing.video_id == VIDEO_ID
    assert pairing.metadata is metadata
    assert pairing.rows[0]["chunk_id"] == "c0"
    chunks[0]["chunk_id"] = "mutated"
    assert pairing.rows[0]["chunk_id"] == "c0"
    with pytest.raises(TypeError):
        pairing.rows[0]["chunk_id"] = "mutated"
    with pytest.raises(ValueError, match="replay mask length"):
        pair_video_with_metadata(
            VIDEO_ID,
            [(0, 0, 50)],
            chunks,
            mask[:-1],
            metadata,
        )
    with pytest.raises(ValueError, match="one-dimensional boolean"):
        pair_video_with_metadata(
            VIDEO_ID,
            [(0, 0, 50)],
            chunks,
            np.zeros(metadata.frame_count, dtype=np.uint8),
            metadata,
        )
    with pytest.raises(ValueError, match="one-dimensional boolean"):
        pair_video_with_metadata(
            VIDEO_ID,
            [(0, 0, 50)],
            chunks,
            None,
            metadata,
        )

    replay_mask = np.zeros(metadata.frame_count, dtype=bool)
    replay_mask[5:45] = True
    masked = pair_video_with_metadata(
        VIDEO_ID,
        [(0, 0, 50)],
        [{"chunk_id": "c0", "start": 2.2, "end": 3.0, "text": "raw"}],
        replay_mask,
        metadata,
    )
    assert masked.rows[0]["chunk_id"] == ""
