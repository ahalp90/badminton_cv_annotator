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
    RALLY_RECORD_SCHEMA,
    RALLY_RECORDS_FILENAME,
    SourceReference,
    assemble_rally_records,
    load_rally_records,
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
    assembly_configuration: dict[str, object] | None = None,
) -> list[dict[str, object]]:
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
        code_version=CODE_VERSION,
        assembly_configuration=(
            {"record_mode": "primitive"}
            if assembly_configuration is None
            else assembly_configuration
        ),
        mask_stage_name="annotation",
    )


def _expected_run() -> dict[str, object]:
    return {
        "run_id": "run-15",
        "created_at_utc": "2026-08-09T00:00:00Z",
        "run_manifest": "run_manifest.json.gz",
        "input_manifest_sha256": run_manifest_sha256(_manifest()),
        "code_version": CODE_VERSION,
        "configuration": {
            "assembly": {"record_mode": "primitive"},
            "stages": {
                "vision": {"device": "cuda"},
                "annotation": {"dead_mask_mode": "replay"},
                "commentary": {"window_seconds": 8},
            },
        },
        "integrity": {
            "vision": {
                "fingerprint": "1" * 64,
                "inputs": [],
                "model_weights": [],
                "outputs": [_artifact("track", "vision/shuttle_track.npy.xz", "1").to_dict()],
            },
            "annotation": {
                "fingerprint": "2" * 64,
                "inputs": [],
                "model_weights": [],
                "outputs": [artifact.to_dict() for artifact in _manifest().stages[1].outputs],
            },
            "commentary": {
                "fingerprint": "3" * 64,
                "inputs": [],
                "model_weights": [],
                "outputs": [_artifact("pairs", "commentary/pairs.csv.gz", "5").to_dict()],
            },
        },
        "stage_outcomes": {
            "vision": {"outcome": "processed", "reason": None},
            "annotation": {"outcome": "processed", "reason": None},
            "commentary": {"outcome": "processed", "reason": None},
        },
    }


def _expected_artifacts() -> dict[str, object]:
    manifest = _manifest()
    return {
        "by_stage": {
            stage.name: [artifact.to_dict() for artifact in stage.outputs]
            for stage in manifest.stages
        },
        "masks": {
            "stage": "annotation",
            "stage_configuration": {"dead_mask_mode": "replay"},
            "raw_replay_mask": manifest.stages[1].outputs[1].to_dict(),
            "definitive_exclusion_mask": manifest.stages[1].outputs[2].to_dict(),
        },
    }


def test_exact_record_fixture_covers_every_mapped_primitive(tmp_path: Path) -> None:
    metadata = _metadata(tmp_path)
    records = _assemble(tmp_path, metadata=metadata)
    expected_source = {
        "source_dataset": "scraped-professional",
        "video_id": VIDEO_ID,
        "source_reference": _source_reference().to_dict(),
        "video_metadata": metadata.to_dict(),
    }
    shared = {
        "schema": RALLY_RECORD_SCHEMA,
        "run": _expected_run(),
        "source": expected_source,
        "artifacts": _expected_artifacts(),
    }
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


def test_persisted_free_form_provenance_is_redacted(tmp_path: Path) -> None:
    provenance = _provenance()
    provenance["transcript_api_token"] = "secret-value"
    records = _assemble(
        tmp_path,
        commentary_provenance=provenance,
        assembly_configuration={"service_password": "secret-value"},
    )
    run = records[0]["run"]
    commentary = records[0]["commentary"]

    assert run["configuration"]["assembly"] == {"service_password": "<redacted>"}
    assert commentary["provenance"]["transcript_api_token"] == "<redacted>"


def test_paired_commentary_requires_transcript_and_cleaning_provenance(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing transcript"):
        _assemble(tmp_path, commentary_provenance={})


def test_assembled_records_do_not_share_mutable_provenance(tmp_path: Path) -> None:
    records = _assemble(tmp_path)
    first_run = records[0]["run"]
    second_run = records[1]["run"]
    first_commentary = records[0]["commentary"]
    second_commentary = records[1]["commentary"]
    assert isinstance(first_run, dict)
    assert isinstance(second_run, dict)
    assert isinstance(first_commentary, dict)
    assert isinstance(second_commentary, dict)

    first_run["code_version"] = "mutated"
    first_provenance = first_commentary["provenance"]
    second_provenance = second_commentary["provenance"]
    assert isinstance(first_provenance, dict)
    assert isinstance(second_provenance, dict)
    first_provenance["transcript"] = "mutated"

    assert second_run["code_version"] == CODE_VERSION
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
    )

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
    records = _assemble(tmp_path)
    run_dir = tmp_path / "run"

    artifacts = write_rally_records(run_dir, _manifest(), records)

    assert artifacts.records == run_dir / RALLY_RECORDS_FILENAME
    assert artifacts.run_manifest == run_dir / "run_manifest.json.gz"
    assert load_rally_records(artifacts.records) == records


def test_duplicate_record_composite_key_is_rejected_before_write(tmp_path: Path) -> None:
    records = _assemble(tmp_path)

    with pytest.raises(ValueError, match="composite key is duplicated"):
        write_rally_records(tmp_path / "run", _manifest(), [records[0], records[0]])


def test_persistence_rejects_a_missing_nested_record_primitive(tmp_path: Path) -> None:
    records = _assemble(tmp_path)
    malformed = deepcopy(records)
    del malformed[0]["contacts"]["accepted"]

    with pytest.raises(ValueError, match="record contacts fields differ"):
        write_rally_records(tmp_path / "rejected", _manifest(), malformed)

    artifacts = write_rally_records(tmp_path / "valid", _manifest(), records)
    payload = load_json_gz(artifacts.records)
    payload["records"][0]["contacts"].pop("accepted")
    tampered = save_json_gz(tmp_path / "tampered.json.gz", payload)
    with pytest.raises(ValueError, match="record contacts fields differ"):
        load_rally_records(tampered)


def test_persistence_rejects_cross_record_provenance_and_span_drift(tmp_path: Path) -> None:
    records = _assemble(tmp_path)
    mismatched_run = deepcopy(records)
    mismatched_run[1]["run"]["code_version"] = "b" * 40
    with pytest.raises(ValueError, match="run provenance differs within"):
        write_rally_records(tmp_path / "run-drift", _manifest(), mismatched_run)

    overlapping = deepcopy(records)
    overlapping[1]["rally"].update({
        "start_frame": 40,
        "duration_frames": 60,
        "duration_seconds": 2.4,
    })
    with pytest.raises(ValueError, match="overlap or are unordered"):
        write_rally_records(tmp_path / "span-drift", _manifest(), overlapping)

    wrong_next_server = deepcopy(records)
    wrong_next_server[0]["outcomes"]["next_server"] = "Top"
    with pytest.raises(ValueError, match="next_server conflicts with the following rally"):
        write_rally_records(tmp_path / "next-server-drift", _manifest(), wrong_next_server)


def test_writer_rejects_a_different_same_id_manifest_before_publication(tmp_path: Path) -> None:
    records = _assemble(tmp_path)
    original = _manifest()
    different = RunManifest(
        run_id=original.run_id,
        created_at_utc=original.created_at_utc,
        stages=original.stages[:1],
    )
    run_dir = tmp_path / "mismatched-manifest"

    with pytest.raises(ValueError, match="supplied input-manifest snapshot"):
        write_rally_records(run_dir, different, records)

    assert not run_dir.exists()


@pytest.mark.parametrize(
    "field",
    ["command", "dependencies", "counts", "elapsed_seconds", "semantic_validation", "fingerprint"],
)
def test_manifest_digest_covers_every_stage_identity_group(tmp_path: Path, field: str) -> None:
    records = _assemble(tmp_path)
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

    with pytest.raises(ValueError, match="supplied input-manifest snapshot"):
        write_rally_records(run_dir, different, records)

    assert not run_dir.exists()


def test_loader_rejects_live_manifest_drift_from_the_input_snapshot(tmp_path: Path) -> None:
    records = _assemble(tmp_path)
    original = _manifest()
    run_dir = tmp_path / "load-manifest-drift"
    artifacts = write_rally_records(run_dir, original, records)
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
    records = _assemble(tmp_path)
    input_manifest = _manifest()
    run_dir = tmp_path / "assembly-extension"
    artifacts = write_rally_records(run_dir, input_manifest, records)
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

    assert load_rally_records(artifacts.records) == records
    reassembled = _assemble(tmp_path, manifest=live_manifest)
    with pytest.raises(ValueError, match="already references the rally-record output"):
        write_rally_records(run_dir, live_manifest, reassembled)
    assert load_rally_records(artifacts.records) == records


def test_empty_collection_detects_live_manifest_drift(tmp_path: Path) -> None:
    original = _manifest()
    run_dir = tmp_path / "empty-manifest-drift"
    artifacts = write_rally_records(run_dir, original, [])
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
    records = _assemble(tmp_path)
    malformed = deepcopy(records)
    for record in malformed:
        record["run"]["stage_outcomes"]["annotation"] = {
            "outcome": "failed",
            "reason": "synthetic failure",
        }

    with pytest.raises(ValueError, match="mask stage must have a reusable"):
        write_rally_records(tmp_path / "failed-mask", _manifest(), malformed)

    manifest = _manifest()
    payload = {
        "schema": RALLY_RECORD_SCHEMA,
        "run_id": manifest.run_id,
        "run_manifest": "run_manifest.json.gz",
        "input_manifest_sha256": run_manifest_sha256(manifest),
        "input_manifest": manifest.to_dict(),
        "records": malformed,
    }
    tampered = save_json_gz(tmp_path / "failed-mask.json.gz", payload)
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
