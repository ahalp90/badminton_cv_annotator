"""Export the frozen v1 dataset from completed ShuttleSet22 artifacts.

ShuttleSet22 has whole-video shuttle, pose, and court primitives from issues
#106 and #120, but no production dataset-builder run. There are therefore no
annotator rallies: every rally row comes from human contacts, with
``rally_origin`` set to ``source_contacts``.

The layout under ``data_root`` is the one the issue #104 comparison consumed:
``extracted-simple/<NN video>/`` holds the primitives and a court receipt,
``annotations/set/<video>/`` holds the set CSVs, and ``sources/`` holds the
source videos, which may be absent because only their paths are recorded.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import logging
from pathlib import Path

from annotator.video_metadata import VideoMetadata
from dataset_builder.export_v1 import (
    DatasetIdentity,
    VideoInputs,
    build_video_tables,
    derive_player_inputs,
    write_dataset,
)
from dataset_builder.manifest import artifact_integrity
from dataset_builder.models import ArtifactIntegrity
from dataset_builder.players import (
    DEFAULT_PLAYERS,
    MATCH_TABLE_FILENAME,
    Player,
    load_match_players,
    load_players,
)
from dataset_builder.schema_v1 import PRIMITIVE_ARTIFACT_NOTES
from dataset_builder.vision import (
    COURT_EVIDENCE_FILENAME,
    COURT_KEEP_VOTE_FILENAME,
    COURT_PRESENT_FILENAME,
    POSE_FILENAMES,
    TRACK_FILENAME,
    load_json_gz,
)
from shuttleset22 import DEFAULT_SOURCES, Source, SourceKind, load_sources, select_sources


log = logging.getLogger(__name__)

SOURCE_DATASET = "ShuttleSet22"
EXTRACTED_DIRECTORY = "extracted-simple"
ANNOTATIONS_DIRECTORY = "annotations"
SET_DIRECTORY = "set"
SOURCES_DIRECTORY = "sources"
COURT_RECEIPT_FILENAME = "court_receipt.json.gz"
INPUT_ARTIFACT_FILENAMES: dict[str, str] = {
    "shuttle_track": TRACK_FILENAME,
    **{f"pose_{name}": filename for name, filename in POSE_FILENAMES.items()},
    "court_evidence": COURT_EVIDENCE_FILENAME,
    "court_keep_vote": COURT_KEEP_VOTE_FILENAME,
    "court_present": COURT_PRESENT_FILENAME,
}
# The base ShuttleSet22 extract was run with InpaintNet off. A later pass over
# the same videos produced these corrected sidecars, kept in a second root
# (--inpainted-root) instead of extracted-simple/ alongside the rest.
INPAINTED_TRACK_FILENAME = "shuttle_track_inpainted.npy.xz"
INPAINTED_GUARD_CODES_FILENAME = "shuttle_guard_codes_inpainted.npy.xz"
INPAINTED_ARTIFACT_FILENAMES: dict[str, str] = {
    "shuttle_track_inpainted": INPAINTED_TRACK_FILENAME,
    "shuttle_guard_codes_inpainted": INPAINTED_GUARD_CODES_FILENAME,
}
# primitive_artifacts.location is normally "input_dir" or "export_dir" (see
# export_v1.build_video_tables). This third value flags that relative_path is
# relative to --inpainted-root, not data_root, so a reader does not mistake
# the two roots for one.
LOCATION_INPAINTED_ROOT = "inpainted_root"
_ARTIFACT_NOTES = {note.artifact: note for note in PRIMITIVE_ARTIFACT_NOTES}


@dataclass(frozen=True)
class ShuttleSet22ExportInputs:
    """Explicit inputs for one ShuttleSet22 export."""

    data_root: Path
    output_dir: Path
    run_id: str
    sources: Path = DEFAULT_SOURCES
    commentary_root: Path | None = None
    replay_mask_root: Path | None = None
    match_ids: tuple[int, ...] | None = None
    players: Path = DEFAULT_PLAYERS
    inpainted_root: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("run_id must be a non-empty label for the artifact set")


def video_id_for(match_id: int) -> str:
    """Return the canonical ShuttleSet22 video identifier, for example ss22_03."""
    return f"ss22_{match_id:02d}"


def export_shuttleset22_v1(inputs: ShuttleSet22ExportInputs) -> dict[str, object]:
    """Write every v1 table and the dataset manifest; return the manifest."""
    data_root = Path(inputs.data_root).resolve(strict=True)
    inpainted_root = (
        None if inputs.inpainted_root is None
        else Path(inputs.inpainted_root).resolve(strict=True)
    )
    if inpainted_root is None:
        log.warning(
            "no --inpainted-root given: this export extracted the plain ShuttleSet22 "
            "shuttle track, which was recorded with InpaintNet off, and is not "
            "referencing the InpaintNet-corrected track or its guard codes. Pass "
            "--inpainted-root to use the corrected track."
        )
    sources = select_sources(load_sources(inputs.sources), inputs.match_ids)
    for source in sources:
        if source.kind is not SourceKind.DOWNLOAD:
            raise ValueError(
                f"source {source.match_id} is {source.kind.value}, not a ShuttleSet22 download"
            )
    annotation_root = data_root / ANNOTATIONS_DIRECTORY
    players_path = Path(inputs.players)
    players = load_players(players_path)
    videos = []
    for source in sources:
        video_tables = build_video_tables(
            inputs.output_dir,
            _video_inputs(data_root, source, inputs.run_id, players, inpainted_root),
        )
        if inpainted_root is not None:
            video_tables.artifacts.extend(_inpainted_artifact_rows(inpainted_root, source))
        videos.append(video_tables)
    identity = DatasetIdentity(
        run_id=inputs.run_id,
        source_dataset=SOURCE_DATASET,
        input_root=data_root,
        sources_manifest=artifact_integrity("shuttleset22_sources", inputs.sources).to_dict(),
        ground_truth_root=annotation_root,
        commentary_root=(
            None if inputs.commentary_root is None
            else Path(inputs.commentary_root).resolve(strict=True)
        ),
        inpainted_root=inpainted_root,
        replay_mask_root=(
            None if inputs.replay_mask_root is None
            else Path(inputs.replay_mask_root).resolve(strict=True)
        ),
        players_table=artifact_integrity("players", players_path).to_dict(),
    )
    video_ids = [video_id_for(source.match_id) for source in sources]
    return write_dataset(inputs.output_dir, identity, videos, video_ids)


def _inpainted_video_path(inpainted_root: Path, source: Source, filename: str) -> Path:
    """Resolve one inpainted sidecar path, failing loudly and naming the video."""
    video_label = f"{source.match_id:02d} {source.video}"
    video_dir = inpainted_root / video_label
    if not video_dir.is_dir():
        raise FileNotFoundError(f"{video_label}: inpainted directory not found: {video_dir}")
    path = video_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"{video_label}: inpainted artifact not found: {path}")
    return path


def _inpainted_artifact_rows(inpainted_root: Path, source: Source) -> list[dict[str, object]]:
    """Build primitive_artifacts rows for one video's InpaintNet-corrected sidecars.

    These are not part of ``VideoInputs.input_artifacts``: that tuple is
    always tagged ``location="input_dir"`` by ``build_video_tables``, which
    would hide that these two files live under a different root entirely.
    """
    video_id = video_id_for(source.match_id)
    rows: list[dict[str, object]] = []
    for name, filename in INPAINTED_ARTIFACT_FILENAMES.items():
        path = _inpainted_video_path(inpainted_root, source, filename)
        integrity = artifact_integrity(name, path, relative_to=inpainted_root)
        note = _ARTIFACT_NOTES[name]
        rows.append({
            "source_dataset": SOURCE_DATASET,
            "video_id": video_id,
            "artifact": integrity.name,
            "location": LOCATION_INPAINTED_ROOT,
            "relative_path": integrity.path,
            "md5": integrity.md5,
            "size_bytes": integrity.size_bytes,
            "reliability": note.reliability.value,
            "note": note.note,
        })
    return rows


def _video_inputs(
    data_root: Path,
    source: Source,
    run_id: str,
    players: Mapping[str, Player],
    inpainted_root: Path | None = None,
) -> VideoInputs:
    annotation_root = data_root / ANNOTATIONS_DIRECTORY
    output = data_root / EXTRACTED_DIRECTORY / f"{source.match_id:02d} {source.video}"
    receipt = load_json_gz(output / COURT_RECEIPT_FILENAME)
    metadata = metadata_from_receipt(receipt, data_root, source)
    track_path = (
        output / TRACK_FILENAME if inpainted_root is None
        else _inpainted_video_path(inpainted_root, source, INPAINTED_TRACK_FILENAME)
    )
    player_inputs = derive_player_inputs(
        track_path,
        output,
        output,
        court_video_id=str(source.match_id),
        metadata=metadata,
    )
    input_artifacts: list[ArtifactIntegrity] = [
        artifact_integrity(name, output / filename, relative_to=data_root)
        for name, filename in INPUT_ARTIFACT_FILENAMES.items()
    ]
    receipt_identity = artifact_integrity(
        "court_receipt", output / COURT_RECEIPT_FILENAME, relative_to=data_root
    )
    model = _mapping(receipt.get("model"), "court receipt model")
    return VideoInputs(
        run_id=run_id,
        source_dataset=SOURCE_DATASET,
        video_id=video_id_for(source.match_id),
        metadata=metadata,
        player_inputs=player_inputs,
        annotator_spans=(),
        input_artifacts=tuple(input_artifacts),
        annotation_dir=annotation_root / SET_DIRECTORY / source.video,
        annotation_root=annotation_root,
        match_players=load_match_players(
            annotation_root / SET_DIRECTORY / MATCH_TABLE_FILENAME, source.video, players
        ),
        identity={
            "match_id": source.match_id,
            "video": source.video,
            "court_receipt": receipt_identity.to_dict(),
            "court_code_id": str(receipt.get("code_id")),
            "court_model_md5": str(model.get("md5")),
        },
    )


def metadata_from_receipt(
    receipt: Mapping[str, object], data_root: Path, source: Source
) -> VideoMetadata:
    """Rebuild canonical video metadata from the court receipt for one source."""
    if receipt.get("match_id") != source.match_id or receipt.get("video") != source.video:
        raise ValueError(f"{source.match_id:02d}: court receipt identity differs")
    if receipt.get("completed") is not True:
        raise ValueError(f"{source.match_id:02d}: court receipt is incomplete")
    metadata = _mapping(receipt.get("metadata"), "court receipt metadata")
    return VideoMetadata(
        source_path=data_root / SOURCES_DIRECTORY / source.filename,
        fps=Fraction(
            _positive_integer(metadata.get("fps_numerator"), "fps_numerator"),
            _positive_integer(metadata.get("fps_denominator"), "fps_denominator"),
        ),
        frame_count=_positive_integer(metadata.get("frame_count"), "frame_count"),
        width=_positive_integer(metadata.get("width"), "width"),
        height=_positive_integer(metadata.get("height"), "height"),
    )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


__all__: Sequence[str] = (
    "ShuttleSet22ExportInputs",
    "export_shuttleset22_v1",
    "metadata_from_receipt",
    "video_id_for",
)
