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
from dataset_builder.vision import (
    COURT_EVIDENCE_FILENAME,
    COURT_KEEP_VOTE_FILENAME,
    COURT_PRESENT_FILENAME,
    POSE_FILENAMES,
    TRACK_FILENAME,
    load_json_gz,
)
from shuttleset22 import DEFAULT_SOURCES, Source, SourceKind, load_sources, select_sources


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


@dataclass(frozen=True)
class ShuttleSet22ExportInputs:
    """Explicit inputs for one ShuttleSet22 export."""

    data_root: Path
    output_dir: Path
    run_id: str
    sources: Path = DEFAULT_SOURCES
    commentary_root: Path | None = None
    match_ids: tuple[int, ...] | None = None
    players: Path = DEFAULT_PLAYERS

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("run_id must be a non-empty label for the artifact set")


def video_id_for(match_id: int) -> str:
    """Return the canonical ShuttleSet22 video identifier, for example ss22_03."""
    return f"ss22_{match_id:02d}"


def export_shuttleset22_v1(inputs: ShuttleSet22ExportInputs) -> dict[str, object]:
    """Write every v1 table and the dataset manifest; return the manifest."""
    data_root = Path(inputs.data_root).resolve(strict=True)
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
        videos.append(
            build_video_tables(
                inputs.output_dir, _video_inputs(data_root, source, inputs.run_id, players)
            )
        )
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
        players_table=artifact_integrity("players", players_path).to_dict(),
    )
    video_ids = [video_id_for(source.match_id) for source in sources]
    return write_dataset(inputs.output_dir, identity, videos, video_ids)


def _video_inputs(
    data_root: Path, source: Source, run_id: str, players: Mapping[str, Player]
) -> VideoInputs:
    annotation_root = data_root / ANNOTATIONS_DIRECTORY
    output = data_root / EXTRACTED_DIRECTORY / f"{source.match_id:02d} {source.video}"
    receipt = load_json_gz(output / COURT_RECEIPT_FILENAME)
    metadata = metadata_from_receipt(receipt, data_root, source)
    player_inputs = derive_player_inputs(
        output / TRACK_FILENAME,
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
