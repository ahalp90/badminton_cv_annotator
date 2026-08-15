"""Tests for pinned ShuttleSet22 annotations and source media."""

from __future__ import annotations

import csv
from fractions import Fraction
from pathlib import Path

import pytest

from annotator.video_metadata import VideoMetadata
from shuttleset22.manifest import (
    EXPECTED_MATCH_IDS,
    EXPECTED_UPSTREAM_COMMIT,
    OFFICIAL_BWF_CHANNEL_ID,
    OFFICIAL_DATASET_PATH,
    SOURCE_MANIFEST_SCHEMA,
    SourceKind,
    annotation_corpus_sha256,
    load_annotation_corpus,
    load_source_context,
    resolve_sources,
)


def _youtube_id(match_id: int) -> str:
    return f"v{match_id:010d}"


def _youtube_url(match_id: int) -> str:
    return f"https://www.youtube.com/watch?v={_youtube_id(match_id)}"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_annotation_fixture(root: Path, *, match_ids: tuple[int, ...] = EXPECTED_MATCH_IDS) -> None:
    rows = []
    for match_id in match_ids:
        video = f"Match_{match_id:02d}"
        rows.append(
            {
                "id": match_id,
                "video": video,
                "tournament": "Test Open",
                "round": "Finals",
                "winner": f"Winner {match_id}",
                "loser": f"Loser {match_id}",
            }
        )
        _write_csv(
            root / "set" / video / "set1.csv",
            ["time", "frame_num"],
            [{"time": "00:00:10", "frame_num": 300}],
        )
    _write_csv(
        root / "set" / "match.csv",
        ["id", "video", "tournament", "round", "winner", "loser"],
        rows,
    )


def _manifest_text(
    annotation_sha256: str,
    *,
    first_video: str = "Match_01",
    duplicate_first_url: bool = False,
    unresolved_id: int | None = None,
) -> str:
    parts = [
        f'schema = "{SOURCE_MANIFEST_SCHEMA}"\n',
        f'upstream_commit = "{EXPECTED_UPSTREAM_COMMIT}"\n',
        f'dataset_path = "{OFFICIAL_DATASET_PATH}"\n',
        f'annotation_sha256 = "{annotation_sha256}"\n',
        'expected_fps = "30/1"\n',
        f'official_channel_id = "{OFFICIAL_BWF_CHANNEL_ID}"\n',
    ]
    for match_id in EXPECTED_MATCH_IDS:
        video = first_video if match_id == 1 else f"Match_{match_id:02d}"
        parts.extend(["\n[[videos]]\n", f"id = {match_id}\n", f'video = "{video}"\n'])
        if match_id == unresolved_id:
            parts.extend(
                [
                    'source_kind = "unresolved"\n',
                    'unresolved_reason = "Exact public source is unavailable"\n',
                ]
            )
            continue
        source_id = 1 if duplicate_first_url and match_id == 2 else match_id
        parts.extend(
            [
                f'url = "{_youtube_url(source_id)}"\n',
                f'youtube_id = "{_youtube_id(source_id)}"\n',
                'source_kind = "shuttleset_overlap"\n'
                if match_id == 1
                else 'source_kind = "download"\n',
            ]
        )
        if match_id == 1:
            parts.append("overlap_shuttleset_id = 23\n")
    return "".join(parts)


def _write_source_fixture(
    tmp_path: Path,
    *,
    first_video: str = "Match_01",
    duplicate_first_url: bool = False,
    overlap_url: str | None = None,
    unresolved_id: int | None = None,
) -> tuple[Path, Path, Path]:
    dataset_root = tmp_path / "official"
    _write_annotation_fixture(dataset_root)
    manifest_path = tmp_path / "sources.toml"
    manifest_path.write_text(
        _manifest_text(
            annotation_corpus_sha256(dataset_root),
            first_video=first_video,
            duplicate_first_url=duplicate_first_url,
            unresolved_id=unresolved_id,
        ),
        encoding="utf-8",
    )
    old_match_csv = tmp_path / "shuttleset" / "match.csv"
    _write_csv(
        old_match_csv,
        ["id", "video", "url"],
        [
            {
                "id": 23,
                "video": "Match_01",
                "url": overlap_url or _youtube_url(1),
            }
        ],
    )
    return dataset_root, manifest_path, old_match_csv


def _metadata(path: Path, *, fps: Fraction = Fraction(30), frames: int = 301) -> VideoMetadata:
    return VideoMetadata(
        source_path=path,
        fps=fps,
        frame_count=frames,
        width=1920,
        height=1080,
    )


def test_load_source_context_cross_checks_all_matches_and_overlap(tmp_path: Path) -> None:
    dataset_root, manifest_path, old_match_csv = _write_source_fixture(tmp_path)

    context = load_source_context(manifest_path, dataset_root, old_match_csv)

    assert tuple(match.match_id for match in context.annotations.matches) == EXPECTED_MATCH_IDS
    assert context.annotations.matches[0].stroke_count == 1
    assert context.annotations.matches[0].max_annotated_frame == 300
    assert context.manifest.videos[0].kind is SourceKind.SHUTTLESET_OVERLAP
    assert context.manifest.videos[0].overlap_shuttleset_id == 23
    assert context.manifest.videos[1].kind is SourceKind.DOWNLOAD


def test_load_annotation_corpus_rejects_missing_match_id(tmp_path: Path) -> None:
    dataset_root = tmp_path / "official"
    _write_annotation_fixture(dataset_root, match_ids=EXPECTED_MATCH_IDS[:-1])

    with pytest.raises(ValueError, match=r"annotation match IDs differ: missing=\[58\]"):
        load_annotation_corpus(dataset_root)


def test_load_annotation_corpus_rejects_non_30_fps_timeline(tmp_path: Path) -> None:
    dataset_root = tmp_path / "official"
    _write_annotation_fixture(dataset_root)
    _write_csv(
        dataset_root / "set" / "Match_01" / "set1.csv",
        ["time", "frame_num"],
        [{"time": "00:00:10", "frame_num": 299}],
    )

    with pytest.raises(ValueError, match=r"match 1 annotations infer 29\.900000 FPS"):
        load_annotation_corpus(dataset_root)


def test_load_annotation_corpus_accepts_single_digit_hour(tmp_path: Path) -> None:
    dataset_root = tmp_path / "official"
    _write_annotation_fixture(dataset_root)
    _write_csv(
        dataset_root / "set" / "Match_01" / "set1.csv",
        ["time", "frame_num"],
        [{"time": "0:00:10", "frame_num": 300}],
    )

    corpus = load_annotation_corpus(dataset_root)

    assert corpus.matches[0].max_annotated_frame == 300


def test_load_source_context_rejects_annotation_changes(tmp_path: Path) -> None:
    dataset_root, manifest_path, old_match_csv = _write_source_fixture(tmp_path)
    _write_csv(
        dataset_root / "set" / "Match_01" / "set1.csv",
        ["time", "frame_num"],
        [{"time": "00:00:11", "frame_num": 330}],
    )

    with pytest.raises(ValueError, match="ShuttleSet22 annotation SHA-256"):
        load_source_context(manifest_path, dataset_root, old_match_csv)


def test_load_source_context_rejects_video_name_mismatch(tmp_path: Path) -> None:
    dataset_root, manifest_path, old_match_csv = _write_source_fixture(
        tmp_path,
        first_video="Wrong_Match_01",
    )

    with pytest.raises(ValueError, match="source manifest video 1"):
        load_source_context(manifest_path, dataset_root, old_match_csv)


def test_load_source_context_rejects_duplicate_urls(tmp_path: Path) -> None:
    dataset_root, manifest_path, old_match_csv = _write_source_fixture(
        tmp_path,
        duplicate_first_url=True,
    )

    with pytest.raises(ValueError, match="source manifest URLs contain duplicates"):
        load_source_context(manifest_path, dataset_root, old_match_csv)


def test_load_source_context_rejects_overlap_url_mismatch(tmp_path: Path) -> None:
    dataset_root, manifest_path, old_match_csv = _write_source_fixture(
        tmp_path,
        overlap_url="https://www.youtube.com/watch?v=different01",
    )

    with pytest.raises(ValueError, match="source 1 overlap URL differs"):
        load_source_context(manifest_path, dataset_root, old_match_csv)


def test_source_context_preserves_an_unresolved_reason(tmp_path: Path) -> None:
    dataset_root, manifest_path, old_match_csv = _write_source_fixture(
        tmp_path,
        unresolved_id=2,
    )

    context = load_source_context(manifest_path, dataset_root, old_match_csv)

    entry = context.manifest.by_id()[2]
    assert entry.kind is SourceKind.UNRESOLVED
    assert entry.url is None
    assert entry.unresolved_reason == "Exact public source is unavailable"


def test_resolve_sources_rejects_unresolved_entry(tmp_path: Path) -> None:
    dataset_root, manifest_path, old_match_csv = _write_source_fixture(
        tmp_path,
        unresolved_id=2,
    )
    context = load_source_context(manifest_path, dataset_root, old_match_csv)
    source_root = tmp_path / "sources"
    overlap_root = tmp_path / "overlap"
    source_root.mkdir()
    overlap_root.mkdir()

    with pytest.raises(ValueError, match="source 2 is unresolved"):
        resolve_sources(
            context,
            source_root=source_root,
            overlap_root=overlap_root,
            video_ids=[2],
            metadata_probe=_metadata,
        )


def test_resolve_sources_uses_explicit_download_and_overlap_paths(tmp_path: Path) -> None:
    dataset_root, manifest_path, old_match_csv = _write_source_fixture(tmp_path)
    context = load_source_context(manifest_path, dataset_root, old_match_csv)
    source_root = tmp_path / "sources"
    overlap_root = tmp_path / "overlap"
    source_root.mkdir()
    overlap_root.mkdir()
    download = source_root / "02 Match_02.mp4"
    overlap = overlap_root / "23 Match_01.mp4"
    download.write_bytes(b"download video")
    overlap.write_bytes(b"overlap video")

    resolved = resolve_sources(
        context,
        source_root=source_root,
        overlap_root=overlap_root,
        video_ids=[2, 1],
        metadata_probe=_metadata,
    )

    assert [source.entry.match_id for source in resolved] == [2, 1]
    assert [source.metadata.source_path for source in resolved] == [download, overlap]
    assert [source.integrity.size_bytes for source in resolved] == [14, 13]
    assert all(len(source.integrity.md5) == 32 for source in resolved)


@pytest.mark.parametrize(
    ("fps", "frames", "message"),
    [
        (Fraction(25), 301, "source 2 FPS 25 != 30"),
        (Fraction(30), 300, "source 2 has 300 frames but annotation requires frame 300"),
    ],
)
def test_resolve_sources_rejects_incompatible_media(
    tmp_path: Path,
    fps: Fraction,
    frames: int,
    message: str,
) -> None:
    dataset_root, manifest_path, old_match_csv = _write_source_fixture(tmp_path)
    context = load_source_context(manifest_path, dataset_root, old_match_csv)
    source_root = tmp_path / "sources"
    overlap_root = tmp_path / "overlap"
    source_root.mkdir()
    overlap_root.mkdir()
    source = source_root / "02 Match_02.mp4"
    source.write_bytes(b"video")

    with pytest.raises(ValueError, match=message):
        resolve_sources(
            context,
            source_root=source_root,
            overlap_root=overlap_root,
            video_ids=[2],
            metadata_probe=lambda path: _metadata(path, fps=fps, frames=frames),
        )


def test_resolve_sources_rejects_symlink(tmp_path: Path) -> None:
    dataset_root, manifest_path, old_match_csv = _write_source_fixture(tmp_path)
    context = load_source_context(manifest_path, dataset_root, old_match_csv)
    source_root = tmp_path / "sources"
    overlap_root = tmp_path / "overlap"
    source_root.mkdir()
    overlap_root.mkdir()
    target = tmp_path / "target.mp4"
    target.write_bytes(b"video")
    (source_root / "02 Match_02.mp4").symlink_to(target)

    with pytest.raises(FileNotFoundError, match="regular non-symlink"):
        resolve_sources(
            context,
            source_root=source_root,
            overlap_root=overlap_root,
            video_ids=[2],
            metadata_probe=_metadata,
        )


def test_resolve_sources_rejects_duplicate_content(tmp_path: Path) -> None:
    dataset_root, manifest_path, old_match_csv = _write_source_fixture(tmp_path)
    context = load_source_context(manifest_path, dataset_root, old_match_csv)
    source_root = tmp_path / "sources"
    overlap_root = tmp_path / "overlap"
    source_root.mkdir()
    overlap_root.mkdir()
    (source_root / "02 Match_02.mp4").write_bytes(b"same video")
    (source_root / "03 Match_03.mp4").write_bytes(b"same video")

    with pytest.raises(ValueError, match="duplicated between matches 2 and 3"):
        resolve_sources(
            context,
            source_root=source_root,
            overlap_root=overlap_root,
            video_ids=[2, 3],
            metadata_probe=_metadata,
        )


def test_resolve_sources_rejects_metadata_for_another_file(tmp_path: Path) -> None:
    dataset_root, manifest_path, old_match_csv = _write_source_fixture(tmp_path)
    context = load_source_context(manifest_path, dataset_root, old_match_csv)
    source_root = tmp_path / "sources"
    overlap_root = tmp_path / "overlap"
    source_root.mkdir()
    overlap_root.mkdir()
    (source_root / "02 Match_02.mp4").write_bytes(b"video")
    other = tmp_path / "other.mp4"
    other.write_bytes(b"other")

    with pytest.raises(ValueError, match="source 2 metadata path"):
        resolve_sources(
            context,
            source_root=source_root,
            overlap_root=overlap_root,
            video_ids=[2],
            metadata_probe=lambda _path: _metadata(other.resolve()),
        )
