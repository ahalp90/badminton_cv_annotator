"""Fixed ShuttleSet source manifest and CPU preflight contracts."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import hashlib
from pathlib import Path

import pytest

from annotator.video_metadata import VideoMetadata
from dataset_builder import vision
from dataset_builder.fixed_sources import (
    FIXED_SOURCE_DATASET,
    FIXED_SOURCE_MANIFEST_SCHEMA,
    FixedSourceManifest,
    load_fixed_acquisition,
    load_fixed_source_manifest,
    preflight_fixed_sources,
    save_fixed_acquisition,
)


def _md5(content: bytes) -> str:
    return hashlib.md5(content).hexdigest()


def _manifest_text(source_md5: str) -> str:
    return f'''schema = "{FIXED_SOURCE_MANIFEST_SCHEMA}"
dataset = "{FIXED_SOURCE_DATASET}"

[[videos]]
video_id = "sset_01"
source_id = "1"
source_url = "https://www.youtube.com/watch?v=source-one"
source_basename = "sset_01_source-one.mp4"
source_available = true
source_md5 = "{source_md5}"
fps = "25/1"
frame_count = 125
eligible = true

[videos.ground_truth]
match_id = "1"
annotation_directory = "set/match-one"

[[videos]]
video_id = "sset_12"
source_id = "12"
source_url = "https://www.youtube.com/watch?v=missing-source"
source_basename = "sset_12_missing-source.mp4"
source_available = false
eligible = false
exclusion_reason = "all frame numbers are incorrect"

[videos.ground_truth]
match_id = "12"
annotation_directory = "set/match-twelve"
'''


def _write_manifest(tmp_path: Path, source_content: bytes = b"video-one") -> FixedSourceManifest:
    manifest_path = tmp_path / "fixed_sources.toml"
    manifest_path.write_text(_manifest_text(_md5(source_content)), encoding="utf-8")
    return load_fixed_source_manifest(manifest_path)


def _source_and_ground_truth(tmp_path: Path, content: bytes = b"video-one") -> tuple[Path, Path]:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    (source_root / "sset_01_source-one.mp4").write_bytes(content)
    ground_truth_root = tmp_path / "annotations"
    ground_truth_dir = ground_truth_root / "set" / "match-one"
    ground_truth_dir.mkdir(parents=True)
    (ground_truth_dir / "set1.csv").write_text("frame,type\n1,serve\n", encoding="utf-8")
    return source_root, ground_truth_root


def _metadata(source_path: Path, *, fps: int = 25, frame_count: int = 125) -> VideoMetadata:
    return VideoMetadata(
        source_path=source_path.resolve(strict=True),
        fps=Fraction(fps),
        frame_count=frame_count,
        width=1920,
        height=1080,
    )


def test_manifest_loads_versioned_exact_string_contract(tmp_path: Path) -> None:
    manifest_path = tmp_path / "fixed_sources.toml"
    manifest_path.write_text(_manifest_text(_md5(b"video-one")), encoding="utf-8")

    manifest = load_fixed_source_manifest(manifest_path)

    assert manifest.path == manifest_path.resolve()
    assert manifest.md5 == _md5(manifest_path.read_bytes())
    assert manifest.size_bytes == manifest_path.stat().st_size
    assert manifest.schema == FIXED_SOURCE_MANIFEST_SCHEMA
    assert manifest.dataset == FIXED_SOURCE_DATASET
    assert tuple(entry.video_id for entry in manifest.videos) == ("sset_01", "sset_12")
    assert manifest.videos[0].source_id == "1"
    assert manifest.videos[0].fps == Fraction(25)
    assert manifest.videos[0].frame_count == 125
    assert manifest.videos[0].ground_truth.match_id == "1"
    assert manifest.videos[1].source_md5 is None
    assert manifest.videos[1].fps is None
    assert manifest.videos[1].exclusion_reason == "all frame numbers are incorrect"


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        (
            f'schema = "{FIXED_SOURCE_MANIFEST_SCHEMA}"',
            'schema = "dataset-builder-fixed-sources/2"',
            "unsupported fixed source manifest schema",
        ),
        (
            f'dataset = "{FIXED_SOURCE_DATASET}"',
            'dataset = "Other"',
            "unsupported fixed source dataset",
        ),
        ('video_id = "sset_01"', "video_id = 1", "video_id must be a non-empty"),
        (
            'source_basename = "sset_01_source-one.mp4"',
            'source_basename = "sset_01_source-one.mkv"',
            "format is unsupported",
        ),
        ('fps = "25/1"', 'fps = "25"', "must be a canonical positive fraction"),
        (
            "eligible = true",
            'eligible = true\nunexpected = "value"',
            r"extra=\['unexpected'\]",
        ),
    ],
)
def test_manifest_rejects_unsupported_or_noncanonical_values(
    tmp_path: Path,
    old: str,
    new: str,
    match: str,
) -> None:
    manifest_path = tmp_path / "fixed_sources.toml"
    manifest_path.write_text(_manifest_text(_md5(b"video-one")).replace(old, new, 1), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_fixed_source_manifest(manifest_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("video_id", "sset_01"),
        ("source_id", "1"),
        ("source_url", "https://www.youtube.com/watch?v=source-one"),
        ("source_basename", "sset_01_source-one.mp4"),
        ("source_md5", _md5(b"video-one")),
        ("ground_truth.annotation_directory", "set/match-one"),
    ],
)
def test_manifest_rejects_duplicate_identities(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    manifest = _write_manifest(tmp_path)
    first, second = manifest.videos
    if field == "source_id":
        duplicate = replace(
            second,
            source_id=value,
            ground_truth=replace(second.ground_truth, match_id=value),
        )
    elif field == "ground_truth.annotation_directory":
        duplicate = replace(
            second,
            ground_truth=replace(second.ground_truth, annotation_directory=value),
        )
    elif field == "source_md5":
        duplicate = replace(
            second,
            source_available=True,
            source_md5=value,
            fps=Fraction(30),
            frame_count=100,
        )
    else:
        duplicate = replace(second, **{field: value})

    with pytest.raises(ValueError, match=f"duplicate {field}"):
        replace(manifest, videos=(first, duplicate))


def test_manifest_requires_explicit_exclusion_and_complete_available_source(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    eligible, ineligible = manifest.videos

    with pytest.raises(ValueError, match="must omit exclusion_reason"):
        replace(eligible, exclusion_reason="should not be present")
    with pytest.raises(ValueError, match="match_id must equal source_id"):
        replace(eligible, ground_truth=replace(eligible.ground_truth, match_id="2"))
    with pytest.raises(ValueError, match="exclusion_reason"):
        replace(ineligible, exclusion_reason=None)
    with pytest.raises(ValueError, match="must be available"):
        replace(
            eligible,
            source_available=False,
            source_md5=None,
            fps=None,
            frame_count=None,
        )
    with pytest.raises(ValueError, match="must omit source_md5"):
        replace(ineligible, source_md5="0" * 32)


def test_preflight_returns_canonical_metadata_in_requested_order(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    source_root, ground_truth_root = _source_and_ground_truth(tmp_path)
    probed: list[Path] = []

    def probe(path: Path) -> VideoMetadata:
        probed.append(path)
        return _metadata(path)

    result = preflight_fixed_sources(
        manifest,
        source_root=source_root,
        ground_truth_root=ground_truth_root,
        requested_video_ids=["sset_01"],
        metadata_probe=probe,
    )

    assert len(result) == 1
    assert result[0].entry is manifest.videos[0]
    assert result[0].source.path == str((source_root / "sset_01_source-one.mp4").resolve())
    assert result[0].source.md5 == manifest.videos[0].source_md5
    assert result[0].metadata == _metadata(source_root / "sset_01_source-one.mp4")
    assert result[0].ground_truth_directory == (ground_truth_root / "set/match-one").resolve()
    assert probed == [(source_root / "sset_01_source-one.mp4").resolve()]


@pytest.mark.parametrize(
    ("requested", "match"),
    [
        ([], "must not be empty"),
        (["sset_01", "sset_01"], "contain duplicates"),
        (["unknown"], "is unknown"),
        (["sset_12"], "is ineligible"),
        ([1], "must be a non-empty trimmed string"),
    ],
)
def test_preflight_rejects_invalid_requested_ids_before_source_access(
    tmp_path: Path,
    requested: list[object],
    match: str,
) -> None:
    manifest = _write_manifest(tmp_path)

    with pytest.raises(ValueError, match=match):
        preflight_fixed_sources(
            manifest,
            source_root=tmp_path / "missing-sources",
            ground_truth_root=tmp_path / "missing-ground-truth",
            requested_video_ids=requested,  # type: ignore[arg-type]
        )


def test_preflight_rejects_missing_source_before_metadata_probe(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    source_root = tmp_path / "sources"
    source_root.mkdir()
    ground_truth_root = tmp_path / "annotations"
    ground_truth_root.mkdir()

    def unexpected_probe(_path: Path) -> VideoMetadata:
        raise AssertionError("metadata probe must not run")

    with pytest.raises(FileNotFoundError, match="fixed source is not a regular file"):
        preflight_fixed_sources(
            manifest,
            source_root=source_root,
            ground_truth_root=ground_truth_root,
            requested_video_ids=["sset_01"],
            metadata_probe=unexpected_probe,
        )


def test_preflight_rejects_source_symlink_before_metadata_probe(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    source_root = tmp_path / "sources"
    source_root.mkdir()
    target = tmp_path / "target.mp4"
    target.write_bytes(b"video-one")
    (source_root / "sset_01_source-one.mp4").symlink_to(target)
    ground_truth_root = tmp_path / "annotations"
    ground_truth_root.mkdir()

    with pytest.raises(FileNotFoundError, match="must not be a symlink"):
        preflight_fixed_sources(
            manifest,
            source_root=source_root,
            ground_truth_root=ground_truth_root,
            requested_video_ids=["sset_01"],
            metadata_probe=lambda path: _metadata(path),
        )


def test_preflight_rejects_digest_mismatch_before_metadata_probe(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    source_root, ground_truth_root = _source_and_ground_truth(tmp_path, b"different-video")
    called = False

    def unexpected_probe(_path: Path) -> VideoMetadata:
        nonlocal called
        called = True
        raise AssertionError("metadata probe must not run")

    with pytest.raises(ValueError, match="digest mismatch"):
        preflight_fixed_sources(
            manifest,
            source_root=source_root,
            ground_truth_root=ground_truth_root,
            requested_video_ids=["sset_01"],
            metadata_probe=unexpected_probe,
        )
    assert not called


@pytest.mark.parametrize("empty", [False, True])
def test_preflight_rejects_missing_or_empty_ground_truth_before_metadata_probe(
    tmp_path: Path,
    empty: bool,
) -> None:
    manifest = _write_manifest(tmp_path)
    source_root = tmp_path / "sources"
    source_root.mkdir()
    (source_root / "sset_01_source-one.mp4").write_bytes(b"video-one")
    ground_truth_root = tmp_path / "annotations"
    ground_truth_root.mkdir()
    if empty:
        (ground_truth_root / "set/match-one").mkdir(parents=True)

    def unexpected_probe(_path: Path) -> VideoMetadata:
        raise AssertionError("metadata probe must not run")

    expected = "contains no CSV files" if empty else "directory is missing"
    with pytest.raises(FileNotFoundError, match=expected):
        preflight_fixed_sources(
            manifest,
            source_root=source_root,
            ground_truth_root=ground_truth_root,
            requested_video_ids=["sset_01"],
            metadata_probe=unexpected_probe,
        )


@pytest.mark.parametrize(
    ("probe", "match"),
    [
        (lambda path: _metadata(path, fps=30), "FPS mismatch"),
        (lambda path: _metadata(path, frame_count=124), "frame-count mismatch"),
        (
            lambda path: VideoMetadata(
                source_path=path.parent / "other.mp4",
                fps=Fraction(25),
                frame_count=125,
                width=1920,
                height=1080,
            ),
            "identity mismatch",
        ),
    ],
)
def test_preflight_rejects_probed_source_mismatches(
    tmp_path: Path,
    probe: object,
    match: str,
) -> None:
    manifest = _write_manifest(tmp_path)
    source_root, ground_truth_root = _source_and_ground_truth(tmp_path)

    with pytest.raises(ValueError, match=match):
        preflight_fixed_sources(
            manifest,
            source_root=source_root,
            ground_truth_root=ground_truth_root,
            requested_video_ids=["sset_01"],
            metadata_probe=probe,  # type: ignore[arg-type]
        )


def test_preflight_propagates_variable_rate_failure(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    source_root, ground_truth_root = _source_and_ground_truth(tmp_path)

    def vfr_probe(_path: Path) -> VideoMetadata:
        raise ValueError("variable frame rate is unsupported")

    with pytest.raises(ValueError, match="variable frame rate is unsupported"):
        preflight_fixed_sources(
            manifest,
            source_root=source_root,
            ground_truth_root=ground_truth_root,
            requested_video_ids=["sset_01"],
            metadata_probe=vfr_probe,
        )


def test_fixed_acquisition_round_trip_can_validate_or_trust_pinned_integrity(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    source_root, ground_truth_root = _source_and_ground_truth(tmp_path)
    resolved = preflight_fixed_sources(
        manifest,
        source_root=source_root,
        ground_truth_root=ground_truth_root,
        requested_video_ids=["sset_01"],
        metadata_probe=lambda path: _metadata(path),
    )
    output = tmp_path / "fixed_acquisition.json.gz"

    save_fixed_acquisition(output, manifest, resolved)

    assert (
        load_fixed_acquisition(
            output,
            manifest,
            source_root=source_root,
            ground_truth_root=ground_truth_root,
            requested_video_ids=["sset_01"],
            validate_source_integrity=True,
        )
        == resolved
    )
    (source_root / "sset_01_source-one.mp4").write_bytes(b"changed")
    assert (
        load_fixed_acquisition(
            output,
            manifest,
            source_root=source_root,
            ground_truth_root=ground_truth_root,
            requested_video_ids=["sset_01"],
            validate_source_integrity=False,
        )
        == resolved
    )
    with pytest.raises(ValueError, match="source integrity differs"):
        load_fixed_acquisition(
            output,
            manifest,
            source_root=source_root,
            ground_truth_root=ground_truth_root,
            requested_video_ids=["sset_01"],
            validate_source_integrity=True,
        )


def test_fixed_acquisition_rejects_cross_manifest_and_metadata_identity(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    source_root, ground_truth_root = _source_and_ground_truth(tmp_path)
    resolved = preflight_fixed_sources(
        manifest,
        source_root=source_root,
        ground_truth_root=ground_truth_root,
        requested_video_ids=["sset_01"],
        metadata_probe=lambda path: _metadata(path),
    )
    output = tmp_path / "fixed_acquisition.json.gz"
    save_fixed_acquisition(output, manifest, resolved)
    payload = vision.load_json_gz(output)
    payload["videos"][0]["metadata"]["frame_count"] = 124
    vision.save_json_gz(output, payload)

    with pytest.raises(ValueError, match="metadata frame count differs"):
        load_fixed_acquisition(
            output,
            manifest,
            source_root=source_root,
            ground_truth_root=ground_truth_root,
            requested_video_ids=["sset_01"],
            validate_source_integrity=False,
        )


def test_manifest_rejects_path_unsafe_video_id(tmp_path: Path) -> None:
    manifest_path = tmp_path / "fixed_sources.toml"
    manifest_path.write_text(
        _manifest_text(_md5(b"video-one")).replace('video_id = "sset_01"', 'video_id = "../escape"'),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="path-safe basename"):
        load_fixed_source_manifest(manifest_path)
