from __future__ import annotations

from dataclasses import replace
import gzip
from fractions import Fraction
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np

from annotator.video_metadata import VideoMetadata
from dataset_builder.vision import CourtDetectorSettings, load_npy_xz, pose_artifact_paths, save_npy_xz
import shuttleset22


def test_real_source_list_records_downloads_overlaps_and_missing_sources() -> None:
    sources = shuttleset22.load_sources()

    assert len(sources) == 58
    assert sum(source.kind is shuttleset22.SourceKind.DOWNLOAD for source in sources) == 47
    overlaps = {
        source.match_id: source.overlap_id
        for source in sources
        if source.kind is shuttleset22.SourceKind.OVERLAP
    }
    assert overlaps == {1: 23, 2: 38, 3: 39, 4: 41, 5: 42, 6: 43, 7: 44, 58: 24}
    assert {
        source.match_id
        for source in sources
        if source.kind is shuttleset22.SourceKind.UNRESOLVED
    } == {14, 45, 56}
    assert {
        source.match_id for source in sources if source.excluded_reason is not None
    } == {15}


def test_default_source_selection_excludes_overlaps_unavailable_and_excluded_sources() -> None:
    selected = shuttleset22.select_sources(shuttleset22.load_sources(), ids=None)

    assert len(selected) == 46
    assert all(source.kind is shuttleset22.SourceKind.DOWNLOAD for source in selected)
    assert 15 not in {source.match_id for source in selected}


def test_explicit_ids_still_select_an_excluded_source() -> None:
    (selected,) = shuttleset22.select_sources(shuttleset22.load_sources(), ids=[15])

    assert selected.match_id == 15
    assert selected.excluded_reason is not None


def test_download_command_uses_pinned_url_and_compressed_mp4_format() -> None:
    source = shuttleset22.Source(
        match_id=8,
        video="match",
        kind=shuttleset22.SourceKind.DOWNLOAD,
        url="https://www.youtube.com/watch?v=7_O5r9CLOVw",
    )

    command = shuttleset22.download_command(
        source,
        Path("08 match.mp4"),
        cookies_from_browser="chrome:/scratch/cmarti56/issue106-youtube-chrome",
        youtube_player_client="web_embedded",
    )

    assert command[-1] == source.url
    assert command[command.index("--format") + 1] == shuttleset22.YOUTUBE_FORMAT
    for alternative in shuttleset22.YOUTUBE_FORMAT.split("/"):
        assert "[vcodec^=avc1]" in alternative
    assert command[command.index("--output") + 1] == "08 match.mp4"
    assert command[command.index("--cookies-from-browser") + 1] == (
        "chrome:/scratch/cmarti56/issue106-youtube-chrome"
    )
    assert command[command.index("--extractor-args") + 1] == "youtube:player_client=web_embedded"


def test_probe_source_defaults_missing_sample_aspect_ratio_to_square_pixels(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    probe = {
        "streams": [
            {
                "avg_frame_rate": "30/1",
                "nb_frames": "100",
                "width": 1920,
                "height": 1080,
            }
        ]
    }
    monkeypatch.setattr(
        shuttleset22.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, json.dumps(probe), ""),
    )

    metadata = shuttleset22.probe_source(source)

    assert metadata.sample_aspect_ratio == Fraction(1, 1)


def test_overlap_command_stream_copies_the_existing_video() -> None:
    source = shuttleset22.Source(
        match_id=1,
        video="match",
        kind=shuttleset22.SourceKind.OVERLAP,
        overlap_id=23,
    )

    command = shuttleset22.overlap_command(
        source,
        Path("overlaps"),
        Path("01 match.part.mp4"),
    )

    assert command[command.index("-i") + 1] == "overlaps/23 match.mp4"
    assert command[command.index("-c") + 1] == "copy"
    assert command[-1] == "01 match.part.mp4"


def test_download_sources_publishes_overlap_and_skips_unresolved(tmp_path: Path) -> None:
    overlap_root = tmp_path / "overlaps"
    overlap_root.mkdir()
    (overlap_root / "23 match.mp4").write_bytes(b"video")
    sources = (
        shuttleset22.Source(
            match_id=1,
            video="match",
            kind=shuttleset22.SourceKind.OVERLAP,
            overlap_id=23,
        ),
        shuttleset22.Source(
            match_id=14,
            video="missing",
            kind=shuttleset22.SourceKind.UNRESOLVED,
            unresolved_reason="not found",
        ),
    )

    def fake_run(command):
        source = Path(command[command.index("-i") + 1])
        destination = Path(command[-1])
        destination.write_bytes(source.read_bytes())
        return subprocess.CompletedProcess(command, 0, "", "")

    failures = shuttleset22.download_sources(
        sources,
        source_root=tmp_path / "sources",
        overlap_root=overlap_root,
        command_runner=fake_run,
    )

    assert failures == 0
    assert (tmp_path / "sources" / "01 match.mp4").read_bytes() == b"video"


def test_download_sources_counts_a_failed_download(tmp_path: Path) -> None:
    source = shuttleset22.Source(
        match_id=8,
        video="match",
        kind=shuttleset22.SourceKind.DOWNLOAD,
        url="https://www.youtube.com/watch?v=7_O5r9CLOVw",
    )

    failures = shuttleset22.download_sources(
        [source],
        source_root=tmp_path / "sources",
        overlap_root=tmp_path / "overlaps",
        command_runner=lambda command: subprocess.CompletedProcess(command, 1),
    )

    assert failures == 1


def test_gzip_csv_uses_level_nine_and_removes_plain_csv(tmp_path: Path) -> None:
    source = tmp_path / "track.csv"
    source.write_bytes(b"Frame,X,Y,Visibility\n0,1,2,1\n")

    output = shuttleset22.gzip_csv(source)

    assert not source.exists()
    assert output.read_bytes()[8] == 2  # gzip XFL=2 means maximum compression.
    with gzip.open(output, "rb") as handle:
        assert handle.read() == b"Frame,X,Y,Visibility\n0,1,2,1\n"


def test_extract_source_writes_requested_compressed_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from bst_x.pipeline import shuttle_extractor
    from dataset_builder import pose_sharding

    source = shuttleset22.Source(1, "match", shuttleset22.SourceKind.DOWNLOAD, "url")
    source_root = tmp_path / "sources"
    source_root.mkdir()
    video = source_root / source.filename
    video.write_bytes(b"video")
    metadata = VideoMetadata(
        video.resolve(),
        Fraction(30, 1),
        2,
        100,
        50,
        Fraction(1, 1),
    )
    monkeypatch.setattr(shuttleset22, "probe_source", lambda _path: metadata)

    calls = {"tracknet": 0, "pose": 0}

    def fake_tracknet(**kwargs) -> None:
        calls["tracknet"] += 1
        assert kwargs["enable_inpainting"] is False
        output = kwargs["output_csv_dir"] / f"{video.stem}_ball.csv"
        output.write_text("Frame,X,Y,Visibility\n0,10,10,1\n1,20,25,1\n")

    def fake_pose(**kwargs) -> None:
        calls["pose"] += 1
        for path in pose_artifact_paths(kwargs["output_dir"]).as_mapping().values():
            save_npy_xz(path, np.zeros((2,), dtype=np.float32))

    monkeypatch.setattr(shuttle_extractor, "extract_all_shuttles", fake_tracknet)
    monkeypatch.setattr(pose_sharding, "extract_sharded_rtmlib_pose_stage", fake_pose)
    output_root = tmp_path / "output"
    output = output_root / "01 match"
    output.mkdir(parents=True)
    (output / f"{video.stem}_ball.csv").write_text(
        "Frame,X,Y,Visibility\n0,10,10,1\n"
    )

    shuttleset22.extract_source(
        source,
        source_root=source_root,
        output_root=output_root,
        tracknet_dir=tmp_path,
        tracknet_python=Path("tracknet-python"),
        pose_python=Path("pose-python"),
        pose_shards=8,
    )

    with gzip.open(output / f"{video.stem}_ball.csv.gz", "rt") as handle:
        assert handle.readline() == "Frame,X,Y,Visibility\n"
    np.testing.assert_array_equal(
        load_npy_xz(output / "shuttle_track.npy.xz"),
        np.array([[0.1, 0.2, 1.0], [0.2, 0.5, 1.0]]),
    )
    assert all(
        path.read_bytes().startswith(b"\xfd7zXZ")
        for path in pose_artifact_paths(output).as_mapping().values()
    )

    monkeypatch.setattr(
        shuttleset22,
        "probe_source",
        lambda _path: (_ for _ in ()).throw(AssertionError("completed source was probed")),
    )
    shuttleset22.extract_source(
        source,
        source_root=source_root,
        output_root=output_root,
        tracknet_dir=tmp_path,
        tracknet_python=Path("tracknet-python"),
        pose_python=Path("pose-python"),
        pose_shards=8,
    )
    assert calls == {"tracknet": 1, "pose": 1}


COURT_SETTINGS = CourtDetectorSettings(
    python=Path("court-python"),
    deeplsd_source=Path("DeepLSD"),
    deeplsd_weights=Path("DeepLSD/weights/deeplsd_md.tar"),
    device="cuda",
    template_device="cpu",
    reuse_courts=False,
)
DEEPLSD_IDENTITY = {
    "name": "deeplsd_weights",
    "path": "deeplsd_md.tar",
    "md5": "0" * 32,
    "size_bytes": 1,
}


def _court_fixture(tmp_path: Path, monkeypatch) -> tuple[shuttleset22.Source, Path, Path]:
    """One extracted source with pose outputs; returns the source, its video and output directory."""
    from dataset_builder import vision

    source = shuttleset22.Source(8, "match", shuttleset22.SourceKind.DOWNLOAD, "url")
    source_root = tmp_path / "sources"
    source_root.mkdir()
    video = source_root / source.filename
    video.write_bytes(b"video")
    output = tmp_path / "output" / "08 match"
    output.mkdir(parents=True)
    for path in pose_artifact_paths(output).as_mapping().values():
        path.write_bytes(b"pose")
    metadata = VideoMetadata(video.resolve(), Fraction(30, 1), 2, 100, 50, Fraction(1, 1))
    monkeypatch.setattr(shuttleset22, "probe_source", lambda _path: metadata)
    monkeypatch.setattr(vision, "load_pose_arrays", lambda *_args: object())
    monkeypatch.setattr(
        vision,
        "load_court_vision",
        lambda *_args, **_kwargs: SimpleNamespace(raw_cuts=((0, 2),)),
    )
    return source, video, output


def _fake_court_build(**kwargs):
    from dataset_builder import vision

    for filename in (
        vision.COURT_EVIDENCE_FILENAME,
        vision.COURT_KEEP_VOTE_FILENAME,
        vision.COURT_PRESENT_FILENAME,
    ):
        (kwargs["output_dir"] / filename).write_bytes(filename.encode())
    return SimpleNamespace(raw_cuts=((0, 2),))


def test_court_source_publishes_receipt_and_validates_resume(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from dataset_builder import vision

    source, _video, output = _court_fixture(tmp_path, monkeypatch)
    builds: list[dict[str, object]] = []

    def fake_build(**kwargs):
        builds.append(kwargs)
        return _fake_court_build(**kwargs)

    monkeypatch.setattr(vision, "build_detected_court_stage", fake_build)
    arguments = {
        "source_root": tmp_path / "sources",
        "output_root": tmp_path / "output",
        "settings": COURT_SETTINGS,
        "model_identity": DEEPLSD_IDENTITY,
        "code_id": "a" * 64,
    }

    assert shuttleset22.court_source(source, **arguments) is True
    assert shuttleset22.court_source(source, **arguments) is False

    (build,) = builds
    assert build["settings"] is COURT_SETTINGS
    assert build["pose_dir"] == output
    assert build["ref_err_px"] == shuttleset22.COURT_REF_ERR_PX
    receipt = vision.load_json_gz(output / shuttleset22.COURT_RECEIPT_FILENAME)
    assert receipt["schema"] == "shuttleset22-court/0.2"
    assert receipt["configuration"] == {
        "device": "cuda",
        "template_device": "cpu",
        "reuse_courts": False,
        "court_mode": "video-robust",
        "ref_err_px": shuttleset22.COURT_REF_ERR_PX,
    }
    assert receipt["model"] == DEEPLSD_IDENTITY
    assert receipt["completed"] is True
    assert receipt["scene_count"] == 1
    assert len(receipt["outputs"]) == 3
    (output / vision.COURT_EVIDENCE_FILENAME).write_bytes(b"tampered")
    with np.testing.assert_raises_regex(ValueError, "output identities"):
        shuttleset22.court_source(source, **arguments)
    assert len(builds) == 1


def test_court_source_rejects_a_stale_completed_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from dataset_builder import vision

    source, video, output = _court_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(vision, "build_detected_court_stage", _fake_court_build)
    arguments = {
        "source_root": tmp_path / "sources",
        "output_root": tmp_path / "output",
        "settings": COURT_SETTINGS,
        "model_identity": DEEPLSD_IDENTITY,
        "code_id": "a" * 64,
    }
    shuttleset22.court_source(source, **arguments)

    for changed_settings in (
        replace(COURT_SETTINGS, reuse_courts=True),
        replace(COURT_SETTINGS, template_device="cuda"),
        replace(COURT_SETTINGS, court_mode="scene-robust"),
    ):
        with np.testing.assert_raises_regex(ValueError, "configuration does not match"):
            shuttleset22.court_source(source, **{**arguments, "settings": changed_settings})

    receipt_path = output / shuttleset22.COURT_RECEIPT_FILENAME
    receipt = vision.load_json_gz(receipt_path)
    old_configuration = {"device": "cuda", "resize_mode": "pad", "parent": "detected_ckn_opencv_consensus"}
    vision.save_json_gz(
        receipt_path,
        {**receipt, "schema": "shuttleset22-court/0.1", "configuration": old_configuration, "extra": 1},
    )
    with np.testing.assert_raises_regex(ValueError, "unsupported court receipt schema 'shuttleset22-court/0.1'"):
        shuttleset22.court_source(source, **arguments)

    vision.save_json_gz(receipt_path, receipt)
    video.write_bytes(b"changed video")
    with np.testing.assert_raises_regex(ValueError, "inputs"):
        shuttleset22.court_source(source, **arguments)


def test_court_sources_pins_the_deeplsd_weights_and_counts_failures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    weights = tmp_path / "deeplsd_md.tar"
    weights.write_bytes(b"weights")
    settings = replace(COURT_SETTINGS, deeplsd_weights=weights)
    sources = (
        shuttleset22.Source(8, "first", shuttleset22.SourceKind.DOWNLOAD, "url"),
        shuttleset22.Source(9, "second", shuttleset22.SourceKind.DOWNLOAD, "url"),
    )
    calls: list[tuple[int, object, object]] = []

    def fake_court_source(source, **kwargs):
        calls.append((source.match_id, kwargs["settings"], kwargs["model_identity"]))
        if source.match_id == 9:
            raise RuntimeError("failed")
        return True

    monkeypatch.setattr(shuttleset22, "court_source", fake_court_source)

    failures = shuttleset22.court_sources(
        sources,
        source_root=tmp_path / "sources",
        output_root=tmp_path / "output",
        settings=settings,
        code_id="a" * 64,
    )

    assert failures == 1
    assert [match_id for match_id, _settings, _identity in calls] == [8, 9]
    assert all(call_settings is settings for _match_id, call_settings, _identity in calls)
    identity = calls[0][2]
    assert isinstance(identity, dict)
    assert identity["name"] == "deeplsd_weights"
    assert identity["path"] == "deeplsd_md.tar"


def test_main_routes_the_court_command_without_extraction_arguments(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured = {}

    def fake_court_sources(sources, **kwargs):
        captured["count"] = len(sources)
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(shuttleset22, "court_sources", fake_court_sources)

    result = shuttleset22.main(
        [
            "court",
            "--source-root",
            str(tmp_path / "sources"),
            "--output-root",
            str(tmp_path / "output"),
            "--court-python",
            "court-python",
            "--deeplsd-source",
            "DeepLSD",
            "--deeplsd-weights",
            "DeepLSD/weights/deeplsd_md.tar",
            "--reuse-courts",
            "--code-id",
            "a" * 64,
        ]
    )

    assert result == 0
    assert captured["count"] == 46
    assert captured["settings"] == replace(COURT_SETTINGS, reuse_courts=True)
    assert captured["code_id"] == "a" * 64
