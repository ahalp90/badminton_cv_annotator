"""Generated-fixture contracts for the reusable validation-overlay core."""

from __future__ import annotations

import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import pytest

import annotator.video_metadata as video_metadata_module
from annotator.validation_overlay.core.cli import make_render_plan, render
from annotator.validation_overlay.core.decode import VideoInfo, iter_span_frames, probe_video
from annotator.validation_overlay.core.timeline import (
    Segment,
    SegmentPlan,
    SpacerPlan,
    SpanState,
    build_timeline,
)
from annotator.video_metadata import VideoMetadata, probe_video_metadata


SOURCE_FPS = Fraction(25)
SOURCE_WIDTH = 64
SOURCE_HEIGHT = 48
SOURCE_FRAMES = 8


def _seek_one(video: Path, timestamp: Fraction) -> np.ndarray:
    command = [
        "ffmpeg", "-v", "error", "-ss", f"{float(timestamp):.6f}", "-i", str(video),
        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.decode("utf-8", errors="replace"))
    return np.frombuffer(completed.stdout, dtype=np.uint8).reshape(SOURCE_HEIGHT, SOURCE_WIDTH, 3)


def _make_video(path: Path, rate: str, n_frames: int = 24, sar: str | None = None) -> Path:
    """Encode a losslessly-distinct test video at an exact frame rate.

    With ``sar`` unset it passes no aspect option at all, so the result carries
    no sample aspect ratio. That is the common case for a plain encode, and it
    is what this tool's own output looks like.
    """
    frames = b"".join(
        np.full((SOURCE_HEIGHT, SOURCE_WIDTH, 3), index * 10, dtype=np.uint8).tobytes()
        for index in range(n_frames)
    )
    aspect = ["-vf", f"setsar={sar}"] if sar is not None else []
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{SOURCE_WIDTH}x{SOURCE_HEIGHT}", "-framerate", rate, "-i", "-",
         "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         *aspect, "-r", rate, str(path)],
        input=frames, check=True, capture_output=True,
    )
    return path


@pytest.mark.parametrize(
    ("sar", "expected_ratio"),
    [(None, Fraction(4, 3)), ("1/1", Fraction(4, 3)), ("0/1", Fraction(4, 3)),
     ("16/15", Fraction(64, 45)), ("2/1", Fraction(8, 3)), ("1/2", Fraction(2, 3))],
)
def test_anamorphic_sources_render_at_true_shape_without_shrinking(
    tmp_path: Path, sar: str | None, expected_ratio: Fraction
) -> None:
    """Non-square pixels decide the output's shape; they are never a rejection.

    Marks are fractions of the coded frame, so pixel shape cannot move them. It
    only decides what shape the render is written at. The no-shrink assertion is
    the point of widening first: sizing off the coded width alone would satisfy
    the ratio by shrinking the other axis and discarding real detail.
    """
    video = _make_video(tmp_path / "anamorphic.mp4", "25", sar=sar)
    info = probe_video_metadata(video)
    plan = make_render_plan(
        info, (Segment(1, 3),), tmp_path / "out.mp4",
        render_width=SOURCE_WIDTH, lead_in=0, lead_out=0, spacer=0,
    )
    # Both dimensions round up to even for yuv420p, which perturbs the ratio by
    # up to a pixel on each axis. That is 1.6% on this 64x48 fixture and far less
    # at any real render width, so compare within tolerance rather than exactly.
    actual_ratio = Fraction(plan.output_width, plan.output_height)
    assert abs(actual_ratio - expected_ratio) / expected_ratio < Fraction(2, 100)
    assert plan.output_width >= info.width
    assert plan.output_height >= info.height
    assert plan.output_width % 2 == 0 and plan.output_height % 2 == 0


def _reference_hashes(video: Path, first: int, last: int) -> list[str]:
    """Hashes for source frames [first, last], decoded from zero and indexed by n."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-vf",
         f"select='between(n\\,{first}\\,{last})'", "-fps_mode", "passthrough",
         "-frames:v", str(last - first + 1), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        capture_output=True, check=True,
    ).stdout
    size = SOURCE_WIDTH * SOURCE_HEIGHT * 3
    return [hashlib.sha256(raw[i * size:(i + 1) * size]).hexdigest() for i in range(len(raw) // size)]


@pytest.mark.parametrize("rate", ["30000/1001", "30", "60", "24000/1001"])
def test_decode_is_exact_at_broadcast_frame_rates(tmp_path: Path, rate: str) -> None:
    """The timestamp maths uses Fraction so NTSC rates stay exact; prove it.

    25 fps divides a second evenly, so it hides any error that only shows up when
    a frame period is not representable in decimal. 30000/1001 does not.
    """
    video = _make_video(tmp_path / "rate.mp4", rate)
    info = probe_video_metadata(video)
    assert info.fps == Fraction(rate)
    for first, last in [(0, 2), (9, 13), (info.frame_count - 3, info.frame_count - 1)]:
        decoded = [
            hashlib.sha256(frame.tobytes()).hexdigest()
            for frame in iter_span_frames(
                video, first, last, info.fps, info.width, info.height,
            )
        ]
        assert decoded == _reference_hashes(video, first, last)


def test_video_without_aspect_metadata_reads_as_square(tmp_path: Path) -> None:
    """ffprobe omits the key entirely when no aspect ratio is recorded.

    Treating absent as an error refuses ordinary files, including this tool's own
    renders, which is how this was found.
    """
    video = _make_video(tmp_path / "no_sar.mp4", "25")
    probed = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "sample_aspect_ratio" not in json.loads(probed)["streams"][0]
    assert probe_video_metadata(video).fps == Fraction(25)


def test_canonical_metadata_round_trips_exact_values(tmp_path: Path) -> None:
    video = _make_video(tmp_path / "fractional.mp4", "30000/1001")
    metadata = probe_video_metadata(video)

    assert metadata.source_path == video.resolve()
    assert metadata.fps == Fraction(30000, 1001)
    assert metadata.frame_count == 24
    assert (metadata.width, metadata.height) == (SOURCE_WIDTH, SOURCE_HEIGHT)
    assert VideoMetadata.from_dict(metadata.to_dict()) == metadata


def test_validation_overlay_metadata_names_remain_compatible() -> None:
    assert VideoInfo is VideoMetadata
    assert probe_video is probe_video_metadata


def test_canonical_metadata_rejects_conflicting_frame_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "conflict.mp4"
    video.touch()
    payload = {
        "streams": [{
            "codec_type": "video",
            "nb_frames": "24",
            "nb_read_frames": "23",
            "width": SOURCE_WIDTH,
            "height": SOURCE_HEIGHT,
            "r_frame_rate": "25/1",
            "avg_frame_rate": "25/1",
            "start_time": "0",
        }],
        "format": {"start_time": "0"},
    }
    monkeypatch.setattr(
        video_metadata_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, json.dumps(payload), ""),
    )

    with pytest.raises(ValueError, match="conflicting frame counts"):
        probe_video_metadata(video)


@pytest.mark.parametrize("header_frame_count", [None, "N/A"])
def test_canonical_metadata_accepts_missing_header_count_when_decoded_count_is_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    header_frame_count: str | None,
) -> None:
    video = tmp_path / "missing-header-count.mp4"
    video.touch()
    payload = {
        "frames": [
            {"best_effort_timestamp": 0},
            {"best_effort_timestamp": 1},
            {"best_effort_timestamp": 2},
        ],
        "streams": [{
            "codec_type": "video",
            "nb_frames": header_frame_count,
            "nb_read_frames": "3",
            "width": SOURCE_WIDTH,
            "height": SOURCE_HEIGHT,
            "r_frame_rate": "25/1",
            "avg_frame_rate": "25/1",
            "time_base": "1/25",
            "start_time": "0",
        }],
        "format": {"start_time": "0"},
    }
    monkeypatch.setattr(
        video_metadata_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, json.dumps(payload), ""),
    )

    assert probe_video_metadata(video).frame_count == 3


def test_canonical_metadata_rejects_equal_rate_vfr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "equal-rate-vfr.mp4"
    video.touch()
    payload = {
        "frames": [
            {"best_effort_timestamp": 0},
            {"best_effort_timestamp": 1},
            {"best_effort_timestamp": 3},
        ],
        "streams": [{
            "codec_type": "video",
            "nb_frames": "3",
            "nb_read_frames": "3",
            "width": SOURCE_WIDTH,
            "height": SOURCE_HEIGHT,
            "r_frame_rate": "25/1",
            "avg_frame_rate": "25/1",
            "time_base": "1/25",
            "start_time": "0",
        }],
        "format": {"start_time": "0"},
    }
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    monkeypatch.setattr(video_metadata_module.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="frame 2 timestamp"):
        probe_video_metadata(video)
    assert "-show_frames" in commands[0]
    entries = commands[0][commands[0].index("-show_entries") + 1]
    assert "frame=best_effort_timestamp" in entries


def test_seek_regression_pins_exact_frame_and_half_frame_behaviour(validation_video: Path) -> None:
    frame_index = 3
    exact = _seek_one(validation_video, Fraction(frame_index, SOURCE_FPS))
    half_frame = _seek_one(validation_video, Fraction(frame_index) / SOURCE_FPS + Fraction(1, 2) / SOURCE_FPS)
    all_frames = np.stack(list(iter_span_frames(
        validation_video, 0, SOURCE_FRAMES - 1, SOURCE_FPS, SOURCE_WIDTH, SOURCE_HEIGHT,
    )))
    assert np.array_equal(exact, all_frames[frame_index])
    assert np.array_equal(half_frame, all_frames[frame_index + 1])


def test_planner_is_pure_and_places_spacers_between_segments() -> None:
    plan = build_timeline(
        (Segment(2, 3, "first"), Segment(7, 8, "second")),
        nb_frames=10,
        fps=SOURCE_FPS,
        lead_in=Fraction(2, 25),
        lead_out=Fraction(2, 25),
        spacer=Fraction(3, 25),
    )
    assert plan.ordered_source_indices == (0, 1, 2, 3, 4, 5, None, None, None, 5, 6, 7, 8, 9)
    assert plan.output_frame_count == 14
    assert plan.distinct_source_indices == frozenset(range(10))
    assert isinstance(plan.parts[0], SegmentPlan)
    assert isinstance(plan.parts[1], SpacerPlan)
    assert plan.parts[0].source_indices == (0, 1, 2, 3, 4, 5)
    assert plan.parts[2].source_indices == (5, 6, 7, 8, 9)
    # Source indices alone would pass a planner that mislabelled every state or
    # showed segment labels through the lead-out, so pin the HUD story too.
    first = plan.parts[0]
    assert [frame.state for frame in first.frames] == [
        SpanState.LEAD_IN, SpanState.LEAD_IN,
        SpanState.TARGET, SpanState.TARGET,
        SpanState.LEAD_OUT, SpanState.LEAD_OUT,
    ]
    assert [frame.show_segment_label for frame in first.frames] == [
        True, True, True, True, False, False,
    ]


def test_frame_zero_lead_context_is_clipped_without_padding() -> None:
    plan = build_timeline(
        (Segment(0, 2),),
        nb_frames=5,
        fps=SOURCE_FPS,
        lead_in=Fraction(3, 25),
        lead_out=Fraction(1, 25),
        spacer=0,
    )
    assert plan.ordered_source_indices == (0, 1, 2, 3)
    assert plan.ordered_source_indices.count(0) == 1
    assert plan.parts[0].effective_first == 0
    assert plan.parts[0].requested_first == -3


def test_short_read_raises_and_exact_eof_span_succeeds(validation_video: Path) -> None:
    with pytest.raises(RuntimeError, match="expected"):
        list(iter_span_frames(
            validation_video, SOURCE_FRAMES - 2, SOURCE_FRAMES, SOURCE_FPS,
            SOURCE_WIDTH, SOURCE_HEIGHT,
        ))
    last = np.stack(list(iter_span_frames(
        validation_video, SOURCE_FRAMES - 1, SOURCE_FRAMES - 1, SOURCE_FPS,
        SOURCE_WIDTH, SOURCE_HEIGHT,
    )))
    assert last.shape == (1, SOURCE_HEIGHT, SOURCE_WIDTH, 3)


@pytest.mark.slow
def test_identity_gate_verifies_distinct_indices_before_rendering(
    validation_video: Path, tmp_path: Path
) -> None:
    info = probe_video_metadata(validation_video)
    segments = (Segment(1, 3), Segment(2, 5))
    plan = make_render_plan(
        info,
        segments,
        tmp_path / "verified.mp4",
        render_width=SOURCE_WIDTH,
        hud_height=4,
        lead_in=Fraction(1, 25),
        lead_out=Fraction(1, 25),
        spacer=Fraction(1, 25),
        verify=True,
    )

    def draw(image: np.ndarray, source_idx: int, in_target_span: bool) -> list[str]:
        image[-1, -1] = (source_idx, int(in_target_span), 255)
        return [f"synthetic={source_idx}"]

    result = render(plan, draw)
    assert result.output.exists()
    assert result.verified_distinct_indices == len(plan.timeline.distinct_source_indices)
    assert result.output_frames == plan.timeline.output_frame_count
    capture = cv2.VideoCapture(str(result.output))
    try:
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == plan.timeline.output_frame_count
    finally:
        capture.release()
