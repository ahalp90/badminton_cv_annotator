"""Generated-fixture contracts for the reusable validation-overlay core."""

from __future__ import annotations

import subprocess
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import pytest

from annotator.validation_overlay.core.cli import make_render_plan, render
from annotator.validation_overlay.core.decode import fetch_span, probe_video
from annotator.validation_overlay.core.timeline import (
    Segment,
    SegmentPlan,
    SpacerPlan,
    SpanState,
    build_timeline,
)


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


def test_seek_regression_pins_exact_frame_and_half_frame_behaviour(validation_video: Path) -> None:
    frame_index = 3
    exact = _seek_one(validation_video, Fraction(frame_index, SOURCE_FPS))
    half_frame = _seek_one(validation_video, Fraction(frame_index) / SOURCE_FPS + Fraction(1, 2) / SOURCE_FPS)
    all_frames = fetch_span(validation_video, 0, SOURCE_FRAMES - 1, SOURCE_FPS)
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
        fetch_span(validation_video, SOURCE_FRAMES - 2, SOURCE_FRAMES, SOURCE_FPS)
    last = fetch_span(validation_video, SOURCE_FRAMES - 1, SOURCE_FRAMES - 1, SOURCE_FPS)
    assert last.shape == (1, SOURCE_HEIGHT, SOURCE_WIDTH, 3)


@pytest.mark.slow
def test_identity_gate_verifies_distinct_indices_before_rendering(
    validation_video: Path, tmp_path: Path
) -> None:
    info = probe_video(validation_video)
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
