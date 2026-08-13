"""Pre-encode frame-stream contracts for the validation-overlay assembler."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import pytest

from annotator.validation_overlay.core.cli import compose_frames, make_render_plan, render
from annotator.validation_overlay.core.decode import iter_span_frames
from annotator.validation_overlay.core.timeline import Segment
from annotator.validation_overlay.overlays.shuttle_track import BOX_COLOUR, load_track, make_draw
from annotator.video_metadata import probe_video_metadata


def test_shuttle_loader_applies_shared_visibility_and_coordinate_contract(tmp_path: Path) -> None:
    valid_path = tmp_path / "valid.npy"
    valid = np.array([[0.0, 1.0, 1.0], [4.0, -3.0, 0.0]])
    np.save(valid_path, valid, allow_pickle=False)

    np.testing.assert_array_equal(load_track(valid_path, 2), valid)

    for name, invalid, reason in (
        ("visibility", np.array([[0.5, 0.5, 0.5]]), "visibility"),
        ("coordinates", np.array([[1.01, 0.5, 1.0]]), r"within \[0, 1\]"),
    ):
        path = tmp_path / f"{name}.npy"
        np.save(path, invalid, allow_pickle=False)
        with pytest.raises(ValueError, match=reason):
            load_track(path, 1)


def test_composed_stream_pairs_every_source_frame_and_marks_before_encoding(
    validation_video: Path, tmp_path: Path
) -> None:
    info = probe_video_metadata(validation_video)
    segments = (Segment(1, 2, "one"), Segment(5, 6, "two"))
    plan = make_render_plan(
        info,
        segments,
        tmp_path / "assembled.mp4",
        render_width=64,
        hud_height=4,
        lead_in=Fraction(1, 25),
        lead_out=Fraction(1, 25),
        spacer=Fraction(2, 25),
    )
    expected_indices = [0, 1, 2, 3, None, None, 4, 5, 6, 7]
    expected_source = np.stack(list(iter_span_frames(
        validation_video, 0, 7, info.fps, info.width, info.height,
    )))
    observed: list[tuple[int, bool]] = []
    mark_counts: list[int] = []

    def draw(image: np.ndarray, source_idx: int, in_target_span: bool) -> list[str]:
        assert np.array_equal(image, expected_source[source_idx])
        observed.append((source_idx, in_target_span))
        if source_idx % 2 == 0:
            image[-3, -3] = (10, 20, 30)
            image[-4, -4] = (40, 50, 60)
            mark_counts.append(2)
        else:
            mark_counts.append(0)
        return [f"synthetic={source_idx}"]

    composed = list(compose_frames(plan, draw))
    assert len(composed) == len(expected_indices)
    assert [entry[0] for entry in observed] == [index for index in expected_indices if index is not None]
    assert observed == [
        (0, False), (1, True), (2, True), (3, False),
        (4, False), (5, True), (6, True), (7, False),
    ]
    assert 2 in mark_counts
    assert 0 in mark_counts
    assert np.array_equal(composed[4], np.zeros((48, 64, 3), dtype=np.uint8))
    assert np.array_equal(composed[5], np.zeros((48, 64, 3), dtype=np.uint8))

    result = render(plan, draw)
    assert result.output_frames == len(expected_indices)
    capture = cv2.VideoCapture(str(result.output))
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    assert frame_count == len(expected_indices)


def test_shuttle_marks_track_their_own_source_row_when_upscaled(
    validation_video: Path, tmp_path: Path
) -> None:
    """Pin the production overlay's array indexing, with the resize branch live.

    The other assembler test runs at the fixture's own width, so it never resizes,
    and its synthetic draw derives marks from ``source_idx`` rather than from an
    array. Between them those two gaps let an off-by-one like ``track[idx + 1]``
    pass the whole suite. This renders the real ``shuttle_track`` draw function at
    four times the source width and checks, on every frame, that the box centre
    sits where that frame's own track row puts it.
    """
    info = probe_video_metadata(validation_video)
    # One distinct position per source frame, spread across the lower half so the
    # HUD block (drawn afterwards, top-left) can never sit on top of a box.
    span = info.frame_count - 1
    track = np.array(
        [
            [0.15 + 0.70 * index / span, 0.55 + 0.35 * index / span, 1.0]
            for index in range(info.frame_count)
        ]
    )
    plan = make_render_plan(
        info,
        (Segment(1, 3, "one"), Segment(5, 6, "two")),
        tmp_path / "upscaled.mp4",
        render_width=info.width * 4,
        hud_height=6,
        lead_in=Fraction(1, 25),
        lead_out=Fraction(1, 25),
        spacer=Fraction(1, 25),
    )
    assert plan.output_width > info.width, "resize branch must be live for this test to mean anything"

    checked = 0
    for image, spec in zip(compose_frames(plan, make_draw(track, "shuttle", plan.hud_style)), plan.timeline.frames):
        if spec is None:
            continue
        x, y, _ = track[spec.source_idx]
        expected_x = round(float(x) * plan.output_width)
        expected_y = round(float(y) * plan.output_height)
        box_rows, box_cols = np.nonzero(np.all(image == BOX_COLOUR, axis=-1))
        assert box_cols.size, f"no box drawn on source frame {spec.source_idx}"
        # Tolerance of 1 absorbs the box's own even/odd pixel-span rounding only;
        # a one-row indexing slip moves the centre by tens of pixels here.
        assert abs((box_cols.min() + box_cols.max()) / 2 - expected_x) <= 1
        assert abs((box_rows.min() + box_rows.max()) / 2 - expected_y) <= 1
        checked += 1
    assert checked == sum(1 for spec in plan.timeline.frames if spec is not None)
