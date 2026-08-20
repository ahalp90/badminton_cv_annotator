from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from experiments.build_track_trials import track_source_frames, tracker_crop_bounds
from experiments.build_trials import (
    EVENT_FPS,
    VideoData,
    _select_event_records,
    _verify_clip,
    _write_event_clip,
    balanced_select,
    broadcast_control_kind,
    broadcast_source_frames,
    distance_stratum,
    event_frame_in_target,
    parse_event_span,
    shift_window,
    strict_broadcast_control_kind,
)


@pytest.mark.parametrize(
    ("distance", "expected"),
    [
        (0.0, "positive"),
        (5.0, "positive"),
        (5.1, "boundary"),
        (15.0, "boundary"),
        (15.1, "negative"),
    ],
)
def test_distance_stratum(distance: float, expected: str) -> None:
    assert distance_stratum(distance) == expected


def test_shift_window_preserves_length_at_edges() -> None:
    assert shift_window(3, 50, 100) == (0, 50)
    assert shift_window(98, 50, 100) == (50, 100)


def test_zero_event_cases_do_not_require_event_candidates() -> None:
    assert _select_event_records([], 0, "filtered_contacts") == []


def test_event_target_matches_ten_base30_frames_at_25_fps() -> None:
    assert event_frame_in_target(92, 100)
    assert event_frame_in_target(108, 100)
    assert not event_frame_in_target(91, 100)
    assert not event_frame_in_target(109, 100)


def test_slow_track_clip_spends_half_its_frames_on_exact_target() -> None:
    frames = track_source_frames(100, 150, 120, 122, slow_target=True)

    assert len(frames) == 50
    assert frames[:25] == list(range(100, 150, 2))
    assert set(frames[25:]) == {120, 121}


def test_slow_track_context_resamples_sixty_native_frames_to_twenty_five() -> None:
    frames = track_source_frames(100, 160, 125, 130, slow_target=True)

    assert len(frames) == 50
    assert len(set(frames[:25])) == 25
    assert min(frames[:25]) == 100
    assert max(frames[:25]) < 160


def test_clean_then_marked_track_replay_reuses_target_frames() -> None:
    frames = track_source_frames(
        100,
        150,
        120,
        125,
        slow_target=True,
        clean_target_replay=True,
    )

    assert len(frames) == 50
    assert frames[10:30] == frames[30:50]


def test_tracker_crop_is_fixed_and_clamped_to_frame() -> None:
    track = np.zeros((10, 3), dtype=float)
    track[2:5] = (0.98, 0.02, 1.0)

    assert tracker_crop_bounds(track, 2, 5, 1920, 1080) == (1280, 0, 1920, 360)


def test_tracker_crop_requires_a_visible_target_claim() -> None:
    with pytest.raises(ValueError, match="no visible tracker claims"):
        tracker_crop_bounds(np.zeros((10, 3)), 2, 5, 1920, 1080)


def test_parse_event_span_requires_known_video_and_non_negative_index() -> None:
    assert parse_event_span("sset_15:43") == ("sset_15", 43)
    with pytest.raises(Exception, match="development video"):
        parse_event_span("sset_21:1")


def test_balanced_select_spreads_across_videos_and_time() -> None:
    rows = [
        {"video_id": video_id, "sort_frame": frame}
        for video_id in ("sset_01", "sset_15")
        for frame in range(10)
    ]
    selected = balanced_select(rows, 6)
    assert {row["video_id"] for row in selected} == {"sset_01", "sset_15"}
    assert [row["sort_frame"] for row in selected if row["video_id"] == "sset_01"] == [
        0,
        5,
        9,
    ]


def test_broadcast_controls_require_clear_visual_scene_truth() -> None:
    assert (
        broadcast_control_kind(
            one_to_one=True, live_fraction=0.8, replay_cutaway_fraction=0.2
        )
        == "positive"
    )


def test_strict_broadcast_controls_require_pure_scene_truth() -> None:
    assert (
        strict_broadcast_control_kind(
            one_to_one=True, scene_fractions={"live": 1.0}
        )
        == "positive"
    )
    assert (
        strict_broadcast_control_kind(
            one_to_one=False, scene_fractions={"replay": 1.0}
        )
        == "negative"
    )
    assert (
        strict_broadcast_control_kind(
            one_to_one=True,
            scene_fractions={"live": 0.9, "cutaway": 0.1},
        )
        is None
    )
    assert (
        strict_broadcast_control_kind(
            one_to_one=True, scene_fractions={"live-non-standard": 1.0}
        )
        is None
    )


def test_dense_broadcast_sampler_marks_thirty_ordered_target_frames() -> None:
    frames, target_outputs = broadcast_source_frames(0, 500, 200, 300)

    assert len(frames) == len(set(frames)) == 50
    assert frames == sorted(frames)
    assert target_outputs == range(10, 40)
    assert all(200 <= frames[index] < 300 for index in target_outputs)


def test_dense_broadcast_sampler_moves_context_quota_at_video_edge() -> None:
    frames, target_outputs = broadcast_source_frames(0, 500, 0, 100)

    assert target_outputs == range(30)
    assert len(frames) == len(set(frames)) == 50
    assert (
        broadcast_control_kind(
            one_to_one=False, live_fraction=0.2, replay_cutaway_fraction=0.8
        )
        == "negative"
    )
    assert (
        broadcast_control_kind(
            one_to_one=False, live_fraction=0.7, replay_cutaway_fraction=0.3
        )
        is None
    )


def _synthetic_source(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), EVENT_FPS, (320, 180)
    )
    assert writer.isOpened()
    for index in range(60):
        frame = np.full((180, 320, 3), index, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_event_clip_has_fixed_geometry_and_frame_count(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "event.mp4"
    _synthetic_source(source)
    track = np.zeros((60, 3), dtype=float)
    track[:, 0] = 0.5
    track[:, 1] = 0.5
    track[:, 2] = 1.0
    bboxes = np.zeros((60, 1, 4), dtype=float)
    bboxes[:, 0] = (120, 40, 200, 170)
    kps = np.zeros((60, 1, 17, 2), dtype=float)
    kps[:, 0, 9] = (150, 90)
    kps[:, 0, 10] = (170, 90)
    video = VideoData(
        name="sset_01",
        numeric_id=1,
        fps=25.0,
        source_path=source,
        result_path=tmp_path / "unused-result",
        track_path=tmp_path / "unused-track",
        bboxes_path=tmp_path / "unused-bboxes",
        kps_path=tmp_path / "unused-kps",
        court_present_path=tmp_path / "unused-court",
        raw_mask_path=tmp_path / "unused-raw",
        definitive_mask_path=tmp_path / "unused-definitive",
        scene_labels_path=tmp_path / "unused-scenes",
        result={},
        track=track,
        bboxes=bboxes,
        kps=kps,
        court_present=np.ones(60, dtype=bool),
        raw_mask=np.zeros(60, dtype=bool),
        definitive_mask=np.zeros(60, dtype=bool),
        scene_labels=None,  # type: ignore[arg-type]
        gt_rallies=[],
        frame_sides={},
    )
    assert _write_event_clip(video, 30, output) == (5, 55)
    _verify_clip(output, EVENT_FPS)
