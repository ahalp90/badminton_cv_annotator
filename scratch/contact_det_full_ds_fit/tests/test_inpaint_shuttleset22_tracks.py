from __future__ import annotations

import csv
import gzip
from pathlib import Path

import numpy as np
import pytest

from scratch.contact_det_full_ds_fit.scripts.inpaint_shuttleset22_tracks import (
    FRAME_HEIGHT,
    VIDEO_IDS,
    discover_videos,
    generate_inpaint_mask,
    mask_spans,
    output_paths,
    read_tracknet_csv,
)


def test_fixed_video_list_has_47_unique_ids() -> None:
    assert len(VIDEO_IDS) == 47
    assert len(set(VIDEO_IDS)) == 47
    assert VIDEO_IDS[0] == 8
    assert VIDEO_IDS[-1] == 57


def test_generate_inpaint_mask_uses_visible_points_around_a_gap() -> None:
    y = np.array([100, 100, 0, 0, 120, 120])
    visibility = np.array([1, 1, 0, 0, 1, 1])

    mask = generate_inpaint_mask(y, visibility)

    assert mask.tolist() == [0, 0, 1, 1, 0, 0]


def test_generate_inpaint_mask_leaves_low_boundary_gap_unselected() -> None:
    threshold = int(FRAME_HEIGHT * 0.05)
    y = np.array([100, 100, 0, 0, threshold, 100])
    visibility = np.array([1, 1, 0, 0, 1, 1])

    mask = generate_inpaint_mask(y, visibility)

    assert not mask.any()


def test_mask_spans_returns_half_open_ranges() -> None:
    mask = np.array([0, 1, 1, 0, 1, 0], dtype=np.float32)

    assert mask_spans(mask) == [[1, 3], [4, 5]]


def test_output_paths_are_siblings_with_distinct_names(tmp_path: Path) -> None:
    paths = output_paths(tmp_path, "08 example")

    assert paths.csv_path.name == "08 example_ball_inpainted.csv.gz"
    assert paths.sidecar_path.name == "08 example_stride8_inpaint_mask.json.gz"
    assert paths.track_path.name == "shuttle_track_inpainted.npy.xz"
    assert len(set(paths.files())) == len(paths.files())


def test_read_tracknet_csv_rejects_nonzero_invisible_coordinate(tmp_path: Path) -> None:
    path = tmp_path / "bad_ball.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("Frame", "Visibility", "X", "Y"))
        writer.writerow((0, 0, 1, 0))

    with pytest.raises(ValueError, match="Invisible TrackNet rows"):
        read_tracknet_csv(path)


def test_read_tracknet_csv_allows_original_inpaint_output_outside_frame(tmp_path: Path) -> None:
    path = tmp_path / "inpainted_ball.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("Frame", "Visibility", "X", "Y"))
        writer.writerow((0, 1, 1920, 1081))

    arrays = read_tracknet_csv(path, coordinates_must_be_in_frame=False)

    assert arrays["X"].tolist() == [1920]
    assert arrays["Y"].tolist() == [1081]


def test_read_tracknet_csv_keeps_input_coordinates_inside_frame(tmp_path: Path) -> None:
    path = tmp_path / "base_ball.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("Frame", "Visibility", "X", "Y"))
        writer.writerow((0, 1, 1920, 1081))

    with pytest.raises(ValueError, match="x coordinates are outside the frame"):
        read_tracknet_csv(path)


def test_discover_videos_requires_every_fixed_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Expected one prepared directory"):
        discover_videos(tmp_path, tmp_path)
