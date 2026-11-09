"""Check single-pass frame selection across a full video's many rally intervals."""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from experiments.annotator.video_models_benchmark import prepare
from experiments.annotator.video_models_benchmark.data import context_intervals


def test_ffmpeg_selects_many_disjoint_intervals_and_reuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is unavailable")
    monkeypatch.setattr(prepare, "IMAGE_SIZE", 8)
    centres = np.arange(140) * 20 + 8
    frame_count = int(centres[-1]) + 1
    frame_ids = np.arange(frame_count)
    pixels = np.zeros((frame_count, 8, 8, 3), dtype=np.uint8)
    pixels[:, :, :, 0] = (frame_ids % 256)[:, None, None]
    pixels[:, :, :, 1] = (frame_ids // 256)[:, None, None]
    source = tmp_path / "many.mkv"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "8x8",
        "-r", "30", "-i", "pipe:0", "-c:v", "ffv1", str(source),
    ], input=pixels.tobytes(), check=True)
    video = {"item_id": "many", "source_video": str(source), "frame_count": frame_count,
             "spans": [{"start_frame": int(centre), "end_frame": int(centre) + 1} for centre in centres]}
    intervals = context_intervals(video)
    assert len(intervals) == 140
    expected = np.concatenate([np.arange(start, end) for start, end in intervals])
    result = prepare.prepare_video(video, tmp_path, tmp_path / "cache")
    assert result["frames"] == len(expected)
    assert not result["reused"]
    cached = np.memmap(tmp_path / "cache/many/frames.rgb", mode="r", dtype=np.uint8,
                       shape=(len(expected), 8, 8, 3))
    decoded = cached[:, 4, 4, 0].astype(int) + cached[:, 4, 4, 1].astype(int) * 256
    np.testing.assert_array_equal(decoded, expected)
    assert decoded[0] == 0
    assert decoded[-1] == frame_count - 1
    assert prepare.prepare_video(video, tmp_path, tmp_path / "cache")["reused"]
