"""Print each ShuttleSet22 video's saved shuttle guard codes as frame runs.

Run on the machine that holds the inpainted ShuttleSet22 extract, with the
extract root as the only argument and this file on stdin. It reads the same
inpainted track, guard codes and fill sidecar that the evaluated annotator run
read, and writes gzipped JSON to stdout. It changes no files.

Each run is ``[start, end, value]`` with ``end`` exclusive, in source frames.
"""

from __future__ import annotations

import gzip
import json
import lzma
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

VIDEO_DIRECTORY = re.compile(r'^(\d{2}) ')


def read_array(path: Path) -> np.ndarray:
    with lzma.open(path, 'rb') as handle:
        return np.load(handle, allow_pickle=False)


def read_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        return json.load(handle)


def value_runs(values: np.ndarray) -> list[list[int]]:
    """Return ``[start, end, value]`` for each run of equal non-zero values."""
    change_points = np.flatnonzero(np.diff(values)) + 1
    starts = np.concatenate(([0], change_points))
    ends = np.concatenate((change_points, [len(values)]))
    return [[int(start), int(end), int(values[start])] for start, end in zip(starts, ends) if values[start] != 0]


def video_runs(directory: Path) -> dict[str, Any]:
    track = read_array(directory / 'shuttle_track_inpainted.npy.xz')
    codes = read_array(directory / 'shuttle_guard_codes_inpainted.npy.xz')
    sidecar_paths = list(directory.glob('*_stride8_inpaint_mask.json.gz'))
    assert len(sidecar_paths) == 1, directory
    sidecar = read_json(sidecar_paths[0])
    frame_count = len(track)
    assert track.shape == (frame_count, 3) and codes.shape == (frame_count,)
    assert np.isin(codes, (0, 1, 2, 3)).all()
    assert sidecar['schema'] == 'inpaint_fill_mask/1' and sidecar['index_space'] == 'frame'
    assert sidecar['n_rows'] == frame_count

    visible = track[:, 2] > 0
    # Guard codes only grade visible positions, so a code on a hidden frame would mean misaligned arrays.
    assert np.all(codes[~visible] == 0)
    selected = np.zeros(frame_count, dtype=np.int8)
    for start, end in sidecar['inpaint_selected']:
        assert 0 <= start < end <= frame_count
        selected[start:end] = 1
    return {
        'frame_count': frame_count,
        'guard_runs': value_runs(codes.astype(np.int8)),
        'inpaint_runs': [run[:2] for run in value_runs(selected)],
        'hidden_runs': [run[:2] for run in value_runs((~visible).astype(np.int8))],
    }


def main() -> None:
    root = Path(sys.argv[1])
    videos = {}
    for directory in sorted(root.iterdir()):
        match = VIDEO_DIRECTORY.match(directory.name)
        if directory.is_dir() and match is not None:
            videos[str(int(match.group(1)))] = video_runs(directory)
    assert len(videos) == 47, sorted(videos)
    payload = {'schema': 'shuttle-guard-runs/1', 'videos': videos}
    with gzip.open(sys.stdout.buffer, 'wt', encoding='utf-8') as handle:
        json.dump(payload, handle)


if __name__ == '__main__':
    main()
