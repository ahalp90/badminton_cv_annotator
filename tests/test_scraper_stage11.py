"""Stage 11 pairing tests: the time-range join, replay exclusions, and the fps sidecar.

fps is fixed at 10 so frame/second arithmetic is easy to read: end_frame 100 is
10.0 s, and the pairing window is `(10.0, 10.0 + PAIR_WINDOW_S]`.
"""
import csv

import numpy as np

from src.scraper import stage11_pairing as stage11
from src.scraper.config import PAIR_WINDOW_S

FPS = 10.0


def _chunk(chunk_id: str, start: float, end: float) -> dict:
    return {'chunk_id': chunk_id, 'start': start, 'end': end, 'text': 'commentary'}


def test_chunk_within_window_pairs():
    rally_spans = [(0, 0, 100)]                          # rally_end_t = 10.0 s
    chunks = [_chunk('c0', 11.0, 15.0)]                  # start inside (10, 18]
    rows = stage11.pair_video('v', rally_spans, chunks, None, FPS)
    assert len(rows) == 1
    assert rows[0]['chunk_id'] == 'c0'
    assert rows[0]['commentary_start'] == 11.0
    assert rows[0]['commentary_end'] == 15.0


def test_chunk_outside_window_leaves_unpaired_blanks():
    rally_spans = [(0, 0, 100)]
    late = 10.0 + PAIR_WINDOW_S + 5.0                    # past the window
    rows = stage11.pair_video('v', rally_spans, [_chunk('c0', late, late + 2)], None, FPS)
    assert rows[0]['chunk_id'] == ''
    assert rows[0]['commentary_start'] == ''
    assert rows[0]['commentary_end'] == ''


def test_chunk_starting_on_replay_frame_is_unpairable():
    rally_spans = [(0, 0, 100)]                          # span frames 0..100, kept clear of the mask
    chunks = [_chunk('c0', 11.0, 15.0)]                  # start frame = int(11.0 * 10) = 110
    replay_mask = np.zeros(400, dtype=bool)
    replay_mask[110] = True
    rows = stage11.pair_video('v', rally_spans, chunks, replay_mask, FPS)
    assert rows[0]['chunk_id'] == ''                     # rally kept, but its only candidate is masked


def test_rally_overlapping_replay_is_held_out():
    rally_spans = [(0, 0, 100)]
    chunks = [_chunk('c0', 11.0, 15.0)]                  # a valid chunk exists
    replay_mask = np.zeros(400, dtype=bool)
    replay_mask[50] = True                               # inside the rally span
    rows = stage11.pair_video('v', rally_spans, chunks, replay_mask, FPS)
    assert rows[0]['chunk_id'] == ''                     # held out despite the available chunk


def test_chunk_claimed_by_earlier_of_two_rallies():
    rally_spans = [(0, 0, 100), (1, 110, 120)]           # windows (10, 18] and (12, 20]
    chunks = [_chunk('c0', 13.0, 16.0)]                  # start 13.0 falls in both
    rows = stage11.pair_video('v', rally_spans, chunks, None, FPS)
    by_id = {row['rally_id']: row for row in rows}
    assert by_id[0]['chunk_id'] == 'c0'                  # earlier rally claims it
    assert by_id[1]['chunk_id'] == ''                    # later rally left unpaired


class _FakeCapture:
    """Stand-in for cv2.VideoCapture returning a fixed fps, no real decode."""

    def __init__(self, path: str):
        self.path = path

    def get(self, prop: int) -> float:
        return 30.0

    def release(self) -> None:
        pass


def test_build_video_fps_csv(tmp_path, monkeypatch):
    video_dir = tmp_path / 'videos'
    video_dir.mkdir()
    (video_dir / 'vid1.mp4').write_bytes(b'')
    (video_dir / 'vid2.mkv').write_bytes(b'')
    (video_dir / 'notes.txt').write_bytes(b'')           # non-video, ignored

    monkeypatch.setattr(stage11.cv2, 'VideoCapture', _FakeCapture)
    out_csv = tmp_path / 'video_fps.csv'
    stage11.build_video_fps_csv(video_dir, out_csv)

    with out_csv.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    fps_by_id = {row['video_id']: float(row['fps']) for row in rows}
    assert fps_by_id == {'vid1': 30.0, 'vid2': 30.0}
