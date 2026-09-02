"""Commentary-pairing tests for joins, replay exclusions, and the FPS sidecar.

fps is fixed at 10 so frame/second arithmetic is easy to read: end_frame 100 is
10.0 s, and the pairing window is `(10.0, 10.0 + PAIR_WINDOW_S]`.
"""
import csv
import json
import logging

import numpy as np
import pytest

from src.scraper import commentary_pairing
from src.scraper.config import PAIR_WINDOW_S

FPS = 10.0


def _chunk(chunk_id: str, start: float, end: float) -> dict:
    return {'chunk_id': chunk_id, 'start': start, 'end': end, 'text': 'commentary'}


def _invoke_commentary_pairing(
    monkeypatch,
    *,
    video_dir,
    fps_csv,
    spans_csv,
    pairs_csv,
    chunks_dir=None,
    masks_dir=None,
    build_fps_from=None,
) -> None:
    argv = [
        'commentary_pairing',
        '--fps-csv', str(fps_csv),
        '--rally-spans', str(spans_csv),
        '--pairs-csv', str(pairs_csv),
    ]
    if chunks_dir is not None:
        argv.extend(['--chunks-dir', str(chunks_dir)])
    if masks_dir is not None:
        argv.extend(['--masks-dir', str(masks_dir)])
    if video_dir is not None:
        argv.extend(['--video-dir', str(video_dir)])
    if build_fps_from is not None:
        argv.extend(['--build-fps-from', str(build_fps_from)])
    monkeypatch.setattr('sys.argv', argv)
    commentary_pairing.main()


def _pair_rows(path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def test_chunk_within_window_pairs():
    rally_spans = [(0, 0, 100)]                          # rally_end_t = 10.0 s
    chunks = [_chunk('c0', 11.0, 15.0)]                  # start inside (10, 18]
    rows = commentary_pairing.pair_video('v', rally_spans, chunks, None, FPS)
    assert len(rows) == 1
    assert rows[0]['chunk_id'] == 'c0'
    assert rows[0]['commentary_start'] == 11.0
    assert rows[0]['commentary_end'] == 15.0


def test_chunk_outside_window_leaves_unpaired_blanks():
    rally_spans = [(0, 0, 100)]
    late = 10.0 + PAIR_WINDOW_S + 5.0                    # past the window
    rows = commentary_pairing.pair_video('v', rally_spans, [_chunk('c0', late, late + 2)], None, FPS)
    assert rows[0]['chunk_id'] == ''
    assert rows[0]['commentary_start'] == ''
    assert rows[0]['commentary_end'] == ''


def test_chunk_start_stray_flag_is_cleared_so_the_chunk_pairs():
    rally_spans = [(0, 0, 100)]                          # span frames 0..100, kept clear of the mask
    chunks = [_chunk('c0', 11.0, 15.0)]                  # start frame = int(11.0 * 10) = 110
    replay_mask = np.zeros(400, dtype=bool)
    replay_mask[110] = True
    rows = commentary_pairing.pair_video('v', rally_spans, chunks, replay_mask, FPS)
    assert rows[0]['chunk_id'] == 'c0'                    # the one-frame stray is not believed


def test_rally_overlapping_replay_is_held_out():
    rally_spans = [(0, 0, 100)]
    chunks = [_chunk('c0', 11.0, 15.0)]                  # a valid chunk exists
    replay_mask = np.zeros(400, dtype=bool)
    replay_mask[50:55] = True                            # sustained at 10 fps (window = 5)
    rows = commentary_pairing.pair_video('v', rally_spans, chunks, replay_mask, FPS)
    assert rows[0]['chunk_id'] == ''                     # held out despite the available chunk


def test_rally_with_short_mask_run_remains_pairable():
    rally_spans = [(0, 0, 100)]
    replay_mask = np.zeros(400, dtype=bool)
    replay_mask[50:54] = True                            # one below the 5-frame window
    rows = commentary_pairing.pair_video('v', rally_spans, [_chunk('c0', 11.0, 15.0)], replay_mask, FPS)
    assert rows[0]['chunk_id'] == 'c0'


def test_two_short_mask_runs_do_not_accumulate():
    rally_spans = [(0, 0, 100)]
    replay_mask = np.zeros(400, dtype=bool)
    replay_mask[40:43] = True
    replay_mask[60:63] = True
    rows = commentary_pairing.pair_video('v', rally_spans, [_chunk('c0', 11.0, 15.0)], replay_mask, FPS)
    assert rows[0]['chunk_id'] == 'c0'


def test_mask_run_at_span_edge_is_trusted_from_the_full_video_run():
    replay_mask = np.zeros(20, dtype=bool)
    replay_mask[0:5] = True
    rows = commentary_pairing.pair_video(
        'v', [(0, 2, 5)], [_chunk('c0', 1.0, 2.0)], replay_mask, FPS,
    )
    assert rows[0]['chunk_id'] == 'c0'  # the believed run sits wholly in boundary grace


def test_duration_filtered_replay_only_disqualifies_a_rally_interior():
    duration_filtered_replay_mask = np.zeros(30, dtype=bool)
    duration_filtered_replay_mask[0:5] = True
    duration_filtered_replay_mask[6] = True

    assert not commentary_pairing._believed_replay_in_rally_interior(
        duration_filtered_replay_mask, 0, 5, grace=5,
    )
    assert commentary_pairing._believed_replay_in_rally_interior(
        duration_filtered_replay_mask, 0, 20, grace=5,
    )
    assert not commentary_pairing._believed_replay_in_rally_interior(
        duration_filtered_replay_mask, 0, 10, grace=5,
    )


def test_none_empty_and_all_false_masks_do_not_hold_out():
    rally_spans = [(0, 0, 100)]
    chunks = [_chunk('c0', 11.0, 15.0)]
    rows = commentary_pairing.pair_video('v', rally_spans, chunks, None, FPS)
    assert rows[0]['chunk_id'] == 'c0'
    rows = commentary_pairing.pair_video('v', [(0, 0, 0)], [_chunk('c0', 1.0, 2.0)], np.zeros(0, dtype=bool), FPS)
    assert rows[0]['chunk_id'] == 'c0'
    rows = commentary_pairing.pair_video('v', rally_spans, chunks, np.zeros(400, dtype=bool), FPS)
    assert rows[0]['chunk_id'] == 'c0'


def test_all_true_mask_holds_out():
    replay_mask = np.ones(400, dtype=bool)
    rows = commentary_pairing.pair_video('v', [(0, 0, 100)], [_chunk('c0', 11.0, 15.0)], replay_mask, FPS)
    assert rows[0]['chunk_id'] == ''


def test_rally_endpoint_beyond_mask_fails_loudly():
    with pytest.raises(ValueError, match='outside replay mask'):
        commentary_pairing.pair_video('v', [(0, 0, 101)], [], np.zeros(100, dtype=bool), FPS)


def test_load_replay_mask_requires_one_dimensional_boolean_array(tmp_path):
    masks_dir = tmp_path / 'masks'
    masks_dir.mkdir()
    np.save(masks_dir / 'bad_shape_replay.npy', np.zeros((2, 2), dtype=bool))
    np.save(masks_dir / 'bad_type_replay.npy', np.zeros(4, dtype=np.uint8))
    with pytest.raises(ValueError, match='one-dimensional boolean'):
        commentary_pairing._load_replay_mask(masks_dir, 'bad_shape')
    with pytest.raises(ValueError, match='one-dimensional boolean'):
        commentary_pairing._load_replay_mask(masks_dir, 'bad_type')


def test_chunk_claimed_by_earlier_of_two_rallies():
    rally_spans = [(0, 0, 100), (1, 110, 120)]           # windows (10, 18] and (12, 20]
    chunks = [_chunk('c0', 13.0, 16.0)]                  # start 13.0 falls in both
    rows = commentary_pairing.pair_video('v', rally_spans, chunks, None, FPS)
    by_id = {row['rally_id']: row for row in rows}
    assert by_id[0]['chunk_id'] == 'c0'                  # earlier rally claims it
    assert by_id[1]['chunk_id'] == ''                    # later rally left unpaired


def test_shorter_window_can_leave_chunk_for_later_rally():
    rally_spans = [(0, 0, 1), (1, 10, 50)]
    chunks = [_chunk('c0', 6.0, 6.5)]

    eight_second = commentary_pairing.pair_video(
        'v', rally_spans, chunks, None, FPS, pair_window_s=8.0,
    )
    five_second = commentary_pairing.pair_video(
        'v', rally_spans, chunks, None, FPS, pair_window_s=5.0,
    )

    assert [row['chunk_id'] for row in eight_second] == ['c0', '']
    assert [row['chunk_id'] for row in five_second] == ['', 'c0']


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

    monkeypatch.setattr(commentary_pairing.cv2, 'VideoCapture', _FakeCapture)
    out_csv = tmp_path / 'video_fps.csv'
    commentary_pairing.build_video_fps_csv(video_dir, out_csv)

    with out_csv.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    fps_by_id = {row['video_id']: float(row['fps']) for row in rows}
    assert fps_by_id == {'vid1': 30.0, 'vid2': 30.0}


def test_commentary_ineligible_video_pairs_with_blank_commentary(tmp_path, monkeypatch):
    video_dir = tmp_path / 'videos'
    video_dir.mkdir()
    (video_dir / 'v.mp4').write_bytes(b'video')
    (video_dir / 'sources.toml').write_text(
        'dataset = "shuttleset"\n\n'
        '[videos."v.mp4"]\n'
        'video_id = "v"\n'
        'commentary_eligible = false\n',
        encoding='utf-8',
    )
    fps_csv = tmp_path / 'fps.csv'
    fps_csv.write_text('video_id,fps\nv,10\n', encoding='utf-8')
    spans_csv = tmp_path / 'spans.csv'
    spans_csv.write_text(
        'video_id,rally_id,start_frame,end_frame\n'
        'v,0,0,100\n',
        encoding='utf-8',
    )
    chunks_dir = tmp_path / 'chunks'
    chunks_dir.mkdir()
    (chunks_dir / 'v.json').write_text(
        json.dumps([{'chunk_id': 'c0', 'start': 11.0, 'end': 15.0, 'text': 'spoken'}]),
        encoding='utf-8',
    )
    pairs_csv = tmp_path / 'pairs.csv'

    monkeypatch.setattr(
        'sys.argv',
        [
            'commentary_pairing',
            '--video-dir', str(video_dir),
            '--fps-csv', str(fps_csv),
            '--rally-spans', str(spans_csv),
            '--chunks-dir', str(chunks_dir),
            '--masks-dir', str(tmp_path / 'masks'),
            '--pairs-csv', str(pairs_csv),
        ],
    )

    commentary_pairing.main()

    with pairs_csv.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]['chunk_id'] == ''
    assert rows[0]['commentary_start'] == ''
    assert rows[0]['commentary_end'] == ''


def test_commentary_eligible_video_pairs_normally(tmp_path, monkeypatch):
    video_dir = tmp_path / 'videos'
    video_dir.mkdir()
    (video_dir / 'v.mp4').write_bytes(b'video')
    (video_dir / 'sources.toml').write_text(
        'dataset = "scraped"\n\n'
        '[videos."v.mp4"]\n'
        'video_id = "v"\n'
        'commentary_eligible = true\n',
        encoding='utf-8',
    )
    fps_csv = tmp_path / 'fps.csv'
    fps_csv.write_text('video_id,fps\nv,10\n', encoding='utf-8')
    spans_csv = tmp_path / 'spans.csv'
    spans_csv.write_text(
        'video_id,rally_id,start_frame,end_frame\n'
        'v,0,0,100\n',
        encoding='utf-8',
    )
    chunks_dir = tmp_path / 'chunks'
    chunks_dir.mkdir()
    (chunks_dir / 'v.json').write_text(
        json.dumps([_chunk('c0', 11.0, 15.0)]),
        encoding='utf-8',
    )
    pairs_csv = tmp_path / 'pairs.csv'

    _invoke_commentary_pairing(
        monkeypatch,
        video_dir=video_dir,
        fps_csv=fps_csv,
        spans_csv=spans_csv,
        pairs_csv=pairs_csv,
        chunks_dir=chunks_dir,
        masks_dir=tmp_path / 'masks',
    )

    assert _pair_rows(pairs_csv)[0]['chunk_id'] == 'c0'


def test_commentary_eligible_video_logs_missing_sidecars(
    tmp_path, monkeypatch, caplog,
):
    video_dir = tmp_path / 'videos'
    video_dir.mkdir()
    (video_dir / 'v.mp4').write_bytes(b'video')
    (video_dir / 'sources.toml').write_text(
        'dataset = "scraped"\n\n'
        '[videos."v.mp4"]\n'
        'video_id = "v"\n'
        'commentary_eligible = true\n',
        encoding='utf-8',
    )
    fps_csv = tmp_path / 'fps.csv'
    fps_csv.write_text('video_id,fps\nv,10\n', encoding='utf-8')
    spans_csv = tmp_path / 'spans.csv'
    spans_csv.write_text(
        'video_id,rally_id,start_frame,end_frame\n'
        'v,0,0,100\n',
        encoding='utf-8',
    )
    chunks_dir = tmp_path / 'chunks'
    masks_dir = tmp_path / 'masks'
    pairs_csv = tmp_path / 'pairs.csv'
    caplog.set_level(logging.INFO)

    _invoke_commentary_pairing(
        monkeypatch,
        video_dir=video_dir,
        fps_csv=fps_csv,
        spans_csv=spans_csv,
        pairs_csv=pairs_csv,
        chunks_dir=chunks_dir,
        masks_dir=masks_dir,
    )

    assert f'commentary-eligible but chunks sidecar is missing: {chunks_dir / "v.json"}' in caplog.text
    assert f'replay mask is missing; pairing without replay filtering: {masks_dir / "v_replay.npy"}' in caplog.text
    assert _pair_rows(pairs_csv)[0]['chunk_id'] == ''


def test_commentary_ineligible_video_leaves_every_rally_blank(tmp_path, monkeypatch):
    video_dir = tmp_path / 'videos'
    video_dir.mkdir()
    (video_dir / 'v.mp4').write_bytes(b'video')
    (video_dir / 'sources.toml').write_text(
        'dataset = "scraped"\n\n'
        '[videos."v.mp4"]\n'
        'video_id = "v"\n'
        'commentary_eligible = false\n',
        encoding='utf-8',
    )
    fps_csv = tmp_path / 'fps.csv'
    fps_csv.write_text('video_id,fps\nv,10\n', encoding='utf-8')
    spans_csv = tmp_path / 'spans.csv'
    spans_csv.write_text(
        'video_id,rally_id,start_frame,end_frame\n'
        'v,0,0,100\n'
        'v,1,200,300\n',
        encoding='utf-8',
    )
    chunks_dir = tmp_path / 'chunks'
    chunks_dir.mkdir()
    (chunks_dir / 'v.json').write_text(
        json.dumps([_chunk('c0', 11.0, 15.0), _chunk('c1', 31.0, 35.0)]),
        encoding='utf-8',
    )
    pairs_csv = tmp_path / 'pairs.csv'

    _invoke_commentary_pairing(
        monkeypatch,
        video_dir=video_dir,
        fps_csv=fps_csv,
        spans_csv=spans_csv,
        pairs_csv=pairs_csv,
        chunks_dir=chunks_dir,
        masks_dir=tmp_path / 'masks',
    )

    rows = _pair_rows(pairs_csv)
    assert len(rows) == 2
    assert all(row['chunk_id'] == '' for row in rows)
    assert all(row['commentary_start'] == '' for row in rows)
    assert all(row['commentary_end'] == '' for row in rows)


def test_ineligible_video_does_not_load_malformed_replay_mask(tmp_path, monkeypatch):
    video_dir = tmp_path / 'videos'
    video_dir.mkdir()
    (video_dir / 'v.mp4').write_bytes(b'video')
    (video_dir / 'sources.toml').write_text(
        'dataset = "scraped"\n\n'
        '[videos."v.mp4"]\n'
        'video_id = "v"\n'
        'commentary_eligible = false\n',
        encoding='utf-8',
    )
    fps_csv = tmp_path / 'fps.csv'
    fps_csv.write_text('video_id,fps\nv,10\n', encoding='utf-8')
    spans_csv = tmp_path / 'spans.csv'
    spans_csv.write_text(
        'video_id,rally_id,start_frame,end_frame\n'
        'v,0,0,100\n',
        encoding='utf-8',
    )
    masks_dir = tmp_path / 'masks'
    masks_dir.mkdir()
    np.save(masks_dir / 'v_replay.npy', np.zeros((2, 2), dtype=np.uint8))
    pairs_csv = tmp_path / 'pairs.csv'

    def fail_load(_masks_dir, _video_id):
        raise AssertionError('ineligible video must not load replay mask')

    monkeypatch.setattr(commentary_pairing, '_load_replay_mask', fail_load)
    _invoke_commentary_pairing(
        monkeypatch,
        video_dir=video_dir,
        fps_csv=fps_csv,
        spans_csv=spans_csv,
        pairs_csv=pairs_csv,
        masks_dir=masks_dir,
    )

    assert _pair_rows(pairs_csv)[0]['chunk_id'] == ''


def test_missing_manifest_fails_for_video_with_fps(tmp_path, monkeypatch):
    video_dir = tmp_path / 'videos'
    video_dir.mkdir()
    (video_dir / 'v.mp4').write_bytes(b'video')
    fps_csv = tmp_path / 'fps.csv'
    fps_csv.write_text('video_id,fps\nv,10\n', encoding='utf-8')
    spans_csv = tmp_path / 'spans.csv'
    spans_csv.write_text(
        'video_id,rally_id,start_frame,end_frame\n'
        'v,0,0,100\n',
        encoding='utf-8',
    )
    pairs_csv = tmp_path / 'pairs.csv'

    with pytest.raises(FileNotFoundError, match='sources.toml'):
        _invoke_commentary_pairing(
            monkeypatch,
            video_dir=video_dir,
            fps_csv=fps_csv,
            spans_csv=spans_csv,
            pairs_csv=pairs_csv,
        )
    assert not pairs_csv.exists()


@pytest.mark.parametrize(
    ('manifest_text', 'video_names', 'error_type'),
    [
        (
            'dataset = "scraped"\n\n'
            '[videos."other.mp4"]\n'
            'video_id = "other"\n'
            'commentary_eligible = true\n',
            ['v.mp4'],
            ValueError,
        ),
        (
            'dataset = "scraped"\n\n'
            '[videos."v.mp4"]\n'
            'video_id = "v"\n'
            'commentary_eligible = true\n\n'
            '[videos."v.mkv"]\n'
            'video_id = "v"\n'
            'commentary_eligible = true\n',
            ['v.mp4', 'v.mkv'],
            ValueError,
        ),
        (
            'dataset = "scraped"\n\n'
            '[videos."v.mp4"]\n'
            'video_id = "v"\n',
            ['v.mp4'],
            ValueError,
        ),
        (
            'dataset = "scraped"\n\n'
            '[videos."v.mp4"]\n'
            'video_id = "v"\n'
            'commentary_eligible = "true"\n',
            ['v.mp4'],
            TypeError,
        ),
        (
            'dataset = 7\n\n[videos]\n',
            ['v.mp4'],
            TypeError,
        ),
        (
            'dataset = ""\n\n[videos]\n',
            ['v.mp4'],
            ValueError,
        ),
    ],
)
def test_strict_manifest_errors_happen_before_pairs_csv_write(
    tmp_path,
    monkeypatch,
    manifest_text,
    video_names,
    error_type,
):
    video_dir = tmp_path / 'videos'
    video_dir.mkdir()
    for video_name in video_names:
        (video_dir / video_name).write_bytes(b'video')
    (video_dir / 'sources.toml').write_text(manifest_text, encoding='utf-8')
    fps_csv = tmp_path / 'fps.csv'
    fps_csv.write_text('video_id,fps\nv,10\n', encoding='utf-8')
    spans_csv = tmp_path / 'spans.csv'
    spans_csv.write_text(
        'video_id,rally_id,start_frame,end_frame\n'
        'v,0,0,100\n',
        encoding='utf-8',
    )
    pairs_csv = tmp_path / 'pairs.csv'
    pairs_csv.write_text('sentinel\n', encoding='utf-8')

    with pytest.raises(error_type):
        _invoke_commentary_pairing(
            monkeypatch,
            video_dir=video_dir,
            fps_csv=fps_csv,
            spans_csv=spans_csv,
            pairs_csv=pairs_csv,
        )
    assert pairs_csv.read_text(encoding='utf-8') == 'sentinel\n'


def test_manifest_extra_keys_and_entry_without_video_id_are_tolerated(tmp_path, monkeypatch):
    video_dir = tmp_path / 'videos'
    video_dir.mkdir()
    (video_dir / 'v.mp4').write_bytes(b'video')
    (video_dir / 'sources.toml').write_text(
        'dataset = "scraped"\n\n'
        '[videos."notes.toml"]\n'
        'description = "not a video"\n'
        'future_flag = true\n\n'
        '[videos."v.mp4"]\n'
        'video_id = "v"\n'
        'commentary_eligible = true\n'
        'future_flag = true\n',
        encoding='utf-8',
    )
    fps_csv = tmp_path / 'fps.csv'
    fps_csv.write_text('video_id,fps\nv,10\n', encoding='utf-8')
    spans_csv = tmp_path / 'spans.csv'
    spans_csv.write_text(
        'video_id,rally_id,start_frame,end_frame\n'
        'v,0,0,100\n',
        encoding='utf-8',
    )
    chunks_dir = tmp_path / 'chunks'
    chunks_dir.mkdir()
    (chunks_dir / 'v.json').write_text(
        json.dumps([_chunk('c0', 11.0, 15.0)]),
        encoding='utf-8',
    )
    pairs_csv = tmp_path / 'pairs.csv'

    _invoke_commentary_pairing(
        monkeypatch,
        video_dir=video_dir,
        fps_csv=fps_csv,
        spans_csv=spans_csv,
        pairs_csv=pairs_csv,
        chunks_dir=chunks_dir,
    )

    assert _pair_rows(pairs_csv)[0]['chunk_id'] == 'c0'


def test_stale_missing_manifest_basename_is_ignored(tmp_path, monkeypatch):
    video_dir = tmp_path / 'videos'
    video_dir.mkdir()
    (video_dir / 'v.mp4').write_bytes(b'video')
    (video_dir / 'sources.toml').write_text(
        'dataset = "scraped"\n\n'
        '[videos."stale.mp4"]\n'
        'video_id = "stale"\n'
        'commentary_eligible = true\n\n'
        '[videos."v.mp4"]\n'
        'video_id = "v"\n'
        'commentary_eligible = true\n',
        encoding='utf-8',
    )
    fps_csv = tmp_path / 'fps.csv'
    fps_csv.write_text('video_id,fps\nv,10\n', encoding='utf-8')
    spans_csv = tmp_path / 'spans.csv'
    spans_csv.write_text(
        'video_id,rally_id,start_frame,end_frame\n'
        'v,0,0,100\n',
        encoding='utf-8',
    )
    pairs_csv = tmp_path / 'pairs.csv'

    _invoke_commentary_pairing(
        monkeypatch,
        video_dir=video_dir,
        fps_csv=fps_csv,
        spans_csv=spans_csv,
        pairs_csv=pairs_csv,
    )

    assert _pair_rows(pairs_csv)[0]['chunk_id'] == ''


def test_manifest_mapping_requires_existing_video_file(tmp_path, monkeypatch):
    video_dir = tmp_path / 'videos'
    video_dir.mkdir()
    (video_dir / 'sources.toml').write_text(
        'dataset = "scraped"\n\n'
        '[videos."v.mp4"]\n'
        'video_id = "v"\n'
        'commentary_eligible = true\n',
        encoding='utf-8',
    )
    fps_csv = tmp_path / 'fps.csv'
    fps_csv.write_text('video_id,fps\nv,10\n', encoding='utf-8')
    spans_csv = tmp_path / 'spans.csv'
    spans_csv.write_text(
        'video_id,rally_id,start_frame,end_frame\n'
        'v,0,0,100\n',
        encoding='utf-8',
    )
    pairs_csv = tmp_path / 'pairs.csv'

    with pytest.raises(ValueError, match='no existing manifest entry'):
        _invoke_commentary_pairing(
            monkeypatch,
            video_dir=video_dir,
            fps_csv=fps_csv,
            spans_csv=spans_csv,
            pairs_csv=pairs_csv,
        )


def test_different_video_and_fps_build_directories_fail_loudly(tmp_path, monkeypatch):
    video_dir = tmp_path / 'videos'
    build_dir = tmp_path / 'other-videos'
    video_dir.mkdir()
    build_dir.mkdir()
    fps_csv = tmp_path / 'fps.csv'
    spans_csv = tmp_path / 'spans.csv'
    pairs_csv = tmp_path / 'pairs.csv'

    with pytest.raises(ValueError, match='must resolve to the same path'):
        _invoke_commentary_pairing(
            monkeypatch,
            video_dir=video_dir,
            build_fps_from=build_dir,
            fps_csv=fps_csv,
            spans_csv=spans_csv,
            pairs_csv=pairs_csv,
        )


def test_video_dir_argument_alone_selects_manifest_directory(tmp_path, monkeypatch):
    video_dir = tmp_path / 'videos'
    video_dir.mkdir()
    (video_dir / 'v.mp4').write_bytes(b'video')
    (video_dir / 'sources.toml').write_text(
        'dataset = "scraped"\n\n'
        '[videos."v.mp4"]\n'
        'video_id = "v"\n'
        'commentary_eligible = false\n',
        encoding='utf-8',
    )
    fps_csv = tmp_path / 'fps.csv'
    fps_csv.write_text('video_id,fps\nv,10\n', encoding='utf-8')
    spans_csv = tmp_path / 'spans.csv'
    spans_csv.write_text(
        'video_id,rally_id,start_frame,end_frame\n'
        'v,0,0,100\n',
        encoding='utf-8',
    )
    pairs_csv = tmp_path / 'pairs.csv'

    _invoke_commentary_pairing(
        monkeypatch,
        video_dir=video_dir,
        fps_csv=fps_csv,
        spans_csv=spans_csv,
        pairs_csv=pairs_csv,
    )

    assert len(_pair_rows(pairs_csv)) == 1


def test_build_fps_from_argument_alone_selects_manifest_directory(tmp_path, monkeypatch):
    video_dir = tmp_path / 'videos'
    video_dir.mkdir()
    (video_dir / 'v.mp4').write_bytes(b'video')
    (video_dir / 'sources.toml').write_text(
        'dataset = "scraped"\n\n'
        '[videos."v.mp4"]\n'
        'video_id = "v"\n'
        'commentary_eligible = false\n',
        encoding='utf-8',
    )
    spans_csv = tmp_path / 'spans.csv'
    spans_csv.write_text(
        'video_id,rally_id,start_frame,end_frame\n'
        'v,0,0,100\n',
        encoding='utf-8',
    )
    fps_csv = tmp_path / 'fps.csv'
    pairs_csv = tmp_path / 'pairs.csv'
    monkeypatch.setattr(commentary_pairing.cv2, 'VideoCapture', _FakeCapture)

    _invoke_commentary_pairing(
        monkeypatch,
        video_dir=None,
        build_fps_from=video_dir,
        fps_csv=fps_csv,
        spans_csv=spans_csv,
        pairs_csv=pairs_csv,
    )

    assert _pair_rows(pairs_csv)[0]['video_id'] == 'v'


def test_missing_fps_skips_without_reading_manifest(tmp_path, monkeypatch, caplog):
    video_dir = tmp_path / 'videos'
    video_dir.mkdir()
    fps_csv = tmp_path / 'fps.csv'
    fps_csv.write_text('video_id,fps\n', encoding='utf-8')
    spans_csv = tmp_path / 'spans.csv'
    spans_csv.write_text(
        'video_id,rally_id,start_frame,end_frame\n'
        'unlisted,0,0,100\n',
        encoding='utf-8',
    )
    pairs_csv = tmp_path / 'pairs.csv'

    monkeypatch.setattr(
        'sys.argv',
        [
            'commentary_pairing',
            '--video-dir', str(video_dir),
            '--fps-csv', str(fps_csv),
            '--rally-spans', str(spans_csv),
            '--pairs-csv', str(pairs_csv),
        ],
    )

    commentary_pairing.main()

    assert 'no fps for unlisted; skipping its rallies' in caplog.text
    with pairs_csv.open(newline='', encoding='utf-8') as handle:
        assert list(csv.DictReader(handle)) == []
