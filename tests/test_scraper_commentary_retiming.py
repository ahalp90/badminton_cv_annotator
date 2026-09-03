"""Commentary re-timing tests: word matching, coarse fallbacks, ordering, and the CLI.

The stream is 80 distinct tokens ``w0`` to ``w79`` with the common pair "of the"
planted at words 25 and 26. Every word lasts 0.4 s with a 0.1 s gap, so word ``i``
starts at ``0.5 * i``.
"""
import json

import pytest

from src.scraper import commentary_retiming as retiming

WORDS = [f'w{index}' for index in range(80)]
WORDS[25:27] = ['of', 'the']


def _aligned_transcript(words=WORDS, untimed: set[int] = frozenset()) -> dict:
    entries = []
    for index, word in enumerate(words):
        entry = {'word': word}
        if index not in untimed:
            entry['start'] = 0.5 * index
            entry['end'] = 0.5 * index + 0.4
        entries.append(entry)
    return {
        'source': retiming.ALIGNED_SOURCE,
        'segments': [{'start': 0.0, 'end': 0.5 * len(words), 'text': ' '.join(words), 'words': entries}],
    }


def _chunk(chunk_id: str, start: float, end: float, text: str) -> dict:
    return {'chunk_id': chunk_id, 'start': start, 'end': end, 'text': text, 'text_clean': text}


def _stream(**kwargs) -> retiming.WordStream:
    return retiming.load_word_stream(_aligned_transcript(**kwargs))


def test_exact_text_snaps_to_word_times():
    phrase = ' '.join(WORDS[10:20])  # words 10..19 start at 5.0 s and end at 9.9 s
    rows = retiming.retime_chunks([_chunk('v_c0', 30.0, 40.0, phrase)], _stream())
    row = rows[0]
    assert row['align_status'] == retiming.AlignStatus.ALIGNED
    assert row['start'] == pytest.approx(5.0)
    assert row['end'] == pytest.approx(9.9)
    assert row['align_match_ratio'] == pytest.approx(1.0)
    assert row['align_shift_s'] == pytest.approx(-25.0)
    assert (row['coarse_start'], row['coarse_end']) == (30.0, 40.0)


def test_noisy_text_still_matches_with_lower_ratio():
    words = list(WORDS[10:22])
    words[3] = 'shuttlecock'
    words[8] = 'ended'
    rows = retiming.retime_chunks([_chunk('v_c0', 5.0, 11.0, ' '.join(words))], _stream())
    assert rows[0]['align_status'] == retiming.AlignStatus.ALIGNED
    assert rows[0]['start'] == pytest.approx(5.0)
    assert retiming.MIN_MATCH_RATIO <= rows[0]['align_match_ratio'] < 1.0


def test_unrelated_text_keeps_coarse_times():
    rows = retiming.retime_chunks([_chunk('v_c0', 5.0, 11.0, 'zebra quantum lattice')], _stream())
    assert rows[0]['align_status'] == retiming.AlignStatus.UNMATCHED
    assert (rows[0]['start'], rows[0]['end']) == (5.0, 11.0)
    assert rows[0]['align_match_ratio'] is None


def test_far_common_words_do_not_stretch_the_span():
    # The chunk opens with "of the", which the stream only has at words 25..26, 30 words
    # before the real phrase at 57..63. That stray block must not pull the start back.
    phrase = ' '.join(['of', 'the', *WORDS[57:64]])
    assert WORDS[25:27] == ['of', 'the']
    rows = retiming.retime_chunks([_chunk('v_c0', 20.0, 40.0, phrase)], _stream())
    assert rows[0]['align_status'] == retiming.AlignStatus.ALIGNED
    assert rows[0]['start'] == pytest.approx(0.5 * 57)
    assert rows[0]['end'] == pytest.approx(0.5 * 63 + 0.4)


def test_window_excludes_words_outside_the_search_pad():
    phrase = ' '.join(WORDS[10:20])  # starts at 5.0 s
    far_start = 5.0 + retiming.SEARCH_PAD_S + 30.0
    rows = retiming.retime_chunks([_chunk('v_c0', far_start, far_start + 5.0, phrase)], _stream())
    assert rows[0]['align_status'] == retiming.AlignStatus.UNMATCHED


def test_untimed_edge_words_take_times_from_neighbours():
    stream = _stream(untimed={10, 19})
    rows = retiming.retime_chunks([_chunk('v_c0', 5.0, 10.0, ' '.join(WORDS[10:20]))], stream)
    assert rows[0]['start'] == pytest.approx(0.5 * 11)
    assert rows[0]['end'] == pytest.approx(0.5 * 18 + 0.4)


def test_below_floor_match_keeps_coarse_but_records_ratio():
    # Two shared words out of twelve: a block exists, but the ratio is far below the floor.
    words = ['zebra'] * 10 + list(WORDS[10:12])
    rows = retiming.retime_chunks([_chunk('v_c0', 5.0, 11.0, ' '.join(words))], _stream())
    assert rows[0]['align_status'] == retiming.AlignStatus.UNMATCHED
    assert (rows[0]['start'], rows[0]['end']) == (5.0, 11.0)
    assert 0 < rows[0]['align_match_ratio'] < retiming.MIN_MATCH_RATIO


def test_span_without_timed_words_keeps_coarse():
    stream = _stream(untimed=set(range(10, 20)))
    rows = retiming.retime_chunks([_chunk('v_c0', 5.0, 10.0, ' '.join(WORDS[10:20]))], stream)
    assert rows[0]['align_status'] == retiming.AlignStatus.UNMATCHED
    assert rows[0]['align_match_ratio'] == pytest.approx(1.0)


def test_aligned_start_yields_to_another_chunks_coarse_start():
    # v_c1 keeps its coarse 5.0 s (unmatched), which is exactly where v_c0 would align.
    chunks = [
        _chunk('v_c0', 30.0, 40.0, ' '.join(WORDS[10:20])),
        _chunk('v_c1', 5.0, 6.0, 'zebra quantum lattice'),
    ]
    rows = retiming.retime_chunks(chunks, _stream())
    by_id = {row['chunk_id']: row for row in rows}
    assert by_id['v_c1']['align_status'] == retiming.AlignStatus.UNMATCHED
    assert by_id['v_c0']['align_status'] == retiming.AlignStatus.COLLISION
    assert by_id['v_c0']['start'] == 30.0
    assert [row['chunk_id'] for row in rows] == ['v_c1', 'v_c0']


def test_better_match_wins_a_shared_aligned_start():
    noisy = list(WORDS[10:20])
    noisy[5] = 'shuttlecock'
    chunks = [
        _chunk('v_c0', 30.0, 35.0, ' '.join(noisy)),  # ratio 0.9, listed first
        _chunk('v_c1', 31.0, 36.0, ' '.join(WORDS[10:20])),  # ratio 1.0
    ]
    rows = retiming.retime_chunks(chunks, _stream())
    by_id = {row['chunk_id']: row for row in rows}
    assert by_id['v_c1']['align_status'] == retiming.AlignStatus.ALIGNED
    assert by_id['v_c0']['align_status'] == retiming.AlignStatus.COLLISION
    assert by_id['v_c0']['start'] == 30.0


def test_duplicate_coarse_starts_are_rejected():
    chunks = [_chunk('v_c0', 5.0, 6.0, 'zebra'), _chunk('v_c1', 5.0, 7.0, 'quantum')]
    with pytest.raises(ValueError, match='duplicate coarse starts'):
        retiming.retime_chunks(chunks, _stream())


def test_quantiles_use_nearest_rank():
    assert retiming._quantiles([float(value) for value in range(1, 11)])['p90'] == 9.0
    assert retiming._quantiles([]) is None


def test_duplicate_aligned_start_marks_second_chunk_as_collision():
    phrase = ' '.join(WORDS[10:20])
    chunks = [_chunk('v_c0', 5.0, 10.0, phrase), _chunk('v_c1', 6.0, 11.0, phrase)]
    rows = retiming.retime_chunks(chunks, _stream())
    by_id = {row['chunk_id']: row for row in rows}
    assert by_id['v_c0']['align_status'] == retiming.AlignStatus.ALIGNED
    assert by_id['v_c1']['align_status'] == retiming.AlignStatus.COLLISION
    assert (by_id['v_c1']['start'], by_id['v_c1']['end']) == (6.0, 11.0)


def test_rows_come_back_sorted_by_aligned_start():
    later = _chunk('v_c0', 40.0, 45.0, ' '.join(WORDS[40:48]))  # aligned to 20.0 s
    earlier = _chunk('v_c1', 50.0, 55.0, ' '.join(WORDS[10:18]))  # aligned to 5.0 s
    rows = retiming.retime_chunks([later, earlier], _stream())
    assert [row['chunk_id'] for row in rows] == ['v_c1', 'v_c0']
    assert rows[0]['start'] < rows[1]['start']


def test_load_word_stream_rejects_other_sources():
    with pytest.raises(ValueError, match='whisperx_aligned'):
        retiming.load_word_stream({'source': 'whisper', 'segments': []})


def test_normalize_tokens_splits_punctuation_and_keeps_apostrophes():
    assert retiming.normalize_tokens("He didn't -- commit, going forward!") == [
        'he', "didn't", 'commit', 'going', 'forward',
    ]


def test_cli_writes_sidecars_and_summary(tmp_path, monkeypatch):
    aligned_dir = tmp_path / 'aligned'
    cleaned_dir = tmp_path / 'cleaned'
    aligned_dir.mkdir()
    cleaned_dir.mkdir()
    (aligned_dir / 'v.json').write_text(json.dumps(_aligned_transcript()), encoding='utf-8')
    chunks = [_chunk('v_c0', 30.0, 40.0, ' '.join(WORDS[10:20])), _chunk('v_c1', 60.0, 70.0, 'zebra quantum')]
    (cleaned_dir / 'v.json').write_text(json.dumps(chunks), encoding='utf-8')
    out_dir = tmp_path / 'retimed'
    summary_path = tmp_path / 'summary.json'
    monkeypatch.setattr('sys.argv', [
        'commentary_retiming',
        '--aligned-dir', str(aligned_dir),
        '--cleaned-dir', str(cleaned_dir),
        '--out-dir', str(out_dir),
        '--summary', str(summary_path),
    ])
    retiming.main()

    rows = json.loads((out_dir / 'v.json').read_text(encoding='utf-8'))
    assert [row['align_status'] for row in rows] == ['aligned', 'unmatched']
    assert rows[0]['text_clean'] == rows[0]['text']  # every cleaned field survives
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    assert summary['videos'] == 1
    assert summary['totals']['status_counts'] == {'aligned': 1, 'unmatched': 1, 'collision': 0}
    assert summary['per_video']['v']['abs_shift_s']['max'] == pytest.approx(25.0)
    assert json.loads((cleaned_dir / 'v.json').read_text(encoding='utf-8')) == chunks  # inputs untouched
