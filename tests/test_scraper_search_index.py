"""Search-index unit tests. CPU-only, no network.

Covers --print line parsing (NA normalisation, wrong-field-count skip), the
dedup provenance join with first-substream-wins, the doubles/duration/upload
screens (including the instructional short-flag exemption), the blank
keep/triage_verdict invariant, and the all-terms-empty block.

Run from repo root::

    ~/.venvs/badminton-cicd/bin/python -m pytest tests/test_scraper_search_index.py -v
"""
from types import SimpleNamespace

import pytest

from annotator import config as annotator_config
from src.scraper import config, search_index


def test_scrape_output_paths_are_annotator_owned() -> None:
    assert config.SCRAPE_DIR is annotator_config.SCRAPE_DIR
    assert config.MASKS_DIR is annotator_config.MASKS_DIR
    assert config.RALLY_SPANS_CSV is annotator_config.RALLY_SPANS_CSV
    assert config.CONTACT_FRAMES_CSV is annotator_config.CONTACT_FRAMES_CSV
    assert annotator_config.MASKS_DIR == annotator_config.SCRAPE_DIR / 'masks'
    assert annotator_config.RALLY_SPANS_CSV == annotator_config.SCRAPE_DIR / 'rally_spans.csv'
    assert annotator_config.CONTACT_FRAMES_CSV == annotator_config.SCRAPE_DIR / 'contact_frames.csv'


def _row_line(video_id: str, title: str = 'Singles match', channel: str = 'Chan',
              duration: str = '1800', upload: str = '20240101') -> str:
    """One tab-separated --print line in FLAT_PRINT_TEMPLATE field order."""
    return f'{video_id}\thttps://y/{video_id}\t{title}\t{channel}\t{duration}\t{upload}'


def _search_run(term_to_lines: dict[str, list[str]]):
    """Fake subprocess.run: return canned --print stdout per search term.

    Rows arrive with every field filled, so enrich_row never fires; a stray
    --dump-json call still gets a benign empty-meta reply.
    """
    def fake_run(cmd, capture_output=False, text=False, timeout=None):
        if '--dump-json' in cmd:
            return SimpleNamespace(returncode=0, stdout='{}', stderr='')
        term = cmd[1].split(':', 1)[1]  # cmd[1] is 'ytsearchN:<term>'
        lines = term_to_lines.get(term, [])
        return SimpleNamespace(returncode=0, stdout='\n'.join(lines), stderr='')
    return fake_run


@pytest.fixture
def patched_paths(tmp_path, monkeypatch):
    """Redirect the scrape output under tmp and neuter yt-dlp checks and sleeps."""
    scrape = tmp_path / 'scrape'
    monkeypatch.setattr(config, 'SCRAPE_DIR', scrape)
    monkeypatch.setattr(config, 'CANDIDATES_CSV', scrape / 'candidates.csv')
    monkeypatch.setattr(config, 'TRANSCRIPTS_DIR', scrape / 'transcripts')
    monkeypatch.setattr(config, 'CHUNKS_DIR', scrape / 'chunks')
    monkeypatch.setattr(config, 'MASKS_DIR', scrape / 'masks')
    monkeypatch.setattr(search_index, 'CANDIDATES_CSV', scrape / 'candidates.csv')
    monkeypatch.setattr(search_index, 'check_ytdlp', lambda: None)
    monkeypatch.setattr(search_index.time, 'sleep', lambda *args, **kwargs: None)
    return scrape


# ---------------------------------------------------------------------------
# search_term_rows: line parsing
# ---------------------------------------------------------------------------
def test_search_term_rows_parses_normalises_na_and_skips_bad_lines(monkeypatch):
    stdout = '\n'.join([
        _row_line('vid1'),
        'vid2\thttps://y/vid2\tTitle Two\tNA\tNA\tNA',  # NA fields normalise to blank
        'badline\ttoo\tfew',                             # wrong field count, skipped
        '',                                              # blank line, skipped
    ])
    monkeypatch.setattr(
        search_index.subprocess, 'run',
        lambda *a, **k: SimpleNamespace(returncode=0, stdout=stdout, stderr=''),
    )
    rows = search_index.search_term_rows('some term', config.SUBSTREAM_INSTRUCTIONAL)

    assert len(rows) == 2
    assert rows[0]['video_id'] == 'vid1'
    assert rows[0]['channel'] == 'Chan'
    assert rows[0]['search_term'] == 'some term'
    assert rows[0]['substream'] == config.SUBSTREAM_INSTRUCTIONAL
    # 'NA' normalised to blank so the enrichment pass knows what to fill.
    assert rows[1]['channel'] == ''
    assert rows[1]['duration_s'] == ''
    assert rows[1]['upload_date'] == ''


def test_search_term_rows_nonzero_exit_returns_empty(monkeypatch):
    monkeypatch.setattr(
        search_index.subprocess, 'run',
        lambda *a, **k: SimpleNamespace(returncode=1, stdout='', stderr='boom'),
    )
    assert search_index.search_term_rows('term', config.SUBSTREAM_MATCH) == []


def test_search_term_rows_accepts_a_per_run_result_count(monkeypatch):
    received = []

    def fake_run(cmd, **_kwargs):
        received.append(cmd)
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    monkeypatch.setattr(search_index.subprocess, 'run', fake_run)

    search_index.search_term_rows('term', config.SUBSTREAM_MATCH, search_count=5)

    assert received[0][1] == 'ytsearch5:term'
    with pytest.raises(ValueError, match='positive integer'):
        search_index.search_term_rows('term', config.SUBSTREAM_MATCH, search_count=0)


# ---------------------------------------------------------------------------
# Doubles keyword screen
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('title, expected', [
    ('Mixed Doubles Final BWF', True),      # phrase substring
    ('A great doubles rally', True),        # 'doubles' substring
    ('MD semifinal highlights', True),      # 'md' as a whole token
    ('WD group stage', True),
    ('XD quarterfinal', True),
    ('The commander takes the court', False),  # 'md' inside 'commander' must not fire
    ('Roaring crowd noise', False),            # 'wd' inside 'crowd' must not fire
    ('Singles final full match', False),
])
def test_flag_doubles(title, expected):
    assert search_index.flag_doubles(title) is expected


# ---------------------------------------------------------------------------
# Duration band screen (with the instructional short-flag exemption)
# ---------------------------------------------------------------------------
def test_duration_out_of_band_match():
    below = str(config.DURATION_MIN_S - 1)
    above = str(config.DURATION_MAX_S + 1)
    inside = str(config.DURATION_MIN_S + 60)
    assert search_index.duration_out_of_band(below, config.SUBSTREAM_MATCH) is True
    assert search_index.duration_out_of_band(above, config.SUBSTREAM_MATCH) is True
    assert search_index.duration_out_of_band(inside, config.SUBSTREAM_MATCH) is False
    assert search_index.duration_out_of_band('', config.SUBSTREAM_MATCH) is False       # unknown
    assert search_index.duration_out_of_band('junk', config.SUBSTREAM_MATCH) is False   # unparseable


def test_duration_out_of_band_instructional_skips_short_keeps_long():
    below = str(config.DURATION_MIN_S - 1)
    above = str(config.DURATION_MAX_S + 1)
    # D24: coach-review clips run short by design, so the short leg is skipped...
    assert search_index.duration_out_of_band(below, config.SUBSTREAM_INSTRUCTIONAL) is False
    # ...but the over-long leg still applies.
    assert search_index.duration_out_of_band(above, config.SUBSTREAM_INSTRUCTIONAL) is True


# ---------------------------------------------------------------------------
# Upload-date screen: floor off means always False
# ---------------------------------------------------------------------------
def test_upload_before_floor_off_is_always_false():
    assert config.UPLOAD_DATE_FLOOR is None  # the shipped default (floor off)
    assert search_index.upload_before_floor('20200101') is False
    assert search_index.upload_before_floor('') is False


# ---------------------------------------------------------------------------
# build_candidates: dedup, provenance, substream, blank columns, block rule
# ---------------------------------------------------------------------------
def test_build_candidates_dedup_join_and_first_substream_wins(patched_paths, monkeypatch):
    term_to_lines = {
        'm1': [_row_line('shared'), _row_line('onlyM')],
        'i1': [_row_line('shared'), _row_line('onlyI')],
    }
    monkeypatch.setattr(search_index.subprocess, 'run', _search_run(term_to_lines))
    search_terms = {
        config.SUBSTREAM_MATCH: ['m1'],
        config.SUBSTREAM_INSTRUCTIONAL: ['i1'],
    }

    rows = search_index.build_candidates(search_terms)
    by_id = {row['video_id']: row for row in rows}

    assert set(by_id) == {'shared', 'onlyM', 'onlyI'}
    # Match family indexed 'shared' first, so its substream wins the tie...
    assert by_id['shared']['substream'] == config.SUBSTREAM_MATCH
    # ...while both provenance terms comma-join so the cross-family hit survives.
    assert by_id['shared']['search_term'] == 'm1,i1'
    assert by_id['onlyM']['substream'] == config.SUBSTREAM_MATCH
    assert by_id['onlyI']['substream'] == config.SUBSTREAM_INSTRUCTIONAL
    # keep and triage_verdict blank at index time (relevance triage / human packet fill them).
    assert all(row['keep'] == '' for row in rows)
    assert all(row['triage_verdict'] == '' for row in rows)

    # The 13-column contract carries substream and round-trips through the CSV.
    written = config.read_candidates()
    assert len(config.CANDIDATES_COLUMNS) == 13
    assert 'substream' in written[0]
    assert {row['video_id'] for row in written} == {'shared', 'onlyM', 'onlyI'}


def test_build_candidates_all_terms_empty_raises(patched_paths, monkeypatch):
    monkeypatch.setattr(
        search_index.subprocess, 'run',
        lambda *a, **k: SimpleNamespace(returncode=0, stdout='', stderr=''),
    )
    search_terms = {config.SUBSTREAM_MATCH: ['m1', 'm2']}
    with pytest.raises(RuntimeError, match='every search term returned zero'):
        search_index.build_candidates(search_terms)
