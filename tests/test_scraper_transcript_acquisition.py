"""Transcript-acquisition unit tests. CPU-only, no network.

Covers json3/vtt parsing (timing-only events skipped, multi-line cues joined),
the json3-over-vtt caption preference, the mid-batch failure circuit breaker and
its floor/fraction gating, resume-skip of existing sidecars, and the clean None
return from whisperx_fallback when whisperx is unavailable.

Run from repo root::

    ~/.venvs/badminton-cicd/bin/python -m pytest tests/test_scraper_transcript_acquisition.py -v
"""
import json
from types import SimpleNamespace

import pytest

from src.scraper import config, transcript_acquisition


# ---------------------------------------------------------------------------
# Caption parsing
# ---------------------------------------------------------------------------
def test_parse_json3_skips_timing_only_and_joins_segs(tmp_path):
    data = {
        'events': [
            {'tStartMs': 0, 'dDurationMs': 1000},              # no segs -> skipped
            {'tStartMs': 1000, 'dDurationMs': 2000,
             'segs': [{'utf8': 'nice '}, {'utf8': 'shot'}]},   # joined across pieces
            {'tStartMs': 5000, 'dDurationMs': 500,
             'segs': [{'utf8': '\n'}]},                        # whitespace only -> skipped
        ],
    }
    path = tmp_path / 'v.json3'
    path.write_text(json.dumps(data), encoding='utf-8')

    segments = transcript_acquisition.parse_json3(path)
    assert len(segments) == 1
    assert segments[0]['text'] == 'nice shot'
    assert segments[0]['start'] == 1.0
    assert segments[0]['end'] == 3.0


def test_parse_vtt_joins_multiline_and_skips_empty_cues(tmp_path):
    vtt = '\n'.join([
        'WEBVTT',
        '',
        '00:00:01.000 --> 00:00:03.000',
        'first line',
        'second line',
        '',
        '00:00:07.000 --> 00:00:08.000',  # timestamp with no text -> skipped
        '',
        '00:00:04.500 --> 00:00:06.000',
        'solo',
        '',
    ])
    path = tmp_path / 'v.vtt'
    path.write_text(vtt, encoding='utf-8')

    segments = transcript_acquisition.parse_vtt(path)
    assert len(segments) == 2
    assert segments[0]['text'] == 'first line second line'
    assert segments[0]['start'] == 1.0
    assert segments[0]['end'] == 3.0
    assert segments[1]['text'] == 'solo'
    assert segments[1]['start'] == 4.5


# ---------------------------------------------------------------------------
# pull_subtitles caption-format preference
# ---------------------------------------------------------------------------
def test_pull_subtitles_prefers_json3_over_vtt(tmp_path, monkeypatch):
    work = tmp_path / 'work'
    work.mkdir()

    def fake_run(cmd, capture_output=False, text=False, timeout=None):
        # yt-dlp would write both formats; the pull must prefer the timestamped json3.
        (work / 'vid.en.json3').write_text('{}', encoding='utf-8')
        (work / 'vid.en.vtt').write_text('WEBVTT', encoding='utf-8')
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    monkeypatch.setattr(transcript_acquisition.subprocess, 'run', fake_run)
    path = transcript_acquisition.pull_subtitles('vid', 'https://y/vid', str(work))
    assert path is not None
    assert path.suffix == '.json3'


def test_pull_subtitles_none_when_no_caption_file(tmp_path, monkeypatch):
    work = tmp_path / 'work'
    work.mkdir()
    monkeypatch.setattr(
        transcript_acquisition.subprocess, 'run',
        lambda *a, **k: SimpleNamespace(returncode=0, stdout='', stderr=''),
    )
    assert transcript_acquisition.pull_subtitles('vid', 'https://y/vid', str(work)) is None


# ---------------------------------------------------------------------------
# whisperx_fallback: unavailable in the test venv
# ---------------------------------------------------------------------------
def test_whisperx_fallback_none_when_whisperx_missing():
    # whisperx is not installed in the test venv, so the function-local import
    # guard trips and the fallback returns None with no exception.
    assert transcript_acquisition.whisperx_fallback('vid', 'https://y/vid') is None


# ---------------------------------------------------------------------------
# run_transcript_acquisition batch behaviour
# ---------------------------------------------------------------------------
@pytest.fixture
def transcript_env(tmp_path, monkeypatch):
    """Redirect the transcripts dir under tmp and neuter yt-dlp checks and sleeps."""
    transcripts = tmp_path / 'transcripts'
    transcripts.mkdir()
    monkeypatch.setattr(transcript_acquisition, 'TRANSCRIPTS_DIR', transcripts)
    monkeypatch.setattr(transcript_acquisition, 'check_ytdlp', lambda: None)
    monkeypatch.setattr(transcript_acquisition, 'ensure_dirs', lambda: None)
    monkeypatch.setattr(transcript_acquisition.time, 'sleep', lambda *a, **k: None)
    monkeypatch.setattr(transcript_acquisition.random, 'uniform', lambda a, b: 0.0)
    return transcripts


def _rows(count: int) -> list[dict]:
    return [{'video_id': f'v{i}', 'url': f'https://y/{i}'} for i in range(count)]


def test_transcript_circuit_breaker_fires_at_floor(transcript_env, monkeypatch):
    calls = {'n': 0}

    def always_fail(video_id, url):
        calls['n'] += 1
        return None

    monkeypatch.setattr(transcript_acquisition, 'acquire_transcript', always_fail)
    with pytest.raises(RuntimeError, match='block threshold'):
        transcript_acquisition.run_transcript_acquisition(_rows(20))
    # Stops exactly at the floor rather than burning all 20 attempts.
    assert calls['n'] == config.TRANSCRIPT_BLOCK_MIN_ATTEMPTS


def test_transcript_below_floor_no_early_stop_then_final_block(transcript_env, monkeypatch):
    calls = {'n': 0}

    def always_fail(video_id, url):
        calls['n'] += 1
        return None

    monkeypatch.setattr(transcript_acquisition, 'acquire_transcript', always_fail)
    below_floor = config.TRANSCRIPT_BLOCK_MIN_ATTEMPTS - 1
    with pytest.raises(RuntimeError, match='block threshold'):
        transcript_acquisition.run_transcript_acquisition(_rows(below_floor))
    # Below the floor the mid-batch breaker never trips: every row is attempted
    # and the end-of-run check raises instead.
    assert calls['n'] == below_floor


def test_transcript_no_block_at_or_below_fraction_and_writes_sidecars(transcript_env, monkeypatch):
    transcripts = transcript_env

    def acquire(video_id, url):
        # Even-indexed videos resolve, odd fail: fraction lands at exactly 0.5,
        # not over the block threshold, so the run completes.
        if int(video_id[1:]) % 2 == 0:
            return {'source': 'youtube_asr', 'segments': [{'start': 0, 'end': 1, 'text': 'x'}]}
        return None

    monkeypatch.setattr(transcript_acquisition, 'acquire_transcript', acquire)
    transcript_acquisition.run_transcript_acquisition(_rows(8))

    written = sorted(path.name for path in transcripts.glob('*.json'))
    assert written == ['v0.json', 'v2.json', 'v4.json', 'v6.json']


def test_transcript_resume_skips_existing_sidecars(transcript_env, monkeypatch):
    transcripts = transcript_env
    (transcripts / 'v0.json').write_text('{}', encoding='utf-8')  # pre-existing sidecar
    seen = []

    def acquire(video_id, url):
        seen.append(video_id)
        return {'source': 'youtube_asr', 'segments': [{'start': 0, 'end': 1, 'text': 'x'}]}

    monkeypatch.setattr(transcript_acquisition, 'acquire_transcript', acquire)
    transcript_acquisition.run_transcript_acquisition([
        {'video_id': 'v0', 'url': 'u0'},
        {'video_id': 'v1', 'url': 'u1'},
    ])
    # v0 already had a sidecar, so only v1 is attempted.
    assert seen == ['v1']
