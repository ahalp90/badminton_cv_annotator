"""Stage 3 (relevance triage) unit tests. CPU-only, no network, no LLM calls.

Covers chunk-window overlap (a boundary segment lands in two windows) and the
empty case, the three-legged D9 keep rule (one test per leg plus boundary and
unknown-duration behaviour), the retry/backoff wrapper (call count, no real
sleep, TriageError after max retries), the keep write-back's blank-preservation
invariant, and the all-calls-failed block.

Run from repo root::

    ~/.venvs/badminton-cicd/bin/python -m pytest tests/test_scraper_stage3.py -v
"""
import pytest

from src.scraper import config, stage3_triage as stage3


# ---------------------------------------------------------------------------
# chunk_windows
# ---------------------------------------------------------------------------
def test_chunk_windows_empty_segments():
    assert stage3.chunk_windows([]) == []


def test_chunk_windows_boundary_segment_lands_in_two_windows():
    step = config.CHUNK_WINDOW_S - config.CHUNK_OVERLAP_S
    # A segment starting exactly on the step boundary falls in both window 0
    # ([0, window)) and window 1 ([step, step + window)).
    segment = {'start': float(step), 'end': float(step) + 1.0}
    windows = stage3.chunk_windows([segment])
    holding = [window for window in windows if segment in window['segments']]
    assert len(holding) == 2


# ---------------------------------------------------------------------------
# Three-legged D9 keep rule
# ---------------------------------------------------------------------------
def test_keep_rule_absolute_leg():
    # A long, sparse video keeps on absolute count alone, regardless of density.
    long_duration = str(config.SHORT_VIDEO_MIN_S * 100)
    assert stage3._keep_decision(config.CHUNKS_ABS_SAFE, long_duration) is True
    # One under the absolute floor, and far too sparse for the density leg.
    assert stage3._keep_decision(config.CHUNKS_ABS_SAFE - 1, long_duration) is False


def test_keep_rule_short_video_count_leg():
    # Boundary: duration == SHORT_VIDEO_MIN_S counts as short.
    short_duration = str(config.SHORT_VIDEO_MIN_S)
    assert config.CHUNKS_MIN_SHORT < config.CHUNKS_ABS_SAFE  # this leg carries the keep
    assert stage3._keep_decision(config.CHUNKS_MIN_SHORT, short_duration) is True
    assert stage3._keep_decision(config.CHUNKS_MIN_SHORT - 1, short_duration) is False


def test_keep_rule_long_video_density_leg():
    seconds = config.SHORT_VIDEO_MIN_S + 60  # just over the short/long boundary
    minutes = seconds / 60.0
    n_pass = int(config.DENSITY_MIN_PER_MIN * minutes) + 1  # clears the density floor
    assert n_pass < config.CHUNKS_ABS_SAFE  # ensure density, not the absolute leg, keeps it
    assert stage3._keep_decision(n_pass, str(seconds)) is True
    assert stage3._keep_decision(0, str(seconds)) is False


def test_keep_rule_unknown_duration_uses_absolute_leg_only():
    # Blank duration can be judged neither short nor long, so only the
    # length-independent absolute leg applies.
    assert stage3._keep_decision(config.CHUNKS_ABS_SAFE, '') is True
    assert stage3._keep_decision(config.CHUNKS_ABS_SAFE - 1, '') is False
    # A count that would clear the short-count floor cannot keep without a duration.
    assert stage3._keep_decision(config.CHUNKS_MIN_SHORT, '') is False


# ---------------------------------------------------------------------------
# Retry / backoff wrapper
# ---------------------------------------------------------------------------
_WINDOW = {'start': 0.0, 'end': 600.0, 'segments': [{'start': 0, 'end': 1, 'text': 'x'}]}


def test_call_triage_llm_retries_then_succeeds(monkeypatch):
    assert config.LLM_MAX_RETRIES >= 3  # the fixture below fails twice before succeeding
    calls = {'n': 0}
    sleeps = []

    def flaky(system_prompt, user_prompt):
        calls['n'] += 1
        if calls['n'] < 3:
            raise RuntimeError('transient')
        return [{'start': 0, 'end': 1, 'text': 'ok'}]

    monkeypatch.setattr(stage3, '_call_once', flaky)
    monkeypatch.setattr(stage3.time, 'sleep', lambda seconds: sleeps.append(seconds))

    out = stage3.call_triage_llm(_WINDOW)
    assert out == [{'start': 0, 'end': 1, 'text': 'ok'}]
    assert calls['n'] == 3          # failed twice, succeeded on the third attempt
    assert len(sleeps) == 2         # one backoff per failed attempt, none real


def test_call_triage_llm_raises_after_max_retries(monkeypatch):
    calls = {'n': 0}

    def always_fail(system_prompt, user_prompt):
        calls['n'] += 1
        raise RuntimeError('endpoint down')

    monkeypatch.setattr(stage3, '_call_once', always_fail)
    monkeypatch.setattr(stage3.time, 'sleep', lambda seconds: None)

    with pytest.raises(stage3.TriageError):
        stage3.call_triage_llm(_WINDOW)
    assert calls['n'] == config.LLM_MAX_RETRIES


# ---------------------------------------------------------------------------
# _write_keep_back
# ---------------------------------------------------------------------------
def test_write_keep_back_preserves_blank_for_untriaged(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'SCRAPE_DIR', tmp_path)
    monkeypatch.setattr(config, 'CANDIDATES_CSV', tmp_path / 'candidates.csv')
    rows = [
        {'video_id': 'a', 'keep': ''},
        {'video_id': 'b', 'keep': ''},
    ]
    stage3._write_keep_back(rows, {'a': True})

    assert rows[0]['keep'] == 'True'
    assert rows[1]['keep'] == ''  # untriaged row keeps its blank keep
    written = {row['video_id']: row for row in config.read_candidates()}
    assert written['a']['keep'] == 'True'
    assert written['b']['keep'] == ''


# ---------------------------------------------------------------------------
# run_stage3 batch behaviour
# ---------------------------------------------------------------------------
@pytest.fixture
def stage3_env(tmp_path, monkeypatch):
    """Redirect transcripts/chunks/candidates under tmp; no-op ensure_dirs."""
    transcripts = tmp_path / 'transcripts'
    transcripts.mkdir()
    chunks = tmp_path / 'chunks'
    chunks.mkdir()
    monkeypatch.setattr(stage3, 'TRANSCRIPTS_DIR', transcripts)
    monkeypatch.setattr(stage3, 'CHUNKS_DIR', chunks)
    monkeypatch.setattr(stage3, 'ensure_dirs', lambda: None)
    monkeypatch.setattr(config, 'SCRAPE_DIR', tmp_path)
    monkeypatch.setattr(config, 'CANDIDATES_CSV', tmp_path / 'candidates.csv')
    return transcripts


def _seed_transcript_rows(transcripts, count: int) -> list[dict]:
    rows = []
    for i in range(count):
        video_id = f'v{i}'
        (transcripts / f'{video_id}.json').write_text('{}', encoding='utf-8')
        rows.append({'video_id': video_id, 'duration_s': '100', 'keep': ''})
    return rows


def test_run_stage3_writes_chunks_and_keep(stage3_env, monkeypatch):
    transcripts = stage3_env
    rows = _seed_transcript_rows(transcripts, 1)

    def ok_triage(video_id, duration_s):
        return True, [{'chunk_id': f'{video_id}_c0', 'start': 0, 'end': 1, 'text': 'nice'}]

    monkeypatch.setattr(stage3, 'triage_video', ok_triage)
    keep_by_id = stage3.run_stage3(rows)

    assert keep_by_id == {'v0': True}
    assert (stage3.CHUNKS_DIR / 'v0.json').exists()
    written = {row['video_id']: row for row in config.read_candidates()}
    assert written['v0']['keep'] == 'True'


def test_run_stage3_all_calls_fail_raises(stage3_env, monkeypatch):
    transcripts = stage3_env
    below_floor = config.STAGE3_BLOCK_MIN_FAILURES - 1  # under the mid-batch floor
    assert below_floor >= 1
    rows = _seed_transcript_rows(transcripts, below_floor)

    def fail_triage(video_id, duration_s):
        raise stage3.TriageError('endpoint down')

    monkeypatch.setattr(stage3, 'triage_video', fail_triage)
    # End-of-run check fires: every video with a transcript failed triage.
    with pytest.raises(RuntimeError, match='triage call'):
        stage3.run_stage3(rows)


def test_run_stage3_mid_batch_breaker_stops_at_floor(stage3_env, monkeypatch):
    transcripts = stage3_env
    rows = _seed_transcript_rows(transcripts, config.STAGE3_BLOCK_MIN_FAILURES + 2)
    calls = {'n': 0}

    def fail_triage(video_id, duration_s):
        calls['n'] += 1
        raise stage3.TriageError('endpoint down')

    monkeypatch.setattr(stage3, 'triage_video', fail_triage)
    with pytest.raises(RuntimeError):
        stage3.run_stage3(rows)
    # Stops at the floor rather than attempting every seeded video.
    assert calls['n'] == config.STAGE3_BLOCK_MIN_FAILURES
