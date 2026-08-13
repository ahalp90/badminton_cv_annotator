"""Relevance-triage unit tests. CPU-only, no network, no LLM calls.

Covers chunk-window overlap (a boundary segment lands in two windows) and the
empty case, the three-legged D9 keep rule (one test per leg plus boundary and
unknown-duration behaviour), the retry/backoff wrapper (call count, no real
sleep, TriageError after max retries), the keep write-back's blank-preservation
invariant, and the all-calls-failed block.

Run from repo root::

    ~/.venvs/badminton-cicd/bin/python -m pytest tests/test_scraper_relevance_triage.py -v
"""
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from src.scraper import _llm_errors, config, relevance_triage


# ---------------------------------------------------------------------------
# chunk_windows
# ---------------------------------------------------------------------------
def test_chunk_windows_empty_segments():
    assert relevance_triage.chunk_windows([]) == []


def test_chunk_windows_boundary_segment_lands_in_two_windows():
    step = config.CHUNK_WINDOW_S - config.CHUNK_OVERLAP_S
    # A segment starting exactly on the step boundary falls in both window 0
    # ([0, window)) and window 1 ([step, step + window)).
    segment = {'start': float(step), 'end': float(step) + 1.0}
    windows = relevance_triage.chunk_windows([segment])
    holding = [window for window in windows if segment in window['segments']]
    assert len(holding) == 2


# ---------------------------------------------------------------------------
# Three-legged D9 keep rule
# ---------------------------------------------------------------------------
def test_keep_rule_absolute_leg():
    # A long, sparse video keeps on absolute count alone, regardless of density.
    long_duration = str(config.SHORT_VIDEO_MIN_S * 100)
    assert relevance_triage._keep_decision(config.CHUNKS_ABS_SAFE, long_duration) is True
    # One under the absolute floor, and far too sparse for the density leg.
    assert relevance_triage._keep_decision(config.CHUNKS_ABS_SAFE - 1, long_duration) is False


def test_keep_rule_short_video_count_leg():
    # Boundary: duration == SHORT_VIDEO_MIN_S counts as short.
    short_duration = str(config.SHORT_VIDEO_MIN_S)
    assert config.CHUNKS_MIN_SHORT < config.CHUNKS_ABS_SAFE  # this leg carries the keep
    assert relevance_triage._keep_decision(config.CHUNKS_MIN_SHORT, short_duration) is True
    assert relevance_triage._keep_decision(config.CHUNKS_MIN_SHORT - 1, short_duration) is False


def test_keep_rule_long_video_density_leg():
    seconds = config.SHORT_VIDEO_MIN_S + 60  # just over the short/long boundary
    minutes = seconds / 60.0
    n_pass = int(config.DENSITY_MIN_PER_MIN * minutes) + 1  # clears the density floor
    assert n_pass < config.CHUNKS_ABS_SAFE  # ensure density, not the absolute leg, keeps it
    assert relevance_triage._keep_decision(n_pass, str(seconds)) is True
    assert relevance_triage._keep_decision(0, str(seconds)) is False


def test_keep_rule_unknown_duration_uses_absolute_leg_only():
    # Blank duration can be judged neither short nor long, so only the
    # length-independent absolute leg applies.
    assert relevance_triage._keep_decision(config.CHUNKS_ABS_SAFE, '') is True
    assert relevance_triage._keep_decision(config.CHUNKS_ABS_SAFE - 1, '') is False
    # A count that would clear the short-count floor cannot keep without a duration.
    assert relevance_triage._keep_decision(config.CHUNKS_MIN_SHORT, '') is False


# ---------------------------------------------------------------------------
# Retry / backoff wrapper
# ---------------------------------------------------------------------------
_WINDOW = {'start': 0.0, 'end': 600.0, 'segments': [{'start': 0, 'end': 1, 'text': 'x'}]}


def test_daily_quota_classifier_rejects_recoverable_or_unstructured_errors():
    class ProviderError(RuntimeError):
        def __init__(self, code, quota_id):
            super().__init__('provider error')
            self.code = code
            self.details = {'violations': [{'quotaId': quota_id}]}

    daily_id = 'GenerateRequestsPerDayPerProjectPerModel'
    free_daily_id = f'{daily_id}-FreeTier'
    minute_id = 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier'

    assert _llm_errors.daily_request_quota_exhausted(ProviderError(429, daily_id)) is True
    assert (
        _llm_errors.daily_request_quota_exhausted(ProviderError(429, free_daily_id))
        is True
    )
    assert _llm_errors.daily_request_quota_exhausted(ProviderError(429, minute_id)) is False
    assert _llm_errors.daily_request_quota_exhausted(ProviderError(503, daily_id)) is False
    assert _llm_errors.daily_request_quota_exhausted(RuntimeError(daily_id)) is False


def test_triage_client_has_a_bounded_request_timeout(monkeypatch):
    captured: dict[str, object] = {}
    google = ModuleType('google')
    genai = ModuleType('google.genai')

    def http_options(*, timeout):
        captured['timeout'] = timeout
        return SimpleNamespace(timeout=timeout)

    def client(**kwargs):
        captured['client_kwargs'] = kwargs
        response = SimpleNamespace(text=json.dumps([]))
        return SimpleNamespace(models=SimpleNamespace(
            generate_content=lambda **_request: response,
        ))

    genai.Client = client
    genai.types = SimpleNamespace(HttpOptions=http_options)
    google.genai = genai
    monkeypatch.setitem(sys.modules, 'google', google)
    monkeypatch.setitem(sys.modules, 'google.genai', genai)
    monkeypatch.setenv(config.API_KEY_ENV, 'fixture-secret')

    assert relevance_triage._call_once('system', 'user') == []
    assert captured['timeout'] == config.LLM_REQUEST_TIMEOUT_S * 1000
    client_kwargs = captured['client_kwargs']
    assert isinstance(client_kwargs, dict)
    assert client_kwargs['http_options'].timeout == captured['timeout']


def test_call_triage_llm_retries_then_succeeds(monkeypatch):
    assert config.LLM_MAX_RETRIES >= 3  # the fixture below fails twice before succeeding
    calls = {'n': 0}
    sleeps = []

    def flaky(system_prompt, user_prompt):
        calls['n'] += 1
        if calls['n'] < 3:
            raise RuntimeError('transient')
        return [{'start': 0, 'end': 1, 'text': 'ok'}]

    monkeypatch.setattr(relevance_triage, '_call_once', flaky)
    monkeypatch.setattr(relevance_triage.time, 'sleep', lambda seconds: sleeps.append(seconds))

    out = relevance_triage.call_triage_llm(_WINDOW)
    assert out == [{'start': 0, 'end': 1, 'text': 'ok'}]
    assert calls['n'] == 3          # failed twice, succeeded on the third attempt
    assert len(sleeps) == 2         # one backoff per failed attempt, none real


def test_call_triage_llm_raises_after_max_retries(monkeypatch):
    calls = {'n': 0}

    def always_fail(system_prompt, user_prompt):
        calls['n'] += 1
        raise TimeoutError('request timed out')

    monkeypatch.setattr(relevance_triage, '_call_once', always_fail)
    monkeypatch.setattr(relevance_triage.time, 'sleep', lambda seconds: None)

    with pytest.raises(relevance_triage.TriageError):
        relevance_triage.call_triage_llm(_WINDOW)
    assert calls['n'] == config.LLM_MAX_RETRIES


@pytest.mark.parametrize('quota_suffix', ['', '-FreeTier'])
def test_call_triage_llm_does_not_retry_exhausted_daily_quota(
    monkeypatch,
    quota_suffix,
):
    calls = 0
    sleeps: list[float] = []

    class DailyQuotaError(RuntimeError):
        code = 429
        details = {
            'error': {
                'details': [{
                    'violations': [{
                        'quotaId': (
                            f'GenerateRequestsPerDayPerProjectPerModel{quota_suffix}'
                        ),
                    }],
                }],
            },
        }

    def exhausted(_system_prompt, _user_prompt):
        nonlocal calls
        calls += 1
        raise DailyQuotaError('daily quota exhausted')

    monkeypatch.setattr(relevance_triage, '_call_once', exhausted)
    monkeypatch.setattr(relevance_triage.time, 'sleep', sleeps.append)

    with pytest.raises(_llm_errors.DailyRequestQuotaError, match='daily request quota'):
        relevance_triage.call_triage_llm(_WINDOW)
    assert calls == 1
    assert sleeps == []


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
    relevance_triage._write_keep_back(rows, {'a': True})

    assert rows[0]['keep'] == 'True'
    assert rows[1]['keep'] == ''  # untriaged row keeps its blank keep
    written = {row['video_id']: row for row in config.read_candidates()}
    assert written['a']['keep'] == 'True'
    assert written['b']['keep'] == ''


# ---------------------------------------------------------------------------
# run_relevance_triage batch behaviour
# ---------------------------------------------------------------------------
@pytest.fixture
def triage_env(tmp_path, monkeypatch):
    """Redirect transcripts/chunks/candidates under tmp; no-op ensure_dirs."""
    transcripts = tmp_path / 'transcripts'
    transcripts.mkdir()
    chunks = tmp_path / 'chunks'
    chunks.mkdir()
    monkeypatch.setattr(relevance_triage, 'TRANSCRIPTS_DIR', transcripts)
    monkeypatch.setattr(relevance_triage, 'CHUNKS_DIR', chunks)
    monkeypatch.setattr(relevance_triage, 'ensure_dirs', lambda: None)
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


def test_run_relevance_triage_writes_chunks_and_keep(triage_env, monkeypatch):
    transcripts = triage_env
    rows = _seed_transcript_rows(transcripts, 1)

    def ok_triage(video_id, duration_s):
        return True, [{'chunk_id': f'{video_id}_c0', 'start': 0, 'end': 1, 'text': 'nice'}]

    monkeypatch.setattr(relevance_triage, 'triage_video', ok_triage)
    keep_by_id = relevance_triage.run_relevance_triage(rows)

    assert keep_by_id == {'v0': True}
    assert (relevance_triage.CHUNKS_DIR / 'v0.json').exists()
    written = {row['video_id']: row for row in config.read_candidates()}
    assert written['v0']['keep'] == 'True'


def test_run_relevance_triage_daily_quota_stops_before_later_video(
    triage_env,
    monkeypatch,
):
    transcripts = triage_env
    rows = _seed_transcript_rows(transcripts, 2)
    calls: list[str] = []

    def exhausted(video_id, _duration_s):
        calls.append(video_id)
        raise _llm_errors.DailyRequestQuotaError('daily quota exhausted')

    monkeypatch.setattr(relevance_triage, 'triage_video', exhausted)

    with pytest.raises(_llm_errors.DailyRequestQuotaError, match='daily quota'):
        relevance_triage.run_relevance_triage(rows)
    assert calls == ['v0']


def test_run_relevance_triage_all_calls_fail_raises(triage_env, monkeypatch):
    transcripts = triage_env
    below_floor = config.TRIAGE_BLOCK_MIN_FAILURES - 1  # under the mid-batch floor
    assert below_floor >= 1
    rows = _seed_transcript_rows(transcripts, below_floor)

    def fail_triage(video_id, duration_s):
        raise relevance_triage.TriageError('endpoint down')

    monkeypatch.setattr(relevance_triage, 'triage_video', fail_triage)
    # End-of-run check fires: every video with a transcript failed triage.
    with pytest.raises(RuntimeError, match='triage call'):
        relevance_triage.run_relevance_triage(rows)


def test_run_relevance_triage_mid_batch_breaker_stops_at_floor(triage_env, monkeypatch):
    transcripts = triage_env
    rows = _seed_transcript_rows(transcripts, config.TRIAGE_BLOCK_MIN_FAILURES + 2)
    calls = {'n': 0}

    def fail_triage(video_id, duration_s):
        calls['n'] += 1
        raise relevance_triage.TriageError('endpoint down')

    monkeypatch.setattr(relevance_triage, 'triage_video', fail_triage)
    with pytest.raises(RuntimeError):
        relevance_triage.run_relevance_triage(rows)
    # Stops at the floor rather than attempting every seeded video.
    assert calls['n'] == config.TRIAGE_BLOCK_MIN_FAILURES
