"""Tests for scraper stage 10 (clean pass + fine timestamps).

CPU only, no network, no WhisperX. The LLM call is faked via monkeypatch; the
WhisperX/torch imports inside refine_timestamps are function-local, so importing
the module here (the test venv has no whisperx) must not fail.
"""
import json

import pytest

from src.scraper import stage10_clean as stage10


def _phrasings():
    """A list of exactly ALT_PHRASINGS_K fake paraphrases."""
    return [f'p{i}' for i in range(stage10.ALT_PHRASINGS_K)]


def _write_sidecar(chunks_dir, video_id, chunks):
    """Write chunks/<video_id>.json and return its path."""
    path = chunks_dir / f'{video_id}.json'
    path.write_text(json.dumps(chunks), encoding='utf-8')
    return path


# -- Clean pass --------------------------------------------------------------


def test_run_clean_extends_in_place_and_keeps_k_phrasings(tmp_path, monkeypatch):
    """A kept video's chunks gain text_clean + K phrasings; original fields survive."""
    monkeypatch.setattr(stage10, 'CHUNKS_DIR', tmp_path)
    monkeypatch.setattr(
        stage10, 'call_clean_llm',
        lambda text: {'text_clean': f'CLEAN::{text}', 'alt_phrasings': _phrasings()},
    )
    sidecar = _write_sidecar(tmp_path, 'v1', [
        {'chunk_id': 'v1_c0', 'start': 1.0, 'end': 2.0, 'text': 'raw one'},
        {'chunk_id': 'v1_c1', 'start': 3.0, 'end': 4.0, 'text': 'raw two'},
    ])

    stage10.run_clean(rows=[{'video_id': 'v1', 'keep': 'True'}])

    out = json.loads(sidecar.read_text(encoding='utf-8'))
    assert out[0]['text_clean'] == 'CLEAN::raw one'
    assert out[1]['text_clean'] == 'CLEAN::raw two'
    # Original coarse timestamps and chunk_id are left in place.
    assert (out[0]['start'], out[0]['end'], out[0]['chunk_id']) == (1.0, 2.0, 'v1_c0')
    assert len(out[0]['alt_phrasings']) == stage10.ALT_PHRASINGS_K
    assert len(out[1]['alt_phrasings']) == stage10.ALT_PHRASINGS_K


def test_run_clean_idempotent_skip_and_force_override(tmp_path, monkeypatch):
    """A chunk already carrying text_clean is skipped unless --force is set."""
    monkeypatch.setattr(stage10, 'CHUNKS_DIR', tmp_path)
    calls = []

    def fake(text):
        calls.append(text)
        return {'text_clean': 'NEW', 'alt_phrasings': _phrasings()}

    monkeypatch.setattr(stage10, 'call_clean_llm', fake)
    sidecar = _write_sidecar(tmp_path, 'v1', [{
        'chunk_id': 'v1_c0', 'start': 1.0, 'end': 2.0, 'text': 'raw',
        'text_clean': 'OLD', 'alt_phrasings': ['x'],
    }])
    rows = [{'video_id': 'v1', 'keep': 'True'}]

    # Idempotent: no call, text_clean untouched.
    stage10.run_clean(rows=rows)
    assert calls == []
    assert json.loads(sidecar.read_text(encoding='utf-8'))[0]['text_clean'] == 'OLD'

    # Force: the chunk is re-cleaned.
    stage10.run_clean(rows=rows, force=True)
    assert calls == ['raw']
    assert json.loads(sidecar.read_text(encoding='utf-8'))[0]['text_clean'] == 'NEW'


def test_run_clean_keep_filter_parses_not_truth_tests(tmp_path, monkeypatch):
    """Only rows whose keep equals the string 'True' are processed."""
    monkeypatch.setattr(stage10, 'CHUNKS_DIR', tmp_path)
    seen = []

    def fake(text):
        seen.append(text)
        return {'text_clean': 'C', 'alt_phrasings': _phrasings()}

    monkeypatch.setattr(stage10, 'call_clean_llm', fake)
    for video_id in ('keep_true', 'keep_false', 'keep_lower', 'keep_blank'):
        _write_sidecar(tmp_path, video_id, [
            {'chunk_id': f'{video_id}_c0', 'start': 0.0, 'end': 1.0, 'text': video_id},
        ])
    rows = [
        {'video_id': 'keep_true', 'keep': 'True'},
        {'video_id': 'keep_false', 'keep': 'False'},   # a non-empty string is truthy, so must NOT slip through
        {'video_id': 'keep_lower', 'keep': 'true'},     # wrong case, does not parse True
        {'video_id': 'keep_blank', 'keep': ''},
    ]

    stage10.run_clean(rows=rows)
    assert seen == ['keep_true']


def test_run_clean_logs_and_skips_failing_video(tmp_path, monkeypatch):
    """One dead video is logged and skipped; the others still clean; no raise."""
    monkeypatch.setattr(stage10, 'CHUNKS_DIR', tmp_path)

    def fake(text):
        if text.startswith('BAD'):
            raise stage10.CleanError('boom')
        return {'text_clean': f'C::{text}', 'alt_phrasings': _phrasings()}

    monkeypatch.setattr(stage10, 'call_clean_llm', fake)
    good = _write_sidecar(tmp_path, 'good', [
        {'chunk_id': 'good_c0', 'start': 0.0, 'end': 1.0, 'text': 'good one'},
    ])
    bad = _write_sidecar(tmp_path, 'bad', [
        {'chunk_id': 'bad_c0', 'start': 0.0, 'end': 1.0, 'text': 'BAD one'},
    ])
    rows = [{'video_id': 'good', 'keep': 'True'}, {'video_id': 'bad', 'keep': 'True'}]

    result = stage10.run_clean(rows=rows)  # must not raise: not every call failed
    assert 'text_clean' in json.loads(good.read_text(encoding='utf-8'))[0]
    assert 'text_clean' not in json.loads(bad.read_text(encoding='utf-8'))[0]
    assert result == {'good': 1}


def test_run_clean_raises_on_dead_endpoint(tmp_path, monkeypatch):
    """Every LLM call failing blocks the run (dead endpoint)."""
    monkeypatch.setattr(stage10, 'CHUNKS_DIR', tmp_path)

    def fake(_text):
        raise stage10.CleanError('dead')

    monkeypatch.setattr(stage10, 'call_clean_llm', fake)
    _write_sidecar(tmp_path, 'v1', [{'chunk_id': 'v1_c0', 'start': 0.0, 'end': 1.0, 'text': 'a'}])
    _write_sidecar(tmp_path, 'v2', [{'chunk_id': 'v2_c0', 'start': 0.0, 'end': 1.0, 'text': 'b'}])
    rows = [{'video_id': 'v1', 'keep': 'True'}, {'video_id': 'v2', 'keep': 'True'}]

    with pytest.raises(RuntimeError, match='all .* LLM calls failed'):
        stage10.run_clean(rows=rows)


def test_call_clean_llm_retries_then_raises(monkeypatch):
    """The retry wrapper retries LLM_MAX_RETRIES times then raises CleanError."""
    attempts = []

    def boom(text):
        attempts.append(text)
        raise RuntimeError('transient')

    monkeypatch.setattr(stage10, '_clean_once', boom)
    monkeypatch.setattr(stage10.time, 'sleep', lambda _s: None)

    with pytest.raises(stage10.CleanError):
        stage10.call_clean_llm('hi')
    assert len(attempts) == stage10.LLM_MAX_RETRIES


# -- Fine-timestamp pass -----------------------------------------------------


def test_padded_span_pads_and_clamps():
    """The span is padded FINE_PAD_S each side and the start never goes negative."""
    pad = stage10.FINE_PAD_S
    assert stage10._padded_span(10.0, 20.0) == (10.0 - pad, 20.0 + pad)
    start, end = stage10._padded_span(0.5, 5.0)  # start within pad of zero clamps
    assert start == 0.0
    assert end == 5.0 + pad


def test_extract_span_ffmpeg_argv(tmp_path, monkeypatch):
    """ffmpeg is called with -ss/-to at the padded bounds and the wav output."""
    captured = {}

    def fake_run(argv, **kwargs):
        captured['argv'] = list(argv)
        captured['kwargs'] = kwargs

    monkeypatch.setattr(stage10.subprocess, 'run', fake_run)
    video = tmp_path / 'vid.mp4'
    wav = tmp_path / 'out.wav'
    stage10._extract_span(video, 12.5, 34.25, wav)

    argv = captured['argv']
    assert argv[0] == 'ffmpeg'
    assert argv[argv.index('-ss') + 1] == '12.500'   # seek to padded span start
    assert argv[argv.index('-to') + 1] == '34.250'   # stop at padded span end
    assert str(video) in argv
    assert argv[-1] == str(wav)
    assert captured['kwargs'].get('check') is True


def test_fine_pass_noop_without_whisperx(tmp_path, monkeypatch):
    """With no whisperx installed, the fine pass skips cleanly: no models, no
    ffmpeg, sidecars untouched."""
    def boom(*_args, **_kwargs):
        raise AssertionError('subprocess.run must not run when whisperx is absent')

    monkeypatch.setattr(stage10.subprocess, 'run', boom)
    assert stage10.load_fine_models() is None  # this venv has no whisperx

    chunks_dir = tmp_path / 'chunks'
    chunks_dir.mkdir()
    sidecar = chunks_dir / 'vid.json'
    original = '[{"chunk_id": "c0", "start": 1.0, "end": 2.0, "text": "t"}]'
    sidecar.write_text(original, encoding='utf-8')
    monkeypatch.setattr(stage10, 'CHUNKS_DIR', chunks_dir)

    rows = [{'video_id': 'vid', 'keep': 'True'}]
    stage10.run_fine(tmp_path, rows)
    assert sidecar.read_text(encoding='utf-8') == original  # never rewritten
