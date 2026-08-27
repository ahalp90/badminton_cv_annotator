"""Tests for scraper commentary cleaning and fine timestamps.

CPU only, no network, no WhisperX. The LLM call is faked via monkeypatch; the
WhisperX/torch imports inside refine_timestamps are function-local, so importing
the module here (the test venv has no whisperx) must not fail.
"""
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from src.scraper import _llm_errors, commentary_cleaning, config


@pytest.fixture(autouse=True)
def fake_bert_score(monkeypatch):
    """Keep commentary-cleaning tests independent of the optional scorer model."""
    batch_sizes: list[int] = []

    class FakeBERTScorer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def score(self, candidates, references, batch_size):
            del references
            batch_sizes.append(batch_size)
            return ([0.9] * len(candidates), [0.9] * len(candidates), [0.9] * len(candidates))

    monkeypatch.setitem(sys.modules, 'bert_score', SimpleNamespace(BERTScorer=FakeBERTScorer))
    return batch_sizes


def _phrasings():
    """A list of exactly ALT_PHRASINGS_K fake paraphrases."""
    return [f'p{i}' for i in range(commentary_cleaning.ALT_PHRASINGS_K)]


def _write_sidecar(chunks_dir, video_id, chunks):
    """Write chunks/<video_id>.json and return its path."""
    path = chunks_dir / f'{video_id}.json'
    path.write_text(json.dumps(chunks), encoding='utf-8')
    return path


def test_bertscore_uses_its_own_batch_size(fake_bert_score) -> None:
    commentary_cleaning._score_chunks([{'text': 'raw', 'text_clean': 'clean'}])
    assert fake_bert_score == [commentary_cleaning.BERTSCORE_BATCH_SIZE]


def test_fine_video_lookup_uses_shared_extensions(tmp_path) -> None:
    video = tmp_path / 'clip.webm'
    video.write_bytes(b'video')
    assert commentary_cleaning.VIDEO_EXTENSIONS is config.VIDEO_EXTENSIONS
    assert commentary_cleaning._find_video(tmp_path, 'clip') == video


def test_fine_video_lookup_accepts_legacy_spaced_basename(tmp_path) -> None:
    video = tmp_path / '0012 Match Name.mkv'
    video.write_bytes(b'video')
    (tmp_path / '0012 Match Name.f137.mp4').write_bytes(b'partial')

    assert commentary_cleaning._find_video(tmp_path, '0012') == video


def test_fine_video_lookup_rejects_duplicate_exact_and_legacy_sources(tmp_path) -> None:
    (tmp_path / '0012.mp4').write_bytes(b'exact')
    (tmp_path / '0012 Match Name.mkv').write_bytes(b'legacy')

    with pytest.raises(ValueError, match='multiple source videos'):
        commentary_cleaning._find_video(tmp_path, '0012')


# -- Clean pass --------------------------------------------------------------


def test_clean_client_has_a_bounded_request_timeout(monkeypatch):
    captured: dict[str, object] = {}
    google = ModuleType('google')
    genai = ModuleType('google.genai')

    def http_options(*, timeout):
        captured['timeout'] = timeout
        return SimpleNamespace(timeout=timeout)

    def client(**kwargs):
        captured['client_kwargs'] = kwargs
        response = SimpleNamespace(text=json.dumps({
            'text_clean': 'clean',
            'alt_phrasings': _phrasings(),
        }))
        return SimpleNamespace(models=SimpleNamespace(
            generate_content=lambda **_request: response,
        ))

    genai.Client = client
    genai.types = SimpleNamespace(HttpOptions=http_options)
    google.genai = genai
    monkeypatch.setitem(sys.modules, 'google', google)
    monkeypatch.setitem(sys.modules, 'google.genai', genai)
    monkeypatch.setenv(config.API_KEY_ENV, 'fixture-secret')

    assert commentary_cleaning._clean_once('raw') == {
        'text_clean': 'clean',
        'alt_phrasings': _phrasings(),
    }
    assert captured['timeout'] == config.LLM_REQUEST_TIMEOUT_S * 1000
    client_kwargs = captured['client_kwargs']
    assert isinstance(client_kwargs, dict)
    assert client_kwargs['http_options'].timeout == captured['timeout']


def test_openrouter_cli_requires_explicit_model_and_key_environment(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['commentary_cleaning', '--provider', 'openrouter', '--clean-only'])
    monkeypatch.setattr(
        commentary_cleaning,
        'run_clean',
        lambda **_kwargs: pytest.fail('cleaning must not run with incomplete OpenRouter settings'),
    )

    with pytest.raises(SystemExit) as caught:
        commentary_cleaning.main()

    assert caught.value.code == 2
    assert 'requires explicit --model and --api-key-environment' in capsys.readouterr().err


def test_run_clean_extends_in_place_and_keeps_k_phrasings(tmp_path, monkeypatch):
    """A kept video's chunks gain text_clean + K phrasings; original fields survive."""
    monkeypatch.setattr(commentary_cleaning, 'CHUNKS_DIR', tmp_path)
    monkeypatch.setattr(
        commentary_cleaning, 'call_clean_llm',
        lambda text: {'text_clean': f'CLEAN::{text}', 'alt_phrasings': _phrasings()},
    )
    sidecar = _write_sidecar(tmp_path, 'v1', [
        {'chunk_id': 'v1_c0', 'start': 1.0, 'end': 2.0, 'text': 'raw one'},
        {'chunk_id': 'v1_c1', 'start': 3.0, 'end': 4.0, 'text': 'raw two'},
    ])

    commentary_cleaning.run_clean(rows=[{'video_id': 'v1', 'keep': 'True'}])

    out = json.loads(sidecar.read_text(encoding='utf-8'))
    assert out[0]['text_clean'] == 'CLEAN::raw one'
    assert out[1]['text_clean'] == 'CLEAN::raw two'
    # Original coarse timestamps and chunk_id are left in place.
    assert (out[0]['start'], out[0]['end'], out[0]['chunk_id']) == (1.0, 2.0, 'v1_c0')
    assert len(out[0]['alt_phrasings']) == commentary_cleaning.ALT_PHRASINGS_K
    assert len(out[1]['alt_phrasings']) == commentary_cleaning.ALT_PHRASINGS_K


def test_run_clean_idempotent_skip_and_force_override(tmp_path, monkeypatch):
    """A chunk already carrying text_clean is skipped unless --force is set."""
    monkeypatch.setattr(commentary_cleaning, 'CHUNKS_DIR', tmp_path)
    calls = []

    def fake(text):
        calls.append(text)
        return {'text_clean': 'NEW', 'alt_phrasings': _phrasings()}

    monkeypatch.setattr(commentary_cleaning, 'call_clean_llm', fake)
    sidecar = _write_sidecar(tmp_path, 'v1', [{
        'chunk_id': 'v1_c0', 'start': 1.0, 'end': 2.0, 'text': 'raw',
        'text_clean': 'OLD', 'alt_phrasings': ['x'],
    }])
    rows = [{'video_id': 'v1', 'keep': 'True'}]

    # Idempotent: no call, text_clean untouched.
    commentary_cleaning.run_clean(rows=rows)
    assert calls == []
    assert json.loads(sidecar.read_text(encoding='utf-8'))[0]['text_clean'] == 'OLD'

    # Force: the chunk is re-cleaned.
    commentary_cleaning.run_clean(rows=rows, force=True)
    assert calls == ['raw']
    assert json.loads(sidecar.read_text(encoding='utf-8'))[0]['text_clean'] == 'NEW'


@pytest.mark.parametrize(('f1', 'expected'), [(0.79, False), (0.80, True)])
def test_run_clean_clean_pass_threshold_boundary(tmp_path, monkeypatch, f1, expected):
    """The stored score controls the flag at the configured inclusive boundary."""
    monkeypatch.setattr(
        commentary_cleaning, '_score_chunks', lambda chunks: {index: f1 for index in range(len(chunks))},
    )
    sidecar = _write_sidecar(tmp_path, 'v1', [{
        'chunk_id': 'v1_c0', 'start': 0.0, 'end': 1.0, 'text': 'raw', 'text_clean': 'clean',
    }])
    monkeypatch.setattr(commentary_cleaning, 'CHUNKS_DIR', tmp_path)

    commentary_cleaning.run_clean(rows=[{'video_id': 'v1', 'keep': 'True'}])

    assert json.loads(sidecar.read_text(encoding='utf-8'))[0]['clean_pass'] is expected
    assert json.loads(sidecar.read_text(encoding='utf-8'))[0]['bert_f1'] == f1


def test_run_clean_scores_existing_clean_without_calling_llm(tmp_path, monkeypatch):
    """A sidecar with text_clean but no score takes the score-only path."""
    monkeypatch.setattr(commentary_cleaning, 'CHUNKS_DIR', tmp_path)
    monkeypatch.setattr(commentary_cleaning, 'call_clean_llm', lambda _text: pytest.fail('LLM must not run'))
    monkeypatch.setattr(commentary_cleaning, '_score_chunks', lambda chunks: {0: 0.81})
    sidecar = _write_sidecar(tmp_path, 'v1', [{
        'chunk_id': 'v1_c0', 'start': 0.0, 'end': 1.0, 'text': 'raw', 'text_clean': 'clean',
    }])

    commentary_cleaning.run_clean(rows=[{'video_id': 'v1', 'keep': 'True'}])

    out = json.loads(sidecar.read_text(encoding='utf-8'))[0]
    assert out['bert_f1'] == 0.81
    assert out['clean_pass'] is True


def test_run_clean_blank_model_output_is_kept_and_fails(tmp_path, monkeypatch):
    """Blank model output is recorded without sending it to BERTScore."""
    monkeypatch.setattr(commentary_cleaning, 'CHUNKS_DIR', tmp_path)
    monkeypatch.setattr(commentary_cleaning, 'call_clean_llm', lambda _text: {
        'text_clean': '', 'alt_phrasings': _phrasings(),
    })
    monkeypatch.setattr(commentary_cleaning, '_score_chunks', lambda _chunks: pytest.fail('blank text must not score'))
    sidecar = _write_sidecar(tmp_path, 'v1', [{
        'chunk_id': 'v1_c0', 'start': 0.0, 'end': 1.0, 'text': 'raw',
    }])

    commentary_cleaning.run_clean(rows=[{'video_id': 'v1', 'keep': 'True'}])

    out = json.loads(sidecar.read_text(encoding='utf-8'))[0]
    assert out['text_clean'] == ''
    assert out['bert_f1'] == 0.0
    assert out['clean_pass'] is False


@pytest.mark.parametrize(('cuda_available', 'arch_list', 'expected'), [
    (False, [], 'cpu'),
    (True, ['sm_75'], 'cpu'),
    (True, ['sm_80'], 'cuda'),
])
def test_bert_score_device_checks_torch_architecture(monkeypatch, cuda_available, arch_list, expected):
    """CUDA is selected only when torch advertises the device architecture."""
    fake_cuda = SimpleNamespace(
        is_available=lambda: cuda_available,
        get_device_capability=lambda: (8, 0),
        get_arch_list=lambda: arch_list,
    )
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=fake_cuda))

    assert commentary_cleaning._bert_score_device() == expected


def test_run_clean_keep_filter_parses_not_truth_tests(tmp_path, monkeypatch):
    """Only rows whose keep equals the string 'True' are processed."""
    monkeypatch.setattr(commentary_cleaning, 'CHUNKS_DIR', tmp_path)
    seen = []

    def fake(text):
        seen.append(text)
        return {'text_clean': 'C', 'alt_phrasings': _phrasings()}

    monkeypatch.setattr(commentary_cleaning, 'call_clean_llm', fake)
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

    commentary_cleaning.run_clean(rows=rows)
    assert seen == ['keep_true']


def test_run_clean_propagates_failure_after_persisting_partial_work(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(commentary_cleaning, 'CHUNKS_DIR', tmp_path)
    calls: list[str] = []

    def fake(text):
        calls.append(text)
        if text == 'BAD one':
            raise commentary_cleaning.CleanError('boom')
        return {'text_clean': f'C::{text}', 'alt_phrasings': _phrasings()}

    monkeypatch.setattr(commentary_cleaning, 'call_clean_llm', fake)
    sidecar = _write_sidecar(tmp_path, 'video', [
        {'chunk_id': 'video_c0', 'start': 0.0, 'end': 1.0, 'text': 'good one'},
        {'chunk_id': 'video_c1', 'start': 1.0, 'end': 2.0, 'text': 'BAD one'},
    ])
    rows = [{'video_id': 'video', 'keep': 'True'}]

    with pytest.raises(commentary_cleaning.CleanError, match='boom'):
        commentary_cleaning.run_clean(rows=rows)
    chunks = json.loads(sidecar.read_text(encoding='utf-8'))
    assert chunks[0]['text_clean'] == 'C::good one'
    assert 'text_clean' not in chunks[1]
    assert calls == ['good one', 'BAD one']


def test_run_clean_daily_quota_stops_before_later_video(tmp_path, monkeypatch):
    monkeypatch.setattr(commentary_cleaning, 'CHUNKS_DIR', tmp_path)
    calls: list[str] = []

    def fake(text):
        calls.append(text)
        raise _llm_errors.DailyRequestQuotaError('daily quota exhausted')

    monkeypatch.setattr(commentary_cleaning, 'call_clean_llm', fake)
    _write_sidecar(tmp_path, 'v1', [
        {'chunk_id': 'v1_c0', 'start': 0.0, 'end': 1.0, 'text': 'first'},
    ])
    _write_sidecar(tmp_path, 'v2', [
        {'chunk_id': 'v2_c0', 'start': 0.0, 'end': 1.0, 'text': 'later'},
    ])
    rows = [{'video_id': 'v1', 'keep': 'True'}, {'video_id': 'v2', 'keep': 'True'}]

    with pytest.raises(_llm_errors.DailyRequestQuotaError, match='daily quota'):
        commentary_cleaning.run_clean(rows=rows)
    assert calls == ['first']


def test_run_clean_persists_partial_work_before_daily_quota(tmp_path, monkeypatch):
    monkeypatch.setattr(commentary_cleaning, 'CHUNKS_DIR', tmp_path)

    def fake(text):
        if text == 'first':
            return {'text_clean': 'clean first', 'alt_phrasings': _phrasings()}
        raise _llm_errors.DailyRequestQuotaError('daily quota exhausted')

    monkeypatch.setattr(commentary_cleaning, 'call_clean_llm', fake)
    sidecar = _write_sidecar(tmp_path, 'video', [
        {'chunk_id': 'video_c0', 'start': 0.0, 'end': 1.0, 'text': 'first'},
        {'chunk_id': 'video_c1', 'start': 1.0, 'end': 2.0, 'text': 'quota'},
    ])

    with pytest.raises(_llm_errors.DailyRequestQuotaError, match='daily quota'):
        commentary_cleaning.run_clean(rows=[{'video_id': 'video', 'keep': 'True'}])
    chunks = json.loads(sidecar.read_text(encoding='utf-8'))
    assert chunks[0]['text_clean'] == 'clean first'
    assert 'text_clean' not in chunks[1]


def test_call_clean_llm_retries_then_raises(monkeypatch):
    """The retry wrapper retries LLM_MAX_RETRIES times then raises CleanError."""
    attempts = []

    def boom(text):
        attempts.append(text)
        raise TimeoutError('request timed out')

    monkeypatch.setattr(commentary_cleaning, '_clean_once', boom)
    monkeypatch.setattr(commentary_cleaning.time, 'sleep', lambda _s: None)

    with pytest.raises(commentary_cleaning.CleanError):
        commentary_cleaning.call_clean_llm('hi')
    assert len(attempts) == commentary_cleaning.LLM_MAX_RETRIES


@pytest.mark.parametrize('quota_suffix', ['', '-FreeTier'])
def test_call_clean_llm_does_not_retry_exhausted_daily_quota(
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

    def exhausted(_text):
        nonlocal calls
        calls += 1
        raise DailyQuotaError('daily quota exhausted')

    monkeypatch.setattr(commentary_cleaning, '_clean_once', exhausted)
    monkeypatch.setattr(commentary_cleaning.time, 'sleep', sleeps.append)

    with pytest.raises(_llm_errors.DailyRequestQuotaError, match='daily request quota'):
        commentary_cleaning.call_clean_llm('hi')
    assert calls == 1
    assert sleeps == []


# -- Fine-timestamp pass -----------------------------------------------------


def test_padded_span_pads_and_clamps():
    """The span is padded FINE_PAD_S each side and the start never goes negative."""
    pad = commentary_cleaning.FINE_PAD_S
    assert commentary_cleaning._padded_span(10.0, 20.0) == (10.0 - pad, 20.0 + pad)
    start, end = commentary_cleaning._padded_span(0.5, 5.0)  # start within pad of zero clamps
    assert start == 0.0
    assert end == 5.0 + pad


def test_extract_span_ffmpeg_argv(tmp_path, monkeypatch):
    """ffmpeg is called with -ss/-to at the padded bounds and the wav output."""
    captured = {}

    def fake_run(argv, **kwargs):
        captured['argv'] = list(argv)
        captured['kwargs'] = kwargs

    monkeypatch.setattr(commentary_cleaning.subprocess, 'run', fake_run)
    video = tmp_path / 'vid.mp4'
    wav = tmp_path / 'out.wav'
    commentary_cleaning._extract_span(video, 12.5, 34.25, wav)

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

    monkeypatch.setattr(commentary_cleaning.subprocess, 'run', boom)
    assert commentary_cleaning.load_fine_models() is None  # this venv has no whisperx

    chunks_dir = tmp_path / 'chunks'
    chunks_dir.mkdir()
    sidecar = chunks_dir / 'vid.json'
    original = '[{"chunk_id": "c0", "start": 1.0, "end": 2.0, "text": "t"}]'
    sidecar.write_text(original, encoding='utf-8')
    monkeypatch.setattr(commentary_cleaning, 'CHUNKS_DIR', chunks_dir)

    rows = [{'video_id': 'vid', 'keep': 'True'}]
    commentary_cleaning.run_fine(tmp_path, rows)
    assert sidecar.read_text(encoding='utf-8') == original  # never rewritten
