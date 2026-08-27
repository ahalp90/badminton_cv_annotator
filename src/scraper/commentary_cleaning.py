"""Clean commentary, generate alternate phrasing, and refine timestamps.

Two passes over the relevance-triage chunk sidecars (`chunks/<video_id>.json`, a list of
`{chunk_id, start, end, text}`), run for every video whose `candidates.csv` keep
column parses True:

  1. Clean pass (LLM): one call per chunk returns cleaned text plus a small pool
     of meaning-preserving paraphrases. Both extend the chunk dict in place
     (`text_clean`, `alt_phrasings`) and the sidecar is rewritten. The clean and
     paraphrase share one call budget per chunk. Idempotent:
     a chunk already carrying `text_clean` is skipped unless `--force`.

  2. Fine-timestamp pass (WhisperX): re-runs alignment on the audio span of each
     kept chunk to snap the coarse start/end to word-level boundaries. GPU only
     (D23); a no-op with a log line when WhisperX or CUDA is absent.

The real provider call is reached only outside the test venv; tests fake it via
monkeypatch. The optional SDK, WhisperX, and torch imports stay function-local
so importing this module must never fail.

Descended from the proof-of-concept relevance-triage skeleton. The LLM
retry/backoff wrapper and Gemini call shape are ported from there.
"""
import argparse
import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path

from ._llm_errors import (
    DailyRequestQuotaError,
    daily_request_quota_exhausted,
    provider_error_is_retryable,
)
from ._llm_provider import (
    LLMProvider,
    LLMSettings,
    generate_structured_json,
    resolve_cli_settings,
)
from .config import (
    ALT_PHRASINGS_K,
    API_KEY_ENV,
    CHUNKS_DIR,
    CLEAN_BERTSCORE_MIN,
    CLEAN_MODEL,
    LLM_BACKOFF_BASE_S,
    LLM_MAX_RETRIES,
    LLM_PROVIDER,
    LLM_REQUEST_TIMEOUT_S,
    TRIAGE_MAX_TOKENS,
    VIDEO_EXTENSIONS,
    WHISPERX_FINE_MODEL,
    read_candidates,
)

BERTSCORE_BATCH_SIZE = 16

# WhisperX fine-pass settings stay local because only the remote-GPU pass reads
# them.
FINE_PAD_S = 2.0  # pad each span so VAD does not clip the first/last word
FINE_BATCH_SIZE = 16
FINE_COMPUTE_TYPE = 'float16'

# The clean and paraphrase instructions share one call. The JSON shape is
# pinned so the response parses deterministically.
CLEAN_SYSTEM_PROMPT = (
    'You process one badminton commentary chunk at a time. Do two things and '
    'return them together as one JSON object.\n'
    '1. Clean this commentary chunk of transcription artefacts and verbal '
    'clutter without changing the meaning. Put the result in "text_clean".\n'
    f'2. Give {ALT_PHRASINGS_K} alternate phrasings that preserve meaning, for '
    'inter-epoch augmentation. Put them in "alt_phrasings" as a list of '
    f'{ALT_PHRASINGS_K} strings.\n'
    'Return only the JSON object '
    '{"text_clean": <string>, "alt_phrasings": [<string>, ...]}.'
)


class CleanError(RuntimeError):
    """Raised when an LLM clean call is rejected or exhausts its retries."""


CLEAN_RESPONSE_SCHEMA: dict[str, object] = {
    'type': 'object',
    'properties': {
        'text_clean': {'type': 'string'},
        'alt_phrasings': {
            'type': 'array',
            'items': {'type': 'string'},
            'minItems': ALT_PHRASINGS_K,
            'maxItems': ALT_PHRASINGS_K,
        },
    },
    'required': ['text_clean', 'alt_phrasings'],
    'additionalProperties': False,
}


def default_llm_settings() -> LLMSettings:
    """Return the unchanged Gemini cleaning defaults."""
    return LLMSettings.from_values(LLM_PROVIDER, CLEAN_MODEL, API_KEY_ENV)


def _clean_once(text: str, llm_settings: LLMSettings | None = None) -> dict:
    """Make one structured clean+paraphrase call for one chunk.

    The provider boundary receives the existing prompt and a JSON schema that
    matches the existing clean result. It reads the configured key without
    logging or persisting it.

    :param text: raw commentary text of one chunk.
    :return: dict with 'text_clean' (str) and 'alt_phrasings' (list of str).
    """
    parsed = generate_structured_json(
        default_llm_settings() if llm_settings is None else llm_settings,
        CLEAN_SYSTEM_PROMPT,
        text,
        CLEAN_RESPONSE_SCHEMA,
        'commentary_cleaning',
        TRIAGE_MAX_TOKENS,
        LLM_REQUEST_TIMEOUT_S,
    )
    if not isinstance(parsed, dict):
        raise TypeError('clean response is not a JSON object')
    return {
        'text_clean': parsed['text_clean'],
        'alt_phrasings': parsed['alt_phrasings'],
    }


def call_clean_llm(text: str, llm_settings: LLMSettings | None = None) -> dict:
    """Call the clean LLM for one chunk with retry and exponential backoff.

    Transient and native Gemini failures back off and retry. Provider failures
    classified as non-retryable stop immediately. An exhausted native Gemini
    daily quota raises ``DailyRequestQuotaError`` immediately.

    :param text: raw commentary text of one chunk.
    :return: dict with 'text_clean' and 'alt_phrasings'.
    """
    last_error: Exception | None = None
    for attempt in range(LLM_MAX_RETRIES):
        try:
            if llm_settings is None:
                return _clean_once(text)
            return _clean_once(text, llm_settings)
        except Exception as error:  # noqa: BLE001 - real code catches SDK errors
            last_error = error
            if daily_request_quota_exhausted(error):
                raise DailyRequestQuotaError(
                    "clean call stopped because the daily request quota is exhausted"
                ) from error
            if not provider_error_is_retryable(error):
                raise CleanError(f'clean call rejected by the provider: {error}') from error
            if attempt == LLM_MAX_RETRIES - 1:
                break  # no backoff after the final attempt; raise straight away
            backoff = LLM_BACKOFF_BASE_S * (2 ** attempt)
            print(f'  LLM retry {attempt + 1}/{LLM_MAX_RETRIES} after {backoff:.1f}s: {error}')
            time.sleep(backoff)
    raise CleanError(f'clean call failed after {LLM_MAX_RETRIES} retries: {last_error}')


def _bert_score_device() -> str:
    """Choose CUDA only when the installed torch build supports the GPU arch."""
    import torch

    capability = 'unavailable'
    device = 'cpu'
    if torch.cuda.is_available():
        capability_tuple = torch.cuda.get_device_capability()
        capability = capability_tuple
        arch = f'sm_{capability_tuple[0]}{capability_tuple[1]}'
        if arch in torch.cuda.get_arch_list():
            device = 'cuda'
    logging.warning('BERTScore device=%s capability=%s', device, capability)
    if device == 'cpu' and capability != 'unavailable':
        logging.warning('BERTScore CUDA architecture is unsupported by this torch build; using CPU')
    return device


def _score_chunks(chunks: list[dict]) -> dict[int, float]:
    """Score each text_clean against its raw text in one batched BERTScorer call.

    The caller passes only chunks needing scores, text and text_clean non-blank.
    """
    if not chunks:
        return {}

    from bert_score import BERTScorer

    scorer = BERTScorer(
        lang='en',
        rescale_with_baseline=False,
        device=_bert_score_device(),
    )
    candidates = [chunk['text_clean'] for chunk in chunks]
    references = [chunk['text'] for chunk in chunks]
    _, _, f1_scores = scorer.score(
        candidates, references, batch_size=BERTSCORE_BATCH_SIZE,
    )
    return {index: float(score) for index, score in enumerate(f1_scores)}


def run_clean(
    rows: list[dict] | None = None,
    force: bool = False,
    llm_settings: LLMSettings | None = None,
) -> dict[str, int]:
    """Run the clean+paraphrase pass over every kept video's chunk sidecar.

    Kept videos are those whose ``keep`` column parses ``== 'True'`` (parse, never
    truth-test: any non-empty cell is truthy, including 'False').
    Exhausting one chunk's bounded request attempts stops this optional stage
    so the coordinator can record it as unavailable instead of publishing a
    partial result as reusable success.

    :param rows: candidate rows; read from candidates.csv when None.
    :param force: re-clean chunks that already carry ``text_clean``.
    :return: cleaned-chunk count per video_id that had work done.
    """
    if rows is None:
        rows = read_candidates()
    kept = [row for row in rows if row.get('keep') == 'True']

    cleaned_by_id: dict[str, int] = {}
    loaded_sidecars: list[tuple[str, Path, list[dict], str]] = []

    for row in kept:
        video_id = row['video_id']
        sidecar = CHUNKS_DIR / f'{video_id}.json'
        if not sidecar.exists():
            print(f'  {video_id}: no chunk sidecar, skipping')
            continue

        chunks = json.loads(sidecar.read_text(encoding='utf-8'))
        for chunk in chunks:
            if 'text' not in chunk:
                raise KeyError(f"{video_id} chunk {chunk.get('chunk_id')}: no raw text field")
        to_clean = sum(1 for chunk in chunks if 'text_clean' not in chunk or force)
        cleaned = 0
        try:
            for chunk in chunks:
                if not chunk['text'].strip():
                    continue
                if 'text_clean' in chunk and not force:
                    continue
                result = (
                    call_clean_llm(chunk['text'])
                    if llm_settings is None
                    else call_clean_llm(chunk['text'], llm_settings)
                )
                chunk['text_clean'] = result['text_clean']
                chunk['alt_phrasings'] = result['alt_phrasings']
                chunk['_score_pending'] = True
                cleaned += 1
        except (CleanError, DailyRequestQuotaError) as error:
            print(f'  CLEAN FAILED {video_id}: {error}')
            if cleaned:  # persist the chunks cleaned before the failure
                for chunk in chunks:
                    chunk.pop('_score_pending', None)
                sidecar.write_text(json.dumps(chunks, indent=2), encoding='utf-8')
            raise

        original = json.dumps(chunks, indent=2)
        loaded_sidecars.append((video_id, sidecar, chunks, original))
        if cleaned:
            cleaned_by_id[video_id] = cleaned
        print(f'  {video_id}: cleaned {cleaned}/{to_clean} chunks')

    pending_chunks: list[dict] = []
    for _, _, chunks, _ in loaded_sidecars:
        for chunk in chunks:
            if 'text_clean' not in chunk:
                continue
            score_pending = 'bert_f1' not in chunk or chunk.pop('_score_pending', False)
            chunk['_score_pending'] = score_pending
            if score_pending and chunk['text_clean'].strip() and chunk['text'].strip():
                pending_chunks.append(chunk)

    scored = _score_chunks(pending_chunks) if pending_chunks else {}
    for index, chunk in enumerate(pending_chunks):
        chunk['bert_f1'] = scored[index]
    for _, _, chunks, _ in loaded_sidecars:
        for chunk in chunks:
            if 'text_clean' not in chunk:
                continue
            if not chunk['text_clean'].strip() and ('bert_f1' not in chunk or chunk.get('_score_pending')):
                chunk['bert_f1'] = 0.0
            if 'bert_f1' in chunk:
                chunk['clean_pass'] = chunk['bert_f1'] >= CLEAN_BERTSCORE_MIN
            chunk.pop('_score_pending', None)

    for _, sidecar, chunks, original in loaded_sidecars:
        updated = json.dumps(chunks, indent=2)
        if updated != original:
            sidecar.write_text(updated, encoding='utf-8')
    return cleaned_by_id


def _padded_span(start: float, end: float) -> tuple[float, float]:
    """Pad a chunk's coarse [start, end] by ``FINE_PAD_S`` each side, clamped at 0.

    Split out so the pad-and-clamp arithmetic is unit-testable without WhisperX.

    :param start: chunk start in absolute video seconds.
    :param end: chunk end in absolute video seconds.
    :return: (padded_start, padded_end); padded_start never negative.
    """
    return max(0.0, start - FINE_PAD_S), end + FINE_PAD_S


def _extract_span(video_path: Path, span_start: float, span_end: float, wav_path: Path) -> None:
    """Cut one padded audio span out of the video to a 16 kHz mono wav via ffmpeg.

    ``-ss``/``-to`` sit before ``-i`` so they seek and stop on the input timeline
    (absolute video seconds). We decode to 16 kHz mono PCM rather than stream-copy
    because a wav container needs PCM and ``whisperx.load_audio`` wants 16 kHz mono
    anyway; a raw copy of an AAC stream into .wav would not be readable.

    :param video_path: source video file.
    :param span_start: padded span start, absolute video seconds.
    :param span_end: padded span end, absolute video seconds.
    :param wav_path: output wav path in the caller's temp dir.
    """
    subprocess.run(
        [
            'ffmpeg', '-nostdin', '-y',
            '-ss', f'{span_start:.3f}',  # input seek to the padded span start
            '-to', f'{span_end:.3f}',  # stop reading the input at the padded end
            '-i', str(video_path),
            '-vn',  # audio only
            '-ac', '1', '-ar', '16000',  # 16 kHz mono, what whisperx.load_audio wants
            str(wav_path),
        ],
        check=True,
        capture_output=True,
    )


def load_fine_models() -> tuple | None:
    """Load the WhisperX fine-pass models once for a whole run, or None off-GPU.

    WhisperX and torch are imported here, not at module scope, so the test venv
    (which has neither) can still import this module. The pass is GPU only
    (D23); diarisation stays off per the settings doc. One load serves every
    video in the run: loading per video leaks VRAM until CUDA OOM (agy F1).

    :return: (model, align_model, align_meta), or None when whisperx or CUDA is
        absent (the caller leaves coarse timestamps in place).
    """
    try:
        import torch
        import whisperx
    except ImportError:
        print('  WhisperX/torch unavailable; fine pass leaves coarse timestamps')
        return None
    if not torch.cuda.is_available():
        print('  No CUDA device; WhisperX fine pass is GPU only (D23)')
        return None

    device = 'cuda'
    model = whisperx.load_model(
        WHISPERX_FINE_MODEL, device, compute_type=FINE_COMPUTE_TYPE,
        vad_method='pyannote',
        asr_options={'suppress_numerals': True, 'hallucination_silence_threshold': 2.0},
    )
    align_model, align_meta = whisperx.load_align_model(language_code='en', device=device)
    return model, align_model, align_meta


def refine_timestamps(video_path: str, chunks: list[dict], models: tuple) -> list[dict]:
    """Snap each chunk's coarse start/end to WhisperX word boundaries.

    Chunk-local by default: only the padded audio span of each kept chunk is
    re-aligned, so compute scales with kept-chunk minutes not full runtime. The
    fallback if span extraction proves unreliable is a single whole-video
    WhisperX pass, which is not built here.

    :param video_path: source video file for the kept chunks.
    :param chunks: chunk dicts carrying coarse 'start'/'end'; mutated in place.
    :param models: (model, align_model, align_meta) from load_fine_models.
    :return: the same chunks, with 'start'/'end' snapped where alignment landed.
    """
    import whisperx  # importable by construction: models came from load_fine_models

    model, align_model, align_meta = models
    device = 'cuda'

    with tempfile.TemporaryDirectory() as tmp_dir:
        for chunk in chunks:
            span_start, span_end = _padded_span(chunk['start'], chunk['end'])
            wav_path = Path(tmp_dir) / f"{chunk['chunk_id']}.wav"
            _extract_span(Path(video_path), span_start, span_end, wav_path)

            audio = whisperx.load_audio(str(wav_path))
            result = model.transcribe(audio, batch_size=FINE_BATCH_SIZE)
            aligned = whisperx.align(
                result['segments'], align_model, align_meta, audio, device,
                return_char_alignments=False,
            )
            words = [
                word for segment in aligned['segments']
                for word in segment.get('words', [])
                if 'start' in word and 'end' in word
            ]
            if not words:
                continue  # nothing aligned in the span; keep the coarse times
            # The wav starts at absolute video time span_start, so word times
            # (measured from the wav's t=0) shift back to absolute by adding it.
            chunk['start'] = words[0]['start'] + span_start
            chunk['end'] = words[-1]['end'] + span_start
    return chunks


def _find_video(video_dir: Path, video_id: str) -> Path | None:
    """Return one exact or legacy spaced-name source, or None when absent."""
    if not video_dir.is_dir():
        return None
    matches = [
        candidate
        for candidate in sorted(video_dir.iterdir(), key=lambda path: path.name)
        if candidate.is_file()
        and (candidate.stem == video_id or candidate.stem.startswith(f'{video_id} '))
        and candidate.suffix.lower() in VIDEO_EXTENSIONS
        and not (
            '.f' in candidate.stem
            and candidate.stem.rpartition('.f')[2].isdecimal()
        )
    ]
    if len(matches) > 1:
        raise ValueError(f'multiple source videos found for {video_id!r}: {matches}')
    return None if not matches else matches[0]


def run_fine(video_dir: Path, rows: list[dict] | None = None) -> None:
    """Run the WhisperX fine-timestamp pass over every kept video's chunks.

    :param video_dir: dir holding the source videos as <video_id>.<ext>.
    :param rows: candidate rows; read from candidates.csv when None.
    """
    if rows is None:
        rows = read_candidates()
    kept = [row for row in rows if row.get('keep') == 'True']

    models = load_fine_models()
    if models is None:
        print(f'  Fine pass skipped for all {len(kept)} kept videos; coarse timestamps stand')
        return

    try:
        for row in kept:
            video_id = row['video_id']
            sidecar = CHUNKS_DIR / f'{video_id}.json'
            if not sidecar.exists():
                print(f'  {video_id}: no chunk sidecar, skipping')
                continue
            video_path = _find_video(video_dir, video_id)
            if video_path is None:
                print(f'  {video_id}: no video file in {video_dir}, skipping fine pass')
                continue
            chunks = json.loads(sidecar.read_text(encoding='utf-8'))
            refine_timestamps(str(video_path), chunks, models)
            sidecar.write_text(json.dumps(chunks, indent=2), encoding='utf-8')
            print(f'  {video_id}: refined {len(chunks)} chunk timestamps')
    finally:
        # One model load serves the whole run; free the VRAM on the way out
        # (mirrors the transcript acquisition fallback's cleanup, whisperx_settings_proposal s6).
        import gc

        import torch
        del models
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Commentary cleaning: LLM clean+paraphrase pass and WhisperX fine '
                    'timestamps over the relevance-triage chunk sidecars.',
    )
    parser.add_argument('--clean-only', action='store_true',
                        help='Run only the LLM clean+paraphrase pass')
    parser.add_argument('--fine-only', action='store_true',
                        help='Run only the WhisperX fine-timestamp pass')
    parser.add_argument('--force', action='store_true',
                        help='Re-clean chunks that already carry text_clean')
    parser.add_argument('--video-dir', type=Path,
                        help='Dir of <video_id>.<ext> videos for the fine pass')
    parser.add_argument(
        '--provider',
        choices=[provider.value for provider in LLMProvider],
        default=LLM_PROVIDER,
    )
    parser.add_argument('--model')
    parser.add_argument('--api-key-environment')
    args = parser.parse_args()

    run_clean_pass = not args.fine_only
    run_fine_pass = not args.clean_only

    if run_clean_pass:
        print('=== Commentary clean pass ===')
        try:
            settings = resolve_cli_settings(
                args.provider,
                args.model,
                args.api_key_environment,
                gemini_model=CLEAN_MODEL,
                gemini_api_key_environment=API_KEY_ENV,
            )
        except ValueError as error:
            parser.error(str(error))
        run_clean(force=args.force, llm_settings=settings)

    if run_fine_pass:
        if args.video_dir is None:
            parser.error('--video-dir is required for the fine-timestamp pass')
        print('=== Commentary fine-timestamp pass ===')
        run_fine(args.video_dir)


if __name__ == '__main__':
    main()
