"""Stage 10: commentary clean pass, alt-phrasings, and fine timestamps (spec s9).

Two passes over the stage-3 chunk sidecars (`chunks/<video_id>.json`, a list of
`{chunk_id, start, end, text}`), run for every video whose `candidates.csv` keep
column parses True:

  1. Clean pass (LLM): one call per chunk returns cleaned text plus a small pool
     of meaning-preserving paraphrases. Both extend the chunk dict in place
     (`text_clean`, `alt_phrasings`) and the sidecar is rewritten. The clean and
     paraphrase share one call budget per chunk (dataset_schema s5). Idempotent:
     a chunk already carrying `text_clean` is skipped unless `--force`.

  2. Fine-timestamp pass (WhisperX): re-runs alignment on the audio span of each
     kept chunk to snap the coarse start/end to word-level boundaries. GPU only
     (D23); a no-op with a log line when WhisperX or CUDA is absent.

The real Gemini call is reached only outside the test venv (google-genai is not
installed there); tests fake it via monkeypatch. The WhisperX and torch imports
are function-local for the same reason: importing this module must never fail.

Descended from the poc stage-3 triage skeleton (poc/stage3_triage.py); the LLM
retry/backoff wrapper and the Gemini call shape are ported from there.
"""
import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from .config import (
    ALT_PHRASINGS_K,
    API_KEY_ENV,
    CHUNKS_DIR,
    CLEAN_MODEL,
    LLM_BACKOFF_BASE_S,
    LLM_MAX_RETRIES,
    TRIAGE_MAX_TOKENS,
    WHISPERX_FINE_MODEL,
    read_candidates,
)

# Video extensions the fine pass will accept for a <video_id>.<ext> lookup.
_VIDEO_EXTS = {'.mp4', '.mkv', '.webm', '.avi', '.mov'}

# WhisperX fine-pass settings signed off in whisperx_settings_proposal.md s2.
# Not in config: whisperx is remote-GPU only and nothing else reads these.
FINE_PAD_S = 2.0          # pad each span so VAD does not clip the first/last word
FINE_BATCH_SIZE = 16      # batched inference batch size (settings doc s2)
FINE_COMPUTE_TYPE = 'float16'

# Merged system prompt: the clean and paraphrase instructions from spec s9 in one
# call (dataset_schema s5 shares the budget). Both spec phrasings are kept
# recognisable; the JSON shape is pinned so the response parses deterministically.
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
    """Raised when an LLM clean call fails after all retries."""


def _clean_once(text: str) -> dict:
    """Single Gemini clean+paraphrase call for one chunk. Never runs in tests.

    Builds the request exactly as poc/stage3_triage.py does: model, contents, and
    a config carrying the system instruction, token cap and a JSON response mime
    type. The client reads the key from ``os.environ[API_KEY_ENV]``; we confirm
    the env var is set by name only and never read or log its value.

    :param text: raw commentary text of one chunk.
    :return: dict with 'text_clean' (str) and 'alt_phrasings' (list of str).
    """
    request_params = {
        'model': CLEAN_MODEL,
        'contents': text,
        'config': {
            'system_instruction': CLEAN_SYSTEM_PROMPT,
            'max_output_tokens': TRIAGE_MAX_TOKENS,
            'response_mime_type': 'application/json',
        },
    }

    from google import genai

    if API_KEY_ENV not in os.environ:
        raise RuntimeError(f'{API_KEY_ENV} is not set')
    client = genai.Client()
    response = client.models.generate_content(**request_params)
    parsed = json.loads(response.text)
    return {
        'text_clean': parsed['text_clean'],
        'alt_phrasings': parsed['alt_phrasings'],
    }


def call_clean_llm(text: str) -> dict:
    """Call the clean LLM for one chunk with retry and exponential backoff.

    Ported from the poc stage-3 wrapper: catch broadly (the real SDK raises typed
    errors on rate limits and transient faults), back off ``LLM_BACKOFF_BASE_S``
    doubled per attempt, and raise CleanError after ``LLM_MAX_RETRIES`` so the
    caller can log-and-skip the video.

    :param text: raw commentary text of one chunk.
    :return: dict with 'text_clean' and 'alt_phrasings'.
    """
    last_error: Exception | None = None
    for attempt in range(LLM_MAX_RETRIES):
        try:
            return _clean_once(text)
        except Exception as error:  # noqa: BLE001 - real code catches SDK errors
            last_error = error
            backoff = LLM_BACKOFF_BASE_S * (2 ** attempt)
            print(f'  LLM retry {attempt + 1}/{LLM_MAX_RETRIES} after {backoff:.1f}s: {error}')
            time.sleep(backoff)
    raise CleanError(f'clean call failed after {LLM_MAX_RETRIES} retries: {last_error}')


def run_clean(rows: list[dict] | None = None, force: bool = False) -> dict[str, int]:
    """Run the clean+paraphrase pass over every kept video's chunk sidecar.

    Kept videos are those whose ``keep`` column parses ``== 'True'`` (parse, never
    truth-test: any non-empty cell is truthy, 'False' included, per config s2).
    Failure is log-and-skip per video; the run blocks (raises) only when every
    LLM call attempted failed, which signals a dead endpoint (mirrors spec s4).

    :param rows: candidate rows; read from candidates.csv when None.
    :param force: re-clean chunks that already carry ``text_clean``.
    :return: cleaned-chunk count per video_id that had work done.
    """
    if rows is None:
        rows = read_candidates()
    kept = [row for row in rows if row.get('keep') == 'True']

    attempted = 0  # LLM calls started this run
    failed = 0     # LLM calls that died after retries
    cleaned_by_id: dict[str, int] = {}

    for row in kept:
        video_id = row['video_id']
        sidecar = CHUNKS_DIR / f'{video_id}.json'
        if not sidecar.exists():
            print(f'  {video_id}: no chunk sidecar, skipping')
            continue

        chunks = json.loads(sidecar.read_text(encoding='utf-8'))
        to_clean = sum(1 for chunk in chunks if 'text_clean' not in chunk or force)
        cleaned = 0
        try:
            for chunk in chunks:
                if 'text_clean' in chunk and not force:
                    continue
                attempted += 1
                result = call_clean_llm(chunk['text'])
                chunk['text_clean'] = result['text_clean']
                chunk['alt_phrasings'] = result['alt_phrasings']
                cleaned += 1
        except CleanError as error:
            failed += 1
            print(f'  CLEAN FAILED {video_id}: {error}')
            if cleaned:  # persist the chunks cleaned before the failure
                sidecar.write_text(json.dumps(chunks, indent=2), encoding='utf-8')
            continue

        if cleaned:
            sidecar.write_text(json.dumps(chunks, indent=2), encoding='utf-8')
            cleaned_by_id[video_id] = cleaned
        print(f'  {video_id}: cleaned {cleaned}/{to_clean} chunks')

    if attempted > 0 and failed == attempted:
        # Every call that was attempted failed: a dead endpoint, not scattered
        # errors (spec s4). failed counts one per dead video, attempted counts
        # every chunk call, so this fires only when the first call of every video
        # died and nothing got through.
        raise RuntimeError(
            f'Stage 10 clean: all {attempted} LLM calls failed. Check the endpoint.'
        )
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
            '-to', f'{span_end:.3f}',    # stop reading the input at the padded end
            '-i', str(video_path),
            '-vn',                        # audio only
            '-ac', '1', '-ar', '16000',  # 16 kHz mono, what whisperx.load_audio wants
            str(wav_path),
        ],
        check=True,
        capture_output=True,
    )


def refine_timestamps(video_path: str, chunks: list[dict]) -> list[dict]:
    """Snap each chunk's coarse start/end to WhisperX word boundaries (spec s9).

    Chunk-local by default: only the padded audio span of each kept chunk is
    re-aligned, so compute scales with kept-chunk minutes not full runtime. The
    documented fallback if span extraction proves fiddly is a single whole-video
    WhisperX pass (spec s9, whisperx_settings_proposal.md s2); not built here.

    WhisperX and torch are imported here, not at module scope, so the test venv
    (which has neither) can still import this module. When WhisperX or CUDA is
    missing the chunks are returned unchanged with a log line: the pass is GPU
    only (D23). Diarisation stays off per the settings doc.

    :param video_path: source video file for the kept chunks.
    :param chunks: chunk dicts carrying coarse 'start'/'end'; mutated in place.
    :return: the same chunks, with 'start'/'end' snapped where alignment landed.
    """
    try:
        import torch
        import whisperx
    except ImportError:
        print(f'  WhisperX/torch unavailable; leaving {len(chunks)} chunks at coarse timestamps')
        return chunks
    if not torch.cuda.is_available():
        print(f'  No CUDA device; WhisperX fine pass is GPU only (D23), '
              f'leaving {len(chunks)} chunks coarse')
        return chunks

    device = 'cuda'
    model = whisperx.load_model(
        WHISPERX_FINE_MODEL, device, compute_type=FINE_COMPUTE_TYPE,
        vad_method='pyannote',
        asr_options={'suppress_numerals': True, 'hallucination_silence_threshold': 2.0},
    )
    align_model, align_meta = whisperx.load_align_model(language_code='en', device=device)

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
    """Return the <video_id>.<ext> file in video_dir, or None when absent."""
    for candidate in sorted(video_dir.glob(f'{video_id}.*')):
        if candidate.suffix.lower() in _VIDEO_EXTS:
            return candidate
    return None


def run_fine(video_dir: Path, rows: list[dict] | None = None) -> None:
    """Run the WhisperX fine-timestamp pass over every kept video's chunks.

    :param video_dir: dir holding the source videos as <video_id>.<ext>.
    :param rows: candidate rows; read from candidates.csv when None.
    """
    if rows is None:
        rows = read_candidates()
    kept = [row for row in rows if row.get('keep') == 'True']

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
        refine_timestamps(str(video_path), chunks)
        sidecar.write_text(json.dumps(chunks, indent=2), encoding='utf-8')
        print(f'  {video_id}: refined {len(chunks)} chunk timestamps')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Stage 10: LLM clean+paraphrase pass and WhisperX fine '
                    'timestamps over the stage-3 chunk sidecars.',
    )
    parser.add_argument('--clean-only', action='store_true',
                        help='Run only the LLM clean+paraphrase pass')
    parser.add_argument('--fine-only', action='store_true',
                        help='Run only the WhisperX fine-timestamp pass')
    parser.add_argument('--force', action='store_true',
                        help='Re-clean chunks that already carry text_clean')
    parser.add_argument('--video-dir', type=Path,
                        help='Dir of <video_id>.<ext> videos for the fine pass')
    args = parser.parse_args()

    run_clean_pass = not args.fine_only
    run_fine_pass = not args.clean_only

    if run_clean_pass:
        print('=== Stage 10 clean pass ===')
        run_clean(force=args.force)

    if run_fine_pass:
        if args.video_dir is None:
            parser.error('--video-dir is required for the fine-timestamp pass')
        print('=== Stage 10 fine-timestamp pass ===')
        run_fine(args.video_dir)


if __name__ == '__main__':
    main()
