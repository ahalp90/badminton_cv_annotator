"""Run LLM relevance triage over acquired transcripts.

For each video with a transcript: split the segments into overlapping windows so a
chunk straddling a boundary is not lost, ask a cheap fast LLM per window for the
qualitative-commentary chunks, aggregate them into chunks/<video_id>.json as
{chunk_id, start, end, text}, then decide keep per video by the three-legged D9
rule and write keep back into the candidates.csv column.

Run as: python -m scraper.relevance_triage (PYTHONPATH=src). The LLM call needs the
GEMINI_API_KEY env var set (referenced by name only, never read or logged) and the
google-genai SDK installed on the calling machine.

Ordinary exhausted calls are logged and skipped per video, moving that video
to a retry list. The batch blocks when every ordinary call fails, checked once
past a small floor so a dead endpoint stops early. A structured daily-request-
quota error instead terminates the batch immediately, before another request.
"""
import argparse
import json
import os
import time

from ._llm_errors import DailyRequestQuotaError, daily_request_quota_exhausted
from .config import (
    API_KEY_ENV,
    CHUNK_OVERLAP_S,
    CHUNK_WINDOW_S,
    CHUNKS_ABS_SAFE,
    CHUNKS_DIR,
    CHUNKS_MIN_SHORT,
    DENSITY_MIN_PER_MIN,
    LLM_BACKOFF_BASE_S,
    LLM_MAX_RETRIES,
    LLM_REQUEST_TIMEOUT_S,
    SHORT_VIDEO_MIN_S,
    TRIAGE_BLOCK_MIN_FAILURES,
    TRANSCRIPTS_DIR,
    TRIAGE_MAX_TOKENS,
    TRIAGE_MODEL,
    ensure_dirs,
    read_candidates,
    write_candidates,
)


class TriageError(RuntimeError):
    """Raised when an LLM triage call fails after all retries."""


def load_transcript(video_id: str) -> dict | None:
    """Load transcripts/<video_id>.json, or None when there is no sidecar.

    :param video_id: yt-dlp id.
    :return: the parsed transcript sidecar, or None when absent.
    """
    path = TRANSCRIPTS_DIR / f'{video_id}.json'
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def chunk_windows(segments: list[dict]) -> list[dict]:
    """Split segments into overlapping windows keyed by their segment timestamps.

    Windows span CHUNK_WINDOW_S seconds and step by (window - overlap), so a chunk
    landing in the overlap zone appears in two adjacent windows. A
    segment joins a window when its start falls inside [window_start, window_end).

    :param segments: the transcript segments, each with a start and end in seconds.
    :return: one window dict {start, end, segments} per non-empty window.
    """
    if not segments:
        return []
    step = CHUNK_WINDOW_S - CHUNK_OVERLAP_S
    end_time = max(segment['end'] for segment in segments)
    windows = []
    window_start = 0.0
    while window_start < end_time:
        window_end = window_start + CHUNK_WINDOW_S
        in_window = [
            segment for segment in segments
            if window_start <= segment['start'] < window_end
        ]
        if in_window:
            windows.append(
                {'start': window_start, 'end': window_end, 'segments': in_window}
            )
        window_start += step
    return windows


def build_triage_prompt(window: dict) -> tuple[str, str]:
    """Build the system and user prompts for one window.

    The prompt asks for qualitative assessments of play and returns commentary
    chunks with coarse start and end timestamps as a list.

    :param window: a window dict with a segments list.
    :return: (system_prompt, user_prompt).
    """
    system_prompt = (
        'You filter badminton match commentary transcripts. Keep only the '
        'commentary chunks that carry a qualitative assessment of play (praise, '
        'criticism, tactical read, shot quality). Return a JSON list of objects, '
        "each with 'start', 'end' (seconds, coarse) and 'text'. Return an empty "
        'list when the window carries no qualitative assessment.'
    )
    segment_lines = [
        f"[{segment['start']:.1f}-{segment['end']:.1f}] {segment['text']}"
        for segment in window['segments']
    ]
    user_prompt = (
        f"Window {window['start']:.1f}-{window['end']:.1f} seconds. "
        'Transcript segments:\n' + '\n'.join(segment_lines)
    )
    return system_prompt, user_prompt


def _call_once(system_prompt: str, user_prompt: str) -> list[dict]:
    """Send one triage window to the LLM and parse its JSON reply.

    genai.Client() reads the key from os.environ[API_KEY_ENV]; we only confirm the
    var is set, by name, and never read or log its value. A JSON parse failure
    raises so call_triage_llm's retry wrapper backs off and retries.

    :param system_prompt: system instruction pinning the filter behaviour.
    :param user_prompt: the window's transcript segments.
    :return: the parsed list of {start, end, text} chunk dicts.
    """
    # Function-local import: google-genai only installs on the machine that runs
    # the real triage. The test/CI venv does not carry it, and a module-level
    # import would break relevance triage there.
    from google import genai
    from google.genai import types

    if API_KEY_ENV not in os.environ:
        raise RuntimeError(f'{API_KEY_ENV} is not set')
    client = genai.Client(http_options=types.HttpOptions(
        timeout=LLM_REQUEST_TIMEOUT_S * 1000,
    ))
    response = client.models.generate_content(
        model=TRIAGE_MODEL,
        contents=user_prompt,
        config={
            'system_instruction': system_prompt,
            'max_output_tokens': TRIAGE_MAX_TOKENS,
            'response_mime_type': 'application/json',
        },
    )
    return json.loads(response.text)


def call_triage_llm(window: dict) -> list[dict]:
    """Call the triage LLM for one window with retry and exponential backoff.

    Real SDK calls hit rate limits and transient errors, and a malformed reply
    fails to parse; all retry unless the provider reports an exhausted daily
    quota. Raises ``TriageError`` after retry exhaustion or
    ``DailyRequestQuotaError`` when the whole batch must stop requesting.

    :param window: a window dict with a segments list.
    :return: the returned chunk dicts for the window.
    """
    system_prompt, user_prompt = build_triage_prompt(window)
    last_error: Exception | None = None
    for attempt in range(LLM_MAX_RETRIES):
        try:
            return _call_once(system_prompt, user_prompt)
        except Exception as error:  # noqa: BLE001 - retry SDK + JSON-parse errors alike
            last_error = error
            if daily_request_quota_exhausted(error):
                raise DailyRequestQuotaError(
                    "triage call stopped because the daily request quota is exhausted"
                ) from error
            if attempt == LLM_MAX_RETRIES - 1:
                break  # no backoff after the final attempt; raise straight away
            backoff = LLM_BACKOFF_BASE_S * (2 ** attempt)
            print(f'  LLM retry {attempt + 1}/{LLM_MAX_RETRIES} after {backoff:.1f}s: {error}')
            time.sleep(backoff)
    raise TriageError(f'triage call failed after {LLM_MAX_RETRIES} retries: {last_error}')


def _keep_decision(n_chunks: int, duration_s: str) -> bool:
    """Three-legged keep rule: keep when any leg passes.

    A duration-less row (rare; enrichment fills nearly all) can be judged
    neither short nor long, so only the length-independent absolute leg applies.

    :param n_chunks: qualitative chunk count for the video.
    :param duration_s: duration in seconds as a string; blank when unknown.
    :return: True to keep the video.
    """
    absolute_safe = n_chunks >= CHUNKS_ABS_SAFE
    if not duration_s:
        return absolute_safe
    seconds = float(duration_s)
    if seconds <= SHORT_VIDEO_MIN_S:
        return absolute_safe or n_chunks >= CHUNKS_MIN_SHORT
    # Long branch: seconds > SHORT_VIDEO_MIN_S > 0, so duration_min is never zero.
    density_per_min = n_chunks / (seconds / 60.0)
    return absolute_safe or density_per_min >= DENSITY_MIN_PER_MIN


def triage_video(video_id: str, duration_s: str) -> tuple[bool, list[dict]] | None:
    """Triage one video: window it, call the LLM per window, aggregate chunks.

    :param video_id: yt-dlp id.
    :param duration_s: the video's duration_s cell, for the keep rule.
    :return: (keep, chunks), or None when the video has no acquired transcript.
        Propagates ``TriageError`` so the batch can retry-list the video, or
        ``DailyRequestQuotaError`` when the entire batch must stop requesting.
    """
    transcript = load_transcript(video_id)
    if transcript is None:
        return None
    chunks: list[dict] = []
    for window in chunk_windows(transcript['segments']):
        for item in call_triage_llm(window):
            chunks.append({
                'chunk_id': f'{video_id}_c{len(chunks)}',
                'start': item['start'],
                'end': item['end'],
                'text': item['text'],
            })
    keep = _keep_decision(len(chunks), duration_s)
    return keep, chunks


def run_relevance_triage(rows: list[dict] | None = None) -> dict[str, bool]:
    """Triage every video with a transcript; write chunk sidecars and keep flags.

    Ordinary exhausted calls remain per-video failures; the run blocks when all
    of them fail, either mid-batch past a small floor or at the end. A structured
    daily-request-quota error terminates the batch immediately.

    :param rows: candidate rows; read from candidates.csv when None.
    :return: the keep decision per video_id.
    """
    ensure_dirs()
    if rows is None:
        rows = read_candidates()

    videos_with_transcript = 0
    failed = 0
    retry_list: list[str] = []
    keep_by_id: dict[str, bool] = {}

    for row in rows:
        video_id = row['video_id']
        if not (TRANSCRIPTS_DIR / f'{video_id}.json').exists():
            continue  # no acquired transcript; nothing to triage
        videos_with_transcript += 1
        try:
            outcome = triage_video(video_id, row.get('duration_s', ''))
        except TriageError as error:
            failed += 1
            retry_list.append(video_id)
            print(f'  TRIAGE FAILED {video_id}: {error}')
            # Circuit-break a dead endpoint: zero successes across the floor means
            # nothing is getting through; stop before burning the batch.
            if failed >= TRIAGE_BLOCK_MIN_FAILURES and failed == videos_with_transcript:
                raise RuntimeError(
                    f'Relevance triage: first {failed} triage calls all failed. '
                    f'Check the LLM endpoint.'
                ) from error
            continue
        if outcome is None:
            continue
        keep, chunks = outcome
        sidecar = CHUNKS_DIR / f'{video_id}.json'
        sidecar.write_text(json.dumps(chunks, indent=2), encoding='utf-8')
        keep_by_id[video_id] = keep
        print(f'  {video_id}: {len(chunks)} chunks, keep={keep}')

    if videos_with_transcript > 0 and failed == videos_with_transcript:
        # Every call failed: a dead endpoint, not scattered errors.
        raise RuntimeError(
            f'Relevance triage: all {failed} triage calls failed. Check the LLM endpoint.'
        )
    if retry_list:
        print(f"Retry list ({len(retry_list)}): {', '.join(retry_list)}")

    _write_keep_back(rows, keep_by_id)
    return keep_by_id


def _write_keep_back(rows: list[dict], keep_by_id: dict[str, bool]) -> None:
    """Fill the keep column for triaged videos and rewrite candidates.csv.

    INVARIANT: rewrites the same CANDIDATES_COLUMNS header; untriaged rows keep
    their blank keep value.

    :param rows: the candidate rows (mutated in place).
    :param keep_by_id: keep decision per triaged video_id.
    """
    for row in rows:
        if row['video_id'] in keep_by_id:
            row['keep'] = str(keep_by_id[row['video_id']])
    write_candidates(rows)
    print('Updated keep column in candidates.csv')


def main() -> None:
    argparse.ArgumentParser(
        description='Relevance triage: LLM relevance triage into chunks/<video_id>.json '
                    'and the keep column.',
    ).parse_args()
    run_relevance_triage()


if __name__ == '__main__':
    main()
