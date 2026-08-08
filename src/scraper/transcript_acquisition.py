"""Acquire one source-flagged coarse transcript per candidate video.

YouTube's own captions are the primary source: yt-dlp pulls the ASR track (plus a
human track where one exists), preferring the timestamped json3 format, parsed
into {start, end, text} segments. Where no English track exists, a WhisperX pass
on the remote GPU fills the gap (D23). One JSON sidecar per video lands at
transcripts/<video_id>.json carrying {source, segments}, source one of
youtube_asr or whisper.

Run as: python -m scraper.transcript_acquisition (PYTHONPATH=src).

Failure behaviour is log-and-skip per video. The batch blocks when more than
50% fails transcript acquisition, which signals an IP ban or systemic break.
This is checked mid-batch once past a small floor, so a banned run stops early
instead of hammering through the rest of the batch.
"""
import argparse
import json
import random
import re
import subprocess
import tempfile
import time
from pathlib import Path

from .config import (
    MAX_SLEEP_INTERVAL_S,
    SLEEP_INTERVAL_S,
    TRANSCRIPT_BLOCK_MIN_ATTEMPTS,
    TRANSCRIPT_FAIL_FRACTION_BLOCK,
    SUB_FORMAT,
    SUB_LANGS,
    SUBTITLE_TIMEOUT_S,
    TRANSCRIPTS_DIR,
    WHISPERX_COARSE_MODEL,
    YTDLP_BIN,
    check_ytdlp,
    ensure_dirs,
    read_candidates,
    ytdlp_throttle_args,
)

# VTT cue timestamp line: "HH:MM:SS.mmm --> HH:MM:SS.mmm".
_VTT_TIME = re.compile(
    r'(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})'
)

# WhisperX coarse-pass knobs (whisperx_settings_proposal.md s3). These belong in
# config beside WHISPERX_COARSE_MODEL, but config is frozen for this build, so
# they are named here instead. float16 on the GPU, batch 24 for the turbo pass.
_WHISPERX_DEVICE = 'cuda'
_WHISPERX_COMPUTE_TYPE = 'float16'
_WHISPERX_BATCH_SIZE = 24


def pull_subtitles(video_id: str, url: str, work_dir: str) -> Path | None:
    """Pull captions for one video into work_dir; return the caption file path.

    Prefers the json3 caption (per-segment start/end) and falls back to vtt.

    :param video_id: yt-dlp id, used for the output filename and log lines.
    :param url: webpage_url passed to yt-dlp.
    :param work_dir: throwaway dir the caption file is written into.
    :return: the caption file path, or None when yt-dlp fails or writes nothing.
    """
    output_template = str(Path(work_dir) / f'{video_id}.%(ext)s')
    cmd = [
        YTDLP_BIN, url,
        '--write-auto-subs',  # YouTube ASR track
        '--write-subs',  # human track where one exists
        '--sub-langs', SUB_LANGS,
        '--sub-format', SUB_FORMAT,
        '--skip-download',
        '--output', output_template,
        *ytdlp_throttle_args(include_subtitles=True),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=SUBTITLE_TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        print(f'  TIMEOUT subs {video_id}')
        return None
    if result.returncode != 0:
        print(f'  ERROR subs {video_id}: {result.stderr.strip()[:200]}')
        return None

    json3 = sorted(Path(work_dir).glob(f'{video_id}*.json3'))
    if json3:
        return json3[0]
    vtt = sorted(Path(work_dir).glob(f'{video_id}*.vtt'))
    if vtt:
        return vtt[0]
    return None


def parse_json3(path: Path) -> list[dict]:
    """Parse a YouTube json3 caption file into {start, end, text} segments.

    json3 events carry tStartMs / dDurationMs plus a segs list of utf8 pieces.
    Timing-only events (no segs, or empty text) are skipped.

    :param path: the json3 caption file.
    :return: one {start, end, text} dict per event with text, times in seconds.
    """
    data = json.loads(path.read_text(encoding='utf-8'))
    segments = []
    for event in data.get('events', []):
        segs = event.get('segs')
        if not segs:
            continue
        text = ''.join(seg.get('utf8', '') for seg in segs).strip()
        if not text:
            continue
        start = event.get('tStartMs', 0) / 1000.0
        end = start + event.get('dDurationMs', 0) / 1000.0
        segments.append({'start': start, 'end': end, 'text': text})
    return segments


def _vtt_seconds(hours: str, minutes: str, seconds: str, millis: str) -> float:
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000.0


def parse_vtt(path: Path) -> list[dict]:
    """Parse a WebVTT caption file into {start, end, text} segments.

    A cue is a timestamp line followed by one or more text lines up to a blank
    line. Basic parser: good enough for the coarse pass; fine timestamps are
    re-derived during commentary cleaning.

    :param path: the vtt caption file.
    :return: one {start, end, text} dict per cue, times in seconds.
    """
    lines = path.read_text(encoding='utf-8').splitlines()
    segments = []
    index = 0
    while index < len(lines):
        match = _VTT_TIME.search(lines[index])
        if not match:
            index += 1
            continue
        start = _vtt_seconds(*match.group(1, 2, 3, 4))
        end = _vtt_seconds(*match.group(5, 6, 7, 8))
        index += 1
        text_lines = []
        while index < len(lines) and lines[index].strip():
            text_lines.append(lines[index].strip())
            index += 1
        text = ' '.join(text_lines).strip()
        if text:
            segments.append({'start': start, 'end': end, 'text': text})
    return segments


def _download_audio(video_id: str, url: str, work_dir: str) -> Path | None:
    """Pull bestaudio for one video into work_dir; return the audio file path.

    Used only by the WhisperX fallback. Unlike the caption pull this needs the
    real audio, so -x extracts the track to a file (no --skip-download). Paces
    itself with a randomised pre-pull sleep even though run_transcript_acquisition already slept
    before the caption pull: the audio pull is a second, separate YouTube
    request, and the D22 pre-download pause applies per request, not per video.

    :param video_id: yt-dlp id, used for the output filename and log lines.
    :param url: webpage_url passed to yt-dlp.
    :param work_dir: throwaway dir the audio file is written into.
    :return: the downloaded audio file path, or None when yt-dlp writes nothing.
    """
    output_template = str(Path(work_dir) / f'{video_id}.%(ext)s')
    cmd = [
        YTDLP_BIN, url,
        '-f', 'bestaudio',
        '-x',
        '--output', output_template,
        *ytdlp_throttle_args(),
    ]
    time.sleep(random.uniform(SLEEP_INTERVAL_S, MAX_SLEEP_INTERVAL_S))
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=SUBTITLE_TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        print(f'  TIMEOUT audio {video_id}')
        return None
    if result.returncode != 0:
        print(f'  ERROR audio {video_id}: {result.stderr.strip()[:200]}')
        return None
    files = sorted(path for path in Path(work_dir).glob(f'{video_id}*') if path.is_file())
    return files[0] if files else None


def whisperx_fallback(video_id: str, url: str) -> list[dict] | None:
    """Coarse WhisperX transcription when no English caption track exists.

    GPU only (D23): the fallback needs an importable whisperx and a CUDA device.
    Absent either, it logs and returns None so the caller records the video as
    unresolved, exactly like a failed caption pull. Otherwise it pulls bestaudio,
    runs the coarse large-v3-turbo pass with diarisation and alignment off (the
    VAD-segment timestamps are enough here; fine times are re-derived during
    commentary cleaning), and maps the segments to {start, end, text}.

    :param video_id: yt-dlp id, for log lines and the temp audio filename.
    :param url: webpage_url passed to yt-dlp.
    :return: list of {start, end, text} segments, or None when unavailable/empty.
    """
    # Function-local imports: the fallback only runs on the remote GPU venv. The
    # test/CI venv carries neither whisperx nor CUDA, and a module-level import
    # would break transcript acquisition there. torch is imported only to read the CUDA flag; gc
    # is used to free VRAM between videos.
    try:
        import gc

        import torch
        import whisperx
    except ImportError:
        print(f'  WHISPERX unavailable, skipping fallback for {video_id}')
        return None
    if not torch.cuda.is_available():
        print(f'  WHISPERX needs CUDA (D23: GPU only), skipping {video_id}')
        return None

    with tempfile.TemporaryDirectory() as work_dir:
        audio_path = _download_audio(video_id, url, work_dir)
        if audio_path is None:
            return None
        model = whisperx.load_model(
            WHISPERX_COARSE_MODEL,
            device=_WHISPERX_DEVICE,
            compute_type=_WHISPERX_COMPUTE_TYPE,
        )
        try:
            audio = whisperx.load_audio(str(audio_path))
            result = model.transcribe(audio, batch_size=_WHISPERX_BATCH_SIZE)
        finally:
            # WhisperX leaks VRAM across a batch loop unless the model is freed
            # (whisperx_settings_proposal.md s6); transcript acquisition calls this per video.
            del model
            gc.collect()
            torch.cuda.empty_cache()

    segments = [
        {'start': seg['start'], 'end': seg['end'], 'text': seg['text'].strip()}
        for seg in result.get('segments', [])
        if seg.get('text', '').strip()
    ]
    return segments or None


def acquire_transcript(video_id: str, url: str) -> dict | None:
    """Return {source, segments} for one video, or None if unresolved.

    Caption files are pulled into a throwaway temp dir; only the parsed JSON
    sidecar is kept. Any YouTube caption (human or ASR) is flagged youtube_asr
    for the coarse pass; distinguishing the two is out of scope here.
    A None WhisperX fallback counts as unresolved.

    :param video_id: yt-dlp id.
    :param url: webpage_url passed to yt-dlp.
    :return: {source, segments} dict, or None when nothing resolved.
    """
    with tempfile.TemporaryDirectory() as work_dir:
        caption_path = pull_subtitles(video_id, url, work_dir)
        if caption_path is None:
            segments = whisperx_fallback(video_id, url)
            if not segments:
                return None
            return {'source': 'whisper', 'segments': segments}

        if caption_path.suffix == '.json3':
            segments = parse_json3(caption_path)
        else:
            segments = parse_vtt(caption_path)
        if not segments:
            print(f'  WARNING {video_id}: caption file parsed to zero segments')
            return None
        return {'source': 'youtube_asr', 'segments': segments}


def run_transcript_acquisition(rows: list[dict] | None = None) -> None:
    """Acquire a transcript per candidate and write the per-video sidecars.

    Blocks when the failure fraction crosses the configured 50% threshold,
    both mid-batch (past a small floor) and at the end of the run.

    :param rows: candidate rows; read from candidates.csv when None.
    """
    check_ytdlp()
    ensure_dirs()
    if rows is None:
        rows = read_candidates()
    total = len(rows)
    if total == 0:
        raise RuntimeError('Transcript acquisition: candidates.csv is empty. Run search indexing first.')

    failures = 0
    attempted = 0
    for row in rows:
        video_id = row['video_id']
        url = row['url']
        sidecar = TRANSCRIPTS_DIR / f'{video_id}.json'
        if sidecar.exists():
            print(f'  Skipping {video_id} (sidecar exists)')
            continue
        attempted += 1
        # Apply the randomised pre-video pause in Python. We pass --skip-download,
        # so yt-dlp's --sleep-interval and --max-sleep-interval would not fire.
        time.sleep(random.uniform(SLEEP_INTERVAL_S, MAX_SLEEP_INTERVAL_S))

        transcript = acquire_transcript(video_id, url)
        if transcript is None:
            failures += 1
            print(f'  FAILED transcript {video_id}')
            # Circuit-break mid-batch: past the floor, a failing majority means an
            # IP-ban or systemic break; carrying on only makes a ban worse.
            past_floor = attempted >= TRANSCRIPT_BLOCK_MIN_ATTEMPTS
            if past_floor and failures / attempted > TRANSCRIPT_FAIL_FRACTION_BLOCK:
                raise RuntimeError(
                    f'Transcript acquisition: {failures}/{attempted} attempted so far failed '
                    f'transcript acquisition, over the '
                    f'{TRANSCRIPT_FAIL_FRACTION_BLOCK:.0%} block threshold.'
                )
            continue
        sidecar.write_text(json.dumps(transcript, indent=2), encoding='utf-8')
        print(
            f"  Wrote {sidecar} ({transcript['source']}, "
            f"{len(transcript['segments'])} segments)"
        )

    if attempted == 0:
        print(f'Transcript acquisition: nothing to do, all {total} sidecars already exist')
        return
    # Fraction over videos attempted THIS run: a resumed run's existing sidecars
    # must not dilute the mass-failure signal.
    fail_fraction = failures / attempted
    print(f'Transcript acquisition: {failures}/{attempted} attempted failed ({fail_fraction:.0%})')
    if fail_fraction > TRANSCRIPT_FAIL_FRACTION_BLOCK:
        # Mass failure signals an IP ban or a systemic break.
        raise RuntimeError(
            f'Transcript acquisition: {fail_fraction:.0%} of the batch failed transcript acquisition, '
            f'over the {TRANSCRIPT_FAIL_FRACTION_BLOCK:.0%} block threshold.'
        )


def main() -> None:
    argparse.ArgumentParser(
        description='Transcript acquisition: pull transcripts to transcripts/<video_id>.json.',
    ).parse_args()
    run_transcript_acquisition()


if __name__ == '__main__':
    main()
