"""Re-time cleaned commentary chunks from word-aligned WhisperX transcripts (issue #136).

The triage LLM wrote each chunk's ``start`` and ``end`` as a coarse guess over a
10-minute window of caption or coarse-Whisper text, so a chunk can sit tens of
seconds from the words it quotes. This module finds each chunk's words in an
aligned transcript (``source: whisperx_aligned``, one time per word from the
wav2vec2 aligner) and replaces the coarse times with the first and last matched
word's times. Cleaned sidecars are never rewritten; a sibling sidecar per video
carries the re-timed rows plus the coarse times and a match score.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import StrEnum
import json
import math
from pathlib import Path
import re
import statistics

ALIGNED_SOURCE = 'whisperx_aligned'
# A coarse time can miss by a full 30 s Whisper window plus the LLM's own slack.
SEARCH_PAD_S = 60.0
# Matched tokens over the longer of the chunk and the matched span. Two ASR systems
# disagree on names and fillers, so a perfect score is not expected.
MIN_MATCH_RATIO = 0.5
MIN_BLOCK_TOKENS = 2  # one shared common word does not anchor a span
P90 = 0.9
_TOKEN_RE = re.compile(r"[^a-z0-9']+")


class AlignStatus(StrEnum):
    ALIGNED = 'aligned'
    UNMATCHED = 'unmatched'  # no span reached MIN_MATCH_RATIO; coarse times kept
    COLLISION = 'collision'  # aligned start duplicates an earlier chunk; coarse times kept


@dataclass(frozen=True)
class TimedToken:
    """One normalized token; times are None when the aligner skipped the word."""

    token: str
    start: float | None
    end: float | None


@dataclass(frozen=True)
class WordStream:
    """Every token of one aligned transcript in order, with a windowing time per token."""

    tokens: list[TimedToken]
    anchor_times: list[float]  # running max of word starts, so bisect can window by time


@dataclass(frozen=True)
class SpanMatch:
    """The densest run of shared token blocks for one chunk, in stream indices."""

    start_index: int
    end_index: int  # one past the last token
    matched_tokens: int
    chunk_tokens: int

    @property
    def ratio(self) -> float:
        return self.matched_tokens / max(self.chunk_tokens, self.end_index - self.start_index)


def normalize_tokens(text: str) -> list[str]:
    """Lower-case and split on anything but letters, digits, and apostrophes.

    :param text: chunk or word text.
    :return: non-empty tokens in order.
    """
    return [token for token in _TOKEN_RE.split(text.lower()) if token]


def load_word_stream(transcript: dict) -> WordStream:
    """Flatten an aligned transcript's words into a token stream.

    :param transcript: ``{source, segments: [{words: [{word, start?, end?}]}]}``.
    :return: the stream; raises when the source is not the aligned WhisperX pass.
    """
    if transcript.get('source') != ALIGNED_SOURCE:
        raise ValueError(f"expected source {ALIGNED_SOURCE!r}, found {transcript.get('source')!r}")
    tokens: list[TimedToken] = []
    anchor_times: list[float] = []
    last_start = 0.0
    for segment in transcript['segments']:
        for word in segment.get('words', ()):
            start = None if word.get('start') is None else float(word['start'])
            end = None if word.get('end') is None else float(word['end'])
            if start is not None:
                last_start = max(last_start, start)
            for token in normalize_tokens(str(word['word'])):
                tokens.append(TimedToken(token, start, end))
                anchor_times.append(last_start)
    return WordStream(tokens, anchor_times)


def match_chunk_tokens(
    chunk_tokens: list[str], stream: WordStream, low_s: float, high_s: float,
) -> SpanMatch | None:
    """Find the densest run of shared token blocks inside the window ``[low_s, high_s]``.

    Matching blocks come back in stream order. They are split into clusters wherever
    the gap between blocks exceeds the chunk length, so a shared "of the" far from the
    real span cannot stretch it, and the cluster with the most matched tokens wins.

    :param chunk_tokens: normalized chunk text.
    :param stream: the video's aligned word stream.
    :param low_s: window start in seconds.
    :param high_s: window end in seconds.
    :return: the best span, or None when no block of MIN_BLOCK_TOKENS exists.
    """
    region_low = bisect_left(stream.anchor_times, low_s)
    region_high = bisect_right(stream.anchor_times, high_s)
    region = [token.token for token in stream.tokens[region_low:region_high]]
    if not region or not chunk_tokens:
        return None
    matcher = SequenceMatcher(None, region, chunk_tokens, autojunk=False)
    blocks = [block for block in matcher.get_matching_blocks() if block.size >= MIN_BLOCK_TOKENS]
    if not blocks:
        return None
    clusters = [[blocks[0]]]
    for block in blocks[1:]:
        previous = clusters[-1][-1]
        if block.a - (previous.a + previous.size) > len(chunk_tokens):
            clusters.append([block])
        else:
            clusters[-1].append(block)
    best = max(clusters, key=lambda cluster: sum(block.size for block in cluster))
    return SpanMatch(
        start_index=region_low + best[0].a,
        end_index=region_low + best[-1].a + best[-1].size,
        matched_tokens=sum(block.size for block in best),
        chunk_tokens=len(chunk_tokens),
    )


def _span_times(stream: WordStream, match: SpanMatch) -> tuple[float, float] | None:
    """Earliest start and latest end among the span's timed words.

    :param stream: the video's aligned word stream.
    :param match: the span to time.
    :return: ``(start, end)`` in seconds, or None when no word in the span carries times.
    """
    starts = []
    ends = []
    for token in stream.tokens[match.start_index:match.end_index]:
        if token.start is not None and token.end is not None:
            starts.append(token.start)
            ends.append(token.end)
    if not starts:
        return None
    return min(starts), max(ends)


def _retime_one(chunk: dict, stream: WordStream) -> dict:
    """Re-time one chunk, recording the coarse times, status, match ratio, and shift.

    The match ratio is recorded even below MIN_MATCH_RATIO so the floor can be tuned
    later; only rows at or above it move.

    :param chunk: a cleaned chunk dict.
    :param stream: the video's aligned word stream.
    :return: a re-timed copy of the chunk.
    """
    row = dict(chunk)
    coarse_start = float(chunk['start'])
    coarse_end = float(chunk['end'])
    row['coarse_start'] = coarse_start
    row['coarse_end'] = coarse_end
    row['align_status'] = AlignStatus.UNMATCHED
    row['align_match_ratio'] = None
    row['align_shift_s'] = None
    tokens = normalize_tokens(str(chunk['text']))
    match = match_chunk_tokens(tokens, stream, coarse_start - SEARCH_PAD_S, coarse_end + SEARCH_PAD_S)
    if match is None:
        return row
    row['align_match_ratio'] = match.ratio
    times = None if match.ratio < MIN_MATCH_RATIO else _span_times(stream, match)
    if times is not None:
        row['start'], row['end'] = times
        row['align_status'] = AlignStatus.ALIGNED
        row['align_shift_s'] = times[0] - coarse_start
    return row


def retime_chunks(chunks: list[dict], stream: WordStream) -> list[dict]:
    """Re-time every chunk, then return the rows sorted by start with unique starts.

    Every coarse start stays reserved, because any row may fall back to it. An aligned
    row whose start lands on another row's coarse start, or on a better-matched aligned
    row's start, reverts to its coarse times as a collision. Duplicate coarse starts are
    an input fault and raise.

    :param chunks: cleaned chunk dicts carrying coarse ``start``, ``end``, and ``text``.
    :param stream: the video's aligned word stream.
    :return: re-timed copies; the inputs are left untouched.
    """
    rows = [_retime_one(chunk, stream) for chunk in chunks]
    _resolve_collisions(rows)
    rows.sort(key=lambda row: (float(row['start']), str(row['chunk_id'])))
    return rows


def _resolve_collisions(rows: list[dict]) -> None:
    """Demote aligned rows whose start is already taken, best match first, in place.

    :param rows: re-timed rows with unique coarse starts.
    :return: None; raises when the coarse starts themselves repeat.
    """
    reserved = {float(row['coarse_start']): str(row['chunk_id']) for row in rows}
    if len(reserved) != len(rows):
        raise ValueError('cleaned chunks carry duplicate coarse starts')
    aligned = [row for row in rows if row['align_status'] == AlignStatus.ALIGNED]
    aligned.sort(key=lambda row: (-float(row['align_match_ratio']), str(row['chunk_id'])))
    claimed: set[float] = set()
    for row in aligned:
        start = float(row['start'])
        owner = reserved.get(start)
        if start in claimed or (owner is not None and owner != str(row['chunk_id'])):
            row['start'], row['end'] = row['coarse_start'], row['coarse_end']
            row['align_status'] = AlignStatus.COLLISION
            row['align_shift_s'] = None
        else:
            claimed.add(start)
    if len({float(row['start']) for row in rows}) != len(rows):
        raise ValueError('re-timed starts are not unique after collision handling')


def _quantiles(values: list[float]) -> dict[str, float | int] | None:
    """Count, median, nearest-rank p90, and max of ``values``, or None when empty.

    :param values: the sample.
    :return: the summary dict, or None for an empty sample.
    """
    if not values:
        return None
    ordered = sorted(values)
    return {
        'count': len(ordered),
        'median': statistics.median(ordered),
        'p90': ordered[math.ceil(P90 * len(ordered)) - 1],
        'max': ordered[-1],
    }


def summarize(rows: list[dict]) -> dict:
    """Status counts plus shift and match-ratio quantiles for a set of re-timed rows.

    :param rows: re-timed rows from one or more videos.
    :return: ``chunks``, ``status_counts``, ``abs_shift_s`` (aligned rows), ``match_ratio``
        (aligned rows), and ``rejected_match_ratio`` (rows that matched below the floor).
    """
    status_counts = {status.value: 0 for status in AlignStatus}
    shifts = []
    ratios = []
    rejected_ratios = []
    for row in rows:
        status_counts[str(row['align_status'])] += 1
        if row['align_status'] == AlignStatus.ALIGNED:
            shifts.append(abs(float(row['align_shift_s'])))
            ratios.append(float(row['align_match_ratio']))
        elif row['align_match_ratio'] is not None:
            rejected_ratios.append(float(row['align_match_ratio']))
    return {
        'chunks': len(rows),
        'status_counts': status_counts,
        'abs_shift_s': _quantiles(shifts),
        'match_ratio': _quantiles(ratios),
        'rejected_match_ratio': _quantiles(rejected_ratios),
    }


def run(aligned_dir: Path, cleaned_dir: Path, out_dir: Path, summary_path: Path) -> dict:
    """Re-time every cleaned sidecar in ``cleaned_dir`` and write the sidecars and a summary.

    :param aligned_dir: ``<video_id>.json`` aligned transcripts.
    :param cleaned_dir: ``<video_id>.json`` cleaned chunk sidecars.
    :param out_dir: destination for the re-timed sidecars, same file names.
    :param summary_path: JSON summary with per-video and total status counts.
    :return: the summary.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    per_video = {}
    all_rows = []
    for cleaned_path in sorted(cleaned_dir.glob('*.json')):
        video_id = cleaned_path.stem
        aligned_path = aligned_dir / f'{video_id}.json'
        if not aligned_path.is_file():
            raise FileNotFoundError(f'{video_id}: no aligned transcript at {aligned_path}')
        stream = load_word_stream(json.loads(aligned_path.read_text(encoding='utf-8')))
        chunks = json.loads(cleaned_path.read_text(encoding='utf-8'))
        rows = retime_chunks(chunks, stream)
        (out_dir / cleaned_path.name).write_text(json.dumps(rows, indent=2), encoding='utf-8')
        per_video[video_id] = summarize(rows)
        all_rows.extend(rows)
        print(f"  {video_id}: {per_video[video_id]['status_counts']}")
    summary = {
        'source': ALIGNED_SOURCE,
        'search_pad_s': SEARCH_PAD_S,
        'min_match_ratio': MIN_MATCH_RATIO,
        'min_block_tokens': MIN_BLOCK_TOKENS,
        'videos': len(per_video),
        'totals': summarize(all_rows),
        'per_video': per_video,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main() -> None:
    """CLI: re-time every cleaned sidecar in a directory and print the totals."""
    parser = argparse.ArgumentParser(
        description='Re-time cleaned commentary chunks from aligned WhisperX transcripts.',
    )
    parser.add_argument('--aligned-dir', type=Path, required=True)
    parser.add_argument('--cleaned-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.aligned_dir, args.cleaned_dir, args.out_dir, args.summary)
    print(json.dumps(summary['totals'], indent=2))


if __name__ == '__main__':
    main()
