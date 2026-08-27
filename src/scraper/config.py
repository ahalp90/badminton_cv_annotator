"""Shared config for the badminton-commentary scraper.

Single source of truth for the file contracts and named constants the components
share: output paths, the candidates.csv column set, yt-dlp throttle flags, the
metadata screens, chunking and keep thresholds, LLM settings, and the
rally-segmentation and replay-mask trajectory rules. Every component imports
from here so the column order, sidecar layout, and rate-limit values live in
one place.

See ``docs/scraper_pipeline/scraper_architecture.md`` for the current public
file contracts.

Comments describe the current contract and identify provisional values that
still need tuning.
"""
import csv
import shutil
from pathlib import Path

from annotator.config import CONTACT_FRAMES_CSV, MASKS_DIR, RALLY_SPANS_CSV, SCRAPE_DIR  # noqa: F401

# ---------------------------------------------------------------------------
# Output layout
# ---------------------------------------------------------------------------
# One scrape root holds the flat CSVs plus the per-video sidecar dirs.
# SCRAPE_DIR, MASKS_DIR, RALLY_SPANS_CSV and CONTACT_FRAMES_CSV are
# annotator-owned (annotator.config); imported inward here so this module's
# own consumers keep the same names and values.
CANDIDATES_CSV = SCRAPE_DIR / 'candidates.csv'
VIDEOS_DIR = SCRAPE_DIR / 'videos'
VIDEO_EXTENSIONS = frozenset({'.mp4', '.mkv', '.webm', '.avi', '.mov'})
SOURCES_MANIFEST_NAME = 'sources.toml'
TRANSCRIPTS_DIR = SCRAPE_DIR / 'transcripts'
CHUNKS_DIR = SCRAPE_DIR / 'chunks'
PAIRS_CSV = SCRAPE_DIR / 'rally_commentary_pairs.csv'

# ---------------------------------------------------------------------------
# candidates.csv contract
# ---------------------------------------------------------------------------
# Column order is fixed here. INVARIANT: search indexing writes this header, relevance triage
# rewrites the same file with the same header (only keep changes), and the
# section 10 human packet later fills triage_verdict.
# Bool columns serialise as the CSV strings 'True'/'False' (keep is also blank
# before relevance triage fills it). Consumers must parse (== 'True'), never truth-test
# a raw cell: any non-empty string is truthy, 'False' included.
CANDIDATES_COLUMNS = [
    'video_id',  # yt-dlp id
    'url',  # webpage_url
    'title',
    'channel',
    'duration_s',
    'upload_date',
    'search_term',  # provenance; comma-joined when several terms surface a video
    'substream',  # 'match' or 'instructional', set by the search family (D24)
    'doubles_suspect',  # bool, title/metadata keyword screen
    'duration_suspect',  # bool, duration outside the match-length band
    'upload_date_suspect',  # bool, always False while the floor is off
    'keep',  # bool, appended by relevance triage; blank at index time
    'triage_verdict',  # keep/drop/uncertain, human packet; blank at index time
]

SUBSTREAM_MATCH = 'match'
SUBSTREAM_INSTRUCTIONAL = 'instructional'

# ---------------------------------------------------------------------------
# Search indexing
# ---------------------------------------------------------------------------
YTDLP_BIN = 'yt-dlp'  # downloader command shared by scraper and ShuttleSet adapter

YTSEARCH_COUNT = 50

# Tab-separated --print template. --print implies --simulate, so the flat index
# downloads no bytes. Field order must match FLAT_PRINT_FIELDS.
FLAT_PRINT_TEMPLATE = (
    '%(id)s\t%(webpage_url)s\t%(title)s\t%(channel)s\t%(duration)s\t%(upload_date)s'
)
FLAT_PRINT_FIELDS = ['video_id', 'url', 'title', 'channel', 'duration_s', 'upload_date']

# Seed search-term families. Terms remain provisional, while the families are
# fixed. Each key is the substream its rows carry.
SEARCH_TERMS = {
    SUBSTREAM_MATCH: [
        # Professional match VODs
        'BWF World Tour final full match',
        'badminton singles full match commentary',
        'olympic badminton singles gold medal match',
        # Amateur games with commentary
        'club badminton singles match commentary',
        'local badminton tournament singles full',
        'amateur badminton singles with commentary',
        # Coaching or analysis videos
        'badminton match analysis breakdown',
        'badminton singles tactics explained',
        'pro badminton point analysis commentary',
    ],
    SUBSTREAM_INSTRUCTIONAL: [
        # Coach-review sub-stream (D24): viewer clips reviewed by coaches
        'badminton clips coach review',
        'badminton coach reacts',
        'rate my badminton',
    ],
}

# Cheap metadata screens. Flag never drop: a dropped row loses
# its provenance. Instructional-substream rows skip the short-duration flag
# (D24; coach-review clips run short by design).
DURATION_MIN_S = 10 * 60  # flag under 10 min
DURATION_MAX_S = 240 * 60  # flag over 240 min
# Upload-date floor off per D8. A YYYYMMDD string when ever set; None disables.
UPLOAD_DATE_FLOOR = None

# Doubles keyword screen. Long phrases match as case-insensitive
# substrings; the short abbreviations match only as whole tokens so 'md'/'wd'/
# 'xd' do not fire inside unrelated words (e.g. 'commander', 'crowd').
DOUBLES_KEYWORD_PHRASES = ['doubles', 'mixed doubles']
DOUBLES_KEYWORD_TOKENS = ['xd', 'md', 'wd']
# Known pair-name patterns need a curated list before they can be added.

# ---------------------------------------------------------------------------
# Transcript acquisition
# ---------------------------------------------------------------------------
SUB_LANGS = 'en.*'
SUB_FORMAT = 'json3/vtt/best'  # prefer timestamped json3
# WhisperX fallback for videos with no English track (D23, signed off
# 2026-07-06): large-v3-turbo for this coarse pass; remote GPU venv only.
WHISPERX_COARSE_MODEL = 'large-v3-turbo'
TRANSCRIPT_FAIL_FRACTION_BLOCK = 0.5  # block when >50% of a batch fails

# ---------------------------------------------------------------------------
# Relevance triage
# ---------------------------------------------------------------------------
# Overlapping windows keep chunks that straddle a boundary.
CHUNK_WINDOW_S = 10 * 60
CHUNK_OVERLAP_S = 60

# Three-legged keep rule: keep when any leg passes. Starting
# values from the mid-July 2026 amateur-footage scoping.
CHUNKS_ABS_SAFE = 15  # enough absolute material regardless of length
SHORT_VIDEO_MIN_S = 20 * 60  # the short/long boundary
CHUNKS_MIN_SHORT = 3  # shorts judged on count
DENSITY_MIN_PER_MIN = 0.15  # longs judged on chunks per minute

# The exact flash ID and low-cost fast tier were selected on 2026-07-05.
# Gemini flash uses GEMINI_API_KEY; gemini-2.5-flash was the known-stable ID.
LLM_PROVIDER = 'gemini'
TRIAGE_MODEL = 'gemini-2.5-flash'
# The documented floor across the candidate seats, kept at the min for fair comparison:
# gemma-4-31b-it :free (OpenRouter) 32,768 < qwen3-32b on Groq 40,960 (hard error above it)
# < nemotron-3-ultra :free / gemini-2.5-flash 65,536. Google-served gemma documents no output
# cap (probe at the gemma re-test) and runs ~14,400 req/day free. Thinking tokens count
# against this budget on the gemini/nemotron seats. Raise to 40,960 if gemma exits.
TRIAGE_MAX_TOKENS = 32768
# The current hand-run s29 chain is the scrape-lane consumer. A promoted wrapper
# must use these values at its subprocess boundary.
SCRAPE_TRACKNET_STRIDE = 8
# streaming builds its median background image from a capped sample of frames (1800) instead of all of them
SCRAPE_TRACKNET_LARGE_VIDEO = True
API_KEY_ENV = 'GEMINI_API_KEY'  # referenced by name only; never a value

# ---------------------------------------------------------------------------
# Commentary cleaning and fine timestamps
# ---------------------------------------------------------------------------
# The clean and paraphrase share one call budget. The clean lane
# earns the stronger tier while the triage filter stays on flash.
CLEAN_MODEL = 'gemma-4-31b-it'
ALT_PHRASINGS_K = 3  # supported range 2 to 4
# Provisional sanity baseline; Curtis tunes this later, not a measured optimum.
CLEAN_BERTSCORE_MIN = 0.80
WHISPERX_FINE_MODEL = 'large-v2'  # D23: fine-timestamp pass, remote GPU only

# A rally pairs with the first commentary chunk
# whose start falls within this many seconds after the rally's end.
PAIR_WINDOW_S = 8

# ---------------------------------------------------------------------------
# Rate limiting and IP-ban mitigation
# ---------------------------------------------------------------------------
# The stack: current pip-installed yt-dlp, Deno >= 2.3.0 user-space, the bgutil
# PO-token provider plugin, cookieless by default. Values are starting points.
SLEEP_INTERVAL_S = 5  # randomised pre-download pause
MAX_SLEEP_INTERVAL_S = 15
SLEEP_REQUESTS_S = 10  # between extraction requests
LIMIT_RATE = '2M'  # byte-transfer cap
CONCURRENT_FRAGMENTS = 1
DOWNLOAD_WORKERS = 2
DOWNLOAD_FAIL_FRACTION_BLOCK = 0.5
SLEEP_SUBTITLES_S = 2  # between subtitle pulls
YTDLP_RETRIES = 3  # existing downloader convention

# Subprocess timeouts. Metadata and caption calls are light; minutes are plenty.
YTDLP_METADATA_TIMEOUT_S = 120
SUBTITLE_TIMEOUT_S = 300

# Mid-batch circuit-breaker floors. The failure fractions say when to block,
# not when to evaluate. Checking once past these floors stops a banned or dead
# endpoint from hammering through the rest of the batch.
TRANSCRIPT_BLOCK_MIN_ATTEMPTS = 10
TRIAGE_BLOCK_MIN_FAILURES = 5

# LLM retry/backoff. Exponential backoff base, doubled per attempt.
LLM_MAX_RETRIES = 3
LLM_BACKOFF_BASE_S = 2.0
# Bound each synchronous SDK request so optional commentary cannot hold the
# visual lane indefinitely when a provider accepts a request but never replies.
LLM_REQUEST_TIMEOUT_S = 120


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def check_ytdlp() -> None:
    """Fail loud if yt-dlp is missing before any scraper component does work."""
    if not shutil.which(YTDLP_BIN):
        raise RuntimeError(
            f'{YTDLP_BIN} not found in PATH. Install with: pip install yt-dlp'
        )


def ensure_dirs() -> None:
    """Create the scrape root and its sidecar dirs if absent."""
    for directory in (SCRAPE_DIR, VIDEOS_DIR, TRANSCRIPTS_DIR, CHUNKS_DIR, MASKS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def ytdlp_throttle_args(include_subtitles: bool = False) -> list[str]:
    """yt-dlp throttle flags shared by the search indexing and transcript acquisition calls.

    Single source for the throttle set so no component hardcodes a magic number.
    --sleep-interval / --max-sleep-interval are deliberately NOT here: they
    pause before a *video* download. Indexing and transcript acquisition both
    pass --skip-download. Those constants belong to the video-download path;
    the two metadata components pace their process spawns from Python instead.
    --limit-rate is a no-op on pure metadata prints (no bytes move) but is kept
    for a single throttle source and does real work on the caption transfer.

    :param include_subtitles: add the between-subtitle-pull sleep (transcript acquisition).
    :return: flag list to splice into a yt-dlp argv.
    """
    flags = [
        '--sleep-requests', str(SLEEP_REQUESTS_S),
        '--limit-rate', LIMIT_RATE,
        '--retries', str(YTDLP_RETRIES),
    ]
    if include_subtitles:
        flags += ['--sleep-subtitles', str(SLEEP_SUBTITLES_S)]
    return flags


def read_candidates(input_path: Path | None = None) -> list[dict]:
    """Read candidates.csv into a list of row dicts (transcript acquisition and triage consume it).

    :param input_path: Optional candidates file override for isolated components.
    :return: one dict per row, keys per CANDIDATES_COLUMNS.
    """
    candidates_path = CANDIDATES_CSV if input_path is None else input_path
    if not candidates_path.exists():
        raise FileNotFoundError(f'{candidates_path} not found. Run search indexing first.')
    with candidates_path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def write_candidates(rows: list[dict]) -> None:
    """Write rows to candidates.csv using the fixed CANDIDATES_COLUMNS header.

    Used by search indexing (initial write) and relevance triage (rewrite with keep filled). Any
    column missing from a row writes blank, which keeps the header stable.

    :param rows: one dict per row; extra keys are ignored, missing keys write blank.
    """
    SCRAPE_DIR.mkdir(parents=True, exist_ok=True)
    with CANDIDATES_CSV.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATES_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, '') for col in CANDIDATES_COLUMNS})
