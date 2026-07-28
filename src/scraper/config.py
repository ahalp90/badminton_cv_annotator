"""Shared config for the badminton-commentary scraper.

Single source of truth for the file contracts and named constants the stages
share: output paths, the candidates.csv column set, yt-dlp throttle flags, the
metadata screens, chunking and keep thresholds, LLM settings, and the stage 8/9
trajectory-rule constants. Every stage imports from here so the column order,
sidecar layout and rate-limit values live in one place.

Constant provenance is cited inline as "spec sN" against the section of
local_scratch/autograder_architecture/scraper_spec.md it came from. OPEN: marks
judgement calls awaiting Ariel; each implements the spec's default.

Descended from the agy-passed poc (local_scratch/autograder_architecture/poc/),
reconciled to the spec's decided values: D22 throttle stack, D23 WhisperX
models, D24 instructional sub-stream, D9 keep rule, D8 metadata screens.
"""
import csv
import os
import shutil
from pathlib import Path
# ---------------------------------------------------------------------------
# Output layout (dataset_schema.md section 2 tree)
# ---------------------------------------------------------------------------
# One scrape root holds the flat CSVs plus the per-video sidecar dirs. Default
# sits under the repo's gitignored data/ tree; BADMINTON_SCRAPE_DIR overrides.
_REPO_ROOT = Path(__file__).resolve().parents[2]
SCRAPE_DIR = Path(os.environ.get('BADMINTON_SCRAPE_DIR', _REPO_ROOT / 'data' / 'scrape_output'))

CANDIDATES_CSV = SCRAPE_DIR / 'candidates.csv'  # spec s2 (stages 1, 3)
VIDEOS_DIR = SCRAPE_DIR / 'videos'
SOURCES_MANIFEST_NAME = 'sources.toml'
TRANSCRIPTS_DIR = SCRAPE_DIR / 'transcripts'  # spec s3 (stage 2)
CHUNKS_DIR = SCRAPE_DIR / 'chunks'  # spec s4 (stages 3, 10)
MASKS_DIR = SCRAPE_DIR / 'masks'  # schema s2 (stage 9)
RALLY_SPANS_CSV = SCRAPE_DIR / 'rally_spans.csv'  # spec s6 (stage 8)
CONTACT_FRAMES_CSV = SCRAPE_DIR / 'contact_frames.csv'  # spec s6 (stage 8)
PAIRS_CSV = SCRAPE_DIR / 'rally_commentary_pairs.csv'  # spec s9 (stage 11)

# ---------------------------------------------------------------------------
# candidates.csv contract (spec s2)
# ---------------------------------------------------------------------------
# Column order is fixed here. INVARIANT: stage 1 writes this header, stage 3
# rewrites the same file with the same header (only keep changes), and the
# section 10 human packet later fills triage_verdict.
# Bool columns serialise as the CSV strings 'True'/'False' (keep is also blank
# before stage 3 fills it). Consumers must parse (== 'True'), never truth-test
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
    'doubles_suspect',  # bool, title/metadata keyword screen (spec s8)
    'duration_suspect',  # bool, duration outside the match-length band (spec s2, D8)
    'upload_date_suspect',  # bool, always False while the floor is off (spec s2, D8)
    'keep',  # bool, appended by stage 3; blank at index time
    'triage_verdict',  # keep/drop/uncertain, human packet; blank at index time
]

SUBSTREAM_MATCH = 'match'
SUBSTREAM_INSTRUCTIONAL = 'instructional'

# ---------------------------------------------------------------------------
# Stage 1: search indexing (spec s2)
# ---------------------------------------------------------------------------
YTDLP_BIN = 'yt-dlp'  # same binary the pipeline already uses (download_videos.py)

YTSEARCH_COUNT = 50  # spec s2 uses ytsearch50

# Tab-separated --print template. --print implies --simulate, so the flat index
# downloads no bytes (spec s2). Field order must match FLAT_PRINT_FIELDS.
FLAT_PRINT_TEMPLATE = (
    '%(id)s\t%(webpage_url)s\t%(title)s\t%(channel)s\t%(duration)s\t%(upload_date)s'
)
FLAT_PRINT_FIELDS = ['video_id', 'url', 'title', 'channel', 'duration_s', 'upload_date']

# Seed search-term families (spec s2; terms OPEN: for Ariel to tune, families
# decided). Keyed by the substream their rows carry (D24).
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

# Cheap metadata screens (spec s2, D8). Flag never drop: a dropped row loses
# its provenance. Instructional-substream rows skip the short-duration flag
# (D24; coach-review clips run short by design).
DURATION_MIN_S = 10 * 60  # flag under 10 min
DURATION_MAX_S = 240 * 60  # flag over 240 min
# Upload-date floor off per D8. A YYYYMMDD string when ever set; None disables.
UPLOAD_DATE_FLOOR = None

# Doubles keyword screen (spec s8). Long phrases match as case-insensitive
# substrings; the short abbreviations match only as whole tokens so 'md'/'wd'/
# 'xd' do not fire inside unrelated words (e.g. 'commander', 'crowd').
DOUBLES_KEYWORD_PHRASES = ['doubles', 'mixed doubles']
DOUBLES_KEYWORD_TOKENS = ['xd', 'md', 'wd']
# spec s8 also lists "known pair-name patterns": needs a curated list, add when
# Ariel supplies one.

# ---------------------------------------------------------------------------
# Stage 2: transcript acquisition (spec s3)
# ---------------------------------------------------------------------------
SUB_LANGS = 'en.*'  # spec s3 --sub-langs
SUB_FORMAT = 'json3/vtt/best'  # spec s3: prefer timestamped json3
# WhisperX fallback for videos with no English track (D23, signed off
# 2026-07-06): large-v3-turbo for this coarse pass; remote GPU venv only.
WHISPERX_COARSE_MODEL = 'large-v3-turbo'
STAGE2_FAIL_FRACTION_BLOCK = 0.5  # spec s3: block when >50% of a batch fails

# ---------------------------------------------------------------------------
# Stage 3: relevance triage (spec s4)
# ---------------------------------------------------------------------------
# Overlapping windows so a chunk straddling a boundary is not lost (spec s4).
CHUNK_WINDOW_S = 10 * 60
CHUNK_OVERLAP_S = 60

# Three-legged keep rule (D9, spec s4): keep when ANY leg passes. Starting
# values, tuned at B5.
CHUNKS_ABS_SAFE = 15  # enough absolute material regardless of length
SHORT_VIDEO_MIN_S = 20 * 60  # the short/long boundary
CHUNKS_MIN_SHORT = 3  # shorts judged on count
DENSITY_MIN_PER_MIN = 0.15  # longs judged on chunks per minute

# OPEN (spec s4, s12): exact flash ID pinned at B5; tier decided (low-cost
# fast, Gemini flash via GEMINI_API_KEY, 2026-07-05). gemini-2.5-flash is the
# known-stable ID at write time.
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
# Stage 10: clean pass and fine timestamps (spec s9)
# ---------------------------------------------------------------------------
# The clean and paraphrase share one call budget (schema s5); the clean lane
# earns the stronger tier while the triage filter stays on flash (spec s4).
CLEAN_MODEL = 'gemma-4-31b-it'
ALT_PHRASINGS_K = 3  # schema s5: 2 to 4, default 3
# Provisional sanity baseline; Curtis tunes this later, not a measured optimum.
CLEAN_BERTSCORE_MIN = 0.80
WHISPERX_FINE_MODEL = 'large-v2'  # D23: fine-timestamp pass, remote GPU only

# Stage 11 pairing (spec s9): a rally pairs with the first commentary chunk
# whose start falls within this many seconds after the rally's end.
PAIR_WINDOW_S = 8

# ---------------------------------------------------------------------------
# Rate limiting / IP-ban mitigation (D22, spec s5)
# ---------------------------------------------------------------------------
# The stack: current pip-installed yt-dlp, Deno >= 2.3.0 user-space, the bgutil
# PO-token provider plugin, cookieless by default. Values are starting points.
SLEEP_INTERVAL_S = 5  # spec s5 --sleep-interval (randomised pre-download pause)
MAX_SLEEP_INTERVAL_S = 15  # spec s5 --max-sleep-interval
SLEEP_REQUESTS_S = 10  # spec s5 --sleep-requests (between extraction requests)
LIMIT_RATE = '2M'  # spec s5 --limit-rate (byte-transfer cap)
CONCURRENT_FRAGMENTS = 1  # spec s5 --concurrent-fragments (stage 4 downloads)
DOWNLOAD_WORKERS = 2  # spec s5: worker count down from 4
SLEEP_SUBTITLES_S = 2  # spec s3 --sleep-subtitles (between subtitle pulls)
YTDLP_RETRIES = 3  # existing downloader convention

# Subprocess timeouts. Metadata and caption calls are light; minutes are plenty.
YTDLP_METADATA_TIMEOUT_S = 120
SUBTITLE_TIMEOUT_S = 300

# Mid-batch circuit-breaker floors. The spec's block rules (s3: over 50% fail;
# s4: every call fails) say when to block, not when to evaluate; checking
# mid-loop once past these floors stops a banned or dead-endpoint run from
# hammering through the rest of the batch.
STAGE2_BLOCK_MIN_ATTEMPTS = 10
STAGE3_BLOCK_MIN_FAILURES = 5

# LLM retry/backoff. Exponential backoff base, doubled per attempt.
LLM_MAX_RETRIES = 3
LLM_BACKOFF_BASE_S = 2.0


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def check_ytdlp() -> None:
    """Fail loud if yt-dlp is missing before any stage does work."""
    if not shutil.which(YTDLP_BIN):
        raise RuntimeError(
            f'{YTDLP_BIN} not found in PATH. Install with: pip install yt-dlp'
        )


def ensure_dirs() -> None:
    """Create the scrape root and its sidecar dirs if absent."""
    for directory in (SCRAPE_DIR, VIDEOS_DIR, TRANSCRIPTS_DIR, CHUNKS_DIR, MASKS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def ytdlp_throttle_args(include_subtitles: bool = False) -> list[str]:
    """yt-dlp throttle flags shared by the stage 1 and stage 2 calls.

    Single source for the throttle set so no stage hardcodes a magic number.
    --sleep-interval / --max-sleep-interval are deliberately NOT here: they
    pause before a *video* download, which stages 1 and 2 never do (both pass
    --skip-download). Those two constants belong to stage 4's download path;
    stages 1 and 2 pace their own process spawns from Python instead.
    --limit-rate is a no-op on pure metadata prints (no bytes move) but is kept
    for a single throttle source and does real work on the caption transfer.

    :param include_subtitles: add the between-subtitle-pull sleep (stage 2).
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
    """Read candidates.csv into a list of row dicts (stages 2, 3 consume it).

    :param input_path: Optional candidates file override for isolated stages.
    :return: one dict per row, keys per CANDIDATES_COLUMNS.
    """
    candidates_path = CANDIDATES_CSV if input_path is None else input_path
    if not candidates_path.exists():
        raise FileNotFoundError(f'{candidates_path} not found. Run stage 1 first.')
    with candidates_path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def write_candidates(rows: list[dict]) -> None:
    """Write rows to candidates.csv using the fixed CANDIDATES_COLUMNS header.

    Used by stage 1 (initial write) and stage 3 (rewrite with keep filled). Any
    column missing from a row writes blank, which keeps the header stable.

    :param rows: one dict per row; extra keys are ignored, missing keys write blank.
    """
    SCRAPE_DIR.mkdir(parents=True, exist_ok=True)
    with CANDIDATES_CSV.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATES_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, '') for col in CANDIDATES_COLUMNS})
