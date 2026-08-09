"""Index seed search families into candidates.csv without downloading media.

Runs the flat yt-dlp metadata index over every seed search term, dedups the hits
by video_id, enriches empty fields with one per-video metadata pull, applies the
cheap screens (doubles / duration / upload-date, flag never drop), and writes
candidates.csv with the fixed column set. keep and triage_verdict stay blank at
index time (relevance triage fills keep, the section 10 human packet fills triage_verdict).

Search terms are grouped by substream ('match' / 'instructional', D24): every row
carries its family's substream, and the instructional family skips the short
duration flag since coach-review clips run short by design.

Run as: python -m scraper.search_index (PYTHONPATH=src).

Failure behaviour is log-and-skip per term. A term returning nothing logs and
moves on. The batch blocks only if every term returns empty, which signals a
broken binary or a network block rather than a thin result.
"""
import argparse
import json
import re
import subprocess
import time

from .config import (
    CANDIDATES_CSV,
    DOUBLES_KEYWORD_PHRASES,
    DOUBLES_KEYWORD_TOKENS,
    DURATION_MAX_S,
    DURATION_MIN_S,
    FLAT_PRINT_FIELDS,
    FLAT_PRINT_TEMPLATE,
    SEARCH_TERMS,
    SLEEP_REQUESTS_S,
    SUBSTREAM_INSTRUCTIONAL,
    UPLOAD_DATE_FLOOR,
    YTDLP_BIN,
    YTDLP_METADATA_TIMEOUT_S,
    YTSEARCH_COUNT,
    check_ytdlp,
    ensure_dirs,
    write_candidates,
    ytdlp_throttle_args,
)


def search_term_rows(
    term: str,
    substream: str,
    *,
    search_count: int = YTSEARCH_COUNT,
) -> list[dict]:
    """Run the flat metadata index for one term; return one row dict per hit.

    On any failure (timeout, non-zero exit) this logs and returns an empty list,
    under the search index's log-and-skip-per-term rule.

    :param term: the seed search term to index.
    :param substream: the family the term belongs to; stamped onto every row.
    :return: one row dict per search hit, each carrying search_term and substream.
    """
    count = _validated_search_count(search_count)
    cmd = [
        YTDLP_BIN,
        f'ytsearch{count}:{term}',
        '--flat-playlist',
        '--print', FLAT_PRINT_TEMPLATE,
        *ytdlp_throttle_args(),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=YTDLP_METADATA_TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT term '{term}': search exceeded {YTDLP_METADATA_TIMEOUT_S}s")
        return []
    if result.returncode != 0:
        print(f"  ERROR term '{term}': {result.stderr.strip()[:200]}")
        return []

    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = line.split('\t')
        if len(fields) != len(FLAT_PRINT_FIELDS):
            print(f"  WARNING term '{term}': unexpected --print line, skipping: {line[:120]}")
            continue
        row = dict(zip(FLAT_PRINT_FIELDS, fields))
        # yt-dlp prints 'NA' for fields missing in flat mode; normalise to blank
        # so the enrichment pass knows what to fill.
        row = {key: ('' if value == 'NA' else value) for key, value in row.items()}
        row['search_term'] = term
        row['substream'] = substream
        rows.append(row)
    return rows


def enrich_row(row: dict) -> bool:
    """Fill empty channel/duration_s/upload_date via one --dump-json pull.

    Flat mode reliably returns id/url/title, but channel, duration and
    upload_date can be empty, so rows with gaps get one full metadata
    extraction each. Mutates the row in place; log-and-skip on failure.

    :param row: the candidate row to fill (mutated in place).
    :return: True when a metadata subprocess was attempted, including failed
        attempts, so the caller paces every request attempt.
    """
    needs_enrich = not row['channel'] or not row['duration_s'] or not row['upload_date']
    if not needs_enrich:
        return False

    cmd = [YTDLP_BIN, row['url'], '--dump-json', '--skip-download', *ytdlp_throttle_args()]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=YTDLP_METADATA_TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT enrich {row['video_id']}")
        return True
    if result.returncode != 0:
        print(f"  ERROR enrich {row['video_id']}: {result.stderr.strip()[:200]}")
        return True
    try:
        meta = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"  WARNING enrich {row['video_id']}: could not parse --dump-json output")
        return True

    row['channel'] = row['channel'] or (meta.get('channel') or '')
    duration = meta.get('duration')
    row['duration_s'] = row['duration_s'] or ('' if duration is None else str(duration))
    row['upload_date'] = row['upload_date'] or (meta.get('upload_date') or '')
    return True


def flag_doubles(title: str) -> bool:
    """True if the title looks like a doubles match.

    Flag not drop: cheap and sometimes wrong, so it flags for the section 7 late
    guard and the human packet rather than dropping. Long phrases match as
    case-insensitive substrings; the short abbreviations match only as whole
    tokens so 'md'/'wd'/'xd' do not fire inside unrelated words (e.g. 'commander',
    'crowd').

    :param title: the video title to screen.
    :return: True when a doubles keyword hits.
    """
    text = title.lower()
    if any(phrase in text for phrase in DOUBLES_KEYWORD_PHRASES):
        return True
    tokens = re.split(r'[^a-z0-9]+', text)
    return any(token in DOUBLES_KEYWORD_TOKENS for token in tokens)


def duration_out_of_band(duration_s: str, substream: str) -> bool:
    """True when a known duration sits outside the sane match-length band.

    Instructional-substream rows skip the short-duration leg (D24): coach-review
    clips run short by design, so only the over-long leg applies to them. An
    unknown (blank or unparseable) duration is never flagged.

    :param duration_s: duration in seconds as a string; blank when unknown.
    :param substream: the row's family; the instructional family skips the short leg.
    :return: True when the duration is out of band and should be flagged.
    """
    if not duration_s:
        return False
    try:
        seconds = int(float(duration_s))
    except ValueError:
        return False
    above_ceiling = seconds > DURATION_MAX_S
    if substream == SUBSTREAM_INSTRUCTIONAL:
        return above_ceiling
    below_floor = seconds < DURATION_MIN_S
    return below_floor or above_ceiling


def upload_before_floor(upload_date: str) -> bool:
    """True if the upload predates the optional floor, which is off by default.

    Both dates are YYYYMMDD strings, which sort lexically the same as by date.
    Returns False whenever the floor is unset (the default), so this screen is a
    no-op unless Ariel sets UPLOAD_DATE_FLOOR.

    :param upload_date: the row's YYYYMMDD upload date; blank when unknown.
    :return: True when the upload predates a set floor.
    """
    if UPLOAD_DATE_FLOOR is None or not upload_date:
        return False
    return upload_date < UPLOAD_DATE_FLOOR


def build_candidates(
    search_terms: dict[str, list[str]] = SEARCH_TERMS,
    *,
    search_count: int = YTSEARCH_COUNT,
) -> list[dict]:
    """Index every search family into candidates.csv and return the rows.

    Dedup: a video_id found by two or more terms collapses to one row. The first
    term's metadata AND substream win; later terms comma-join into search_term so
    provenance survives. INVARIANT: a video surfaced by both families therefore
    keeps whichever family indexed it first, while its cross-family provenance
    stays auditable through the joined search_term.

    :param search_terms: substream -> list of seed terms; defaults to config.
    :return: the written candidate rows.
    """
    count = _validated_search_count(search_count)
    check_ytdlp()
    ensure_dirs()

    by_id: dict[str, dict] = {}
    terms_with_hits = 0
    total_terms = sum(len(terms) for terms in search_terms.values())
    for substream, terms in search_terms.items():
        for term in terms:
            print(f'Searching [{substream}]: {term}')
            rows = search_term_rows(term, substream, search_count=count)
            # Each count feeds the per-term keep-rate
            # report (relevance-triage keeps over rows surfaced).
            print(f'  {term!r}: {len(rows)} rows')
            if rows:
                terms_with_hits += 1
            for row in rows:
                video_id = row['video_id']
                if video_id in by_id:
                    existing_terms = by_id[video_id]['search_term'].split(',')
                    if term not in existing_terms:
                        # First row already owns the metadata and substream; only
                        # the provenance term joins on. Dupe terms are dropped so
                        # the join stays clean.
                        by_id[video_id]['search_term'] += f',{term}'
                else:
                    by_id[video_id] = row
            time.sleep(SLEEP_REQUESTS_S)  # pace our own process spawns

    if terms_with_hits == 0:
        # Every term returned nothing, so fail loudly.
        raise RuntimeError(
            'Search indexing: every search term returned zero rows. Check yt-dlp and network.'
        )

    rows = list(by_id.values())
    print(f'{len(rows)} unique videos across {terms_with_hits}/{total_terms} terms')

    flagged_duration = 0
    flagged_upload = 0
    for row in rows:
        if enrich_row(row):  # mutates in place; True when an enrichment attempt ran
            # SLEEP_REQUESTS_S, not the 5-15 s interval pause: enrichment is a
            # metadata extraction request, matching --sleep-requests semantics.
            # The randomised pre-download pause belongs to transcript acquisition's caption pull.
            time.sleep(SLEEP_REQUESTS_S)
        row['doubles_suspect'] = flag_doubles(row['title'])
        row['duration_suspect'] = duration_out_of_band(row['duration_s'], row['substream'])
        row['upload_date_suspect'] = upload_before_floor(row['upload_date'])
        flagged_duration += row['duration_suspect']
        flagged_upload += row['upload_date_suspect']
        # INVARIANT: keep and triage_verdict are blank at index time. Relevance triage
        # fills keep; the section 10 human packet fills triage_verdict.
        row['keep'] = ''
        row['triage_verdict'] = ''

    print(
        f'Duration screen flagged {flagged_duration}/{len(rows)} rows '
        f'(outside {DURATION_MIN_S // 60}-{DURATION_MAX_S // 60} min)'
    )
    if UPLOAD_DATE_FLOOR is not None:
        print(f'Upload-date screen flagged {flagged_upload}/{len(rows)} rows')

    write_candidates(rows)
    print(f'Wrote {CANDIDATES_CSV} ({len(rows)} rows)')
    return rows


def _validated_search_count(value: int) -> int:
    """Return one positive per-term search cap."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'search_count must be a positive integer, got {value!r}')
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Search indexing: index the seed search families into candidates.csv.',
    )
    parser.add_argument('--search-count', type=int, default=YTSEARCH_COUNT)
    arguments = parser.parse_args()
    build_candidates(search_count=arguments.search_count)


if __name__ == '__main__':
    main()
