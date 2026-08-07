"""Gate: is seek-based range decoding frame-identical to a sequential decode?

Two subcommands::

    baseline <video> <ledger.txt>
        Decode the whole video sequentially; write one MD5 per frame plus a
        final "TOTAL <n>" line. This is the ground truth for frame identity
        (and the true decodable frame count, vs the container's metadata).

    check <video> <ledger.txt> [--ranges a:b,c:d,...] [--mode seek|scan]
        Decode each half-open range and compare every frame's MD5 against the
        ledger. Without --ranges, a default probe set is used: frame 0, an
        unaligned prime start mid-video, a range crossing the video's midpoint,
        the last frames, and a deliberate EOF-crossing range (expected SHORT).

Exit code 0 only if every range matched exactly (an EOF-crossing probe passes
by being detectably short, never by fabricating frames).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from shared.video_sharding.range_decode import (
    iter_frame_range,
    md5_frame,
    metadata_frame_count,
    open_capture,
)


def write_baseline(video: Path, ledger: Path) -> int:
    cap = open_capture(video)
    count = 0
    with ledger.open("w") as out:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out.write(md5_frame(frame) + "\n")
            count += 1
            if count % 10000 == 0:
                print(f"  hashed {count} frames", flush=True)
        out.write(f"TOTAL {count}\n")
    cap.release()
    meta = metadata_frame_count(video)
    print(f"baseline: {count} decoded frames (container metadata says {meta}) -> {ledger}")
    return 0


def read_baseline(ledger: Path) -> list[str]:
    lines = [line.strip() for line in ledger.open() if line.strip()]
    if not lines or not lines[-1].startswith("TOTAL "):
        raise ValueError(f"{ledger} is not a complete baseline ledger")
    return lines[:-1]


def default_ranges(n_frames: int) -> list[tuple[int, int]]:
    """Awkward probe boundaries: unaligned starts, midpoint, tail, EOF-crossing."""
    prime_start = 12347 % max(n_frames - 100, 1)
    mid = n_frames // 2
    return [
        (0, min(50, n_frames)),
        (prime_start, prime_start + 53),
        (mid - 7, mid + 11),
        (max(0, n_frames - 41), n_frames),
        (max(0, n_frames - 5), n_frames + 20),  # crosses EOF: must read short, loudly
    ]


def check_ranges(video: Path, ledger: Path, ranges: list[tuple[int, int]], mode: str) -> int:
    baseline = read_baseline(ledger)
    n_frames = len(baseline)
    all_ok = True
    for start, end in ranges:
        digests = [md5_frame(f) for f in iter_frame_range(video, start, end, mode)]
        expected_n = max(0, min(end, n_frames) - start)
        matches = sum(
            1 for i, d in enumerate(digests)
            if start + i < n_frames and d == baseline[start + i]
        )
        first_bad = next(
            (start + i for i, d in enumerate(digests)
             if start + i >= n_frames or d != baseline[start + i]),
            None,
        )
        ok = len(digests) == expected_n and matches == len(digests)
        all_ok &= ok
        crosses_eof = end > n_frames
        print(
            f"RANGE {mode} [{start},{end}) read={len(digests)} expected={expected_n} "
            f"match={matches} first_bad={'-' if first_bad is None else first_bad} "
            f"ok={int(ok)}{' (EOF-crossing probe: short read is the pass condition)' if crosses_eof else ''}"
        )
    print(f"GATE decode-identity {mode}: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


def parse_ranges(spec: str) -> list[tuple[int, int]]:
    ranges = []
    for token in spec.split(","):
        start_s, end_s = token.split(":")
        ranges.append((int(start_s), int(end_s)))
    return ranges


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_base = sub.add_parser("baseline")
    p_base.add_argument("video", type=Path)
    p_base.add_argument("ledger", type=Path)
    p_check = sub.add_parser("check")
    p_check.add_argument("video", type=Path)
    p_check.add_argument("ledger", type=Path)
    p_check.add_argument("--ranges", type=parse_ranges, default=None,
                         help="comma-separated start:end pairs; default probe set otherwise")
    p_check.add_argument("--mode", choices=("seek", "scan"), default="seek")
    args = parser.parse_args()

    if args.cmd == "baseline":
        return write_baseline(args.video, args.ledger)
    ranges = args.ranges or default_ranges(len(read_baseline(args.ledger)))
    return check_ranges(args.video, args.ledger, ranges, args.mode)


if __name__ == "__main__":
    sys.exit(main())
