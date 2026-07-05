"""Build a stratified Phase-A stem sample for the GPU parity gates (G8/G9).

G9's authoritative decision needs coverage of **every distinct video-id** (court
homography is per-video), but the full extract list is ~30k clips, far more than
a decision run needs. This takes the first ``--per-video`` clips of each video-id
present in the source list, giving all-court coverage at ~10k frames (a few clips
per video) instead of a full re-extract.

Deterministic (sorted), so the same source + ``--per-video`` always yields the
same sample. Feed the output to G8 via ``RTMLIB_GATE_STEMFILE``.

Env / args:
  --source     stem list to sample from (default: provenance stems_to_extract.txt)
  --per-video  clips to keep per video-id (default: 5)
  --out        output stem file (default: ./phase_a_stems.txt)

Run (on Bourbaki):
  PYTHONPATH=src/bst_x:src <env>/bin/python \\
      src/bst_x/validation_scripts/rtmlib_migration/make_phase_a_sample.py \\
      --per-video 5 --out phase_a_stems.txt
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

DEFAULT_SOURCE = os.environ.get(
    "RTMLIB_GATE_EXTRACT_LIST",
    "/srv/mergerfs/main_pool/320_cosc594_data-bourbaki/"
    "ShuttleSet_keypoints_raw_provenance/stems_to_extract.txt",
)


def build_sample(stems: list[str], per_video: int) -> list[str]:
    """First ``per_video`` stems of each video-id, in sorted (deterministic) order.

    :param stems: candidate stems (``<vid>_<game>_<rally>_<stroke>``).
    :param per_video: cap of clips kept per video-id.
    :return: the stratified sample, sorted by (video-id, stem).
    """
    by_vid: dict[int, list[str]] = defaultdict(list)
    for stem in sorted(stems):
        vid = int(stem.split("_", 1)[0])
        if len(by_vid[vid]) < per_video:
            by_vid[vid].append(stem)
    picked: list[str] = []
    for vid in sorted(by_vid):
        picked.extend(by_vid[vid])
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, default=Path(DEFAULT_SOURCE),
                        help="Stem list to sample from (one stem per line).")
    parser.add_argument("--per-video", type=int, default=5,
                        help="Clips to keep per video-id (default 5).")
    parser.add_argument("--out", type=Path, default=Path("phase_a_stems.txt"),
                        help="Output stem file (default ./phase_a_stems.txt).")
    args = parser.parse_args()

    if not args.source.exists():
        parser.error(f"source stem list not found: {args.source}")
    stems = [ln.strip() for ln in args.source.read_text().splitlines() if ln.strip()]
    if not stems:
        parser.error(f"source stem list is empty: {args.source}")

    sample = build_sample(stems, args.per_video)
    args.out.write_text("\n".join(sample) + "\n")

    vids = sorted({int(s.split("_", 1)[0]) for s in sample})
    src_vids = {int(s.split("_", 1)[0]) for s in stems}
    print(f"source: {args.source}  ({len(stems)} stems, {len(src_vids)} video-ids)")
    print(f"sample: {len(sample)} stems across {len(vids)} video-ids "
          f"(<= {args.per_video} per video)  -> {args.out}")
    print(f"video-ids: {vids}")
    print(f"est. frames @ ~60/clip: ~{len(sample) * 60:,}")
    missing = sorted(src_vids - set(vids))
    if missing:
        print(f"WARNING: {len(missing)} source video-ids absent from the sample: {missing}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
