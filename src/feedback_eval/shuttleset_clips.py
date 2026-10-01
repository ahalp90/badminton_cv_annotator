"""Cut the rally clips a split names out of the ShuttleSet source matches.

A model version needs video to watch, and the reference set was derived from
annotations alone: the CSVs are in the repo, the match videos are not. This
module closes that gap for exactly the clips one side of a split needs, so a
predictions file can be produced for the same clips the scorer will select.

The rally bounds come from ShuttleSet itself, not from the rally segmenter.
Each `set/<video>/set<N>.csv` row is one stroke with its `frame_num`; a clip
`<video>_s<N>_r<R>` (the id `shuttleset_faults` writes) spans its first stroke
to its last, padded by `--pre` and `--post` seconds. The padding is asymmetric
by default because the terminal shot is the fault: the clip has to run long
enough after it to show the shuttle land out or in the net.

Frame numbers are converted to seconds with the fps ShuttleSet annotated at
(`video_metadata.csv`), and the cut is made in seconds. A source downloaded at a
different frame rate today still cuts at the right moment; one that is a
different *edit* of the match would not, and nothing here can detect that, so
spot-check a few clips before running the model on all of them.

Two ways to get the video:

- `--fetch` (the default to reach for): yt-dlp downloads only each clip's span
  at up to 720p, straight from the match URL. 100 clips are ~18 minutes of
  video rather than 11 whole matches, which matters on a nearly full disk, and
  720p is well above what the model samples at.
- Whole matches: `--write-match-csv` writes `match.csv` rows for only the
  matches the split needs. Pass that file to
  `bst_x/pipeline/download_adapter.py --match-csv`, which fetches them into
  `data/shuttleset/raw_video/<id>.mp4`; then run again without the flag to cut
  locally with ffmpeg.

Either way, clips already on disk are skipped, so an interrupted run resumes,
and `clips.csv` in the output directory records every clip's source and bounds.

Run as `python -m feedback_eval.shuttleset_clips` (see --help).
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .splits import load_split

logger = logging.getLogger(__name__)

CLIP_ID = re.compile(r"^(?P<video>.+)_s(?P<set>\d+)_r(?P<rally>\d+)$")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm")

Runner = Callable[[list[str]], None]


class ShuttleSetClipError(ValueError):
    """Raised when a clip cannot be planned or cut as asked."""


@dataclass(frozen=True)
class ClipPlan:
    """Where one rally clip comes from and the span to cut, in seconds."""

    clip_id: str
    video: str
    video_id: str
    url: str
    start_s: float
    end_s: float


def parse_clip_id(clip_id: str) -> tuple[str, str, str]:
    """`<video>_s<set>_r<rally>` -> (video, set, rally)."""
    match = CLIP_ID.match(clip_id)
    if match is None:
        raise ShuttleSetClipError(f"not a ShuttleSet clip id: {clip_id!r}")
    return match["video"], match["set"], match["rally"]


def load_matches(path: Path) -> dict[str, dict[str, str]]:
    """Read `match.csv` into video -> its row, keeping every column for re-writing."""
    if not path.is_file():
        raise ShuttleSetClipError(f"no such match file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = {"id", "video", "url"} - set(reader.fieldnames or ())
        if missing:
            raise ShuttleSetClipError(f"{path}: missing column(s) {sorted(missing)}")
        return {row["video"].strip(): row for row in reader}


def load_fps(path: Path) -> tuple[dict[str, float], dict[str, str]]:
    """Read `video_metadata.csv` into (video -> annotated fps, video -> exclusion note).

    Excluded matches carry a blank fps and a note such as "excluded: all frame
    numbers incorrect". They are kept apart rather than dropped so a clip that
    points at one fails with the reason, not just "no fps".
    """
    if not path.is_file():
        raise ShuttleSetClipError(f"no such metadata file: {path}")
    fps: dict[str, float] = {}
    excluded: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            video = row["video"].strip()
            text = (row.get("fps") or "").strip()
            if not text:
                excluded[video] = (row.get("note") or "").strip() or "no fps recorded"
                continue
            value = float(text)
            if value <= 0:
                raise ShuttleSetClipError(f"{path}: non-positive fps for {video!r}")
            fps[video] = value
    return fps, excluded


def rally_frames(set_file: Path) -> dict[str, tuple[float, float]]:
    """Read one set file into rally -> (first stroke frame, last stroke frame)."""
    if not set_file.is_file():
        raise ShuttleSetClipError(f"no such set file: {set_file}")
    bounds: dict[str, tuple[float, float]] = {}
    with set_file.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rally = (row.get("rally") or "").strip()
            frame_text = (row.get("frame_num") or "").strip()
            if not rally or not frame_text:
                continue
            frame = float(frame_text)
            low, high = bounds.get(rally, (frame, frame))
            bounds[rally] = (min(low, frame), max(high, frame))
    return bounds


def plan_clips(
    clip_ids: Sequence[str],
    *,
    set_root: Path,
    matches: dict[str, dict[str, str]],
    fps: dict[str, float],
    pre_s: float,
    post_s: float,
    excluded: dict[str, str] | None = None,
) -> tuple[ClipPlan, ...]:
    """Resolve every clip id to a source match and a span in seconds.

    Fails on the first clip it cannot resolve rather than skipping it: a clip
    missing from the predictions file is refused by the scorer later anyway,
    and here the reason is still known.
    """
    if pre_s < 0 or post_s < 0:
        raise ShuttleSetClipError("--pre and --post must not be negative")
    frames_by_set: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}
    plans: list[ClipPlan] = []
    for clip_id in clip_ids:
        video, set_number, rally = parse_clip_id(clip_id)
        if video not in matches:
            raise ShuttleSetClipError(f"{clip_id}: video {video!r} is not in match.csv")
        if excluded and video in excluded:
            raise ShuttleSetClipError(f"{clip_id}: video {video!r} is {excluded[video]}")
        if video not in fps:
            raise ShuttleSetClipError(f"{clip_id}: video {video!r} has no fps in the metadata")
        key = (video, set_number)
        if key not in frames_by_set:
            frames_by_set[key] = rally_frames(set_root / video / f"set{set_number}.csv")
        bounds = frames_by_set[key].get(rally)
        if bounds is None:
            raise ShuttleSetClipError(f"{clip_id}: rally {rally} not in set{set_number}.csv")
        first, last = bounds
        plans.append(
            ClipPlan(
                clip_id=clip_id,
                video=video,
                video_id=matches[video]["id"].strip(),
                url=matches[video]["url"].strip(),
                start_s=max(0.0, first / fps[video] - pre_s),
                end_s=last / fps[video] + post_s,
            )
        )
    return tuple(plans)


def find_source(videos_dir: Path, video_id: str) -> Path | None:
    """Find `<id>.mp4`, or the downloader's legacy `<id> <name>.mp4`."""
    if not videos_dir.is_dir():
        return None
    for path in sorted(videos_dir.iterdir()):
        if (
            path.is_file()
            and path.suffix.lower() in VIDEO_EXTENSIONS
            and (path.stem == video_id or path.stem.startswith(f"{video_id} "))
        ):
            return path
    return None


def cut_command(source: Path, plan: ClipPlan, out: Path) -> list[str]:
    """ffmpeg arguments for one clip.

    Re-encoded rather than stream-copied: a stream copy can only start on a
    keyframe, which on broadcast footage can be seconds before the rally.
    Audio is dropped; the model is given video only.
    """
    return [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-ss", f"{plan.start_s:.3f}",
        "-i", str(source),
        "-t", f"{plan.end_s - plan.start_s:.3f}",
        "-map", "0:v:0", "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        str(out),
    ]


def fetch_command(plan: ClipPlan, out: Path) -> list[str]:
    """yt-dlp arguments to download one clip's span, video only, at most 720p.

    `--force-keyframes-at-cuts` re-encodes at the boundaries so the clip starts
    on the requested moment rather than the nearest earlier keyframe.
    """
    return [
        "yt-dlp", "--quiet", "--no-warnings", "--no-playlist",
        "-f", "bv*[height<=720][ext=mp4]/bv*[height<=720]",
        "--download-sections", f"*{plan.start_s:.3f}-{plan.end_s:.3f}",
        "--force-keyframes-at-cuts",
        "--remux-video", "mp4",
        "-o", str(out),
        plan.url,
    ]


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def cut_clips(
    plans: Sequence[ClipPlan],
    *,
    videos_dir: Path,
    out_dir: Path,
    runner: Runner = _run,
) -> tuple[list[ClipPlan], list[str]]:
    """Cut every planned clip that is not already on disk.

    Returns (clips now on disk, video ids with no source file). Each clip is
    written to a temporary name and renamed, so an interrupted run never leaves
    a truncated clip that a resumed run would then skip.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    done: list[ClipPlan] = []
    missing_sources: dict[str, None] = {}
    for plan in plans:
        out = out_dir / f"{plan.clip_id}.mp4"
        if out.is_file():
            done.append(plan)
            continue
        source = find_source(videos_dir, plan.video_id)
        if source is None:
            missing_sources.setdefault(plan.video_id, None)
            continue
        partial = out.with_name(f"{out.stem}.partial.mp4")
        runner(cut_command(source, plan, partial))
        partial.replace(out)
        done.append(plan)
    return done, list(missing_sources)


def fetch_clips(
    plans: Sequence[ClipPlan],
    *,
    out_dir: Path,
    runner: Runner = _run,
) -> tuple[list[ClipPlan], list[str]]:
    """Download every planned clip that is not already on disk.

    Returns (clips now on disk, clip ids that failed). A failed clip is reported
    and skipped rather than aborting the run: one unavailable span should not
    cost the other 99, and the scorer refuses a predictions file with a clip
    missing, so the gap cannot go unnoticed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    done: list[ClipPlan] = []
    failed: list[str] = []
    for plan in plans:
        out = out_dir / f"{plan.clip_id}.mp4"
        if out.is_file():
            done.append(plan)
            continue
        partial = out.with_name(f"{out.stem}.partial.mp4")
        try:
            runner(fetch_command(plan, partial))
        except subprocess.CalledProcessError as error:
            logger.warning("%s: yt-dlp failed (exit %s)", plan.clip_id, error.returncode)
            failed.append(plan.clip_id)
            continue
        if not partial.is_file():
            logger.warning("%s: yt-dlp wrote no file", plan.clip_id)
            failed.append(plan.clip_id)
            continue
        partial.replace(out)
        done.append(plan)
        logger.info("fetched %d/%d %s", len(done), len(plans), plan.clip_id)
    return done, failed


def write_manifest(path: Path, plans: Sequence[ClipPlan], out_dir: Path) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["clip_id", "path", "video", "video_id", "start_s", "end_s"])
        for plan in plans:
            writer.writerow(
                [
                    plan.clip_id,
                    str(out_dir / f"{plan.clip_id}.mp4"),
                    plan.video,
                    plan.video_id,
                    f"{plan.start_s:.3f}",
                    f"{plan.end_s:.3f}",
                ]
            )
    return path


def write_match_subset(path: Path, matches: dict[str, dict[str, str]], videos: Sequence[str]) -> Path:
    """Write the `match.csv` rows for `videos`, in the original column order."""
    rows = [matches[video] for video in videos]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--split", type=Path, required=True, help="split (or subset) JSON")
    parser.add_argument("--side", choices=("train", "test"), default="test")
    parser.add_argument("--set-root", type=Path, default=Path("data/shuttleset/set"))
    parser.add_argument(
        "--match", type=Path, default=None, help="match.csv (default: <set-root>/match.csv)"
    )
    parser.add_argument(
        "--metadata", type=Path, default=Path("data/shuttleset/video_metadata.csv")
    )
    parser.add_argument("--videos-dir", type=Path, default=Path("data/shuttleset/raw_video"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/feedback_eval/clips"))
    parser.add_argument("--pre", type=float, default=1.0, help="seconds before the first stroke")
    parser.add_argument("--post", type=float, default=2.0, help="seconds after the last stroke")
    parser.add_argument(
        "--write-match-csv",
        type=Path,
        help="write match.csv rows for the needed matches here (for the downloader) and stop",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="download each clip's span with yt-dlp instead of cutting local matches",
    )
    parser.add_argument("--dry-run", action="store_true", help="plan and report; cut nothing")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    clip_ids = load_split(args.split).clips(args.side)
    matches = load_matches(args.match or args.set_root / "match.csv")
    fps, excluded = load_fps(args.metadata)
    plans = plan_clips(
        clip_ids,
        set_root=args.set_root,
        matches=matches,
        fps=fps,
        pre_s=args.pre,
        post_s=args.post,
        excluded=excluded,
    )
    videos = sorted({plan.video for plan in plans})
    seconds = sum(plan.end_s - plan.start_s for plan in plans)
    print(f"{len(plans)} clip(s) from {len(videos)} match(es), {seconds / 60:.1f} min of video")

    if args.write_match_csv:
        write_match_subset(args.write_match_csv, matches, videos)
        print(
            f"wrote {args.write_match_csv}; fetch with:\n"
            f"  PYTHONPATH=src:src/bst_x python src/bst_x/pipeline/download_adapter.py "
            f"--match-csv {args.write_match_csv} --output-dir {args.videos_dir}"
        )
        return 0

    if args.dry_run:
        absent = sorted({p.video_id for p in plans if find_source(args.videos_dir, p.video_id) is None})
        print(f"dry run: {len(absent)} source match(es) not yet downloaded {absent}")
        return 0

    if args.fetch:
        for tool in ("yt-dlp", "ffmpeg"):
            if shutil.which(tool) is None:
                raise ShuttleSetClipError(f"{tool} not found in PATH")
        done, failed = fetch_clips(plans, out_dir=args.out_dir)
        manifest = write_manifest(args.out_dir / "clips.csv", done, args.out_dir)
        print(f"{len(done)} of {len(plans)} clip(s) on disk; manifest {manifest}")
        if failed:
            print(f"{len(failed)} clip(s) failed; re-run to retry: {failed[:5]}")
            return 1
        return 0

    if shutil.which("ffmpeg") is None:
        raise ShuttleSetClipError("ffmpeg not found in PATH")
    done, missing = cut_clips(plans, videos_dir=args.videos_dir, out_dir=args.out_dir)
    manifest = write_manifest(args.out_dir / "clips.csv", done, args.out_dir)
    print(f"{len(done)} of {len(plans)} clip(s) on disk; manifest {manifest}")
    if missing:
        print(
            f"{len(missing)} source match(es) not downloaded, ids {missing}; "
            "run with --write-match-csv to fetch them"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
