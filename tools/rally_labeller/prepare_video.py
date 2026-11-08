"""Make browser copies of source videos for the rally labeller.

Run from the repository root::

    PYTHONPATH=src python3 tools/rally_labeller/prepare_video.py path/to/source.mp4 [more.mp4 ...]

Frame N in the labeller is frame N of the source, which is the numbering the
pipeline uses. To keep that true, every source frame passes through in order,
and the copy's frame count must equal the source's packet count. The copy is
at most 720 pixels high, has a keyframe every 15 frames so single-frame steps
stay quick, and has exact constant-rate timestamps so the browser can convert
a playback time into a frame number.

Each copy goes to ``<out-root>/<video_id>/<first frame>-<end frame>/video.mp4``,
where the end frame is exclusive, with its details in ``video.json.gz``:
frame rate, frame numbers, the source file's SHA-256 and frame size, and its
YouTube title, channel and upload date. YouTube details come
from the latest ``yt-dlp`` (run through ``uvx``) when the file name, or
``--youtube-id``, is a YouTube ID.

Copies that already have both files are skipped, so a rerun picks up where a
failed batch stopped. ``--out-root`` defaults to ``$RALLY_LABELLER_MEDIA``, or
``local_scratch/rally_labeller`` when that is unset.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from shared.video_metadata import probe_video_fps, probe_video_metadata

VIDEO_INFO_SCHEMA = "rally-labeller-video/1"
# Same folder the labeller serves copies from; relative paths count from the repository root.
DEFAULT_OUT_ROOT = Path(os.environ.get("RALLY_LABELLER_MEDIA", "local_scratch/rally_labeller"))
MAX_HEIGHT = 720
# Short keyframe spacing makes stepping back one frame cheap: the browser
# decodes at most this many frames from the previous keyframe.
KEYFRAME_INTERVAL = 15
CRF = 23
AUDIO_BITRATE = "96k"
YOUTUBE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")


@dataclass(frozen=True)
class SourceInfo:
    """Source timing and contents measured before encoding."""

    fps: Fraction
    packet_count: int
    width: int
    height: int
    has_audio: bool
    # Timestamp of the first video frame. Usually 0, but a cut made by copying
    # packets can start later, and frames are picked by timestamp.
    stream_start_seconds: float


def probe_source(source: Path) -> SourceInfo:
    """Read the source frame rate, video packet count and whether it has audio.

    ``probe_video_fps`` rejects a source whose declared rates differ, which is
    how ffprobe reports a variable frame rate.
    """
    fps = probe_video_fps(source)
    video = _ffprobe_json(
        ["-select_streams", "v:0", "-count_packets", "-show_entries", "stream=nb_read_packets,width,height,start_time"],
        source,
    )["streams"][0]
    audio = _ffprobe_json(["-select_streams", "a", "-show_entries", "stream=index"], source)
    return SourceInfo(
        fps=fps,
        packet_count=int(video["nb_read_packets"]),
        width=int(video["width"]),
        height=int(video["height"]),
        has_audio=bool(audio["streams"]),
        stream_start_seconds=float(video.get("start_time", 0)),
    )


def youtube_details(youtube_id: str) -> dict[str, object]:
    """Look up a YouTube video's title, channel, upload date and length with yt-dlp.

    ``uvx`` runs the latest yt-dlp, which matters because YouTube changes
    often break older versions.
    """
    uvx = shutil.which("uvx")
    if uvx is None:
        raise RuntimeError("uvx is not on PATH; install uv or pass --no-youtube")
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    command = [uvx, "yt-dlp@latest", "--dump-json", "--skip-download", "--no-warnings", url]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"yt-dlp could not read {url}: {completed.stderr.strip()}")
    info = json.loads(completed.stdout)
    return {
        "id": info["id"],
        "url": info["webpage_url"],
        "title": info["title"],
        "channel": info.get("channel") or info["uploader"],
        "upload_date": info["upload_date"],
        "duration_seconds": info["duration"],
    }


def file_sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


# Seek this far before a clip's first frame, then select frames by timestamp.
# Input seeking lands on a keyframe, so it needs room before the wanted frame.
SEEK_LEAD_SECONDS = 10


def parse_time(text: str) -> float:
    """Read seconds from ``"95"``, ``"1:35"`` or ``"1:01:35"``."""
    seconds = 0.0
    for part in text.split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def encode_command(source: Path, output: Path, info: SourceInfo, first_frame: int, end_frame: int) -> list[str]:
    """Build the ffmpeg command for one browser copy of source frames ``[first_frame, end_frame)``.

    ``-copyts`` keeps source timestamps, so ``select`` can pick frames by time:
    frame N starts at the stream's start time plus N / fps, and each window
    edge sits half a frame before a frame's start, well clear of millisecond
    rounding in MKV timestamps. ``settb`` and ``setpts`` then place
    clip frame N at exactly N / fps, counted in a time base of 1 / fps
    numerator, so every timestamp is a whole number of ticks.
    ``-fps_mode passthrough`` stops ffmpeg dropping or repeating frames.
    B-frames are off so the browser never has to reorder frames to show one.
    """
    numerator, denominator = info.fps.numerator, info.fps.denominator
    first_second = info.stream_start_seconds + float(first_frame / info.fps)
    end_second = info.stream_start_seconds + float(end_frame / info.fps)
    window_start = info.stream_start_seconds + float((first_frame - 0.5) / info.fps)
    window_end = info.stream_start_seconds + float((end_frame - 0.5) / info.fps)
    video_filters = ",".join((
        f"select='gte(t,{window_start})*lt(t,{window_end})'",
        f"settb=1/{numerator}",
        f"setpts=N*{denominator}",
        f"scale=-2:'min({MAX_HEIGHT},ih)'",
    ))
    seek_to = max(first_second - SEEK_LEAD_SECONDS, 0)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-stats", "-nostdin",
        "-copyts", "-ss", f"{seek_to:.3f}", "-i", os.fspath(source),
        "-map", "0:v:0",
    ]
    if info.has_audio:
        audio_filter = f"atrim=start={first_second}:end={end_second},asetpts=PTS-STARTPTS"
        command += ["-map", "0:a:0", "-af", audio_filter, "-c:a", "aac", "-b:a", AUDIO_BITRATE]
    command += [
        "-sn", "-dn", "-map_metadata", "-1",
        "-vf", video_filters,
        "-fps_mode", "passthrough",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(CRF),
        "-g", str(KEYFRAME_INTERVAL), "-bf", "0", "-pix_fmt", "yuv420p",
        "-video_track_timescale", str(numerator),
        "-movflags", "+faststart",
        "-f", "mp4", os.fspath(output),
    ]
    return command


def prepare_video(
    source: Path,
    video_id: str,
    youtube_id: str | None,
    out_root: Path,
    *,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    overwrite: bool = False,
) -> Path | None:
    """Encode and check one browser copy of a whole video or part of one.

    :param source: Source video file.
    :param video_id: Folder name and label-file name for this video.
    :param youtube_id: YouTube ID to look up, or None for a video from elsewhere.
    :param out_root: Folder that receives one subfolder per video.
    :param start_seconds: Start of the part to copy, or None for the beginning.
    :param end_seconds: End of the part to copy, or None for the end of the video.
    :param overwrite: Replace an existing copy instead of skipping it.
    :return: The new clip folder, or None when that clip already existed.
    :raises ValueError: The requested part is empty or outside the video, or
        the copy's frames or timing differ from the source.
    """
    source_info = probe_source(source)
    first_frame = 0 if start_seconds is None else round(start_seconds * source_info.fps)
    end_frame = source_info.packet_count if end_seconds is None else round(end_seconds * source_info.fps)
    end_frame = min(end_frame, source_info.packet_count)
    if not 0 <= first_frame < end_frame:
        raise ValueError(f"{video_id}: no frames between {start_seconds} s and {end_seconds} s")

    # Folder names use frame numbers so clips of one video sort in order and never collide.
    clip_dir = out_root / video_id / f"{first_frame:07d}-{end_frame:07d}"
    video_path = clip_dir / "video.mp4"
    info_path = clip_dir / "video.json.gz"
    if video_path.is_file() and info_path.is_file() and not overwrite:
        print(f"{video_id}: frames {first_frame}-{end_frame} already prepared, skipping")
        return None

    clip_dir.mkdir(parents=True, exist_ok=True)
    # Look up YouTube first: a network failure should stop before a long encode.
    youtube = youtube_details(youtube_id) if youtube_id is not None else None
    partial_path = clip_dir / "video.part.mp4"
    clip_frames = end_frame - first_frame
    print(f"{video_id}: encoding frames {first_frame}-{end_frame} ({clip_frames} frames at {source_info.fps} fps)")
    try:
        subprocess.run(encode_command(source, partial_path, source_info, first_frame, end_frame), check=True)
        copy = probe_video_metadata(partial_path)
        if copy.fps != source_info.fps or copy.frame_count != clip_frames:
            raise ValueError(
                f"{video_id}: copy has {copy.frame_count} frames at {copy.fps} fps, "
                f"expected {clip_frames} at {source_info.fps}"
            )
        os.replace(partial_path, video_path)
    finally:
        partial_path.unlink(missing_ok=True)

    video_info = {
        "schema": VIDEO_INFO_SCHEMA,
        "video_id": video_id,
        "fps": f"{copy.fps.numerator}/{copy.fps.denominator}",
        "first_frame": first_frame,
        "frame_count": clip_frames,
        "source_frame_count": source_info.packet_count,
        "has_audio": source_info.has_audio,
        "source": {
            "file_name": source.name,
            "sha256": file_sha256(source),
            "width": source_info.width,
            "height": source_info.height,
            "youtube": youtube,
        },
    }
    with gzip.open(info_path, "wt", encoding="utf-8") as file:
        json.dump(video_info, file, indent=2)
    print(f"{video_id}: done, {clip_frames} frames checked")
    return clip_dir


def _ffprobe_json(arguments: list[str], source: Path) -> dict:
    command = ["ffprobe", "-v", "error", *arguments, "-of", "json", os.fspath(source)]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    return json.loads(completed.stdout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="+", type=Path, help="source video files")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help=f"default: {DEFAULT_OUT_ROOT}")
    parser.add_argument("--video-id", help="video ID when preparing one source (default: file name without extension)")
    parser.add_argument("--youtube-id", help="YouTube ID when preparing one source (default: the video ID if it is one)")
    parser.add_argument("--no-youtube", action="store_true", help="skip the YouTube lookup")
    parser.add_argument("--start", type=parse_time, help="start of the part to prepare, as seconds or m:ss")
    parser.add_argument("--end", type=parse_time, help="end of the part to prepare, as seconds or m:ss")
    parser.add_argument("--overwrite", action="store_true", help="replace copies that already exist")
    args = parser.parse_args(argv)
    if (args.video_id is not None or args.youtube_id is not None) and len(args.sources) != 1:
        parser.error("--video-id and --youtube-id need exactly one source")

    failures: list[str] = []
    for source in args.sources:
        video_id = args.video_id or source.stem
        youtube_id = args.youtube_id or (video_id if YOUTUBE_ID_PATTERN.match(video_id) else None)
        if args.no_youtube:
            youtube_id = None
        try:
            prepare_video(
                source,
                video_id,
                youtube_id,
                args.out_root,
                start_seconds=args.start,
                end_seconds=args.end,
                overwrite=args.overwrite,
            )
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
            # A batch can run for an hour, so report each failure and carry on.
            print(f"{video_id}: FAILED: {error}", file=sys.stderr)
            failures.append(video_id)
    if failures:
        print(f"{len(failures)} of {len(args.sources)} videos failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
