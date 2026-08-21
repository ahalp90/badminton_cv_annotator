"""Build paired cut-aware broad-context cases without exposing human truth."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .build_trials import (
    VIDEO_IDS,
    VideoData,
    _letterbox,
    _load_video,
    _open_writer,
    _sha256,
    _write_json,
    balanced_select,
    shift_window,
)
from .multiscale_sampling import segment_for_frame, storyboard_source_frames
from .multiscale_schema import (
    MANIFEST_SCHEMA,
    MultiscaleCase,
    Segment,
    load_manifest,
    validate_context_pairs,
)

TRUTH_SCHEMA = "vlm-multiscale-truth-v1"
PROVENANCE_SCHEMA = "vlm-multiscale-provenance-v1"
STORYBOARD_FPS = 8.0
WIDTH = 512
HEIGHT = 288
CYAN = (220, 200, 20)
GOLD = (30, 190, 240)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DEFAULT_PILOT_CASES = 12


def _load_json_gz(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"expected an object in {path}")
    return payload


def load_raw_cut_segments(path: Path) -> tuple[Segment, ...]:
    """Load persisted PySceneDetect intervals with stable source-global IDs."""
    raw_cuts = _load_json_gz(path)["raw_cuts"]
    segments = tuple(
        Segment(
            segment_id=f"S{index:05d}",
            source_start_frame=int(interval[0]),
            source_end_frame=int(interval[1]),
        )
        for index, interval in enumerate(raw_cuts)
    )
    if not segments:
        raise ValueError(f"no raw cut intervals in {path}")
    for previous, current in pairwise(segments):
        if previous.source_end_frame != current.source_start_frame:
            raise ValueError(
                f"raw cut intervals are not contiguous at {current.segment_id}"
            )
    return segments


def clip_segments(
    segments: tuple[Segment, ...],
    start_frame: int,
    end_frame: int,
) -> tuple[Segment, ...]:
    """Clip source-global cut segments to one context window."""
    clipped = []
    for segment in segments:
        start = max(start_frame, segment.source_start_frame)
        end = min(end_frame, segment.source_end_frame)
        if start < end:
            clipped.append(Segment(segment.segment_id, start, end))
    if not clipped:
        raise ValueError("context window overlaps no cut segments")
    if (
        clipped[0].source_start_frame != start_frame
        or clipped[-1].source_end_frame != end_frame
    ):
        raise ValueError("cut segments do not cover the context window")
    for previous, current in pairwise(clipped):
        if previous.source_end_frame != current.source_start_frame:
            raise ValueError("clipped cut segments contain a gap")
    return tuple(clipped)


def _scene_fractions(
    labels: pd.DataFrame, start_frame: int, end_frame: int
) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for row in labels.itertuples(index=False):
        overlap = max(
            0,
            min(end_frame, int(row.end_frame)) - max(start_frame, int(row.start_frame)),
        )
        if overlap:
            counts[str(row.truth)] += overlap
    length = end_frame - start_frame
    if sum(counts.values()) != length:
        raise ValueError(
            f"scene truth covers {sum(counts.values())}/{length} target frames"
        )
    return {name: count / length for name, count in sorted(counts.items())}


def scene_stratum(scene_fractions: dict[str, float]) -> str | None:
    """Place a target in one of the three fixed pilot groups."""
    live_fraction = scene_fractions.get("live", 0.0) + scene_fractions.get(
        "live-non-standard", 0.0
    )
    replay_cutaway_fraction = scene_fractions.get("replay", 0.0) + scene_fractions.get(
        "cutaway", 0.0
    )
    if live_fraction >= 0.90:
        return "clear_live"
    if replay_cutaway_fraction >= 0.75:
        return "replay_or_cutaway"
    material_classes = sum(fraction >= 0.05 for fraction in scene_fractions.values())
    if material_classes >= 2:
        return "mixed"
    return None


def _truth_intervals(
    labels: pd.DataFrame, start_frame: int, end_frame: int
) -> list[dict[str, Any]]:
    intervals = []
    for row in labels.itertuples(index=False):
        start = max(start_frame, int(row.start_frame))
        end = min(end_frame, int(row.end_frame))
        if start < end:
            intervals.append(
                {
                    "source_start_frame": start,
                    "source_end_frame": end,
                    "truth": str(row.truth),
                }
            )
    return intervals


def _alignment_report(
    video: VideoData,
    raw_segments: tuple[Segment, ...],
) -> dict[str, Any]:
    labels = video.scene_labels.sort_values("start_frame")
    starts = [int(frame) for frame in labels["start_frame"]]
    ends = [int(frame) for frame in labels["end_frame"]]
    frame_count = len(video.track)
    if (
        raw_segments[0].source_start_frame != 0
        or raw_segments[-1].source_end_frame != frame_count
    ):
        raise ValueError(
            f"{video.name}: raw cut intervals do not cover the complete source"
        )
    if starts[0] != 0 or ends[-1] != frame_count:
        raise ValueError(
            f"{video.name}: scene truth does not cover the complete source"
        )
    if any(left_end != right_start for left_end, right_start in zip(ends, starts[1:])):
        raise ValueError(f"{video.name}: scene truth intervals are not contiguous")
    cut_boundaries = np.asarray(
        [segment.source_start_frame for segment in raw_segments[1:]],
        dtype=int,
    )
    transition_frames = np.asarray(starts[1:], dtype=int)
    nearest_errors = []
    if len(cut_boundaries):
        for frame in transition_frames:
            nearest_errors.append(int(np.min(np.abs(cut_boundaries - frame))))
    return {
        "video_id": video.name,
        "fps": video.fps,
        "decoded_frames": frame_count,
        "truth_start_frame": starts[0],
        "truth_end_frame": ends[-1],
        "truth_transitions": len(transition_frames),
        "detected_cut_boundaries": len(cut_boundaries),
        "transition_nearest_cut_error_frames": nearest_errors,
        "transition_within_2_frames": sum(error <= 2 for error in nearest_errors),
        "transition_within_10_base30_frames": sum(
            error * 30 / video.fps <= 10 for error in nearest_errors
        ),
        "source_identity_limit": (
            "FPS, decoded extent, full truth coverage, and transition-to-cut distances are "
            "checked. The reviewed MP4 is a different encode and is not available here."
        ),
    }


def _write_storyboard(
    video: VideoData,
    case: MultiscaleCase,
) -> None:
    capture = cv2.VideoCapture(str(video.source_path))
    writer = _open_writer(case.clip_path, STORYBOARD_FPS)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"could not open source video {video.source_path}")
        for source_frame in case.source_frames:
            capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"failed to read source frame {source_frame}")
            composed, _, _, _ = _letterbox(frame, WIDTH, HEIGHT)
            segment = segment_for_frame(case.segments, source_frame)
            in_target = case.target_start_frame <= source_frame < case.target_end_frame
            label = f"{segment.segment_id}  {source_frame / video.fps:07.2f}s"
            if in_target:
                label += "  TARGET"
                cv2.rectangle(composed, (2, 2), (WIDTH - 3, HEIGHT - 3), GOLD, 3)
            cv2.rectangle(composed, (0, 0), (WIDTH, 28), BLACK, -1)
            cv2.putText(
                composed,
                label,
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                CYAN if in_target else WHITE,
                1,
                cv2.LINE_AA,
            )
            writer.write(composed)
    finally:
        writer.release()
        capture.release()
    _verify_storyboard(case)


def _verify_storyboard(case: MultiscaleCase) -> None:
    capture = cv2.VideoCapture(str(case.clip_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"could not reopen clip {case.clip_path}")
        observed = (
            int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        expected = (case.expected_frames, case.width, case.height)
        if observed != expected:
            raise ValueError(
                f"{case.clip_path}: clip geometry/count is {observed}, expected {expected}"
            )
        if abs(capture.get(cv2.CAP_PROP_FPS) - case.sample_fps) > 0.01:
            raise ValueError(
                f"{case.clip_path}: clip FPS differs from {case.sample_fps}"
            )
    finally:
        capture.release()


def _automatic_priors(
    video: VideoData, span_index: int, start_frame: int, end_frame: int
) -> dict[str, Any]:
    raw_contacts = sum(
        int(row["rally_id"]) == span_index for row in video.result["contacts"]
    )
    filtered_contacts = sum(
        int(row["rally_id"]) == span_index for row in video.result["filtered_contacts"]
    )
    span = video.result["spans"][span_index]
    return {
        "span_id": span_index,
        "span_duration_seconds": (int(span[1]) - int(span[0])) / video.fps,
        "raw_contact_count": raw_contacts,
        "filtered_contact_count": filtered_contacts,
        "court_present_fraction": float(
            np.mean(video.court_present[start_frame:end_frame])
        ),
        "raw_mask_fraction": float(np.mean(video.raw_mask[start_frame:end_frame])),
        "definitive_mask_fraction": float(
            np.mean(video.definitive_mask[start_frame:end_frame])
        ),
        "track_visible_fraction": float(
            np.mean(video.track[start_frame:end_frame, 2] > 0)
        ),
    }


def _candidate_records(
    videos: list[VideoData],
    cuts_by_video: dict[str, tuple[Segment, ...]],
    context_seconds: tuple[int, ...],
    max_frames: int,
    *,
    include_unstratified: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible = []
    excluded = []
    for video in videos:
        total_frames = len(video.track)
        raw_segments = cuts_by_video[video.name]
        for span_index, raw_span in enumerate(video.result["spans"]):
            target_start, target_end = (int(raw_span[0]), int(raw_span[1]))
            scene_fractions = _scene_fractions(
                video.scene_labels, target_start, target_end
            )
            stratum = scene_stratum(scene_fractions)
            if stratum is None and not include_unstratified:
                excluded.append(
                    {
                        "pair_id": f"context-{video.name}-r{span_index:03d}",
                        "video_id": video.name,
                        "span_id": span_index,
                        "sort_frame": target_start,
                        "stratum": None,
                        "reason": "scene stratum is undefined",
                    }
                )
                continue
            if stratum is None:
                stratum = "other_or_unstratified"
            contexts: dict[int, dict[str, Any]] = {}
            exclusion = None
            for seconds in context_seconds:
                context_length = round(video.fps * seconds)
                start, end = shift_window(
                    (target_start + target_end) // 2, context_length, total_frames
                )
                if not start <= target_start < target_end <= end:
                    exclusion = f"target does not fit {seconds}-second context"
                    break
                segments = clip_segments(raw_segments, start, end)
                source_frames = storyboard_source_frames(
                    segments,
                    start,
                    end,
                    target_start,
                    target_end,
                    max_frames,
                )
                if source_frames is None:
                    exclusion = f"required cut and target frames exceed {max_frames} at {seconds} seconds"
                    break
                contexts[seconds] = {
                    "source_start_frame": start,
                    "source_end_frame": end,
                    "source_frames": source_frames,
                    "segments": segments,
                }
            pair_id = f"context-{video.name}-r{span_index:03d}"
            if exclusion is not None:
                excluded.append(
                    {
                        "pair_id": pair_id,
                        "video_id": video.name,
                        "span_id": span_index,
                        "sort_frame": target_start,
                        "stratum": stratum,
                        "reason": exclusion,
                    }
                )
                continue
            eligible.append(
                {
                    "pair_id": pair_id,
                    "video_id": video.name,
                    "sort_frame": target_start,
                    "span_id": span_index,
                    "target_start_frame": target_start,
                    "target_end_frame": target_end,
                    "stratum": stratum,
                    "scene_fractions": scene_fractions,
                    "truth_intervals": _truth_intervals(
                        video.scene_labels, target_start, target_end
                    ),
                    "contexts": contexts,
                }
            )
    return eligible, excluded


def _select_pilot(
    records: list[dict[str, Any]], pilot_cases: int
) -> list[dict[str, Any]]:
    if pilot_cases % 3:
        raise ValueError("pilot case count must divide evenly across three strata")
    per_stratum = pilot_cases // 3
    selected = []
    for stratum in ("clear_live", "replay_or_cutaway", "mixed"):
        candidates = [record for record in records if record["stratum"] == stratum]
        selected.extend(balanced_select(candidates, per_stratum))
    return sorted(
        selected,
        key=lambda record: (str(record["video_id"]), int(record["sort_frame"])),
    )


def _select_all_eligible(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select every eligible record in stable source order."""
    return sorted(
        records,
        key=lambda record: (
            str(record["video_id"]),
            int(record["sort_frame"]),
            int(record["span_id"]),
        ),
    )


def _manifest_case(case: MultiscaleCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "pair_id": case.pair_id,
        "video_id": case.video_id,
        "context_seconds": case.context_seconds,
        "clip_path": str(Path("clips") / case.clip_path.name),
        "source_start_frame": case.source_start_frame,
        "source_end_frame": case.source_end_frame,
        "target_start_frame": case.target_start_frame,
        "target_end_frame": case.target_end_frame,
        "sample_fps": case.sample_fps,
        "source_frames": list(case.source_frames),
        "segments": [asdict(segment) for segment in case.segments],
        "pipeline_priors": case.pipeline_priors,
    }


def build_multiscale_trials(
    artifacts_root: Path,
    repo_root: Path,
    scene_labels_dir: Path,
    output_dir: Path,
    *,
    video_names: tuple[str, ...],
    pilot_cases: int,
    context_seconds: tuple[int, ...],
    max_frames: int,
    all_eligible: bool = False,
) -> None:
    """Build immutable paired clips, a truth-blind manifest, and scoring sidecars."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    if tuple(sorted(context_seconds)) != (90, 120):
        raise ValueError("the first multiscale comparison requires 90 and 120 seconds")
    unknown_videos = set(video_names) - set(VIDEO_IDS)
    if unknown_videos:
        raise ValueError(f"unknown development videos: {sorted(unknown_videos)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    inference_dir = output_dir / "inference"
    scoring_dir = output_dir / "scoring"
    clips_dir = inference_dir / "clips"
    clips_dir.mkdir(parents=True)
    scoring_dir.mkdir()

    shots_master_path = (
        repo_root / "training/data/shuttleset/annotations/shots_master.csv"
    )
    shots_master = pd.read_csv(shots_master_path)
    videos = [
        _load_video(artifacts_root, scene_labels_dir, shots_master, video_name)
        for video_name in video_names
    ]
    by_name = {video.name: video for video in videos}
    court_paths = {
        video.name: artifacts_root
        / "stages"
        / "court"
        / video.name
        / "court_evidence.json.gz"
        for video in videos
    }
    cuts_by_video = {
        video.name: load_raw_cut_segments(court_paths[video.name]) for video in videos
    }
    alignments = [
        _alignment_report(video, cuts_by_video[video.name]) for video in videos
    ]
    eligible, excluded = _candidate_records(
        videos,
        cuts_by_video,
        context_seconds,
        max_frames,
        include_unstratified=all_eligible,
    )
    if all_eligible:
        selected = _select_all_eligible(eligible)
        selection_mode = "all_eligible"
    else:
        selected = _select_pilot(eligible, pilot_cases)
        selection_mode = "pilot"

    manifest_cases = []
    truth_cases = []
    for record in selected:
        video = by_name[record["video_id"]]
        span_index = int(record["span_id"])
        priors = _automatic_priors(
            video,
            span_index,
            int(record["target_start_frame"]),
            int(record["target_end_frame"]),
        )
        for seconds in context_seconds:
            context = record["contexts"][seconds]
            case_id = f"{record['pair_id']}--{seconds}"
            case = MultiscaleCase(
                case_id=case_id,
                pair_id=str(record["pair_id"]),
                video_id=video.name,
                context_seconds=seconds,
                clip_path=clips_dir / f"{case_id}.mp4",
                source_start_frame=int(context["source_start_frame"]),
                source_end_frame=int(context["source_end_frame"]),
                target_start_frame=int(record["target_start_frame"]),
                target_end_frame=int(record["target_end_frame"]),
                sample_fps=STORYBOARD_FPS,
                expected_frames=max_frames,
                width=WIDTH,
                height=HEIGHT,
                source_frames=tuple(context["source_frames"]),
                segments=tuple(context["segments"]),
                pipeline_priors=priors,
            )
            _write_storyboard(video, case)
            manifest_cases.append(_manifest_case(case))
        truth_cases.append(
            {
                "pair_id": record["pair_id"],
                "video_id": video.name,
                "span_id": span_index,
                "target_start_frame": record["target_start_frame"],
                "target_end_frame": record["target_end_frame"],
                "stratum": record["stratum"],
                "scene_fractions": record["scene_fractions"],
                "truth_intervals": record["truth_intervals"],
            }
        )

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "expected_frames": max_frames,
        "width": WIDTH,
        "height": HEIGHT,
        "cases": manifest_cases,
    }
    truth = {
        "schema": TRUTH_SCHEMA,
        "cases": truth_cases,
        "excluded": excluded,
    }
    _write_json(inference_dir / "manifest.json", manifest)
    _write_json(scoring_dir / "truth.json", truth)
    loaded_cases = load_manifest(inference_dir / "manifest.json")
    validate_context_pairs(loaded_cases)

    input_paths = [shots_master_path]
    for video in videos:
        input_paths.extend(
            (
                video.source_path,
                video.result_path,
                video.track_path,
                video.bboxes_path,
                video.kps_path,
                video.court_present_path,
                video.raw_mask_path,
                video.definitive_mask_path,
                video.scene_labels_path,
                court_paths[video.name],
            )
        )
    clip_paths = sorted(clips_dir.glob("*.mp4"))
    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "settings": {
            "videos": video_names,
            "pilot_cases": None if all_eligible else pilot_cases,
            "selection_mode": selection_mode,
            "selected_count": len(selected),
            "eligible_count": len(eligible),
            "excluded_count": len(excluded),
            "context_seconds": context_seconds,
            "max_frames": max_frames,
            "storyboard_fps": STORYBOARD_FPS,
            "width": WIDTH,
            "height": HEIGHT,
        },
        "alignment": alignments,
        "inputs": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in input_paths
        ],
        "outputs": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in clip_paths
        ],
        "manifest_sha256": hashlib.sha256(
            (inference_dir / "manifest.json").read_bytes()
        ).hexdigest(),
    }
    _write_json(scoring_dir / "provenance.json", provenance)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--scene-labels-dir", type=Path, required=True)
    parser.add_argument("--video", action="append", dest="videos", required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--pilot-cases", type=int)
    selection.add_argument("--all-eligible", action="store_true")
    parser.add_argument("--context-seconds", action="append", type=int, required=True)
    parser.add_argument("--max-frames", type=int, default=96)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pilot_cases = DEFAULT_PILOT_CASES if args.pilot_cases is None else args.pilot_cases
    build_multiscale_trials(
        args.artifacts_root,
        args.repo_root,
        args.scene_labels_dir,
        args.out,
        video_names=tuple(args.videos),
        pilot_cases=pilot_cases,
        context_seconds=tuple(args.context_seconds),
        max_frames=args.max_frames,
        all_eligible=args.all_eligible,
    )


if __name__ == "__main__":
    main()
