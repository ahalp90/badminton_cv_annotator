"""Draw the three court trial methods' saved courts on review clips and still frames.

``plan`` picks the three rallies with the most scene cuts, one per video, and each
method's representative court for the eight trial videos. ``fetch`` runs on the
host that holds the source videos and decodes only the frames the plan needs.
``render`` draws each method's saved courts locally with the evaluator's
``draw_court``, so clips and stills share the drawing used for the original gallery.

A clip shows its rally plus a short lead and tail for context. The cut ranking uses
the unpadded rally: first labelled contact to last labelled contact plus one.

Run from the repo root with ``PYTHONPATH=.:src``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from scripts import evaluate_courts_fast_robust as evaluator

EXPERIMENT_DIR = Path("experiments/court_detector/evidence")
EVALUATION_DIR = EXPERIMENT_DIR / "baseline"
TRIAL_RESULTS = EXPERIMENT_DIR / "inputs" / "player_tiebreak_results"
LABEL_ROOTS = {
    "ShuttleSet": Path("data/shuttleset/set"),
    "ShuttleSet22": Path("data/shuttleset22/set"),
}
HOST_VIDEOS = {
    "carmack": ("sset_11", "sset_21", "sset_30", "sset_36"),
    "bourbaki": ("ss22_27", "ss22_43", "ss22_44", "ss22_51"),
}
# method -> (stage, output directory) inside each host's trial results
METHODS = {
    "court_sharing_patched": ("stage_a", "videos"),
    "score_first": ("stage_a", "trial_videos"),
    "search_without_player_rejection": ("stage_b", "trial_videos"),
}
SCENE_KEY = ["view_id", "start_frame", "end_frame", "frame_index"]
REVIEW_RALLIES = 3
LEAD_SECONDS = 1
TAIL_SECONDS = 2
# The fetched copy is only an intermediate; near-lossless keeps the review encode's
# losses from stacking on a second visible generation.
EXTRACT_CRF = 12
REVIEW_CRF = 18
REPRESENTATIVE_METRICS = ["scene_index", "view_id", "view_group", "frame_index_in_rally", *evaluator.SCORE_COLUMNS]


def frame_name(video_id: str, frame_index: int) -> str:
    return f"{video_id}_frame_{frame_index}.png"


def rally_stem(clip: dict) -> str:
    return f"{clip['video_id']}_{clip['set']}_rally{clip['rally']:02d}"


def scene_partition(output: dict) -> list[list]:
    """:return: One (view id, start, end, sampled frame) row per scene of a detector output."""
    return [[scene[key] for key in SCENE_KEY] for scene in output["scenes"]]


def read_outputs(video_id: str) -> dict[str, dict]:
    """:return: Each method's detector output for one trial video, checked to share one scene partition."""
    host = next(host for host, video_ids in HOST_VIDEOS.items() if video_id in video_ids)
    outputs = {}
    for method, (stage, directory) in METHODS.items():
        outputs[method] = evaluator.read_json_gz(TRIAL_RESULTS / host / stage / directory / f"{video_id}.json.gz")
    partitions = [scene_partition(output) for output in outputs.values()]
    if any(partition != partitions[0] for partition in partitions):
        raise ValueError(f"{video_id}: the methods' scene partitions differ")
    return outputs


def rank_rallies(per_scene: pd.DataFrame, per_rally: pd.DataFrame) -> pd.DataFrame:
    """Order trial-video rallies by scene cuts inside the rally, then by length, then by video id."""
    cuts = []
    for rally in per_rally.itertuples():
        scenes = per_scene[per_scene["video_id"] == rally.video_id]
        starts_inside = (scenes["start_frame"] > rally.start_frame) & (scenes["start_frame"] < rally.end_frame)
        cuts.append(int(starts_inside.sum()))
    ranked = per_rally.assign(scene_cuts=cuts, rally_frames=per_rally["end_frame"] - per_rally["start_frame"])
    return ranked.sort_values(["scene_cuts", "rally_frames", "video_id"], ascending=[False, False, True])


def plan_clip(rally: pd.Series, video: pd.Series, outputs: dict[str, dict]) -> dict[str, object]:
    """Pad one rally for review and list every scene the padded clip touches, with each method's court."""
    fps = float(video["fps"])
    frame_count = int(video["frame_count"])
    rally_start, rally_end = int(rally["start_frame"]), int(rally["end_frame"])
    clip_start = max(0, rally_start - round(LEAD_SECONDS * fps))
    clip_end = min(frame_count, rally_end + round(TAIL_SECONDS * fps))
    scenes = []
    for scene_index, scene in enumerate(outputs["court_sharing_patched"]["scenes"]):
        overlap_start, overlap_end = max(scene["start_frame"], clip_start), min(scene["end_frame"], clip_end)
        if overlap_start >= overlap_end:
            continue
        sampled_in_clip = clip_start <= scene["frame_index"] < clip_end
        still_frame = scene["frame_index"] if sampled_in_clip else (overlap_start + overlap_end) // 2
        courts = {}
        for method, output in outputs.items():
            method_scene = output["scenes"][scene_index]
            courts[method] = {"status": method_scene["status"], "corners_native_px": method_scene["corners_native_px"]}
        scenes.append({
            "scene_index": scene_index,
            "view_id": scene["view_id"],
            "start_frame": scene["start_frame"],
            "end_frame": scene["end_frame"],
            "sampled_frame": scene["frame_index"],
            "still_frame": still_frame,
            "still_is_sampled_frame": sampled_in_clip,
            "overlaps_unpadded_rally": scene["start_frame"] < rally_end and scene["end_frame"] > rally_start,
            "courts": courts,
        })
    return {
        "video_id": rally["video_id"],
        "dataset": rally["dataset"],
        "set": rally["set"],
        "rally": int(rally["rally"]),
        "source_video": video["source_video"],
        "fps": fps,
        "frame_count": frame_count,
        "native_size": [int(video["native_width"]), int(video["native_height"])],
        "rally_start_frame": rally_start,
        "rally_end_frame": rally_end,
        "scene_cuts_in_rally": int(rally["scene_cuts"]),
        "clip_start_frame": clip_start,
        "clip_end_frame": clip_end,
        "lead_frames": rally_start - clip_start,
        "tail_frames": clip_end - rally_end,
        "scenes": scenes,
    }


def plan_representatives(video_id: str, video: pd.Series, outputs: dict[str, dict]) -> list[dict[str, object]]:
    """Choose each method's representative court for one video, as the evaluator's analysis does."""
    dataset = video["dataset"]
    labels = LABEL_ROOTS[dataset]
    source_id = int(video["source_id"])
    homography = evaluator.read_label_table(labels, "homography").set_index("id").loc[source_id]
    official, _ = evaluator.read_official_corners(homography, dataset)
    match_dir = labels / evaluator.read_label_table(labels, "match").set_index("id").loc[source_id, "video"]
    rallies, _ = evaluator.read_rallies(match_dir, int(video["frame_count"]))
    entry = {"id": video_id, "dataset": dataset}
    representatives = []
    for method, output in outputs.items():
        scenes, _ = evaluator.analyse_video(entry, output, official, rallies)
        chosen = scenes[scenes["is_representative"]]
        if len(chosen) != 1:
            raise ValueError(f"{video_id} {method}: expected one representative scene, found {len(chosen)}")
        scene = chosen.iloc[0]
        representative: dict[str, object] = {
            "method": method,
            "video_id": video_id,
            "source_video": video["source_video"],
            "native_size": [int(video["native_width"]), int(video["native_height"])],
            "frame_index": int(scene["frame_index"]),
            "corners_native_px": scene[evaluator.CORNER_COLUMNS].to_numpy(dtype=float).reshape(4, 2).tolist(),
        }
        # pandas' JSON writer turns NumPy scalars into plain values.
        representative |= json.loads(scene[REPRESENTATIVE_METRICS].to_json())
        representatives.append(representative)
    return representatives


def plan(output_path: Path, reusable_frames: Path | None) -> None:
    """Write the clip and representative-frame plan, and note which raw frames already exist locally."""
    per_video = pd.read_csv(EVALUATION_DIR / "per_video.csv.gz").set_index("video_id")
    per_scene = pd.read_csv(EVALUATION_DIR / "per_scene.csv.gz")
    per_rally = pd.read_csv(EVALUATION_DIR / "per_rally.csv.gz")
    trial_videos = [video_id for video_ids in HOST_VIDEOS.values() for video_id in video_ids]
    outputs = {video_id: read_outputs(video_id) for video_id in trial_videos}
    for video_id, video_outputs in outputs.items():
        original = evaluator.read_json_gz((EXPERIMENT_DIR / "inputs" / "videos") / f"{video_id}.json.gz")
        if scene_partition(video_outputs["court_sharing_patched"]) != scene_partition(original):
            raise ValueError(f"{video_id}: trial scenes differ from the original run's scenes")

    ranked = rank_rallies(per_scene, per_rally[per_rally["video_id"].isin(trial_videos)])
    chosen = ranked.drop_duplicates("video_id").head(REVIEW_RALLIES)
    print(ranked.head(8)[["video_id", "set", "rally", "start_frame", "end_frame", "scene_cuts", "rally_frames"]])
    print("best rally per video:", ranked.drop_duplicates("video_id")[["video_id", "scene_cuts"]].values.tolist())
    clips = []
    for _, rally in chosen.iterrows():
        clips.append(plan_clip(rally, per_video.loc[rally["video_id"]], outputs[rally["video_id"]]))

    # The original evaluation fetched raw frames under its request names; reuse
    # those that show the same frame of the same video.
    original_requests = evaluator.read_json_gz(EVALUATION_DIR / "render_requests.json.gz")
    reusable = {(request["id"], request["frame_index"]): request["output"] for request in original_requests}
    representatives = []
    for video_id in trial_videos:
        for representative in plan_representatives(video_id, per_video.loc[video_id], outputs[video_id]):
            request_output = reusable.get((video_id, representative["frame_index"]))
            reuse_path = None
            if request_output is not None and reusable_frames is not None:
                reuse_path = reusable_frames / (Path(request_output).stem + ".png")
            representative["reused_frame"] = None if reuse_path is None or not reuse_path.exists() else str(reuse_path)
            representatives.append(representative)

    payload = {
        "schema": "court-trial-rally-review/1",
        "methods": {method: f"{stage}/{directory}" for method, (stage, directory) in METHODS.items()},
        "rally_rule": "most scene starts strictly inside [first contact, last contact + 1), then longer, then video id",
        "padding_seconds": {"lead": LEAD_SECONDS, "tail": TAIL_SECONDS},
        "clips": clips,
        "representatives": representatives,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluator.write_json_gz(output_path, payload)
    print(f"{len(clips)} clips, {len(representatives)} representatives, "
          f"{sum(item['reused_frame'] is not None for item in representatives)} representative frames reused")


def probe_frame_rate(video_path: str) -> str:
    """:return: The video stream's exact frame rate as ffprobe's rational, such as ``30000/1001``."""
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
               "-of", "default=noprint_wrappers=1:nokey=1", video_path]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def x264_writer(path: Path, width: int, height: int, frame_rate: str, crf: int) -> subprocess.Popen:
    """Start an ffmpeg encoder that takes raw BGR frames on stdin at the source's frame rate."""
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", frame_rate, "-i", "-",
               "-c:v", "libx264", "-threads", "2", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(path)]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def probe_encoded(path: Path) -> dict[str, object]:
    """:return: A written clip's decoded frame count, size, frame rate, duration and file size."""
    command = ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
               "-show_entries", "stream=nb_read_frames,width,height,r_frame_rate,pix_fmt,codec_name:format=duration",
               "-of", "json", str(path)]
    probed = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    stream = probed["streams"][0]
    return {
        "frames": int(stream["nb_read_frames"]),
        "size": [stream["width"], stream["height"]],
        "frame_rate": stream["r_frame_rate"],
        "pix_fmt": stream["pix_fmt"],
        "codec": stream["codec_name"],
        "duration_seconds": float(probed["format"]["duration"]),
        "bytes": path.stat().st_size,
    }


def finish_writer(writer: subprocess.Popen, path: Path) -> None:
    assert writer.stdin is not None
    writer.stdin.close()
    if writer.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}")


def fetch(plan_path: Path, output_dir: Path) -> None:
    """Decode each clip once from its requested first frame, plus missing representative frames.

    Each clip's frames go to a near-lossless intermediate and its still frames to PNG.
    OpenCV's reported next-frame position is checked after every read, using the
    same seeking convention as the detector. This does not independently verify
    a seek against decoding the whole source from its first frame.
    """
    payload = evaluator.read_json_gz(plan_path)
    (output_dir / "clips").mkdir(parents=True, exist_ok=True)
    (output_dir / "frames").mkdir(parents=True, exist_ok=True)
    frame_rates = {}
    for clip in payload["clips"]:
        width, height = clip["native_size"]
        frame_rates[clip["video_id"]] = probe_frame_rate(clip["source_video"])
        destination = output_dir / "clips" / f"{rally_stem(clip)}.mp4"
        still_frames = {scene["still_frame"] for scene in clip["scenes"]}
        capture = cv2.VideoCapture(clip["source_video"])
        capture.set(cv2.CAP_PROP_POS_FRAMES, clip["clip_start_frame"])
        writer = x264_writer(destination, width, height, frame_rates[clip["video_id"]], EXTRACT_CRF)
        assert writer.stdin is not None
        for frame_index in range(clip["clip_start_frame"], clip["clip_end_frame"]):
            ok, image = capture.read()
            next_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
            if not ok or next_frame != frame_index + 1 or image.shape[:2] != (height, width):
                raise RuntimeError(f"{clip['video_id']}: could not decode frame {frame_index}; next={next_frame}")
            writer.stdin.write(image.tobytes())
            still_path = output_dir / "frames" / frame_name(clip["video_id"], frame_index)
            if frame_index in still_frames and not cv2.imwrite(str(still_path), image):
                raise OSError(f"cannot write {still_path}")
        capture.release()
        finish_writer(writer, destination)
        print(f"{destination.name}: frames [{clip['clip_start_frame']}, {clip['clip_end_frame']})", flush=True)

    requests = []
    for representative in payload["representatives"]:
        if representative["reused_frame"] is None:
            output = frame_name(representative["video_id"], representative["frame_index"])
            requests.append({"id": representative["video_id"], "source_video": representative["source_video"],
                             "frame_index": representative["frame_index"], "output": output})
    unique_requests = list({request["output"]: request for request in requests}.values())
    requests_path = output_dir / "frame_requests.json.gz"
    evaluator.write_json_gz(requests_path, unique_requests)
    evaluator.fetch_frames(requests_path, output_dir / "frames")
    evaluator.write_json_gz(output_dir / "frame_rates.json.gz", frame_rates)


def compress_png(path: Path) -> None:
    """Quantise without dithering, so 1 px lines stay solid, then optimise losslessly."""
    quantise = ["pngquant", "--force", "--ext", ".png", "--speed", "3", "--nofs", "256", "--", str(path)]
    subprocess.run(quantise, check=True)
    subprocess.run(["oxipng", "-q", "-o", "2", "--strip", "safe", str(path)], check=True)


def draw_still(raw_path: Path, corners: list | None, native_size: list[int], destination: Path) -> None:
    """Draw one saved court on a raw frame, or copy the frame unchanged when the scene has no court."""
    image = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
    if image is None or list(image.shape[1::-1]) != native_size:
        raise ValueError(f"{raw_path} is missing or not {native_size}")
    if corners is not None:
        evaluator.draw_court(image, np.array(corners, dtype=float))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), image):
        raise OSError(f"cannot write {destination}")
    compress_png(destination)


def render_clip(clip: dict, extract: Path, frame_rate: str, output_dir: Path) -> dict[str, str]:
    """Decode one extracted clip once and encode one review clip per method from it.

    :return: Review clip paths by method, relative to ``output_dir``.
    """
    width, height = clip["native_size"]
    # The scene that holds each clip frame; the scenes partition the video.
    clip_start, clip_end = clip["clip_start_frame"], clip["clip_end_frame"]
    scene_of_frame = []  # one per clip frame: position in clip["scenes"]
    for scene_position, scene in enumerate(clip["scenes"]):
        first, last = max(scene["start_frame"], clip_start), min(scene["end_frame"], clip_end)
        scene_of_frame += [scene_position] * (last - first)
    if len(scene_of_frame) != clip_end - clip_start:
        raise ValueError(f"{rally_stem(clip)}: scenes do not cover the clip")

    paths = {method: Path("clips") / f"{rally_stem(clip)}_{method}.mp4" for method in METHODS}
    writers = {}
    for method, path in paths.items():
        writers[method] = x264_writer(output_dir / path, width, height, frame_rate, REVIEW_CRF)
    capture = cv2.VideoCapture(str(extract))
    decoded = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        if decoded >= len(scene_of_frame) or image.shape[:2] != (height, width):
            raise ValueError(f"{extract}: more frames than planned or wrong size at frame {decoded}")
        scene = clip["scenes"][scene_of_frame[decoded]]
        for method, writer in writers.items():
            corners = scene["courts"][method]["corners_native_px"]
            drawn = image.copy()
            if corners is not None:
                evaluator.draw_court(drawn, np.array(corners, dtype=float))
            assert writer.stdin is not None
            writer.stdin.write(drawn.tobytes())
        decoded += 1
    capture.release()
    for method, writer in writers.items():
        finish_writer(writer, paths[method])
    if decoded != len(scene_of_frame):
        raise ValueError(f"{extract}: decoded {decoded} frames, planned {len(scene_of_frame)}")
    return {method: str(path) for method, path in paths.items()}


def render(plan_path: Path, fetched_dir: Path, output_dir: Path) -> None:
    """Draw every method's courts on the fetched clips and frames, then write the manifest."""
    payload = evaluator.read_json_gz(plan_path)
    frame_rates = evaluator.read_json_gz(fetched_dir / "frame_rates.json.gz")
    (output_dir / "clips").mkdir(parents=True, exist_ok=True)
    for clip in payload["clips"]:
        extract = fetched_dir / "clips" / f"{rally_stem(clip)}.mp4"
        clip["frame_rate"] = frame_rates[clip["video_id"]]
        clip["review_clips"] = render_clip(clip, extract, clip["frame_rate"], output_dir)
        clip["review_clip_checks"] = {}
        for method, path in clip["review_clips"].items():
            checks = probe_encoded(output_dir / path)
            planned_frames = clip["clip_end_frame"] - clip["clip_start_frame"]
            if checks["frames"] != planned_frames or checks["size"] != clip["native_size"]:
                raise ValueError(f"{path}: wrote {checks['frames']} frames at {checks['size']}")
            clip["review_clip_checks"][method] = checks
        for scene in clip["scenes"]:
            raw_path = fetched_dir / "frames" / frame_name(clip["video_id"], scene["still_frame"])
            scene["stills"] = {}
            for method in METHODS:
                still = (Path("scene_stills") / method /
                         f"{rally_stem(clip)}_scene{scene['scene_index']:04d}_frame{scene['still_frame']}_{method}.png")
                corners = scene["courts"][method]["corners_native_px"]
                draw_still(raw_path, corners, clip["native_size"], output_dir / still)
                scene["stills"][method] = str(still)
        print(f"rendered {rally_stem(clip)}", flush=True)

    for representative in payload["representatives"]:
        fetched = fetched_dir / "frames" / frame_name(representative["video_id"], representative["frame_index"])
        raw_path = Path(representative["reused_frame"] or fetched)
        method = representative["method"]
        still = (Path("representative_courts") / method /
                 f"{method}_{representative['video_id']}_frame{representative['frame_index']}.png")
        draw_still(raw_path, representative["corners_native_px"], representative["native_size"], output_dir / still)
        representative["png"] = str(still)
    evaluator.write_json_gz(output_dir / "manifest.json.gz", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan", help="choose rallies and representative frames")
    plan_parser.add_argument("--output", type=Path, required=True, help="plan .json.gz")
    plan_parser.add_argument("--reusable-frames", type=Path,
                             help="raw frames fetched for the original evaluation")
    fetch_parser = commands.add_parser("fetch", help="decode the planned frames on the video host")
    fetch_parser.add_argument("--plan", type=Path, required=True)
    fetch_parser.add_argument("--output", type=Path, required=True)
    render_parser = commands.add_parser("render", help="draw the saved courts locally")
    render_parser.add_argument("--plan", type=Path, required=True)
    render_parser.add_argument("--fetched", type=Path, required=True)
    render_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "plan":
        plan(args.output, args.reusable_frames)
    elif args.command == "fetch":
        fetch(args.plan, args.output)
    else:
        render(args.plan, args.fetched, args.output)


if __name__ == "__main__":
    main()
