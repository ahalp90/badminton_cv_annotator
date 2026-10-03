"""Plan representative courts and three complete-rally galleries from saved results."""

import argparse
from pathlib import Path

import pandas as pd

from scripts import evaluate_courts_fast_robust as evaluator
from scripts.render_court_trial_rallies import rank_rallies


def plan(input_root: Path, dataset_root: Path, output_dir: Path) -> None:
    """Rebuild the frame requests and captions, including courtless rally scenes."""
    scenes = pd.read_csv(input_root / "per_scene.csv.gz")
    videos = pd.read_csv(input_root / "per_video.csv.gz")
    rallies = pd.read_csv(input_root / "per_rally.csv.gz")
    requests = evaluator.build_render_requests(scenes, videos)
    ranked = rank_rallies(scenes, rallies).drop_duplicates("video_id").head(3)
    cohort = {entry["id"]: entry for entry in evaluator.read_json_gz(dataset_root / "cohort.json.gz")}
    captions = []
    for rally in ranked.to_dict("records"):
        video_id = rally["video_id"]
        output = evaluator.read_json_gz(dataset_root / "videos" / f"{video_id}.json.gz")
        folder = f"rally_samples/{video_id}_{rally['set']}_rally{rally['rally']:02d}"
        covered = 0
        for scene_index, scene in enumerate(output["scenes"]):
            start = max(scene["start_frame"], rally["start_frame"])
            end = min(scene["end_frame"], rally["end_frame"])
            if start >= end:
                continue
            frame_index = (start + end) // 2
            covered += end - start
            stem = f"{video_id}_scene{scene_index:04d}_frame{frame_index}"
            image = f"{folder}/{stem}.png"
            requests.append({
                "id": stem,
                "source_video": cohort[video_id]["video"],
                "frame_index": frame_index,
                "native_size": output["native_size"],
                "corners_native_px": scene["corners_native_px"],
                "output": image,
            })
            captions.append({
                "video_id": video_id, "dataset": rally["dataset"], "set": rally["set"], "rally": rally["rally"],
                "rally_start": rally["start_frame"], "rally_end": rally["end_frame"],
                "scene_index": scene_index, "overlap_start": start, "overlap_end": end,
                "frame_index": frame_index, "status": scene["status"], "image": image,
            })
        if covered != rally["end_frame"] - rally["start_frame"]:
            raise ValueError(f"{video_id}: saved scenes do not cover the selected rally")
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator.write_json_gz(output_dir / "render_requests.json.gz", requests)
    pd.DataFrame(captions).to_csv(output_dir / "rally_samples.csv.gz", index=False)
    print(f"{len(requests)} stills; {len(captions)} rally sub-scenes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="evaluation tables directory")
    parser.add_argument("--dataset", type=Path, required=True, help="court predictions: cohort.json.gz and videos/")
    parser.add_argument("--output", type=Path, required=True, help="frame requests and rally captions")
    args = parser.parse_args()
    plan(args.input, args.dataset, args.output)


if __name__ == "__main__":
    main()
