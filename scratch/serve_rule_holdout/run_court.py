"""Build CourtKeyNet evidence from the frozen held-out pose arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from annotator.video_metadata import probe_video_metadata
from courtkeynet.wrapper import CourtKeyNetDetector
from dataset_builder.vision import PoseArrays, build_detected_court_stage


VIDEO_IDS = ("sset_20", "sset_22")


def load_pose_arrays(data_root: Path, video_id: str) -> PoseArrays:
    """Load raw pose arrays from the successful sharded publication."""
    run_id = f"issue90_{video_id.replace('_', '')}_v2"
    root = data_root / "stages" / "pose_raw_v2" / video_id / f"publish_{run_id}"
    return PoseArrays(
        kps=np.load(root / "pose_raw_kps.npy", allow_pickle=False),
        bboxes=np.load(root / "pose_raw_bboxes.npy", allow_pickle=False),
        scores=np.load(root / "pose_raw_scores.npy", allow_pickle=False),
        kp_scores=np.load(root / "pose_raw_kp_scores.npy", allow_pickle=False),
        ndet=np.load(root / "pose_raw_ndet.npy", allow_pickle=False),
    )


def run(data_root: Path, repo_root: Path) -> None:
    """Build and persist court evidence for both held-out videos."""
    detector = CourtKeyNetDetector(
        weights_path=repo_root / "src" / "courtkeynet" / "weights" / "courtkeynet_finetuned.safetensors",
        device="cuda",
        resize_mode="pad",
    )
    for video_id in VIDEO_IDS:
        metadata = probe_video_metadata(data_root / "videos" / f"{video_id}.mp4")
        result = build_detected_court_stage(
            video_id=video_id,
            metadata=metadata,
            pose=load_pose_arrays(data_root, video_id),
            detector=detector,
            output_dir=data_root / "stages" / "court" / video_id,
        ).require_value()
        print(f"{video_id} complete scenes={len(result.raw_cuts)}", flush=True)


def main() -> None:
    """Parse paths and run the frozen court stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.data_root, arguments.repo_root)


if __name__ == "__main__":
    main()
