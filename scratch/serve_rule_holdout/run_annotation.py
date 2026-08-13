"""Build label-blind annotator evidence for the held-out videos."""

from __future__ import annotations

import argparse
from pathlib import Path

from annotator.video_metadata import probe_video_metadata
from dataset_builder.vision import (
    convert_tracknet_csv_stage,
    load_court_vision,
    run_full_annotation_stage,
)
from run_court import VIDEO_IDS, load_pose_arrays


def run(data_root: Path) -> None:
    """Run the unchanged annotator from frozen vision evidence."""
    for video_id in VIDEO_IDS:
        metadata = probe_video_metadata(data_root / "videos" / f"{video_id}.mp4")
        proxy = probe_video_metadata(
            data_root / "stages" / "tracknet_input" / video_id / f"{video_id}.avi"
        )
        track = convert_tracknet_csv_stage(
            data_root / "stages" / "shuttle_csv" / f"{video_id}_ball.csv",
            video_id=video_id,
            metadata=proxy,
            output_path=data_root / "stages" / "shuttle" / video_id / "shuttle_track.npy.xz",
        ).require_value().track
        court = load_court_vision(
            data_root / "stages" / "court" / video_id,
            video_id=video_id,
            frame_count=metadata.frame_count,
            resolution=(float(metadata.width), float(metadata.height)),
        )
        annotation = run_full_annotation_stage(
            video_id=video_id,
            metadata=metadata,
            track=track,
            pose=load_pose_arrays(data_root, video_id),
            court=court,
            output_dir=data_root / "stages" / "annotation" / video_id,
        ).require_value()
        print(f"{video_id} complete rallies={len(annotation.run.result.spans)}", flush=True)


def main() -> None:
    """Parse the evidence root and run annotation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.data_root)


if __name__ == "__main__":
    main()
