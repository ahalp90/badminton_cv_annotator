"""Run the complete auto-annotator on one video's saved vision inputs."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from annotator.excluded_frames.shuttle_hallucination import grade_track
from annotator.models import load_models
from dataset_builder.shuttle_evidence import (
    GUARD_CODES_FILENAME,
    inpaint_mask_from_record,
)
from dataset_builder.vision import (
    TRACK_FILENAME,
    AnnotationOutput,
    load_court_vision,
    load_json_gz,
    load_npy_xz,
    load_pose_arrays,
    run_full_annotation_stage,
)
from shared.shuttle_track_validation import validate_shuttle_track
from shared.video_metadata import VideoMetadata


def annotate_saved_video(run_dir: Path, video_id: str, model_dir: Path, output_dir: Path) -> AnnotationOutput:
    """Read saved extraction stages and write the same outputs as the dataset builder.

    The extraction directory needs metadata, shuttle, pose and court stages. No
    run manifest, labels, source video or extraction model weights are needed.
    The output directory must be new or empty so a separate annotation run cannot
    silently replace the dataset builder's saved results.
    """
    if not video_id or video_id in ('.', '..') or Path(video_id).name != video_id:
        raise ValueError('video ID must be one directory name')
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f'Output directory must be new or empty: {output_dir}')
    models = load_models(model_dir)
    stages = run_dir / 'stages'
    metadata = VideoMetadata.from_dict(load_json_gz(stages / 'metadata' / video_id / 'video_metadata.json.gz'))
    shuttle_dir = stages / 'shuttle' / video_id
    sidecars = list(shuttle_dir.glob('*_inpaint_mask.json.gz'))
    if len(sidecars) != 1:
        raise ValueError(f'{shuttle_dir}: expected one *_inpaint_mask.json.gz file; found {len(sidecars)}')
    fill_mask = inpaint_mask_from_record(
        load_json_gz(sidecars[0]), metadata.frame_count, metadata.source_path.name,
    )
    track = load_npy_xz(shuttle_dir / TRACK_FILENAME)
    validate_shuttle_track(track, metadata.frame_count)
    guard_codes = load_npy_xz(shuttle_dir / GUARD_CODES_FILENAME)
    expected_codes, _ = grade_track(track)
    if not np.array_equal(guard_codes, expected_codes):
        raise ValueError('persisted shuttle guard codes differ from the final track')
    pose = load_pose_arrays(stages / 'pose' / video_id, metadata.frame_count)
    court = load_court_vision(
        stages / 'court' / video_id,
        video_id=video_id,
        frame_count=metadata.frame_count,
        resolution=(float(metadata.width), float(metadata.height)),
    )
    return run_full_annotation_stage(
        video_id=video_id,
        metadata=metadata,
        track=track,
        inpaint_fill_mask=fill_mask,
        guard_codes=guard_codes,
        pose=pose,
        court=court,
        models=models,
        output_dir=output_dir,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Annotate saved inputs without running acquisition or vision extraction."""
    parser = argparse.ArgumentParser(prog='python -m annotator', description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path,
                        help='Extraction directory containing stages/metadata, shuttle, pose and court')
    parser.add_argument('--video-id', required=True, help='Video directory name within each extraction stage')
    parser.add_argument('--models', required=True, type=Path, help='Directory containing the fitted annotator bundle')
    parser.add_argument('--output-dir', required=True, type=Path, help='New or empty directory for annotation results')
    parser.add_argument('--progress', choices=('auto', 'on', 'off'), default='auto',
                        help='Terminal progress: auto on a TTY, on to force, off for plain logs')
    arguments = parser.parse_args(argv)
    from dataset_builder.progress import RunProgress

    display = RunProgress(SimpleNamespace(fixed_sources=None, commentary_enabled=True, pose_shards=1),
                          ('annotation',), arguments.progress,
                          log_dir=arguments.output_dir.parent / 'terminal-logs')
    name = f'annotation:{arguments.video_id}'
    with display.stage(name) as stage:
        output = annotate_saved_video(arguments.run_dir, arguments.video_id, arguments.models, arguments.output_dir)
    if display.enabled:
        from dataset_builder.models import StageOutcome

        display.finish(name, SimpleNamespace(outcome=StageOutcome.PROCESSED, reused=False, reason=None),
                       SimpleNamespace(counts=(('rallies', len(output.run.result.rallies)),)), stage)
    print(f'Saved {len(output.run.result.rallies)} rallies to {output.artifacts.result}')
    return 0
