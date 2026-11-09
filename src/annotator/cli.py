"""Run the complete auto-annotator on one video's saved vision inputs."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from annotator.excluded_frames.shuttle_hallucination import grade_track
from annotator.models import load_models
from annotator.saved_inputs import load_saved_video_inputs
from dataset_builder.shuttle_evidence import inpaint_mask_from_record
from dataset_builder.vision import (
    AnnotationOutput,
    load_json_gz,
    run_full_annotation_stage,
)


def annotate_saved_video(run_dir: Path, video_id: str, model_dir: Path | None, output_dir: Path) -> AnnotationOutput:
    """Read saved extraction stages and write the same outputs as the dataset builder.

    Use heuristic-only annotation when model_dir is None.

    The extraction directory needs metadata, shuttle, pose and court stages. No
    run manifest, labels, source video or extraction model weights are needed.
    The output directory must be new or empty so a separate annotation run cannot
    silently replace the dataset builder's saved results.
    """
    if not video_id or video_id in ('.', '..') or Path(video_id).name != video_id:
        raise ValueError('video ID must be one directory name')
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f'Output directory must be new or empty: {output_dir}')
    models = load_models(model_dir) if model_dir is not None else None
    saved = load_saved_video_inputs(run_dir, video_id)
    metadata = saved.metadata
    shuttle_dir = run_dir / 'stages' / 'shuttle' / video_id
    sidecars = list(shuttle_dir.glob('*_inpaint_mask.json.gz'))
    if len(sidecars) != 1:
        raise ValueError(f'{shuttle_dir}: expected one *_inpaint_mask.json.gz file; found {len(sidecars)}')
    fill_mask = inpaint_mask_from_record(
        load_json_gz(sidecars[0]), metadata.frame_count, metadata.source_path.name,
    )
    expected_grades, _ = grade_track(saved.track)
    if not np.array_equal(saved.shuttle_grades, expected_grades):
        raise ValueError('persisted shuttle guard codes differ from the final track')
    return run_full_annotation_stage(
        video_id=video_id,
        metadata=metadata,
        track=saved.track,
        inpaint_fill_mask=fill_mask,
        shuttle_grades=saved.shuttle_grades,
        pose=saved.pose,
        court=saved.court,
        models=models,
        heuristic_only=model_dir is None,
        output_dir=output_dir,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Annotate saved inputs without running acquisition or vision extraction."""
    parser = argparse.ArgumentParser(prog='python -m annotator', description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path,
                        help='Extraction directory containing stages/metadata, shuttle, pose and court')
    parser.add_argument('--video-id', required=True, help='Video directory name within each extraction stage')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--models', type=Path, help='Directory containing the fitted annotator bundle')
    mode.add_argument('--heuristic-only', action='store_true', help='Run heuristic contacts without a model bundle')
    parser.add_argument('--output-dir', required=True, type=Path, help='New or empty directory for annotation results')
    parser.add_argument('--progress', choices=('auto', 'on', 'off'), default='auto',
                        help='Terminal progress: auto on a TTY, on to force, off for plain logs')
    arguments = parser.parse_args(argv)
    from dataset_builder.progress import RunProgress

    display = RunProgress(('annotation',), arguments.progress,
                          log_dir=arguments.output_dir.parent / 'terminal-logs')
    name = f'annotation:{arguments.video_id}'
    with display.stage(name) as stage:
        output = annotate_saved_video(arguments.run_dir, arguments.video_id, arguments.models, arguments.output_dir)
    if display.enabled:
        display.finish(name, stage, counts=(('rallies', len(output.run.result.rallies)),))
    print(f'Saved {len(output.run.result.rallies)} rallies to {output.artifacts.result}')
    return 0
