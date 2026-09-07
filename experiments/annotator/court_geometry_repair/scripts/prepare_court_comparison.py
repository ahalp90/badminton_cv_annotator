"""Build fresh court-dependent stages and contact features for one SS22 video."""

from __future__ import annotations

import argparse
import base64
import importlib
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from annotator import replay_mask
from annotator.court_evidence import SceneEvidence, build_detected_court_evidence
from annotator.court_views import HASH_SIZE, VIEW_RESOLUTION, CourtView
from annotator.video_metadata import VideoMetadata, probe_video_fps
from courtkeynet.court_corners import CourtQuad, FallbackDiagnostics
from dataset_builder.vision import (
    CourtVision,
    load_court_vision,
    load_json_gz,
    load_npy_xz,
    load_pose_arrays,
    persist_court_vision,
    run_full_annotation_stage,
    save_json_gz,
    save_npy_xz,
)
from scratch.contact_det.scripts.freeze_contact_evidence import FixtureSpec
from scratch.contact_det.scripts.freeze_tree_contact_features import _fixture_rows
from scratch.contact_det_full_ds_fit.scripts.inpaint_shuttleset22_tracks import (
    output_paths,
)
from scratch.contact_det_full_ds_fit.scripts.prepare_shuttleset22_predictions import (
    fill_mask_from_sidecar,
)

EVIDENCE_SCHEMA = 'scene-court-evidence/2'
PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'


def _quad(payload: dict[str, object] | None, *, path: Path | None = None) -> CourtQuad | None:
    if payload is None:
        return None
    record = payload  # Producer: rebuild_scene_courts.py.
    for field_name in ('alternative_corners_px', 'line_segments_px'):
        if field_name not in record:
            location = f' in {path}' if path is not None else ''
            raise ValueError(
                f'cached court evidence{location} omits {field_name}; '
                'regenerate with rebuild_scene_courts.py'
            )
    diagnostics_payload = record["diagnostics"]
    diagnostics = None
    if diagnostics_payload is not None:
        diagnostics = FallbackDiagnostics(
            reproj_line_px=float(diagnostics_payload["reproj_line_px"]),
            reproj_anchor_px=float(diagnostics_payload["reproj_anchor_px"]),
            gate_line_frac=float(diagnostics_payload["gate_line_frac"]),
            gate_anchor_frac=float(diagnostics_payload["gate_anchor_frac"]),
            n_lines_used=int(diagnostics_payload["n_lines_used"]),
            n_correspondences=int(diagnostics_payload["n_correspondences"]),
            max_sagitta_px=float(diagnostics_payload["max_sagitta_px"]),
        )
    alternatives = tuple(
        np.asarray(corners, dtype=np.float32) for corners in record['alternative_corners_px']
    )
    return CourtQuad(
        corners_px=np.asarray(record["corners_px"], dtype=np.float32),
        peak=np.asarray(record["peak"], dtype=np.float32),
        source=str(record["source"]),
        corner_source=tuple(str(value) for value in record["corner_source"]),
        diagnostics=diagnostics,
        line_segments_px=tuple(np.asarray(segments, dtype=float).reshape(-1, 4)
                               for segments in record['line_segments_px']),
        alternative_corners_px=alternatives,
    )


def _view(payload: dict[str, object] | None, *, path: Path) -> CourtView | None:
    """Restore a CourtView from the lossless image cache."""
    if payload is None:
        return None
    try:
        hashes = np.asarray(payload['hashes'])
        encoded = base64.b64decode(str(payload['image_png_base64']), validate=True)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f'cached court evidence view is malformed in {path}; '
            'regenerate with rebuild_scene_courts.py'
        ) from error
    if (
        hashes.dtype != np.bool_
        or hashes.ndim != 3
        or not 1 <= hashes.shape[0] <= 3
        or hashes.shape[1:] != (HASH_SIZE, HASH_SIZE)
    ):
        raise ValueError(
            f'cached court evidence view has invalid hashes in {path}; '
            'regenerate with rebuild_scene_courts.py'
        )
    if not encoded.startswith(PNG_SIGNATURE):
        raise ValueError(f'cached court view in {path} must be PNG; regenerate with rebuild_scene_courts.py')
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None or tuple(image.shape[::-1]) != VIEW_RESOLUTION:
        raise ValueError(
            f'cached court evidence view has invalid image in {path}; '
            'regenerate with rebuild_scene_courts.py'
        )
    return CourtView(hashes, image)


def _scene_evidence(
    nn_root: Path,
    cuts: Sequence[tuple[int, int]],
    video_id: int,
    *,
    frame_wh: tuple[int, int],
) -> tuple[SceneEvidence, ...]:
    evidence = []
    for scene_index, interval in enumerate(cuts):
        path = nn_root / f"video_{video_id:02d}_scene_{scene_index:04d}.json.gz"
        record = load_json_gz(path)
        if tuple(record["interval"]) != tuple(interval):
            raise ValueError(f"scene interval differs in {path}")
        samples = tuple(int(frame) for frame in record["sampled_frame_indices"])
        if record.get('evidence_schema') != EVIDENCE_SCHEMA:
            raise ValueError(
                f'{path}: cached court evidence lacks schema {EVIDENCE_SCHEMA}; '
                'regenerate with rebuild_scene_courts.py'
            )
        if tuple(record.get('frame_wh', ())) != frame_wh:
            raise ValueError(
                f'{path}: cached frame dimensions differ from {frame_wh}; '
                'regenerate with rebuild_scene_courts.py'
            )
        if 'view' not in record:
            raise ValueError(
                f'{path}: cached court evidence omits view; '
                'regenerate with rebuild_scene_courts.py'
            )
        quad = _quad(record["court_quad"], path=path)
        view = _view(record['view'], path=path)
        if quad is not None and view is None:
            raise ValueError(
                f'{path}: cached court evidence with a quad omits view; '
                'regenerate with rebuild_scene_courts.py'
            )
        evidence.append(SceneEvidence(*interval, samples, quad, view))
    return tuple(evidence)


def _find_prepared(root: Path, video_id: int) -> Path:
    matches = sorted(path for path in root.glob(f"{video_id:02d} *") if path.is_dir())
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {video_id:02d} * directory under {root}")
    return matches[0]


def _link_raw_stages(
    output: Path,
    fixture: FixtureSpec,
    prepared: Path,
    inpaint: Path,
) -> None:
    pose_dir = output / "stages" / "pose" / fixture.name
    shuttle_dir = output / "stages" / "shuttle" / fixture.name
    pose_dir.mkdir(parents=True)
    shuttle_dir.mkdir(parents=True)
    pose_files = (
        "pose_kps.npy.xz",
        "pose_bboxes.npy.xz",
        "pose_scores.npy.xz",
        "pose_kp_scores.npy.xz",
        "pose_ndet.npy.xz",
    )
    for filename in pose_files:
        (pose_dir / filename).symlink_to((prepared / filename).resolve(strict=True))
    (shuttle_dir / "shuttle_track.npy.xz").symlink_to(
        (inpaint / "shuttle_track_inpainted.npy.xz").resolve(strict=True)
    )


def _load_shuttle(
    inpaint: Path,
    frame_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = output_paths(inpaint, inpaint.name)
    track = load_npy_xz(paths.track_path)
    guard_codes = load_npy_xz(paths.guard_codes_path)
    sidecar = load_json_gz(paths.sidecar_path)
    fill_mask = fill_mask_from_sidecar(sidecar, frame_count)
    return track, fill_mask, guard_codes


def _baseline_combine_mask(
    court_present: np.ndarray | None,
    homography_rows: list[dict] | None,
    track: np.ndarray | None,
    rally_spans: list[tuple[int, int]] | None,
    n_frames: int,
    fps: float,
    *,
    non_evidence: np.ndarray | None = None,
) -> np.ndarray:
    """Reproduce the a111181 replay-mask composition for baseline runs."""
    court = replay_mask.court_absence_signal(court_present, n_frames, fps)
    perspective = replay_mask.perspective_shift_signal(homography_rows, n_frames)
    velocity = replay_mask.velocity_drop_signal(
        track,
        rally_spans,
        n_frames,
        fps,
        non_evidence=non_evidence,
        baseline_exclude=court | perspective,
    )
    return court | perspective | velocity


@contextmanager
def _baseline_replay_mask_mode(enabled: bool) -> Iterator[None]:
    """Temporarily restore the historical perspective veto for baseline runs."""
    if not enabled:
        yield
        return
    dead_mask_module = importlib.import_module("annotator.dead_mask")
    original_replay = replay_mask.combine_mask
    original_dead_mask = dead_mask_module.combine_mask
    replay_mask.combine_mask = _baseline_combine_mask
    dead_mask_module.combine_mask = _baseline_combine_mask
    try:
        yield
    finally:
        replay_mask.combine_mask = original_replay
        dead_mask_module.combine_mask = original_dead_mask


@contextmanager
def _global_tracker_mode(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    scene_module = importlib.import_module("annotator.scene_courts")
    run_module = importlib.import_module("annotator.run_video")
    original_scene = scene_module.build_scene_courts
    original_run = run_module.build_scene_courts
    scene_module.build_scene_courts = lambda *_args, **_kwargs: None
    run_module.build_scene_courts = lambda *_args, **_kwargs: None
    try:
        yield
    finally:
        scene_module.build_scene_courts = original_scene
        run_module.build_scene_courts = original_run


def prepare(arguments: argparse.Namespace) -> None:
    prepared = _find_prepared(arguments.prepared_root, arguments.video_id)
    source = arguments.sources / f"{prepared.name}.mp4"
    inpaint = arguments.inpaint_root / prepared.name
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f'cannot open {source}')
    metadata = VideoMetadata(
        source.resolve(), probe_video_fps(source),
        int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), Fraction(1),
    )
    capture.release()
    if (metadata.width, metadata.height, float(metadata.fps)) != (1920, 1080, 30.0):
        raise ValueError("This frozen comparison requires the original 1920x1080, 30 fps video")
    print(f'video {arguments.video_id}: loading cached arrays ({metadata.frame_count} frames)', flush=True)
    frame_count = metadata.frame_count
    resolution = (float(metadata.width), float(metadata.height))
    fixture = FixtureSpec(
        str(arguments.video_id), arguments.video_id, float(metadata.fps), *resolution
    )
    output = arguments.output
    output.mkdir(parents=True, exist_ok=False)
    _link_raw_stages(output, fixture, prepared, inpaint)
    pose = load_pose_arrays(prepared, frame_count)
    track, fill_mask, guard_codes = _load_shuttle(inpaint, frame_count)
    court_dir = output / "stages" / "court" / fixture.name
    court_dir.mkdir(parents=True)
    original = load_court_vision(
        prepared, video_id=fixture.name, frame_count=frame_count, resolution=resolution
    )
    if arguments.baseline:
        court = original
    else:
        result = build_detected_court_evidence(
            case_id=f"ss22_{fixture.video_id}_court_comparison",
            parent="cached_scene_nn",
            video_id=fixture.name,
            resolution=resolution,
            raw_cuts=original.raw_cuts,
            scene_evidence=_scene_evidence(
                arguments.nn_results, original.raw_cuts, fixture.video_id,
                frame_wh=(metadata.width, metadata.height),
            ),
            bboxes=pose.bboxes,
            scores=pose.scores,
            ndet=pose.ndet,
            detector_resolution=resolution,
        )
        court = CourtVision(original.raw_cuts, result)
    persist_court_vision(
        court_dir,
        video_id=fixture.name,
        court=court,
        frame_count=frame_count,
        resolution=resolution,
    )
    annotation_dir = output / "stages" / "annotation" / fixture.name
    annotation_dir.mkdir(parents=True)
    with _baseline_replay_mask_mode(arguments.baseline), _global_tracker_mode(arguments.baseline):
        run_full_annotation_stage(
            video_id=fixture.name,
            metadata=metadata,
            track=track,
            inpaint_fill_mask=fill_mask,
            guard_codes=guard_codes,
            pose=pose,
            court=court,
            output_dir=annotation_dir,
        )
        print("Annotation complete; deriving contact features", flush=True)
        rows, summary = _fixture_rows(output, fixture, "raw_per_frame")
    save_npy_xz(output / "contact_features.npy.xz", rows)
    summary.update(
        video_id=arguments.video_id, baseline=arguments.baseline, source=str(source)
    )
    save_json_gz(output / "contact_features_summary.json.gz", summary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--inpaint-root", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--nn-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-id", type=int, required=True)
    parser.add_argument("--baseline", action="store_true")
    arguments = parser.parse_args(argv)
    prepare(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
