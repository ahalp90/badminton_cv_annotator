"""TrackNetV3 shuttle trajectory extraction and normalization.

Runs TrackNetV3 inference on clip .mp4 files to produce per-clip shuttle
trajectory arrays. Both architectures share this step.

TrackNetV3 is included in the repo at ``src/shared/tracknetv3`` (trimmed to
inference only) and shares the BST training venv. Pretrained weights must be
downloaded separately — see ``src/shared/tracknetv3/README.md``.

Usage:
    python -m pipeline.shuttle_extractor [--clips-dir DIR] \
        [--tracknet-python /path/to/bst-venv/bin/python] [--profile {bst,scrape}]
"""
import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import FrameType

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from annotator.shuttle_track import validate_shuttle_track  # noqa: E402
from pipeline.config import (  # noqa: E402
    CLIPS_OUTPUT_DIR, SHUTTLE_OUTPUT_DIR, RESOLUTION_CSV_PATH,
)
from scraper.config import SCRAPE_TRACKNET_STRIDE, SCRAPE_TRACKNET_LARGE_VIDEO  # noqa: E402

DEFAULT_TRACKNET_DIR = _SRC / 'shared' / 'tracknetv3'
_DEFAULT_TRACKNET_SUBPATH = Path('ckpts') / 'TrackNet_best.pt'
_DEFAULT_INPAINTNET_SUBPATH = Path('ckpts') / 'InpaintNet_best.pt'
TRACKNET_STRIDE = 1
# The batch subprocess (batch_predict.py) is in-RAM only; per-file vendored predict.py is the
# streaming route. streaming builds its median background image from a capped sample of frames
# (1800) instead of all of them
TRACKNET_LARGE_VIDEO = False

PROFILE_DEFAULTS = {'bst': (TRACKNET_STRIDE, TRACKNET_LARGE_VIDEO), 'scrape': (SCRAPE_TRACKNET_STRIDE, SCRAPE_TRACKNET_LARGE_VIDEO)}

def _tracknet_eval_mode(stride: int) -> str:
    if stride == 1:
        return 'weight'
    if stride == 8:
        return 'nonoverlap'
    raise ValueError(f'tracknet stride must be 1 or 8, got {stride}')


def _default_csv_dir(clips_dir: Path) -> Path:
    """Default location for TrackNetV3 CSV outputs: clips_dir/../shuttle_csv."""
    return clips_dir.parent / 'shuttle_csv'


def normalize_shuttlecock(arr: np.ndarray, v_width: float, v_height: float) -> np.ndarray:
    """Normalize shuttle coordinates by video resolution.

    Scales x and y by the frame dimensions. In-frame positions become values
    in [0, 1]. A third visibility column passes through unchanged.

    :param arr: (t, 2) or (t, 3) array. Columns: x, y, [visibility].
    :param v_width: Video width in pixels.
    :param v_height: Video height in pixels.
    :return: Array with same shape, xy columns normalized.
    """
    result = arr.astype(float)
    result[:, 0] /= v_width
    result[:, 1] /= v_height
    return result


@dataclass(frozen=True)
class WholeVideoShuttle:
    """One exact source ID and its frame-ordered annotator shuttle track."""

    video_id: str
    track: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.video_id, str) or not self.video_id:
            raise ValueError('whole-video shuttle video_id must be a non-empty string')
        track = np.asarray(self.track)
        validate_shuttle_track(track)
        object.__setattr__(self, 'track', np.ascontiguousarray(track).copy())


def whole_video_csv_to_shuttle(
    csv_path: Path,
    *,
    video_id: str,
    frame_count: int,
    width: float,
    height: float,
) -> WholeVideoShuttle:
    """Validate and reindex a whole-video TrackNet CSV for the annotator.

    Frame rows may arrive in any order, but must identify each integer frame in
    ``0..frame_count-1`` exactly once. This adapter is deliberately separate
    from :func:`shuttle_csvs_to_npy`, whose clip-specific duplicate handling and
    numeric ShuttleSet IDs remain a legacy contract.
    """
    if not Path(csv_path).is_file():
        raise FileNotFoundError(f'whole-video TrackNet CSV is not a regular file: {csv_path}')
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        raise ValueError(f'frame_count must be a positive integer, got {frame_count!r}')
    for name, value in (('width', width), ('height', height)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'{name} must be a finite positive number, got {value!r}')
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f'{name} must be a finite positive number, got {value!r}')

    frame = pd.read_csv(csv_path)
    columns = ('Frame', 'X', 'Y', 'Visibility')
    missing_columns = sorted(set(columns).difference(frame.columns))
    if missing_columns:
        raise ValueError(f'whole-video TrackNet CSV is missing columns: {missing_columns}')

    frame_values = pd.to_numeric(frame['Frame'], errors='coerce').to_numpy(dtype=float)
    if not np.isfinite(frame_values).all():
        raise ValueError('TrackNet Frame values must be finite numbers')
    if not np.equal(frame_values, np.floor(frame_values)).all():
        raise ValueError('TrackNet Frame values must be integers')
    if ((frame_values < 0) | (frame_values >= frame_count)).any():
        invalid = frame_values[(frame_values < 0) | (frame_values >= frame_count)]
        raise ValueError(
            f'TrackNet Frame values must be in [0, {frame_count - 1}], got {invalid.tolist()}'
        )
    frame_ids = frame_values.astype(np.int64)
    duplicate_ids = np.unique(frame_ids[pd.Index(frame_ids).duplicated(keep=False)])
    if duplicate_ids.size:
        raise ValueError(f'TrackNet Frame values contain duplicates: {duplicate_ids.tolist()}')
    if len(frame_ids) != frame_count:
        missing_ids = np.setdiff1d(np.arange(frame_count, dtype=np.int64), frame_ids)
        raise ValueError(f'TrackNet Frame values have gaps: {missing_ids.tolist()}')

    ordered = frame.assign(_frame_id=frame_ids).set_index('_frame_id').reindex(range(frame_count))
    values = ordered.loc[:, ['X', 'Y', 'Visibility']].apply(pd.to_numeric, errors='coerce')
    shuttle_camera = values.to_numpy(dtype=float)
    shuttle_norm = normalize_shuttlecock(shuttle_camera, float(width), float(height))
    return WholeVideoShuttle(video_id=video_id, track=shuttle_norm)


# ---------------------------------------------------------------------------
# TrackNetV3 subprocess invocation
# ---------------------------------------------------------------------------
def extract_all_shuttles(
    tracknet_dir: Path,
    clips_dir: Path = CLIPS_OUTPUT_DIR,
    output_csv_dir: Path | None = None,
    model_path: Path | None = None,
    inpaintnet_path: Path | None = None,
    tracknet_python: Path | None = None,
    max_workers: int = 2,
    batch_size: int = 32,
    tracknet_stride: int = TRACKNET_STRIDE,
    large_video: bool = TRACKNET_LARGE_VIDEO,
    dry_run: bool = False,
    video_paths: Sequence[Path] | None = None,
    enable_inpainting: bool = True,
    input_mode: str = 'persisted_ffv1_proxy',
    ffmpeg: str | Path = 'ffmpeg',
    expected_frame_count: int | None = None,
    input_video_identity: Path | None = None,
) -> None:
    """Run TrackNetV3 on all clips using batch mode.

    Uses batch_predict.py to load models once per worker and iterate
    over clips in-process, avoiding the ~8s model-reload overhead per
    clip that subprocess-per-clip mode incurred.

    Each worker loads its own model copy onto the GPU, so max_workers > 1
    requires enough VRAM for multiple models (e.g. A100 40GB). On V100
    16GB, use max_workers=1.

    :param tracknet_dir: Path to the cloned TrackNetV3 repository.
    :param clips_dir: Root clips directory to scan for .mp4 files.
    :param video_paths: Optional explicit source paths. This preserves the legacy
        ``clips_dir`` scan by default while allowing selected whole-video files
        whose download container is not ``.mp4``.
    :param enable_inpainting: Whether to resolve and pass InpaintNet weights.
        Defaults to the legacy enabled behavior; dataset-builder callers can
        explicitly disable it without falling back to the default checkpoint.
    :param output_csv_dir: Directory for TrackNetV3 CSV outputs.
        Defaults to clips_dir/../shuttle_csv.
    :param model_path: Path to TrackNet weights. Defaults to tracknet_dir/ckpts/TrackNet_best.pt.
    :param inpaintnet_path: Path to InpaintNet weights. Defaults to tracknet_dir/ckpts/InpaintNet_best.pt.
    :param tracknet_python: Python executable in BST venv (shared with TrackNetV3).
        Defaults to sys.executable (assumes shared environment).
    :param max_workers: Number of parallel batch workers (default 2).
        Each worker loads its own model copy — needs enough GPU memory.
    :param batch_size: Batch size for TrackNet DataLoader (default 32).
        Safe at 32 with max_workers=2; use 64 with max_workers=1.
    :param large_video: Use large video mode (default TRACKNET_LARGE_VIDEO).
    :param input_mode: Exact in-memory stream or persisted proxy input.
    :param ffmpeg: FFmpeg executable used by exact stream mode.
    :param expected_frame_count: Canonical frame count required by exact stream mode.
    :param input_video_identity: Canonical source path recorded in the Inpaint sidecar.
    """
    # Preflight: verify TrackNetV3 is set up correctly
    if not tracknet_dir.is_dir():
        raise FileNotFoundError(f'TrackNetV3 directory not found: {tracknet_dir}')
    if not (tracknet_dir / 'batch_predict.py').exists():
        raise FileNotFoundError(f'batch_predict.py not found in: {tracknet_dir}')

    resolved_model = model_path or (tracknet_dir / _DEFAULT_TRACKNET_SUBPATH)
    if not resolved_model.exists():
        raise FileNotFoundError(f'TrackNet weights not found: {resolved_model}')

    if not isinstance(enable_inpainting, bool):
        raise ValueError('enable_inpainting must be boolean')
    if not enable_inpainting:
        resolved_inpaint = None
    else:
        resolved_inpaint = inpaintnet_path or (tracknet_dir / _DEFAULT_INPAINTNET_SUBPATH)
        if not resolved_inpaint.exists():
            print(f'  WARNING: InpaintNet weights not found: {resolved_inpaint}')
            print(f'  Running TrackNet only (no inpainting of occluded frames)')
            resolved_inpaint = None

    if not output_csv_dir:
        output_csv_dir = _default_csv_dir(clips_dir)
    output_csv_dir.mkdir(parents=True, exist_ok=True)

    if video_paths is None:
        all_clips = sorted(clips_dir.rglob('*.mp4'))
    else:
        all_clips = sorted((Path(path) for path in video_paths), key=lambda path: path.name)
        missing = [path for path in all_clips if not path.is_file()]
        if missing:
            raise FileNotFoundError(f'explicit TrackNet videos are not regular files: {missing}')
        stems = [path.stem for path in all_clips]
        if len(stems) != len(set(stems)):
            raise ValueError('explicit TrackNet video paths must have unique stems')
    if input_mode not in {'exact_ffv1_stream', 'persisted_ffv1_proxy'}:
        raise ValueError(f'unsupported TrackNet input mode: {input_mode}')
    if input_video_identity is not None and len(all_clips) != 1:
        raise ValueError('canonical TrackNet input identity requires exactly one video')
    if input_mode == 'exact_ffv1_stream':
        if len(all_clips) != 1:
            raise ValueError('exact FFV1 stream requires exactly one explicit video')
        if tracknet_stride != 8 or not large_video:
            raise ValueError('exact FFV1 stream requires stride 8 and large-video mode')
        if (
            isinstance(expected_frame_count, bool)
            or not isinstance(expected_frame_count, int)
            or expected_frame_count <= 0
        ):
            raise ValueError('exact FFV1 stream requires a positive expected frame count')
    # Filter to clips that don't already have results (dry_run processes all)
    if dry_run:
        pending = all_clips
    else:
        pending = [c for c in all_clips
                   if not (output_csv_dir / (c.stem + '_ball.csv')).exists()]

    print(f'TrackNetV3 extraction: {len(pending)} pending of {len(all_clips)} total clips')
    if not pending:
        return

    # Split pending clips across workers (round-robin so each worker
    # processes a mix of short and long clips from different videos).
    chunks = [pending[i::max_workers] for i in range(max_workers)] #TODO fix inline nesting
    chunks = [c for c in chunks if c]  # drop empty if fewer clips than workers

    python_exe = str(tracknet_python) if tracknet_python else sys.executable #TODO is this check necessary?
    batch_script = tracknet_dir / 'batch_predict.py'

    # Install signal handling before launch so partial startup is also cleaned up.
    list_files: list[Path] = []
    processes: list[subprocess.Popen[str]] = []
    handled_signals = (signal.SIGINT, signal.SIGTERM)
    previous_handlers = {signum: signal.getsignal(signum) for signum in handled_signals}
    installed_handlers = threading.current_thread() is threading.main_thread()
    pending_signal: int | None = None

    def cancel(signum: int, _frame: FrameType | None) -> None:
        nonlocal pending_signal
        pending_signal = signum if pending_signal is None else pending_signal
        raise SystemExit(128 + pending_signal)

    try:
        if installed_handlers:
            for signum in handled_signals:
                signal.signal(signum, cancel)

        for worker_i, chunk in enumerate(chunks):
            list_file = output_csv_dir / f'_pending_clips_{worker_i}.txt'
            list_file.write_text('\n'.join(str(path) for path in chunk))
            list_files.append(list_file)

            process_args = [
                python_exe, str(batch_script),
                '--video_list', str(list_file),
                '--tracknet_file', str(resolved_model),
                '--save_dir', str(output_csv_dir),
                '--batch_size', str(batch_size),
                '--eval_mode', _tracknet_eval_mode(tracknet_stride),
            ]
            if large_video:
                process_args.append('--large_video')
            process_args.extend(['--input_mode', input_mode])
            if input_mode == 'exact_ffv1_stream':
                process_args.extend([
                    '--ffmpeg', os.fspath(ffmpeg),
                    '--expected_frame_count', str(expected_frame_count),
                ])
            if input_video_identity is not None:
                process_args.extend([
                    '--input_video_identity',
                    os.fspath(input_video_identity),
                ])
            if resolved_inpaint:
                process_args.extend(['--inpaintnet_file', str(resolved_inpaint)])
            if dry_run:
                process_args.append('--dry_run')

            # stdout and stderr inherit the terminal. Piping either one can
            # deadlock if a worker fills the OS pipe while the parent waits.
            processes.append(
                subprocess.Popen(
                    process_args,
                    text=True,
                    env=_tracknet_subprocess_environment(),
                    start_new_session=True,
                )
            )

        print(f'Launched {len(processes)} batch worker(s)')
        for proc in processes:
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f'TrackNet worker exited with status {proc.returncode}')
    except BaseException:
        _terminate_process_groups(processes)
        if pending_signal is not None:
            raise SystemExit(128 + pending_signal) from None
        raise
    finally:
        if installed_handlers:
            for signum in handled_signals:
                signal.signal(signum, previous_handlers[signum])
        for f in list_files:
            f.unlink(missing_ok=True)

    # Count results from disk (authoritative, regardless of worker output)
    done = sum(1 for c in all_clips
               if (output_csv_dir / (c.stem + '_ball.csv')).exists())
    print(f'Extraction complete: {done}/{len(all_clips)} clips have CSVs')


def _terminate_process_groups(processes: Sequence[subprocess.Popen[str]]) -> None:
    """Terminate every owned TrackNet worker group and reap direct children."""
    for process in processes:
        _signal_process_group(process.pid, signal.SIGTERM)
    for process in processes:
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            _signal_process_group(process.pid, signal.SIGKILL)
            process.wait(timeout=2.0)
    deadline = time.monotonic() + 2.0
    for process in processes:
        while _process_group_exists(process.pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        if _process_group_exists(process.pid):
            _signal_process_group(process.pid, signal.SIGKILL)


def _tracknet_subprocess_environment() -> dict[str, str]:
    """Expose repository packages to the isolated TrackNet interpreter."""
    environment = os.environ.copy()
    required = [os.fspath(_SRC)]
    existing = environment.get('PYTHONPATH')
    if existing:
        required.append(existing)
    environment['PYTHONPATH'] = os.pathsep.join(required)
    return environment


def _signal_process_group(process_group: int, signum: signal.Signals) -> None:
    try:
        os.killpg(process_group, signum)
    except ProcessLookupError:
        pass


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ---------------------------------------------------------------------------
# CSV -> NPY conversion
# ---------------------------------------------------------------------------
def shuttle_csvs_to_npy(
    clips_dir: Path = CLIPS_OUTPUT_DIR,
    csv_dir: Path | None = None,
    npy_output_dir: Path = SHUTTLE_OUTPUT_DIR,
    resolution_csv_path: Path = RESOLUTION_CSV_PATH,
) -> None:
    """Convert TrackNetV3 CSV outputs to normalised .npy files, one per clip.

    Regenerates every npy on every run (no skip-existing): a re-extract that
    pops a fresh CSV then pops a fresh npy rather than leaving a stale one.

    Each saved npy holds resolution-NORMALISED coordinates -- x divided by the
    video width, y by the height, both in [0, 1] -- with the TrackNetV3
    Visibility flag passed through untouched. Raw pixel coordinates are always
    rederivable from the source CSV, so saving normalised loses nothing.

    Writes flat: each clip gets one .npy named after its stem. Split and
    class labels are carried by clips_master.csv at collation time, not by
    directory structure.

      clips/train/Top_smash/1_1_3_2.mp4  ->  shuttle_npy/1_1_3_2.npy

    :param clips_dir: Root clips directory (used to discover all clips).
    :param csv_dir: Directory containing TrackNetV3 CSV outputs.
        Defaults to clips_dir/../shuttle_csv.
    :param npy_output_dir: Output directory for normalised .npy files (flat).
    :param resolution_csv_path: Path to video resolution CSV (for normalisation).
    """
    if not csv_dir:
        csv_dir = _default_csv_dir(clips_dir)

    npy_output_dir.mkdir(parents=True, exist_ok=True)

    res_df = pd.read_csv(resolution_csv_path).set_index('id')
    converted = 0
    missing = 0

    for clip_path in sorted(clips_dir.rglob('*.mp4')):
        npy_path = npy_output_dir / (clip_path.stem + '.npy')

        # Find corresponding TrackNetV3 CSV
        csv_path = csv_dir / (clip_path.stem + '_ball.csv')
        if not csv_path.exists():
            # A clip with no CSV must not keep an old npy: collation reads
            # the npy as its source now, so a missing file fails loud there
            # instead of serving outdated coordinates.
            npy_path.unlink(missing_ok=True)
            missing += 1
            continue

        # Get video resolution for normalization
        vid_id = int(clip_path.stem.split('_')[0])
        if vid_id not in res_df.index:
            print(f'  WARNING: No resolution data for video {vid_id}')
            continue

        v_width = res_df.loc[vid_id, 'width']
        v_height = res_df.loc[vid_id, 'height']

        # Read TrackNetV3 CSV and normalize
        df = pd.read_csv(str(csv_path))
        expected_cols = {'Frame', 'X', 'Y', 'Visibility'}
        if not expected_cols.issubset(df.columns):
            print(f'  WARNING: Unexpected CSV format in {csv_path.name}, '
                  f'expected columns {expected_cols}, got {set(df.columns)}')
            continue

        df = df.drop_duplicates('Frame').set_index('Frame')
        # Keep Visibility column -- save as (t, 3): [x, y, visibility].
        # Consumers that only need xy can slice [:, :2].
        shuttle_camera = df[['X', 'Y', 'Visibility']].to_numpy().astype(float)
        shuttle_norm = normalize_shuttlecock(shuttle_camera, v_width, v_height)

        np.save(str(npy_path), shuttle_norm)
        converted += 1

    print(f'Shuttle NPY conversion: {converted} files written, {missing} missing CSVs')


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Extract shuttle trajectories from ShuttleSet clips using TrackNetV3.',
    )
    parser.add_argument('--tracknet-dir', type=Path, default=DEFAULT_TRACKNET_DIR,
                        help='TrackNetV3 directory (default: src/shared/tracknetv3)')
    parser.add_argument('--clips-dir', type=Path, default=CLIPS_OUTPUT_DIR,
                        help='Directory containing generated clips')
    parser.add_argument('--csv-dir', type=Path, default=None,
                        help='Directory for TrackNetV3 CSV outputs (default: clips_dir/../shuttle_csv)')
    parser.add_argument('--npy-dir', type=Path, default=SHUTTLE_OUTPUT_DIR,
                        help='Output directory for normalized .npy files')
    parser.add_argument('--resolution-csv', type=Path, default=RESOLUTION_CSV_PATH,
                        help='Path to video resolution CSV')
    parser.add_argument('--model-path', type=Path, default=None,
                        help='Path to TrackNet weights (default: tracknet-dir/ckpts/TrackNet_best.pt)')
    parser.add_argument('--inpaintnet-path', type=Path, default=None,
                        help='Path to InpaintNet weights (default: tracknet-dir/ckpts/InpaintNet_best.pt)')
    parser.add_argument('--workers', type=int, default=2,
                        help='Parallel workers for TrackNetV3 (default 2, GPU-bound)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for TrackNet DataLoader (default 32, use 64 with --workers 1)')
    parser.add_argument('--tracknet-python', type=Path, default=None,
                        help='Python executable in BST venv (shared with TrackNetV3)')
    parser.add_argument('--tracknet-stride', choices=(1, 8), type=int, default=None)
    parser.add_argument('--large-video', action=argparse.BooleanOptionalAction, default=None,
                        help='Streaming TrackNet mode for long videos (default: profile setting)')
    parser.add_argument('--profile', choices=('bst', 'scrape'), default='bst',
                        help='Lane defaults for stride and large-video mode (default bst)')
    parser.add_argument('--skip-extraction', action='store_true',
                        help='Skip TrackNetV3 extraction, only convert existing CSVs to NPY')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run inference without writing output files (test that pipeline works)')
    args = parser.parse_args()

    default_stride, default_large_video = PROFILE_DEFAULTS[args.profile]
    resolved_stride = default_stride if args.tracknet_stride is None else args.tracknet_stride
    resolved_large_video = default_large_video if args.large_video is None else args.large_video

    if not args.skip_extraction:
        print('=== Extracting shuttle trajectories ===')
        extract_all_shuttles(
            clips_dir=args.clips_dir,
            tracknet_dir=args.tracknet_dir,
            output_csv_dir=args.csv_dir,
            model_path=args.model_path,
            inpaintnet_path=args.inpaintnet_path,
            tracknet_python=args.tracknet_python,
            max_workers=args.workers,
            batch_size=args.batch_size,
            tracknet_stride=resolved_stride,
            large_video=resolved_large_video,
            dry_run=args.dry_run,
        )

    if not args.dry_run:
        print('\n=== Converting shuttle CSVs to NPY ===')
        shuttle_csvs_to_npy(
            clips_dir=args.clips_dir,
            csv_dir=args.csv_dir,
            npy_output_dir=args.npy_dir,
            resolution_csv_path=args.resolution_csv,
        )


if __name__ == '__main__':
    main()
