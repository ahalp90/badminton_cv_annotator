"""Composition dead-mask: broadcast-cut segments filtered by a court-view vote.

A companion producer to replay_mask: a second way to build the same
`(frames,)` bool dead-time mask that `segment_video(exclusion_mask=...)` accepts
(True = dead). Where the replay mask unions three per-frame signals, this one
works per SEGMENT of the broadcast.

The idea: a broadcast-cut detector (PySceneDetect's content detector, run on a
288p scale-only downsample) gives crisp segment boundaries; the homography
court-view vote then labels each whole segment live or dead by a single fraction.
So the smoothing comes free: one vote per segment, no per-frame flicker to clean
up. A segment is LIVE when at least `vote` of its frames read as court view, else
DEAD, and every frame of a dead segment joins the mask.

The scoping pass picked content threshold 27 with vote 0.5 (comp_content27_v0p5)
as the best config on sset_01; those are the config defaults.

The cut detector needs scenedetect (and its cv2 backend), which lives only in the
dedicated detect venv. Its import is function-local so the model/test venv, which
carries neither, can still import this module (matching transcript acquisition's whisperx
fallback). The court-view vote is a precomputed input here, the same way replay masking
takes its court and homography inputs precomputed.

Run as `python -m experiments.annotator.scene_cut_mask --video-id ... --video <288p>
--keep-vote <court_view.npy>` with PYTHONPATH=src, in the detect venv.
"""

import argparse
import itertools
import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np

from scraper.config import MASKS_DIR
from shared.camera_cuts import (
    COMPOSITION_CONTENT_THRESHOLD,
    MIN_SCENE_FRAMES_AT_30_FPS,
    detect_cuts,
)
from shared.frame_timing import frames_at_fps
from shared.video_metadata import probe_fps

COMPOSITION_KEEP_VOTE = 0.5

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cut detection
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Composition mask build
# ---------------------------------------------------------------------------
class CompositionSegment(NamedTuple):
    """One cut-to-cut segment [start, end) with its court-view verdict."""

    start: int
    end: int
    keep_fraction: float
    is_dead: bool


def build_composition_mask(
    cut_frames: np.ndarray, keep_vote: np.ndarray, n_frames: int, vote: float,
) -> tuple[np.ndarray, list[CompositionSegment]]:
    """Cut-segment the timeline; label each segment live/dead by the court-view vote.

    Boundaries are the cut frames plus 0 and n_frames (np.unique folds any
    coincidence and sorts). Segment [start, end) is LIVE when its court-view keep
    fraction is at or above `vote`, else DEAD; the mask is True on every frame of a
    dead segment. Equality is deliberately live (`is_dead = fraction < vote`): a
    segment exactly on the threshold gets the benefit of the doubt and stays in.

    Fail loud on an all-dead result: no live segment means nothing for the
    segmenter to anchor to (apply_replay_mask rejects an all-True mask anyway), so
    a fully-dead mask is almost always a bad vote input rather than a real answer.

    :param cut_frames: cut frame indices, each the first frame of a new segment (order irrelevant).
    :param keep_vote: `(n_frames,)` bool homography court-view vote, True = court view (live-like).
    :param n_frames: video frame count (the mask length).
    :param vote: live threshold on a segment's court-view keep fraction.
    :return: (`(n_frames,)` bool mask True = dead, one `CompositionSegment` per segment).
    """
    boundaries = np.unique(np.concatenate([[0], cut_frames.astype(int), [n_frames]]))
    mask = np.zeros(n_frames, dtype=bool)
    segments: list[CompositionSegment] = []
    for start, end in itertools.pairwise(boundaries):
        start, end = int(start), int(end)
        keep_fraction = float(keep_vote[start:end].mean())
        is_dead = keep_fraction < vote
        if is_dead:
            mask[start:end] = True
        segments.append(CompositionSegment(start, end, keep_fraction, is_dead))
    if mask.all():
        raise ValueError(
            f'composition mask (vote={vote}) is all dead: no segment cleared the court-view vote, '
            f'so there is nothing live to anchor the segmenter to'
        )
    return mask, segments


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description='Composition dead-mask: cut-segment the timeline and keep court-view segments.'
    )
    parser.add_argument('--video-id', required=True)
    parser.add_argument('--video', type=Path, required=True,
                        help='288p scale-only downsample for the cut detector (frame count preserved)')
    parser.add_argument('--keep-vote', type=Path, required=True,
                        help='<video_id> homography court-view vote (frames,) bool npy, True = court view')
    parser.add_argument('--content-threshold', type=float, default=COMPOSITION_CONTENT_THRESHOLD)
    parser.add_argument('--vote', type=float, default=COMPOSITION_KEEP_VOTE)
    parser.add_argument('--fps', type=float, default=None)
    parser.add_argument('--out-dir', type=Path, default=MASKS_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    keep_vote = np.load(args.keep_vote)
    if keep_vote.dtype != bool:
        raise ValueError(f'keep-vote must be bool (True = court view), got {keep_vote.dtype}')
    n_frames = len(keep_vote)

    fps = args.fps if args.fps is not None else probe_fps(args.video)
    min_scene_len = frames_at_fps(MIN_SCENE_FRAMES_AT_30_FPS, fps)
    cut_frames = detect_cuts(args.video, n_frames, args.content_threshold, min_scene_len)
    mask, segments = build_composition_mask(cut_frames, keep_vote, n_frames, args.vote)
    n_live = sum(1 for seg in segments if not seg.is_dead)

    # Feeds rally segmentation's existing dead-mask slot (segment_video reads <video_id>_dead_mask.npy).
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f'{args.video_id}_dead_mask.npy'
    np.save(out_path, mask)
    log.info('%s: %d cuts, %d/%d segments live, %d/%d frames dead -> %s',
             args.video_id, len(cut_frames), n_live, len(segments),
             int(mask.sum()), n_frames, out_path)


if __name__ == '__main__':
    main()
