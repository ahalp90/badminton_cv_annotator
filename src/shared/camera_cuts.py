"""Find camera-cut frame numbers in a video."""
from pathlib import Path

import numpy as np


def detect_cuts(video_path: Path, expected_frames: int, threshold: float, min_scene_len: int) -> np.ndarray:
    """Broadcast-cut frames from a PySceneDetect content pass over the downsample.

    Fail loud unless the container-declared duration and the frames actually read
    both equal `expected_frames` (the source track length). A scale-only 288p
    downsample must preserve the frame count for the cut frames to index the same
    timeline as the shuttle track and the court-view vote.

    :param video_path: the 288p scale-only downsample to run the detector over.
    :param expected_frames: the source frame count (track / vote length) to assert against.
    :param threshold: PySceneDetect ContentDetector threshold (lower = more cuts).
    :return: `(n_cuts,)` int cut frame indices, ascending; each the first frame of a new scene.
    """
    # scenedetect and its cv2 backend live only in the dedicated detect venv; the
    # model/test venv carries neither. A module-level import would break every
    # importer that never runs cut detection (the whole test suite included), so
    # it is function-local, matching transcript acquisition's whisperx fallback.
    from scenedetect import ContentDetector, SceneManager, open_video

    video = open_video(str(video_path))
    if video.duration.frame_num != expected_frames:
        raise ValueError(
            f'{video_path.name}: container duration {video.duration.frame_num} frames '
            f'!= expected {expected_frames} (the downsample must preserve the source frame count)'
        )
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
    n_read = manager.detect_scenes(video, show_progress=False)
    if n_read != expected_frames:
        raise ValueError(
            f'{video_path.name}: detect_scenes read {n_read} frames != expected {expected_frames}'
        )
    scene_list = manager.get_scene_list()
    # A cut is the boundary between scene i-1 and scene i, recorded at scene i's
    # first frame (0-based). scene_list[0] is the opening scene, so cuts start at 1.
    cut_frames = [scene[0].frame_num for scene in scene_list[1:]]
    return np.array(cut_frames, dtype=int)
