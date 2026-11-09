"""Find camera-cut frame numbers in a video."""
from pathlib import Path

import numpy as np

COMPOSITION_CONTENT_THRESHOLD = 27.0  # PySceneDetect ContentDetector default
MIN_SCENE_FRAMES_AT_30_FPS = 15


def detect_cuts(video_path: Path, expected_frames: int, threshold: float, min_scene_len: int) -> np.ndarray:
    """Detect camera cuts and return the first frame of each new scene.

    Check both the container frame count and the count actually read against
    expected_frames. A resized video must keep the original frames so cuts
    line up with shuttle positions and court/player measurements.

    :param video_path: Video to inspect, typically a smaller resized copy.
    :param expected_frames: Frame count shared by the source and vision arrays.
    :param threshold: Content-change threshold; lower means more cuts.
    :param min_scene_len: Minimum scene length in video frames.
    :return: Integer cut frames in ascending order, zero-based.
    """
    # Load the optional scene detector only when cut detection is requested.
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
    # The opening scene is not a cut; each later scene starts at one.
    cut_frames = [scene[0].frame_num for scene in scene_list[1:]]
    return np.array(cut_frames, dtype=int)
