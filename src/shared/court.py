"""Convert image positions to court coordinates for annotation and classifiers.

``project_pixels_to_court`` scales video pixels to the 1280x720 reference,
applies the homography and normalises against the supplied court bounds.
``get_court_info`` and the CSV loaders read saved ShuttleSet homographies.

The older REF_COURT_M rectangle uses (length, width). Court detection uses
(width, length) from ``shared.court_model``. Keep the coordinate convention
matched to the homography being used.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Reference image size for the saved homographies. Scale source-video pixels
# to this size before applying H.
HOMOGRAPHY_RESOLUTION: tuple[int, int] = (1280, 720)

# Doubles-court rectangle in metres, using the older (length, width) axes.
# Corners follow top-left, top-right, bottom-right, bottom-left order.
REF_COURT_M: tuple[float, float] = (13.4, 6.1)
REF_COURT_CORNERS_M = np.array(
    [[0.0,            0.0],
     [REF_COURT_M[0], 0.0],
     [REF_COURT_M[0], REF_COURT_M[1]],
     [0.0,            REF_COURT_M[1]]],
    dtype=np.float32,
)


def project_pixels_to_court(
    px_xy: np.ndarray, resolution: tuple[float, float], court_info: dict,
) -> np.ndarray:
    """Project video-pixel xy onto the court, then normalise by the court bounds.

    Scale to the homography's 1280x720 reference before applying its matrix.
    For the annotator's court bounds, the doubles outline becomes a unit square.

    :param px_xy: (2, points) pixel coordinates at the supplied video size.
    :param resolution: Video width and height.
    :param court_info: H matrix and border_L/R/U/D values.
    :return: (2, points) normalised court xy, possibly outside [0, 1].
    """
    scaled = scale_pos_by_resolution(px_xy, width=resolution[0], height=resolution[1])
    court = project(court_info['H'], convert_homogeneous(scaled))
    return normalize_position(court, court_info)


def normalize_position(arr: np.ndarray, court_info: dict) -> np.ndarray:
    """Map court xy to a unit square using the supplied left/right/top/bottom bounds.

    Input and output: (2, points). Allow values outside [0, 1] for off-court positions.
    """
    x_dist = court_info['border_R'] - court_info['border_L']
    y_dist = court_info['border_D'] - court_info['border_U']
    x_normalized = (arr[0, :] - court_info['border_L']) / x_dist
    y_normalized = (arr[1, :] - court_info['border_U']) / y_dist
    return np.stack((x_normalized, y_normalized))


def scale_pos_by_resolution(
    arr: np.ndarray, width: float, height: float,
) -> np.ndarray:
    """Scale xy rows from video pixels to 1280x720 reference pixels; keep any third row."""
    aim_w, aim_h = HOMOGRAPHY_RESOLUTION
    new_arr = arr.copy()
    new_arr[0, :] *= aim_w / width
    new_arr[1, :] *= aim_h / height
    return new_arr


def convert_homogeneous(arr: np.ndarray) -> np.ndarray:
    """Append a row of ones to (2, points) xy so a 3x3 homography can transform it."""
    return np.concatenate((arr, np.full((1, arr.shape[-1]), 1.0)), axis=0)


def project(H: np.ndarray, P_prime: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to (3, points) coordinates and return (2, points) xy."""
    P = H @ P_prime
    P = P[:2, :] / P[-1, :]
    return P


# Saved ShuttleSet court geometry

def build_all_court_info(set_info_dir: Path, res_df: pd.DataFrame) -> dict:
    """Read homography.csv and build court geometry for each video in res_df.

    Use the resolution table's video IDs. A missing homography row raises
    KeyError, so videos without court geometry are not silently dropped.
    """
    homo_df = pd.read_csv(set_info_dir / 'homography.csv').set_index('id')
    return {vid: get_court_info(homo_df, vid) for vid in res_df.index}


def load_all_court_info(homo_csv_path: Path) -> dict:
    """Load court info for all videos from BST's homography.csv."""
    homo_df = pd.read_csv(homo_csv_path).set_index('id')
    return {vid: get_court_info(homo_df, vid) for vid in homo_df.index}


def get_court_info(homo_df: pd.DataFrame, vid: int) -> dict:
    """Get homography matrix and court boundary coordinates for a video."""
    homography_info = homo_df.loc[vid]
    H = get_H(homography_info)
    corner_camera = get_corner_camera(homography_info)
    corner_camera = convert_homogeneous(corner_camera)
    corner_court = project(H, corner_camera)
    # CSV corner columns arrive (upleft, upright, downleft, downright), so R
    # reads from column 1 (upright) and D from column 2 (downleft).
    return {
        'H': H,
        'border_L': corner_court[0, 0],
        'border_R': corner_court[0, 1],
        'border_U': corner_court[1, 0],
        'border_D': corner_court[1, 2],
    }


def get_H(homography_info: pd.Series) -> np.ndarray:
    """Parse the 3x3 homography matrix from a homography.csv row."""
    h_str: str = homography_info['homography_matrix']
    clean_str = h_str.replace('[', '').replace(']', '').replace(',', ' ')
    return np.fromstring(clean_str, sep=' ').reshape((3, 3))


def get_corner_camera(homography_info: pd.Series) -> np.ndarray:
    """Read the four saved image corners as a (2, 4) array from a homography.csv row."""
    corner_camera = homography_info.loc['upleft_x':'downright_y']
    return corner_camera.to_numpy(dtype=float).reshape((2, 4))


def to_court_coordinate(
    arr_camera: np.ndarray,
    vid: int,
    all_court_info: dict,
    res_df: pd.DataFrame,
) -> np.ndarray:
    """Transform camera pixel coordinates (2, N) to court coordinates (2, N)."""
    res_info = res_df.loc[vid]
    H = all_court_info[vid]['H']
    arr_camera = scale_pos_by_resolution(arr_camera, width=res_info['width'], height=res_info['height'])
    arr_camera = convert_homogeneous(arr_camera)
    return project(H, arr_camera)


def check_pos_in_court(
    keypoints: np.ndarray,
    vid: int,
    all_court_info: dict,
    res_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Project each person's mean ankle position and check whether it is on court.

    :param keypoints: (people, joints, 2) camera pixels, with COCO ankle joints
        in the last two positions.
    :return: Boolean mask (people,) and normalised court xy (people, 2).
        Allow 1% beyond the court boundary on each axis.
    """
    n_people = keypoints.shape[0]

    # Estimate each person's court position from the midpoint between their ankles.
    feet_camera = keypoints[:, -2:, :].reshape(-1, 2).T  # (2, n_people*2)
    feet_court = to_court_coordinate(feet_camera, vid, all_court_info, res_df)
    feet_court = feet_court.reshape(2, n_people, -1)  # (2, n_people, 2)

    pos_court = feet_court.mean(axis=-1)  # midpoint between feet, (2, n_people)
    pos_court_normalized = normalize_position(pos_court, court_info=all_court_info[vid]).T  # (m, 2)

    eps = 0.01  # soft border tolerance
    dim_in_court = (pos_court_normalized > -eps) & (pos_court_normalized < (1 + eps))
    in_court = dim_in_court[:, 0] & dim_in_court[:, 1]
    return in_court, pos_court_normalized
