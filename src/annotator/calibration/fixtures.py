"""Digest-validated input fixtures for annotator calibration."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from shared.court import REF_COURT_M, convert_homogeneous, get_H, get_corner_camera, project

RootKind = Literal["fixtures", "repo"]


@dataclass(frozen=True)
class FilePin:
    """A relative file path and the bytes it is expected to contain."""

    path: Path
    md5: str
    root: RootKind


@dataclass(frozen=True)
class CalibrationGeometry:
    """Derived camera geometry and the tracked source resolution for one fixture."""

    court_box: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]
    net_band: tuple[float, float]
    resolution: tuple[float, float]


@dataclass(frozen=True)
class Fixture:
    """All external and repository-local inputs for one scoring fixture.

    The ``court_present_path`` field is a pose-derived court-view proxy (True =
    court view); the scene-gated tracker's producer choice is re-approved at
    its activation commit.
    """

    name: str
    video_id: int
    fps: float
    track_path: Path
    pose_dir: Path
    pose_prefix: str
    mask_path: Path
    court_present_path: Path
    scene_rows_path: Path
    gt_set_dir: Path
    court_box: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]
    net_band: tuple[float, float]
    resolution: tuple[float, float]
    n_rallies: int
    n_strokes: int
    files: tuple[FilePin, ...]


def fixtures_root() -> Path:
    """Return the configured external fixture root."""
    value = os.environ.get("ANNOTATOR_FIXTURES_ROOT")
    if not value:
        raise RuntimeError(
            "ANNOTATOR_FIXTURES_ROOT is unset; external annotator fixtures are unavailable"
        )
    return Path(value).expanduser().resolve()


REPO_ROOT = Path(__file__).resolve().parents[3]

_HOMOGRAPHY_SOURCE = REPO_ROOT / "training/data/shuttleset/annotations/set/homography.csv"
_RESOLUTION_SOURCE = REPO_ROOT / "training/data/shuttleset/annotations/my_raw_video_resolution.csv"
# Homography camera coordinates are on 1280x720; fixture coordinates are 1920x1080.
# Worked arithmetic: local_scratch/autograder_architecture/records/s6_b7a_mapping_report.txt.
_HOMOGRAPHY_TO_FIXTURE_MULTIPLIER = 1.5
# The centre band spans one metre along the 13.4 m court-length axis.
_NET_BAND_HALF_WIDTH_M = 0.5
_COMPATIBILITY_HEIGHT = (84.0, 336.0)
_CALIBRATION_VIDEO_IDS = (1, 15, 21)
_CORNER_COLUMNS = (
    "upleft_x", "upright_x", "downleft_x", "downright_x",
    "upleft_y", "upright_y", "downleft_y", "downright_y",
)


def _read_source_frame(path: Path, source_name: str) -> pd.DataFrame:
    """Read and validate the id column in one tracked calibration source."""
    if not path.is_file():
        raise ValueError(f"{source_name} source file missing: {path}")

    frame = pd.read_csv(path)
    if "id" not in frame.columns:
        raise ValueError(f"{source_name} source is missing the id column: {path}")

    numeric_ids = pd.to_numeric(frame["id"], errors="coerce")
    if not np.isfinite(numeric_ids.to_numpy()).all():
        raise ValueError(f"{source_name} source has malformed id values: {path}")
    if not (numeric_ids == numeric_ids.astype(int)).all():
        raise ValueError(f"{source_name} source has non-integer id values: {path}")

    frame = frame.copy()
    frame["id"] = numeric_ids.astype(int)
    duplicate_ids = frame.loc[frame["id"].duplicated(keep=False), "id"].unique()
    if duplicate_ids.size:
        duplicate_text = ", ".join(str(int(video_id)) for video_id in duplicate_ids)
        raise ValueError(f"{source_name} source has duplicate rows for id {duplicate_text}: {path}")
    return frame


def _source_row(frame: pd.DataFrame, video_id: int, source_name: str, path: Path) -> pd.Series:
    """Return exactly one source row for a requested video id."""
    matching_rows = frame.loc[frame["id"] == video_id]
    if matching_rows.empty:
        raise ValueError(f"{source_name} source row missing for id {video_id}: {path}")
    return matching_rows.iloc[0]


def _finite_float(value: object, field: str, video_id: int, source_name: str) -> float:
    """Parse one finite source value with an error that names its input."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source_name} value {field!r} is malformed for id {video_id}") from error
    if not np.isfinite(parsed):
        raise ValueError(f"{source_name} value {field!r} is not finite for id {video_id}")
    return parsed


def _derive_calibration_geometry(
    homography_row: pd.Series,
    resolution_row: pd.Series,
    video_id: int,
) -> CalibrationGeometry:
    """Derive one fixture's camera bounds and net band from its tracked rows."""
    for column in _CORNER_COLUMNS:
        _finite_float(homography_row[column], column, video_id, "homography")

    try:
        homography = get_H(homography_row)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"homography matrix is malformed for id {video_id}") from error
    if not np.isfinite(homography).all():
        raise ValueError(f"homography matrix is malformed for id {video_id}")

    corner_camera = get_corner_camera(homography_row)
    corner_camera_fixture = corner_camera * _HOMOGRAPHY_TO_FIXTURE_MULTIPLIER
    x_bounds = _rounded_bounds(corner_camera_fixture[0])
    y_bounds = _rounded_bounds(corner_camera_fixture[1])

    try:
        corner_court = project(homography, convert_homogeneous(corner_camera))
        inverse_homography = np.linalg.inv(homography)
    except (ValueError, np.linalg.LinAlgError) as error:
        raise ValueError(f"homography matrix cannot project id {video_id}") from error
    if not np.isfinite(corner_court).all():
        raise ValueError(f"homography matrix cannot project id {video_id}")

    court_x_min, court_x_max = float(corner_court[0].min()), float(corner_court[0].max())
    court_y_min, court_y_max = float(corner_court[1].min()), float(corner_court[1].max())
    x_centre = (court_x_min + court_x_max) / 2.0
    y_centre = (court_y_min + court_y_max) / 2.0
    court_units_per_metre = (court_y_max - court_y_min) / REF_COURT_M[0]
    net_half_band = court_units_per_metre * _NET_BAND_HALF_WIDTH_M
    court_band_points = np.array(
        [[x_centre, x_centre], [y_centre - net_half_band, y_centre + net_half_band]],
        dtype=float,
    )
    camera_band = project(inverse_homography, convert_homogeneous(court_band_points))
    if not np.isfinite(camera_band).all():
        raise ValueError(f"homography matrix cannot project net band for id {video_id}")
    net_band = _rounded_bounds(camera_band[1] * _HOMOGRAPHY_TO_FIXTURE_MULTIPLIER)

    width = _finite_float(resolution_row["width"], "width", video_id, "resolution")
    height = _finite_float(resolution_row["height"], "height", video_id, "resolution")
    if width <= 0 or height <= 0:
        raise ValueError(f"resolution values must be positive for id {video_id}")

    resolution = (width, height)
    court_box = (x_bounds, y_bounds, _COMPATIBILITY_HEIGHT, net_band)
    return CalibrationGeometry(court_box=court_box, net_band=net_band, resolution=resolution)


def _rounded_bounds(values: np.ndarray) -> tuple[float, float]:
    """Return the minimum and maximum values rounded to one decimal place."""
    return round(float(values.min()), 1), round(float(values.max()), 1)


def _load_calibration_geometry(
    homography_path: Path = _HOMOGRAPHY_SOURCE,
    resolution_path: Path = _RESOLUTION_SOURCE,
) -> dict[int, CalibrationGeometry]:
    """Load and derive calibration geometry for every Wave 1 fixture."""
    homography_frame = _read_source_frame(homography_path, "homography")
    resolution_frame = _read_source_frame(resolution_path, "resolution")
    required_homography_columns = {"homography_matrix", *_CORNER_COLUMNS}
    missing_homography_columns = required_homography_columns.difference(homography_frame.columns)
    if missing_homography_columns:
        missing = ", ".join(sorted(missing_homography_columns))
        raise ValueError(f"homography source is missing columns: {missing}")
    required_resolution_columns = {"width", "height"}
    missing_resolution_columns = required_resolution_columns.difference(resolution_frame.columns)
    if missing_resolution_columns:
        missing = ", ".join(sorted(missing_resolution_columns))
        raise ValueError(f"resolution source is missing columns: {missing}")

    geometry: dict[int, CalibrationGeometry] = {}
    for video_id in _CALIBRATION_VIDEO_IDS:
        homography_row = _source_row(homography_frame, video_id, "homography", homography_path)
        resolution_row = _source_row(resolution_frame, video_id, "resolution", resolution_path)
        geometry[video_id] = _derive_calibration_geometry(homography_row, resolution_row, video_id)
    return geometry


_CALIBRATION_GEOMETRY = _load_calibration_geometry()


def _pose_files(directory: Path, prefix: str, *, kp_scores: bool = False) -> tuple[Path, ...]:
    names = ("bboxes", "scores", "kps")
    if kp_scores:
        names += ("kp_scores",)
    names += ("ndet",)
    return tuple(directory / f"{prefix}_{name}.npy" for name in names)


def _fixture_files(
    track_path: Path,
    pose_dir: Path,
    pose_prefix: str,
    mask_path: Path,
    court_present_path: Path,
    scene_rows_path: Path,
    md5s: dict[Path, str],
    *,
    kp_scores: bool = False,
) -> tuple[FilePin, ...]:
    paths = (
        track_path,
        *_pose_files(pose_dir, pose_prefix, kp_scores=kp_scores),
        mask_path,
        court_present_path,
        scene_rows_path,
    )
    return tuple(FilePin(path, md5s[path], "fixtures") for path in paths)


# Digests marked "ported" are present in INPUT_MD5S or
# local_scratch/stage1_reference/sset21_substrate_md5s.txt. The remaining digests were newly
# computed from the live fixture root on 2026-07-18: pilot/vid15 tracks and pose arrays, the
# sset21 R dead mask, and all repository-local GT CSVs.
_PILOT_ROOT = Path("pilot_results")
_PILOT_TRACK = Path("pilot_track_npy/1.npy")
_PILOT_POSE = Path("pilot_pose_raw")
_PILOT_PREFIX = "pilot_1080p_raw"
_VID15_TRACK = Path("vid15_track_npy/15.npy")
_VID15_POSE = Path("vid15_pose_raw")
_VID15_PREFIX = "vid15_1080p_raw"
_SSET_TRACK = Path("sset21_track_npy/21.npy")
_SSET_POSE = Path("sset21_pose_raw")
_SSET_PREFIX = "sset_21_gloiZ_gTJaE_raw"

PILOT = Fixture(
    name="pilot",
    video_id=1,
    fps=25.0,
    track_path=_PILOT_TRACK,
    pose_dir=_PILOT_POSE,
    pose_prefix=_PILOT_PREFIX,
    mask_path=Path("pilot_results/composition_mask/comp_content27_v0p5.npy"),
    court_present_path=Path("pilot_results/homography_smoothing/raw_keep_hard_any_m0p10.npy"),
    scene_rows_path=Path("pilot_results/scene_rows_content27_refcorners.csv"),
    gt_set_dir=Path("training/data/shuttleset/annotations/set/Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2019_Finals"),
    court_box=_CALIBRATION_GEOMETRY[1].court_box,
    net_band=_CALIBRATION_GEOMETRY[1].net_band,
    resolution=_CALIBRATION_GEOMETRY[1].resolution,
    n_rallies=113,
    n_strokes=1641,
    files=_fixture_files(
        _PILOT_TRACK, _PILOT_POSE, _PILOT_PREFIX,
        Path("pilot_results/composition_mask/comp_content27_v0p5.npy"),
        Path("pilot_results/homography_smoothing/raw_keep_hard_any_m0p10.npy"),
        Path("pilot_results/scene_rows_content27_refcorners.csv"),
        {
            _PILOT_TRACK: "08c5afced66b561517a43571df567b2f",
            Path("pilot_pose_raw/pilot_1080p_raw_bboxes.npy"): "4c9525949d1c79f0161f81b2bb63d5ef",
            Path("pilot_pose_raw/pilot_1080p_raw_scores.npy"): "03e655b3429f9482c5a3f4df766a3534",
            Path("pilot_pose_raw/pilot_1080p_raw_kps.npy"): "621427713fc617d81d4081db15613b06",
            Path("pilot_pose_raw/pilot_1080p_raw_ndet.npy"): "5cc366f2cd459ea9be44876bc07e74ea",
            Path("pilot_results/composition_mask/comp_content27_v0p5.npy"): "a5043d329752a4e202c8566515b37231",
            Path("pilot_results/homography_smoothing/raw_keep_hard_any_m0p10.npy"): "095f6ee3a3a3042c06f42e6e4467e88d",
            Path("pilot_results/scene_rows_content27_refcorners.csv"): "378cfeb29a44e90ef9f9694344cca649",
        },
    ),
)

VID15 = Fixture(
    name="vid15",
    video_id=15,
    fps=25.0,
    track_path=_VID15_TRACK,
    pose_dir=_VID15_POSE,
    pose_prefix=_VID15_PREFIX,
    mask_path=Path("vid15_results/composition_mask/comp_content27_v0p7.npy"),
    court_present_path=Path("vid15_results/composition_mask/vid15_keep_hard_any_m0p10.npy"),
    scene_rows_path=Path("vid15_results/scene_rows_content27_refcorners.csv"),
    gt_set_dir=Path("training/data/shuttleset/annotations/set/Anthony_Sinisuka_GINTING_Anders_ANTONSEN_Indonesia_Masters_2020_Final"),
    court_box=_CALIBRATION_GEOMETRY[15].court_box,
    net_band=_CALIBRATION_GEOMETRY[15].net_band,
    resolution=_CALIBRATION_GEOMETRY[15].resolution,
    n_rallies=104,
    n_strokes=824,
    files=_fixture_files(
        _VID15_TRACK, _VID15_POSE, _VID15_PREFIX,
        Path("vid15_results/composition_mask/comp_content27_v0p7.npy"),
        Path("vid15_results/composition_mask/vid15_keep_hard_any_m0p10.npy"),
        Path("vid15_results/scene_rows_content27_refcorners.csv"),
        {
            _VID15_TRACK: "0b9c0966ffc58a36c65f97a5a9a78deb",
            Path("vid15_pose_raw/vid15_1080p_raw_bboxes.npy"): "031d4f61f71f7e3f2e18a0af5e52b138",
            Path("vid15_pose_raw/vid15_1080p_raw_scores.npy"): "5c3c7895312abbd28045968426fc21c4",
            Path("vid15_pose_raw/vid15_1080p_raw_kps.npy"): "1d74ceef0fdd53dab60e3afd64e4a6fc",
            Path("vid15_pose_raw/vid15_1080p_raw_ndet.npy"): "71f7f8a9e7f270fc0ffea868da437e08",
            Path("vid15_results/composition_mask/comp_content27_v0p7.npy"): "c01914b9788afef3bca6e0b5bd88dc7f",
            Path("vid15_results/composition_mask/vid15_keep_hard_any_m0p10.npy"): "8268eeed2c48914d165c31899ce9417b",
            Path("vid15_results/scene_rows_content27_refcorners.csv"): "a893afaf12920658338586e4b9b0d6d6",
        },
    ),
)

SSET21 = Fixture(
    name="sset21",
    video_id=21,
    fps=30.0,
    track_path=_SSET_TRACK,
    pose_dir=_SSET_POSE,
    pose_prefix=_SSET_PREFIX,
    mask_path=Path("sset21_results/R/21_dead_mask.npy"),
    court_present_path=Path("sset21_results/keep_vote_hard_any_m0p10.npy"),
    scene_rows_path=Path("sset21_results/scene_rows_content27_refcorners.csv"),
    gt_set_dir=Path("training/data/shuttleset/annotations/set/An_Se_Young_Ratchanok_Intanon_YONEX_Thailand_Open_2021_QuarterFinals"),
    court_box=_CALIBRATION_GEOMETRY[21].court_box,
    net_band=_CALIBRATION_GEOMETRY[21].net_band,
    resolution=_CALIBRATION_GEOMETRY[21].resolution,
    n_rallies=75,
    n_strokes=663,
    files=_fixture_files(
        _SSET_TRACK, _SSET_POSE, _SSET_PREFIX, Path("sset21_results/R/21_dead_mask.npy"),
        Path("sset21_results/keep_vote_hard_any_m0p10.npy"),
        Path("sset21_results/scene_rows_content27_refcorners.csv"),
        {
            _SSET_TRACK: "ad00846dc78b08de728cf59ea773ad61",
            Path("sset21_pose_raw/sset_21_gloiZ_gTJaE_raw_bboxes.npy"): "3ee48b9637a49157ed494cbc0fbfab9a",
            Path("sset21_pose_raw/sset_21_gloiZ_gTJaE_raw_scores.npy"): "86ba65b4e902067853a51308db864a69",
            Path("sset21_pose_raw/sset_21_gloiZ_gTJaE_raw_kps.npy"): "6f5b60e0b2ae04ead4a3523aad744fa4",
            Path("sset21_pose_raw/sset_21_gloiZ_gTJaE_raw_kp_scores.npy"): "014561d30e74bd6811933d68dfd19525",
            Path("sset21_pose_raw/sset_21_gloiZ_gTJaE_raw_ndet.npy"): "1844e00ffd6cddfa1dd52e26442fef14",
            Path("sset21_results/R/21_dead_mask.npy"): "9a6b43bc14f795d8c5e4d62e86005798",
            Path("sset21_results/keep_vote_hard_any_m0p10.npy"): "93f5cbea19f8b7e65e272df9a5d0b252",
            Path("sset21_results/scene_rows_content27_refcorners.csv"): "f9fb06285637076c5817301ae7a7b41b",
        },
        kp_scores=True,
    ),
)

FIXTURES = (PILOT, VID15, SSET21)

SHARED_FILES = (
    FilePin(Path("training/data/shuttleset/annotations/shots_master.csv"), "39cdc201057050abfe4c6f8770734fde", "repo"),
    FilePin(Path("training/data/shuttleset/annotations/set/homography.csv"), "07de7edf7951f4f5ca2d76d9f5490600", "repo"),
    FilePin(Path("training/data/shuttleset/annotations/my_raw_video_resolution.csv"), "d252694e01497e43aedcdd01c6dce251", "repo"),
    FilePin(Path("training/data/shuttleset/annotations/set/Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2019_Finals/set1.csv"), "cd627c256043d128b4eeb05895b3e8d7", "repo"),
    FilePin(Path("training/data/shuttleset/annotations/set/Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2019_Finals/set2.csv"), "c91b420295ec6366960c52a5985f07d7", "repo"),
    FilePin(Path("training/data/shuttleset/annotations/set/Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2019_Finals/set3.csv"), "6eab3bb513555a24dd970d8b330a2874", "repo"),
    FilePin(Path("training/data/shuttleset/annotations/set/Anthony_Sinisuka_GINTING_Anders_ANTONSEN_Indonesia_Masters_2020_Final/set1.csv"), "7c2e7348ff336f4100ef9ef54c07d6f5", "repo"),
    FilePin(Path("training/data/shuttleset/annotations/set/Anthony_Sinisuka_GINTING_Anders_ANTONSEN_Indonesia_Masters_2020_Final/set2.csv"), "37cc02ee4354763091c24135672c1945", "repo"),
    FilePin(Path("training/data/shuttleset/annotations/set/Anthony_Sinisuka_GINTING_Anders_ANTONSEN_Indonesia_Masters_2020_Final/set3.csv"), "e88c93225f1796d1b3e9bccfb70c3965", "repo"),
    FilePin(Path("training/data/shuttleset/annotations/set/An_Se_Young_Ratchanok_Intanon_YONEX_Thailand_Open_2021_QuarterFinals/set1.csv"), "5724e218db02fa8311551a20faa5207c", "repo"),
    FilePin(Path("training/data/shuttleset/annotations/set/An_Se_Young_Ratchanok_Intanon_YONEX_Thailand_Open_2021_QuarterFinals/set2.csv"), "d0010e431200a471f06e6b4ab4557b16", "repo"),
)


def _file_path(pin: FilePin) -> Path:
    return (fixtures_root() if pin.root == "fixtures" else REPO_ROOT) / pin.path


def verify_file(pin: FilePin) -> None:
    """Assert that one pinned file exists and has its recorded digest."""
    path = _file_path(pin)
    if not path.is_file():
        raise ValueError(f"fixture file missing: {pin.path}")
    actual = hashlib.md5(path.read_bytes()).hexdigest()
    if actual != pin.md5:
        raise ValueError(f"fixture file md5 mismatch: {pin.path}")


def verify_fixture(fixture: Fixture) -> None:
    """Assert every external file named by a fixture."""
    for pin in fixture.files:
        verify_file(pin)
