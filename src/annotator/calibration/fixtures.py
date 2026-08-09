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

    court_geo: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    net_band: tuple[float, float]
    resolution: tuple[float, float]


@dataclass(frozen=True)
class FixtureDigests:
    """The nine pinned file digests for one fixture, in ``Fixture.files`` order."""

    track: str
    bboxes: str
    scores: str
    kps: str
    kp_scores: str
    ndet: str
    dead_mask: str
    court_present: str
    scene_rows: str


@dataclass(frozen=True)
class Fixture:
    """All external and repository-local inputs for one scoring fixture.

    ``name`` is the fixture's sole local identity: every operational path and
    pin path below is a derived property of it (the stem-plus-role grammar),
    so copying a fixture with a new ``name`` (e.g. via ``dataclasses.replace``)
    moves every path consistently. The ``court_present_path`` property is a
    pose-derived court-view proxy (True = court view); the scene-gated
    tracker's producer choice is re-approved at its activation commit.
    """

    name: str
    video_id: int
    fps: float
    gt_set_dir: Path
    court_geo: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    net_band: tuple[float, float]
    resolution: tuple[float, float]
    n_rallies: int
    n_strokes: int
    digests: FixtureDigests

    @property
    def track_path(self) -> Path:
        return Path(f"{self.name}_track_npy/{self.name}_track.npy")

    @property
    def pose_dir(self) -> Path:
        return Path(f"{self.name}_pose_raw")

    def pose_path(self, role: str) -> Path:
        """Return the canonical path for one pose-role array (e.g. ``"kps"``)."""
        return self.pose_dir / f"{self.name}_pose_raw_{role}.npy"

    @property
    def mask_path(self) -> Path:
        return Path(f"{self.name}_results/{self.name}_dead_mask.npy")

    @property
    def court_present_path(self) -> Path:
        return Path(f"{self.name}_results/{self.name}_court_present.npy")

    @property
    def scene_rows_path(self) -> Path:
        return Path(f"{self.name}_results/{self.name}_scene_rows.csv")

    @property
    def files(self) -> tuple[FilePin, ...]:
        """The nine pinned external files, in the fixed pin order."""
        return (
            FilePin(self.track_path, self.digests.track, "fixtures"),
            FilePin(self.pose_path("bboxes"), self.digests.bboxes, "fixtures"),
            FilePin(self.pose_path("scores"), self.digests.scores, "fixtures"),
            FilePin(self.pose_path("kps"), self.digests.kps, "fixtures"),
            FilePin(self.pose_path("kp_scores"), self.digests.kp_scores, "fixtures"),
            FilePin(self.pose_path("ndet"), self.digests.ndet, "fixtures"),
            FilePin(self.mask_path, self.digests.dead_mask, "fixtures"),
            FilePin(self.court_present_path, self.digests.court_present, "fixtures"),
            FilePin(self.scene_rows_path, self.digests.scene_rows, "fixtures"),
        )

    @property
    def run_video_files(self) -> tuple[FilePin, ...]:
        """The eight pinned files consumed by ``build_run_video_inputs``."""
        unused_path = self.pose_path("kp_scores")
        return tuple(pin for pin in self.files if pin.path != unused_path)


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
# Worked arithmetic: docs/scraper_pipeline/annotator_fixture_geometry.md.
_HOMOGRAPHY_TO_FIXTURE_MULTIPLIER = 1.5
# The centre band spans one metre along the 13.4 m court-length axis.
_NET_BAND_HALF_WIDTH_M = 0.5
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
    court_geo = (x_bounds, y_bounds, net_band)
    return CalibrationGeometry(court_geo=court_geo, net_band=net_band, resolution=resolution)


def _rounded_bounds(values: np.ndarray) -> tuple[float, float]:
    """Return the minimum and maximum values rounded to one decimal place."""
    return round(float(values.min()), 1), round(float(values.max()), 1)


def _load_calibration_geometry(
    homography_path: Path = _HOMOGRAPHY_SOURCE,
    resolution_path: Path = _RESOLUTION_SOURCE,
) -> dict[int, CalibrationGeometry]:
    """Load and derive calibration geometry for every pinned ShuttleSet fixture."""
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


def _build_fixture(
    name: str, *, video_id: int, fps: float, gt_set_dir: Path,
    n_rallies: int, n_strokes: int, digests: FixtureDigests,
) -> Fixture:
    """Build one fixture; every path is a derived property of ``name`` (the stem-plus-role scheme)."""
    geometry = _CALIBRATION_GEOMETRY[video_id]
    return Fixture(
        name=name,
        video_id=video_id,
        fps=fps,
        gt_set_dir=gt_set_dir,
        court_geo=geometry.court_geo,
        net_band=geometry.net_band,
        resolution=geometry.resolution,
        n_rallies=n_rallies,
        n_strokes=n_strokes,
        digests=digests,
    )


# External digest provenance and pre/post move equality are recorded in the
# 2026-07-29 fixture-move state packet under "Core external move map and
# pre-move MD5s". All
# values below match that 2026-07-29 evidence.
SSET_01 = _build_fixture(
    "sset_01",
    video_id=1,
    fps=25.0,
    gt_set_dir=Path("training/data/shuttleset/annotations/set/Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2019_Finals"),
    n_rallies=113,
    n_strokes=1641,
    digests=FixtureDigests(
        track="08c5afced66b561517a43571df567b2f",
        bboxes="4c9525949d1c79f0161f81b2bb63d5ef",
        scores="03e655b3429f9482c5a3f4df766a3534",
        kps="621427713fc617d81d4081db15613b06",
        kp_scores="deb1ab46efcbe34a19bd4590b2f1b384",
        ndet="5cc366f2cd459ea9be44876bc07e74ea",
        dead_mask="a5043d329752a4e202c8566515b37231",
        court_present="095f6ee3a3a3042c06f42e6e4467e88d",
        scene_rows="378cfeb29a44e90ef9f9694344cca649",
    ),
)

SSET_15 = _build_fixture(
    "sset_15",
    video_id=15,
    fps=25.0,
    gt_set_dir=Path("training/data/shuttleset/annotations/set/Anthony_Sinisuka_GINTING_Anders_ANTONSEN_Indonesia_Masters_2020_Final"),
    n_rallies=104,
    n_strokes=824,
    digests=FixtureDigests(
        track="0b9c0966ffc58a36c65f97a5a9a78deb",
        bboxes="031d4f61f71f7e3f2e18a0af5e52b138",
        scores="5c3c7895312abbd28045968426fc21c4",
        kps="1d74ceef0fdd53dab60e3afd64e4a6fc",
        kp_scores="088d65a108d3be668dc71c93f4b9beb0",
        ndet="71f7f8a9e7f270fc0ffea868da437e08",
        dead_mask="c01914b9788afef3bca6e0b5bd88dc7f",
        court_present="8268eeed2c48914d165c31899ce9417b",
        scene_rows="a893afaf12920658338586e4b9b0d6d6",
    ),
)

SSET_21 = _build_fixture(
    "sset_21",
    video_id=21,
    fps=30.0,
    gt_set_dir=Path("training/data/shuttleset/annotations/set/An_Se_Young_Ratchanok_Intanon_YONEX_Thailand_Open_2021_QuarterFinals"),
    n_rallies=75,
    n_strokes=663,
    digests=FixtureDigests(
        track="ad00846dc78b08de728cf59ea773ad61",
        bboxes="3ee48b9637a49157ed494cbc0fbfab9a",
        scores="86ba65b4e902067853a51308db864a69",
        kps="6f5b60e0b2ae04ead4a3523aad744fa4",
        kp_scores="014561d30e74bd6811933d68dfd19525",
        ndet="1844e00ffd6cddfa1dd52e26442fef14",
        dead_mask="9a6b43bc14f795d8c5e4d62e86005798",
        court_present="93f5cbea19f8b7e65e272df9a5d0b252",
        scene_rows="f9fb06285637076c5817301ae7a7b41b",
    ),
)

FIXTURES = (SSET_01, SSET_15, SSET_21)

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


def verify_run_video_fixture(fixture: Fixture) -> None:
    """Assert the external files read while assembling ``run_video`` inputs."""
    for pin in fixture.run_video_files:
        verify_file(pin)
