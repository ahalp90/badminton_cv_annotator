"""Link and verify the frozen inputs for the corrected contact analysis."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from annotator.calibration.fixtures import FIXTURES, Fixture

RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[1]
ASSET_ROOT = RUN_DIR / "assets/shuttleset-current-annotator-reference-v1"
RELEASE_RESULTS = ASSET_ROOT / "measurement/current_annotator_8config_288p/static_shuttleset_homography"
PRODUCER_EVIDENCE = ASSET_ROOT / "inputs/tracknet_producer_evidence"
FIXTURE_ROOT = RUN_DIR / "inputs/fixtures"
LOCAL_POSE_ROOT = REPO_ROOT / "local_scratch/autograder_architecture"

POSE_ROLES = ("bboxes", "scores", "kps", "ndet")
STATIC_ARRAY_NAMES = (
    "court_present.npy",
    "keep_vote.npy",
    "raw_replay_mask.npy",
    "definitive_exclusion_mask.npy",
)
STATIC_FILE_NAMES = ("annotations.json", "scene_rows.csv", *STATIC_ARRAY_NAMES)


def md5(path: Path) -> str:
    """Return the MD5 digest of one input without loading it all at once."""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_path(destination: Path, source: Path) -> None:
    """Create one link, replacing an existing link but never a real path."""
    if not (source.is_file() or source.is_dir()):
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        destination.unlink()
    elif destination.exists():
        raise FileExistsError(f"refusing to replace non-symlink: {destination}")
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    print(f"linked {destination} -> {source.resolve()}")


def pin_digest(fixture: Fixture, relative_path: Path) -> str:
    """Return the recorded digest for one canonical fixture path."""
    for pin in fixture.files:
        if pin.path == relative_path:
            return pin.md5
    raise KeyError(f"no fixture pin for {fixture.name}: {relative_path}")


def verify_pinned_input(fixture: Fixture, relative_path: Path) -> None:
    """Verify one linked track or pose array against its fixture pin."""
    path = FIXTURE_ROOT / relative_path
    actual = md5(path)
    expected = pin_digest(fixture, relative_path)
    if actual != expected:
        raise ValueError(f"{fixture.name}: {relative_path} MD5 {actual} != {expected}")


def frame_count(path: Path) -> int:
    """Return the first dimension of an uncompressed NumPy array."""
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim == 0:
        raise ValueError(f"array has no frame dimension: {path}")
    return int(array.shape[0])


def verify_release_files(fixture: Fixture, n_frames: int) -> None:
    """Check the required static leaves and their frame-aligned arrays."""
    release_dir = RELEASE_RESULTS / fixture.name / "tracknet-stride-8"
    for name in STATIC_FILE_NAMES:
        path = release_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)

    for name in STATIC_ARRAY_NAMES:
        path = release_dir / name
        if frame_count(path) != n_frames:
            raise ValueError(f"{fixture.name}: {name} is not frame-aligned with {n_frames} frames")

    with (release_dir / "annotations.json").open(encoding="utf-8") as handle:
        annotations = json.load(handle)
    if not isinstance(annotations, dict) or "spans" not in annotations:
        raise ValueError(f"{fixture.name}: malformed annotations.json")


def verify_sidecar(fixture: Fixture, n_frames: int) -> None:
    """Check the stride-8 producer inpaint sidecar and its frame count."""
    sidecar = (
        PRODUCER_EVIDENCE
        / fixture.name
        / "tracknet-stride-8"
        / f"{fixture.video_id}_stride8_inpaint_mask.json.gz"
    )
    if not sidecar.is_file():
        raise FileNotFoundError(sidecar)
    with gzip.open(sidecar, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if (
        payload.get("schema") != "inpaint_fill_mask/1"
        or payload.get("index_space") != "frame"
        or payload.get("stride") != 8
        or payload.get("n_rows") != n_frames
    ):
        raise ValueError(f"{fixture.name}: misaligned stride-8 producer sidecar {sidecar}")


def prepare_inputs() -> None:
    """Create the ignored links and verify every analysis input."""
    if not ASSET_ROOT.is_dir():
        raise FileNotFoundError(f"missing frozen investigation asset: {ASSET_ROOT}")
    for fixture in FIXTURES:
        track_source = ASSET_ROOT / "inputs" / fixture.track_path
        link_path(FIXTURE_ROOT / fixture.track_path, track_source)
        for role in POSE_ROLES:
            pose_source = LOCAL_POSE_ROOT / fixture.pose_path(role)
            link_path(FIXTURE_ROOT / fixture.pose_path(role), pose_source)

        pinned_paths = (fixture.track_path, *(fixture.pose_path(role) for role in POSE_ROLES))
        for relative_path in pinned_paths:
            verify_pinned_input(fixture, relative_path)

        n_frames = frame_count(FIXTURE_ROOT / fixture.track_path)
        for role in POSE_ROLES:
            pose_path = FIXTURE_ROOT / fixture.pose_path(role)
            if frame_count(pose_path) != n_frames:
                raise ValueError(f"{fixture.name}: {role} is not frame-aligned with {n_frames} frames")
        verify_release_files(fixture, n_frames)
        verify_sidecar(fixture, n_frames)
        print(f"{fixture.name}: verified {n_frames} aligned frames")


if __name__ == "__main__":
    prepare_inputs()
