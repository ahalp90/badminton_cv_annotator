"""BST-X data paths, split membership, and pipeline constants."""
from pathlib import Path

from classifier_shared.dataset import parse_flaw_records


# ---------------------------------------------------------------------------
# Default paths (anchored to project root, not cwd)
# ---------------------------------------------------------------------------
# PROJECT_ROOT = src/bst_x/. REPO_ROOT walks up two more levels to the repo top;
# SHUTTLESET_DIR is the shared on-disk data dir at data/shuttleset/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent.parent
SHUTTLESET_DIR = REPO_ROOT / 'data' / 'shuttleset'

SET_INFO_DIR = SHUTTLESET_DIR / 'set'
RAW_VIDEO_DIR = SHUTTLESET_DIR / 'raw_video'
CLIPS_OUTPUT_DIR = SHUTTLESET_DIR / 'clips'
SHUTTLE_OUTPUT_DIR = SHUTTLESET_DIR / 'shuttle_npy'
SHUTTLE_CSV_DIR = SHUTTLESET_DIR / 'shuttle_csv'
FLAW_RECORDS_PATH = SHUTTLESET_DIR / 'flaw_shot_records.csv'
RESOLUTION_CSV_PATH = SHUTTLESET_DIR / 'my_raw_video_resolution.csv'


# ---------------------------------------------------------------------------
# Player-side rules
# ---------------------------------------------------------------------------
PLAYERS = ('Top', 'Bottom')

# Raw stroke types that get one flat folder at clip generation instead of
# split Top_/Bottom_ folders. Disk-layout concern, NOT a taxonomy property.
# 'unknown' lacks meaningful player attribution; 'driven_flight' is a transient
# type that's merged into 'drive' before training, so its raw folder exists
# unprefixed at clip-gen and disappears afterward.
NOSIDE_FOLDERS: frozenset[str] = frozenset({'unknown', 'driven_flight'})


# ---------------------------------------------------------------------------
# Pipeline scalars (clip window, homography reference)
# ---------------------------------------------------------------------------
CLIP_WINDOW = 'between_2_hits_with_max_limits'

# COCO-17 keypoint count. MMPoseInferencer('human') / RTMPose-L return 17
# joints per person; every pose tensor's joint axis is sized by this.
COCO_N_JOINTS = 17

# homography.csv matrices were computed at this resolution; coordinates must
# scale to match before applying the homography.
HOMOGRAPHY_RESOLUTION = (1280, 720)


# ---------------------------------------------------------------------------
# Flaw record parsing -- CSV is the single source of truth for exclusions
# ---------------------------------------------------------------------------
# Load lazily at import so path-only inspection works without the flaw CSV.
# A missing file leaves both sets empty, so execution can produce incorrect results.
try:
    EXCLUDED_VIDEOS, REMOVED_SHOTS = parse_flaw_records(FLAW_RECORDS_PATH)
    # Read-only downstream (membership + iteration only); frozenset makes the
    # download_adapter frozenset[int] annotation honest.
    EXCLUDED_VIDEOS = frozenset(EXCLUDED_VIDEOS)
except FileNotFoundError:
    import warnings
    warnings.warn(
        f'{FLAW_RECORDS_PATH} not found. '
        f'EXCLUDED_VIDEOS and REMOVED_SHOTS are empty. '
        f'This is fine for inspecting config, but the pipeline '
        f'will produce incorrect results without this file.',
        stacklevel=2,
    )
    EXCLUDED_VIDEOS, REMOVED_SHOTS = frozenset(), set()


# ---------------------------------------------------------------------------
# Match-level train/val/test splits
# ---------------------------------------------------------------------------
# Define with full intended ranges -- excluded videos are stripped automatically
# below, so you never need to manually skip them.
_SPLITS_RAW: dict[str, list[int]] = {
    'train': list(range(1, 35)),
    'val':   list(range(35, 39)) + [41],
    'test':  [39, 40, 42, 43, 44],
}

# Strip excluded videos so SPLITS and EXCLUDED_VIDEOS can never desync.
SPLITS: dict[str, list[int]] = {
    name: [v for v in ids if v not in EXCLUDED_VIDEOS]
    for name, ids in _SPLITS_RAW.items()
}


# ---------------------------------------------------------------------------
# Collated-dir naming -- writer + reader derive the same basename
# ---------------------------------------------------------------------------

def derive_npy_collated_dir_basename(
    *, seq_len: int, split_column: str, collation_id: str,
) -> str:
    """Format the collated dir basename: ``npy_[seq{N}_]{split}_{collation_id}``.

    Taxonomy lives in the parent dir (``ShuttleSet_data_<tax>/``), so isn't
    repeated here. ``seq_len=100`` is canonical and skips the ``seq{N}_`` tag.
    ``split_column`` has its ``split_`` prefix stripped at tag.
    Example ``collation_id`` values: ``'taxon_pinned_w_preds'``, ``'wipe_drop'``.
    """
    seq_tag = '' if seq_len == 100 else f'seq{seq_len}_'
    split_tag = split_column.removeprefix('split_')
    return f'npy_{seq_tag}{split_tag}_{collation_id}'


def collation_id_from_manifest(manifest: dict) -> str | None:
    """Resolve a run's collation generation tag from its manifest, current or pre-2026-06 format.

    New-schema manifests carry it directly as ``config.collation_id``. Pre-refactor
    manifests stored it in ``config.ablation_id`` or
    ``extra.data_provenance.effective_ablation_id`` instead. Reading the
    new-schema field first means a new manifest's training ``ablation_id``
    (different meaning) never gets misread as the collation tag.

    For internal scripts that read historical run data; the live FE registry
    sees new-schema manifests only and reads ``config.collation_id`` directly.

    :param manifest: a parsed run manifest (e.g. ``yaml.safe_load`` of manifest.yaml).
    :return: the collation tag, or None when none is present.
    """
    config = manifest.get('config') or {}
    if config.get('collation_id'):
        return config['collation_id']
    if config.get('ablation_id'):
        return config['ablation_id']
    provenance = (manifest.get('extra') or {}).get('data_provenance') or {}
    return provenance.get('collation_id') or provenance.get('effective_ablation_id')
