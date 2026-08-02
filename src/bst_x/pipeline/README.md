# pipeline/

Shared data pipeline for the ShuttleSet badminton stroke classification project. Produces labelled video clips and shuttle trajectory files consumed by both team architectures.

New here? [`data_pipeline_and_model_train_overview.md`](../data_pipeline_and_model_train_overview.md) gives the 5-minute picture; come back here for the `pipeline/` folder specifics, or see [`data_pipeline_to_model_train.md`](../data_pipeline_to_model_train.md) for the end-to-end run (pose, collation, training).

## Contents

- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Pipeline Steps](#pipeline-steps)
- [CLI Flags](#cli-flags)
- [Resuming after a crash](#resuming-after-a-crash)
- [Output Structure](#output-structure)
- [Pre-existing Input Data](#pre-existing-input-data)
- [Configuration](#configuration)
- [Module Reference](#module-reference)
- [Running Individual Steps](#running-individual-steps)
- [For Downstream Consumers](#for-downstream-consumers)

## Quick Start

Run these commands from the repository root with both source roots available:

```bash
export PYTHONPATH=src:src/bst_x

# Preview what the pipeline will do (no files created)
python -m pipeline.build_dataset --skip-shuttle --dry-run

# Run steps 1-5 (download, resolution CSV, clips, merge, verify)
python -m pipeline.build_dataset --skip-shuttle

# Run everything including shuttle extraction (uses BST venv for TrackNetV3)
python -m pipeline.build_dataset \
    --tracknet-python /path/to/bst-venv/bin/python
```

## Prerequisites

| Dependency | Install | Used by |
|---|---|---|
| yt-dlp | `pip install yt-dlp` | Step 1: video download |
| OpenCV | `pip install opencv-python` | Step 2: resolution scanning |
| MoviePy | `pip install moviepy` | Step 3: clip generation |
| pandas, numpy | `pip install pandas numpy` | All steps |
| TrackNetV3 | Included in repo (inference only). **Pretrained weights (~150 MB) must be downloaded separately** — see Step 6. Shares BST venv. | Step 6: shuttle extraction (optional) |

## Pipeline Steps

### Step 1: Download Videos

Downloads 40 ShuttleSet match videos from YouTube using yt-dlp. Checks that yt-dlp is installed before spawning workers. Skips videos that already exist on disk.

```bash
python -m pipeline.download_adapter --workers 4
```

Output: `data/shuttleset/raw_video/{id}.mp4`. Existing
`{id} {match_name}.mp4` files remain supported.

### Step 2: Build Resolution CSV

Scans downloaded videos with OpenCV and writes `my_raw_video_resolution.csv`. Replaces the need to manually create this file.

```bash
python -m pipeline.video_metadata
```

Output: `data/shuttleset/my_raw_video_resolution.csv`

### Step 3: Generate Clips

For each video in each split, extracts individual stroke clips using temporal boundaries from adjacent shots. Filters out excluded videos and individually removed shots automatically.

```bash
python -m pipeline.clip_generator --clip-window between_2_hits_with_max_limits
```

Three clip window options:
- `middle_in_a_sec` -- fixed 1-second window centered on the shot frame
- `between_2_hits` -- from previous shot's frame to next shot's frame
- `between_2_hits_with_max_limits` -- same as above, max 1.5s before / 1.75s after shot frame (default). See `data_pipeline_to_model_train.md` Key Concepts for the full windowing table.

Output: `data/shuttleset/clips/{train,val,test}/{Player}_{stroke_type}/{vid}_{set}_{rally}_{ball_round}.mp4`

### Step 4: Class Merge

Merges rare stroke subtypes into their parent types according to the active taxonomy's `merge_map`. The default taxonomy (`une_v1_14`) applies `UNE_MERGE_V1_MAP`, folding 4 subtypes into existing types:

| Subtype | Merged into |
|---|---|
| defensive_return_lob | lob |
| driven_flight | drive |
| back_court_drive | drive |
| defensive_return_drive | drive |

This reduces the 19 raw types to 14 (no sides, `unknown` excluded). `une_v1_15` is the same 14 plus `unknown`. The BST-paper family uses `MERGE_MAP_25` (6 merges, 19 -> 12 base types): `bst_25` (12 x Top/Bottom + `unknown` = 25), `bst_24` (no unknown = 24), `bst_12` (nosides = 12). `shuttleset_18` keeps all 18 raw types with no merge. Whether a taxonomy carries sides or keeps `unknown` is fixed per taxonomy (`has_sides`, `excluded_base_stroke_types`), not a runtime flag.

### Step 5: Verify

Checks that:
- All splits (train/val/test) exist and contain clips
- No clips from excluded videos (IDs from `flaw_shot_records.csv`)
- No individually removed shots present
- Merged subtype folders are empty
- No orphan files with unexpected naming patterns

### Step 6: Shuttle Extraction (Optional)

Runs TrackNetV3 on each clip to extract shuttle trajectories, then normalises to `(t, 3)` numpy arrays: `[x_norm, y_norm, visibility]`.

TrackNetV3 shares the BST training venv (`requirements.txt`) rather than maintaining a separate environment. The original repo's dependencies (torch 1.10, numpy 1.22) are incompatible with Python 3.11 and CUDA 12.1; the code has been verified to work with torch 2.3.1. See `src/shared/tracknetv3/requirements.txt` for the full version rationale and standalone setup instructions.

The pipeline calls TrackNetV3 as a subprocess via `batch_predict.py`, which loads models once and iterates over all clips in-process. This avoids the ~8s model-reload overhead per clip that the old subprocess-per-clip approach had. The pipeline passes `--batch_size` (default 32; configurable via `--batch-size`) and uses the default `eval_mode='weight'` (full temporal ensemble) for maximum detection accuracy. Inference runs in FP32 to preserve detection accuracy on fast-moving shuttles (>400 km/h at 25-30fps produces faint heatmap responses where FP16 rounding could flip the 0.5 visibility threshold). Frames are pre-resized during loading using PIL BICUBIC, which is bit-identical to the Dataset's own resize and avoids redundant full-resolution array operations. VideoCapture handles are explicitly released after use, and `gc.collect()` + `torch.cuda.empty_cache()` run between clips to prevent resource exhaustion over long batch runs. TrackNetV3's imports don't affect the pipeline venv. Point `--tracknet-python` at the BST venv's Python.

#### One-time setup

1. **Download pretrained weights** from [Google Drive](https://drive.google.com/file/d/1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA/view?usp=sharing) (~150 MB zip). These are too large for the git repo (`ckpts/` is gitignored).

   ```bash
   cd ../shared/tracknetv3
   pip install gdown              # if not already installed
   gdown 1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA
   unzip TrackNetV3_ckpts.zip -d ckpts/
   # Expected: ckpts/TrackNet_best.pt, ckpts/InpaintNet_best.pt
   cd ../../bst_x
   ```

   Without InpaintNet weights the pipeline will warn and fall back to TrackNet-only (no gap-filling for occluded frames). Without TrackNet weights step 6 will fail.

2. **Create output directories.** Step 6 writes intermediate CSVs to `data/shuttleset/shuttle_csv/` and final `.npy` files to `data/shuttleset/shuttle_npy/`. On HPC nodes these should live on scratch storage and be symlinked:

   ```bash
   # Example for engelbart (adjust paths for your setup)
   mkdir -p /scratch/comp320a/ShuttleSet/shuttle_csv
   mkdir -p /scratch/comp320a/ShuttleSet/shuttle_npy
   ln -s /scratch/comp320a/ShuttleSet/shuttle_csv data/shuttleset/shuttle_csv
   ln -s /scratch/comp320a/ShuttleSet/shuttle_npy data/shuttleset/shuttle_npy
   ```

#### Running

```bash
# Run from the pipeline's own venv (batch mode, single GPU)
python -m pipeline.shuttle_extractor \
    --tracknet-python /path/to/bst-venv/bin/python --workers 1 --batch-size 16

# Retry any OOM failures with a smaller batch size (resume picks up where it left off)
python -m pipeline.shuttle_extractor \
    --tracknet-python /path/to/bst-venv/bin/python --workers 1 --batch-size 8

# Dry run (processes clips but writes no files — test that the pipeline works)
python -m pipeline.shuttle_extractor \
    --tracknet-python /path/to/bst-venv/bin/python --workers 1 --batch-size 16 --dry-run
```

`--workers N` launches N parallel batch processes, each loading its own model copy. Use `--workers 1` on V100 16GB (two copies OOM). On A100 40GB or multi-GPU nodes, `--workers 2` roughly halves wall time. `--batch-size` controls the TrackNet DataLoader batch size (default 32). FP32 inference on V100 16GB fits batch_size 16 comfortably; a small number of clips may OOM at 16, so re-run with batch_size 8 to pick up the stragglers (the resume logic skips clips that already have CSVs).

If omitted, `--tracknet-python` defaults to the current interpreter (`sys.executable`).

Single-clip inference is still available via `predict.py` directly (e.g. for deployment):

```bash
cd ../shared/tracknetv3
python predict.py --video_file clip.mp4 --tracknet_file ckpts/TrackNet_best.pt \
    --inpaintnet_file ckpts/InpaintNet_best.pt --save_dir output/
```

**Frame-level guarantees:** TrackNetV3's output CSVs always contain a contiguous Frame column `[0, 1, ..., N-1]` matching the input video length. Frames where the shuttle is undetected are written with zeroed coordinates and `Visibility=0` (never skipped), and buffer flushing ensures trailing frames are included. This means `shuttle_csvs_to_npy` can safely call `.set_index('Frame').to_numpy()` without gap-filling or reindexing.

Output: `data/shuttleset/shuttle_npy/{vid}_{set}_{rally}_{ball_round}.npy` (flat). Split and label assignment are carried by `notebooks/clips_master.csv` at collation time, not by the on-disk directory layout. See `docs/archive/completed_general_refactors/dir_flatten_refactor.md` for the migration.

Each `.npy` file has shape `(t, 3)`. To get xy-only coordinates: `shuttle[:, :2]`. To get the visibility mask: `shuttle[:, 2]`.

## CLI Flags

```
python -m pipeline.build_dataset [OPTIONS]

--tracknet-dir PATH    Optional TrackNetV3 override (default: src/shared/tracknetv3)
--tracknet-python PATH Python executable in BST venv (default: sys.executable)
--workers N            Parallel workers (default 2, safe for shared GPU nodes)
--batch-size N         Batch size for TrackNet DataLoader (default 32; use 16 on V100 16GB)
--skip-download        Skip YouTube download (videos must already exist)
--skip-resolution      Skip resolution CSV rebuild (keep existing CSV)
--skip-clips           Skip clip generation and class merge (steps 3-4)
--skip-verify          Skip verification checks (step 5)
--skip-shuttle         Skip TrackNetV3 shuttle extraction
--no-merge             Keep all 19 stroke types (skip class merging)
--taxonomy NAME        Taxonomy (default une_v1_14): bst_25, bst_24, bst_12, une_v1_14, une_v1_15, shuttleset_18.
--dry-run              Preview what the pipeline would do without executing
--force                Continue past verification failures
```

```
python -m pipeline.shuttle_extractor [OPTIONS]

--tracknet-dir PATH    Optional TrackNetV3 override (default: src/shared/tracknetv3)
--clips-dir PATH       Directory containing generated clips
--csv-dir PATH         Directory for TrackNetV3 CSV outputs
--npy-dir PATH         Output directory for normalised .npy files
--resolution-csv PATH  Path to video resolution CSV
--model-path PATH      Path to TrackNet weights
--inpaintnet-path PATH Path to InpaintNet weights
--workers N            Parallel batch workers (default 2)
--batch-size N         Batch size for TrackNet DataLoader (default 32)
--tracknet-python PATH Python executable in BST venv
--skip-extraction      Skip TrackNetV3 extraction, only convert existing CSVs to NPY
--dry-run              Run inference without writing output files (test pipeline)
```

### Resuming after a crash

Class merge (step 4) is destructive — it moves clips from subtype folders (e.g. `Top_wrist_smash/`) into parent folders (e.g. `Top_smash/`) and removes the source folders. If the pipeline crashes after step 4 and you re-run without `--skip-clips`, step 3 will not find the merged clips at their original paths and will **re-generate them from video** (hours of re-encoding).

To resume safely after steps 3-5 have completed:

```bash
# Skip straight to shuttle extraction (step 6)
python -m pipeline.build_dataset \
    --skip-download --skip-resolution --skip-clips --skip-verify \
    --tracknet-python /path/to/bst-venv/bin/python

# Or run step 6 directly via its own CLI
python -m pipeline.shuttle_extractor \
    --tracknet-python /path/to/bst-venv/bin/python
```

## Output Structure

```
data/shuttleset/
  raw_video/                                    # Step 1
    {id}.mp4
    sources.toml                                # Download resume metadata
  my_raw_video_resolution.csv                   # Step 2
  clips/                                        # Steps 3-4 (still nested)
    train/{Top,Bottom}_{stroke_type}/*.mp4
    val/{Top,Bottom}_{stroke_type}/*.mp4
    test/{Top,Bottom}_{stroke_type}/*.mp4
  shuttle_csv/                                  # Step 6 (intermediate, flat)
    {vid}_{set}_{rally}_{ball_round}_ball.csv
  shuttle_npy/                                  # Step 6 (final, flat)
    {vid}_{set}_{rally}_{ball_round}.npy
```

Clip filenames: `{video_id}_{set}_{rally}_{ball_round}.mp4`

Split and label assignment for `shuttle_npy/` (and the downstream pose npys) come from `notebooks/clips_master.csv` at collation time. The clips directory stays nested for now; flattening it is deferred. See `docs/archive/completed_general_refactors/dir_flatten_refactor.md` for the migration plan.

## Pre-existing Input Data

These files ship with the ShuttleSet dataset and are required by the pipeline. Do not delete them.

| File | Read by | Contents |
|---|---|---|
| `data/shuttleset/set/match.csv` | `download_adapter.py`, `clip_generator.py` | Match metadata: video IDs, YouTube URLs, player court orientation (`downcourt` flag). 44 matches. |
| `data/shuttleset/set/{match_folder}/set[1-3].csv` | `clip_generator.py`, `classifier_shared.player_mapping` | Per-set stroke annotations: stroke type (Chinese), rally/ball_round numbers, frame timestamps, player A/B labels. One folder per match, up to 3 CSVs per folder. |
| `data/shuttleset/set/homography.csv` | `shared.court`, `prepare_train_on_shuttleset.py` | Homography matrices and court corner coordinates for camera-to-court projection. Computed at 1280x720 (W x H) resolution. Optional for basic pipeline; required for court-normalised features. |
| `data/shuttleset/flaw_shot_records.csv` | `classifier_shared.dataset` via `pipeline/config.py` | Data quality records: 4 whole-video exclusions and 25 individual shot removals. Drives `EXCLUDED_VIDEOS` and `REMOVED_SHOTS` constants. |
| `data/shuttleset/my_raw_video_resolution.csv` | `shared.court`, `prepare_train_on_shuttleset.py` | Video dimensions (id, width, height). Auto-regenerated by Step 2, but the pre-existing copy is useful as a reference before videos are downloaded. |

The original repo's pre-refactor scripts and spreadsheets (`gen_my_dataset.py`, `get_each_class_total.py`, `class_total.xlsx`, etc.) were relocated to `scratch/project_history/shuttleset_deprecated/` by step 3 of the pre-phase-2 tidy. Nothing in the active pipeline reads from them.

## Configuration

BST-X paths and split constants live in `pipeline/config.py`. Taxonomy data and
label derivation live in `classifier_shared/taxonomy.py`.

| Constant | Description |
|---|---|
| `BST_X_TAXONOMIES` | BST-X command registry containing `bst_25`, `bst_24`, `bst_12`, `une_v1_14`, `une_v1_15`, and `shuttleset_18`. |
| `TAXONOMIES` | Complete classifier registry containing the BST-X taxonomies and BRIC label spaces. |
| `taxonomy_lookup()` | Look up a `Taxonomy` by canonical name; raises `KeyError` for unknown names. |
| `derive_class_index()` | The single per-row label decision: `excluded_base_stroke_types` (drop), then `merge_map`, then side-prefixing. Shared by the collator and `data_access`. |
| `NOSIDE_FOLDERS` | Frozenset of raw types that get one flat folder at clip generation instead of split `Top_`/`Bottom_` folders (`{'unknown', 'driven_flight'}`). Disk-layout concern; not a taxonomy property. |
| `SPLITS` | Train/val/test video ID lists (excluded videos auto-stripped) |
| `EXCLUDED_VIDEOS` | Parsed from `flaw_shot_records.csv` at import time |
| `REMOVED_SHOTS` | Individual bad shots, also from `flaw_shot_records.csv` |
| `UNE_MERGE_V1_MAP` | Taxonomy-module merge map for `une_v1_14` and `une_v1_15`. |
| `MERGE_MAP_25` | Taxonomy-module BST-paper merge map used by `bst_25`, `bst_24`, and `bst_12`. |
| `CLIP_WINDOW` | Default temporal clipping strategy |
| `EN_TO_ZH` / `ZH_TO_EN` | English-Chinese stroke name translation (used at CSV I/O boundary only) |
| `HOMOGRAPHY_RESOLUTION` | Resolution (1280, 720) at which homography matrices were computed. Coordinates must be scaled before applying homography. Used by `shared.court`. |

### Changing Splits

Edit `_SPLITS_RAW` in `config.py`. Use full ranges -- excluded videos are stripped automatically:

```python
_SPLITS_RAW = {
    'train': list(range(1, 35)),        # Videos 1-34 minus exclusions
    'val':   list(range(35, 39)) + [41],
    'test':  [39, 40, 42, 43, 44],
}
```

### Adding Exclusions

Update `data/shuttleset/flaw_shot_records.csv`. The pipeline reads it at import time -- no code changes needed.

## Module Reference

| Module | Purpose |
|---|---|
| `config.py` | BST-X paths, splits, and pipeline constants |
| `classifier_shared/taxonomy.py` | Classifier taxonomy definitions, stroke mappings, and label derivation |
| `classifier_shared/dataset.py` | ShuttleSet paths, flaw parsing, split metadata, and clip bounds |
| `classifier_shared/player_mapping.py` | A/B to Top/Bottom mapping with set 3 court-switch handling |
| `download_adapter.py` | ShuttleSet adapter to the scraper-owned yt-dlp downloader |
| `video_metadata.py` | Resolution CSV builder and missing-video report |
| `clip_generator.py` | Clip extraction, flaw filtering, class merging |
| `shuttle_extractor.py` | TrackNetV3 wrapper + CSV-to-NPY normalisation |
| `shared/court.py` | Shared homography-based court projection utilities |
| `verify.py` | Post-generation sanity checks |
| `build_dataset.py` | One-command orchestrator |
| `clip_index.py` | `build_clip_path_index(clips_dir)` helper: one-time rglob to build a `{clip_stem -> mp4 Path}` lookup for CSV-driven video-loading Datasets. Used by downstream arch code (Arch 2 3D CNN, Arch 1 wrist crop). |
| `data_access.py` | CSV-aware access layer on top of `clip_index.py`. `get_clip_records(paths, split=..., taxonomy_class=..., split_column=..., taxonomy_name=...)` reads `clips_master.csv`, derives the folder-style class label under the active taxonomy, and returns `ClipRecord`s pairing clip / flat shuttle / flat rtmpose paths. Also exposes a CLI + TUI (`python -m pipeline.data_access`) and a `.env` mechanism for per-environment path config. |

## Running Individual Steps

Each module can be run standalone:

```bash
python -m pipeline.download_adapter --workers 4
python -m pipeline.video_metadata
python -m pipeline.clip_generator --clip-window between_2_hits
python -m pipeline.shuttle_extractor \
    --tracknet-python /path/to/bst-venv/bin/python
python -m pipeline.verify --clips-dir data/shuttleset/clips
```

## For Downstream Consumers

Both architectures read from the same `clips/` and `shuttle_npy/` directories. The pipeline doesn't care what you do with the output.

**Next step for BST-X:** Run `preparing_data/prepare_train_on_shuttleset.py` to extract poses (rtmlib) and collate into batch-ready arrays. See `data_pipeline_to_model_train.md` at the project root for the full pipeline-to-training walkthrough. For the COCO 17-keypoint joint index map, bone pairs, and JnB representations, see [`keypoints_schema.md`](../preparing_data/keypoints_schema.md). To run the pose extract across several GPU workers at once (per-node worker counts, sharding, command blocks), see [`extraction_saturation_runbook.md`](../../../docs/architecture_notes/rtmlib_migration/extraction_saturation_runbook.md).

### Split + label source

Split (train/val/test) and class label for every clip come from `notebooks/clips_master.csv` at collation time, not from the on-disk folder layout. The clips directory is still nested as `{split}/{Top,Bottom}_{stroke_type}/*.mp4` (Phase 3 flattening is deferred), but the `{split}/` parent reflects the historical `split_bst_baseline` partition only — any new ablation split (e.g. `split_v2`) is applied via the CSV.

### Loading shuttle / label data (flat)

```python
# Loading shuttle trajectories (flat: one file per clip, named after clip stem)
import numpy as np

shuttle = np.load('data/shuttleset/shuttle_npy/1_1_3_2.npy')
xy = shuttle[:, :2]  # (t, 2) normalised coordinates
visibility = shuttle[:, 2]  # (t,) detection confidence

# Getting class labels: each Taxonomy pins its full ordered class list in .classes
from classifier_shared.taxonomy import TAXONOMIES, taxonomy_lookup

labels_14 = TAXONOMIES['une_v1_14'].classes  # 14 classes (current default)
labels_25 = TAXONOMIES['bst_25'].classes  # 25 classes (BST-paper family)
labels_18 = TAXONOMIES['shuttleset_18'].classes  # 18 raw types
classes = taxonomy_lookup('bst_25').classes  # equivalent canonical-name lookup
```

### Loading clip frames (video-Dataset pattern)

Any `Dataset` that needs per-clip video frames should use `pipeline.clip_index.build_clip_path_index` to get an O(1) `clip_stem -> Path` lookup once at `__init__` against the still-nested clips dir, then pair it with a CSV-driven split + label. Skeleton:

```python
import pandas as pd
from torch.utils.data import Dataset

from pipeline.clip_index import build_clip_path_index
from classifier_shared.taxonomy import TAXONOMIES
from pipeline.config import CLIPS_OUTPUT_DIR

class ClipVideoDataset(Dataset):
    def __init__(self, clips_csv, split_column, taxonomy_name,
                 split='train', clips_dir=CLIPS_OUTPUT_DIR):
        df = pd.read_csv(clips_csv)
        df = df[df[split_column] == split]
        taxonomy = TAXONOMIES[taxonomy_name]
        self._path_by_stem = build_clip_path_index(clips_dir)
        self.items = [
            (row.clip_stem, _derive_label(row, taxonomy))
            for row in df.itertuples()
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        stem, label = self.items[i]
        return load_video(self._path_by_stem[stem]), label
```

`derive_class_index(taxonomy, row.raw_type_en, row.player_side)` produces the int label (or `None` to skip a filtered row), applying exclusion + merge + side in one place, matching what `collate_npy` does for the pose/shuttle npys (see `preparing_data/prepare_train_on_shuttleset.py`). Pick your own video backend (cv2, decord, torchvision.io) for `load_video`.

This pattern means the nested clips layout is transparent: the same `ClipVideoDataset` works for any `split_column` in `clips_master.csv` without needing to flatten or reorganise `clips/`.

### Higher-level access (`pipeline/data_access.py`)

For ad-hoc "give me the clip / shuttle / rtmpose paths for this split and this class" queries, `pipeline.data_access` wraps the CSV read + `build_clip_path_index` + flat-path resolution in one call:

```python
from pipeline.data_access import DataPaths, get_clip_records

records = get_clip_records(
    DataPaths(),
    split='train',
    taxonomy_class='smash',
    split_column='split_v2',
    taxonomy_name='une_v1_14',
)
for r in records:
    print(r.clip_stem, r.clip, r.shuttle_npy, r.rtmpose_joints)
```

`DataPaths` resolves paths in priority order: constructor arg > environment variable (or `.env` file entry: `BST_X_CLIPS_DIR`, `BST_X_SHUTTLE_NPY_DIR`, `BST_X_RTMPOSE_NPY_DIR`, `BST_X_CLIPS_CSV`) > `pipeline.config` defaults. Copy `.env.example` at the repo root to `.env` to pin paths per environment.

CLI + TUI for quick inspection:

```bash
python -m pipeline.data_access --summary                        # counts per split+class
python -m pipeline.data_access --split val --class smash    # list paths
python -m pipeline.data_access --split-column split_v2 --summary
python -m pipeline.data_access --list-classes                   # active taxonomy's classes
python -m pipeline.data_access                                   # interactive TUI
```

`clip_index.build_clip_path_index` remains the zero-dep pathlib helper for Datasets that just need `{stem -> Path}`; `data_access` is the CSV-aware layer above it.
