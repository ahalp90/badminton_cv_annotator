# TrackNetV3 (Inference Only)

Shuttle trajectory extraction for the ShuttleSet stroke classification pipeline. This is a trimmed fork of [TrackNetV3](https://github.com/qaz812345/TrackNetV3) — training, evaluation, preprocessing, and error analysis modules have been removed. See the original repo for the full codebase.

## Setup

TrackNetV3 shares the BST training venv rather than maintaining a separate environment. See `../../bst_x/requirements.txt` for the full dependency list, and `requirements.txt` in this directory for standalone setup instructions if needed.

```bash
# From the repository root with the BST training venv active:
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r src/bst_x/requirements.txt
```

### Pretrained Weights

Download the [checkpoints](https://drive.google.com/file/d/1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA/view?usp=sharing) and unzip into `ckpts/`:

```bash
unzip TrackNetV3_ckpts.zip -d src/shared/tracknetv3/ckpts/
# Expected: ckpts/TrackNet_best.pt, ckpts/InpaintNet_best.pt
```

## Usage

### Via the pipeline (recommended)

The pipeline's `shuttle_extractor.py` calls `batch_predict.py` as a subprocess (`predict.py` remains for manual single-clip runs). Point `--tracknet-python` at the BST venv's Python:

```bash
cd src/bst_x
python -m pipeline.shuttle_extractor \
    --tracknet-python /path/to/bst-venv/bin/python
```

### Standalone

```bash
cd src/shared/tracknetv3
python predict.py --video_file test.mp4 \
    --tracknet_file ckpts/TrackNet_best.pt \
    --inpaintnet_file ckpts/InpaintNet_best.pt \
    --save_dir prediction
```

For large videos (prevents memory errors):

```bash
python predict.py --video_file test.mp4 \
    --tracknet_file ckpts/TrackNet_best.pt \
    --inpaintnet_file ckpts/InpaintNet_best.pt \
    --save_dir prediction \
    --large_video --video_range 324,330
```

Output: `{save_dir}/{video_name}_ball.csv` with columns `Frame, Visibility, X, Y`.

## What was removed

This fork retains only the inference path (`predict.py`, `model.py`, `dataset.py`, `utils/general.py`). The following modules were removed as they are not needed for shuttle coordinate extraction:

- `train.py` — model training loop
- `test.py` — evaluation and metrics (3 inference functions extracted to `inference_utils.py`)
- `preprocess.py` — dataset frame extraction
- `generate_mask_data.py` — InpaintNet training data generation
- `error_analysis.py`, `correct_label.py` — Dash-based analysis UIs
- `utils/visualize.py`, `utils/metric.py` — training visualization and loss functions

## Performance

From the [original paper](https://dl.acm.org/doi/10.1145/3595916.3626370), on the [Shuttlecock Trajectory Dataset](https://hackmd.io/Nf8Rh1NrSrqNUzmO0sQKZw) test split:

| Model | Accuracy | Precision | Recall | F1 | FPS |
|---|---|---|---|---|---|
| TrackNetV2 | 94.98% | **99.64%** | 94.56% | 97.03% | 27.70 |
| TrackNetV3 | **97.51%** | 97.79% | **99.33%** | **98.56%** | 25.11 |

## Reference

**Paper:** Yu-Huan Lin, et al. "TrackNetV3: Enhancing ShuttleCock Tracking with Augmentations and Trajectory Rectification." ACM MM 2023. [[link](https://dl.acm.org/doi/10.1145/3595916.3626370)]

**Original repo:** https://github.com/qaz812345/TrackNetV3
