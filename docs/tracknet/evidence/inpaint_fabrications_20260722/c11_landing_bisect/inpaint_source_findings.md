# Source of InpaintNet's bobbing fill loop

Code tree: `/home/ariel/Documents/COSC594/wt_annotator`
Branch: `feature/commentary-scraper`
Commit: `d04a78903160bc4948d23a1bf684b8e329b3d577`

All `src/...` citations below refer to that commit. The fixture provenance files sit outside the pinned worktree. They are named separately where used.

## Headline answers

### Mechanism

The source code supports the window-length explanation, the recorded run logs pin the window length at 16, and the checkpoint confirms it directly (question 1).

The fixture was made in `nonoverlap` mode. This mode divides the track into abutting model windows. Their outputs are appended without blending (`src/bric/perception/_vendor/tracknetv3/predict.py:263-281`). The known pilot rerun passed `--eval_mode nonoverlap` and used InpaintNet (`remote_logs/pilot_inpaint_redo_chain.sh:26-36`). That rerun matched the original pilot track element for element (`pilot_track_npy/README.md:3-5`).

A fully missing window contains zero shuttle coordinates and an all-one fill mask. A mask value of one tells InpaintNet to replace that coordinate. InpaintNet receives those three channels and no video pixels (`src/bric/perception/_vendor/tracknetv3/dataset.py:463-473`; `src/bric/perception/_vendor/tracknetv3/model.py:113-128`). Every fully missing window therefore gives the model the same input. In evaluation mode, the same input and weights give the same output. The learned weights produce the same position at each place within the window. Non-overlap stitching repeats that position sequence on one fixed set of frame offsets.

The measured track repeats one bit-identical 16-value vertical cycle across all three videos. It is phase-locked to the affected contacts. Phase-locked means the events keep the same offsets within the repeated cycle (`findings.txt:36-65`; `findings.txt:121-125`). With the window length confirmed at 16 and the checkpoint reproducing the saved cycle numerically (question 8), the mechanism is fully closed.

### Sidecar

The chain knows fill provenance when it creates `tracknet_pred_dict['Inpaint_Mask']` (`src/bric/perception/_vendor/tracknetv3/predict.py:257-261`). The chain loses that provenance on the next line. `inpaint_pred_dict` contains only frame, x, y and visibility (`src/bric/perception/_vendor/tracknetv3/predict.py:261`).

A separate, frame-aligned mask can be saved beside the track. This separate file is the sidecar. The existing `(n, 3)` array can stay byte-identical. The sidecar must use the same frame ordering and duplicate-frame rule as the track writer (`src/bst_x/pipeline/shuttle_extractor.py:260-274`).

## Numbered questions

### 1. Is `inpaintnet_seq_len` 16?

Yes, pinned from recorded artefacts (audit addition, 2026-07-22). The source does not set this value. `load_models` reads `param_dict['seq_len']` from the InpaintNet checkpoint (`src/bric/perception/_vendor/tracknetv3/predict.py:103-109`), and the laptop has no checkpoint at the runtime path. The rerun's console logs carry the value indirectly. In `nonoverlap` mode the inpaint loader's progress bar prints its batch total. That total equals the window count divided by the batch size, rounded up. The batch size defaulted to 16 (`src/bric/perception/_vendor/tracknetv3/predict.py:371`; the rerun passed no override, `remote_logs/pilot_inpaint_redo_chain.sh:30-37`). Pilot has 154,393 frames and its inpaint loader shows 604 batches (`remote_logs/pilot_inpaint_redo_console.log`). Only a window length of exactly 16 gives 604. Vid15 has 149,487 frames (`remote_logs/vid15_framecounts.txt`) and shows 584 batches (`remote_logs/vid15_step3_tracknet.log`). Again only 16 gives 584. The counted loader cannot be the TrackNet pass: its streaming dataset defines no length for the progress bar to print (`src/bric/perception/_vendor/tracknetv3/dataset.py:705` sets `video_len` but no `__len__`), and its window length of 8 would give 1,207 batches on pilot.

The direct checkpoint read ran on bourbaki the same day:

```bash
cd "$REPO/src/bst_x/TrackNetV3" && \
"$HOME/.venvs/venv-bst-x/bin/python" - <<'PY'
import torch

checkpoint = torch.load("ckpts/InpaintNet_best.pt", map_location="cpu")
print(checkpoint["param_dict"]["seq_len"])
PY
```

It printed 16, confirming the log arithmetic. The stronger weights check also ran; question 8 records its result.

### 2. How do consecutive inpaint windows connect?

The fill mask is made over the complete TrackNet result before windows are selected (`src/bric/perception/_vendor/tracknetv3/predict.py:257-265`). The dataset walks the coordinate sequence by `sliding_step`. It copies consecutive coordinates and mask values into each window (`src/bric/perception/_vendor/tracknetv3/dataset.py:357-396`).

In `nonoverlap` mode, `sliding_step` equals `seq_len` (`src/bric/perception/_vendor/tracknetv3/predict.py:263-266`). Each output window is converted to rows and appended in loader order (`src/bric/perception/_vendor/tracknetv3/predict.py:268-281`). The last short window repeats its final input frame for padding (`src/bric/perception/_vendor/tracknetv3/dataset.py:380-386`). All earlier windows abut exactly.

This selection explains the phase lock. Window starts are fixed from frame zero. Gaps do not start their own windows. Identical fully missing windows therefore repeat on the same global lattice.

The other modes differ. `average` and `weight` start a window every frame (`src/bric/perception/_vendor/tracknetv3/predict.py:283-287`). They blend the overlapping predictions for each frame (`src/bric/perception/_vendor/tracknetv3/predict.py:307-345`; `src/bric/perception/_vendor/tracknetv3/inference_utils.py:14-32`). They do not preserve the same abutting-window loop.

### 3. What does the model see for a fully missing window?

TrackNet records a miss as coordinate `(0, 0)` with visibility zero (`src/bric/perception/_vendor/tracknetv3/predict.py:57-73`). `generate_inpaint_mask` marks a qualifying missing span with ones (`src/bric/perception/_vendor/tracknetv3/inference_utils.py:73-87`). The coordinate dataset divides x and y by image width and height, so zeros stay zero (`src/bric/perception/_vendor/tracknetv3/dataset.py:463-473`).

InpaintNet then receives two coordinate channels and one mask channel. It receives no image or heatmap (`src/bric/perception/_vendor/tracknetv3/model.py:100-128`). A fully missing window is therefore identical across videos. Its output depends on the checkpoint weights, the window length and the fixed input. The output can vary by position inside the window. The model uses learned filters across neighbouring frames. It also pads the window boundaries (`src/bric/perception/_vendor/tracknetv3/model.py:76-86`; `src/bric/perception/_vendor/tracknetv3/model.py:100-128`).

This conclusion applies only when the whole model window is missing. A window that contains a real detection carries video-dependent coordinates.

### 4. Where does the fill mask go?

`generate_inpaint_mask` reads TrackNet visibility and y coordinates. It marks eligible leading and internal gaps (`src/bric/perception/_vendor/tracknetv3/inference_utils.py:59-90`). `predict_video` saves the result in `tracknet_pred_dict['Inpaint_Mask']` (`src/bric/perception/_vendor/tracknetv3/predict.py:257-261`). The dataset carries the mask beside the coordinate sequence (`src/bric/perception/_vendor/tracknetv3/dataset.py:357-398`). InpaintNet replaces coordinates only where the mask is one (`src/bric/perception/_vendor/tracknetv3/predict.py:268-276`).

The exact visibility assignment is `src/bric/perception/_vendor/tracknetv3/predict.py:69`. Any final coordinate other than `(0, 0)` gets visibility one. A successful fill therefore becomes indistinguishable from a detection in the visibility column. A fill thresholded back to `(0, 0)` stays invisible (`src/bric/perception/_vendor/tracknetv3/predict.py:59-73`; `src/bric/perception/_vendor/tracknetv3/predict.py:274-279`).

The exact provenance-loss point is `src/bric/perception/_vendor/tracknetv3/predict.py:261`. The new output dictionary omits `Inpaint_Mask`. The later selection chooses that reduced dictionary (`src/bric/perception/_vendor/tracknetv3/predict.py:348-357`). The normal CSV writer writes only frame, visibility, x and y (`src/bric/perception/_vendor/tracknetv3/utils/general.py:368-373`). Its `save_inpaint_mask` branch is for training data and also requires ground-truth fields, so the inference path cannot enable it unchanged (`src/bric/perception/_vendor/tracknetv3/utils/general.py:358-367`).

The current BRIC preprocessing path reads only the four normal CSV columns and writes x, y and visibility arrays (`src/bric/preprocessing/extract_shuttle.py:140-165`). The `(n, 3)` converter selects only x, y and visibility, then saves the result (`src/bst_x/pipeline/shuttle_extractor.py:260-274`). The recorded pilot writer performed the same selection and divided x and y by 512 and 288 (`remote_logs/pilot_inpaint_redo_chain.sh:42-56`). No downstream step ever receives the mask.

### 5. Does inpainting read `eval_mode`?

Yes. The dependence is direct.

`nonoverlap` selects windows one sequence length apart (`src/bric/perception/_vendor/tracknetv3/predict.py:263-266`). Other modes select one-frame steps and call `get_ensemble_weight` (`src/bric/perception/_vendor/tracknetv3/predict.py:283-318`). `average` uses uniform weights. `weight` favours central window positions (`src/bric/perception/_vendor/tracknetv3/inference_utils.py:14-32`).

The batch lane has no different inpaint path. It passes `eval_mode` into the same `predict_video` function (`src/bric/perception/_vendor/tracknetv3/batch_predict.py:45-49`; `src/bric/perception/_vendor/tracknetv3/batch_predict.py:71-77`).

A weight-mode retrack can change both the missing spans and how remaining fills are stitched. Remaining fully missing inputs still contain no video pixels. Their final track is not guaranteed to keep the non-overlap loop.

### 6. Is there a cap on fill-run length?

There is no length cap. The mask generator marks the whole eligible interval from `i` to `j`, with no span-length test (`src/bric/perception/_vendor/tracknetv3/inference_utils.py:73-88`). A long eligible gap crosses as many model windows as needed (`src/bric/perception/_vendor/tracknetv3/dataset.py:368-396`). Batch size changes only how many windows run together (`src/bric/perception/_vendor/tracknetv3/predict.py:265-268`).

Eligibility still has boundaries. A gap is filled only when its bounding detections sit clear of the top of the frame: numerically, y greater than a threshold that `predict_video` sets to 5 percent of the frame height (`src/bric/perception/_vendor/tracknetv3/predict.py:260`; `src/bric/perception/_vendor/tracknetv3/inference_utils.py:81-87`). A leading gap needs only its following detection clear. An internal gap needs both neighbours clear. A gap bounded by a top-edge detection reads as the shuttle leaving the picture and is left unfilled. A trailing gap ends at a miss, whose y of zero fails the same test, so it is also left unfilled. These are view-boundary rules, not length caps.

The last short non-overlap window is padded by repeating its final input (`src/bric/perception/_vendor/tracknetv3/dataset.py:380-386`). The code does not truncate a qualifying long run or switch to another filler.

### 7. Does filling happen before or after the 288-row snap?

The fill is computed before the saved coordinate snaps to an integer row.

The coordinate dataset first normalises TrackNet x and y by the input image size (`src/bric/perception/_vendor/tracknetv3/dataset.py:463-473`). InpaintNet produces continuous normalised values. Its sigmoid output confines each value between zero and one (`src/bric/perception/_vendor/tracknetv3/model.py:126-128`). `predict` then multiplies those values by the 512 by 288 model size and the image scale. The image scale maps the model grid to the input dimensions. `predict` converts the result with `int` (`src/bric/perception/_vendor/tracknetv3/predict.py:57-60`). The model size is fixed at height 288 and width 512 (`src/bric/perception/_vendor/tracknetv3/utils/general.py:23-28`).

The fixture command used a 512 by 288 input (`remote_logs/pilot_inpaint_redo_chain.sh:14-17`; `remote_logs/pilot_inpaint_redo_chain.sh:30-36`). Its image scale was therefore one. The integer conversion in `predict` snapped each fill to a 288-row y coordinate. The pilot writer later divided that integer by 288 (`remote_logs/pilot_inpaint_redo_chain.sh:42-55`). This order explains why the saved normalised fills sit exactly on grid values.

### 8. Does the code explain the measured loop coordinates?

No. The code contains no fixed x near 0.475 and no fixed y rows 67 to 84. InpaintNet applies learned one-dimensional convolutions and a sigmoid coordinate head (`src/bric/perception/_vendor/tracknetv3/model.py:76-128`). The only coordinate constants here are the 512 by 288 output scale and a near-origin rejection threshold (`src/bric/perception/_vendor/tracknetv3/utils/general.py:23-28`; `src/bric/perception/_vendor/tracknetv3/predict.py:274-276`). Those constants do not select the measured location.

The specific loop is therefore an arbitrary property of the trained weights for the fixed fully missing input. Integer conversion explains the grid alignment. It does not explain why the model chose those rows or that x position; nothing needs to, since the checkpoint experiment reproduced them on 2026-07-22.

The experiment fed the production checkpoint the fully missing input on a CPU: 16 zero coordinate pairs and an all-ones mask (`probe_inpaint_cycle.py`, this directory). Its output, converted exactly as `predict.py:60` converts it, is the saved loop. Across the 3,289 pilot windows whose y cycle matches at 14 or more of 16 positions, the track equals the probe at all 16 positions in y and 15 of 16 in x (`probe_vs_track.py`, this directory). The one exception is position 5: the probe gives x 245 and every saved window holds 244. The production GPU run and the CPU probe land on opposite sides of the 245.0 float boundary before the integer conversion truncates. The loop is the weights' fixed response to the evidence-free input, confirmed numerically.

## Fill-mask sidecar feasibility verdict

The sidecar is straightforward and does not require changing the existing `(n, 3)` schema. Save the frame-aligned `Inpaint_Mask` while `tracknet_pred_dict` still holds it at `src/bric/perception/_vendor/tracknetv3/predict.py:260`. Carry that separate artefact through the same frame ordering and duplicate-frame selection used by `src/bst_x/pipeline/shuttle_extractor.py:260-274`. Keep the current x, y and visibility selection and its `np.save` call unchanged. The track bytes then stay identical. The mask sidecar records one for a frame selected for filling. For a saved visible position, zero means the position came from TrackNet. An invisible mask-zero frame is an unfilled miss. A successful fill may have visibility one, but consumers can now distinguish its source.
