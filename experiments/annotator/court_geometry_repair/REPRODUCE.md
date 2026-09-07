# Regenerate the court repair comparison

The [saved-output checks](README.md#check-the-saved-results) are the quickest
way to evaluate the published evidence. This recipe regenerates the heavier
court-dependent pipeline with the same frozen models.

## Inputs and environment

Use the repository revision containing this bundle. The source implementation
measured by the report is `2b486d5`; the baseline evidence is from `a111181`.
The experiment scripts retain the comparison logic while accepting portable
input locations. Run every command from the repository root.

| Input | Required contents |
| --- | --- |
| `SOURCES` | Original ShuttleSet22 MP4s for videos 17 and 53 |
| `PREPARED` | Matching per-video directories containing the original `court_evidence.json.gz`, `court_receipt.json.gz` and pose arrays |
| `INPAINTED` | Matching per-video directories containing `shuttle_track_inpainted.npy.xz`, its guard codes and sidecar |
| Repository | CourtKeyNet weights, existing selection-model files and the evaluation code imported by the scripts |
| Bundle `models/` | Frozen contact model and fit/setting receipts |

The source basenames are:

- `17 CHEN_Yu_Fei_HE_Bing_Jiao_DAIHATSU_Indonesia_Masters_2022_Semifinals.mp4`
- `53 AN_Se_Young_Pornpawee_CHOCHUWONG_Korea_Open_Badminton_Championships_2022_Finals.mp4`

Prepared and inpainted directories use the same basenames without `.mp4`.
Both ShuttleSet22 sources must be 1920×1080 at 30 fps; preparation rejects other
formats because the frozen scorer assumes those dimensions and frame rate.
The pose files are `pose_kps.npy.xz`, `pose_bboxes.npy.xz`, `pose_scores.npy.xz`,
`pose_kp_scores.npy.xz` and `pose_ndet.npy.xz`. Preserve the original frame
alignment. Replacing these arrays with a fresh pose/shuttle extraction is a
new experiment rather than an exact rerun of the paired comparison.

The large video, pose and shuttle inputs are not distributed in this Git bundle.
Their absence does not prevent the saved-output checks. This bundle does not
currently provide a separate downloadable archive of those prepared inputs.

The frozen contact loader requires the recorded versions: Python 3.11.13,
NumPy 2.2.6, scikit-learn 1.6.1 and joblib 1.5.3. Use the project's inference
dependencies for OpenCV, PyTorch and CourtKeyNet as well. The original detector
runs used CUDA and `resize_mode="pad"`; other detector options used the defaults
at the measured revision. No tree-model fitting or tuning occurs in this recipe.

## Prepare the frozen contact model

Set `SOURCES`, `PREPARED` and `INPAINTED` to your input directories, then choose
a fresh output directory. `MODEL_REPO` points to this checkout because the other
selection models are already versioned here.

```bash
export PYTHONPATH="$PWD/src:$PWD"
BUNDLE="$PWD/experiments/annotator/court_geometry_repair"
MODEL_REPO="$PWD"
OUT="$(mktemp -d)"
mkdir -p "$OUT/contact_model"
cp "$BUNDLE/models/contact_model.joblib" "$OUT/contact_model/"
gzip -dc "$BUNDLE/models/final_contact_model_result.json.gz" \
  > "$OUT/contact_model/final_contact_model_result.json"
gzip -dc "$BUNDLE/models/final_contact_setting_result.json.gz" \
  > "$OUT/contact_model/final_contact_setting_result.json"
```

The model loader verifies its existing receipt identities and runtime versions.
The saved run metadata lists the opening, later and local selection-model paths
and the fixed policy file. Keep those versions for the comparison.

The historical measured report was produced from source revision `2b486d5`
with the public evidence bundle at `c9dc6ac`. The commands below run the
current checkout against the same kind of frozen inputs; current output is not
required to reproduce every historical score exactly. Treat any difference as
an item to inspect against the recorded evidence.

## Regenerate annotations and predictions

For each video, first sample the recorded scene frames and regenerate neural
court predictions. Current fallback outputs include the finite line fragments
needed by the acceptance rule. The original experiment added those fragments
to cached predictions in a separate step; the current producer includes them.
The scene cache also records the lossless image view summary and retained
alternative corners required by current grouping and final validation. The
preparation step rejects older or incomplete cache records with a regeneration
message; rerun `rebuild_scene_courts.py` instead of silently replaying a
historical cache without those fields. Cached frame dimensions must match the
source video used for replay; thumbnail dimensions are a separate check.

```bash
for VIDEO in 17 53; do
  python "$BUNDLE/scripts/rebuild_scene_courts.py" \
    --prepared-root "$PREPARED" --sources "$SOURCES" \
    --video-ids "$VIDEO" --device cuda --output "$OUT/nn_$VIDEO"

  for ROLE in before after; do
    BASELINE_ARGS=()
    if [ "$ROLE" = before ]; then
      BASELINE_ARGS=(--baseline)
    fi
    python "$BUNDLE/scripts/prepare_court_comparison.py" \
      --prepared-root "$PREPARED" --inpaint-root "$INPAINTED" \
      --sources "$SOURCES" --nn-results "$OUT/nn_$VIDEO" \
      --video-id "$VIDEO" --output "$OUT/${ROLE}_$VIDEO" \
      "${BASELINE_ARGS[@]}"
    python "$BUNDLE/scripts/score_court_comparison.py" \
      --prepared "$OUT/${ROLE}_$VIDEO" --model-repo "$MODEL_REPO" \
      --contact-model-dir "$OUT/contact_model" --video-id "$VIDEO" \
      --output "$OUT/${ROLE}_${VIDEO}_scores" "${BASELINE_ARGS[@]}"
  done

  python "$BUNDLE/scripts/evaluate_court_comparison.py" \
    --video-id "$VIDEO" --baseline "$OUT/before_${VIDEO}_scores" \
    --changed "$OUT/after_${VIDEO}_scores" \
    --labels-root "$BUNDLE/evidence/labels" \
    --output "$OUT/video_${VIDEO}_evaluation.json.gz"
done
```

Baseline mode loads the saved old court evidence and restores the old global
tracking and perspective-mask behaviour within the experiment process. Final
mode uses the repaired scene-specific pipeline. Both regenerate annotations and
features before applying the same models. This is a reconstruction of the
baseline, not a checkout-wide execution of historical source.

Compare the new evaluation with `evidence/video_17/expected.json.gz` and
`evidence/video_53/expected.json.gz`. The original reconstructed baselines
matched the saved contact and section records. Fresh neural inference may vary
with the runtime or source decoding, so inspect any differences rather than
silently accepting a changed score.

## Rerun the original ShuttleSet geometry controls

These controls use original ShuttleSet videos 3 and 21, distinct from the
ShuttleSet22 video numbering above. Their exact filenames and frame metadata
are recorded in each control's `comparison.json.gz`.

Export the old fallback directly from Git, then run both implementations on
the same freshly sampled frames and neural outputs:

```bash
git show a111181:src/courtkeynet/court_corners.py \
  > "$OUT/baseline_court_corners.py"

# Repeat for VIDEO=21 with its corresponding MP4.
VIDEO=3
python "$BUNDLE/scripts/rebuild_control_courts.py" \
  --video "$CONTROL_VIDEO" --video-id "$VIDEO" \
  --scenes-csv "$BUNDLE/evidence/controls/video_$VIDEO/scenes.csv.gz" \
  --homography-csv data/shuttleset/set/homography.csv \
  --baseline-module "$OUT/baseline_court_corners.py" \
  --device cuda --output "$OUT/control_$VIDEO"
```

`CONTROL_VIDEO` must name the corresponding original ShuttleSet MP4. Frames
are resized to 1280×720 with `INTER_AREA`. This command reproduces the boundary
comparison before donor repair. The saved control checker separately evaluates
the recorded donor candidates and painted-line evidence behind the final
geometry table; it requires no source video.
