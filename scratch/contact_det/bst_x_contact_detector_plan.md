# BST-X contact detector implementation plan

## Table of contents

- [Recommendation](#recommendation)
- [Decisions for the first build](#decisions-for-the-first-build)
- [What exists now](#what-exists-now)
- [Model design](#model-design)
- [Contact window preparation](#contact-window-preparation)
- [Labels and examples](#labels-and-examples)
- [Search regions and event decoding](#search-regions-and-event-decoding)
- [Splits and leakage controls](#splits-and-leakage-controls)
- [Metrics and retained outputs](#metrics-and-retained-outputs)
- [Implementation stages](#implementation-stages)
- [Tests](#tests)
- [Compute and remote use](#compute-and-remote-use)
- [Failure and stop criteria](#failure-and-stop-criteria)
- [Expected commits](#expected-commits)
- [First-run recipe](#first-run-recipe)
- [Approval checklist](#approval-checklist)

## Recommendation

For the first inside-region pilot, build BST-X as a local, frame-aligned contact scorer over 21 consecutive source frames. Keep the current player, shuttle and court streams. Replace the clip-level readout with a temporal head that emits one contact logit per frame, then use only the centre-frame logit for each stride-1 window.

The first executable experiment should test whether temporal physical evidence beats histogram boosting inside one shared set of label-blind regions. On region version 2 at ±10, HGB physics reaches 84.5% precision, 90.5% recall and 87.4% F1. An eligible-court-only sensitivity refit reaches 86.3% precision, 89.1% recall and 87.7% F1. The harder acceptance floor remains 89.9% F1 with at least 87.5% precision. A validity-only control must remain clearly below the full model.

Region version 2 is now measured. Its 45-base-30-frame pre-roll before court-view intervals raises pooled operational ±10 coverage to 98.4% for non-serves and 97.9% for serves. Every fixture passes the 90% serve gate. `sset_21` remains at 93.7% non-serve coverage, below the 97% gate. A bounded BST-X pilot can compare classifiers on this region, but full training and whole-video claims remain blocked until a separate off-court shuttle search is defined.

The version-2 tree result also sets the physical controls. HGB physics reaches 92.9% non-serve recall and 67.5% serve recall. Adding context lowers F1 to 85.5%. Context-only F1 is 59.1%; missingness-only F1 is 27.5%. The primary BST-X comparison is therefore the physics tree.

Use ShuttleSet for model development. Keep ShuttleSet22 separate for the first cross-dataset test. ShuttleSet22 has the same stroke annotation format and is prepared separately. Holding it out gives a much stronger generalisation result than mixing it into the first training run.

## Decisions for the first build

| Item | First build | Reason |
| --- | --- | --- |
| Input width | 21 consecutive frames | An odd width gives an unambiguous centre and remains close to the proposed 20-frame window |
| Contact output | One logit per frame; gather the centre logit | The output stays time-aligned and does not use a whole-clip CLS summary |
| Contact labels | Positive within ±2 base-30 frames; ignore from 3 to 5; negative from 6 onward | This tolerates small annotation error while still training a narrow peak |
| Main loss | Mean binary cross-entropy on sampled eligible centre labels | The fixed sampler already controls the class mix |
| Side output | Separate Top/Bottom logit, supervised only on positive centres | Contact confidence and side confidence can abstain independently |
| Search | Frozen region version 2, then stride-1 centre scoring | The bounded surface greatly improves serve coverage but still omits 37 `sset_21` non-serves |
| Peak selection | Held-out logit threshold, local maxima, then temporal NMS | It preserves the strongest frame for each event |
| Development split | Whole source videos and fixtures | Neighbouring windows from one rally must not cross splits |
| External test | ShuttleSet22 held out in full | This tests dataset and match generalisation |

The 21-frame choice is fixed for the first result. Only run a 15/21/31 width ablation if the first model finds the right events but produces broad or consistently shifted peaks.

## What exists now

The current BST-X code is a stroke classifier, not a contact detector.

- [`BST.forward`](../../src/bst_x/model/bst.py) receives flattened joint and bone features, shuttle `x/y`, player court positions and `video_len`. It uses temporal and player-shuttle cross attention, then applies the interactional transformer separately to each player's sequence. It joins three CLS summaries before `mlp_head` emits clip logits.
- [`build_bst_x_network`](../../src/bst_x/bst_x_common.py) constructs the current classifier. [`flatten_pose_features`](../../src/bst_x/bst_x_common.py) applies the shared pose flattening step.
- [`Dataset_npy_collated`](../../src/bst_x/preparing_data/shuttleset_dataset.py) loads whole stroke clips. [`make_seq_len_same`](../../src/bst_x/preparing_data/shuttleset_dataset.py) pads short clips and linspace-samples long clips. Setting its target length to 21 would resample a whole stroke clip. It would not make a consecutive contact-centred window.
- [`detect_players_2d`](../../src/bst_x/preparing_data/prepare_train_on_shuttleset.py) zero-fills a frame when it cannot select both players and records the frame in `_failed.npy`.
- [`get_shuttle_result`](../../src/bst_x/preparing_data/prepare_train_on_shuttleset.py) discards the shuttle visibility column. `_align_shuttle_and_truncate` retains visible shuttle coordinates when pose fails, but the collated model input still cannot distinguish an invisible shuttle from the saved `(0, 0)` sentinel.
- [`build_hit_frame_lookup`](../../src/bst_x/validation_scripts/hit_frame_lookup.py) re-derives the annotated hit's local clip index from [`compute_clip_bounds`](../../src/classifier_shared/dataset.py). It is useful, but it assumes the generated clip begins on the nominal source frame.
- [`_frame_to_time`](../../src/bst_x/pipeline/clip_generator.py) documents an unresolved possible one-frame MoviePy offset. Frame alignment therefore needs a measured gate before the local hit index is trusted.
- [`impulse_cell_candidates`](../../src/annotator/rally/contacts.py) makes the current raw shuttle-impulse proposals. [`assemble_contacts`](../../src/annotator/rally/contacts.py) applies the wrist gate and temporal suppression.
- [`build_sticky_result`](../../src/annotator/rally/evidence.py) retains frame-aligned player picks, wrist distances and missingness for full-video annotation. It does not expose the normalised joint and player-position tensors expected by BST-X.

The new detector should live beside the stroke classifier. It should not change the current stroke data, checkpoints or inference contract.

## Model design

### Inputs

Each example is one 21-frame consecutive window with these tensors:

```text
pose                 (21, 2, 36, 2)   JnB_bone, Top then Bottom
player_position      (21, 2, 2)       normalised court coordinates
shuttle_xy           (21, 2)          normalised image coordinates
pose_valid           (21, 2)          one flag per player slot
shuttle_valid        (21,)            TrackNet visibility
source_frame_valid   (21,)            false only for boundary padding
centre_index         scalar           always 10 in the first build
```

For the legacy `detect_players_2d` extract, `_failed.npy` is a frame-level two-player failure flag and both slots are zero-filled together. Repeat `~failed` across the two slots only for that producer. The sticky-anchor producer can retain one slot while marking the frame failed. For that producer, build per-slot validity from its raw accepted picks or rerun the adapter; do not collapse it to the frame flag. Bind the producer and mask rule in the dataset manifest. The full-video feature adapter can provide per-slot validity because the sticky picker retains each slot separately.

Keep missing numerical values zero-filled for compatibility, but carry validity separately. Before the TCN, zero pose and player-position values where `source_frame_valid` or the relevant `pose_valid` slot is false. Zero shuttle coordinates where `source_frame_valid` or `shuttle_valid` is false. The TCN convolutions mix neighbouring frames before any attention mask exists, so this ordering is required. Then add a small learned validity embedding to each pose and shuttle frame after the TCN projection. This preserves the existing TCN input widths and leaves warm-starting possible. It also makes a real coordinate near zero different from missing evidence.

Use validity in attention as follows:

- temporal player keys: `source_frame_valid & pose_valid[:, player]`
- temporal shuttle keys: `source_frame_valid & shuttle_valid`
- player-shuttle interaction keys: valid shuttle frames
- interactional player keys: `source_frame_valid & (pose_valid[:, player] | shuttle_valid)`
- CLS remains valid internally so an entirely missing stream cannot create an all-masked softmax

The all-missing-window test must remain finite. If cross-attention has no valid shuttle key in a sample, replace the centre shuttle representation with one learned missing-shuttle vector and mark only that centre key valid for the cross-attention call. This avoids changing the sequence length or unmasking arbitrary zero frames. It is a real data case around long tracking gaps.

### Backbone and temporal head

Refactor the existing `BST` computation at one clear boundary. Add a private encoder method that returns both the current CLS summaries and aligned frame features. Keep `BST.forward` behaviour and checkpoint keys unchanged where practical.

Add `BSTContactDetector` in a new `src/bst_x/contact/model.py` module. It should:

1. run the existing pose and shuttle TCNs
2. add position and validity information
3. run the current temporal transformer
4. run player-to-shuttle cross-attention
5. run the current interactional transformer separately for each player's player-shuttle sequence
6. apply explicit framewise Aim Player and Clean Gate equations
7. concatenate the Top-player, Bottom-player and cleaned shuttle features at each frame
8. emit `contact_logits` with shape `(batch, 21)`

The current interactional transformer does not directly cross-attend one player to the other. It runs the same encoder independently on the two player-shuttle streams. Preserve that behaviour in the first detector.

Define the framewise Aim Player and Clean Gate operations as the direct temporal analogue of the current CLS equations. Let `p1_tem[t]`, `p2_tem[t]` and `shuttle_tem[t]` be temporal-transformer frame outputs. Let `p1_inter[t]` and `p2_inter[t]` be the corresponding interactional-transformer frame outputs after removing its CLS slot. Then:

```text
p1_conclusion[t] = p1_tem[t] + p1_inter[t]
p2_conclusion[t] = p2_tem[t] + p2_inter[t]
alpha[t] = (cos(p1_inter[t], shuttle_tem[t])
            - cos(p2_inter[t], shuttle_tem[t]) + 2) / 4
p1_conclusion[t] *= ap_factor * alpha[t] + (1 - ap_factor)
p2_conclusion[t] *= ap_factor * (1 - alpha[t]) + (1 - ap_factor)
dirt[t] = mlp_clean(minimum(p1_inter[t], p2_inter[t]))
shuttle_clean[t] = shuttle_tem[t] - cg_factor * dirt[t]
```

Concatenate `p1_conclusion[t]`, `p2_conclusion[t]` and `shuttle_clean[t]` for the temporal head. Reuse the current AP/CG schedule factors. Unit tests should compare this equation with the current CLS equation on synthetic tensors.

The contact head should be `LayerNorm -> Linear -> GELU -> Dropout -> Linear(1)` applied independently at every frame. It shares weights across time. Training gathers index 10 from `contact_logits`; inference also gathers index 10 from each stride-1 window. The other logits are retained for diagnostics and a later dense-region optimisation, but they do not enter the first loss.

This is a centre-supervised, time-aligned contact head because the readout is tied to a source frame. The current classifier's three CLS summaries remain available only to the stroke head.

### Contact and side outputs

Use two outputs:

- `contact_logits`: binary contact evidence at each frame
- `side_logits`: Top versus Bottom at each frame

Compute side loss only when the gathered centre is a positive contact and the annotation has a valid side. At inference:

1. accept an event only when the contact logit passes the held-out threshold
2. emit Top or Bottom only when the side probability is outside a held-out abstention interval
3. otherwise emit the contact with side `None`

Do not force `None`, Top and Bottom into one three-class target. A weak side answer must not suppress a strong contact.

Direct geometry gives **89.0%** side accuracy on temporally matched current-final contacts. On the frozen region-v2 HGB event stream, the same shipped side rule gives **83.7%** conditional side accuracy. The current rally alternation mainly fails after contacts are missed. Contact recall is therefore the primary decision metric. The side head is an auxiliary output and a possible replacement for direct geometry later.

One small head ablation is required in the first pilot:

- contact-only head
- separate contact plus side head with side-loss weight `0.25`

Keep the version with the better held-out contact event F1. If they tie, keep contact-only. A coupled pair of `Top_contact` and `Bottom_contact` logits is a later option only if the separate side head is clearly useful but poorly calibrated.

For side abstention, search confidence cut-offs `{0.70, 0.75, 0.80, 0.90}` on validation. Emit Top above the cut-off, Bottom below one minus the cut-off, and `None` between them. Select the highest side accuracy that retains at least 80% side coverage. Freeze the cut-off before test scoring.

### Initialisation

Train the first contact model from scratch. This isolates the new task and avoids silently cropping 100-frame positional embeddings from a stroke checkpoint.

If the scratch model underfits, run one warm-start ablation. Load compatible TCN and transformer weights from the current BST-X classifier, initialise the 21-frame positional embeddings afresh, and leave validity embeddings and both heads random. Record every skipped key. Do not make warm-starting a prerequisite for the first end-to-end run.

## Contact window preparation

### New package and records

Add a focused package:

```text
src/bst_x/contact/
    __init__.py
    data.py
    labels.py
    model.py
    search.py
    train.py
    evaluate.py
```

`data.py` should define:

- `ContactExample`: dataset, source video, set, rally, contact index, source contact frame, window centre, label status, `clip_stem` and nominal source clip interval
- `extract_consecutive_window`: one shared boundary-padding and mask operation
- `build_contact_streams_from_raw`: full-video raw pose to BST-X joints, positions and per-slot validity
- `ContactWindowDataset`: load prepared shards and return the tensor contract above
- `build_contact_shards`: read each clip once and write deterministic compressed shards

Use the repository compression convention: each tensor shard is an `.npy.xz` written with `lzma`, and metadata is `json.gz`. Store dataset roots only in an ignored run record or pass them as CLI arguments. Tracked prose and manifests should contain dataset names, public video or fixture identifiers, schema versions and hashes, but no private filesystem paths.

### Consecutive centre extraction

For centre frame `c`, the first window is the exact half-open slice `[c - 10, c + 11)`. Do not call `make_seq_len_same`. Do not interpolate, decimate or linspace-sample within a window.

When the slice crosses a clip or video boundary:

- zero-pad the missing side
- set `source_frame_valid` false for the pad
- set both evidence masks false for the pad
- retain centre index 10

Reject a positive training example if its annotated contact cannot be mapped to a decoded source frame. Boundary padding is valid; an unresolved source-to-clip offset is not.

### Frame alignment gate

Before preparing the full dataset:

1. extend `build_hit_frame_lookup` or add a sibling `build_contact_frame_lookup` that records nominal clip start, source contact frame and local contact frame
2. select clips across source videos, frame rates, first/middle/last rally contacts and short clip boundaries; sort each stratum by a hash of `clip_stem` and retain the sample list in the manifest
3. compare decoded clip frames with the corresponding source-video frames
4. determine whether the MoviePy path has a zero-frame or one-frame start offset
5. store the measured offset in the contact dataset manifest
6. fail preparation if the offset is inconsistent within one dataset build

Then verify feature alignment on at least 100 clips:

- pose, position, shuttle and both masks have the same decoded length after the existing tail truncation rule
- the local contact frame lies inside that shared length
- the 21-frame slice uses identical local indices for every stream
- a known shuttle visibility transition occurs at the same local index in raw and prepared data

ShuttleSet and ShuttleSet22 need separate alignment checks. A shared format does not prove that both video-generation runs used the same decoder behaviour.

### Full-video inference features

Training can use the prepared per-clip pose and shuttle files. Fixture and ShuttleSet22 event evaluation need full-video timelines.

`build_contact_streams_from_raw` should accept the same raw contracts used by the current annotator path:

```text
shuttle track     (time, 3)              normalised x, y, visibility
pose keypoints    (time, detections, 17, 2)
pose boxes        (time, detections, 4)
pose scores       (time, detections)
pose counts       (time,)
court_present     (time,)
homography rows, court information and resolution table
scene intervals   half-open source-frame ranges
```

These are the raw pose fields consumed by `build_sticky_result`, plus its full-video shuttle and court evidence. Validate every time dimension before feature construction.

The adapter should reuse the sticky player-picker rules over the court-present tracker segments. Reset the picker at each segment boundary. For every accepted Top or Bottom slot, write:

- bbox-diagonal and centre-aligned joints using the same normalisation as `normalize_joints`
- normalised court position from the same homography projection
- a per-slot pose validity flag

Then derive JnB features with the existing `create_bones` function. Keep the original shuttle `x/y/visibility` without passing it through `get_shuttle_result`, because that function drops visibility.

Test this adapter against `preparing_data.heuristics.sticky_anchor.apply` on a single uninterrupted synthetic segment. The arrays should match for frames where both implementations accept the same slots.

## Labels and examples

### Label states

`labels.py` should define one frame-labelling function over all reviewed contacts in a rally. Every candidate centre receives exactly one of these states:

| Distance to nearest reviewed visible contact | State | Loss |
| --- | --- | --- |
| 0 to 2 base-30 frames | positive | Contact BCE; side BCE when side exists |
| 3 to 5 base-30 frames | ignore | No contact or side loss |
| 6 or more base-30 frames | negative | Contact BCE only |
| Exact contact is offscreen, omitted or unresolved | ignore region | No contact or side loss |

Scale the bands with the existing FPS helpers. Check every centre against every contact in its rally. An offset from one contact is not a negative if it is positive or ignored for another contact.

Resolve overlaps in this order: visible-contact positive, visible-contact ignore band, unresolved-contact ignore interval, then negative. Union all ignore intervals before assigning negatives. A reviewed visible contact therefore remains positive even when an uncertain offscreen interval overlaps it.

Use mean binary cross-entropy over the sampled eligible centres with `pos_weight=1.0`. Log positive and negative counts for each split. Change the loss weight only through a recorded later ablation.

Keep the original contact frame in metadata even when centres within ±2 are labelled positive. This allows peak-offset analysis against the unexpanded annotation.

### Positive examples

For each reviewed visible contact, include centres at offsets `-2, -1, 0, 1, 2` after FPS scaling and de-duplication. Keep all valid positives. Group every centre derived from one real contact under one `contact_group_id` so sampling and split audits can detect leakage.

Do not turn an offscreen or broadcast-omitted serve into a visible-contact target. Mark its uncertain interval ignored. It can still be used to measure whether a search region was proposed around a likely serve, but not to train an exact impact claim.

### Negative mix

Build a deterministic negative pool with four named sources:

- easy negatives: quiet in-rally centres at least 15 base-30 frames from every contact
- known failure negatives: rejected raw impulses, suppressed neighbouring proposals, scene changes, shuttle gap boundaries, tracking resets and strong wrist motion without a contact
- offset hard negatives: centres 6 to 12 base-30 frames before and after a real contact
- search-region negatives: all other eligible centres inside the broadened training regions

For the first run, sample four negative centres per exact contact:

- one easy negative
- one known-failure negative
- two offset or search-region hard negatives

If one pool is empty, record that fact and draw from the other hard-negative pool. Do not silently replace missing hard negatives with broadcast-wide quiet frames.

Represent every sampled centre in source-video coordinates before resolving a prepared clip. When several overlapping stroke clips contain the same 21-frame source window, choose the clip with the greatest context margin, then the lexicographically first `clip_stem` on a tie. Keep one example for that source centre. If no prepared clip contains the complete window, use the full-video feature adapter when its inputs exist; otherwise record and omit the centre.

This rule applies to full-rally easy and search-region negatives as well as positives. It prevents overlapping per-stroke clips from duplicating one physical frame and states how negatives outside a target clip are sourced.

Sample from the training split only. Never use a held-out label to decide which frame is a hard negative. Evaluation scores every eligible centre in a label-blind region, so it does not use sampled negatives.

### Shortcut controls

Prepare these input variants from the same examples and split manifest:

- full physical model: pose, positions, shuttle and validity
- shuttle-only
- pose-plus-position only
- validity-only

The validity-only model is the direct missingness shortcut control. If it finishes within five event-F1 points of the full model on held-out fixtures, treat the full result as shortcut-dependent until a cross-dataset result disproves that concern.

## Search regions and event decoding

### Region seeds

The measured `tree-contact-features/1` contract stores six expanded region channels made from these seeds:

- `current_raw`: the current post-impulse-dedup proposal list, before the wrist gate and final suppression
- `relaxed_impulse`: every finite frame with impulse ratio at least `1.25`
- `wrist`: a local minimum in shuttle-to-wrist distance at no more than `3.0` body heights, using a ±3 base-30 local-minimum radius
- `visibility`: every shuttle visibility transition
- `rally_start`: the first frame of each detected rally span
- `scene_start`: the first frame of each scene interval

Version 1 expands those seeds within each detected rally span by these base-30 radii:

| Seed | Radius |
| --- | ---: |
| current raw | 15 |
| relaxed impulse | 15 |
| wrist | 10 |
| visibility | 15 |
| rally start | 45 |
| scene start | 15 |

Version 1 clears the pooled operational ±10 gate at 98.3% non-serve and 91.4% serve coverage. It fails the per-fixture gate on `sset_21`, at 93.7% and 80.0%. Its hard detected-span boundary is the main restriction.

The measured `tree-contact-features/2` freeze implements region version 2:

- retain the six version-1 seed definitions and radii
- emit eligible rows across court-present, non-replay tracker intervals rather than only inside detected rally spans
- compute relaxed impulse, wrist and visibility seeds across those full eligible intervals
- add a `serve_lookback` channel covering the 45 base-30 frames before every eligible interval
- merge overlapping pre-rolls and clamp them to the source timeline

The backwards pre-roll is important. A forwards-only window cannot recover a serve shown in a close-up immediately before the court view appears. Move this frozen contract into `contact/search.py` rather than maintaining a neural copy.

The final freeze contains 130,624 rows and has SHA-256 `4a5efbd6582701a708270a3b273be2d2572bc3753085ec449b7db815dffec722`. Two independent freezes are byte-identical. At ±10, pooled coverage is 98.4% for non-serves and 97.9% for serves. `sset_21` reaches 93.7% and 94.7%. The HGB physics rerun reaches 84.5% precision, 90.5% recall and 87.4% F1.

The tree's “physics” input includes explicit validity masks. Its search intervals also include the non-court serve pre-roll. A court-view-only refit still reaches 87.7% F1, so boundary context does not explain the baseline. Keep the same validity information available to BST-X, but mask context that crosses a scene, replay or source-video boundary.

Use region version 2 as the shared surface for a classifier pilot. Keep the failed `sset_21` non-serve gate explicit. Do not compare a version-2 BST-X score with version-1 tree predictions.

The 37 uncovered `sset_21` non-serves all have visible shuttle evidence within ±10, but no sticky player analysis and no detected rally span. A whole-video relaxed-impulse diagnostic covers every non-serve and 291 of 292 serves, but searches 90.6% of the broadcasts. That is a ceiling check, not the production region. A final whole-video detector needs a stricter shuttle-only fallback with replay rejection, or an RGB scene path for live close-ups.

The generator must not accept ground-truth rows or a scorer callback. Coverage is measured later in `evaluate.py` after the frozen region file has passed its forbidden-field and provenance checks.

Report two ceilings separately for serves and later contacts:

- strict coverage: the exact reviewed frame is an eligible scored centre
- operational coverage at ±5, ±10 and ±15: at least one eligible scored centre lies within the evaluation tolerance

Use operational ±10 coverage for the per-fixture training gate. Keep strict coverage so the result remains comparable with the current tree scorer.

### Stride-1 scoring

For every merged region:

1. read ten context frames before and after the region so boundary centres receive a full window where video evidence exists
2. score every eligible centre in the original region at stride 1
3. retain the raw centre logit, side logit, frame index, region id and input-validity counts
4. score an overlapping centre once after regions are merged

Context may cross a detected rally-span boundary, because the span is not evidence validity. It may not cross a scene boundary, replay interval or source-video boundary. Clip the context at those hard boundaries, zero-pad the missing side and clear `source_frame_valid` there.

Batch windows on GPU. A later optimisation may run a longer region through the temporal head once, but it must first match stride-1 window logits within a stated tolerance. Window-boundary effects from the TCN and positional embeddings make that optimisation non-trivial.

### Local maxima and temporal NMS

Decode events in this order:

1. keep centre logits above the threshold chosen on validation fixtures
2. keep deterministic local maxima within a small peak radius
3. rank peaks by descending logit, then ascending frame for ties
4. apply temporal NMS with the radius chosen on validation fixtures
5. return accepted events in frame order

A frame is a local maximum when no eligible neighbour within the peak radius has a higher logit. If equal logits form a plateau, keep the earliest frame and drop the other equal frames. At a region edge, compare only eligible neighbours inside the region. Temporal NMS suppresses a later-ranked peak when its absolute frame distance from an accepted peak is less than or equal to the NMS radius. This matches the tree scorer's suppression inequality and frame tie-break.

Tune the threshold, peak radius and NMS radius on the validation fixtures only. Use base-30 radii and scale them per fixture FPS. Search peak radii `{1, 2, 3}` and NMS radii `{3, 5, 7, 9}`. Do not set NMS radius to the 21-frame input width; two real contacts can occur within one context window.

Select the operating point that maximises ±10 event F1 on validation. Freeze it before scoring the held-out fixture. Also report the validation precision-recall curve so a high-recall operating point remains available for the autograder.

## Splits and leakage controls

### Three-fixture development result

The three current fixtures are `sset_01`, `sset_15` and `sset_21`. For the BST-X event result:

- train on ShuttleSet source videos outside all three fixtures
- use two fixtures to select threshold and NMS settings
- hold the third fixture out for the reported fold
- rotate the held-out fixture three times

This gives three grouped test folds without training on a different clip from the test source video. Pool event counts across the three test folds. Do not average the three recalls.

Every split check must enforce:

- one source video belongs to one split
- one match belongs to one split
- all windows from one contact belong to one split
- all contacts and negatives from one rally belong to one split
- duplicated or re-encoded videos share one group
- normalisation statistics, sampler weights, thresholds and NMS settings come from training or validation only

Player overlap may remain because professional players recur across tournaments. Report player overlap, but do not move matches after looking at test performance.

### ShuttleSet22 generalisation

Preferred use:

1. develop the data path, model and event decoder on ShuttleSet
2. freeze architecture, masks, label bands, region rules, threshold selection rule and NMS search set
3. train the chosen model on all eligible ShuttleSet development videos, retaining whole-match validation videos for early stopping and threshold selection
4. evaluate once on ShuttleSet22

The ShuttleSet22 score must use whole-match or source-timeline evidence. Prepared positive stroke clips alone are not enough for event precision because they do not test label-blind region generation or false peaks between contacts.

Run one bounded remote inventory before implementation. Record only:

- public match or fixture identifiers
- annotation schema and FPS distribution
- which full videos, shuttle tracks, raw pose arrays and filtered pose arrays exist
- content hashes or counts needed to bind the run

Keep exact data paths and environment-specific access commands in the ignored `local_agents/implementation_paths.md`. Do not put them in tracked documents.

If only per-clip ShuttleSet22 pose is prepared, choose and freeze a whole-match test subset before extracting the missing full-timeline pose or shuttle evidence. Do not select the subset after seeing model results.

Two fallbacks remain valid, but they answer weaker questions:

- If ShuttleSet alone is too small or too imbalanced, train on ShuttleSet plus a predeclared ShuttleSet22 development partition while keeping a whole-match ShuttleSet22 test partition untouched.
- If full-timeline ShuttleSet22 extraction cannot finish, report centre classification on held-out ShuttleSet22 clips as a data-path check. Do not call it event generalisation.

## Metrics and retained outputs

Use one-to-one event matching at ±5, ±10 and ±15 base-30 frames. Report:

- search-region ceiling before model scoring
- serve recall and non-serve recall separately
- pooled event precision, recall and F1 after NMS
- accepted predictions per rally and false predictions per minute of active rally time
- absolute peak offset from the reviewed contact
- side accuracy and side-answer coverage, conditional on a temporally matched event
- metrics per fixture and dataset, plus pooled counts
- full, shuttle-only, pose-only and validity-only controls
- one-seed pilot and three-seed mean and range for the chosen configuration

Retain raw per-centre logits and decoded events in compressed, ignored artefacts. Each run manifest should bind the code commit, dataset manifest, split groups, seed, window width, label bands, feature variant, model settings, threshold and NMS settings.

Compare BST-X with unchanged histogram boosting and random forest on the same frozen version-2 regions and held-out fixture folds. Do not tune the random forest further.

The version-2 HGB physics rerun is complete. The minimum BST-X acceptance target at ±10 is therefore:

```text
required F1 = max(89.9%, version-2 HGB physics F1 + 2.0 points)
required precision = max(87.5%, version-2 HGB physics precision)
```

Also report against the version-2 HGB physics breakdown: 90.5% overall recall, 92.9% non-serve recall and 67.5% serve recall. The key comparison is event precision and recall at the selected ±10 operating point, not centre-row accuracy.

## Implementation stages

### Stage 0: bind data and frame alignment

Files:

- existing `src/bst_x/validation_scripts/hit_frame_lookup.py`
- new `src/bst_x/contact/data.py`
- new `tests/test_bst_contact_data.py`

Work:

- add the source-frame, clip-start and local-centre record
- measure the decoder offset for ShuttleSet and ShuttleSet22
- define the 21-frame extraction contract and validity tensors
- write a dry-run manifest and counts without model tensors

Gate:

- alignment sample passes across both datasets
- no inconsistent decoder offset
- no positive centre outside the aligned feature length
- split and duplicate-video checks pass

Stop here if alignment is unresolved.

### Stage 1: build labelled contact shards

Files:

- `src/bst_x/contact/data.py`
- new `src/bst_x/contact/labels.py`
- `tests/test_bst_contact_data.py`
- new `tests/test_bst_contact_labels.py`

Work:

- build consecutive windows and masks
- implement label bands and negative pools
- write deterministic compressed shards and provenance
- add the full-video feature adapter

Gate:

- rebuild with the same seed gives the same manifest and row identities
- positives, ignored centres and negatives match hand-built examples
- adjacent contacts cannot create false hard negatives
- raw-to-filtered feature adapter matches the existing heuristic on a fixed segment

### Stage 2: add the temporal detector

Files:

- existing `src/bst_x/model/bst.py`
- new `src/bst_x/contact/model.py`
- existing `src/bst_x/bst_x_common.py` only if a shared constructor is genuinely useful
- new `tests/test_bst_contact_model.py`

Work:

- expose aligned frame features without changing current classifier outputs
- add validity embeddings and safe attention masks
- add temporal contact and optional side heads
- add centre-logit gathering

Gate:

- current BST classifier shape and a fixed forward output remain unchanged
- contact output shape is `(batch, 21)`
- all-visible, partly missing and entirely missing windows remain finite
- changing padded values under a false validity mask does not change the centre logit
- a tiny synthetic dataset can be overfit

### Stage 3: train and score events

Files:

- new `src/bst_x/contact/train.py`
- new `src/bst_x/contact/search.py`
- new `src/bst_x/contact/evaluate.py`
- new `tests/test_bst_contact_search.py`
- new `tests/test_bst_contact_training.py`

Work:

- port and verify the measured region-version-2 contract before launching a BST-X job
- bind the pilot to the retained region SHA and HGB baseline
- train one contact-only and one contact-plus-side pilot
- score merged search regions at stride 1
- select local maxima and apply temporal NMS
- run three grouped fixture folds
- run physical input and shortcut controls

Gate:

- the region identity and measured ceiling match the retained version-2 result
- the pilot report states that `sset_21` remains below the non-serve coverage gate
- no full multi-seed or whole-video claim proceeds until the off-court search path is resolved
- the scorer reproduces synthetic one-to-one event counts at all three tolerances
- threshold and NMS selection never read the held-out fixture
- deterministic tie handling and overlapping-region de-duplication pass
- raw centre logits can be re-decoded without rerunning inference

### Stage 4: run the ShuttleSet22 generalisation test

Start this stage only after the off-court search path clears the coverage gate, or after the user explicitly accepts a bounded inside-region generalisation result.

Files:

- dataset adapter additions in `src/bst_x/contact/data.py`
- result-only changes in the approved report location
- tests only where ShuttleSet22 exposes a real schema difference

Work:

- bind the untouched ShuttleSet22 match list
- complete any predeclared full-timeline extraction
- run the frozen region, scoring and NMS path
- compare per-dataset failures and missingness rates

Gate:

- no ShuttleSet22 label entered model selection or threshold tuning
- event metrics come from source timelines, not isolated positive clips
- missingness-only control stays materially below the full model

### Stage 5: production decision

Do not integrate BST-X into `annotator` during these experiments. Decide after comparing the tree and BST-X results.

Proceed to integration only if BST-X gives a useful gain at comparable precision and the gain remains on ShuttleSet22. At that point, define one stable inference interface that returns frame, contact logit, optional side and provenance. Rally alternation should remain downstream and should be reassessed only after the missing-contact rate improves.

## Tests

The focused suite should cover these cases.

Data and labels:

- centre index and exact consecutive source indices
- left and right boundary padding
- pose failure with a still-visible shuttle
- invisible shuttle with non-zero stale coordinates
- one valid player slot and two valid player slots
- clip-tail truncation shared across all streams
- positive, ignore and negative band boundaries at 25 and 30 fps
- two contacts close enough for their bands to overlap
- offscreen or unresolved contact ignore intervals
- stable negative sampling and split grouping

Model:

- contact and side output shapes
- no NaN for an entirely missing stream or window
- masked-value invariance
- centre readout uses the frame token rather than the CLS token
- contact-only checkpoint loads without a side head
- optional warm-start reports all incompatible keys
- current stroke classifier regression test

Search and evaluation:

- region creation has no label argument
- region schema, seed thresholds and expansion radii match the frozen version
- region merging and video-boundary clamping
- stride-1 coverage with no duplicate centres
- local-max tie behaviour
- NMS keeps the stronger peak and preserves contacts outside its radius
- one-to-one matching at ±5, ±10 and ±15
- threshold and NMS choices use validation groups only
- pooled counts are sums rather than fold-average recalls

Run the repository gates after each coherent implementation commit:

```bash
ruff check .
uvx --from pyrefly==1.1.1 --with jaxtyping==0.3.11 pyrefly check -c pyproject.toml
pytest
```

The repository's current baseline failures remain in the ignored worklog. Report them separately from new failures.

## Compute and remote use

Laptop work is enough for:

- schema and alignment inspection
- shard-builder tests on a small subset
- synthetic model tests
- search-region ceiling and event scoring from frozen logits
- tree comparison and report generation

Use a GPU server for:

- a full ShuttleSet contact-shard build if source arrays are remote
- full BST-X training
- stride-1 inference over all fixture regions
- any missing ShuttleSet22 full-timeline pose or shuttle extraction

The model is small enough that the first 100-example overfit and one-batch forward can run on CPU. The full experiment belongs on the GPU data environment because the data are already there and stride-1 windows multiply inference rows.

Use one bounded remote command per stage. Let the training command run to completion and retrieve its manifest and outputs once. Do not poll. Keep private paths and command details in the ignored `local_agents/implementation_paths.md`.

## Failure and stop criteria

Stop before full model training when:

- any fixture's version-2 regions remain below 97% non-serve or 90% serve operational coverage at ±10
- source-to-clip frame alignment is inconsistent
- more than 1% of otherwise eligible positive contacts fall outside aligned prepared arrays
- ShuttleSet and ShuttleSet22 masks have incompatible meanings that the adapter cannot state explicitly

The measured version-2 region passes the serve gate but fails the non-serve gate on `sset_21`. One bounded classifier pilot is still useful. Label it as an inside-region comparison, and do not use it for a whole-video claim or a full multi-seed run.

Stop or redesign the model when:

- the model cannot overfit a small clean subset
- any missing-data case produces NaN
- centre peaks have a consistent offset that survives the alignment audit
- validity-only performance is within five event-F1 points of the full input
- BST-X misses the version-2 acceptance formula: at least `max(89.9%, HGB F1 + 2.0 points)` F1 and `max(87.5%, HGB precision)` precision at ±10

Do not claim cross-dataset generalisation when:

- ShuttleSet22 was used to choose architecture, thresholds or NMS
- only positive clips were scored
- held-out ShuttleSet22 event F1 drops by more than ten points from the pooled ShuttleSet fixture result at a comparable operating point
- the gain disappears under the ShuttleSet22 missingness-only control

A failed neural result is still useful. Keep the tree if it gives the same event quality with a simpler runtime.

## Expected commits

These exact messages are proposals only. Every commit still requires the user's approval of its exact message before it is created.

1. `Plan the BST-X contact detector`
2. `Add BST-X contact window preparation`
3. `Add the temporal BST-X contact detector`
4. `Add BST-X contact training and event scoring`
5. `Evaluate BST-X contact detection on ShuttleSet22`

Keep each commit limited to the stage named by its message. Do not mix the existing stroke classifier refactor with training or evaluation changes.

## First-run recipe

This is the first real run after Stages 0 to 3 exist.

1. Reuse the verified region-version-2 contract and port it into `contact/search.py`. The isolated tree freeze has already passed its label-blind and reproducibility gates.
2. Preserve the measured coverage warning: `sset_21` has 93.7% non-serve and 94.7% serve coverage at ±10. Use the pilot only to compare classifiers inside that region.
3. Build 21-frame ShuttleSet shards from source videos outside all three fixtures. Use positive ±2, ignore through ±5 and four negatives per exact contact.
4. Run the alignment report and inspect its failed-row list. It must be empty or contain only explicitly ignored unresolved contacts.
5. Run the CPU 100-example overfit for the full model and validity-only control.
6. On one GPU, train one seed of the contact-only model and one seed with the separate side head. Use identical examples, optimiser settings and early-stopping groups.
7. Keep contact-only on a tie. Train the selected head for three seeds only after the one-seed event scorer works end to end.
8. For each held-out fixture, choose threshold and NMS from the other two fixtures, decode events, then score ±5, ±10 and ±15.
9. Run the full, shuttle-only, pose-only and validity-only controls on the same folds.
10. Compare pooled BST-X events with the version-2 trees. Require the acceptance formula: F1 at least `max(89.9%, HGB F1 + 2.0 points)` and precision at least `max(87.5%, HGB precision)`.
11. If BST-X passes the gain and shortcut gates, freeze the configuration and run ShuttleSet22 once.

Suggested CLI shape after implementation:

```bash
PYTHONPATH=src:src/bst_x python -m contact.data \
    --dataset shuttleset \
    --window-frames 21 \
    --positive-radius-base30 2 \
    --ignore-radius-base30 5 \
    --exclude-fixtures sset_01,sset_15,sset_21 \
    --output-root <ignored-contact-data-root>

PYTHONPATH=src:src/bst_x python -m contact.train \
    --manifest <ignored-contact-data-root>/manifest.json.gz \
    --feature-set full \
    --side-head off \
    --seed 1 \
    --run-dir <ignored-run-dir>

PYTHONPATH=src:src/bst_x python -m contact.evaluate \
    --run-dir <ignored-run-dir> \
    --regions <ignored-region-v2-freeze> \
    --fixture-folds sset_01,sset_15,sset_21 \
    --tolerances-base30 5,10,15
```

The final flag names can change during implementation. The data, split and scoring contracts should not.

## Approval checklist

Before implementation starts, approve or change these choices:

- 21 consecutive frames with centre index 10
- positive ±2 and ignore through ±5 base-30 frames
- region version 2 with a 45-base-30-frame pre-roll before eligible court-view intervals
- one bounded classifier pilot despite the known `sset_21` non-serve gate failure; full training remains gated
- the BST-X target formula, with a current minimum of 89.9% F1 and 87.5% precision at ±10
- separate contact and optional Top/Bottom heads
- contact-only versus contact-plus-side as the first head ablation
- ShuttleSet22 held out for the first full event test
- the five proposed exact commit messages

The main unresolved external-data question is whether the prepared ShuttleSet22 material includes full-match timelines or only per-stroke clips. A bounded server inventory can answer it without changing the design. Stage 0 must also bind ShuttleSet22 column names and mask meanings in its adapter manifest. Full-match evidence is required for the claimed external event result.
