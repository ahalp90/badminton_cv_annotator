# BST-X 3D pose stream: design and revival

Revival record for the dual 2D+3D pose extraction path (`use_3d_pose`).
The next commit strips this feature out of the tree in favour of a
2D-only pipeline. This doc captures what the 3D path did and every seam
it touched, so re-adding it later is a wiring job rather than an
archaeology dig.

Written against the live code as it stands the moment before removal.
Line numbers are as-read at that point; find everything by symbol, the
line numbers are just a starting nudge.

**Why it's going.** No recorded run ever used it. `use_3d_pose` is
present in all 64 `manifest.yaml` files under `experiments/bst_x/`, and
its value is `false` in all 64 and `true` in none. The 3D path carried
a whole parallel MMPose extraction, an `in_channels` fork, and two
naming tags, none of which any experiment exercised. It was scaffolding
for a stream we never trained, so it comes out until we actually want it.

---

## 1. What the 3D extraction did

The pipeline's Step 1 turns clips into per-clip pose `.npy` files. In 2D
that means COCO 17-joint keypoints as `(x, y)` per joint; the 3D path
did the same job but emitted `(x, y, z)` per joint instead. Everything
downstream (collation, the pose-style bone derivations, the model's
input projection) then carried one extra channel, at least on paper;
two gaps in that story are flagged below and in §4.

### `detect_players_3d`

`detect_players_3d` in
`src/bst_x/preparing_data/prepare_train_on_shuttleset.py` (~L217) was the
3D twin of `detect_players_2d`. Per frame it ran two MMPose generators
side by side: the 2D inferencer to locate and order the two on-court
players, and a 3D inferencer to lift their keypoints into `xyz`. It
reused the shared per-frame decision `_order_two_on_court` (which is
2D-only by design: the court projection needs pixel coordinates, so the
3D caller still hands it the 2D keypoints), then stored the 3D keypoints
for whichever two players `_order_two_on_court` picked. Output shape was
`(t, m, J, xyz)` for joints and the usual `(t, m, xy)` for court
positions.

Two things the 2D path does that this one didn't, worth knowing before
any revival. First, no normalisation: `detect_players_2d` passes its
picked keypoints through `normalize_joints` (bbox-anchored, and 2D-only
by its `'m j 2'` annotation), while the 3D path appended
`keypoints_3d[in_court_pid]` raw, so the stored values sat on MMPose's
own 3D coordinate scale, nothing like the normalised 2D features. Second,
the slice assumes alignment: `in_court_pid` indexes the 2D detection
list, but it's used to slice the 3D generator's output, and nothing
checks that the two inferencers detected the same people in the same
order. Fewer 3D detections than 2D would crash with an IndexError; a
different ordering would silently swap one player's joints for the
other's. Neither ever fired in practice because the path never ran end
to end.

### The per-call MMPoseInferencer reload

The one odd bit worth preserving. `detect_players_2d` takes its
inferencer as a parameter: it's built once in the wrapper and reused
across every clip. `detect_players_3d` took the 2D inferencer the same
way, but built the 3D one fresh inside the function on every call:

```python
inferencer_3d = MMPoseInferencer(pose3d="human3d")
```

That rebuild reloads the 3D model weights from disk on every single clip,
which is slow across ~33k clips. It was a deliberate workaround, not an
oversight: the original author found that threading the 3D inferencer in
as a parameter (the way the 2D one is passed) tripped an MMPose bug. So
the 3D inferencer got rebuilt per clip to sidestep it.

The scaffolding for the fast path was left in place, commented out: the
`# inferencer_3d: MMPoseInferencer,` parameter on `detect_players_3d`,
and the matching `# pose_inferencer_3d = MMPoseInferencer(pose3d='human3d')`
build plus `# "inferencer_3d": ...` kwarg in the wrapper. If MMPose ever
fixes the bug, the intended move was to uncomment those, build the 3D
inferencer once in the wrapper, pass it in like the 2D one, and drop the
per-call rebuild.

### `prepare_3d_dataset_npy_from_raw_video`

`prepare_3d_dataset_npy_from_raw_video` (~L390) was the 3D wrapper. It
built the 2D inferencer and handed `detect_players_3d` to a shared
`_prepare_dataset_from_raw_video` helper that owned the per-clip loop: skip
if the `_failed.npy` resume marker already exists, otherwise call the
passed-in detect function, save the three `.npy` outputs (`_pos`,
`_joints`, `_failed`), and free GPU memory. The 2D wrapper handed
`detect_players_2d` to the same helper, so the `detect_fn` parameter is
what let one loop serve both streams:

```
def _prepare_dataset_from_raw_video(
    my_clips_folder, save_root_dir, detect_fn, detect_kwargs,
):
    save_root_dir.mkdir(parents=True, exist_ok=True)
    all_mp4_paths = sorted(my_clips_folder.glob("**/*.mp4"))
    for video_path in tqdm(all_mp4_paths, desc="Yield .npy files", unit="video"):
        save_branch = str(save_root_dir / video_path.stem)
        if not Path(save_branch + "_failed.npy").exists():
            failed_ls, players_positions, joints = detect_fn(
                video_path=video_path, **detect_kwargs,
            )
            np.save(save_branch + "_pos.npy", players_positions)
            np.save(save_branch + "_joints.npy", joints)
            np.save(save_branch + "_failed.npy", np.array(failed_ls, dtype=bool))
            gc.collect()
            torch.cuda.empty_cache()
```

The structure-and-guards pass inlined this shared helper into the single
public `prepare_dataset_npy_from_raw_video` (the old 2D wrapper, renamed)
once the 3D stream's second caller was gone, so the `detect_fn` seam no
longer exists. A revival has to re-open it; see the §4 checklist. Only the
3D wrapper and `detect_players_3d` leave here.

---

## 2. How the `use_3d_pose` flag fanned out

The flag started as one `Hyp` field defaulting False and reached into
four places: a collated-dir name tag, a weight-file name tag, the
`in_channels` choice, and the extraction dispatch. Here's each.

### The `Hyp` field

`use_3d_pose: bool = False` on the `Hyp` NamedTuple in
`src/bst_x/bst_x_train.py` (~L86). Because the manifest is written from
`config_payload = dict(hyp._asdict())` (~L1319), this field is what put
`use_3d_pose:` into every run's `manifest.yaml`. Drop the field and new
manifests stop recording the key, which is the source of the infer-side
back-compat note in the checklist below.

### `in_channels`: 3 for 3D, 2 for 2D

This is the fork that matters, and the reason it's `3` for 3D is just the
channel count: 3D keypoints carry `(x, y, z)`, three numbers per joint,
against 2D's `(x, y)`, two numbers. The model's input width is
`in_dim = (n_joints + n_bones) * in_channels` in `build_bst_x_network`
(`src/bst_x/bst_x_common.py`, ~L80). Build the net with `in_channels=3`
and the first input projection is sized to read the 3-wide-per-joint
feature vectors the 3D npys produce; build it with `2` and a 3D npy
wouldn't line up. Note the model itself (`src/bst_x/model/bst.py`) takes
the already-computed `in_dim`, not `in_channels`, so there's no
3D-conditional plumbing inside the model; the fork lives entirely in the
`in_dim` calculation.

`in_channels` was threaded through every place that builds the network:

- `Task.get_network_architecture` in `bst_x_train.py` (~L924), called at
  the train site as `in_channels=(3 if hyp.use_3d_pose else 2)` (~L1359)
- `build_bst_x_network` in `bst_x_common.py` (~L63), the shared builder
- the inference-engine `get_network_architecture` in
  `src/bst_x/bst_x_infer.py` (~L109, default 2)
- `dump_run_predictions` in `bst_x_infer.py` (~L229),
  `in_channels=(3 if config['use_3d_pose'] else 2)`
- `_build_model` in `src/api/bst_x_inference.py` (~L100, hardwired 2)

### The collated-dir `3d_` tag

`derive_npy_collated_dir_basename` in `src/bst_x/pipeline/config.py`
(~L384) formats the collated dir as `npy_[3d_][seq{N}_]{split}_{id}`. The
`3d_` slice comes from `three_d_tag = '3d_' if use_3d_pose else ''`
(~L394). This is a shared writer/reader contract: `prepare_train` writes
the dir, and both `bst_x_train` and the infer fallback
(`_resolve_collated_dir`) re-derive the same basename to read it, so the
`use_3d_pose` argument has to stay in step across all three call sites.
Beyond that runtime trio, three more files pass the kwarg and need the
threading on revival: `validation_scripts/verify_bst_x_train_target.py`,
`tests/test_taxonomy.py` (the parametrised basename tests) and
`tests/test_remote_preflight.py`.
Every collated dir on disk was written 2D, so none carry the tag.

### The weight-name `_3d` tag

In `bst_x_train.py` (~L1268), `str_3d = '_3d' if hyp.use_3d_pose else ''`
fed the weight-file name via
`model_info_parts.append(f'{CLIP_WINDOW}_seq_100{str_3d}')`, with a dead
`elif hyp.use_3d_pose: model_info_parts.append('3d')` for the
non-seq-100 case. Every weight file on disk is 2D, so none carry `_3d`;
dropping the tag renames nothing that exists.

### Touch-point table

| Site | File | Symbol / line |
|---|---|---|
| `Hyp` field | `bst_x_train.py` | `use_3d_pose: bool = False` (~L86) |
| manifest write | `bst_x_train.py` | `dict(hyp._asdict())` (~L1319) |
| dir-tag derive call | `bst_x_train.py` | `derive_npy_collated_dir_basename(use_3d_pose=...)` (~L1259) |
| weight-name tag | `bst_x_train.py` | `str_3d` block (~L1268-1273) |
| `in_channels` train fork | `bst_x_train.py` | `get_network_architecture(..., in_channels=(3 if ...))` (~L1359) |
| `in_channels` param | `bst_x_train.py` | `get_network_architecture(self, ..., in_channels=2)` (~L924) |
| `in_channels` builder | `bst_x_common.py` | `build_bst_x_network(..., in_channels)`, `in_dim` (~L63, L80) |
| infer config read | `bst_x_infer.py` | `use_3d_pose=config['use_3d_pose']` (~L169) |
| infer `in_channels` fork | `bst_x_infer.py` | `in_channels=(3 if config['use_3d_pose'] else 2)` (~L229) |
| infer `in_channels` param | `bst_x_infer.py` | engine `get_network_architecture(..., in_channels=2)` (~L109) |
| api `in_channels` | `src/api/bst_x_inference.py` | `_build_model(..., in_channels=2)` (~L100) |
| dir-tag helper | `pipeline/config.py` | `derive_npy_collated_dir_basename`, `three_d_tag` (~L384, L394) |
| extraction | `prepare_train_on_shuttleset.py` | `detect_players_3d` (~L217), `prepare_3d_dataset_npy_from_raw_video` (~L390) |
| CLI | `prepare_train_on_shuttleset.py` | `--use-3d-pose` arg + dispatch (see §3) |

---

## 3. The CLI surface

`prepare_train_on_shuttleset.py` exposed the flag on its argument parser:

```python
parser.add_argument(
    "--use-3d-pose",
    action="store_true",
    help="Use 3D pose estimation instead of 2D",
)
```

From there `args.use_3d_pose` drove three things:

- **The default flat-dir name.** `str_3d = "_3d" if args.use_3d_pose else ""`
  (~L985) tagged the default per-clip output dir,
  `dataset{str_3d}_npy_{CLIP_WINDOW}_flat` (~L1017-1020)
- **The collated-dir derive.** `use_3d_pose=args.use_3d_pose` passed into
  `derive_npy_collated_dir_basename` (~L1011), the writer half of the
  shared naming contract
- **The dry-run echo.** `print(f"  use_3d_pose:      {args.use_3d_pose}")`
  (~L1033)

The Step 1 dispatch branched on it (~L1058): `if args.use_3d_pose:` ran
`prepare_3d_dataset_npy_from_raw_video`, else the 2D wrapper. The
`--use-3d-pose` help text and the `npy_[3d_]...` / `dataset[_3d]...`
mentions in the `--collation-id` and `--clip-npy-dir` help strings
described the tag as well.

---

## 4. Re-adding it: the checklist

Post-removal state to re-wire against: `in_dim` is hardwired to
`(n_joints + n_bones) * 2`, there is no `in_channels` parameter anywhere,
and `derive_npy_collated_dir_basename` has no `3d_` tag. A revival needs
to re-open each of these seams:

- Re-add `use_3d_pose: bool = False` to the `Hyp` NamedTuple in
  `bst_x_train.py` so it serialises back into new manifests
- Re-add the `in_channels` parameter to `build_bst_x_network` in
  `bst_x_common.py` and restore `in_dim = (n_joints + n_bones) * in_channels`
- Re-thread `in_channels` through `Task.get_network_architecture` and its
  train call site (`3 if hyp.use_3d_pose else 2`)
- Re-thread `in_channels` through the infer path: the engine
  `get_network_architecture` and `dump_run_predictions` in
  `bst_x_infer.py`, and `_build_model` in `api/bst_x_inference.py`
- Re-add the `3d_` tag to `derive_npy_collated_dir_basename` and re-thread
  `use_3d_pose` to the writer (`prepare_train`) and both readers
  (`bst_x_train`, `_resolve_collated_dir`)
- Re-add the `_3d` weight-name tag in `bst_x_train.py`
- Re-add `detect_players_3d` and `prepare_3d_dataset_npy_from_raw_video`
  in `prepare_train`. The shared `_prepare_dataset_from_raw_video` seam is
  gone (inlined into `prepare_dataset_npy_from_raw_video` by the
  structure-and-guards pass), so first re-extract the per-clip loop back
  into a `detect_fn`-parameterised helper (§1) or fork the merged function;
  the `_order_two_on_court` seam is still present. Carry the per-call
  inferencer-reload workaround (§1) unless MMPose has since fixed the bug
- Re-add the `--use-3d-pose` CLI arg, its Step 1 dispatch branch, the
  `str_3d` default-dir tag, and the dry-run echo

Things to check that removal doesn't tell you about:

- **3D normalisation.** The 3D path stored raw MMPose coordinates;
  `normalize_joints` is 2D-only and never had a 3D counterpart. Decide a
  3D normalisation scheme before training on the stream, or the joint
  features arrive on a scale unrelated to everything the 2D path produces
- **2D/3D detection alignment.** `keypoints_3d[in_court_pid]` reuses the
  2D detection indices on the 3D generator's output. The two inferencers
  are independent models, so associate their detections explicitly
  (bbox or projection matching) rather than assuming same-people,
  same-order
- **Tolerant infer reads.** Any manifest written between removal and
  revival won't carry `use_3d_pose`. If the infer-side reads go back to a
  bare `config['use_3d_pose']`, they'll `KeyError` on those manifests.
  Use `config.get('use_3d_pose', False)` at the two `bst_x_infer.py` read
  sites, or accept that only manifests written after the revival will load
  on the derive-fallback path
- **Downstream channel width.** The 3D npys carry a 3-wide channel through
  collation and the pose-style bone/limb derivations
  (`pad_and_derive_pose_styles` and friends). `in_dim` already accounts
  for the extra channel via `in_channels`, so it flows through as long as
  the bone derivation is channel-agnostic. Confirm that on revival rather
  than assuming it; the 3D path was never run end-to-end, so this leg is
  untested in practice
