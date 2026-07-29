# Recipe: carry the inpaint flag into the track write-out

Written 2026-07-22 for the TODO item "Inpaint fill flag into the
track write-out" (Curtis). Why this matters:
../inpaint_fabrications_investigation.md. Code lines pin to the
wt_annotator worktree at commit d04a789. Every claim below was
verified against that tree this session.

## What exists today

The mask that knows which frames are fills already exists at run
time and dies before any writer sees it:

- predict.py builds it over the whole video:
  `tracknet_pred_dict['Inpaint_Mask'] = generate_inpaint_mask(...)`
  (predict.py:260; the dict initialises the key empty at :159)
- one line later the output dict is built without the key
  (predict.py:261), so the CSV writer never receives it
- the CSV writer's non-training branch writes Frame/Visibility/X/Y
  only (utils/general.py:368-373)
- downstream, extract_shuttle.py reads those four columns into a
  dense NPZ (src/bric/preprocessing/extract_shuttle.py:140-165),
  and shuttle_extractor.py selects X/Y/Visibility into the (t, 3)
  npy (src/bst_x/pipeline/shuttle_extractor.py:260-274)

## The change, step by step

Step 0, two copies. predict.py and utils/general.py exist in BOTH
src/bst_x/TrackNetV3/ and src/bric/perception/_vendor/tracknetv3/,
byte-identical today. Apply every vendored edit to both and diff
the copies afterwards to prove they stayed identical.

Step 1, carry the mask to the writer (predict.py). After the
inpaint loop and before the write call (predict.py:348-357):

```
if inpaintnet is not None:
    assert len(inpaint_pred_dict['Frame']) == len(tracknet_pred_dict['Inpaint_Mask'])
    inpaint_pred_dict['Inpaint_Mask'] = tracknet_pred_dict['Inpaint_Mask']
else:
    tracknet_pred_dict['Inpaint_Mask'] = [0] * len(tracknet_pred_dict['Frame'])
```

The assert is the alignment guard. The inpaint output has one row
per tracknet row in the same order (verified on pilot: 154,393 in,
154,393 out), and the mask is indexed by that same row order. A
tracknet-only run has no fills, so its mask is all zeros.

Step 2, write the column (utils/general.py:368-373). Add
`'Inpaint_Mask': pred_dict['Inpaint_Mask']` to the non-training
DataFrame. The CSV gains a fifth column on every run: 1 on filled
frames, 0 everywhere else.

Step 3, the dense NPZ lane (extract_shuttle.py:140-165). Read the
new column beside the other four, scatter it through the same
clip-frame to source-frame re-keying as visibility, and add
`inpaint=` to the np.savez_compressed call. Old NPZs simply lack
the key, which is what the step-5 decision handles.

Step 4, the (t, 3) npy lane (shuttle_extractor.py:260-274). The
flag joins the array per the TODO ruling: select
`['X', 'Y', 'Visibility', 'Inpaint_Mask']` and save (t, 4) as
x, y, visibility, inpaint. The `drop_duplicates('Frame')` step runs
before the selection, so the mask stays row-aligned for free.
normalize_shuttlecock passes extra columns through untouched (it
divides columns 0 and 1 only; shuttle_extractor.py:54-68), so only
its docstring needs updating to mention the fourth column.

Step 5, the compatibility decision. NEEDS ARIEL'S RULING before
build. Existing caches are (t, 3) npys and mask-less NPZs. Two ways
to accommodate readers:

- shim: readers accept both shapes, and a missing mask means all
  zeros. Old caches keep working, with fills invisible there, which
  is exactly today's behaviour
- regenerate: re-track the fixtures with the new writer (about 22
  minutes per video at 288p nonoverlap, per
  ../../pilot_track_npy/README.md) and re-pin whatever recorded
  numbers move

The shim is the minimal-change reading of the TODO. For existing
nonoverlap caches there is also a no-retrack retrofit: the
signature filter in probe_vs_track.py (this directory) recovers the
mask for fully-fabricated windows.

Step 6, the consumer sweep. Before shipping, find every reader of
the two artefact shapes: PyCharm find-references on the writers
(the np.save in shuttle_extractor.py, the np.savez_compressed in
extract_shuttle.py) and on their load sites; grep alone misses
re-exports. Known consumers to check first: the auto-annotator's
run_video shuttle input, the BST-X dataloader lane, and the
measurement harnesses under measurements/ (frozen harnesses are
records: note, do not fix).

Step 7, gates. ruff, whole-project pyrefly, full pytest with
ANNOTATOR_FIXTURES_ROOT set, then the capture. Under the shim the
capture must stay byte-identical, because nothing reads the flag
yet. The moment a consumer starts reading it, that is a behaviour
change with its own commit and re-pin, per the measured-change
protocol.

## Out of scope here

- what consumers DO with the flag: per-rule design work, sketches
  in ../inpaint_fabrications_investigation.md
- the upstream contribution: upstream_issue_draft.md beside this
  file
- the stride=1 vs stride=8 extract question: its own TODO item
