# The inpaint fill-mask sidecar

Single source of truth for the sidecar files that TrackNetV3 extraction
writes beside its shuttle CSVs from commit 9475036 onward. It covers
what the files mean, how to consume them, and the settled boundary
choices.

Contents: [Why it exists](#why-it-exists) |
[What gets written](#what-gets-written) |
[Reading a sidecar correctly](#reading-a-sidecar-correctly) |
[Provenance manifests](#provenance-manifests-sourcestoml) |
[Boundary choices](#boundary-choices-settled-do-not-relitigate) |
[Landscape notes](#landscape-notes-from-the-build) |
[Verification record](#verification-record)

## Why it exists

TrackNetV3 detects the shuttle per frame and writes a per-frame CSV
(`Frame, Visibility, X, Y`). Its companion model InpaintNet invents
positions for frames the detector missed, and those inventions are
saved with Visibility = 1. An invented position is therefore
indistinguishable from a real detection in every saved artefact. On the
three whole-video reference tracks the raw fill mask covers 46-53% of
frames. The mask that knows which frames were filled existed only in
memory inside `predict.py` and was discarded before saving. The sidecar
saves it.

The mask is exact provenance, not a guess. `predict.py` computes the
saved coordinates as `coor_inpaint * mask + coor_pred * (1 - mask)`:
where the mask is 1 the saved value is InpaintNet's output, where it is
0 the detector's own value passes through untouched.

## What gets written

One gzipped JSON file beside each `{stem}_ball.csv`, named
`{video_stem}_stride{N}_inpaint_mask.json.gz`. The writer is
`src/shared/tracknetv3/write_inpaint_metadata.py`. It runs inside
`predict_video`, immediately before the CSV write. Both callers of that
function therefore produce sidecars: standalone `predict.py` and
`batch_predict.py`, which the extract pipeline's `shuttle_extractor.py`
drives.

Example, with every field:

    {
        "schema": "inpaint_fill_mask/1",
        "index_space": "frame",
        "inpaint_status": "applied",
        "n_rows": 154393,
        "eval_mode": "nonoverlap",
        "stride": 8,
        "th_h_px": 14.4,
        "tracknet_ckpt": "TrackNet_best.pt",
        "inpaintnet_ckpt": "InpaintNet_best.pt",
        "input_video": "1.mp4",
        "dataset": "shuttleset",
        "video_id": 1,
        "title": "Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2019_Finals",
        "url": "https://www.youtube.com/watch?v=O669aZhH0LI",
        "fps": 25.0,
        "extracted_utc": "2026-07-23T00:05:11Z",
        "inpaint_selected": [
            [5, 7],
            [11, 13]
        ]
    }

Field notes:

- `inpaint_selected`: sorted, non-overlapping, half-open `[start, end)`
  spans over frame id values from the CSV's Frame column. This is the
  raw switch mask, recorded before the near-origin coordinate zeroing
  and never intersected with visibility
- `stride`: the TrackNet checkpoint's sequence length under nonoverlap
  mode, 1 under weight and average modes (the three temporal ensemble
  settings of `--eval_mode`). It is a property of the weights, not a
  CLI flag
- `th_h_px`: the pixel threshold (5% of frame height) that decides
  whether a detection gap qualifies for filling; gaps where the shuttle
  exited through the top of frame stay unfilled
- `inpaint_status`: "applied" when InpaintNet ran; "disabled" when it
  did not, with empty spans and a null `inpaintnet_ckpt`. The extract
  pipeline falls back to TrackNet-only when InpaintNet weights are
  missing. That fallback used to be a console warning and nothing else;
  it now leaves this visible record
- the five source fields (`dataset` through `fps`) appear only when a
  manifest supplied them; their absence means "unknown", not "error"

## Reading a sidecar correctly

Every frame in the CSV falls into exactly one of three classes, and the
sidecar makes them separable. InpaintNet only ever fills holes; a
threshold-passing TrackNet detection is never overwritten.

- real detection: Visibility 1 and the frame is not in any span
- invented: the frame is in a span. Usually Visibility 1; the exception
  is a filled coordinate that landed near the origin, which `predict.py`
  zeroes back to (0, 0) as a no-detection. Such a frame shows
  Visibility 0 but stays in the spans
- honest no-detection: Visibility 0 and not in any span

"Frames TrackNet failed on" is the union of the last two classes.
"Invented positions currently masquerading as detections" is
span-membership AND Visibility 1, which is the derivation any consumer
should use. The one thing no consumer can recover is what TrackNet
almost detected: its confidence map below the detection cutoff is never
saved, so separating a weak detection from a pure invention inside a
filled gap needs a producer-side change.

Two operational rules follow from the design. First, keep one stride
per save_dir. The stride is in the sidecar name but not the CSV name,
so a second stride into the same directory leaves one CSV beside two
sidecars, one of them stale. Second, the sidecar-then-CSV write order
only guarantees a matched pair when the CSV did not already exist.
Batch mode guarantees that through its existing-CSV skip; standalone
reruns into a populated directory do not.

## Provenance manifests (sources.toml)

Source facts are asserted, never inferred: a numeric stem like `1.mp4`
proves nothing about which video it is. The writer looks for
`sources.toml` in the input video's own directory:

    dataset = "shuttleset"

    [videos."1.mp4"]
    video_id = 1
    title = "Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2019_Finals"
    url = "https://www.youtube.com/watch?v=O669aZhH0LI"
    fps = 25.0

The omit-versus-raise line: a missing manifest or an unlisted basename
omits the five fields quietly, because unsupplied provenance is normal.
Everything else raises: unreadable file, malformed TOML, wrong shapes or
types. A permission error must not read as absence. One directory holds
one dataset. ShuttleSet's generated clips live in per-stroke-class
folders, so covering them would need a manifest in each of those leaf
folders, keyed by clip filenames. Nothing currently generates manifests
anywhere in the download or extract pipeline, so pipeline-produced
sidecars carry no source fields today.

## Boundary choices, settled (do not relitigate)

Each of these was ruled during the 2026-07-22 scoping and survived three
pre-build red-teams; reopen only with a concrete reachable failure.

- one sidecar, beside the CSV only; nothing beside the npy (the
  normalised per-clip numpy array the training lane reads). The npys
  are regenerated wholesale from CSVs on every extract run, so a
  CSV-side record is the durable one
- the raw switch mask, never intersected with visibility; consumers
  derive the intersection
- write-before-CSV ordering instead of atomic-write machinery
- sidecar absence means "unknown", never "pending". Roughly 30,000
  legacy clip CSVs deliberately have no sidecar. A skip-if-exists check
  that treated absence as pending would requeue all of them; the
  extractor's TrackNet-only fallback could then overwrite inpainted
  CSVs with detector-only ones
- no artefact hashing or two-phase commit. The manifest is trusted by
  directory and basename: it cannot prove the listed file is the same
  bytes it originally described, and that trust boundary is recorded
  and accepted
- writer only: nothing downstream reads the flag yet

## Landscape notes from the build

Facts about the surrounding code that the scoping verified first-hand
and that future work here will need:

- `src/shared/tracknetv3/` is the authoritative TrackNetV3 tree for BRIC,
  BST-X, and the scrape lane
- the TrackNetV3 tree is excluded from the lint and type gates
  (ruff, pyrefly) by repo config, so `tests/test_inpaint_sidecar.py` is
  the only real coverage of the writer. Several of its checks inspect
  the source structurally instead of running it: call order inside
  `predict_video` and checkpoint propagation at the call sites, because
  a live ordering test would need GPU inference
- the extract pipeline reaches TrackNet through `batch_predict.py`
  worker processes.
  Its asymmetry (CSVs skip-if-exists, npys regenerated every run) is
  what makes the CSV the right sidecar anchor
- whole-video (t, 3) npys have no in-repo producer; they came from a
  hand-run chain. The CSV-to-npy converter is clip-specific: it parses
  the video id from an underscore-delimited stem, which whole-video
  filenames break
- `predict.py` derives `video_name` by stripping the last four
  characters of the filename. The writer uses `os.path.splitext`
  instead; the two agree on every `.mp4` name
- the BRIC lane reaches the same tree through the subprocess wrapper in
  `src/bric/perception/shuttle.py`

## Verification record

The stride-8 backfill (2026-07-23, on bourbaki, the group's GPU
server) re-ran videos 1, 15 and 21 through the new writer. All three
fresh CSVs reproduced the recorded reference tracks exactly, 0
differing rows of 154,393 / 149,487 / 100,349. The sidecars therefore
describe exactly the tracks every downstream consumer already uses.

Raw fill fractions measured there: 51.87%, 53.33%, 45.96% of frames.
The 2026-07 fabrication investigation, which graded each frame by
whether invention could be PROVEN, put the fabricated share at 34-37%.
The gap is by construction: the raw mask counts every filled gap,
provable or not. Build, audit and red-team records live in the
campaign's local scratch (`local_scratch/inpaint_sidecar/`, not
committed).
