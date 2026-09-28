# Sampling and scene reuse

The video runner produces **at most one court projection for each scene**.
A projection describes where the court's four corners lie in the image.
Returning camera views can reuse an earlier court after checks against the
current scene. Each scene keeps its own result, which may differ slightly from
the earlier projection.

This guide describes the current video runner and dataset-builder integration.
See the [detector README](README.md) for commands, input files and output schemas.

## What counts as a scene?

A scene runs from one detected cut to the next. **PySceneDetect finds the cuts**;
the court detector then analyses the resulting scenes. The adapter uses
ContentDetector with threshold 27 and a minimum scene length of about half a
second. That minimum controls cut detection, not whether a scene is long enough
for the court detector's player checks.

| How you run it | Where boundaries come from |
| --- | --- |
| Standalone with `--pyscenedetect` | The runner invokes PySceneDetect |
| Standalone with `--scenes FILE` | Your saved scene boundaries |
| Standalone with neither option | The entire video becomes one scene |
| Dataset builder | PySceneDetect, unless supplied with saved boundaries |

Frames are numbered from zero. A scene range `[start, end)` includes `start`
through `end - 1`. The cut frame belongs to the following scene. Saved boundaries
must cover the whole video without gaps or overlaps.

### A short clip or single image without PySceneDetect

For a clip, omit both `--pyscenedetect` and `--scenes`. The runner analyses its
middle frame and treats the clip as one scene. Keep the usual player checks if
the clip can hold the three-second window. For a shorter clip, add
`--no-require-people` to allow detection from lines and geometry:

```bash
PYTHONPATH=.:src python -m court_detector.run_video \
  --video CLIP.mp4 \
  --deeplsd-source DEEPLSD_CHECKOUT \
  --deeplsd-weights DEEPLSD_WEIGHTS.tar \
  --no-require-people \
  --output courts.json.gz
```

For a single image file, use `python -m court_detector.run_image`; the README's
[image section](README.md#run-from-an-image) gives the command and output.
It runs DeepLSD on the image. RTMLib runs only with `--with-people`, once on
that image. Those people mask occlusions and support proposals. One image has
no three-second window, so the video mode's player requirement does not apply.

A caller with its own line segments can instead use the Python `CourtDetector`
API with `Switches(require_people=False)`; see the README's
[inputs](README.md#inputs-and-result) and [API example](README.md#run-it).
Neither image path nor the clip command above invokes PySceneDetect.

Detection without player checks is supported, but has not been tuned for the
same precision as detection with player evidence.

## What gets sampled?

The runner chooses the scene's middle frame as its **anchor**. For an even number
of frames, it uses the later of the two middle frames.

| Purpose | Images used |
| --- | --- |
| Detect court lines, search and fit the court | The anchor image |
| Check player positions | 31 scheduled frames at approximately 10 per second, spanning three seconds around the anchor |
| Compare a returning camera view | The first, middle and last images of that sampling window |

The player window shifts when necessary to stay inside the scene. Frame indices
are rounded to the video's frame rate. For example, in a 30 fps video with an
anchor at frame 300, a centred window samples frames 255, 258, …, 345.

An additional check compares small greyscale images with the anchor. It keeps
the continuous run around the anchor whose average difference is at most eight
grey levels. This can shorten the 31-frame window. It helps exclude frames across
a missed cut or transition, but moving players and camera motion can also cause
differences. A discarded sample does not prove a cut was missed.

Player checks use the retained frames. The detector filters seated people and
restores brief crouches on tracks whose observations are mostly standing. It
checks the resulting feet against each candidate court. These samples provide
player evidence; they are not 31 independent court detections.

With players required, a scene that cannot hold the scheduled window gets
`scene_too_short_for_feet`. This means **unanalysed**, not that the scene contains
no court. With `--no-require-people`, short scenes can still be analysed. If poses
are supplied in that mode, a short scene uses only its anchor's people. Its
reuse image uses the scene's first, middle and last frames instead.

Saved pose arrays and sparse live pose extraction supply the same requested
frame positions. A full pose prerun extracts every frame for later pipeline use;
it does not increase the court detector's sampling schedule.

## How returning views reuse a court

Reuse is optional in the standalone runner: enable `--reuse-courts`.
The supplied [ShuttleSet configuration](../../configs/dataset_builder/shuttleset_fixed.toml)
and [trial configuration](../../configs/dataset_builder/trial.toml) enable it.

The runner processes scenes in time order:

1. **Gather the new scene's evidence.** Read its anchor, detect its lines and
   gather its player samples. Combine three window images into a median image
   to reduce interference from moving players when comparing views.
2. **Choose earlier courts to try.** Keep up to eight recent courts from full
   searches. Try at most three for the new scene. When scene histograms are
   available, their similarity sets the order; otherwise try the most recent
   first. A histogram describes image brightness and cannot establish a match.
3. **Align and refit.** Align each reference image to the new scene's median
   image. Carry its corners across, then refit them to the stripes in the new
   anchor image.
4. **Check the result.** Require acceptable image alignment, court geometry,
   camera orientation and paint support. Check player positions when required.
   Also limit how far the stripe refit moves the court, including its far end.
5. **Accept the first passing reuse, or search afresh.** Record the earlier
   reference in `reused_from` when reuse succeeds. If every attempt fails, run
   the full court search using the already prepared evidence.

Only a court found through a full search can become a reference. A reused court
never becomes another reference, which avoids accumulating adjustments through
a chain of reused results. The reference store starts empty for each video.

### Example: wide view, close-up, wide view

Suppose scene 1 shows the whole court, scene 2 is a player close-up, and scene 3
returns to the wide view. Scene 1 can supply the reference for scene 3 even
though they are not consecutive. Scene 3 still gets fresh line and player
evidence. If alignment and court checks pass, it uses an adjusted version of
scene 1's court. Otherwise it gets a full search.

Scene 3 does not trigger another sampling round or a reread of scene 1. The
runner does not pool evidence across the two scenes or revise scene 1's result.
An older reference may also have fallen out of the eight-court store, or may
not be among the three tried.

## What the output does and does not establish

Every scene gets a separate result: a court, no accepted court, a short-scene
status or a detection failure. Accepted scenes retain their own corners. The
annotation pipeline applies a scene's fixed projection across its frame range;
it also applies its own player-vote acceptance rule. That rule requires exactly
two people inside the court's margin in at least half of the scene's frames.
It uses the full pose arrays, separately from the detector's 31-frame window.

There is **no video-wide grouping pass that assigns one identical projection
to every occurrence of a camera view**. The older hash-grouping helpers remain
in `annotator/court_views.py`, but the new runner does not call them. Reuse is a
forward pass through scenes with a limited set of earlier references.

The runner also does not track a pan or zoom within a scene. Its extra image
check can restrict player samples, but it does not create new scene boundaries
or additional court projections. Long scenes still receive one anchor-based
projection. Missed cuts, camera movement and poor visibility can therefore
make that projection unsuitable for parts of the scene.

### What has been checked

A 24-video comparison on 29 September 2026 covered 533 scenes and returned 188 courts. Of 76
accepted reuse pairs, the older hash-grouping method placed 74 in the same group.
This supports agreement between the methods on those pairs; it does not establish
that every reused court was accurate. The image-consistency check shortened 244
of 359 eligible player windows, so removing it would materially change the
player evidence.

Reuse has been exercised on real videos, but its paint-support and movement
thresholds remain heuristic choices. Difficult views can still produce false
courts. Reuse checks reduce risk; they do not guarantee a correct projection.

## Where the behaviour lives

- [scene_sources.py](scene_sources.py): PySceneDetect adapter and saved boundaries
- [run_video.py](run_video.py): anchor selection, median images and reference store
- [feet.py](feet.py): sample schedule, image-consistency check and player evidence
- [reuse.py](reuse.py): alignment, stripe refitting and reuse acceptance checks
- [court_evidence.py](../annotator/court_evidence.py): per-scene pipeline evidence
