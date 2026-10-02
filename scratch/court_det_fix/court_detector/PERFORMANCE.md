# Make the detector faster

This is the design reference for speed-up work. It owns the proposed changes,
constraints and checks. [pickup.md](../pickup.md) owns what to do next;
[the decisions](../DETECTOR_DECISIONS.md#d18) own accepted targets and past
results. This document replaces the design scattered across the old handovers.

The detector searches using all line fragments and using paint-like fragments.
The code calls these searches `all_lines` and `painted_lines`; older records and
notes call them G0 and G1. It also builds courts from crossing lines. It scores
the resulting courts against painted stripes, refits them, then chooses one. The
code calls that stage scoring; older records and notes call it W5.

## Current measurements — 28 September 2026

The latest measured run keeps worker processes alive across stages and scenes,
on top of fp32 bulk arrays, simpler projection/pairing operations and serial
cached Numba support scoring. It processes the same five-minute interval:

| Timing | CUDA templates | CPU templates |
| --- | ---: | ---: |
| Court detector, including DeepLSD | 173.226 s | 219.878 s |
| People provider setup and calls | 35.006 s | 36.554 s |
| Scene detection | 19.163 s | 18.453 s |
| Whole run | 227.396 s | 274.885 s |

Both configurations use GPU neural inference. These are single runs on an L40
with at most eight CPU cores. Hardware checks warmed compiler caches before the
timed runs; neural model setup and worker startup/shutdown remain included.
Detector setup takes 4.303 s with CUDA templates and 5.386 s with CPU templates.
DeepLSD setup and calls contribute 6.792 s and 6.827 s respectively.

All 33 scene statuses, choices, reuse sources and corner arrays match between
CPU and CUDA, and exactly reproduce the preceding fp32/API/Numba run. That
preceding change moved corners by less than 0.000004 px. There are still 13
courts, 11 rejected scenes, 9 short scenes and 8 reused courts. The latest
hardware checks passed 43 tests without skips.

Worker reuse reduces detector time from 206.788 s to 173.226 s with CUDA
(16.2%), and from 251.818 s to 219.878 s with CPU templates (12.7%). The previous
whole-run totals were 265.695 s and 307.375 s. The original total was 872.829 s.

The 30-second goal and 90-second upper target remain unmet. Direction searches
take 41.8 s and candidate scoring/refitting 63.5 s in the CUDA run. Reaching
90 seconds would need another 48% reduction; no straightforward change with
that expected gain has been established. The bounded optimisation pass is
moving towards completion and broader integration/evaluation. These runs do
not establish accuracy across varied videos or performance on a consumer GPU.

## Worker data transfer

Each scoring view uses one read-only pickle snapshot, about 6 MB in three
sampled views. A laptop probe measured 8–9 ms to write it and 3.5–6 ms for each
worker to load it once. The measured server uses memory-backed temporary
storage. Keep this simple transfer: shared memory would add lifecycle handling
for a small expected saving. Workers keep private measurement state; there are
no shared writes or locks.

A separate two-view audit found a more useful saving in the parent's repeated
conversion of scoring records to JSON-compatible values. Plain scalars and
containers now take a short path through that conversion. NumPy values,
subclasses and non-finite values retain the previous handling. Search results
also omit unused axis diagnostics. These two changes follow the timed run
above; no whole-run speed gain is claimed for them. Existing archived diagnostic
readers may still use axis records from older saved outputs.

Repeated search inputs and scoring return arrays have small measured transfer
costs. Keep them for this pass. Pickle already preserves aliases within a
returned payload, so the measurement-cache return does not duplicate its arrays.

## Constraints

The target is 30 seconds of court detection per five minutes of video, with
90 seconds as the upper target. Include DeepLSD, cutaways and repeated camera
views. Report RTMLib and PySceneDetect separately, along with the whole-run
total. This timing scope supersedes the original whole-video interpretation in
[D18](../DETECTOR_DECISIONS.md#d18). Report setup and warm throughput separately.
A fast isolated scoring function is only an intermediate result.

CPU-only use must keep working. The GPU target is a consumer card with about
16 GB and CUDA 13; development uses the full L40 with 48 GB. The earlier V100
reference describes a rough performance level, not a requirement to support
that card. Measure memory and throughput separately. A memory cap on the L40
does not reproduce the speed of a weaker GPU.

Keep one maintained definition of the scoring maths. Backend wrappers are
justified when a second backend exists. The earlier proposals for plug-in
classes and four deployment modes add no useful requirement here.

## Run independent work in parallel

Search pairs are independent until their kept courts are merged. Collect
results in the original pair order, then apply the existing ranking and tie
rules. Otherwise equal scores can change which court survives.

After duplicate courts have been merged, each court can be measured and
refitted independently. Restore the original order before ranking their
results. These are two separate opportunities for parallel work.

Measure scaling with eight workers on Carmack. Parallel work reduces elapsed
time for a view; it does not inherently reduce CPU work. Old estimates came
from concurrent jobs on a shared server and do not establish the gain.

## Choose how to use the GPU and Numba

Keep CuPy for GPU line-template scoring. It shares the array implementation
with NumPy and has demonstrated useful whole-path savings. CUDA-MLIR produced
promising component timings and three final-court checks, but has less complete
path evidence. The measured Torch scoring gain did not justify adoption. Further
backend comparisons are closed for this pass.

CPU direction-pair support now uses one serial cached Numba kernel. Existing
worker processes provide parallelism. It supports both 16-sample cheap scoring
and 64-sample full scoring. Its isolated speed-up is modest; the combined fp32,
API and Numba run above is the current whole-path check. It does not isolate
Numba's contribution.

Bulk templates, support arrays and stripe evidence use fp32. Fitting,
ill-conditioned direction geometry and sensitive checks retain fp64. The compiled
CPU scorer also uses fp64 for its small projection/clipping calculations.
Template scores sum integer sample counts before one fp32 division, preserving
exact ties. Camera errors near the acceptance boundary are rechecked in fp64.

Comparing CPU and GPU scores on the courts kept can expose drift between the
implementations. It cannot find a court the GPU discarded. Measure both risks.
Judge GPU changes by final court quality, rather than require identical scores
or candidate IDs. Measure calibration displacement in court metres as well as
image pixels, including the far end separately. Perspective makes a pixel shift
near the far baseline matter more than the same shift near the camera.

**ShuttleSet's supplied ground-truth homography is static per video.** Camera
perspective changes can make it wrong for the current shot, especially outside
rallies. Use it as a yardstick and check visible court lines in the current
frame. PySceneDetect cuts suggest shot changes, but do not prove that camera
geometry changed or stayed the same. Keep CPU-baseline differences separate
from accuracy against a reference homography.

Check frame identity when combining saved poses, annotations and trimmed video.
The [BST-X decoder note](../../../src/bst_x/preparing_data/mmpose_changes.md)
documents a 1–2-frame tail-count difference between older MMPose and TrackNet
decoders. It reports matching indices before the tail, rather than a shifted
sequence. A separate comment in
[clip_generator.py](../../../src/bst_x/pipeline/clip_generator.py) records an
unverified one-frame start-offset concern in the MoviePy clipping path.
Neither justifies silently shifting pose indices or accepting a larger trim
discrepancy. Check decoded source-frame identity before using a new clip.

If minor precision or rounding changes alter court quality, retain the case and
its calibration impact as a detector-stability issue for the stage after
performance tuning. Do not hide the issue with tolerance adjustments.

The former web-UI proposed two initial speed checks: each hot function at
least 10× faster than one CPU core, including transfers, and the whole search
at least 2× faster. These are proposed prototype checks, not measured gains.
Measure peak memory and confirm results on the target hardware before relying
on the L40 result. Keep the GPU in one serving process when several scenes
need it.

## Score cheaply before scoring in full

The project owner agreed on 26 September to try 16 samples per marking for
all courts, then the usual 64 samples for the best 2,048 per direction pair.
Provide a command-line setting for that limit and a way to score every court.
[D24](../DETECTOR_DECISIONS.md#d24) records the supporting replay and its limits.

The change belongs in `propose_role` in [proposals.py](../../../src/court_detector/proposals.py).
Pass the setting through `generation.generate` to the view runner. Keep the selected courts
in their original order before full scoring, with their source arrays sliced
the same way. Leave pairs below the limit unchanged.

Rerun all 28 views against a fresh current control. Compare the courts kept by
each complete search, the later scores and the chosen courts. Pair-level
records can differ when discarded courts were already below the overall cut.

Log how deeply each search's kept courts ranked in their pair's cheap score.
This exposes cases close to the limit, but cannot reveal a court already
lost. Measure more views with full scoring alongside before trusting the
default more widely. The old estimated saving predates the camera filter.
Use the actual cheap score: the average line-match score failed as a substitute.

## Reuse a court when the camera returns

The project owner agreed to this design on 26 September. Apply the same policy
on CPU and GPU.

1. Before searching a scene, compare it with earlier scenes that have courts.
   Reuse the image-matching pieces in `src/annotator/court_views.py`
2. Return the fitted image warp from `_view_alignment` as well as its match
   result. Apply that warp to the earlier court's corners
3. Refit the moved court to the new scene's stripes and run the full-court
   checks with the new scene's feet
4. Require stripe support close to the earlier scene's and a small corner
   movement during refitting. Tune the tolerance on known same-camera pairs
5. If the checks fail, run the full detector. If that search chooses a
   different court for the same camera, flag the camera group

The proposed setting for full searches before reuse defaults to one.
Increasing it to three restores the protection of combining three separately
searched courts. One search is faster but can propagate its mistake; tests
must cover that tradeoff.

The existing grouping happens after court detection and requires at least
three valid scenes. It does not implement this pre-search step. Passing a
borrowed court into the existing exhaustive search alone would save nothing.

Test repeated camera views, changing light or player cover, crops and
letterboxing, different cameras, replays and cutaways. A cutaway must not gain
a court merely because an earlier scene had one. The tolerance and the cost
of handling scenes without courts remain open.

## Check each change

Use the API and runner in [README.md](../../../src/court_detector/README.md). The
[research wiring](../archive/20260927_code/d17_timing/WIRING.md#how-to-check-an-integrated-detector)
and [28-view check](check_20260925/README.md) define the original comparison.

- For a change intended to preserve results exactly, compare before and after
  in the same environment. Use
  `../court_detector_optimisation_handover/claude_evidence/exact_rewrites/compare_exact_runs.py`
  for saved research runs. It compares float bits, array types, shapes and
  bytes, while excluding named timing and path fields
- The old research baseline needs `--any-camera-roll --geometry-weight 0`.
  For a new speed-up, compare against the current defaults too. Passing the
  historical baseline alone does not check the behaviour now in use
- For a change that alters scores, compare chosen courts and floor-coordinate
  errors, then inspect meaningful changes. Keep the eight control views
  in the test. Control membership does not mean no court is visible: the wide
  arena control contains multiple courts. Preserve existing useful courts,
  including backups
- Compare paired before/after runs. At most eight jobs in total was the
  recorded protocol. Shared-server timing varies too much to infer a saving
  from one view or an unmatched old run
- The recorded Carmack environment was `court_det`: Python 3.12.13, NumPy
  2.5.3, SciPy 1.17.1 and OpenCV 5.0.0.93. Verify it before resuming, and use
  the same environment for both arms

The live video runner now supplies lines, people, poses and scene cuts. Its
matched five-minute runs are the current end-to-end timing comparison. Broader
video evaluation and migration into the shared project pipeline remain open.

## Sources

The [archived speed-up account](../archive/20260926/originals/court_detector_optimisation_handover/README.md)
and [web-UI comparison](../archive/20260926/webui_final_opt_handover/README.md)
retain the full reasoning, measurements and rejected options. The
[measurement index](../court_detector_optimisation_handover/claude_evidence/README.md)
locates scripts and saved outputs. Read them for a specific question; neither
owns the live work queue.
