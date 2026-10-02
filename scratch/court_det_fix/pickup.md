# Court detector: resume here

Updated 28 September 2026. The detector processes five minutes of video in
**173.2 seconds with CUDA templates**, or 219.9 seconds with CPU templates.
These times include DeepLSD and exclude RTMLib and PySceneDetect. Whole-run
totals are 227.4 and 274.9 seconds. Both configurations use GPU neural inference.
The 30-second goal and 90-second upper target remain unmet.

CPU and CUDA return identical scene results and corners. The fp32/API and Numba
changes preserve the preceding version's courts within 0.000004 px. Persistent
workers now avoid repeated startup across stages and scenes, with identical
courts and reuse decisions on the measured interval. No further GPU backend
comparison is planned; keep CuPy.

Continue on `fix/court-det`. The latest measured implementation is `c24b45fd`.
Check current Git state before editing. The [detector guide](../../src/court_detector/README.md)
owns the API and settings. [PERFORMANCE.md](court_detector/PERFORMANCE.md) records
measurements and timing scope.

## What changed since the previous pickup

The earlier 743.3-second result has been superseded by further matched runs.
Reusing direction masks and retaining brief crouches reduced the total to
694.9 seconds. CUDA template scoring reduced it to 536.7 seconds. Median-image
reuse took 300.0 seconds in the experiment and 279.9 seconds after integration
and ordered frame reads. Each configuration ran once on an L40 GPU with at
most eight CPU cores. These observations do not establish consumer-GPU speed.

All runs cover 7,500 frames at 25 fps across 33 scenes, with live DeepLSD lines,
RTMLib poses and PySceneDetect cuts. They return 13 courts, reject 11 scenes and
leave nine scenes unanalysed because they are too short for the feet window.
The persistent-pool run exactly reproduces the preceding numerical checkpoint.
The numerical changes altered two chosen IDs between effectively identical courts;
scene statuses and reuse decisions remain unchanged.

The previous request to investigate frame 12636 is resolved for this pass.
The standing-person filter dropped a brief crouch. A simple nearest-person
tracker now retains such samples when most of the track is standing. It fixes
the advertising-wall court while preserving the final player-position rule.
The tracker can swap identities at crossings. The earlier referee-elbow idea
was not established.

The previous "no GPU backend adopted" and incomplete-coverage notes are also
superseded. CuPy is now an explicit option, with one implementation shared by
NumPy and CuPy. CPU remains the default. Final courts were checked on all 28
development views; 17 hardware tests passed without skips. Earlier CuPy and
CUDA-MLIR prototypes were compared on smaller components. No further backend
is required for the accepted implementation.

Median alignment is now adopted when video court reuse is enabled. It uses the
first, middle and last frames of the existing feet window for both the current
image and stored references. Reuse increased from one court to eight. The
alignment and acceptance thresholds are unchanged, and reused courts never
become later references. Court search and refitting still use the actual middle
frame. The user reviewed frames 12636, 15011 and 15873 as usable, with small
baseline or corner offsets. Frame 17820 has wrong cross-court boundaries in a
difficult distorted view. That is an accepted limitation for this pass, and
its court was not reused elsewhere.

## Remaining work

Descriptive names replace the historical W5/G0/G1 stages in live code and new
outputs. Readers translate the old fields explicitly. Bulk template, support
and stripe evidence arrays use float32. Fitting, ill-conditioned direction
geometry and sensitive checks retain float64. Numba CPU support scoring uses
one serial cached kernel, with float64 projection/clipping calculations.

Worker reuse passed 2,516 tests with 35 skips before small review fixes. Those
fixes passed 21 recovery tests and the other 72 focused tests. They ensure later
saved views recover from a crashed worker and input files outlive running tasks.
Changed-file Ruff, whole-project Pyrefly and 43 hardware tests passed. The live
CPU/CUDA comparison preserves all 33 scene results exactly.

Direction search now takes about 42 seconds and scoring/refitting about 63 seconds
with CUDA templates. The transfer audit supports keeping the current small
read-only input snapshot. A faster record conversion and removal of unused search
diagnostics follow the timed run; their whole-run gain has not been measured.
These transfer changes passed 55 focused tests, changed-file Ruff and whole-project
plus explicit changed-source Pyrefly checks, all with exit code 0.
The remaining path to 90 seconds needs another 48% reduction. No straightforward
change with that expected gain has been established. Finish the bounded efficiency
work, then focus on the integration and broad evaluation below.

## Runtime integration

The maintained detector now lives in `src/court_detector`. Normal runtime imports
no scratch, experiment or CourtKeyNet modules. Saved-view loaders and comparison
tools remain in scratch. Court dimensions and painted segments live in
`shared.court_model`; DeepLSD loading and inference helpers belong to the runtime
package. The old pipeline still uses CourtKeyNet until its replacement batch.

The move passed 2,518 tests with 35 skips and whole-project Pyrefly, both exit 0.
Two saved views returned identical courts and records after excluding timings.
Changed-file Ruff retains 11 existing findings in four CourtKeyNet files; the
migration introduced none. A live CUDA DeepLSD check also passed after the move.

Scene ranges now use zero-based `[start, end)` throughout the detector. The
initial conversion keeps the same sampled frames and centralises midpoint choice
in `SceneInfo.middle_frame`. Boundary tests cover cuts, short scenes, foot windows,
reuse samples and the last video frame. Focused tests passed 228 cases with six
GPU skips; Ruff and Pyrefly passed. The full suite passed 2,541 tests with 35 skips
and one interpreter-PATH failure. The affected 15-test file then passed with the
venv on PATH.

Independent review of the range change remains pending. The simpler midpoint
`(start + end) // 2` is preferred if a bounded comparison shows comparable court
quality. The current implementation retains the lower midpoint until that check;
its parity results do not establish the effect of choosing the next frame.
Shared pose extraction, the complete entry point and old-pipeline replacement
follow the range review and midpoint decision.

## Follow-up after satisfactory optimisation

These are required follow-ups once the CPU and GPU paths perform satisfactorily
on videos with multiple scenes and on batches of independent videos. The current
28-view development set and single five-minute interval do not fulfil this work.

- **Evaluate a broad population.** Run the complete collected amateur videos.
  Also test many ShuttleSet and ShuttleSet22 videos, potentially through
  contiguous segments lasting several minutes from a variety of videos.
  Budget about two hours per evaluation run.
- **Use the right references.** Existing extracted court detections come from
  the broken old detector and are unsuitable as truth. The bundled static
  single-homography-per-video files are useful only where the current view
  agrees. They can be wrong even inside normal rally bounds, with little warning.
- **Keep visual review small.** Agents should inspect only a small subset of
  frames. Use numerical findings to prepare a bundle of 20–30 images that merit
  the user's inspection. Keep court overlays as plain 1px red dashed outlines.
- **Provide one complete entry point.** Run end to end with or without supplied
  bounding boxes and keypoints. Also support an optional RTMLib/RTMPose pre-run
  to obtain the required player detections, using the configuration settings
  already used elsewhere in this project.
- **Evaluate scene splitting and grouping.** Integrate useful PySceneDetect
  boundaries and compare grouping with the detector's existing hashing approach
  across a substantial sample. Scene cuts and hashes may have different value.
  Keep only useful parts, and make the integration optional if measurements show
  the existing approach is faster and equally useful. Record that evidence.
- **Finish source integration.** The runtime detector is now in `src/` and the
  research harness stays in scratch. Complete shared pose extraction and replace
  the old pipeline call sites, as described above.
- **Replace CourtKeyNet throughout the project.** Remove that dependency and
  OpenCV support written solely for it that is no longer needed. Move any court
  geometry or other shared functionality still required by the homegrown
  detector into an appropriate shared location. Wire the replacement into every
  call site that previously ran the old court-detection pipeline.

## Continuing limits

ShuttleSet homographies are static templates per video. Camera perspective can
change during a match, and scene cuts can miss changes or split unchanged views.
Judge current visible paint and geometry; baseline agreement is not accuracy.

The development set has 20 court views and eight unlabelled controls. Controls
may contain real courts. Dark markings and unseen cameras remain insufficiently
evaluated. Required people remains the default. Optional mode has known false
courts and has not had an equivalent full-video timing trial. No separate
optional-mode precision sweep is requested.

The detector temporarily uses the neighbouring BST-X classifier's pose extractor.
Move the extractor and constants into shared code later. Keep scene and people
providers optional interfaces. Exhaustive scoring remains the default. Do not
reopen closed colour, net-weight or search-depth sweeps without new evidence.

[INDEX.md](INDEX.md) maps the documents.
[DETECTOR_DECISIONS.md](DETECTOR_DECISIONS.md) records retained choices.
