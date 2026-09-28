# Court detector: resume here

Updated 28 September 2026. The detector processes five minutes of video in
**206.8 seconds with CUDA templates**, or 251.8 seconds with CPU templates.
These times include DeepLSD and exclude RTMLib and PySceneDetect. Whole-run
totals are 265.7 and 307.4 seconds. Both configurations use GPU neural inference.
The 30-second goal and 90-second upper target remain unmet.

CPU and CUDA return identical scene results and corners. The fp32/API and Numba
changes preserve the preceding version's courts within 0.000004 px. The next
bounded change reuses worker processes across stages and scenes to avoid repeated
startup. No further GPU backend comparison is planned; keep CuPy.

Continue on `fix/court-det`. The latest measured implementation is `6cfc7a89`.
Check current Git state before editing. The [detector guide](court_detector/README.md)
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
The latest integration exactly reproduces the median experiment's scene
statuses, chosen keys, reuse sources and corners.

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
outputs. Readers translate the old fields explicitly. The fp32/API and Numba
changes passed the full CPU suite before the final small template tie fix:
2,499 tests passed, 34 skipped. That fix passed relevant local tests, and the
combined Carmack hardware checks passed 43 tests with no skips. Changed-file
Ruff and whole-project Pyrefly passed. Independent review found the tie issue
and it is fixed.

Bulk template, support and stripe evidence arrays use float32. Fitting,
ill-conditioned direction geometry and sensitive acceptance checks retain
float64. Numba CPU support scoring uses one serial cached kernel. Its small
projection/clipping calculations also use float64. The combined live run
preserves the accepted courts; no datatype-specific ablation is needed.

Direction search still takes about 62 seconds and candidate scoring/refitting
about 71 seconds. Reusing worker processes is next. Check ordering, fresh state
for each view and failure/cleanup paths, then compare the existing live interval.
Later broad evaluation and integration remain required as listed below.

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
- **Move the detector into `src/`.** Give the subproject a substantive tidy and
  coherent organisation so normal use no longer depends on disparate scratch
  scripts.
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
