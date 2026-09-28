# Court detector: resume here

Updated 28 September 2026. The live detector now processes five minutes of video
in **279.9 seconds**, down from 872.8 seconds. Court reuse uses median images,
and line-template scoring has an explicit CUDA option. The 30-second goal and
90-second upper target remain unmet.

Continue on `fix/court-det`. Check the current Git state before editing. The
latest measured implementation is `d0b8c4ce`; naming and readability cleanup is
in progress. The [detector guide](court_detector/README.md) owns the API and
settings. [PERFORMANCE.md](court_detector/PERFORMANCE.md) records measurements.

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

Finish descriptive names for the historical W5/G0/G1 stages and their saved
outputs. Preserve deliberate readers of old records. Complete the current
readability pass, including explicit array-module names and boolean masks.
Check the final changes on CPU and CUDA and reconcile the performance report.

The broader float32 pass remains deferred. Retain float64 only where a specific
numerical need is demonstrated; assess final court quality, especially far-end
errors in metres. A later pass should also check hand-written operations against
library equivalents where that would simplify the code without hurting speed
or CPU/GPU compatibility.

Candidate scoring and fitting still take about 79 seconds in the latest live
run, and the two direction searches take about 67 seconds. These are substantial
remaining costs. Do not present cleanup work as achieving the speed target.

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
