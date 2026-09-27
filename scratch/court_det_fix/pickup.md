# Court detector: resume here

Updated 27 September 2026. The detector now runs from source video with live
DeepLSD lines, RTMLib poses and optional PySceneDetect scene information.
The latest five-minute rally trial finished successfully. Performance and
court-selection accuracy still need work; the campaign is not complete.

## Current state

Continue on `fix/court-det`. The tested implementation is `976f3b8a`.
Check Git state before editing. The latest remote comparison has finished;
no benchmark needs to be awaited before resuming.

The full live trial covers 7,500 frames at 25fps across 33 scenes. It took
743.3 seconds, compared with 872.8 seconds before the required-player-count
shortcut: 14.8% less elapsed time in this pair of runs. All scene statuses,
corner coordinates, chosen keys, reuse sources and rejection reasons match.
It returns 13 courts, rejects 11 scenes and skips nine short scenes.
This remains well above the 30-second goal and 90-second upper target.
The court and camera-view suite passed all 167 tests before this run.

Implemented changes include cheap-score shortlisting, exhausted-pool handling,
less JSON conversion, court reuse before search, and optional people inputs.
Exhaustive scoring remains the default. `--no-require-people` provides a
plausible fallback; required people remains the default. The early count check
skips impossible required-player searches after input validation. Optional mode
and full diagnostic runs retain their normal searches.

No GPU scoring backend has been adopted. Two four-core CPU processes improved
warm independent-scene throughput by 1.54 times in a small saved-input trial.
That excludes video decoding, inference and reuse. Torch did not establish a
useful gain. Compare CuPy and Numba-CUDA-MLIR performance and complexity before
choosing a backend; GPU coverage remains incomplete.

## Next work

Investigate frame 12636: fresh search gets the verticals right but selects
unusable horizontal boundaries. A median-aligned and refitted previous court
has correct near/left/right edges, but slightly undershoots the far paint edge,
especially far-left. The user rates that fit only minimally acceptable for
this easy court. The player-position gate rejects it anyway.

Sample 12604 has no retained foot inside that plausible court. Inspect its
original boxes/keypoints and standing-person filter before changing the rule.
The user's suggestion that fresh search follows referees' elbows is unverified.
No median-alignment policy or threshold change has been adopted.

Continue matched full-pipeline performance measurements after reviewing that
failure. Evaluate numerical differences through final court quality, especially
far-end errors in metres. Preserve evidence of brittle scoring rather than
hiding qualitative failures with numerical tolerances. Rename historical
labels such as w5/g0/g1 after behaviour has been evaluated.

Detailed run records and the bounded handover are kept locally under
`local_scratch/campaigns/court-det-speed/` from the repository root. Start with
`HANDOVER.md`; read deeper records only for the current question. If
`opus-review-result.md` exists, assess its findings before repeating that review.
These working records and large artefacts are not committed. The detector guide
and this pickup file remain available from Git.

## Read before judging quality

**ShuttleSet homographies are static templates per video.** Camera perspective
can change during a match and often changes outside rallies. Supplied ground
truth can therefore be wrong for the current shot. PySceneDetect cuts also
miss changes or split unchanged views. Judge current-frame court lines and
alignment; baseline agreement does not establish accuracy.

The prepared set contains 20 court views and eight control views. Controls can
contain real courts and are not verified negative examples. Optional-people
renders have known limitations: one catches the net top, another overshoots a
back corner. No separate no-people precision-tuning sweep is requested.

The detector still temporarily uses BST-X's pose extractor. Decouple that later;
keep scene and people providers optional interfaces. Dark court markings and
performance on unseen cameras remain insufficiently evaluated. Do not reopen
closed colour, net-weight or search-depth sweeps without new evidence.

[INDEX.md](INDEX.md) maps the documents. The
[detector guide](court_detector/README.md) owns API and settings;
[DETECTOR_DECISIONS.md](DETECTOR_DECISIONS.md) records retained choices.
