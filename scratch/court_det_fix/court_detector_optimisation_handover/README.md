# Court detector speed-ups

In the 28-view check, the joined court detector used about 230 process
seconds per view, with eight single-threaded views running at a time. The
target is about 30 s for a whole five-minute video, with 90 s as the upper end
([pickup.md](../pickup.md#compute-requirement-and-next-work)). The measured
exact speed-ups are built in, and a few smaller exact ones remain. The larger
savings left need a design decision or a first measurement.

This page lists what is built in, what is left to try and how to check a
speed-up. [claude_evidence/README.md](claude_evidence/README.md) names the
script and result behind each number. The detector itself is described in its
own [README](../court_detector/README.md).

## Names used here

- **Joined detector**: `court_detector/`, the accepted method as one piece of
  code that keeps each step's results in memory.
- **Research chain**: `d17_timing/run_d17.py`, which runs the same method
  through the research scripts and saves each stage's results to files. The
  research calls the method the D17 chain.
- **Court search**: the step that pairs up a view's main line directions and
  builds candidate courts from each pair. It runs once on all line fragments
  (G0) and once on paint-coloured fragments (G1).
- **Shortlist**: the 256 best distinct candidate courts that one direction
  pair keeps.
- **Scoring**: the stage after the search (W5). It measures each shortlisted
  court against the painted stripes, refits it and ranks the results.
- **Exact**: the compared saved results are bit-identical to the run before
  the change, apart from timings and a few named diagnostic fields.
- **Camera check**: a test of whether a real camera with a plausible lens
  could see a candidate court the way it appears.
- **Items 1–17**: the numbered sections of the archived
  [speed-up list](../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md).
- **F1–F12**: proposals in the static audit of the web-UI handover packet of
  23 September, which this folder started as. The packet is archived too.
- **Patches 1 and 2**: the first two exact fixes, from the archived
  [evaluation](../archive/20260925_optimisation_handover/CLAUDE_EVALUATION.md)
  of that packet.
- **Times**: process seconds summed over views, on Carmack, the shared compute
  server, unless a row says otherwise.

## Where the time goes

The court search and scoring take 93% of the joined detector's time. Over the
28 test views, with self-checks off and 8 processes at a time, it took
6,434 s ([timing.txt](../court_detector/check_20260925/timing.txt)). That
leaves out start-up and video decoding.

| Step | Seconds | Share |
| --- | ---: | ---: |
| Court search on all fragments (G0) | 3,346 | 52% |
| Court search on paint fragments (G1) | 785 | 12% |
| Scoring (W5) | 1,837 | 29% |
| Line templates | 409 | 6% |
| Feet, set-up, net choice and stripe refit | 53 | 1% |

Single views range from 20 s, on a view with no court, to 546 s on
`gxBQ_window_00_frame_689`.

The joined detector does not time single functions, but the research chain
does. In its last full run, 8,250 s over the 28 views, three functions took
56% of the time
([stage times](claude_evidence/w5_savings/carmack_stage_times_vs_item12.txt)):

| Function | What it does | Seconds |
| --- | --- | ---: |
| `finite_scores` | Scores each candidate court against distance maps of the line fragments | 2,054 |
| `match_axis` | Matches each direction's lines to the court's markings | 1,384 |
| `stripe_measure` | Measures how well a court's markings sit on painted stripes | 1,156 |

That run also spent about 550 s in its file-writing and file-reading
functions, which the joined detector skips.

## What is left to try

| Idea | Likely gain | Exact? | What it needs | In the joined detector |
| --- | --- | --- | --- | --- |
| Reuse a court across the scenes of one camera ([item 9](../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md#9-view-reuse-before-full-search-f12)) | The largest for whole videos: it skips whole searches | No; a design decision | Rules for "same camera", and a check that catches a wrong reuse | No |
| Search direction pairs in parallel ([item 8](../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md#8-run-generator-pairs-in-parallel)) | Shorter wait for one video, perhaps several-fold on 8 cores; not measured. No CPU saving | Yes, if results merge back in pair order | A process pool in the court search | No |
| float32 in court scoring and axis matching ([item 17](../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md#17-tested-float32-and-float16)) | Perhaps 10–20% of the run; a guess | No | A cheap timing test first (below) | No |
| Build candidate objects only for shortlisted courts ([item 6](../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md#6-build-candidate-objects-only-for-survivors-f8)) | At most 179 s over 28 views (2%) | Yes, with the same tie order | A change to the search so it builds candidate objects only after the shortlist is chosen | Only the half item 12 did |
| Larger axis-scoring batches (F4, [verdict](../archive/20260925_optimisation_handover/CLAUDE_EVALUATION.md#claim-by-claim-verdicts)) | A few per cent; not measured | Yes | A batch-size change and a timing check | No |
| Compiled loops (numba) for court scoring ([open list](../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md#open)) | Not measured | Not automatically: loop sums round differently from numpy's | A new dependency. Worth it only if the ideas above miss the target | No |
| Working image below 960×540 ([input resolution](../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md#input-resolution)) | Not measured | No | An accuracy check, since reference errors are about 2 working pixels now | No |

**Reusing a court across scenes** is the lever for whole videos. Today the
pipeline groups recurring views only after it has searched each one
([pickup.md](../pickup.md#compute-requirement-and-next-work)). On one core,
one court view in 20 takes under 90 s; the rest take 130–546 s
([timing.txt](../court_detector/check_20260925/timing.txt)). So a five-minute
video needs few searches, as well as faster ones.

**Parallel pairs** cut one video's wait on a multi-core machine. The direction
pairs are independent until the shortlists are merged. Results stay exact if
each pair's candidates merge back in their original order.

**float32** is a guess, not a measurement. The distance maps are already
float32, and court scoring mostly reads from them. The cheap test is to time
`continuous_support` (inside `finite_scores`) and `score_axes` (inside
`match_axis`) in float32 on saved inputs. A 28-view run is worth doing only if
both reach about 1.4×. Fitting must stay float64. That run would compare each
view's chosen court, net choice and reference error with the baseline.

**The working image** is the image shrunk until its longest side is at most
960 px (`experiments/annotator/independent_court/export_lines.py:17`). A
smaller one cuts work that scales with pixels, but axis matching and court
scoring scale with candidate counts instead.

## Built in

All changes are exact unless noted. The last column says how each one
reaches the joined detector. Rows with a commit also run in the research
chain, since both call the same research modules. Over the 28 views, the joined detector ran 8–12% faster than
the research chain, about the size of run-to-run noise
([check](../court_detector/check_20260925/README.md)).

| Change | Commit or file | Result | Detail | In the joined detector |
| --- | --- | --- | --- | --- |
| Player test only on courts with valid geometry (patch 1). The matrix product below replaced it | a2e24ddf | With patch 2: six views, 6,255 → 2,572 s (2.4×) | [Item 1](../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md#1-land-patches-1-and-2) | Through item 3, which replaced it |
| Stripe evidence measures only fragments with a matching direction (patch 2) | a2e24ddf | Shared with patch 1 | [Item 1](../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md#1-land-patches-1-and-2) | Yes |
| Distance maps built without IPP, Intel's optimised routines inside OpenCV | 7b56d58e | Maps 12–16% faster. Repeat runs became bit-identical, so later changes could be checked bit for bit | [Item 2](../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md#2-make-the-distance-maps-deterministic) | Yes |
| Player test as one matrix product per frame (F5) | 888cf999 | Player test 6,429 → 52 s over the 20 court views | [Item 3](../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md#3-replace-the-player-test-with-a-matrix-product) | Yes |
| Player check before axis scoring (F1) | 888cf999 | Axis matching 4,579 → 3,553 s. With the row above: 20 court views, 20,647 → 13,942 s | [Item 5](../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md#5-gate-axis-hypotheses-on-players-before-scoring-them-f1) | Yes |
| Refit distance maps built once per view | d1609d6c | 412 → 85 s over 28 views | [Item 4](../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md#4-build-the-refit-distance-maps-once-per-view) | Yes |
| x and y kept as separate arrays in three hot spots; shortlist records built on demand | d1609d6c | 13,002 → 8,931 s over 28 views (−31%) | [Item 12](../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md#12-tested-stop-scoring-courts-that-cannot-make-the-shortlist) | Yes |
| Camera check with x, y and w as separate arrays | 44254b42 | 564 → 92 s over 28 views | [Item 13](../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md#13-done-rewrite-the-template-camera-check-commit-44254b42) | Yes |
| Cache for `prepare_observations` | 7c74b015 | 100 → 21 s over 28 views. With the row above: 8,931 → 8,250 s | [Item 14](../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md#14-done-cache-prepare_observations-commit-7c74b015) | Yes |
| Seated people left out of the player test. Not exact | `court_detector/feet.py` | Axis matching −27%, run −5%, all 20 courts kept. Three courts moved 0.5–3.9 px, each as close to the reference or closer, bar 0.04 px on one. This 5% is already in the later timings | [Item 10](../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md#10-bake-in-leave-seated-people-out-of-the-player-test) | Yes |
| No research-only files, such as populations, case records and arrays | Joined detector | About 580 s (6.5%) of the 8,931 s research run | [Deployment blocker](../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md#deployment-blocker-w5-reads-git-ignored-control-files); [STRIPPED.md](../court_detector/STRIPPED.md) | Yes. It writes files only when asked to |
| No legacy evidence on the search's kept courts | Joined detector (`legacy_evidence=False`) | 480 s in the 13,942 s run; smaller since item 12, not re-measured | [Item 7](../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md#7-skip-legacy-pool-evidence-in-a-deployment-mode) | Yes (`legacy_evidence=False`) |

## Decided against

The ideas that would save the most drop candidates early. They can lose the
right court on a view nobody has tested. The others save too little for their
cost.

| Idea | Why not | Detail | In the joined detector |
| --- | --- | --- | --- |
| A screen that keeps only 12 of the up to 16 line directions before the search (SVD12) | It roughly halved run time in the 24 September research run, before items 3 and 5; its gain on the joined detector is not measured. It changes the court on 5 of 20 court views, and `letterboxed_short_frame_78` and `shuttleset_21_scene_0044` get clearly worse | [Evaluation](../archive/20260925_optimisation_handover/CLAUDE_EVALUATION.md#svd12-screen-against-full-search) | No. It searches all directions |
| Upper bound to skip courts that cannot make the shortlist | The bound holds but is loose. It costs 53–60% of full scoring and skips little | [Item 12](../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md#12-tested-stop-scoring-courts-that-cannot-make-the-shortlist) | No |
| Coarse scores, then exact scores for the top K only | It rebuilt every shortlist in the probe, with a projected saving of about 8% of the run; no end-to-end speed-up was measured. But K is fixed before a view is seen, and one call needed K = 4,511 with no warning at 4,096 | [Item 16](../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md#16-tested-score-coarsely-then-exactly-only-the-top-k) | No |
| Count only the people who move most | Top 6 movers fails the player test on 9 of 20 real courts | [Item 11](../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md#11-rejected-count-only-the-people-who-move-most) | No |
| Cap people by detector score | Players in motion, occluded or behind the net can score below spectators (the project owner's call) | [What not to do](../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md#what-not-to-do) | No |
| Shrink the 256-court shortlists | The accepted court's parent ranked as low as 41st in its pair and 107th overall. A cap of 32 per pair loses 1 of 20 courts | [What not to do](../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md#what-not-to-do) | No. It keeps 256 per pair and 256 overall |
| Camera check before court scoring | A court that fails the camera check can yield a refit that passes | [What not to do](../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md#what-not-to-do) | No |
| Refit only the courts near the best score | Skips 14–58% of refits, but the safe margin is empirical | [Item 15](../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md#15-w5-savings-not-worth-doing-now) | No |
| Camera-check only the courts selection reads | About 1% left after item 13. Loses metadata that `compare_directional_runs.py` validates | [Item 15](../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md#15-w5-savings-not-worth-doing-now) | No |
| Faster scoring-stage measurement internals | No single hotspot is left after item 12 | [Item 15](../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md#15-w5-savings-not-worth-doing-now) | No |
| gzip level 6 for research records | About 0.6%. It breaks byte comparison with older runs, and the joined detector skips these files anyway | [Item 15](../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md#15-w5-savings-not-worth-doing-now) | No |
| Score bit-identical axis hypotheses once | Two sampled views had 0 duplicates among 321,224 scored hypotheses | [Item 12](../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md#12-tested-stop-scoring-courts-that-cannot-make-the-shortlist) | No |
| float16 camera check | 5–6× slower than float64 on a CPU. It overflows on real homographies, with errors up to 0.074 against a 0.1 gate | [Item 17](../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md#17-tested-float32-and-float16) | No |
| float32 for the camera check alone | 2.3–3.8× faster, but saves only about 50 s (0.6%). Stored values would no longer match byte for byte | [Item 17](../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md#17-tested-float32-and-float16) | No |
| Hoist the basis inverse (F2) | Under 1% of pair time | [Verdicts](../archive/20260925_optimisation_handover/CLAUDE_EVALUATION.md#claim-by-claim-verdicts) | No |
| Affine scoring kernel (F3) | Changes rounding and needs fallback code. F1 got the same saving exactly | [Verdicts](../archive/20260925_optimisation_handover/CLAUDE_EVALUATION.md#claim-by-claim-verdicts) | No |
| Project corners once (F5b) | 2–4% of pair time | [Verdicts](../archive/20260925_optimisation_handover/CLAUDE_EVALUATION.md#claim-by-claim-verdicts) | No |
| Per-pair distance-map cache (F7) | About 20 ms per pair, and keys rarely repeat | [Verdicts](../archive/20260925_optimisation_handover/CLAUDE_EVALUATION.md#claim-by-claim-verdicts) | No |
| Batched gate evidence (F9) | 1% of pool evidence | [Verdicts](../archive/20260925_optimisation_handover/CLAUDE_EVALUATION.md#claim-by-claim-verdicts) | No |
| Batched stripe measurement (F10) | Patch 2 removed the same cost in about 10 lines | [Verdicts](../archive/20260925_optimisation_handover/CLAUDE_EVALUATION.md#claim-by-claim-verdicts) | No |
| Refit cache and analytic Jacobian (F11) | `least_squares` is about 4% of the refit stage | [Verdicts](../archive/20260925_optimisation_handover/CLAUDE_EVALUATION.md#claim-by-claim-verdicts) | No |
| `cv2.polylines` for map drawing | The distance transform is 90% of map cost | [What not to do](../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md#what-not-to-do) | No |
| Split `player_fractions` into chunks to cap memory | The matrix-product player test no longer builds the large array | [Item 3](../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md#3-replace-the-player-test-with-a-matrix-product) | No |
| Prepared-structure types; caches with hit-rate reports | The costs they target are under 1% of pair time | [What not to do](../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md#what-not-to-do) | No |

## How to check a speed-up

Follow the joined detector's check: the steps are in
[WIRING.md](../d17_timing/WIRING.md#how-to-check-an-integrated-detector) and
the 25 September results are in
[check_20260925](../court_detector/check_20260925/README.md). A speed-up also
needs these checks:

- **Exact changes in the research modules.** Run the research chain on all 28
  views before and after the change. Compare the two run folders with
  `claude_evidence/exact_rewrites/compare_exact_runs.py`. It compares floats
  by their bits and arrays by dtype, shape and bytes. It skips timings and
  memory, and ignores the run folder's own path. It lists how often each
  stage ran for you to review, rather than failing on a change.
- **Exact changes in the joined detector.** Run `run_views.py` with
  `--baseline ARM_DIR --feet FEET_FILE --artefacts` against the saved research
  run, as the [check](../d17_timing/WIRING.md#how-to-check-an-integrated-detector)
  did.
- **Speed.** Carmack is shared, so one view's time can swing by up to 40%
  between runs. Totals over many views agree to about 5%. Run the before and
  after arms side by side, at most 8 processes in all, and compare totals
  ([verification plan](../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md#verification-plan)).
- **Python.** Use the `court_det` venv on Carmack: Python 3.12.13, numpy
  2.5.3, SciPy 1.17.1 and opencv-contrib-python 5.0.0.93. Compare only runs
  made in the same venv, since other numpy builds can differ in the last bits.
- **Changes that are not exact.** Compare each view's chosen court and
  reference error with the baseline. `claude_evidence/fresh_feet/compare_feet_runs.py`
  and `claude_evidence/d17/court_plane_error.py` do this. Recheck the eight
  non-court controls too. Fresh person detections already put a court on
  control 100347 ([finding](../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md#fresh-detections-put-a-court-on-a-non-court-control)),
  so any change to which people count needs that recheck.

New speed-up work goes on `exp/court-det-opt2`. The merged work above is on
`fix/court-det`.

## Where the history went

| Record | Where it is now |
| --- | --- |
| Every speed-up tried, with its measurements and reasoning (items 1–17) | [Archived speed-up list](../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md) |
| The 24 September evaluation of the web UI's packet: profiles, whole-view runs, the cause of score drift, verdicts on F1–F12 | [Archived evaluation](../archive/20260925_optimisation_handover/CLAUDE_EVALUATION.md) |
| The web UI's 23 September packet: overview, static audit, patch queue, experiment protocol and source map | [Archived packet](../archive/20260925_optimisation_handover/webui_packet/README.md) |
| Measurement scripts and their results | [claude_evidence/](claude_evidence/README.md), still in this folder |
| The packet's single-file copy, local-model prompt, task manifest and `tools/` | Deleted. Git history has them at commit 92535b6e |
