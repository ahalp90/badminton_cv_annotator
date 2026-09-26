# Make the detector faster

This is the design reference for speed-up work. It owns the proposed changes,
constraints and checks. [pickup.md](../pickup.md) owns what to do next;
[the decisions](../DETECTOR_DECISIONS.md#d18) own accepted targets and past
results. This document replaces the design scattered across the old handovers.

The detector searches using all line fragments and using paint-like fragments.
The code calls these searches G0 and G1. It also builds courts from crossing
lines. It scores the resulting courts against painted stripes, refits them,
then chooses one. The code calls that scoring stage W5.

## Constraints

[D18](../DETECTOR_DECISIONS.md#d18) gives the whole-video time budget. It covers
a five-minute video, including cutaways and repeated camera views. Start-up
and warmup must be reported separately. A fast isolated scoring function is
only an intermediate result.

CPU-only use must keep working. The recorded GPU brief is roughly V100-class
speed with 16 GB memory and CUDA 13 or later; development uses an L40 with
48 GB. Check the actual supported hardware and libraries before choosing a
backend. In particular, an L40 and a V100 have very different float64 speeds.

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

Two approaches deserve a measured comparison:

- Run shared array operations on the CPU and GPU, with a small argument
  selecting the array library. Process courts in chunks to bound GPU memory
- Express the scoring maths once as loops that can be compiled for CPU and
  GPU. Numba is the proposed compiler; compilation, supported operations and
  rounding behaviour need a prototype

Neither approach has been measured in this detector. The old suggestion to
try array code first is a proposal, not a settled backend decision. Check
installed source and current official documentation before adopting a library.

The proposed split leaves bookkeeping, fitting and the final ranking on the
CPU. GPU code would first handle the large batches that score line matches
and possible courts. Keep the scoring stage on the CPU initially unless a
profile gives a reason to move it.

One proposed safeguard is to rescore the GPU's best courts with the CPU code
before choosing. The width of that set is unresolved. Tiny numerical changes
can send a projected sample to a different pixel, so a small floating-point
error does not itself bound score error. During tests, also run the full
CPU calculation and count any courts the GPU discarded too early.

Comparing CPU and GPU scores on the courts kept can expose drift between the
implementations. It cannot find a court the GPU discarded. Measure both risks.
Do not promise identical court choices until these checks support that claim.

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

The change belongs in `propose_role` in [proposals.py](proposals.py).
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

Use the API and runner in [README.md](README.md). The
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
  errors, then inspect meaningful changes. Keep the eight non-court views
  in the test. Preserve existing useful courts, including backups
- Compare paired before/after runs. At most eight jobs in total was the
  recorded protocol. Shared-server timing varies too much to infer a saving
  from one view or an unmatched old run
- The recorded Carmack environment was `court_det`: Python 3.12.13, NumPy
  2.5.3, SciPy 1.17.1 and OpenCV 5.0.0.93. Verify it before resuming, and use
  the same environment for both arms

A whole-video test still needs code that supplies lines, people, poses and
scene cuts. Include that remaining work when reporting gains.

## Sources

The [archived speed-up account](../archive/20260926/originals/court_detector_optimisation_handover/README.md)
and [web-UI comparison](../archive/20260926/webui_final_opt_handover/README.md)
retain the full reasoning, measurements and rejected options. The
[measurement index](../court_detector_optimisation_handover/claude_evidence/README.md)
locates scripts and saved outputs. Read them for a specific question; neither
owns the live work queue.
