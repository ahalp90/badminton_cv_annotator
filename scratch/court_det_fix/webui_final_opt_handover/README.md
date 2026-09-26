# Web-UI speed-up proposals and where they fit

On 25 September a web-UI model wrote two sets of speed-up proposals for the
court detector. It reviewed commit `a53b8a0e`, just before the joined detector
was merged. This page sorts its proposals by where they stand now, then
suggests an order for what to try next.

The short version:

- Its first recommendation, one detector that passes results in memory, is
  built. That is the joined detector
- Its coarse-then-exact scoring cascade is item 16, which was tested and set
  aside. It is now to be tried, with K = 2,048 as the default and a switch to
  score every court
- Its 128-court cap on each search's overall shortlist was replayed on 26
  September. It kept every chosen court but cut close backups on six hard
  amateur views, so both caps stay at 256
- Parallel direction pairs and court reuse across scenes are already on the
  open list. The web-UI adds useful design detail to both
- Not from the web-UI: an upright-camera filter, added on 26 September, cut
  the 28 views' time by 42%. It exposed a scoring weakness: two views now slip
  one line at the far end. The cascade and cap numbers below were measured
  before it, so they need measuring again
- Other new ideas: running the scoring stage's candidates in parallel, a small
  rewrite of court scoring, and a GPU version of the search
- The web-UI sized everything against 90 s per scene. The target is 30 s per
  five-minute video, with 90 s as the upper end. So a video can afford only a
  few full searches, and each must be much faster than now

The two source folders are
[`court_detector_architecture_handover_2026-09-25/`](court_detector_architecture_handover_2026-09-25/README.md)
and [`webui_numba_cuda_128.md`](webui_numba_cuda_128.md). The current speed-up
status is in the
[speed-up README](../court_detector_optimisation_handover/README.md).

## Terms

- **Joined detector**: `court_detector/`, the accepted method as one piece of
  code that passes results between steps in memory
- **Research chain**: `d17_timing/run_d17.py`, which runs the same method
  through the research scripts and writes files between stages. The web-UI's
  timings come from the research chain
- **Court search**: finds up to 16 main line directions in a view and builds
  candidate courts from each **direction pair**, about 240 pairs a view. It
  runs on all line fragments (G0) and again on paint-coloured ones (G1)
- **Shortlist**: the best distinct courts kept at a step. Each pair keeps up to
  256, then each search (G0 or G1) keeps up to 256 overall from those
- **Line templates**: up to 256 more candidates, built from rectangles of
  crossing lines
- **Scoring stage** (W5): measures each candidate against the painted stripes,
  refits it to them and ranks the results. A candidate is a **parent**; its
  refit is a **child**
- **Net choice**: picks the final court from the scoring stage's ranked courts,
  with a small reward for net posts that line fragments support
- **Control views**: the 8 test views with no court
- **Exact**: saved results bit-identical to the run before a change
- **Item N**: a numbered entry in the archived
  [speed-up list](../archive/20260925_optimisation_handover/CLAUDE_FOLLOWUPS.md)
- **Times**: process seconds on one core, summed over the 28 test views, on
  Carmack, the shared compute server

## What the target means

The target is about 30 s for a whole five-minute video, with 90 s as the upper
end and start-up reported separately
([pickup.md](../pickup.md#compute-requirement-and-next-work)). The web-UI read
90 s as a limit per scene. A broadcast video has many scenes, so the real
budget is much tighter.

In the 25 September check, the joined detector took 6,434 s over the 28 views
([timing.txt](../court_detector/check_20260925/timing.txt)):

| Step | Seconds | Share |
| --- | ---: | ---: |
| Court search, all fragments (G0) | 3,346 | 52% |
| Court search, paint fragments (G1) | 785 | 12% |
| Scoring stage | 1,837 | 29% |
| Line templates | 409 | 6% |
| Feet, set-up, net choice and stripe refit | 53 | 1% |

That is 230 s a view on average and 546 s at worst
(`gxBQ_window_00_frame_689`). The web-UI quotes 295 s and 666 s, which are the
research chain's earlier figures. No whole video has been timed yet, because
the joined detector still takes prepared inputs for one view at a time.

The upright-camera filter (26 September) skips courts that need a camera on
its side or upside down. With it, the same run took 3,703 s: 132 s a view on
average and 264 s at worst. The G0 search fell to 951 s and G1 to 284 s, so
scoring, at 1,968 s, is now the largest step
([check](../court_detector/check_20260926_upright/README.md)).

Two things follow from the per-video budget:

- **A video can afford few full searches.** Reusing a court across scenes from
  the same camera is what makes that possible
- **Scenes with no court still cost time.** The 8 no-court test views took
  20–65 s each, mostly line templates and scoring the 256 templates. Neither
  web-UI document covers this, and a broadcast video has many cutaways

The hardware brief: CPU-only running must keep working, even if slow. The GPU
target is a card of roughly V100 speed with 16 GB of memory, on CUDA 13 or
later. Development runs on an L40 with 48 GB.

## Where each proposal stands

| Proposal | Where it stands | Suggested next step |
| --- | --- | --- |
| One in-memory detector | Built: the joined detector | None |
| Coarse 16-sample score, then exact scores for the top K | Item 16 set it aside. Now to be tried (your decision, 26 September) | Build with K = 2,048 as the default and a switch for no cap |
| Cap each search's overall shortlist at 128 | Replayed 26 September: same chosen courts, but cuts close backups on six hard views | Keep 256 |
| Cap each pair's shortlist at 128 | New. Saves about 0.5% | Drop |
| Search direction pairs in parallel | Item 8, open | Build |
| Score parents and refits in parallel | New | Build |
| Reuse a court across scenes | Item 9. Plan agreed 26 September | Build: warp the borrowed court, re-check it, fall back to a full search |
| GPU search | New | Prototype |
| Compiled loops (numba) on the CPU | Open list; web-UI adds synthetic timings | Decide with the CPU/GPU question below |
| Gaussian response computed once per map | New. Bit-identical in a synthetic test; untested in the detector; at most about 3% | Optional, on large pairs only |
| Candidate objects only for survivors; larger batches | Item 6 and the open list | Unchanged: small |
| Scoring courts from 1-D line profiles | New, not exact, speculative | Skip for now |
| Plug-in backend classes and four deployment modes | New | Skip: more structure than needed now |

The sections below cover the rows that need more than a line.

## Already built: one in-memory detector

The web-UI said the method ran end to end only through the research chain,
with code swapped in at run time and files written between stages. That was
true of the commit it reviewed. The joined detector now does what it asked
for. It calls the unchanged research functions in order and passes their
results in memory. On all 28 views it passed the 13 checks against the
research chain, with the chosen courts identical bit for bit
([check](../court_detector/check_20260925/README.md)). Two limits: the line
templates were compared by count and metadata only, and the scoring-stage
comparison leaves out research-only fields.

Skipping the research files and the legacy pool evidence was the saving the
web-UI expected from this. In the check, the joined detector ran 8–12% faster
than the research chain, which is about the size of run-to-run noise.

The web-UI also sketched plug-in backend classes and four deployment modes.
The joined detector's steps (search, templates, scoring, net choice, stripe
refit) already give the seams those were for. A backend switch is worth adding
once a second backend exists, as a plain argument.

## The cascade: item 16, now to be tried

The web-UI's cascade scores every court in a pair cheaply, with 16 samples per
marking, then scores only the top K exactly with the usual 64. So full scoring
is capped at K courts a pair. The cheap pass is what picks those K: a pair
builds up to 262,144 courts in the order it combines line guesses, not by
quality.

Item 16 tested it by replaying 3,369 scoring calls on the 23 views that reach
a search ([summary](../court_detector_optimisation_handover/claude_evidence/prefilter/summary.txt)).
At K = 8,192 every pair's shortlist came out the same. At K = 4,096 one call
lost a court: it needed 4,511, and the missed court's coarse score sat 0.001
below the cut.

Item 16 set this aside because K is fixed before a view is seen, and a miss
gives no warning. The web-UI agrees that K = 4,096 needs a fallback, and lists
possible triggers but tested none. Its "shadow mode" runs the full exact pass
alongside, so it is a test tool rather than a saving.

**Which courts need to get through (26 September).** Item 16 asked for every
pair's whole shortlist to survive. Most of those courts never reach the
search's overall shortlist, so they cannot change the result. Joining the same
replay to the search records shows how deep in its pair's cheap order each
court that matters sat
([k_depth_summary.txt](../court_detector_optimisation_handover/claude_evidence/prefilter/k_depth_summary.txt)):

| Courts | Deepest cheap rank within its pair |
| --- | ---: |
| The chosen court | 189 |
| The net choice's top five | 367 |
| Each search's overall shortlist | 1,211 |
| Every court any pair kept | 4,511 |

At K = 2,048 the pairs lose 217 courts from their shortlists. Every one scores
below its search's overall cut, so on all 23 views the overall shortlists stay
the same. The scoring stage's inputs and the chosen courts should then stay the
same too. At K = 1,024, 18 lost
courts could change an overall shortlist; at K = 512, 1,233 could
([dropped_courts.txt](../court_detector_optimisation_handover/claude_evidence/prefilter/dropped_courts.txt)).

The costs below are estimates from timed pieces, not an end-to-end timing.
Court scoring is roughly 30–33% of the joined detector's run.

| K | Cost against today's court scoring | Saving on the joined detector's run | Overall shortlists unchanged on the 23 views |
| --- | ---: | ---: | --- |
| 8,192 | 66% | 10–11% | Yes, and every pair's shortlist too |
| 2,048 | 51% | 15–16% | Yes |
| 512 | about 45% | 16–18% | No, 12 of 23 change |

The cheap pass alone costs about 43% of today's court scoring, so a K below
about 2,048 buys little. This is a CPU saving: on a GPU, scoring every court
is cheap anyway.

Eight samples at K = 4,096 also keeps every overall shortlist, at about 47%.
Sixteen samples stays, because its cheap order tracks the full score more
closely. At K = 4,096 it keeps every pair's own shortlist whole on 22 of 23
views, against 2 of 23 with 8 samples. Four samples is too coarse: one
top-five court sat at cheap rank 16,169.

**Decision (26 September): try it, at K = 2,048.** A command-line switch
scores every court for a hard video. On the 23 views, 2,048 is 1.7 times the
deepest court in an overall shortlist, 5.6 times the deepest top-five court and
11 times the deepest chosen court.

What it needs:

- **The change** is in `propose_role` (`run_given.py`), which the joined
  detector and the research chain share. When a pair has more usable courts
  than K, score them all with 16 samples and keep the top K. Put those back in
  their original order before the exact scoring, since the shortlist breaks
  score ties by input order. The arrays that record each candidate's origin
  take the same subset
- **The setting** passes down through `automatic_generation.generate`, with a
  flag on `run_views.py` that sets K or turns the cap off
- **The check** is a 28-view run at the default K. The replay predicts the
  same overall shortlists as the baseline, and so the same scoring results and
  chosen courts. Pair records differ where a pair lost courts below the cut,
  and so do timings and scoring counts
- **A depth log.** For each search, log the deepest cheap rank, within its
  pair, of any court in the overall shortlist. A run then shows how close each
  view came to K. It cannot show a court the cut already dropped, so a view
  that comes close is worth rerunning with no cap

The switch only helps if someone suspects a view is hard. Before relying on
the default, measure these depths on more views. The observation-only hook
(`../court_detector_optimisation_handover/claude_evidence/prefilter/measure_prefilter.py`)
records the cheap ranks, and `k_depth.py` beside it joins them to the search
records. If the deepest overall-shortlist court stays well under 2,048 (1,211
here), the default holds. If it creeps up, raise K.

**Experiment to do: rank by the line-guess scores instead.** A cap needs a
ranking to choose which K courts get the full score, and the cheap pass costs
about 43% of today's court scoring. The order a pair builds its courts in is a
poor ranking. It takes two lists of line guesses, each sorted by score, and
builds horizontal first: every court from the best horizontal guess, then
every court from the second best, and so on. Capping at 2,048 in that order,
with no cheap pass, misses the chosen court on 6 of the 16 court views the
search wins
([build_order.tsv](../court_detector_optimisation_handover/claude_evidence/prefilter/build_order.tsv)).
On `gxBQ_window_00_frame_0` the right court is built 5,047th in its pair, but
ranks 3rd on the cheap score.

Every court also has a free combined score: the average of its two line-guess
scores, which `propose_role` already computes (`axis_scores`). If the courts
that matter rank shallow by that average, full scoring could be capped with no
cheap pass. That might save roughly 27–30% of the run instead of 15–16%, by the
same rough estimate. To test it from the saved records:

- Rank each pair's courts by that average, and find how deep the chosen,
  top-five and overall-shortlist courts sit, as `k_depth.py` does for the
  cheap score
- Find the smallest K that leaves every overall shortlist unchanged, as
  `dropped_courts.py` does

The web-UI advises against this average as the final ranking. Here it would
only choose which courts get the full score.

## Shortlist caps of 128

**Result (26 September): keep both caps at 256.** A replay of a 128 cap kept
every chosen court on the 28 views. But on six hard amateur views, the courts
it cuts include close backups for the right court.

The search has two caps, both 256: one on each pair's shortlist and one on
each search's overall shortlist.

**The per-pair cap barely matters.** Every court is still scored; only the
greedy pick that builds the shortlist does less work. That pick took 86 s of
the research chain's 8,250 s, so halving it saves about 0.5%. The web-UI's
estimate of 1–3% also counts smaller savings in building and pooling
candidates, which nobody has measured.

**The overall cap would cut the scoring stage's work.** G0 and G1 each send up
to 256 parents to the scoring stage, and the line templates another 256.
Capping G0 and G1 at 128 cuts the 28 views' parents from 17,741 to 12,491,
after merging duplicates. If the stage's time follows its parent count, that
saves about 540 s, or 8% of the joined detector. The web-UI estimated 5–15%.
Neither figure is timed. The cut parents also yield valid refits more often
(87% against 80%), so they may cost more than average.

**The replay needed no rescoring.** Each court is scored on its own, and the
net choice takes the best combined score, so a cap can only delete courts. The
first 128 of each saved 256 list are exactly what a cap of 128 keeps, because
the pick walks courts in score order. The replay reruns the ranking on the
surviving courts and filters the saved net-choice rows
([replay](../court_detector_optimisation_handover/claude_evidence/shortlist_cap_replay/)).

What it found:

- **Every chosen court survives.** That covers all 22 views where the net
  choice picks a court: the 20 court views and two control views. The other
  six control views still get none
- **The margin is thin.** The deepest winning parent sat at rank 107 of 128 in
  G1, and at 96 in G0
- **On 14 of the 20 court views, places 2–5 of the net choice stay the same**
- **On the other 6, the cap removes close backups.** All six are amateur or
  hall footage (am2, am3 and gxBQ)

`gxBQ_window_00_frame_0` shows why. The net choice's top six are all the right
court, reached by different routes and within 21 px of each other. The caps
keep two of them (G0 ranks 96 and 104) and cut four (G0 145; G1 198, 237 and
247). G0 ranked the right court 96th, behind wrong courts from other pairs,
most of them largely off-frame. Its search scores are bunched: 0.631 at rank 1,
0.492 at 96 and 0.483 at 128.

Capping one search at a time:

| Across the 20 court views | Cap G0 only | Cap G1 only |
| --- | ---: | ---: |
| Parents kept | 84% | 86% |
| Deepest winning parent | 96 | 107 |
| Views where a cut court placed 2nd or 3rd | 0 | 5 |
| Views where the top five change | 0 | 6 |

Losing the winner's own parent would move the pick 1–5 px on the nine
`shuttleset_*_scene_*` views, and 12–34 px on the other eleven court views.

So the cap might save about 8% on these 28 views, but it removes the backups on the
hard frames this detector exists for. The web-UI's fallback, keeping 32 more
courts from ranks 129–256 for variety, is moot at 256. If the search moves to
a GPU, the scoring stage becomes most of the CPU time, and the cap may be worth
another look on a larger set of hard views.

## Parallel work inside one view

**Direction pairs (item 8).** Pairs are independent until their shortlists
merge. Results stay exact if they merge back in the original pair order. The
recorded run gives a fair idea of the gain. No pair took more than 7.8 s, and
pairs made up 91% of the search time. Packed onto 16 workers, the slowest
view's pairs (418 s in that run) would take about 26 s, plus about 28 s of
other search work. So no single slow pair would hold it back. The recorded
times come from runs sharing Carmack with 7 other processes, so real scaling
needs a measurement.

**Scoring-stage parents and refits (new).** After duplicates are merged, each
parent's measurement is independent, and so is each refit. The work comes in
small pieces: on the slowest view, 768 parents took about 60 s to measure and
69 s to refit. So it should spread well over workers, though that is
untested. The web-UI's plan keeps the result exact: merge duplicates first,
restore the original order, then rank on one core as now.

Parallel work cuts one view's wait, not its CPU seconds. It pays off when
fewer views than cores are running, which court reuse makes the usual case. It
helps both the CPU-only and the GPU setups, because the GPU plan leaves the
scoring stage on the CPU.

The web-UI's latency table assumes 92% of the work runs in parallel. The
recorded pair times above are firmer evidence for the search part.

## Court reuse across scenes (item 9)

The web-UI's design: when a scene's camera matches an earlier view that has a
court, try that court first. Re-measure it on the new frame, refit it, and try
a small grid of nearby courts. Accept it only if the usual checks pass;
otherwise run the full search. A reused court must never skip the no-court
checks just because an earlier scene had a court.

The pieces for camera matching exist in `src/annotator/court_views.py`, but not
the whole step. It compares perceptual hashes of scene frames, then aligns two
images around an accepted court's corners. A match needs a correlation of at
least 0.8, with the corners moving at most 1 px. Today the pipeline runs it
after every scene has been searched, to group three or more scenes that
already have courts. Reuse needs a different step built from those pieces:
match one new scene against earlier scenes with a court, before searching it.

Putting the borrowed court first saves nothing by itself, because the search
scores every candidate before it picks. The saving comes from a rule that
skips the search when the borrowed court is good enough. Today's detector has
no absolute score threshold, only a ranking, so that rule is new.

**The plan (agreed 26 September).** The same policy runs on the CPU and GPU
paths, so a video gets the same court on both and the hardware only changes the
speed.

- **Move the court with the camera.** The alignment step in `court_views.py`
  (`_view_alignment`) already fits a warp between the two images, though it
  returns only pass or fail today. Have it return the warp too, and apply the
  warp to the borrowed court's corners, rather than trying the web-UI's grid
  of nearby courts
- **Re-check it on the new scene.** Accept the moved court when, after the
  stripe refit, it passes the usual full-court checks with the new scene's
  feet. Its stripe evidence must also be close to what it scored on its own
  scene, and the refit must move its corners only a little. "Close" is one
  tolerance, tuned on scene pairs known to share a camera
- **Otherwise run the full detector.** No-court answers stay as they are only
  if the re-check never passes a court on a cutaway, so cutaways belong in its
  tests
- **Flag disagreement.** When a fallback search finds a different court for
  the same camera, flag that camera's group. The first search may have been
  the wrong one
- **A setting for full searches per camera before reuse.** The default is 1.
  Raising it to 3 brings back today's check, where each group takes the median
  of at least three searched courts, so one bad search is outvoted. At about
  35 s a search even with the GPU (estimate below), 3 does not fit the budget
  yet

This trades today's outvoting for speed. It matters less than it sounds:
scenes from one fixed camera have near-identical lines, so their searches tend
to fail the same way. The outvoting mainly protects against failures in one
scene, such as a player hiding a line, an overlay or a lighting change.

The web-UI's
[runbook](court_detector_architecture_handover_2026-09-25/ACCEPTANCE_AND_BENCHMARK_RUNBOOK.md#7-previous-view-reuse-gate)
lists test cases worth keeping: adjacent scenes from one camera, lighting or
player changes, a crop or letterbox change, a cut to another camera, and
replays.

## GPU search

The idea is to move the array-heavy parts to the GPU: axis scoring, court
scoring and later stripe measurement. Python bookkeeping, the greedy picks,
fitting and ranking stay on the CPU. The web-UI ran no GPU code, so all its GPU
figures are estimates:

- **Memory.** One pair has at most 262,144 candidate courts. Their geometry and
  scores take a few tens of MB. Only one array is large: all courts × 12
  markings × 64 samples, about 768 MiB in float32. Processing courts in chunks
  avoids it. By this estimate 16 GB is ample, but peak use with several scenes
  at once needs measuring
- **Gain.** If the three hottest functions ran 20× faster, the research chain
  would run 2.1× faster. The arithmetic is right; the 20× is an assumption. The
  joined detector skips the research chain's file writing, so those functions
  are probably a larger share of its time. It does not time single functions,
  so that is unmeasured
- **One GPU process for all scenes.** Several processes on one GPU take turns,
  so one worker serving every scene makes sense

My rough extension, also an estimate: an average view spends 148 s in the
search, 66 s scoring and 15 s on line templates. With the search 20× faster and
scoring on 8 cores, an average view would take about 35 s and the slowest
about 55 s. The line templates, which run on one core, would then be the
largest piece.

Card choice matters for one thing. Apart from datacenter cards such as the
A100, most cards of that class compute float64 slowly. The L40 runs it at 1/64
of its float32 speed, while the V100 itself ran it at half speed. The
detector's geometry is float64. Heavy fused kernels would suffer most, and
plain array code, which mostly waits on memory, less. How much either suffers
needs a measurement on a card of the target class.

## Keeping the CPU and GPU paths in step

CPU-only running must keep working, both paths should get faster, and the
maths should not be copied where the copies could drift. There are three ways
to split the work:

| Approach | Copies of the scoring maths | CPU path faster? | Drift risk | Main cost |
| --- | --- | --- | --- | --- |
| Run the same array code on numpy or a GPU array library (such as CuPy), passing the library in | One | No | Lowest | A port: the hot functions build maps with OpenCV, loop in Python and call numpy directly. Chunk to limit GPU memory |
| Write each court's maths once as a loop body and compile it for both CPU and GPU (the Numba idea) | One | Yes | Low | A compiler dependency; results shift once in the last bits, since compiled loops sum in a different order |
| Separate GPU kernels beside the numpy code | Two | No | Highest | Fastest GPU code, but two copies to keep aligned |

The web-UI's split between GPU and CPU helps with the rest. The GPU scores
every court, then the CPU rescores the best of them with its own code and
makes the pick. The final pick then always comes from the CPU code, so both
setups share one reference.

The catch is how many courts to rescore. GPU and CPU scores will not match bit
for bit, even from the same source. The live code turns each projected sample
into a whole pixel, so a last-bit difference at a pixel boundary reads a
different map value. The rescored set must be wide enough to cover those
differences, and nothing bounds them yet. That is the same kind of question as
item 16's K. Both sides compute the same formula, so the gaps may well be
small, but only a measurement will tell. During evaluation, run the full CPU
pass alongside and count how often the rescored set misses a court that a
CPU-only run keeps.

Drift between the paths is easier to catch. Each run can compare GPU and CPU
scores on the rescored courts and stop if they differ by more than the
evaluation found normal. A formula changed on one side only shows up there
straight away. This check cannot find a court the GPU ranked too low to be
rescored; that is the width question above.

I suggest starting with the same array code, because it measures the GPU gain
without a second copy of the maths. The compiled approach is the next step if
that falls short, or if CPU-only speed matters enough.

## Suggested order

This is my suggestion, from the evidence above. It was written before the
upright-camera filter. The filter cut the search's time by about 70%, so the
cascade saves much less than estimated below, and scoring is now the largest
step. Re-measure both before building. The filter also moved
`gxBQ_window_00_frame_0`'s right court from G0 #96 to #1, so smaller G0 and G1
shortlists may now be safe. That is a question for later. Until scoring can
tell a court that slips one line at the far end from the right one, any change
to what reaches scoring can move results by chance, as the filter did.

1. **Build the cascade** with K = 2,048 as the default and a switch for no
   cap. Check it on the 28 views, time it, and measure the depth of the
   overall shortlists' courts on more views
2. **Search pairs and score parents in parallel** in the joined detector. Both
   are exact and help both setups. Measure real scaling at 8 workers on
   Carmack
3. **Build court reuse** as planned above: the pre-search match from the pieces
   in `court_views.py`, the warp, the re-check and the fallback. Tune the
   re-check tolerance on same-camera pairs, and look at the cost of no-court
   scenes at the same time
4. **Prototype GPU court and axis scoring** on the L40, using the same array
   code and the CPU rescore check. The web-UI's bar is a fair first gate: each
   function at least 10× faster than one CPU core, including copies to and
   from the GPU, and the whole search at least 2× faster. The L40 differs from
   the target class in memory and float64 speed, so confirm speed and a 16 GB
   peak on a target-class card before deciding
5. **Profile again**, then revisit compiled loops and float32

These steps only work towards the target. The real test is a whole
five-minute video, cutaways and reused scenes included, against the 30 s goal
and the 90 s upper end. That needs the input steps the joined detector still
lacks: line fragments, people and poses, and scene cuts from a new video. The
estimate of about 35 s for an average view, with the search on the GPU and
scoring on 8 cores, is progress towards the 90 s end, not the 30 s goal.

## Open questions

- How close must a borrowed court's stripe evidence be to its own scene's
  score? That tolerance needs tuning on known same-camera pairs
- On new footage, do the courts that reach each overall shortlist stay well
  within K = 2,048 in their pairs' cheap order?
- Does the free line-guess average rank the courts that matter shallow enough
  to replace the cheap pass? (experiment above)
- How many no-court scenes does a typical five-minute video have, and must
  each get the full detector?
