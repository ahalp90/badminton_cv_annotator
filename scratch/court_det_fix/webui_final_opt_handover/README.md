# Web-UI speed-up proposals and where they fit

On 25 September a web-UI model wrote two sets of speed-up proposals for the
court detector. It reviewed commit `a53b8a0e`, just before the joined detector
was merged. This page sorts its proposals by where they stand now, then
suggests an order for what to try next.

The short version:

- Its first recommendation, one detector that passes results in memory, is
  built. That is the joined detector
- Its coarse-then-exact scoring cascade is item 16, which was tested and set
  aside. The web-UI adds a fallback plan but has not tested it
- Parallel direction pairs and court reuse across scenes are already on the
  open list. The web-UI adds useful design detail to both
- New ideas: a 128-court cap on each search's overall shortlist, running the
  scoring stage's candidates in parallel, a small rewrite of court scoring,
  and a GPU version of the search
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
| Coarse 16-sample score, then exact scores for the top 8,192 | Item 16 tested it and set it aside | None unless a fallback rule can be proven |
| Cap each search's overall shortlist at 128 | New. Reopens "shrink the shortlists", decided against at 32 a pair | Cheap replay on saved shortlists; adopting it is your call |
| Cap each pair's shortlist at 128 | New. Saves about 0.5% | Drop |
| Search direction pairs in parallel | Item 8, open | Build |
| Score parents and refits in parallel | New | Build |
| Reuse a court across scenes | Item 9, open. The web-UI adds a design | Needs your rules for "same camera" |
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

## The cascade is item 16 again

The web-UI's cascade scores every court cheaply with 16 samples per marking,
then scores only the top 8,192 exactly. Its evidence is item 16's own replay
of 3,369 scoring calls
([summary](../court_detector_optimisation_handover/claude_evidence/prefilter/summary.txt)).
At K = 8,192 every pair's shortlist came out the same, for about 66% of the
exact scoring cost, or about 8% off the run. At K = 4,096 one call lost a
court: it needed 4,511, and the missed court's coarse score sat 0.001 below the
cut.

Item 16 set this aside because K is fixed before a view is seen, and nothing
cheap proves a new view stays inside it. The web-UI agrees that K = 4,096
needs a fallback. It lists possible triggers but has tested none. Its "shadow
mode" runs the full exact pass alongside, so it is a test tool rather than a
saving. Nothing here answers item 16's objection. If the search moves to a GPU,
scoring every court with all 64 samples becomes cheap, and the cascade matters
less.

## Shortlist caps of 128

The search has two caps, both 256: one on each pair's shortlist and one on
each search's overall shortlist.

**The per-pair cap barely matters.** Every court is still scored; only the
greedy pick that builds the shortlist does less work. That pick took 86 s of
the research chain's 8,250 s, so halving it saves about 0.5%. The web-UI's
estimate of 1–3% also counts smaller savings in building and pooling
candidates, which nobody has measured.

**The overall cap cuts the scoring stage's work.** In the recorded run, G0 and
G1 each sent up to 256 parents to the scoring stage, and the line templates
another 256. Over the 28 views that was 18,026 parents, or 17,741 once
duplicates were merged (`../court_detector/check_20260925/run_d17/logs/`).
Capping G0 and G1 at 128 leaves about 70% of them. If the stage's time follows
its parent count, that saves about 540 s, or 8% of the joined detector. The
web-UI estimated 5–15%.

**The accuracy question is a margin.** On the 20 court views the accepted
parent ranked no lower than 107th in its search's overall shortlist, so a cap
of 128 keeps every one. But "shrink the shortlists" is on the decided-against
list, and a 21-place margin measured on 20 views is the same kind of rule as
item 16's K. The web-UI's replay is still cheap and worth running on Carmack:

- Keep the first 128 of each saved G0 and G1 shortlist. The pick walks courts
  in score order, so this is exactly what a cap of 128 would have kept
- Rerun scoring, net choice and stripe refit
- Compare each view's chosen court, including the 8 no-court views

The replay can show the cap is harmless on these views, but not on new ones.
Its value grows if the search moves to a GPU, because the scoring stage would
then take most of the remaining CPU time. If plain 128 loses a court, the
web-UI suggests keeping 32 more, chosen for variety from ranks 129–256.

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

Reuse would then re-check the borrowed court on the new frame, and run the full
detector when the match or the re-check fails. No-court answers stay as they
are only if the re-check never passes a court on a cutaway, so cutaways belong
in its tests.

Two decisions are yours: what counts as the same camera, and what check a
reused court must pass. The web-UI's
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

This is my suggestion, from the evidence above.

1. **Replay the 128 cap** from saved shortlists on Carmack. It is cheap and
   shows whether the cap changes any chosen court on the 28 views. Adopting it
   is your call, since it reopens a decided-against item
2. **Search pairs and score parents in parallel** in the joined detector. Both
   are exact and help both setups. Measure real scaling at 8 workers on
   Carmack
3. **Settle the reuse rules** and build the pre-search match from the pieces in
   `court_views.py`. Look at the cost of no-court scenes at the same time
4. **Prototype GPU court and axis scoring** on the L40, using the same array
   code and the CPU rescore check. The web-UI's bar is a fair first gate: each
   function at least 10× faster than one CPU core, including copies to and
   from the GPU, and the whole search at least 2× faster. The L40 differs from
   the target class in memory and float64 speed, so confirm speed and a 16 GB
   peak on a target-class card before deciding
5. **Profile again**, then revisit compiled loops, float32 and the cascade

These steps only work towards the target. The real test is a whole
five-minute video, cutaways and reused scenes included, against the 30 s goal
and the 90 s upper end. That needs the input steps the joined detector still
lacks: line fragments, people and poses, and scene cuts from a new video. The
estimate of about 35 s for an average view, with the search on the GPU and
scoring on 8 cores, is progress towards the 90 s end, not the 30 s goal.

## Open questions

- Which rules decide that two scenes share a camera, and what must a reused
  court pass?
- Is a 21-place margin on 20 views enough to adopt a cap of 128?
- How many no-court scenes does a typical five-minute video have, and must
  each get the full detector?
