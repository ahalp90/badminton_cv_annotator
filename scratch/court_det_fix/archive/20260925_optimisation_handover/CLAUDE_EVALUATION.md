> Historical record, filed on 25 September 2026. The
> [speed-up README](../../court_detector_optimisation_handover/README.md)
> owns current status; the [archive map](../README.md) records the original
> paths. Content is unchanged. `claude_evidence/` paths are under
> `court_detector_optimisation_handover/`; the deleted `tools/` folder is in
> git history at commit 92535b6e.

# Evaluation of the web-UI optimisation handover

**Bottom line.** The handover is right that the detector wastes a lot of work.
Its ranking of that waste is wrong, because it came from counting operations
instead of profiling. Two costs dominate the measured runtime:

- the player test projects every detected person's feet into every combined
  court, and about nine in ten of those people are off the court
- stripe evidence measures distances to every line fragment, then discards
  most of them for pointing the wrong way

The handover lists the first as a side note (F6). It misses the second. Its
first-wave items (F1–F4) target 0.5–21% of generator time on the pairs I
profiled. Its refit item (F11) targets about 4% of the refit stage.

Two small patches, about 20 changed lines, fix both dominant costs. On
Carmack's six-view SVD12 G0 run they cut total wall time from 6,255 s to
2,572 s (2.4×). Every case record, including the selected court, is identical
to the 23 September baseline. The only difference is float noise below 1e-8
in generator shortlist scores. That noise comes from OpenCV's IPP distance
transform, not from the patches (see "Score drift and its cause"). Commit
7b56d58e removes that noise, and repeat runs are now bit-identical.

A matrix-product form of the player test (the handover's F5) cut the heaviest
remaining cost by a further 100–225× on the pairs tested. It has since landed,
with the handover's F1; the last paragraph of this summary gives the whole-run
result.

The full D17 chain was then timed stage by stage on 28 views. When it searches
all 16 direction groups, it reproduces the accepted gallery court on every
court view, or improves on it. That run takes a median of 40 minutes per GX
view. The player test is 30% of its time and axis matching is 22%. The SVD12
screen searches only 12 of the 16 groups and halves run time. It also changes
the court on five of 20 court views, and two get worse. One is letterboxed,
where the court cuts into the playing area. See "The full D17 chain: full
search against the SVD12 screen".

Follow-ups 3 and 5 then landed as commit 888cf999. Rerunning the full search
with them gives bit-identical outputs on all 28 views. It cuts the 20 court
views from 20,647 to 13,942 s, a third less. The player test drops from
6,429 s to 52 s. Peak memory on GX views falls from 5.2–7.1 GB to
1.8–2.2 GB. See "After follow-ups 3 and 5".

Later exact rewrites (follow-up 12, commit d1609d6c) cut the full search on
all 28 views from 13,002 to 8,931 s, again bit-identical. Follow-ups 13 and
14 then took it to 8,250 s, still bit-identical. That run used the
seated-removed feet of follow-up 10, so its times do not compare directly with
the 13,942 s above. `CLAUDE_FOLLOWUPS.md` covers that work and what is left.

Date: 24 September 2026. Repository revision: `fix/court-det` @ `2a00f771`.
The patches are now commits on branch `exp/court-det-opt`: a2e24ddf (patches 1
and 2), 7b56d58e (deterministic distance maps) and 888cf999 (follow-ups 3
and 5).

## Conventions

- **View**: one frozen video frame with its line fragments and sampled player
  feet
- **Pair**: one ordered pair of vanishing directions. The fresh G0 generator
  lists 240 of these per view. In the SVD12 run its direction screen skips
  108, leaving 132 to try, and 22–80 of those reach matching
- **Axis hypothesis**: one way to line up court markings along one direction
- **Combined court**: one axis hypothesis from each direction, combined into a
  court homography. A pair can produce up to 262,144
- **Usable court**: a combined court that passes geometry and the player test
- **Foot samples**: one image position per detected person per sampled frame,
  including spectators, officials and people on neighbouring courts
- **Pool evidence**: `run_automatic.evaluate_pool`, which measures the 256
  courts the generator keeps
- **Refit stage**: W5 parent measurement, stripe refit and ranking, as run by
  `svd_search/run.py`
- **Bit-identical**: every stored number, array and string equal, checked
  recursively
- **Full search and SVD12 screen**: the generator sorts line fragments into
  16 vanishing-direction groups. The full search tries pairs from all 16. The
  SVD12 screen keeps 12 groups and skips every pair that uses the other four
- **D17 chain**: the accepted detector, end to end. Fresh G0 and
  paint-filtered G1 generation, seeded line templates, W5 measurement, refit
  and ranking, the bounded net choice, then the stripe-polarity refit
- **Working pixels**: pixels in the detector's working image. GX frames are
  1920×1080 and work at 960×540, so one working pixel is two frame pixels
- **Floor error**: how far a court point lands from its true position on the
  court floor, in centimetres. See "Pixel error and floor error"
- **IPP**: Intel's Integrated Performance Primitives, a vendor library that
  OpenCV calls for some image operations
- **Laptop**: Intel i7-6700HQ (AVX2), numpy 2.4.4 then 2.5.3, one numerical
  thread. **Carmack**: Xeon Gold 6448H (AVX-512), one numerical thread per
  worker. The six-view runs used numpy 2.4.6 in the shared `venv-rtmlib`; the
  D17 runs used numpy 2.5.3 in the `court_det` venv

## What I measured

I profiled the actual imported modules, confirmed with the handover's own
`tools/resolve_hotpath.py`. The active `projective_seed.py` is the
`next_steps_20260916/webui_seed/source/` copy. It is byte-identical to the
`frozen_helpers_20260914/axis_matching/` copy the handover names.

Measurements, in order:

- laptop profiles of single generator pairs on a broadcast view
  (`shuttleset_03_scene_0034`), an amateur view (`am2_window_01_frame_28019`)
  and a GX view (`gxBQ_window_00_frame_0`)
- laptop profiles of pool evidence and the complete refit stage on
  `shuttleset_03_scene_0034`, from its saved generation record
- a Carmack rerun of all six SVD12 views with both patches, compared with the
  23 September baseline run on the same machine and revision
- Carmack profiles and probes of the slowest remaining GX pairs

Harnesses, the patch diff, the Codex red-team report and every probe are in
`claude_evidence/`.

## Whole-view results on Carmack

Both runs used `svd_search/run.py --arm baseline`, at most six workers on 32
cores, one numerical thread per worker, numpy 2.4.6 and revision `14310f73`.
The only code difference is the two patches. The 23 September run also ran two
other arms in the same worker pool, so background load may differ slightly.

Wall seconds per view (Carmack):

| View | Generation before → after | Refit stage before → after | Total before → after | Speed-up |
| --- | ---: | ---: | ---: | ---: |
| am1 (`am1_window_00_frame_54`) | 524 → 211 | 59 → 49 | 585 → 263 | 2.2× |
| am2 (`am2_window_01_frame_28019`) | 1,090 → 142 | 108 → 29 | 1,199 → 173 | 6.9× |
| GX0 (`gxBQ_window_00_frame_0`) | 1,727 → 788 | 99 → 41 | 1,829 → 831 | 2.2× |
| GX5 (`gxBQ_window_00_frame_5`) | 1,929 → 745 | 83 → 63 | 2,015 → 811 | 2.5× |
| ss19 (`shuttleset_03_scene_0019`) | 284 → 210 | 84 → 59 | 370 → 271 | 1.4× |
| ss34 (`shuttleset_03_scene_0034`) | 178 → 166 | 78 → 55 | 258 → 223 | 1.2× |
| **All six** | 5,731 → 2,263 | 511 → 295 | 6,255 → 2,572 | 2.4× |

Generation here includes pool evidence. Preparation takes 1–3 s per view and
did not change. The patched run's largest single process peaked at 5.0 GB
(`/usr/bin/time -v`). The baseline run did not record memory.

GX views are now the slowest by far. "Finding 1" explains why.

## The full D17 chain: full search against the SVD12 screen

With the full search, the fresh D17 chain matches or improves on the accepted
gallery court on all 20 court views. Search dominates its run time: the G0 and
G1 search takes 78%, and the player test alone takes 30%. The SVD12 screen
halves the time but is not equivalent. It moves the corrected court on five
views and makes two of them clearly worse.

What ran:

- code: branch `exp/court-det-opt` at f72ed1c6, which holds patches 1 and 2,
  the deterministic distance maps and the timing harness
  (`scratch/court_det_fix/d17_timing/`)
- views: the 20 court views of the accepted gallery, plus the eight labelled
  non-court controls
- settings: net weight 0.04 and overrun 4.0 for the bounded net choice, and a
  visibility floor of (4, 3) for the line templates
- one process per view and search setting, up to 12 at a time on Carmack, in
  the `court_det` venv. Full-search and SVD12 jobs were interleaved in one
  batch, so both saw the same node load

All 56 jobs and the comparison exited 0. Outputs and scripts are in
`claude_evidence/d17/`.

### Where the time goes

Seconds summed over the 20 court views (process wall time, start-up
included):

| Stage | What it does | Role | Full search | SVD12 screen |
| --- | --- | --- | ---: | ---: |
| G0 search | Try direction pairs and keep the best 256 courts | Needed | 12,794 (62%) | 5,508 |
| G1 search | The same search on paint-filtered fragments | Needed | 3,341 (16%) | 1,143 |
| W5 refit | Refit each valid candidate to its stripes, then re-measure it | Needed | 1,571 (8%) | 1,227 |
| W5 measurement | Paint and stripe evidence for each merged candidate | Needed | 1,081 (5%) | 908 |
| Line templates | Candidate courts built from line-crossing rectangles | Needed | 666 (3%) | 637 |
| Research-only work | Population files, full records, determinism re-checks, refit replay | Research | 578 (3%) | 446 |
| Pool evidence | Legacy evidence on each generator's 256 kept courts | Likely avoidable (follow-up 7) | 476 (2%) | 428 |
| Everything else | Start-up, view loading, directions, net choice, polarity correction | Needed | 141 (1%) | 117 |
| **Total** | | | 20,648 | 10,413 |

Median wall time per view, full search → SVD12 screen:

- GX: 2,416 → 1,295 s (worst 3,550 s, on GX689)
- amateur and letterboxed: 754 → 318 s
- broadcast: 641 → 328 s
- non-court controls: 90 → 90 s

In the full-search run, three functions inside the search take 65% of the
time across all 28 views:

- the player test (`player_fractions`, G0 and G1): 6,429 s, 30%
- axis matching (`match_axis`): 4,808 s, 22%
- scoring combined courts against the distance maps (`finite_scores`):
  2,891 s, 13%

Follow-up 3 replaces the player test. Follow-up 5 skips part of axis
matching. Both are exact, and both have now landed (see "After follow-ups 3
and 5").

Peak memory per process in the full-search run was 5.2–7.1 GB on GX views, 4.7–6.6 GB on
amateur views and 1.1–1.9 GB on broadcast views.

Research-only work costs 3%. Stripping it for deployment saves little time,
but one part of it breaks a fresh checkout (follow-ups, "Deployment blocker").

Per-view times are noisy on Carmack, by up to 40% between runs (see the
follow-ups' verification plan). Totals over many views are reliable to about
5%.

### After follow-ups 3 and 5

Commit 888cf999 lands both. The full search was rerun on all 28 views with it,
up to 8 jobs at a time. Its outputs were then compared with the run above,
file by file and number by number.

Exactness: all 168 saved files are identical. They cover inputs, G0 and G1
populations, case records, D17 summaries and arrays. The one expected
difference is a diagnostic count: 11,656 axis records now store
`pattern_supported` as None. Players now rule hypotheses out before scoring,
so that count is no longer measured. The bounded choice, corrected court and
non-court abstention are therefore unchanged on every view.

Seconds summed over the 20 court views, before → after:

| Step | Before | After |
| --- | ---: | ---: |
| Player test in the generator | 6,429 | 52 |
| Axis matching | 4,579 | 3,553 |
| Whole generator pair loop | 15,770 | 8,980 |
| **Whole D17 run** | **20,647** | **13,942** |

Median wall time per view, before → after:

- GX: 2,416 → 1,414 s
- amateur and letterboxed: 754 → 409 s
- broadcast: 641 → 537 s
- non-court controls: 90 → 101 s, where the search is already tiny

Peak memory per process, before → after:

- GX: 5.2–7.1 → 1.8–2.2 GB
- amateur: 1.7–6.6 → 0.5–1.7 GB
- broadcast: unchanged at 1.0–1.9 GB

The two runs were not side by side: the earlier one ran 12 jobs at a time
with the SVD12 screen mixed in. Steps the patches do not touch took about 10%
longer in the new run, so the true gain is, if anything, larger.

Inside the pair loop, time now splits into three parts:

- axis matching: 3,553 s
- scoring combined courts against the distance maps: 3,318 s
- untimed candidate building: about 1,100 s, which is follow-up 6's target
  (commit d1609d6c later cut the pair loop's own time to 179 s over 28 views)

The generator search is still 67% of the run.

### Full search against the accepted gallery

- 17 views: the same corrected court, within 1e-9 working pixels
- GX0 and am2 28019: the same court within 0.8 working pixels
- GX77876: 3.5 working pixels apart. The fresh court is closer to the
  reference, with median error 1.9 against 3.2 working pixels

The accepted gallery was built from older source records, so small
differences are expected. None is a regression.

### SVD12 screen against full search

The earlier SVD12 evaluation was the project owner's visual review after
refit. It judged SVD12 courts as good as the full search, within a few pixels.
That review came before the net choice existed, and SVD12 was never
benchmarked through the D17 chain until now.

The SVD12 screen gives the same corrected court as the full search on 15 of
20 court views. Five differ:

| View | Corners apart (working px) | Pixel error, full → SVD12, median / worst (working px) | Floor error, full → SVD12, median / worst (cm) | Verdict for SVD12 |
| --- | ---: | --- | --- | --- |
| letterboxed 78 | 38.6 | 1.6 / 2.9 → 3.6 / 8.5 | 5 / 12 → 10 / 25 | Worse. The court cuts visibly into the playing area |
| ss21 0044 | 4.7 | 2.2 / 4.0 → 3.6 / 4.7 | 6 / 10 → 21 / 36 | Worse |
| ss03 0032 | 1.3 | 1.7 / 2.7 → 2.3 / 2.9 | 5 / 11 → 6 / 10 | Equivalent |
| ss21 0039 | 1.9 | No reference | No reference | Equivalent within a few pixels |
| GX0 | 6.7 | 3.6 / 7.8 → 2.2 / 4.5 | 13 / 110 → 5 / 12 | Better |

On letterboxed, the screen dropped direction group 0, and the best court's
lines need it. The polarity refit keeps each fragment's line assignment, so it
cannot bring that court back. On GX0, the SVD12 screen happens to pick a
different candidate that fits better. There the accepted court overshoots the
same way the full search's court does.

So the direction screen changes which court wins on a quarter of views, in
both directions. It fails the "same court within a few pixels" test on
letterboxed, an easy view.

Non-court controls: seven of eight abstain with both search settings. Frame
00014336 gets a court with both. D17 has always accepted that frame: the
paired statistics' primary setting chose the same line-template court as the
fresh full search. The SVD12 screen picks a different wrong court there.
Abstention is unchanged.

### Pixel error and floor error

Pixel error under-weights the far end of the court. On GX0, one metre of court
spans 5 working pixels at the far baseline and 161 at the near end. The
full search's GX0 court is only 7.8 working pixels out at worst. On the floor, that
worst point, the far-baseline centre, lands 1.1 m beyond the far baseline.

Floor error uses one homography fitted through the reference clicks, by least
squares on landmark views and exactly through the four corners on broadcast
views. Each predicted reference point is mapped back onto the floor through
it:

```python
reference_h, _ = cv2.findHomography(court_m, clicked_px, 0)
predicted_m = cv2.perspectiveTransform(predicted_px[None], np.linalg.inv(reference_h))[0]
floor_error_cm = 100 * np.linalg.norm(predicted_m - court_m, axis=1)
```

Floor error over-weights the far end instead, because it amplifies click
noise there. On GX, one working pixel at the far baseline is roughly 10–20 cm.
Far-end floor errors up to about 20 cm are therefore within annotation noise.
Neither measure matches a visual check alone. Pixel error says whether the
court looks right on the frame. Floor error says whether court positions are
right in metres.

## Finding 1: the player test scales with everyone in the frame

The frozen packs keep every detected person in every sampled frame. Counted
against the reference court (`claude_evidence/feet_census.py`):

| View | Frames | People per frame (median) | Inside the court per frame (median) | Foot samples off the court |
| --- | ---: | ---: | ---: | ---: |
| GX0, GX5 | 30 | 18 | 2 | 88% |
| am1 | 31 | 32 | 4 | 85% |
| am2 | 29 | 42 | 2 | 95% |
| ss19, ss34 | 3 | 10 | 2 | 80% |

`run_given.propose_role` projects every foot sample slot into every combined
court through `zone_net.player_fractions`. That is 30 slots per court on
broadcast views, 660 on GX, 1,085 on am1 and 1,508 on am2. This explains the
baseline timings better than axis counts do. Amateur pair 16 had only 25,500
axis hypotheses but took 89.9 s. Broadcast pair 1 had 72,830 and took 10.9 s.

Spectators also weaken the test as a filter. A court passes if some foot is
inside it in every frame. With people seated around the court, most court-sized
guesses pass. GX pairs keep 82,000–90,000 usable courts, and every one of them
is then scored and turned into an object.

**Patch 1** runs the player test only on geometry-valid courts, since only
those can become usable (`run_given.py:153`). Geometry rejects 54–93% of
combined courts on the six pairs probed on Carmack. Invalid courts keep NaN fractions,
meaning "not measured".

| Pair | Carmack baseline | Laptop baseline | Laptop patched |
| --- | ---: | ---: | ---: |
| broadcast 64 | 1.5 s | 2.9 s | 2.2 s |
| amateur 175 | 11.0 s | 14.7 s | 4.2 s |
| GX 46 | 22.9 s | not run | 1.9 s |
| amateur 16 | 89.9 s | not run | 4.7 s |
| amateur 0 | 92.9 s | not run | 9.5 s |

Unpatched, amateur pair 0 needs one intermediate array of about 9.5 GB:
262,144 combined courts × 1,508 foot slots × 3 coordinates × 8 bytes.

Patch 1 helps least where geometry passes most courts. On GX0 pair 80 and GX5
pair 213, about 100,000 courts are geometry-valid. There the player test still
costs 14.4–14.8 s of 26–39 s per pair (Carmack profile).

## Finding 2: stripe evidence measures fragments it then discards

`stripe_observations.interval_evidence` measures distances from each projected
marking to all ~650 fragments (`stripe_observations.py:76` and `:80`). It then
masks out fragments more than 5° from the marking direction (`:78`, `:82`).
Most fragments fail that direction test, so most of the distance work is thrown
away.

This one function dominates both later stages on `shuttleset_03_scene_0034`
(laptop):

| Stage | Baseline | Stripe measurement share | Patched |
| --- | ---: | ---: | ---: |
| Pool evidence (256 courts) | 46.8 s | 44.8 s (96%) | 16.8 s |
| Refit stage (256 parents, 255 children) | 110.7 s | 88.4 s (80%) | 52.0 s |

**Patch 2** computes distances only for fragments compatible with at least one
of the three stripe positions. The skipped fragments get the same `0.0` and
`inf` values the old masks produced.

## Exactness evidence

**Laptop, same process layout.** Baseline and patched runs were bit-identical
for:

- retained candidate IDs, scores and corners on broadcast pair 64 and amateur
  pair 175
- all 256 pool-evidence entries, every W5 parent and child record, and the
  final ranking on `shuttleset_03_scene_0034`

**Codex red team.** Codex (gpt-6-sol, high effort) attacked both patches
independently (`claude_evidence/CODEX_REDTEAM.md`). It confirmed patch 2 byte
for byte across 24 synthetic interval cases, including no compatible fragments,
custom templates, boundary tolerance, NaN, inf and negative zero. It confirmed
that pool stripe and profile evidence never reach the accepted choice.

Codex found one real consequence of patch 1, which I verified. The closed
pre-gate loss diagnostic reports player fractions for the nearest combined
court even when geometry rejected it
(`line_identity/prior_checks/pregate_loss/analyse.py:148-150`). Rerun on
patched code, that report would show NaN. The deployment path never writes
those pool sidecar files, and no selection reads them. The matrix-product test
below removes this caveat.

**Carmack, six whole views** (`claude_evidence/compare_runs.py`,
`score_drift.py`):

- all six case records are identical, excluding timing fields: candidates,
  refits, ranking and selected court
- all six generation records are identical in every field except
  `shortlist_score`: IDs, homographies, corners, pool evidence, pool order and
  both generator winners
- `shortlist_score` differs on 1.4–6.2% of scores, by at most 7.5e-9

## Score drift and its cause

Every numeric difference observed in this work, and its cause:

| Where | Size | Cause | How established |
| --- | --- | --- | --- |
| Carmack generator `shortlist_score`, patched vs baseline | 221–382 scores per view; max 2.3e-10 to 7.5e-9 | IPP distance transform rounds differently depending on the output buffer's address | Reproduced and isolated on the laptop; IPP behaviour confirmed on Carmack |
| Laptop, identical inputs, one process, am2 pair 175 | 38 of 8,900 scores; max 1.35e-9 | Same | Same |
| Laptop fresh run vs saved Carmack record: `camera_error` | 56 entries; max 4.2e-17 | Probably different CPU instruction sets and numpy builds | Not isolated. No distance maps involved |
| Laptop fresh run vs saved Carmack record: pool stripe score | max 7.8e-15 | Same as above | Not isolated |

Per-view Carmack drift (`score_drift.py`):

| View | Scores compared | Scores drifted | Largest difference |
| --- | ---: | ---: | ---: |
| am1 | 5,036 | 314 | 3.7e-9 |
| am2 | 7,326 | 247 | 5.4e-9 |
| GX0 | 16,223 | 304 | 7.5e-9 |
| GX5 | 16,456 | 234 | 5.4e-9 |
| ss19 | 11,510 | 221 | 2.3e-10 |
| ss34 | 12,851 | 382 | 3.7e-9 |

The IPP attribution rests on this chain:

1. **The scoring inputs were identical.** Both Carmack generation records store
   each court's full homography and axis IDs, and those match bit for bit. So
   `run_given.finite_scores` got identical inputs and returned different
   outputs. The patches do not touch `finite_scores`.
2. **The distance maps change, not the arithmetic around them.** In one laptop
   process, `finite_scores` on identical inputs changed 38 of 8,900 scores on a
   later call (`claude_evidence/history_probe.py`). The rebuilt distance maps
   differed at 420 pixels by 9.5e-7, one float32 rounding step. Projection,
   division, `exp` and the float32 mean gave identical bits at every memory
   offset (`alignment_probe.py`, `stage_probe.py`).
3. **IPP makes the map depend on memory placement.** OpenCV calls IPP for
   `cv2.distanceTransform(..., DIST_L2, DIST_MASK_PRECISE)` when it runs
   single-threaded, as the pipeline does. With IPP on, 16 output-buffer offsets
   gave 8 different maps. The pattern repeats every 32 bytes on the laptop,
   which is one AVX2 register. With IPP off, every offset gave the same map,
   equal at every pixel to the exact Euclidean distance rounded to float32
   (`ipp_probe.py`). Carmack's AVX-512 IPP build also gave 8 variants
   (`ipp_timing.py`).
4. **Memory placement varies between processes.** OpenCV's Python binding
   allocates outputs as numpy arrays, which are only guaranteed 16-byte
   alignment. Where an array lands depends on earlier allocations. Python picks
   a random hash seed per process, which changes those allocations. The same
   code, data and call gave 3 different maps across 4 hash seeds.
5. **Float32 carries the step into the score.** `continuous_support` reads
   float32 maps and averages float32 responses. One rounding step at a few
   pixels moves a court's score by up to about 1e-8.

So neither patch causes the drift. The patches change allocation history, which
decides which of IPP's rounding variants a run gets. By the same mechanism, two
runs of unpatched code can drift from each other. I showed this on the map
builder, which neither patch touches, rather than by rerunning the unpatched
pipeline.

Impact: no pool, order, winner or selection changed in six views. A swap is
possible only if two courts' scores fall within about 1e-8 of each other at a
retention boundary. The later IPP-off reference run showed such near-ties
exist: on `shuttleset_03_scene_0019`, three shortlist entries swapped order
against the IPP-on baseline. The set was unchanged, and so was the case record.

Turning IPP off for this call removes the drift and is also faster. On Carmack,
15 interleaved rounds per mask gave 12–15% lower median time without IPP, with
non-overlapping ranges (`ipp_bench.py`). `CLAUDE_FOLLOWUPS.md` item 2 covers
the change, now committed as 7b56d58e. Two six-view runs with it, under
different hash seeds, gave identical records.

## Claim-by-claim verdicts

Verdicts: **Adopt** (worth doing now), **Later** (correct, secondary),
**Reject** (wrong or not worth its complexity).

| ID | Handover claim | Verdict | Evidence |
| --- | --- | --- | --- |
| F1 | Apply the player gate before axis scoring | **Adopted** (follow-up 5, commit 888cf999) | Exact for retained axes. On GX5 pair 213, 60% of 243,840 vertical hypotheses fail the gate, and axis scoring takes 15.6 of 39 s. On large broadcast pairs, 63–69% fail; axis scoring is 21% of patched time there. Changes the `pattern_supported` diagnostic count unless it is counted on gated hypotheses only |
| F2 | Hoist the basis inverse and frame geometry | Reject | Correct observation. A 3×3 inverse per 256-row batch is well under 1% of pair time. The proposed prepared-structure types add code for no measurable gain |
| F3 | Affine endpoint-coefficient scoring kernel | Later, maybe never | Changes float rounding, so it needs frontier fallback code. It targets the same cost as F1, which is simpler and exact |
| F4 | Larger scoring batches | Later | Exact. Per-call overhead in `score_axes` is small; expected gain is a few per cent |
| F5 | Joint player test from per-axis values | **Adopted** (follow-up 3, commit 888cf999) | Measured as a matrix product (next section). Zero disagreements with the current test on 371,024 valid and 60,000 sampled invalid courts. 100–225× faster on the pairs tested. Bit-identical D17 outputs on all 28 views |
| F5b | Do not project corners twice | Reject | Correct, but `canonicalise` plus `geometry` cost 2–4% of patched pair time |
| F6 | Player test only on geometry-valid courts | **Adopt (patch 1)**, superseded by F5 | Exact for selection. Validated on six whole views. The handover calls it a low-risk side patch, then folds it into the Tier-B P5 rewrite |
| F7 | Cache per-pair distance maps | Reject | Generation maps cost about 20 ms per pair. Keys built from retained group IDs would rarely repeat. A different, real repeat exists in the refit stage (see "Missed by the handover") |
| F8 | Build candidate objects only for survivors | **Partly done** (commit d1609d6c) | Correct. It was about 13% of patched time on large broadcast pairs. Commit d1609d6c now builds provenance records only for courts that reach a shortlist. Candidate objects are still built for every usable court, but the pair loop's own time is down to 179 of 8,931 s (2%), which caps what is left |
| F9 | Batch the gate evidence | Reject | Correct that feet are rebuilt per candidate. All gate work is 0.45 of 46.8 s (1%) of pool evidence |
| F10 | Batch stripe measurement across candidates | Replace | Right stage, wrong mechanism. Patch 2 removes most of the cost exactly in about 10 lines |
| F11 | Refit cache and analytic Jacobian | Reject | `least_squares` is 4.3 s of the 110.7 s baseline refit stage. Stripe measurement is 80%. Parents are at least 2 px apart, so exact duplicate fits cannot occur within one source |
| F12 | Move view reuse before full search | Agree | Correct: current grouping happens after inference and saves no detector calls. Needed for the video budget. Outside this evaluation's measurements |

Two lower-priority remarks are also off. `cv2.polylines` would not speed up
distance maps, because `cv2.distanceTransform` takes 90% of map cost. Cached
`linspace` fractions and homogeneous constants cost microseconds.

## F5 as a matrix product

A combined court's homography is `basis @ diag-scale-shift` (`combine`,
`projective_seed.py:190`). So a foot's court x depends only on the horizontal
axis hypothesis, and its court y only on the vertical one.
`projective_seed.necessary_players` already tests each axis hypothesis's band
with the same 0.15 margin as `zone_net.player_fractions`
(`projective_seed.py:122-133`). The 180-degree relabelling and depth sign flip
in `canonicalise` and `combine` leave both tests unchanged. The margin is
symmetric, and the two-halves test needs both halves.

"Some foot is inside this court in this frame" then means "some foot is inside
both bands". For every pair of axis hypotheses at once, that is one matrix
product per frame:

```python
# inside_x: (horizontal hypotheses, frames, foot samples); inside_y and far_y: (vertical hypotheses, frames, foot samples).
# Each product counts, per frame, the foot samples inside both bands for every (horizontal, vertical) pair.
by_frame_x = inside_x.transpose(1, 0, 2).astype(np.float32)
anyone = (by_frame_x @ inside_y.transpose(1, 2, 0).astype(np.float32)) > 0
far = (by_frame_x @ far_y.transpose(1, 2, 0).astype(np.float32)) > 0
near = (by_frame_x @ (inside_y & ~far_y).transpose(1, 2, 0).astype(np.float32)) > 0
one = anyone.mean(axis=0).reshape(-1)  # horizontal-major, the same order as combine()
two = (far & near).mean(axis=0).reshape(-1)
```

The counts are small whole numbers, so float32 products are exact. Each product
is (512 × feet) @ (feet × 512) per frame. The cost no longer grows with
courts × people.

Measured on Carmack against the current test (`joint_players_probe.py`):

| Pair | Valid courts | Disagreements (either fraction or usable) | Current test | Matrix product |
| --- | ---: | ---: | ---: | ---: |
| GX0 pair 80 | 100,760 | 0 | 14.58 s | 0.065 s |
| GX5 pair 213 | 97,038 | 0 | 14.40 s | 0.073 s |
| am2 pair 0 | 18,658 | 0 | 6.47 s | 0.100 s |
| am2 pair 175 | 9,244 | 0 | 3.15 s | 0.021 s |
| ss34 pair 64 | 34,400 | 0 | 0.27 s | 0.003 s |
| ss34 pair 1 | 110,924 | 0 | 0.84 s | 0.006 s |

A random sample of 20,000 geometry-invalid courts on each of GX0 pair 80, GX5
pair 213 and am2 pair 0 also had zero disagreements. So the matrix product can
measure every combined court, which restores real values for the pre-gate
diagnostic.

The two tests use different arithmetic, so they could disagree for a foot
within rounding distance of a band edge. None did here. After the matrix
product landed, the 28-view D17 rerun gave bit-identical outputs (see "After
follow-ups 3 and 5").

## Missed by the handover

These are measured, and each is exact or selection-neutral.

- **Everyone in the frame drives cost.** Amateur and GX views are slow because
  of 660–1,508 foot slots per court, mostly spectators. Axis enumeration is
  not the cause
- **Direction-incompatible fragments** (patch 2 above)
- **The refit rebuilds the view's distance maps for every child.**
  `run_w5.attempt_refit` calls `_distance_maps` per child
  (`run_w5.py:747`). Its inputs are fixed for the view. This costs 6.7 of
  52.0 s in the patched laptop refit stage. Codex confirmed the repeat
- **Pool evidence computes scores the accepted selection never reads.** The
  pool's centre-line stripe score and appearance profile feed only legacy
  report fields, comparison arm A and the generator's own winner IDs. Codex
  confirmed this independently
- **Distance maps were not reproducible.** IPP's rounding depends on memory
  placement (previous sections). Exact-equivalence checks saw score noise
  near 1e-8 until commit 7b56d58e switched IPP off for that call

`CLAUDE_FOLLOWUPS.md` covers what to do with these.

## The handover's process advice

These parts are sound:

- compare baseline and patch on the same machine, with one numerical thread
- start with one fast and one slow view before any corpus run
- keep search caps, sources, thresholds and ranking out of exact patches
- treat G0-only timings as partial, since G1 and templates also run

These parts are more machinery than the problem needs:

- an 11-step serial patch queue with a JSON dependency manifest
- prepared-geometry structure types, three diagnostics levels, an LRU cache
  with hit-rate reports, and numerical-frontier fallback paths for small gains
- a bootstrap script that records pip freeze, CPU details and checksums for
  every session, plus four standing log files per session
- Tier A/B/C labels, where "same pools and winners" does the job

One profile of one amateur pair would have reordered the whole queue. The
handover says openly that it ran no profiler. The cost of that is visible:
operation counts missed the per-court × per-person factor.

Its `tools/summarise_generation.py` counts "redundant basis inversions" as a
cost proxy, which misleads for the reason under F2. `tools/resolve_hotpath.py`
was useful. `tools/compare_records.py` works, but it compares JSON only.

Its bit-identity goal for Tier A patches did not hold at first, because IPP
built the distance maps. With IPP off for that call (commit 7b56d58e), it
holds. Bit-identical outputs have been the gate for every exact change since.

## Limits of this evaluation

- The patch 1 and 2 measurements cover G0 generation and the refit stage only.
  The D17 chain was timed before and after follow-ups 3 and 5, but not before
  patches 1 and 2
- The D17 run used one frame per view. The SVD12 screen's differences could be
  larger or smaller on other frames of the same videos
- Broadcast floor errors rest on four clicked corners, so click noise is not
  averaged out there
- Refit-stage profiles come from one broadcast view. The whole-view Carmack
  run times the refit stage on all six but does not break it down
- The baseline wall times come from a run that shared its worker pool with
  two other arms
- The laptop-versus-Carmack differences near 1e-17 and 1e-15 are attributed
  to hardware and numpy builds without isolation
- Video-level cost (decoding, cuts, view reuse) is outside this evaluation
