# Court detector speed-ups: what was tried, and why

**Where things stand.** The full search now takes 8,250 s summed over 28
views, down from 8,931 s at the current baseline (defined below). Every
change behind those numbers is exact. Four runs, over three different sets
of views, trace the path:

- patches 1 and 2: six whole views, 6,255 → 2,572 s
- items 3 and 5: the 20 court views, 20,647 → 13,942 s
- item 12: all 28 views, 13,002 → 8,931 s
- items 13 and 14: all 28 views, 8,931 → 8,250 s

The exact savings left add up to about 10% of the baseline, from a deployment
mode that skips research-only files and legacy evidence. After that, the
remaining time is the search itself.

Two larger levers remain, and neither is a micro-optimisation. Running pairs
in parallel (item 8) cuts one video's wait but not total CPU. Reusing a view's
court across scenes (item 9) avoids whole searches, but it is a design
decision. Most ideas that would cut more were decided against, because they
can drop the right court on a view nobody has tested.

The detector is not yet one unit. Its stages run end to end only through a
timing script, `scratch/court_det_fix/d17_timing/run_d17.py`, on the 28 test
views. `d17_timing/WIRING.md` on `exp/court-det-opt` says how those stages join
up, and what a single detector must keep or can drop. The web UI's
`SOURCE_MAP.md` says no such runner exists; that line predates the script.

The tables below list every idea tried, with its outcome and the reason.
Numbered sections after them hold the detail.

**Conventions**

- **Current baseline**: item 12's Carmack run at commit d1609d6c. Full search
  (all 16 direction groups) on all 28 views, seated-removed feet (item 10),
  8 jobs at a time. 8,931 s of process wall time, summed over views
- **Exact**: every saved result bit-identical to the run before the change,
  checked across all output files. The comparer skips timing, memory and
  stage-time fields and run-folder paths, so it says nothing about speed or
  memory
- **Commits**: all on branch `exp/court-det-opt`. This folder is also on
  `fix/court-det`, which does not hold them
- **Item N**: a numbered section below. **F-numbers** are the web-UI
  handover's proposals, judged in `CLAUDE_EVALUATION.md`
- **Times** are summed process seconds. Carmack is shared, so one view's time
  can vary by up to 40% between runs. Totals over many views agree to about 5%
- **View, pair, axis hypothesis, combined court, D17 chain**: as defined under
  "Conventions" in `CLAUDE_EVALUATION.md`. In short, the search tries about 240
  pairs of line directions per view and builds candidate courts from each
- **Shortlist**: the 256 best distinct courts a pair keeps, by distance-map
  score
- **W5**: the stage after the search. It measures each shortlisted court (the
  parent), refits it to the painted stripes (the child), and ranks the results
- **Line templates**: extra candidate courts built from rectangles of crossing
  lines. The template camera check rejects those that no camera with a
  plausible focal length could produce

## Built in

All exact unless noted.

| Change | Commit and item | Result | Why it was kept |
| --- | --- | --- | --- |
| Player test only on geometry-valid courts (patch 1) | a2e24ddf, item 1 | With patch 2: six views 2.4× faster | Exact. Item 3 later replaced it |
| Stripe evidence measures only direction-compatible fragments (patch 2) | a2e24ddf, item 1 | With patch 1: six views 2.4× faster. Stripe measurement had been 80–96% of pool-evidence and refit time | Exact. The skipped fragments were discarded anyway |
| Distance maps built without IPP | 7b56d58e, item 2 | Maps 12–16% faster; repeat runs bit-identical | Removed run-to-run score noise, so every later change could be checked bit for bit |
| Player test as one matrix product per frame (F5) | 888cf999, item 3 | Player test 6,429 → 52 s over the 20 court views | Exact. Cost no longer grows with courts × people |
| Player gate before axis scoring (F1) | 888cf999, item 5 | Axis matching 4,579 → 3,553 s | Exact. The skipped hypotheses could never be kept |
| Refit distance maps built once per view | d1609d6c, item 4 | 412 → 85 s over 28 views | Exact. The maps were identical for every refit |
| x and y kept as separate arrays in three hot spots; provenance records built on demand | d1609d6c, item 12 | 13,002 → 8,931 s over 28 views (−31%) | Exact. numpy is 2–3× slower on a trailing axis of length 2 |
| Cache for `prepare_observations` | 7c74b015, item 14 | 100 → 21 s over 28 views, under 1% | Exact and a few lines. The project owner judged it worth it |
| Camera check with x, y and w as separate arrays | 44254b42, item 13 | 564 → 92 s over 28 views (5.3% of the run) | Exact on all 28 views. numpy is several times slower on a trailing axis of length 3 or 2 |

## Decided or being checked

| Change | Status | Result so far | What is left |
| --- | --- | --- | --- |
| Leave seated people out of the player test | Decided by the project owner, item 10 | Axis matching −27%, run −5%, all 20 courts kept. Not exact: three courts moved 0.5–3.9 px, each as close to the reference or closer, bar 0.04 px on one | Wire in pose keypoints for the frames the player test uses. The baseline already uses these feet, so the 5% is already counted |

## Open

| Idea | Likely gain | Exact? | What it needs |
| --- | --- | --- | --- |
| Deployment mode: skip research-only files | About 580 s (6.5%) | Yes; records only | Pass populations and records between stages in memory. Also removes the git-ignored control files blocker |
| Deployment mode: skip legacy pool evidence (item 7) | Under 480 s; not re-measured | Yes; records only | A mode flag |
| Run pairs in parallel (item 8) | One video's wait, several-fold on 8 cores (not measured). No CPU saving | Yes, if merged in order | A process pool |
| Reuse a view's court across scenes (item 9, F12) | The large lever for whole videos | No; a design decision | Rules for "same camera" and for catching a wrong reuse |
| Finish building candidate objects only for survivors (item 6, F8) | At most 179 s (2%) | Yes | An interface change between `propose_role` and `select_pool` |
| Larger axis-scoring batches (F4) | A few per cent; not measured | Yes | A batch-size change and a timing check |
| float32 in the hot scoring code (item 17) | Perhaps 10–20%; a guess | No | Time the two hottest functions first, then a 28-view accuracy check |
| Compiled loops (numba) for court scoring | Unmeasured | Not automatically: loop sums round differently from numpy's | A new dependency. Only worth it if the options above miss the time budget |
| Working image below 960×540 | Unmeasured | No | Likely costs accuracy: reference errors are about 2 working px now |

## Decided against

| Idea | Why not | Detail |
| --- | --- | --- |
| SVD12 direction screen, or any pre-run that drops directions | Halves run time, but changes the court on 5 of 20 court views. Letterboxed and ss21 0044 get clearly worse | `CLAUDE_EVALUATION.md`, "SVD12 screen against full search" |
| Upper bound to skip courts that cannot make the shortlist | The bound holds but is loose. It costs 53–60% of full scoring and skips little | Item 12 |
| Coarse scores, then exact scores for the top K only | Kept every shortlist here, saving about 8%. But K is fixed before a view is seen, and one call needed K = 4,511 with no warning at 4,096 | Item 16 |
| Count only the people who move most | Top 6 movers fails the player test on 9 of 20 real courts | Item 11 |
| Cap people by detector score | Players in motion, occluded or behind the net can score below spectators (the project owner's call) | "What not to do" |
| Shrink the 256-court shortlists | The accepted court's parent ranked as low as 41st in its pair and 107th overall. A cap of 32 per pair loses 1 of 20 courts | "What not to do" |
| Camera gate before court scoring | A parent that fails the camera gate can yield a child that passes | "What not to do" |
| Refit only parents near the best score | Skips 14–58% of refits, but the safe margin is empirical | Item 15 |
| Camera-check only the courts selection reads | About 1% left after item 13. Loses metadata that `compare_directional_runs.py` validates | Item 15 |
| Faster W5 measurement internals | No single hotspot is left after item 12 | Item 15 |
| gzip level 6 for research records | About 0.6%. It breaks byte comparison with older runs, and deployment skips these files anyway | Item 15 |
| Score bit-identical axis hypotheses once | There are none: 0 duplicates among 321,224 scored hypotheses | Item 12 |
| float16 camera check | 5–6× slower than float64 on a CPU. It overflows on real homographies, with errors up to 0.074 against a 0.1 gate | Item 17 |
| float32 for the camera check alone | 2.3–3.8× faster, but saves only about 50 s (0.6%). Stored values would no longer match byte for byte | Item 17 |
| Hoist the basis inverse (F2) | Under 1% of pair time | `CLAUDE_EVALUATION.md` |
| Affine scoring kernel (F3) | Changes rounding and needs fallback code. F1 got the same saving exactly | `CLAUDE_EVALUATION.md` |
| Project corners once (F5b) | 2–4% of pair time | `CLAUDE_EVALUATION.md` |
| Per-pair distance-map cache (F7) | About 20 ms per pair, and keys rarely repeat | `CLAUDE_EVALUATION.md` |
| Batched gate evidence (F9) | 1% of pool evidence | `CLAUDE_EVALUATION.md` |
| Batched stripe measurement (F10) | Patch 2 removed the same cost in about 10 lines | `CLAUDE_EVALUATION.md` |
| Refit cache and analytic Jacobian (F11) | `least_squares` is about 4% of the refit stage | `CLAUDE_EVALUATION.md` |
| `cv2.polylines` for map drawing | The distance transform is 90% of map cost | "What not to do" |
| Prepared-structure types; LRU caches with hit-rate reports | The costs they target are under 1% of pair time | "What not to do" |

## 1. Land patches 1 and 2

Done. Commit a2e24ddf on branch `exp/court-det-opt`, also in
`claude_evidence/exact_patches.diff`.

- **Patch 1**, `run_given.propose_role`: run the player test only on
  geometry-valid courts
- **Patch 2**, `stripe_observations.interval_evidence`: measure distances only
  to direction-compatible fragments

Benefit: 2.4× over six whole views on Carmack, with 6.9× on am2. Every case
record, including the selected court, matches the 23 September baseline.

Cost: none beyond review. Item 3 later replaces patch 1.

## 2. Make the distance maps deterministic

`cv2.distanceTransform` with `DIST_MASK_PRECISE` goes through IPP when OpenCV
runs single-threaded. IPP's rounding then depends on where numpy placed the
output array, which changes between processes. That caused all the score
drift in the Carmack comparison.

Done. Commit 7b56d58e on branch `exp/court-det-opt`. Two hot-path sites
called it: `detector._distance_maps` (generation scoring and the W5 refit) and
`line_template_source.union_distance_map` (line templates). Both now use one
helper in `detector.py`, and the duplicate is gone.

IPP is switched off for this one call and restored afterwards. The switch is
process-wide, and other OpenCV calls such as resizing also use IPP, so a
global switch could change unrelated outputs.

```python
def distance_map(segments: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Give each working-image pixel its distance to the nearest drawn fragment."""
    width, height = size
    mask = np.full((height, width), 255, dtype=np.uint8)
    for x1, y1, x2, y2 in np.rint(segments).astype(int):
        cv2.line(mask, (x1, y1), (x2, y2), 0, 1)
    use_ipp = cv2.ipp.useIPP()
    cv2.ipp.setUseIPP(False)
    try:
        return cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    finally:
        cv2.ipp.setUseIPP(use_ipp)
```

A test in `tests/test_independent_court.py` checks the distances and that the
caller's IPP setting survives.

Benefit:

- identical maps on every run, and equal at every pixel to the exact
  Euclidean distance rounded to float32 (`claude_evidence/ipp_probe.py`)
- 12–15% faster per map on Carmack across 15 interleaved rounds
  (`ipp_bench.py`), and 12–16% in a later check in both Carmack venvs
- later equivalence checks can demand bit-identical records instead of "same
  pools and winners"

Result (verification plan step 1): two six-view runs with `PYTHONHASHSEED`
1 and 2 gave identical records on every non-timing field. Against the
23 September baseline, all six case records match bit for bit, apart from
`jacobian_condition`. That field is the condition number of a near-singular
fitting matrix. It differs between the old and new venv's maths libraries, and
nothing reads it for a decision. Generator shortlist scores moved by up to
1.1e-8. On `shuttleset_03_scene_0019`, three near-tied shortlist entries
swapped order, and the case record built from them is unchanged.

## 3. Replace the player test with a matrix product

This is the handover's F5, in a simpler form. A foot is inside a combined
court when it is inside both axis bands. `necessary_players` already computes
each band per axis hypothesis. One matrix product per frame then gives the
answer for every pair of axis hypotheses at once. The derivation is in
`CLAUDE_EVALUATION.md`, "F5 as a matrix product".

Done. Commit 888cf999 on branch `exp/court-det-opt`, together with item 5. A
unit test (`test_projective_seed.py`) checks it against `zone_net` on 1,600
combined courts, including rotated ones. The D17 rerun gave bit-identical
outputs on all 28 views. Over the 20 court views, the player test fell from
6,429 s to 52 s.

The landed code follows the shape below. `joint_player_fractions` takes the
basis, both axes and the feet, and does the rectification itself through a
shared `rectify_feet` helper.

```python
def band_masks(parameters: np.ndarray, rectified_feet: np.ndarray, extent: float) -> tuple[np.ndarray, np.ndarray]:
    """Per axis hypothesis: foot samples inside this axis's court band, and those in its first half.

    :return: two boolean arrays of shape (axis hypotheses, frames, foot samples).
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        position = (rectified_feet[None] - parameters[:, None, None, 1]) / parameters[:, None, None, 0] / extent
    inside = np.isfinite(position) & (position >= -.15) & (position <= 1.15)
    return inside, inside & (position < .5)


def joint_player_fractions(inside_x: np.ndarray, inside_y: np.ndarray, far_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Player fractions for every combined court, in combine() order (horizontal-major).

    Each product counts, per frame, the foot samples inside both bands. Counts are
    small whole numbers, so float32 products are exact.
    """
    by_frame_x = inside_x.transpose(1, 0, 2).astype(np.float32)
    anyone = (by_frame_x @ inside_y.transpose(1, 2, 0).astype(np.float32)) > 0
    far = (by_frame_x @ far_y.transpose(1, 2, 0).astype(np.float32)) > 0
    near = (by_frame_x @ (inside_y & ~far_y).transpose(1, 2, 0).astype(np.float32)) > 0
    return anyone.mean(axis=0).reshape(-1), (far & near).mean(axis=0).reshape(-1)
```

`necessary_players` can then call `band_masks`, so the band test lives in one
place. In `propose_role`, the `zone.player_fractions` call becomes:

```python
inside_x, _ = seed.band_masks(horizontal.parameters[horizontal.retained], rectified[..., 0], float(detector.X_COORDS.max()))
inside_y, far_y = seed.band_masks(vertical.parameters[vertical.retained], rectified[..., 1], float(detector.Y_COORDS.max()))
one, two = seed.joint_player_fractions(inside_x, inside_y, far_y)
usable = valid & (one == 1) & (two >= .5)
```

`rectified` is the feet mapped through `inv(basis)` and divided by the third
coordinate, as `match_axis` already does. Moving that into a small shared
helper keeps it computed the same way in both places.

Benefit, measured on Carmack (`joint_players_probe.py`):

- GX0 pair 80: 14.58 s → 0.065 s; GX5 pair 213: 14.40 s → 0.073 s
- am2 pair 0: 6.47 s → 0.100 s; ss34 pair 1: 0.84 s → 0.006 s
- every combined court gets real fractions again, which removes patch 1's NaN
  caveat for the pre-gate diagnostic
- memory is about 31 MB per product (30 frames × 512 × 512 × 4 bytes),
  against about 9.5 GB for unpatched amateur pair 0

Equivalence: zero disagreements in either fraction or the usable mask, on
371,024 geometry-valid courts across six pairs plus 60,000 sampled invalid
courts. The two tests use different arithmetic. They could still disagree for
a foot within rounding distance of a band edge. The 28-view D17 rerun found no
such case: every output was bit-identical.

Cost: about 30 lines, replacing patch 1. The generator no longer calls
`zone_net.player_fractions`.

## 4. Build the refit distance maps once per view

Done in commit d1609d6c (`run_w5.view_line_maps`, shared read-only across
refits). The Carmack check under item 12 covers it: `distance_maps` fell from
412 to 85 s over 28 views. The original proposal follows.

`run_w5.attempt_refit` rebuilds the view's distance maps for every child
(`run_w5.py:747`). Its inputs, the view's segments and size, are fixed for the
view. Codex confirmed the maps are identical for every child.

Pass the maps in and let each caller build them once per view:

```python
# run_w5.attempt_refit: take the maps instead of rebuilding them.
def attempt_refit(context, parent, runtime, cache, maps):
    ...  # delete the per-child detector._distance_maps(...) line

# In each caller, once per view, before the parent loop:
maps = detector._distance_maps(detector._wide_line_families(context.segments), context.size)
row, child, _ = run_w5.attempt_refit(context, parent, runtime, cache, maps)
```

Callers: `w5_holistic/run_w5.py:924`, `svd_search/run.py:98` and
`colour_consistency/am1_recovery_trial.py:237`. The existing `cache` argument
is the wrong home: in `run_w5` it is typed as keyed by homography bytes.

Benefit: 6.7 of 52.0 s in the patched ss34 refit stage on the laptop (13%).
On Carmack, two maps per child at 13–16 ms each come to about 7–8 s per
view.

Cost: about 6 lines across three files. Exact.

## 5. Gate axis hypotheses on players before scoring them (F1)

`match_axis` scores every axis hypothesis, then drops those that fail
`necessary_players`. Testing players first skips scoring for hypotheses that
can never be retained.

Done. Commit 888cf999, together with item 3. Over the 20 court views, axis
matching fell from 4,579 s to 3,553 s, with bit-identical D17 outputs. The
gain is smallest in crowded views, where the player test rejects little (see
item 10). With pruning on, `pattern_supported` is saved as None, because
player-incompatible hypotheses are no longer scored. The given-direction
diagnostic in `run_given.py` reports that stage as None for the same reason.

```python
if rectified_feet is not None:
    for start in range(0, len(parameters), settings.batch):
        player_compatible[start:start + settings.batch] = necessary_players(
            parameters[start:start + settings.batch], rectified_feet, axis, float(coordinates.max()))
to_score = np.flatnonzero(player_compatible)
for start in range(0, len(to_score), settings.batch):
    rows = to_score[start:start + settings.batch]
    batch_scores, group_indexes, counts = score_axes(parameters[rows], coordinates, endpoints, basis, axis)
    scores[rows] = batch_scores
    matches[rows] = np.where(group_indexes >= 0, ids[np.maximum(group_indexes, 0)], -1)
    supported[rows] = counts
```

Benefit:

- GX5 pair 213: 60% of 243,840 vertical hypotheses fail the gate. Axis
  scoring takes 15.6 of 39 s there (Carmack profile)
- large broadcast pairs: 63–69% fail; axis scoring is about 21% of patched
  pair time (laptop profile)
- am2 pair 0: 14% of horizontal and 26% of vertical hypotheses fail. All
  axis matching took 0.07 s on am2 pair 175 (laptop), so the gain is small there

Cost: about 10 lines. Retained axes are unchanged, because `eligible` already
requires `player_compatible`. Gated-out hypotheses keep their initial values
(`supported` 0, `matches` −1). The `pattern_supported` diagnostic therefore
counts only player-compatible hypotheses. Also initialise `scores` with NaN
rather than `np.empty`, so the skipped rows are clearly unmeasured.

## 6. Build candidate objects only for survivors (F8)

Half done, and little is left to gain. Commit d1609d6c builds each court's
provenance record only when the court reaches a shortlist. `propose_role`
still creates one `detector.Candidate` per usable court. The pair loop's own
time, which includes that, fell from 986 to 179 s over 28 views (item 12).
So the rest of this item can save at most about 2% of the run.

The original proposal follows. `propose_role` created one `detector.Candidate`
and one provenance dict per usable court, then kept at most 256 per pair. GX
pairs created 82,000–90,000 objects each. On large broadcast pairs, object
creation and `tolist` took about 13% of patched pair time.

The greedy retention in `scan_population.retain` only needs corners and
scores. An array version that returns retained indices lets the generator
build objects for the survivors alone:

```python
order = np.argsort(-scores, kind="stable")  # same tie order as sorted(..., key=-score)
retained = []
for index in order:
    if retained and np.linalg.norm(corners[retained] - corners[index], axis=2).max(axis=1).min() <= distinct_px:
        continue
    retained.append(index)
    if len(retained) == keep:
        break
```

Cost: moderate. It changes the interface between `propose_role` and
`select_pool`. Exact as long as the stable tie order is kept.

## 7. Skip legacy pool evidence in a deployment mode

`run_automatic.evaluate_pool` computes a centre-line stripe score and an
appearance profile for each of 256 pooled courts. The accepted selection never
reads them (confirmed by Codex). They feed legacy report fields, comparison
arm A and the generator's own winner IDs.

A deployment mode could skip them and keep the gate evidence. In the older
13,942 s run, pool evidence for G0 and G1 took 480 s over the 20 court views,
3.4% of their time. The largest part is stripe measurement on the pooled
courts. Item 12 made stripe measurement faster, so the saving is now smaller.
It has not been re-measured.

Cost: a mode flag, and deployment records lose those fields. Keep the full
evidence for experiment runs.

## 8. Run generator pairs in parallel

The 240 pairs per view are independent until the global retention step. A
process pool over pairs would cut wall time for one video on a multi-core
machine. It does not reduce total CPU. The SVD-search runner already
parallelises across views, so this matters only for single-view deployment.

Of the exact options left, this is the largest for one video's waiting time.
The slowest view in the current baseline, `gxBQ_window_00_frame_689`, took
772 s (666 s after items 13 and 14), and the pair search is most of that. On 8 cores the search part could shrink
several-fold (not measured). Results stay exact if the pairs' candidates are
merged back in their original order before retention.

## 9. View reuse before full search (F12)

The handover is right that grouping happens after inference and saves no
detector calls. Needed for the video budget, and a separate design question.

## 10. Bake in: leave seated people out of the player test

The project owner has decided to bake this rule in. The result is below the
original proposal.

Every timing run after this one, including item 12's 8,931 s baseline, used
the seated-removed feet. Its 5% saving is therefore already inside those
figures, not extra to them. What remains is plumbing: pose keypoints for the
frames the player test uses.

The player test counts every detected person, including spectators. In a
crowd, spectators at both ends satisfy the test for almost any large court,
so the test prunes little and more courts must be scored. Item 5's
measurements show the effect:

| View | Detected people per frame | Axis hypotheses the player test rejects |
| --- | --- | --- |
| Broadcast, large pairs | up to 10 (capped) | 63–69% |
| GX5 pair 213 | 15–22 | 60% (vertical axis) |
| am2 pair 0 | 25–37 | 14% horizontal, 26% vertical |

sticky_anchor already has a seated-person rule, `is_sitting` in
`src/bst_x/preparing_data/heuristics/base.py`. It compares the hip-to-knee
direction with the hip-to-shoulder direction. The idea is to drop seated
people's feet before the player test. Dropping people can only make the test
stricter, so fewer oversized courts pass.

What it needs:

- **Pose keypoints** for every frame whose feet the player test uses. The
  court packs hold person boxes and scores only. The saved ShuttleSet pose
  arrays do not fit: the pose step keeps only the ten highest-scoring people
  per frame (`src/bst_x/preparing_data/raw_extract.py:88`). The test therefore
  ran a fresh detection and pose pass on every view. In deployment, keypoints
  are free if court detection runs after the pipeline's pose step; that
  ordering is unchecked
- **A one-line filter** where each view's feet array is built

Expected effect: large for seated broadcast and GX crowds. Amateur halls are
mostly standing bystanders, so am2 may gain little.

Recall risk: a deep lunge can make a real player look seated for a frame.
The right court fails "anyone inside" only if both players are flagged in
the same frame. "Both halves" tolerates misses in up to half the frames.

### Result: every court kept, axis matching down 27%

Both runs below use the same fresh detections, so the seated rule is the only
difference. Each is a full-search D17 run on all 28 views, run side by side on
Carmack. The method is in "How items 10 and 11 were tested".

| Measure | Everyone | Seated removed | Change |
| --- | ---: | ---: | ---: |
| Court views keeping their court, of 20 | 20 | 20 | — |
| Axis hypotheses scored | 77.7 M | 64.5 M | −17% |
| Axis matching time | 3,967 s | 2,878 s | −27% |
| Combined courts scored | 63.3 M | 62.5 M | −1.3% |
| Pair loop time | 8,766 s | 7,881 s | −10% |
| Whole run | 13,696 s | 13,008 s | −5% |

Carmack is shared, and per-view times vary by up to 40%. Trust the counts over
the times.

The rule prunes most where crowds sit. Median detected people per sample:

| Views | Everyone | Seated removed |
| --- | ---: | ---: |
| GX (4 views) | 18–25 | 6–9 |
| ShuttleSet 03 (6 views) | 57–81 | 23–37 |
| ShuttleSet 21 (3 views) | 13–14 | 10–11 |
| Amateur (5 views) | 13–41 | 10–34 |

Combined courts barely fall because `match_axis` keeps a fixed number of
distinct axes per pair (`settings.keep_axes`). When the seated rule removes an
axis, the next-best axis takes its slot. On seven of the 20 court views, more
combined courts reach scoring with seated people removed. The largest rise is
21%, on GX5. The saving therefore comes from axis scoring alone.

Accuracy holds. 17 of 20 corrected courts match the baseline run on the
frozen feet to 0.01 px, although some pick IDs change. Three ShuttleSet 03
courts shift by 0.5–3.9 px. Against the
reference landmarks, each is as close as the baseline's or closer, apart from
0.04 px on 03/0032:

| View | Baseline | Everyone | Seated removed |
| --- | ---: | ---: | ---: |
| 03/0017 | 2.66 | 3.44 | 2.05 |
| 03/0032 | 1.68 | 2.19 | 1.72 |
| 03/0034 | 2.53 | 2.33 | 2.33 |

Values are median reference errors in working pixels.

Precision is not settled; see "Fresh detections put a court on a non-court
control". Against the same detections with everyone kept, the seated rule
removed one false court (control 38228) and added none.

A Codex review (Sol, high effort) found no other recall-safe rule. Two
alternatives each have a direct way to reject the correct court:

- **A cap on people inside each court:** spectators, officials or players on
  the next court can sit inside the correct court's 15% margin
- **A both-halves vote without the margin:** a correct court whose players
  stand just past a line fails, especially with few frames

A player test can reject courts in the wrong place. It cannot reject an
oversized court that still contains the players.

## 11. Rejected: count only the people who move most

Keeping only the top movers fails the player test on real courts, so this
item is rejected. The screen replayed the player test on each view's accepted
baseline court. D17 cannot return a court that fails it. It might return a
nearby court instead, but full runs were not done for these variants. The
screen is a diagnostic only; it is not a proposed player test.

| Feet kept | Accepted courts failing the test, of 20 |
| --- | ---: |
| Top 6 movers | 9 |
| Seated removed, then top 6 movers | 6 |
| Seated removed, then top 10 | 3 |
| Seated removed, then top 15 | 1 |
| Seated removed, then top 20 | 0 |
| Seated removed, no cap | 0 |

The cap needed grows with how busy the hall is, not with the two players.
Over 3 s, a player waiting to serve or receive barely moves. Players on
neighbouring courts, officials and passers-by often move more. am2 28019, the
last view to fail, has a median of 34 standing people per sample. A cap of 20
keeps every court, but it trims little there. Track linking also ends a track
at one missed detection, so an occluded player's motion is split and
undercounted. Evidence:
`claude_evidence/fresh_feet/mover_caps_screen.txt` and `screen_output.txt`.

The original proposal follows.

Players cover ground in a rally, while crowds and most bystanders stay put.
The rule keeps only the top movers across the sampled window:

1. Match each person's box to the nearest box in the next frame, for example
   with `scipy.optimize.linear_sum_assignment` on box centres. The match needs a
   distance limit, so unrelated boxes are never joined across a gap
2. Measure each step in body heights (step length divided by box height). In
   pixels, the far player moves less than a near spectator
3. Keep the top four to eight movers, then run the existing player test on
   their feet only

This prunes before axis scoring, like item 10. The "one player per end" check
works on each vertical-axis hypothesis's far and near halves, so it needs no
detected court. The guaranteed saving is in axis scoring. Combined-court
scoring shrinks only when the joint player test also rejects more courts.

Recall risks:

- a serving or paused player ranks low
- players on an active neighbouring court rank high
- a missed detection splits one player's track in two

A fallback to everyone when motion is weak brings the crowd back exactly
where pruning would help. The rule therefore has to show zero lost courts on
its own. Items 10 and 11 complement each other: the seated rule needs no
matching across frames, and the movers rule also catches standing bystanders.

## 12. Tested: stop scoring courts that cannot make the shortlist

An exact score bound does not pay here, although the room is large. Each pair
keeps its 256 best distinct courts by distance-map score. Only about 1% of a
pair's courts score at or above the 256th. The rest could be skipped if a cheap
upper bound on their score fell below that cutoff.

The bound held: across the tested pairs, no court scored above it. But it is
too loose to skip much, and it costs about half of full scoring. Local run on
`shuttleset_21_scene_0044`, first three matched pairs of G0 and G1:

| Courts in pair | At or above the cutoff (whole batches of 256) | Must score with the bound | Bound cost / full scoring |
| ---: | ---: | ---: | ---: |
| 94,524 | 1,280 | 73,472 | 56% |
| 104,538 | 1,024 | 32,768 | 60% |
| 53,408 | 1,024 | 5,888 | 53% |

The bound reads 1 sample in 4 and bounds the rest by Lipschitz continuity: a
distance-map value changes by at most the pixel distance between two samples.
It stays loose because every candidate court is built from detected lines. Its
lines sit within a few pixels of real lines, and scores differ by pixel-level
alignment that sparse samples cannot see. Most of the scoring cost is also
fixed per court, so reading fewer samples saves little.

Exact rewrites help instead. Commit d1609d6c keeps x and y, or a segment's
two endpoints, as separate arrays in three hot spots:

- the continuous-support sampler (`scan_population.continuous_support`)
- segment distances (`assignment.distances_to_segments`)
- axis scoring (`projective_seed.score_axes`)

numpy is 2–3× slower on an innermost axis of length 2. Each rewrite keeps the
original arithmetic order. A test for each compares it with the original
formula byte for byte, so signed zeros and NaN bit patterns must match too.
The same commit builds role-proposal records only when a court reaches a
shortlist, and lands item 4.

```python
sample_x = starts[..., 0, None] + fractions * vectors[..., 0, None]
sample_y = starts[..., 1, None] + fractions * vectors[..., 1, None]
```

Carmack check: full search on all 28 views, 8 jobs, with the seated-removed
feet from item 10, against the same run at commit 6178fda5. All 168 output
files are bit-identical. The only differences are the intended stage-call
moves: refit `distance_maps` went from about 753 calls per view to one per
view. Summed wall time fell from 13,002 to 8,931 s (−31%).

| Stage (summed over 28 views) | Before (s) | After (s) |
| --- | ---: | ---: |
| Axis matching (`match_axis`) | 2,878 | 1,466 |
| Court scoring (`finite_scores`) | 2,978 | 2,077 |
| Stripe measurement (`stripe_measure`) | 1,870 | 1,139 |
| Pair loop, own time | 986 | 179 |
| Distance maps | 412 | 85 |

Stages the commit did not touch ran 6–16% slower in the second run. That is
most likely node load (see the verification plan), though one pair of runs
cannot prove it. Per view, the new/old wall-time ratio ran from 0.50 to 1.47.
The view that was slowest before, `gxBQ_window_00_frame_0`, went from 1,344 to
674 s.

Before that run, a Sol red-team (high effort) found two gaps in the checking,
both since fixed. The saved benchmarks had imported the rewritten functions,
so they compared new against new. The run comparer also let some differences
through: `-0.0` against `0.0`, unread `.npy` files, and run-folder prefixes
replaced anywhere in a string. The rewritten comparer (float bits, dtype,
shape and raw bytes) produced the 168/168 result above.

A non-exact route remains: rank each pair's courts by a coarse score, then
score exactly only the top few thousand. Item 16 measures it.

Evidence: `claude_evidence/shortlist_bound/`. `per_sample_bound_rows.jsonl`
used one bound per sample, which was tighter but cost as much as full scoring.
`gap_bound_rows.jsonl` used one bound per gap between measured samples, as in
the script.

Checked and dropped: scoring bit-identical axis hypotheses only once. Two
hypotheses get identical (scale, shift) bits only if different observed lines
happen to give exactly the same floats. An observation-only hook counted
none: 0 duplicates among 321,224 scored hypotheses. That covered 24
`match_axis` calls, from the first three matched pairs of
`shuttleset_21_scene_0044` and `gxBQ_window_00_frame_0` (local runs). Evidence:
`claude_evidence/axis_duplicates/`.

## 13. Done: rewrite the template camera check (commit 44254b42)

This is the one W5 saving worth doing now. On Carmack it cut the check from
564 to 92 s over 28 views (5.3% of the run), with every output file
bit-identical.

After item 12, W5 takes 3,281 of 8,931 s (37%). Summed over the 28 views of
the Carmack run in item 12:

| W5 part | Seconds | Share of run |
| --- | ---: | ---: |
| Line templates | 975 | 10.9% |
| of which the template camera check | 564 | 6.3% |
| Measuring parents | 805 | 9.0% |
| Measuring children | 639 | 7.2% |
| Refit (`refine` 351, `prepare` 42) | 393 | 4.4% |
| Writing the case record and arrays | 373 | 4.2% |

The camera check (`line_template_source.vector_camera_errors`) tests each
template court against 200 focal lengths. It builds arrays shaped (courts,
200 focal lengths, 3, 2) and reduces over the axes of length 3 and 2. That is
the same slow layout as item 12. A probe on `shuttleset_21_scene_0044` put
28.1 of the check's 28.5 s in this vectorised part. The scalar rechecks near
the camera limit take the rest.

The rewrite keeps each court direction's x, y and w parts as separate
(courts, focal lengths) arrays and sums them left to right, as numpy does.
It refuses float32 input, because the old form kept float32 results and the
new one would silently return float64:

```python
if homographies.dtype != np.float64:
    raise TypeError(f"camera errors need float64 homographies, got {homographies.dtype}")
width_x = (homographies[:, 0, 0] - image_width / 2.0 * homographies[:, 2, 0])[:, None] / focals
width_y = (homographies[:, 1, 0] - image_height / 2.0 * homographies[:, 2, 0])[:, None] / focals
width_w = homographies[:, 2, 0, None]
# ... the same for the length direction (column 1)
width_norm = np.sqrt(np.square(width_x) + np.square(width_y) + np.square(width_w))
dot = width_x * length_x + width_y * length_y + width_w * length_w
cosine = dot / (width_norm * length_norm)
```

It is in `scratch/court_det_fix/w5_holistic/line_template_source.py`. Two tests in
`test_w5.py` go with it. One compares the new function with a verbatim copy of
the old one, byte for byte, on 400 homographies that include affine,
collapsed, infinite, NaN and `-0.0` rows. It fails if the sum is regrouped as
`x² + (y² + w²)`. The other checks that float32 input raises.

Checks so far:

- **Byte-identical** on 80 of 80 real batches saved from a D17 run, and on
  30 perturbed ones with zero, infinite, NaN and `-0.0` entries
- **About 6–7× faster** on the 80 real batches (37,895 courts): 2.80 → 0.43 s
  and 2.71 → 0.39 s in two laptop runs
- **Tests**: the camera tests pass locally and on Carmack's numpy build (4
  passed). The full pytest suite shows only the one old failure (`python` not
  on PATH). Pyrefly reports 0 errors, and Ruff is clean on the changed files
- **Local D17 run** of `shuttleset_21_scene_0044` (first three matched pairs):
  bit-identical to the run with item 14, and the camera check fell from 31 to
  8 s

Two Sol red-teams (high effort) reviewed it. The first, on the candidate,
raised two caveats:

- **float32 input would change dtype.** The only caller passes float64
  (`line_template_source.py:436`), so the function now refuses anything else
- **Bit identity rests on numpy's summation order.** numpy 2.5.3 sums a
  length-3 axis left to right (its pairwise helper runs a plain loop below 8
  items). That is how this build works, not a documented promise. Only a
  Carmack identity run confirms Carmack's build

The second, on the built code, found it byte-identical across image sizes
given as Python or numpy numbers, empty batches, strided and Fortran-ordered
input, and extreme magnitudes. It flagged that the first version's float64
`assert` disappears under `python -O`, so that became the `TypeError` above.

**Carmack identity run.** The full search on all 28 views, with this change
applied on top of item 14's commit, against the current baseline:

- every saved result is bit-identical across all 168 output files
  (`compare_exact_runs.py`). This covers items 13 and 14 together
- the camera check fell from 564 to 92 s (−84%), and `prepare_observations`
  from 100 to 21 s (item 14). Both fell on every view. Together they saved
  551 s, 6.2% of the baseline
- summed over views, the run fell from 8,931 to 8,250 s (−7.6%). Stages
  neither change touches moved by −7% to +7% between the two runs, which were
  on different days. That variation is unexplained, so only the two stage
  times above are firm
- peak memory, which the comparer does not check, moved between −11% and +18%
  per view, with a median of −0.1%. The highest peak stayed at about 2.0 GB

Nine views ran separately. The first launch failed on the nine `sset_21`
views, because the new checkout lacked a git-ignored folder of control frames.
That folder was copied from the baseline checkout, checked identical, and only
those nine views were rerun. They failed before writing any output.

A Sol red-team (high effort) of these results found them sound with caveats.
It corrected the wording above: "bit-identical" holds for saved results, not
raw file bytes, and the other stages' variation is unexplained rather than
proven noise. It found no result that differs.

Evidence: `claude_evidence/w5_savings/`, including
`carmack_compare_vs_item12.txt` and `carmack_stage_times_vs_item12.txt`.

## 14. Done: cache `prepare_observations` (commit 7c74b015)

One D17 view calls `prepare_observations` seven times. Five of those calls
repeat an earlier call's input exactly. `prepare_observations` now caches its
result for the default fragment IDs:

- `functools.lru_cache(maxsize=8, typed=True)`, keyed on the float64 segment
  bytes, the image width and the image height
- each call gets a `copy.deepcopy`, so callers never share arrays
- calls with explicit fragment IDs take the original, uncached path

A Sol red-team found one real key collision in the first version, which
keyed on `tuple(size)`. Sizes `(2050, 100)` and `(np.float16(2050), 100)`
compare equal, but float16 rounds `2050 - 1` to 2048 when clipping fragments to
the image. The second call then got the first call's result. D17 passes plain
integers, so no D17 result was affected. Passing width and height as separate
arguments with `typed=True` keys different number types apart. A regression
test covers Sol's case and failed before the fix.

Checks:
- the new test fails when the copy is removed
- the full pytest suite shows only the one old failure (`python` not on PATH)
- Pyrefly reports 0 errors, and Ruff is clean on the changed files
- a local D17 run of `shuttleset_21_scene_0044` (first three matched pairs) is
  bit-identical to the run before the change

Time in `prepare_observations` for that view fell from 3.08 to 0.56 s. Over
28 views on Carmack it fell from 100 to 21 s, under 1% of the run. That run,
under item 13, also confirmed the cache bit-identical on every view. That is below the 1% bar in "What not to do", but the project
owner judged the few lines worth it.

## 15. W5 savings not worth doing now

- **Camera-check only the courts selection reads.** Selection walks template
  courts in score order and stops at 256 kept. On `shuttleset_21_scene_0044`
  it read the top 1,058 of 277,192 visibility-admitted courts (0.38%), and the
  floor-zero comparison pass read 1,306 of 412,333 (0.32%). Checking cameras
  score-order chunk by chunk would give the same selection, because each
  court's camera error depends only on that court. But after item 13 only
  about 90 s (1%) is left to save. It would also lose the template-wide counts
  in the metadata. One of them, `combined_admission_hypotheses`, is read and
  validated by `compare_directional_runs.py`. Sol agreed the selection would be
  the same, provided both greedy passes run until they keep 256 courts or run
  out of courts. Revisit only inside a deployment path
- **Measurement internals.** A profile of 200 parent measurements put 65% of
  the time in stripe measurement, 24% in physical marking evidence and 10% in
  the stripe score model. The largest single part, segment distances, is
  already rewritten (item 12). The rest is spread over many small array steps
  with no single hotspot. More speed would need a sparser evidence layout that
  every consumer of the stripe evidence adopts
- **Refit only parents near the best score.** D17's net choice can only pick
  a court within 0.04 paint score of the best eligible court. On the 28 views,
  every child within 0.04 of the best court had a parent within 0.288 of the
  best parent. A cut at 0.288, 1.5× or 2× that skips 58%, 32% or 14% of refits.
  The safe margin is empirical, though: a new view could need a wider one.
  Not exact, so not recommended
- **gzip level 6 for research records.** `write_json_gz` compresses at level
  9. On one 43 MB record, level 6 takes 1.3 s instead of 3.9 s for a file 3%
  larger with the same content. Scaled to the Carmack run, that is about 55 s
  (0.6%). It changes the compressed bytes, which breaks byte-level comparison
  with older runs. Not worth it if deployment skips these files (see
  "Deployment blocker" below)

Refit (`refine`, 351 s) has not been profiled since item 12.

## 16. Tested: score coarsely, then exactly only the top K

Not recommended. On these views it would have kept every shortlist and saved
about a third of court scoring. But nothing at run time can show that a new
view stays within the chosen depth, so it can lose courts.

The idea: score every court in a pair with the same distance-map support
score, but with 4, 8 or 16 samples per marking instead of 64. Score exactly
only the K courts with the best coarse scores, then keep the shortlist from
those as now. The shortlist is unchanged whenever the K courts include every
court the full search keeps. `retain` walks courts in exact-score order and
skips any within 2 px of one already kept, so dropping courts it would never
keep changes nothing.

Measurement: an observation-only hook on the full D17 chain, all 28 views on
Carmack at commit d1609d6c. That is 3,369 scoring calls on 23 views; five
non-court views never scored a pair. The hook changed nothing: all 168
output files are bit-identical to the run in item 12. For each call it recorded
the coarse depth needed to hold every kept court, and rebuilt the shortlist
from exactly that many courts.

| Samples per marking | Shortlist rebuilt identically | Deepest K needed | Fixed K that covers every call | Estimated cost at that K |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 3,369 of 3,369 | 70,121 | none that pays | — |
| 8 | 3,369 of 3,369 | 22,863 | 32,768 | about 90% of exact scoring |
| 16 | 3,369 of 3,369 | 4,511 | 8,192 | about 66% of exact scoring |

At 16 samples the worst depth per view ranged from 91 to 4,511, so K = 8,192
has a margin of 1.8× over the worst call. The cost estimate adds one map build,
the coarse pass, and exact scoring of K courts, scaled linearly from a timed
exact rescoring of the needed courts. It is not an end-to-end measurement.
In item 12's run, court scoring took about 2,160 of 8,931 s: 2,077 s in
`finite_scores` plus 85 s building its distance maps. A third of that is
roughly 8% of the run.

Why not:

- **K is fixed before the view is seen.** `shuttleset_03_scene_0017`, call
  88, needs K = 4,511. At K = 4,096 it would drop a kept court. That court's
  coarse score sits 0.001 below the 4,096th, so neither rank nor score gap
  gives warning
- **The only certificate is item 12's bound.** Excluded courts are provably
  safe only if an upper bound on their exact scores falls below the 256th kept
  court's score. That bound costs about half of exact scoring and is too loose
  to skip much
- **It is the same kind of rule as the refit margin in item 15**: safe on the
  views measured, with a margin chosen by eye

A Sol red-team (high effort) confirmed the shortlist argument and the saved
results, with caveats. The results are empirical for these views, not a
guarantee. It also found that the first cost estimate counted the distance
maps twice (78% instead of about 66%). If this is ever built, pass the top K
courts to `retain` in their original proposal order. `retain` sorts stably, so
tied exact scores are broken by input order, and coarse-rank order could flip
which of two tied courts is kept.

Evidence: `claude_evidence/prefilter/`.

## 17. Tested: float32 and float16

Neither pays for the camera check, and float16 is unusable. float32 across the
whole pipeline is untested.

The test ran the item 13 camera check on 80 real batches (37,895 courts),
saved from a local D17 run of `shuttleset_21_scene_0044`. Each lower-precision
version did all its arithmetic in that type. Three laptop runs:

| Type | Time for 80 batches | Largest difference from float64 | Camera-gate changes (3,319 courts pass) |
| --- | ---: | ---: | --- |
| float64 (item 13) | 0.42–0.73 s | — | — |
| float32 | 0.17–0.32 s | 8.4e-7 | None |
| float16 | 3.6–4.3 s | 0.074 | 2, plus 4 courts that overflow to unusable |

**float16 fails three ways:**

- CPUs have little native float16 arithmetic, so numpy converts every value
  to float32 and back. That makes it 5–6× slower than float64
- homography entries reach 481,000 in these batches, and float16 tops out at
  65,504
- it keeps about 3 significant digits, so a court that truly scores 0.03
  could read above the 0.1 gate

**float32 for the camera check alone is not worth it.** Gate decisions would
not change: courts within ±0.001 of the gate are already rechecked in float64,
and float32's error is far smaller than that. But the stored camera errors
would change in their last bits, so runs would no longer match byte for byte.
The saving is about 50 s (0.6%).

**float32 across the whole pipeline is an open question, not a measured
one.** Court scoring, axis matching and stripe measurement are about half the
baseline. The distance maps are already float32, and court scoring is mostly
lookups into them, which gain little from narrower numbers. A plausible gain is
1.2–1.6× on those stages, or 10–20% of the run. That is a guess. The risks:

- fitting must stay float64. Homography entries span 9e-6 to 4.8e5, and
  least-squares fits lose real accuracy in float32
- tiny score changes can cross hard cut-offs: pixel rounding in scoring, the
  2 px duplicate radius, the 256-court cut, the camera gate and the final
  pick. Near-tied courts can then swap, though they score the same to about
  six digits
- numpy turns float32 back into float64 whenever it meets a float64 array,
  such as `np.linspace` defaults or constant arrays. Keeping float32 means
  auditing every array the scoring code creates
- checks against earlier runs would need tolerances instead of byte
  comparisons

The cheapest test is to time court scoring (`continuous_support`) and axis
scoring (`score_axes`) in float32 on saved inputs. Only if both reach about
1.4× is a 28-view run worth doing, with float32 inside those two functions and
fitting left in float64. That run would compare each view's winning court,
net choice and reference error with the baseline.

Evidence: `claude_evidence/precision/`.

## How items 10 and 11 were tested

The frozen feet windows differed by view type: about 30 frames for GX and
amateur views, 3 frames across a scene for ShuttleSet, and 1 frame for the
non-court controls. So every view got a fresh window instead:

1. **Frames.** 31 samples, one every 0.1 s, centred on the analysed frame
   and shifted to fit inside the video. The analysed frame is always included.
   Decoded frames were checked against the frozen images; ShuttleSet composites
   are grey medians and were checked in grey
2. **People.** RTMDet-M boxes above score 0.2, then RTMPose-L keypoints, on
   the full-size frame. A foot is the bottom centre of a box, scaled to the
   view's pack size. A foot outside the image is unknown, as in the frozen packs
3. **Same shot only.** A sample counts only if it and every sample between it
   and the analysed frame stay within 8 grey levels of the analysed frame, on
   64×36 thumbnails. In-shot samples of the court views differ by at most 5.1.
   This drops a dissolve (control frame 1) and a passer-by at the lens (am3
   frame 0)
4. **Variants.** Everyone; seated removed (`is_sitting` at −0.3, used as
   sticky_anchor uses it); top movers (Hungarian matching on feet, gated at one
   body height per step, ranked by summed steps in body heights); both
5. **Screen.** Replay the player test on each view's accepted baseline court
6. **D17.** Full search on all 28 views for the variants that pass the screen.
   `run_d17.py --feet FILE` swaps in the feet, and `match_axis` records how
   many axis hypotheses were scored (commit 6178fda5)

Two Codex red-teams (Sol, high effort) checked this. The first found no code
errors in steps 1–5 or the swap, and confirmed the swap reaches every player
test in the chain. The second checked the conclusions against the logs and
code; its corrections are folded in here. Their caveats:

- the screen only shows that D17 cannot return that exact court
- the fresh results describe the fresh detections, not the frozen packs
- the second review read the saved logs, not the raw run files

Scripts and logs are in `claude_evidence/fresh_feet/`.

## Fresh detections put a court on a non-court control

On non-court control 100347, both fresh-detection runs accept a court. The
baseline, on the frozen feet, correctly returns none. This is a precision
loss, and the seated rule did not cause it.

| Non-court control | Baseline (frozen feet) | Everyone (fresh) | Seated removed (fresh) |
| --- | --- | --- | --- |
| 100347 | none | court | court |
| 38228 | none | court | none |
| 14336 | wrong court | same wrong court | different wrong court |
| Five others | none | none | none |

The fresh pass finds 3–7 people per sample on 100347, where the frozen pack
had none. On the analysed frame they score 0.21–0.31, with boxes 34–112 px
tall in the 1920 px frame. Only 6 samples count as the same shot there: the
image changes steadily across the 3 s before the analysed frame.

The fresh input differs from the frozen control packs in three ways at once
(`wider_evaluation/prepare_controls.py`):

- full-size video frames instead of one 960×540 image
- a person score cutoff of 0.2 instead of 0.3
- up to 31 samples instead of one

Which change lets the court through is untested. The frozen packs already had
people on 14336 (two) and 38228 (three), so "no people" does not explain
every control.

Deployment input is arbitrary video, so precision depends on whatever the
person detector finds there. Any rule that uses fresh detections needs the
non-court controls rechecked with those detections.

## Input resolution

The court search already shrinks every input until its longest side is at
most 960 px (`experiments/annotator/independent_court/export_lines.py:17`).
A 720p video and a 1080p video therefore reach the same 960×540 working image.
A 720p benchmark would mainly test the person detector, whose far players get
smaller. It has not been run.

Going below 960×540 would cut the work that scales with pixels: resizing, line
finding and the two distance maps built per pair in court scoring. The two
largest stages, axis matching and per-court scoring, scale with hypothesis
counts instead. The net speed-up is unmeasured. The accuracy cost is likely:
reference errors are about 2 working px now, and each working pixel would
cover more of the frame.

## Deployment blocker: W5 reads git-ignored control files

W5's `load_control_entry` loads diagnostic control candidates for views listed
in `KNOWN_CONTROLS`. It reads them from `automatic_axes_20260914/all_camera`,
which git ignores. A fresh checkout therefore fails on both am2 views with
`FileNotFoundError`. Every Carmack D17 run so far needed those three files
copied in by hand.

This work is research-only and takes under 20 s over 28 views. A deployment
mode must skip it. The same mode can drop the other research-only work:
population files, full records and arrays, the determinism re-check and the
refit replay. Before the speed-ups, these cost 578 s of 20,648 s on court
views (3%).

After item 12 they matter for speed too. In the Carmack run under
item 12, file writing and re-reading took about 580 of 8,931 s (6.5%):

- case records: 314 s
- population files: 119 s
- arrays: 59 s
- reading populations and records back in later stages: about 89 s

The research-only computation left in `run_w5.process_case`, such as rank
sensitivity (8 s), is small. Skipping the files means handing populations and
the case record between stages in memory. That is the same work whether it is
a flag in the research harness or a separate deployment entry point, so do it
once, in the deployment entry point.

## What not to do

- **SVD12 screen as the speed-up**: it halved the run time before items 3 and
  5, but the corrected court changes on five of 20 court views. Letterboxed
  and ss21 0044 get clearly worse. Items 3 and 5 took a third off exactly

- **Prepared-structure types, basis-inverse hoisting (F2), LRU caches with
  hit-rate reports**: the costs they target are under 1% of pair time
- **Affine scoring kernel (F3)**: needs rounding-frontier fallback code; item
  5 targets the same cost exactly
- **Per-pair generation map cache (F7)**: about 20 ms per pair, rare key
  repeats
- **Batched gate evidence (F9)**: 1% of pool evidence
- **Refit cache and analytic Jacobian (F11)**: `least_squares` is about 4% of
  the refit stage
- **`cv2.polylines` for map drawing**: the distance transform takes 90% of
  map cost
- **Chunking `player_fractions` to cap memory**: item 3 removes the large
  array
- **Capping people by detector score**: players in motion, occluded or behind
  the net can score below spectators (the project owner's call). This sample
  does not show the failure: a top-6 cap passed the screen and top 4 failed
  both am2 views. Note that the frozen ShuttleSet feet already carry a top-10
  cap by score from the pose step
- **Capping people by movement**: see item 11
- **Shrinking the 256-court shortlists**: the accepted court's parent ranked
  as low as 41st in its pair (03/0023) and 107th of 256 overall (am2 150), by
  court score. W5's measuring and refit reorder courts heavily. A per-pair cap
  of 32 would already drop the parent on one of 20 views. Four accepted courts
  came from line templates instead
- **Moving the camera gate before court scoring** (Codex): W5 refits every
  valid parent, and a parent that fails the camera gate can yield a child that
  passes

## Verification plan

Use the new Carmack venv `~/.venvs/court_det` for every run from here:
Python 3.12.13, numpy 2.5.3, scipy 1.17.1, opencv-contrib-python 5.0.0.93,
plus pytest, jaxtyping, beartype and ripgrep. The SVD-search runtime loads only
numpy, scipy and OpenCV. Runs in the old `venv-rtmlib` (numpy 2.4.6) are not a
bit-level baseline for it.

1. **New reference (done; see item 2).** Apply item 2 on top of patches 1 and 2. Run all six views
   twice with different `PYTHONHASHSEED` values. Expect bit-identical records
   between the two runs, which confirms the maps were the only source of
   noise. Against the 23 September baseline, expect the same pools, winners and
   selected courts, with score differences up to about 1e-8
2. **Exact items (done for 4, 5, 12, 13 and 14).** Run the full
   search on all 28 views and compare with the run before the change, using
   `claude_evidence/exact_rewrites/compare_exact_runs.py`. It checks float
   bits, dtype, shape and raw bytes. Expect bit-identical files, apart from
   timing and any named diagnostic fields
3. **Item 3 (done, covered by step 4).** Expect bit-identical records.
   Any pool or winner difference would point at a band-edge disagreement
4. **End to end (done).** After items 3 and 5, the D17 harness was rerun with
   the full search on all 28 views and compared with the 24 September run,
   using `claude_evidence/d17/compare_d17_runs.py`. All 168 files were
   identical, apart from the expected `pattern_supported` None. Results are in
   `claude_evidence/d17/players_rerun_results.txt`

Each six-view run took 14 minutes on Carmack with six workers after patches 1
and 2. The two item-2 reference runs took 19 and 20 minutes. That gap is node
noise, not item 2. Carmack is a shared virtual machine, and the same work took
up to 40% more or less CPU time per view between runs. For example, the
`shuttleset_03_scene_0034` generation took 165, 168 and 101 CPU seconds in three
runs. Six-view totals agreed within about 5%. Compare speed only between arms
run side by side at the same time, and prefer totals over single views.
