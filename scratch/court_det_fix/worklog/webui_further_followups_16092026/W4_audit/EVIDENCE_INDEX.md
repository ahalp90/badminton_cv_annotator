# Where the findings come from

This file is for checking a number or finding the code behind it. The [main report](W4_FINAL_REVIEW.md) explains the findings without these source details.

## Reviewed version

Repository: `ahalp90/badminton_cv_annotator`, branch `fix/court-det`.

The review used commit **`b36402f1c994f2b000044589cdb6b001f190d0dd`**, “Preserve W2 pixel review evidence,” committed **17 September 2026, 06:23:50 UTC / 16:23:50 Melbourne time**. Every repository link below points to that commit, not to a moving branch. This rewrite did not check for a newer commit.

Most trial files sit under:

```text
scratch/court_det_fix/next_steps_20260916/
```

The old helper code sits under:

```text
scratch/court_det_fix/frozen_helpers_20260914/
```

Files with the same name in different folders are not necessarily the same code. The notes below explain which copy each test imports.

<a id="corrections"></a>
## Earlier corrections: what is still open

The original W4 review read the changed text in these files, not only the summary saying that changes had been made:

- [GX direction note](https://github.com/ahalp90/badminton_cv_annotator/blob/b36402f1c994f2b000044589cdb6b001f190d0dd/scratch/court_det_fix/worklog/webui_evaluation_returns_15092026/CLAUDE_FOLLOWUPS/gx_directions/note.md).
- [Camera calibration assessment](https://github.com/ahalp90/badminton_cv_annotator/blob/b36402f1c994f2b000044589cdb6b001f190d0dd/scratch/court_det_fix/worklog/webui_evaluation_returns_15092026/camera_calibration_assessment.md).
- [Follow-up schedule](https://github.com/ahalp90/badminton_cv_annotator/blob/b36402f1c994f2b000044589cdb6b001f190d0dd/scratch/court_det_fix/worklog/webui_evaluation_returns_15092026/CLAUDE_FOLLOWUPS/SCHEDULE.md) and [status](https://github.com/ahalp90/badminton_cv_annotator/blob/b36402f1c994f2b000044589cdb6b001f190d0dd/scratch/court_det_fix/worklog/webui_evaluation_returns_15092026/CLAUDE_FOLLOWUPS/status.md).

The [C1 change list](https://github.com/ahalp90/badminton_cv_annotator/blob/b36402f1c994f2b000044589cdb6b001f190d0dd/scratch/court_det_fix/next_steps_20260916/C1_corrections/changes.md) gives the surrounding context.

These sources distinguish lower bounds from achieved fits, and achieved fits from generated courts. They leave the separate Amateur-2 test unrun rather than treating it as disproved. They describe the distortion methods as inconclusive. They also keep line-spacing and ambiguity warnings separate from rules that have actually been tested for rejecting courts.

The camera assessment is also the source for the older visual judgement that GX0 `22:4579` and `22:4588` are skewed. W4 did not make that judgement from fresh image viewing.

<a id="c2"></a>
## C2: checking the earlier explanation

### Data and code

The saved corners, assignment IDs and scores are in [webui_seed/witnesses.json](https://github.com/ahalp90/badminton_cv_annotator/blob/b36402f1c994f2b000044589cdb6b001f190d0dd/scratch/court_det_fix/next_steps_20260916/webui_seed/witnesses.json). That file records `d92979bcffee197e0748118ad452093b27a453d4` as the commit present when C2 was checked. That is different from the later commit reviewed by W4.

The check is implemented in [C2_traces/check_traces.py](https://github.com/ahalp90/badminton_cv_annotator/blob/b36402f1c994f2b000044589cdb6b001f190d0dd/scratch/court_det_fix/next_steps_20260916/C2_traces/check_traces.py). In particular, `record_pool_witness` loops over every matched pair with a saved shortlist. It checks the total against `pooled_candidates`. `nearest_product_witness` examines the combinations from the specified pair; it does not represent the whole set.

The duplicate rule is in the frozen [axis_matching/projective_seed.py](https://github.com/ahalp90/badminton_cv_annotator/blob/b36402f1c994f2b000044589cdb6b001f190d0dd/scratch/court_det_fix/frozen_helpers_20260914/axis_matching/projective_seed.py): `corner_errors` at lines 14–18, `match_axis` at 135–187 and `combine` at 190–201. It sorts eligible assignments by descending score before removing repeated assignments. Both scale signs are considered, and direction roles are kept when forming courts.

### Which helper was used?

C2 uses the import setup in [line_identity/shared.py](https://github.com/ahalp90/badminton_cv_annotator/blob/b36402f1c994f2b000044589cdb6b001f190d0dd/scratch/court_det_fix/line_identity/shared.py), the current `line_identity/axis_replay.py`, and the frozen axis matcher. The [seed's source map](https://github.com/ahalp90/badminton_cv_annotator/blob/b36402f1c994f2b000044589cdb6b001f190d0dd/scratch/court_det_fix/next_steps_20260916/webui_seed/source/SOURCE_MAP.md) identifies the relevant copies.

The saved records before the geometry/player checks came from the copies of `run_automatic.py` and `run_given.py` in:

```text
scratch/court_det_fix/worklog/webui_evaluation_returns_15092026/
  CLAUDE_FOLLOWUPS/pregate_loss/remote_src/
```

Those copies write extra intermediate results. They should not be confused with another file of the same name elsewhere in the repository.

### What W4 recalculated

W4 recalculated the four GX corner errors and compared the scores and assignment IDs for the Am3 duplicates. The copied fields are in [inputs/c2_compact.json](inputs/c2_compact.json); the calculation is in [results/c2_check.json](results/c2_check.json).

It did **not** download and rerun the large searches, recheck every reported rank, or reconstruct the Am3 geometric errors. The large inputs named by the saved examples are:

```text
scratch/court_det_fix/frozen_views/baseline_generation/
  gxBQ_window_00_frame_0.json.gz

scratch/court_det_fix/line_identity/runs/line_identity_20260915_222437/
  matcher/paint_observations/results/gxBQ_window_00_frame_0.json.gz

scratch/court_det_fix/worklog/webui_evaluation_returns_15092026/
  CLAUDE_FOLLOWUPS/pregate_loss/records/new/R/results/
  am3_window_00_frame_0.json.gz
```

W4 did not recompute them. Missing access to those files is not a request to commit them.

<a id="l1"></a>
## L1: changing which line assignments survive

The key files are [run_l1_admission.py](https://github.com/ahalp90/badminton_cv_annotator/blob/b36402f1c994f2b000044589cdb6b001f190d0dd/scratch/court_det_fix/next_steps_20260916/L1_admission/run_l1_admission.py), [comparison.csv](https://github.com/ahalp90/badminton_cv_annotator/blob/b36402f1c994f2b000044589cdb6b001f190d0dd/scratch/court_det_fix/next_steps_20260916/L1_admission/comparison.csv) and [result.md](https://github.com/ahalp90/badminton_cv_annotator/blob/b36402f1c994f2b000044589cdb6b001f190d0dd/scratch/court_det_fix/next_steps_20260916/L1_admission/result.md), all under the trial folder.

The code's constants and `PAIR_SPECS` define the size and scope of the test. `evaluate_selection` forms courts, applies the existing checks and keeps the per-pair shortlist. The reference corners are loaded after the selected assignment IDs have been fixed. The six direction pairs were still chosen for this test; that does not make it a run over every possible pair.

The table contains 12 rows: two rules on each of six direction pairs. Every row keeps 512 assignments per axis, forms 262,144 courts and ends with 256 courts per pair. The main report uses the closest court after that last step.

For Am2 frame 28019, the closest court after geometry/player checks changes from 61.620681 to 7.529112 px. After the shortlist limit, it changes from 81.262555 to 7.529112 px. For Am3 B pair 43, the corresponding values are 4.270800 to 23.248379 px after checks, and 4.270903 to 23.248379 px after the limit.

The frozen [axis_matching/run_given.py](https://github.com/ahalp90/badminton_cv_annotator/blob/b36402f1c994f2b000044589cdb6b001f190d0dd/scratch/court_det_fix/frozen_helpers_20260914/axis_matching/run_given.py) supplies `finite_scores` and `canonicalise`. The latter changes the court's labels by its 180-degree symmetry, rather than trying arbitrary corner permutations.

The new selection rule is imported as `retention_probe_reference.select_score_plus_strata`. W4 read its caller and the results, but did not recover that helper's own implementation or rerun the job. The saved comparison with an older run is unavailable for the Am2 and broadcast pairs. These are limits on the checks, not reasons to invent a new rerun requirement.

<a id="l2"></a>
## L2: new courts versus new scoring inputs

The key files are [run_l2_scoring.py](https://github.com/ahalp90/badminton_cv_annotator/blob/b36402f1c994f2b000044589cdb6b001f190d0dd/scratch/court_det_fix/next_steps_20260916/L2_scoring/run_l2_scoring.py), [comparison.csv](https://github.com/ahalp90/badminton_cv_annotator/blob/b36402f1c994f2b000044589cdb6b001f190d0dd/scratch/court_det_fix/next_steps_20260916/L2_scoring/comparison.csv) and [result.md](https://github.com/ahalp90/badminton_cv_annotator/blob/b36402f1c994f2b000044589cdb6b001f190d0dd/scratch/court_det_fix/next_steps_20260916/L2_scoring/result.md).

W4 read all 24 table rows. The code to check is `add_experiment_paths`, `reconstruct_generation_entries`, `load_generation_population`, `score_population`, `winner_entry`, `run_case` and `winner_delta`.

**Which helper was used?** L2 puts `next_steps_20260916/webui_seed/source` ahead of the frozen helper folders. Its `run_automatic` import therefore uses the seed copy. The stripe scorer comes from `experiments/annotator/independent_court/stripe_observations.py` at the reviewed revision.

`score_population` changes the fragment-based stripe score. It keeps the candidate geometry, eligibility fields, paint profile and order. The separate caches keep scores from one set of observations from being reused for the other. Where the original generation set has to be reconstructed, the script uses the saved per-pair shortlists and the existing overall selection rule.

The original set and the filtered-generation set each contain 256 entries. Their combined set keeps the original entries first, then the filtered entries. It is not a new set of 512 demonstrably different court shapes.

The checks against saved runs compare the winning IDs, not every score. `winner_delta` also compares IDs, so it counts a change of ID even when two entries might describe the same court. The report therefore uses actual corner distances and does not count changed IDs as successes.

<a id="l3"></a>
## L3: choosing across seven frames

The key files are [run_l3_temporal.py](https://github.com/ahalp90/badminton_cv_annotator/blob/b36402f1c994f2b000044589cdb6b001f190d0dd/scratch/court_det_fix/next_steps_20260916/L3_temporal/run_l3_temporal.py), [W3_PACKET.md](https://github.com/ahalp90/badminton_cv_annotator/blob/b36402f1c994f2b000044589cdb6b001f190d0dd/scratch/court_det_fix/next_steps_20260916/L3_temporal/W3_PACKET.md), [score_matrix.csv](https://github.com/ahalp90/badminton_cv_annotator/blob/b36402f1c994f2b000044589cdb6b001f190d0dd/scratch/court_det_fix/next_steps_20260916/L3_temporal/score_matrix.csv) and [result.md](https://github.com/ahalp90/badminton_cv_annotator/blob/b36402f1c994f2b000044589cdb6b001f190d0dd/scratch/court_det_fix/next_steps_20260916/L3_temporal/result.md).

W4 read all 210 CSV rows. It read the script in ranges 1–430 and 600–845, including how the courts are chosen, which entries are eligible, how another frame's observations are used, and when reference errors are added.

Unlike L2, L3 imports the frozen `automatic_axes/run_automatic.py`. Most scores on a candidate's own source frame come from the saved record. One candidate from each origin is rescored to check that those numbers repeat. Other-frame scoring uses that frame's image, detected fragments and feet.

The 30 entries have different source IDs. W4 did not remove entries that might describe the same geometry. Every method in this comparison uses the same entries and the same ordering rules.

### Camera/view evidence

The saved registration is in [view_alignment.json](https://github.com/ahalp90/badminton_cv_annotator/blob/b36402f1c994f2b000044589cdb6b001f190d0dd/scratch/court_det_fix/next_steps_20260916/L3_temporal/view_alignment.json). W4 read its settings and the detailed frame-5 entry. That entry reports 2,368 of 2,414 fitted matches agreeing with the transform, 604 held-out matches, coverage of all 12 image-grid cells, and a held-out median residual of 0.10209 native pixels.

The wider six-pair range comes from the producer's result report. W4 did not rerun image alignment or view the alignment images. This evidence concerns the sampled instants, not continuous camera stability between them.

### The added calculation

`W3_PACKET.md` already specifies the rank-sum rule. W4 applied it without choosing new frames, adding candidates or changing cue weights. The copied score fields are in [inputs/l3_scores_compact.json](inputs/l3_scores_compact.json). The saved reference errors are in a [separate file](inputs/l3_recorded_reference_errors.json).

The code writes the selected court to `results/selection_lock.json` before reading the errors. The final comparison is in [results/l3_rank_sum.json](results/l3_rank_sum.json), with a smaller rank table in [results/l3_ranks.csv](results/l3_ranks.csv).

These are calculations on real saved scores, not scores computed again from images. The reference-error medians use the recorded errors; W4 did not reconstruct the corners. This rewrite reran the same calculation and checked that every numerical result and ranking stayed the same.

<a id="w2"></a>
## W2: what the earlier image review says

The source is [w2_pixel_review/REVIEW.md](https://github.com/ahalp90/badminton_cv_annotator/blob/b36402f1c994f2b000044589cdb6b001f190d0dd/scratch/court_det_fix/worklog/webui_further_followups_16092026/w2_pixel_review/REVIEW.md), particularly sections 3–6. Its README and handover were also available in the reviewed commit's patch.

That review says false Am2 `184:4123` follows a raised net band, while approved `30:33` includes support from a sock and a crossing sideline. It also records tests for two expected markings using the same source pixels. Four detailed intervals were returned; the earlier table covering all profiles is a separate calculation. The return did not contain scene19 pixels.

The manifest identifies an edited `build_pair_atlas.py`, but its exact executed source was not returned. Its recorded SHA-256 is:

```text
80c0e9c0b97135e08bef4d9f03da2b38b73ddc5b59526d3e2e44800e1a091252
```

That distinction matters when identifying the producer. It is not a reason to discard the returned images or repeat the job.

**W4 did not view those pixels or run that verifier.** The physical explanations remain that earlier review's observations. W4 uses the approved example to explain why the simple rejection rule is not justified. It does not adopt the earlier suggestion of manual approval as the answer to detector development.

## Which references the distances use

| Test | Reference used |
|---|---|
| C2, GX | Visually approved supplied-direction court 89. |
| C2, Am3 | The manual reference saved by the direction experiment. |
| L3 | The annotated reference for each of the seven frames. |

These references are not interchangeable. In particular, a GX distance in C2 need not equal the L3 distance for the same court. The metric throughout uses 960 × 540 working coordinates and permits the project's 180-degree relabelling.

## What was and was not done

In the original W4 review, the archive read failed and the review used small GitHub reads at explicit paths and a fixed commit. No attached seed was available then. For this rewrite, the previous pack was supplied and available locally.

The local Python calculation ran on copied real-data fields. It did not run a model, generate new courts, score pixels again, use a remote host, view images or change the repository. The optional mode for reading the original local CSV is included in the script, but it was not run against the original CSV here.
