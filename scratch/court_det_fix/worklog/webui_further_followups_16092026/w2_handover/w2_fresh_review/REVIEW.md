# W2 fresh-data review — coherence is informative, but not the missing identity rule

> **Historical review:** This report was written before the bounded pixel export ran.
> The current execution status and unpacked evidence are in `../pickup.md` and
> `w2_pair_atlas/`.

## Decision

**Do not promote the connected-run probe to a court scorer or veto.** The fresh data
show a strong distinction for scene19, but the approved Am2 contrary example defeats
the straightforward replacement of the existing paint score. In particular, the
false Am2 candidate has **stronger connected support on its suspect pair** than the
approved candidate has on its crowded far pair.

One bounded follow-up is justified: **finish the paired pixel-ownership audit**, using
a single scripted export of two candidates and four intervals. Do not rerun the
matcher, expand to full pools, fit weights, or launch a learned semantic model on
the strength of this table.

The calculation is in `analyze_fresh.py`; exact results are in
`results/fresh_analysis.json`. The job is ready in `run_followup.sh`, with its complete
scope in `LUNA_EXECUTOR.md`. No original pixels or remote job were accessed in this
review. [F1, P1]

## 1. What was actually checked

I read the supplied `real_interval_probe.csv` and the exact `w2_probe.py` from the
previous handover. I independently checked all row identities, interval-to-marking
mappings, missing states, counts, fractions, span calculations and run bounds. The
CSV contains **48 rows: 41 measured intervals and seven geometrically unavailable
intervals**. Forty measured intervals pass the legacy test; one fails it. The one
measured failure is the usable scene19 candidate's far long service. [F1]

The input CSV has SHA-256:

```
41fd27c0387d5dc9864bde9cd0031a04b8669c37698fca7e5173aaaa15644e61
```

The supplied probe has SHA-256:

```
81e454e5e7876bbae4f7347c0feead3895ae764230edf4b64e73cfe932439d48
```

The unmodified probe refuses a witness file whose Git blob differs from
`739af3d1dacc7f3fedb551eab01a2b4bad94fd06`. It also checks the saved replay flags,
sampling settings, passing-offset predicates and panel identities. Your reported
execution is therefore a fresh calculation on the pinned L2 evidence, not a new
extraction, new candidate pool, or new video observation. The uploaded CSV alone
cannot attest that the local script was unmodified, and it does not contain its
producing commit, dirty state, dependency versions or execution log. I record those
as **not supplied**, rather than inventing a provenance match. [P1]

I re-resolved `fix/court-det`; it remains
`bb6787adeca9a19b5f1d3c3324d54cfcb2ea6e68`, dated 17 September 2026 at 02:47:11 UTC.
That is the repository/packaging scope, not a substitute for your local producing
identity. The old `d0c9a12…` to `872ea806…` comparison remains historical context;
it was not re-audited here. [R1]

I have not reopened the original witness JSON or viewed its source pixels. The new
results below are **real-data calculations on your derived table**. The six image
unit tests in this pack use constructed images and are labelled separately. There
were no repository writes, detector runs, model fits, or remote executions.

## 2. The direct contrary-example test fails

The probe's longest run connects adjacent longitudinal stations while allowing at
most a two-working-pixel change in normal offset per station. It is a connected path
in the saved 24-by-5 sampling lattice, not a proof of physical paint continuity. [P1]

Test a precise, unfitted replacement: keep the existing 40% interval requirement,
but apply it to the longest connected run rather than to the number of stations
having any passing offset. Keep the original aggregation: average available
intervals within each marking, then average available markings. The centre's two
painted intervals still share one marking identity. [R3]

For interval `i`, marking `m`, available-interval set `A_m`, and longest run `L_i`:

```
b_i = int(L_i / 24 >= 0.4)  # equivalently L_i >= 10
P_run = mean(mean(b_i for i in A_m) for each available marking m)
```

This is a **counterfactual calculation**, not a recommended new threshold or a
rerun of winner selection.

| Original candidate | Existing ruling | Legacy passing intervals | Intervals with run >=10 | Counterfactual P_run |
| --- | --- | ---: | ---: | ---: |
| Am2-28019 `184:4123` | False | 12/12 | 6/12 | **0.500000** |
| Am2-150 `30:33` | Approved | 12/12 | 6/12 | **0.454545** |
| SS03-19 `165:6702` | False | 5/5 available | 0/5 available | 0 |
| SS03-19 `1:60` | Usable | 11/12 | 11/12 | 0.909091 |

The false Am2 candidate comes out **above the approved one**. Equal interval counts
do not imply equal marking scores. The false candidate contributes five whole
passing markings plus one passing centre half: `(5 + 0.5)/11 = 0.5`. The approved
candidate contributes five passing markings, including the centre with both halves
passing: `5/11`. These calculations preserve the existing identity weighting. [F1]

Using the probe's possible upper runs gives the **same four counterfactual scores**.
Unknown offset tests do not rescue this replacement. A veto requiring every marking
to retain a sufficiently long run would also reject both approved Am2 and usable
scene19.

This result narrows the conclusion: it falsifies this natural drop-in use of the
probe. It does not prove that every possible use of spatial support is futile.

## 3. The targeted Am2 pair is the most important new result

| Candidate and nominal marking | Interval | Passing stations | Longest run, lower–upper | Run endpoint span, working px |
| --- | ---: | ---: | ---: | ---: |
| False `184:4123`: near long service | 10 | 18/24 | **16–16** | 82.850 |
| False `184:4123`: near baseline | 11 | 18/24 | **12–12** | 54.208 |
| Approved `30:33`: far baseline | 6 | 13/24 | **4–4** | 28.115 |
| Approved `30:33`: far long service | 7 | 10/24 | **3–3** | 19.497 |

**All four intervals have all 120 offset tests measured.** Their lower and upper
runs coincide, so this ordering is not a missing-evidence artefact. The reported
endpoint spans are distances between sampled path endpoints; they do not assert
unbroken paint between them. [F1]

The false court does not merely accumulate isolated positives everywhere. Its
suspect near pair retains moderately long paths. Conversely, the approved court's
crowded far pair has very short paths. Therefore, “the wrong pair will fragment;
the right pair will remain coherent” is not supported by this probe.

What these rows **do not** establish is that the false pair uses one physical ridge,
or that the approved pair uses two. A path count contains neither ridge ownership
nor the locations of its support. Short runs on the approved pair could reflect
failed stations, jumps between passing offsets, weak/crowded paint, occlusion,
alignment or sampling effects. The CSV does not distinguish those causes. It also
cannot identify positive pixels as net tape, court paint, a mat boundary, or another
structure. “Near the net top” is an existing visual description of a bad court,
not a per-sample physical attribution.

The earlier alias calculation remains a useful **target**, not the answer: W reports
nominal near-baseline/long-service separation of 0.378426–0.452247 working pixels for
false `184:4123`, and 1.861547–2.243938 for approved `30:33`'s far pair. Those numbers
were not recalculated here and do not prove which ridges passed. The approved
candidate's earlier exclusive-support loss also remains a contrary example to an
every-marking exclusivity veto. [R2]

## 4. Scene19 has a real signal, even on the same available markings

Compare only the **same five nominal markings** in both scene19 candidates:
left doubles, left singles, right singles, right doubles and far short service
(intervals 0, 1, 4, 5, 8). Do not give the usable court credit for its seven additional
available intervals, and do not punish the false court for missing ones.

| Quantity on those five markings only | False `165:6702` | Usable `1:60` |
| --- | --- | --- |
| Passing stations by marking | 13, 15, 13, 13, 12 | 24, 24, 24, 24, 22 |
| Longest runs by marking | 4, 9, 5, 5, 5 | 24, 24, 24, 24, 14 |
| Mean passing-station fraction | **0.550000** | **0.983333** |
| Mean connected-run fraction | **0.233333** | **0.916667** |

The run bounds are equal for each of these rows. This difference is not a reward
for completeness and does not rely on inventing failed observations outside the
image. The usable candidate also dominates these five markings in the simpler
passing-station counts, before adding any connectivity rule. The old binary
interval pass discarded a substantial difference in support strength. [F1]

That is useful development evidence, but not a universal classifier. The Am2
contrary example prevents transferring the conclusion unchanged. Scene19's separate
extent ambiguity also remains: four longitudinal lines plus one transverse line
can leave court length underconstrained when actual terminations are not observed.
A high score on that subset would not supply the missing constraint. A genuine
cropped court can have the same observability limitation. [R2]

## 5. Continuous summaries help describe the result, not calibrate it

Here are marking-balanced means of the hit fraction and the connected-run fraction,
using only available intervals. Centre halves are averaged within one marking.

| Original candidate | Mean passing-station fraction | Mean run fraction, lower–upper |
| --- | ---: | ---: |
| Am2-28019 false `184:4123` | 0.683712 | **0.420455–0.422348** |
| Am2-150 approved `30:33` | 0.691288 | **0.486742–0.494318** |
| SS03-19 false `165:6702` | 0.550000 | **0.233333–0.233333** |
| SS03-19 usable `1:60` | 0.903409 | **0.810606–0.810606** |

The continuous run mean orders these four candidates more favourably than the
counterfactual binary score. For Am2, its lower-bound gap is about **0.0663**; the
same-five-marking scene19 gap is about **0.6833**. These are descriptive differences,
not margins with known operating meaning. The displayed bounds cover unknown
sample tests, not statistical confidence or detector uncertainty. Choosing a cutoff
between the two Am2 values now would be fitting to this very comparison. [F1]

Several useful cautions are measurable rather than hypothetical:

* There are **44 unknown offset tests** across 11 measured intervals. Only three
  intervals have unequal run bounds, and each difference is one station. Seven
  entirely unavailable intervals remain null. Unknowns are not a general
  explanation for the observed short runs.
* Station spacing ranges from **2.355 to 41.697 working pixels**. A fixed permitted
  offset step per station is not a fixed geometric-direction tolerance. For
  scene19's four false sidelines the spacing is 41.697 px; for the usable sidelines
  it is about 12.6–13.0 px. The same two-pixel offset step corresponds to about
  2.75 degrees in the former and 8.8–9.0 degrees in the latter. The lattice rule is
  therefore stricter per unit length on the long false spans. That does not make
  the false court correct; it limits how physically one should interpret the run.

The table cannot say whether a run ends because a station fails, the allowed offset
transition fails, or another local condition intervenes. Repeated averages of this
CSV will not recover that discarded arrangement.

## 6. My assessment of the approaches and the project

**The diagnostic work is making progress; a replacement acceptance rule is not yet
demonstrated.** That is an important distinction, not a reason to discard the work.
The original all-pass counterexample, the L2 fixed-population comparison and this
fresh contrary-example test each answer a different question. Together they stop
three plausible but unsupported stories: complete paint passes prove identity;
filtering helps only by recovering capped candidates; and requiring coherent runs
will simply fix the false paint winners. [R2, R7, F1]

I would retain the current geometry, line scores and paint scores as proposal and
diagnostic machinery. I would not spend the next iteration on a weighted blend of
coverage, run length, exclusivity, rank and floor outcome. The existing few labels
could support many attractive fitted formulas; that would not resolve the observed
correspondence ambiguity. Nor would I require a learned semantic system before
looking at the pixels already available locally.

The earlier L2 separation of population effects from scoring-input effects was the
right experimental design. Its mixed outcomes are not evidence that filtering is
pointless; they are a reason to keep those scopes separate. No formula is truly
“unchanged evidence” when the fragment population and its normalised weights change.
The fresh CSV does not re-evaluate those full pools or establish a new winner. [R7]

For a resource-constrained project, I would keep a manual-review boundary around
these proposals until the ownership question has an actual positive-versus-negative
witness. The extra compute permission is welcome: it makes a straightforward batch
preferable to elaborate caching and repeated interactive reviews. It does not make
a larger candidate sweep the best next scientific question. The immediate bottleneck
is the interpretation of existing positive evidence, not a lack of generated numbers.

## 7. ONE bounded follow-up: finish the two-pair pixel ownership witness

**Single changed component: the diagnostic export.** There is no new score or
acceptance threshold. The script places both nominal markings and their actual
saved passing tests in the same image-coordinate view, preserving the unannotated
pixels beside the overlay. It also exports a shared grayscale strip covering their
finite overlap. Unlike comparing station 0 with station 0 on unequal spans, this
uses one image coordinate chart for both intervals.

The panel is exactly false Am2-28019 `184:4123`, intervals 10/11, and approved
Am2-150 `30:33`, intervals 6/7. It is the smallest panel addressing the specific
all-pass alias with an approved crowded contrary example. There is no extension to
the ten-candidate guardrail panel, nine development views or full pools in this job.

`build_pair_atlas.py` reads the pinned witness JSON and the two pinned native frames,
checks the uploaded CSV against all 48 saved trace rows, and replays the original
centre/flank sampling only for the four target intervals. It records actual input
and script hashes, HEAD/dirty identity and environment versions. It writes one small
return bundle, with no repository changes. The current branch and immutable frame
blobs were checked through the GitHub connector. [R1, R8]

The output adds information the CSV lacks: where support lies along each finite
interval, how both candidates' nearby marking tests sit on the same image features,
and enough native/context pixels to distinguish court paint from a competing
structure where the image permits it. Interpolated strips are explicitly display
sampling, not extra resolution. Unknown regions retain masks; the script does not
infer semantic absence from dark pixels or a failed contrast test.

Readout is limited to two pairs. The key possibilities are:

- A visible shared ridge in the false pair, with separately identifiable paint in
  the approved pair, would support investigating joint correspondence on this
  mechanism. It would not establish a general rejection threshold.
- Shared or unresolved ridges in **both** pairs would defeat ridge multiplicity as
  a sufficient discriminator at this resolution.
- Distinct but wrong scene ridges supporting the false pair would shift the problem
  from evidence reuse to surface/object association. The diagnostic itself would
  still not decide acceptance.

These are readout alternatives within one job, not three proposed follow-ups. A
result of unresolved ownership is legitimate. Do not start a second experiment from
inside the executor task.

### Historical one-command execution plan

> This section records the pre-export plan. The export is complete. Use `../pickup.md`
> and `w2_pair_atlas/` for the current packet.

Use the existing approved host, launcher, execution policy and NumPy/OpenCV
environment. L2's state records Carmack; this handover does not invent or change
connection details or host permissions. From the repository checkout:

```bash
bash /absolute/path/to/w2_fresh_review/run_followup.sh
```

At the time of this review, the wrapper chose a new output directory outside the checkout.
Explicit `REPO`,
`PYTHON` and `OUT` overrides are documented in `LUNA_EXECUTOR.md`. It runs one CPU
process; no model install, GPU job, full-pool replay or new extraction is needed.
The pre-export plan requested `return_pack.zip`; the committed packet now keeps the
unpacked output in `w2_pair_atlas/`. Semantic image judgement is not required of the
executor. One image-capable review of the resulting two sections is the intended
human-attention cost.

The code compiled and **all ten tests passed here**: four on the real CSV and six
on constructed images, including one-ridge/two-ridge views, shared-coordinate
alignment, unavailable pixels, native scaling and deliberate replay mismatch.
At the time of this review, the original two-pair pixel job **had not run**. These tests
exercise the export logic; they were not a successful replay on the source frames. See
`results/execution_status.json`.

## Remaining uncertainty

We now know that the proposed connected-run replacement fails the approved Am2
contrary example, and that scene19 has a strong difference even on its common
marking subset. What remains unknown is **which physical structures supplied the
Am2 passing tests and why the approved far pair has weak lattice connectivity**.
The CSV does not answer that. The next bundle asks for the exact two-pair evidence
that can, without requesting a new dataset or repeating a full experiment.

## Source and execution references

`S = scratch/court_det_fix`, `N = S/next_steps_20260916`. Repository sources below
are pinned to `bb6787adeca9a19b5f1d3c3324d54cfcb2ea6e68` unless explicitly labelled
as an earlier producing scope. Full paths and hashes are also in
`source_manifest.json` and the labelled prior `inputs/prior_source_index.json`.

- **F1:** User upload `real_interval_probe.csv`, byte-identical copy in `inputs/`;
  fresh SHA-256 above. Calculations actually executed in `analyze_fresh.py` and
  `results/fresh_analysis.json`.
- **P1:** Previous handover `w2_probe.py`, SHA-256 above; `longest_path`,
  `interval_probe`, `witness_calculation`. Read fully in this review.
- **R1:** Current branch resolution through GitHub's commits collection,
  `sha=fix/court-det`, one result; current full SHA and timestamp above.
- **R2:** `S/worklog/webui_evaluation_returns_15092026/can_available_evidence_separate_court_from_false_match.md`;
  Git blob `9a8a2e22df019f4414242fa6005aeb532c022ac8`. Earlier source/audit read;
  appended geometric measurements not recalculated in this review.
- **R3:** `S/frozen_helpers_20260914/axis_matching/inspect_appearance.py`,
  `profiles` lines 21–38; blob `4a65a9281e89be4a1ac2fbf77585b457c25500dc`.
  Original interval/marking aggregation read in the preceding source review.
- **R7:** `N/webui_seed/L2_scoring/comparison.csv`, blob
  `7b1716becb2fae350a41935600fcd343efdb20af`; included unchanged for provenance.
  `N/L2_scoring/run_l2_scoring.py`, blob
  `1e5c948ff813306421819d9bcfd9d07eff0cd99d`, especially `score_population`,
  `winner_entry`, `profile_trace`, `run_case`.
- **R8:** `N/webui_seed/frames` Git tree
  `7f6f982e6baa9447e8c1049c81aee21780dbdc92`; frame metadata read in this review.
  `amateur/am2/frame_00028019.png`: blob
  `a6c1ace2fad3a6f350b7b91fad991840955fd866`;
  `amateur/am2/frame_00000150.png`: blob
  `08ab1d43d219320bc18d2bfb25fbde5f5f3cf456`. Pixels not viewed.
- **R9:** `N/L2_scoring/STATE.md`, blob
  `b6c26165700f01756e8169b6b6909d4d895b9449`; read in this review. It records L2's
  Carmack execution and the distinction between runtime replay and local mismatch.
