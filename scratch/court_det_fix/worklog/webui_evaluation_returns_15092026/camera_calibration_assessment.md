# Camera calibration assessment

## Revision and evidence inspected

All reads were pinned to **`d0c9a12fb62eea64e1e4cec523d6f29a42e37a2a`** in `ahalp90/badminton_cv_annotator`, with `fix/court-det` as the requested branch context—not an assumed current branch head. Below, `R` denotes your `projective_patterns/` evidence root.

| Inspection level                         | Evidence actually inspected                                                                                                                                                                                                                                        |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Text read                                | `evaluation/temporal_assessment.md`, `evaluation/README.md`, `automatic_axes_results.md`, `evaluation/temporal_view.html`, `automatic_axes_visual_judgements.md`, relevant `evaluation/method_excerpts.md` sections, and `../README.md`.                           |
| Completely decoded                       | `../summary.json.gz`; archive entry `../replay.zip!joint_short.py`, with its ZIP-entry CRC verified.                                                                                                                                                               |
| **Partially decoded only**               | Prefixes of `evaluation/temporal_records.json.gz` and `evaluation/ranking_records.json.gz`; the beginning of archive `joint_short/results.json.gz`, including complete initial Yellow floor-only candidate objects.                                                |
| **Not inspected visually or completely** | The gallery’s linked images did not become viewable. The complete 13-frame numerical records, complete ranking records and remaining archive contents were not recovered. No detector execution, registration, temporal scoring or timing benchmark was performed. |

Consequently, modern case findings below are **documented repository results**, not a fresh inspection of all 13 frames. The specific intermittent-occlusion and cross-frame line/direction comparisons remain unverified here. The handover manifest records that access boundary.

## Recommendation

**Yes—calibration per automatically verified stable view is a credible nearer-term architecture to test, but this evidence does not demonstrate that it is the faster successful route.** It reduces how often correct calibration is needed; it does not remove the need to generate, select and accept a correct court automatically.

The smallest defensible later-audit comparison is **independent versus cross-frame scoring of the same frozen automatic candidate union**, followed by guarded reuse. Start there rather than changing pooled-line generation or refinement simultaneously. This prioritizes an interpretable test, **not an empirically superior algorithm**. Current evidence cannot rank the options by eventual accuracy or development time. Leave the independent direction experiments unchanged.

**Current scope correction (2026-09-16).** The seven cached GX direction checks are descriptive only. They did not test the proposed calibration frames 30, 60 and 90, the modern score matrix, or independent versus shared scoring on the same automatic candidate union within a verified view. They therefore do not show that neighbouring frames cannot help or that those proposed frames are bad choices. The temporal audit remains deferred, not disproved. A smaller cached-frame pilot may be considered separately; a full stability and reuse framework is not required for this correction.

## 1. Missing observations are repairable in principle; the diagnosed failures are not all missing-observation failures

**Documented observations.** These cases put distinct constraints on the temporal hypothesis:

| Exact case and population                                                         | Consequential evidence                                                                                                                                                                                               |
| --------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `gxBQ_window_00_frame_0`, `automatic_all_camera`                                  | `22:4579` and `22:4588` are rejected for shear. Discarded observed directions `1183/122` produce approved control winners, but their selection was label-guided.                                                     |
| `gxBQ_window_00_frame_5`, `automatic_all_camera`                                  | `181:29836` and `10:1274` are rejected as wall/seat-area courts. The nearest camera-eligible saved candidate remains **89.94 working pixels** from its control: global retention alone does not explain the failure. |
| `am2_window_00_frame_150` and `am2_window_01_frame_28019`, `automatic_all_camera` | Line winners `30:30` and `16:1800` confuse mat borders with court boundaries. Paint winner `30:33` is approved at frame150, whereas frame28019’s `184:4123` is rejected **despite passing all 11 paint profiles**.   |
| `am3_window_00_frame_0`, `automatic_all_camera`                                   | Line winner `43:22603` is already essentially perfect; paint winner `43:22627` is very usable. This is a mandatory regression case.                                                                                  |

These are the recorded judgements and diagnostics; suspected post/person correspondences and distortion remain hypotheses, not established causes.

**Deduction.** Pooling can help when moving players uncover different painted intervals, extraction misses vary, or new observations improve direction constraints. It cannot recover paint always outside a fixed image, turn repeated background structure into correct line identity, or guarantee recovery of directions discarded by `vp_pruning.retain_pencils`. More similar observations can reinforce the same misleading support. Fixed-candidate rescoring cannot create geometry absent from the entire candidate union.

**Hypothesis and falsifier.** Another automatically processed frame might supply useful geometry missing from GX5, or better distinguish paint from the Amateur-2 mat boundary. The available inspection does **not** establish a particular intermittently missing stripe. The hypothesis fails for the frozen comparison if no useful automatic geometry enters its union, or persistent false geometry still wins. Improvement on GX accompanied by degradation of the already-good Amateur-3 case is not an unqualified advance.

Missing modern outputs on the other eight supplement frames are **unmeasured**, not failures; historical candidates cannot be substituted into a modern automatic pool.

## 2. The older shared-court experiment supplies caution, not the missing temporal control

**Direct code observation.** `replay.zip!joint_short.py::main` unions retained per-frame proposals, excludes `_median` proposals, suppresses near-duplicates and uses `choose_pairs(record)['central']`. **Every arm selects one court across three frames.** `floor_only` uses median floor scores; `initial` adds median net evidence at 3:1 floor-to-net weighting; `refined` additionally changes corners through `refine(corners, pooled, size)`. References enter evaluation metrics, not the selection score. There is no independent-frame selection arm.

**Direct saved-record observation.** In the decoded results prefix:

```text
records[0].window                 = "yellow_short"
records[0].stage                  = "floor_only"
records[0].candidates[0].source_image = "yellow_short_frame_90"
records[0].candidates[0].source_rank  = 0
records[0].candidates[0].floor_scores =
    [0.7777777778, 0.8541666667, 0.7847222222]
```

That complete candidate object has worst corner error **221.1893 pixels at 1280×720** across reference keys `14`, `90`, `156`, matching the fully decoded summary. Thus substantial repeated support already coexists with wrong geometry in the older pipeline. This does not establish its exact failure mechanism or camera stability.

**Contrary evidence matters.** Yellow improves from `221.19 → 26.01 → 21.22`, but Letterboxed worsens from `10.06 → 17.38`, then reaches `14.75`; Centre changes `4.04 → 4.04 → 3.20`. These are mixed effects of added cues and changed geometry—not measured benefits of aggregation over independent selection. The historical 15-pixel cutoff is not a current acceptance criterion. Nor are these floor/net scores comparable with modern stripe/profile scores.

**Deduction.** The operations have different capabilities:

* **Pooled-line generation** can create a new geometry, but changes observation merging, direction selection and search.
* **Shared scoring** chooses among fixed geometries and isolates whether observations elsewhere support the *same* candidate.
* **Reuse** simply retains the chosen geometry; it can preserve an error as efficiently as a correct court.

For a fixed pool \(U\), the relevant comparison is

$$
H^*=\arg\max_{H\in U}\operatorname{median}_t s_t(H),
$$

not choosing whichever candidate achieved the largest score anywhere in the video.

**Falsifiable implication.** If a useful candidate exists in \(U\) but loses on other independently same-view observations, this frozen shared scorer has not solved ranking. If none exists, the limitation is generation. Neither outcome justifies silently adding pooled generation or refinement to rescue the comparison.

## 3. Automatic view verification and court correctness require different evidence

**Recorded fact.** No shared projection is verified for the 13-frame supplement. GX0/5 are only about **0.084 nominal seconds** apart; later GX samples extend to roughly 24 minutes. Amateur-2/3 seconds are unavailable. Annotated renderings and matching sampled endpoints cannot exclude intervening cuts, pan/zoom, or departure and return.

**Proposed minimum automatic evidence.** Use pristine decoded pixels and spatially distributed static-feature tracks, automatically excluding people and moving outliers. Check both consecutive frames **and the original calibration anchor**, with held-out feature matches testing the estimated registration. A low-residual registration is insufficient: pan/zoom may register well while invalidating the old pixel court. The guard must bound motion relative to identity over adequately covered image regions.

Cuts, match collapse, detectable pan/zoom, changed crop/dimensions, insufficient coverage or uncertainty should invalidate reuse. Near-identical adjacent frames should not count as independent calibration support. Registration tolerances must be explicit and frozen before court outcomes are examined; they are a new engineering assumption, not an already-validated detector component.

**Circularity and contrary evidence.** Persistent walls and mat edges are useful evidence that the *view* is unchanged, but not that a particular structure is court paint. Using estimated-court agreement to form groups and then treating those groups as validation of that court is circular. `saved_reference` must never decide grouping, candidate selection, rank choice, rejection or restart.

**Falsifiable implication.** A cut or detectable pan/zoom must invalidate the cached court even when its court score stays high. A guard evaluated only at sampled instants supports only within-sample agreement; uninterrupted reuse requires checks throughout the claimed interval.

## One bounded experiment for the later audit

**Sampling.** Use four local windows from the existing development videos:

| Window, inclusive        | Existing evaluation anchors | Three prescribed calibration frames |
| ------------------------ | --------------------------- | ----------------------------------- |
| GX `0..90`               | `0`, `5`                    | `30`, `60`, `90`                    |
| Amateur-2 `150..240`     | `150`                       | `180`, `210`, `240`                 |
| Amateur-2 `28019..28109` | `28019`                     | `28049`, `28079`, `28109`           |
| Amateur-3 `0..90`        | `0`                         | `30`, `60`, `90`                    |

This bounds acquisition to **364 original frames** and **12 additional calibration-cache positions**. Offsets are frame indices, not inferred Amateur seconds. It is an **offline/backfill audit**, not a causal or zero-latency streaming result.

**Grouping.** Run the independent guard on every decoded frame. Windows are acquisition limits, not pre-certified views. Combine prescribed observations only when the guard assigns them to the anchor’s view. Splits, inadequate support or near-duplicate calibration samples make the window ineligible or uninformative—no manual correction, replacement frames or adaptive extension.

**Matched comparison.** Freeze original B direction selection, extraction, matcher budgets, coordinate conventions, camera/player checks and per-frame scoring. Obtain complete modern automatic candidate pools for the five anchors and 12 calibration positions; winner-only exports are insufficient. Exclude historical and label-guided candidates.

Within each automatically confirmed view, construct a deterministic union \(U\), then record:

| Output                       | Candidate access   | Selection evidence                                        |
| ---------------------------- | ------------------ | --------------------------------------------------------- |
| Native independent baseline  | Its own \(P_t\)    | Evaluation frame \(t\)                                    |
| Matched independent baseline | \(U\)              | Evaluation frame \(t\)                                    |
| Shared selection             | **The same \(U\)** | Median score over the three prescribed calibration frames |

The decisive contrast is the last two rows. The first two separately measure candidate-access effects. Predeclare the existing complete-line score as the primary diagnostic rule, with deterministic identity-based ties; retain paint components as diagnostics, **not a human-selectable alternative output**. Apply identical eligibility in matched arms and log missing scores rather than averaging over a favorable subset. This audit does not assert that line ranking is globally best.

**Separate outcomes.** Report candidate coverage and origin IDs (**generation**), locked winner and complete cross-frame score vector (**ranking**), unchanged acceptance decisions and abstentions (**acceptance**), and reset/state trace (**reuse**). References evaluate only after outputs are locked. Do not turn diagnostic camera-eligible winners into accepted detections: `run_automatic.winner_ids` is not an acceptance policy. If an existing CourtKeyNet-free acceptance function cannot be identified and replayed, report acceptance as **not established**, without inventing a new threshold. The old floor gate alone is not a solution: it rejects both extreme false winners and approved GX0 controls.

Include three fixed guard sanity checks within this audit—a cut, translation and zoom applied temporarily to a copy of the GX sequence, then removed. These test invalidation mechanics, not natural unseen intervals; exact perturbations are specified in the handover.

**Stop** on baseline replay mismatch, unavailable required inputs, failed automatic grouping, absent useful automatic geometry, persistent wrong shared winners, or Amateur-3 regression. Report acceptance and guard failures even if ranking improves. Do not add net scoring, refinement, pooled generation or tuned thresholds to rescue this run. Final hard holdouts remain reserved.

## Minimum missing evidence and deciding observation

The packet already supplies development references/judgements, cached line/direction evidence, settings and identities, five modern winner sets, and the older shared-court implementation. It does **not** supply the modern controlled result proposed above; my incomplete compressed-record/image access further limits this assessment.

The minimum missing evidence is **a fixed-geometry cross-frame score matrix coupled to an independent automatic view-state trace**, including Amateur-3 and unchanged acceptance outcomes. Producing it requires pristine frames for the bounded windows, the 12 additional observation caches and full matched candidate pools—not new annotations.

**The observation that would change the judgement:** a reference-blind shared selection receives support on genuinely separated, independently verified same-view observations, improves difficult-anchor results over the *same-pool* independent comparator, and preserves Amateur-3. That would support investment in this simpler route. Until then, its practicality is a credible hypothesis—not an empirical advantage or a completed automatic detector.

### Handover

[Markdown assessment](sandbox:/mnt/data/calibration_per_view_assessment.md) · [Handover bundle: assessment, bounded protocol, evidence manifest and decoded saved-candidate excerpt](sandbox:/mnt/data/calibration_per_view_handover.zip)


## Revision and evidence actually inspected

All reads remained pinned to **`d0c9a12fb62eea64e1e4cec523d6f29a42e37a2a`** in `ahalp90/badminton_cv_annotator`, with `fix/court-det` as the branch context. The new work concerns the older committed `player_guided/replay.zip`, not newly generated modern results.

| Newly recovered evidence            | Inspection and verification                                                                                                                                                                                                             |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `joint_short.py`                    | Complete source; ZIP-entry checksum verified. Inspected, not executed.                                                                                                                                                                  |
| `people_short/yellow_short.json.gz` | Complete record; ZIP and gzip checksums verified. Establishes anchor order **14, 90, 156** and dimensions **1280 × 720**.                                                                                                               |
| `joint_short/results.json.gz`       | Recovered the **complete first record: `yellow_short / floor_only`, containing all 33 saved candidates**. Also recovered 16 complete candidate objects from the following `initial` record and cross-checked their duplicated evidence. |

The entire results gzip remains incompletely recovered, so its whole-file checksum is **not verified**. The full modern 13-frame records and repository images remain uninspected. These access boundaries, including the verified checksums, are recorded in the [recovery manifest](sandbox:/mnt/data/court_extension/results/recovery_checks.json).

**Historical extension scope.** This yielded a meaningful conditional extension within the saved older Yellow population: a matched independent-versus-shared selection comparison, a stronger mathematical diagnosis of the floor scorer, and a candidate-origin sensitivity check. I ran these calculations locally from the saved numbers; no detector inference, new annotations, manual candidate selection, or acceptance-threshold tuning was involved.

**Recommendation:** the older Yellow comparison gives a conditional reason to run the later audit, rather than only an architectural reason. Its strongest negative finding is limited to the saved Yellow floor-score matrix: **changing the temporal aggregation rule alone cannot rescue the closer Yellow candidate from that matrix**. It does not establish modern Amateur-2 line- or paint-score dominance, and it does not rule out useful new observations.

## 1. Holding the candidates and cues fixed, shared scoring does improve selection in the older Yellow case

**New computation.** The complete 33-candidate record contains each fixed geometry’s floor and net scores on all three anchors. That permits a comparison missing from the original headline table.

I compared independent selection using one observation against shared selection using all three, separately within each historical cue definition. For the net-inclusive arm, I preserved the source expression exactly:

$$
S(H)=\frac{3\,\operatorname{median}_t f_t(H)
+\operatorname{median}_t n_t(H)}{4}.
$$

This is the historical **combination of componentwise medians**, not a substituted median of per-frame composite scores. Geometry, candidate access and the 3:1 weighting remain fixed.

The decisive candidates, all within this **older Yellow population**, are:

| Label | Exact origin                                             | Saved floor-record index   |
| ----- | -------------------------------------------------------- | -------------------------- |
| **A** | `source_image="yellow_short_frame_90"`, `source_rank=0`  | `records[0].candidates[0]` |
| **B** | `source_image="yellow_short_frame_90"`, `source_rank=2`  | `records[0].candidates[4]` |
| **C** | `source_image="yellow_short_frame_156"`, `source_rank=0` | `records[0].candidates[1]` |

References were joined **after selection**. The resulting maximum-corner errors use the existing 1280 × 720 evaluation coordinates:

| Frozen scoring cues            | Independent winners at frames 14 / 90 / 156 | Shared winner | Worst independent error | Worst shared error |
| ------------------------------ | ------------------------------------------- | ------------- | ----------------------: | -----------------: |
| Floor only                     | A / A / C                                   | A             |         **1,271.00 px** |      **221.19 px** |
| Floor plus net, historical 3:1 | A / B / B                                   | B             |           **221.19 px** |       **26.01 px** |

These are newly computed selections from the recovered complete record, not a reinterpretation of the original three-arm headline table. Exact scores, margins and per-anchor errors are in the [computed results](sandbox:/mnt/data/court_extension/results/analysis.json).

Floor-only sharing prevents the particularly displaced frame-156 selection but still chooses A. With the net cue held fixed, sharing changes frame 14 to B; frames 90 and 156 already select B independently. Existing landmark errors also improve: worst landmark RMS changes from **206.06 to 61.82 pixels** in the floor comparison, and **59.40 to 4.60 pixels** in the net-inclusive comparison.

**Important limitation:** this is a **matched comparison within the saved surviving population**, not a native independent-frame detector baseline. The 33 candidates already survived the archive’s shared-player check, deduplication and nonnegative median-floor filter. Those earlier exclusions cannot be undone from this table.

**Contrary evidence:** B’s 26.01-pixel error is not a new visual approval or acceptance result. Adding net evidence previously worsened Letterboxed from 10.06 to 17.38 pixels. The new Yellow comparison therefore supports testing shared selection—not adopting net scoring universally.

**Falsifiable implication:** a modern temporal benefit should appear **within a fixed scorer and fixed candidate pool**. Comparing independent floor scoring against temporal net scoring plus refinement would still leave the cause unresolved.

## 2. The floor-score failure is stronger than “median did not work”

**New computation.** These are the saved floor-score vectors:

| Candidate | Frame 14 | Frame 90 | Frame 156 | Worst corner error |
| --------- | -------: | -------: | --------: | -----------------: |
| **A**     | 0.777778 | 0.854167 |  0.784722 |          221.19 px |
| **B**     | 0.663194 | 0.777778 |  0.760417 |           26.01 px |
| **C**     | 0.715278 | 0.775694 |  0.785417 |        1,271.00 px |

Across the complete population, **A scores strictly higher in every observation than 31 of its 32 competitors—including B**. Only C escapes, because its frame-156 score is slightly higher. Thus the floor-score *Pareto frontier*—the candidates not beaten component by component—contains only **A and C**. The [dominance certificates](sandbox:/mnt/data/court_extension/results/dominance_certificates.csv) record every dominated candidate and its three score margins.

Reference-only evaluation shows that B has both the lowest worst-corner error and the lowest worst-landmark RMS among these 33 candidates. That evaluation did not select B: B is also the unique automatic winner of the historical net-inclusive shared score.

**Mathematical deduction.** Because

$$
f_t(A)>f_t(B)\quad\text{for every saved observation }t,
$$

a positive-weight average, mean, minimum, maximum, median or quantile of these same floor observations cannot prefer B to A. More generally, a coordinatewise nondecreasing aggregation cannot strictly prefer a dominated score vector. A flat rule could create a tie, but resolving it would require another criterion.

**Consequently, replacing median with another ordinary temporal floor-score aggregation cannot recover B from this matrix.** This is a stronger diagnosis than observing that one particular aggregation failed.

**Boundary and contrary evidence:** this does not prove that additional frames cannot help. A new observation could reverse the ordering. Another cue already distinguishes the pair here: B’s net scores are approximately **0.873 / 0.873 / 0.859**, compared with A’s **0.550 / 0.530 / 0.545**. Nor does the numerical ordering establish whether the underlying floor confusion comes from occlusion, line identity, background structure or another mechanism. Those causal explanations remain unverified.

**Falsifiable implication:** once the modern cross-frame matrix exists, check whether useful alternatives are dominated. When they are, stop tuning the aggregation rule for that frozen matrix. The missing ingredient must be discriminating observations, different cues, better geometry, or another explicitly changed criterion—not merely a different average.

## 3. Score-only leave-one-out can hide dependence on the generating frame

**New computation.** Removing each anchor’s score in turn, while retaining all 33 candidates, the historical net-inclusive scorer selects B in **all three** two-observation comparisons.

That sounds reassuring, but B is credited to **frame 90**. Holding out frame 90’s score does not hold out the geometry generated from it.

I therefore performed a separate, bounded origin sensitivity: exclude frame 90’s score **and the 14 saved candidates credited to `yellow_short_frame_90`**. Nineteen candidates remain.

| Frame-90 diagnostic                          | Automatically selected origin       | Error at frame 90 |
| -------------------------------------------- | ----------------------------------- | ----------------: |
| Hold out its score only                      | `yellow_short_frame_90`, rank 2 — B |      **25.79 px** |
| Also exclude candidates credited to frame 90 | `yellow_short_frame_156`, rank 1    |     **288.30 px** |

The closest remaining geometry by **evaluation only** is `yellow_short_frame_156`, rank 4, at **63.05 pixels** on frame 90. Thus this sensitivity exposes both a loss of available geometry quality and a remaining ranking loss. That reference-nearest candidate was never substituted for the automatic output. Full details are in the [sensitivity results](sandbox:/mnt/data/court_extension/results/sensitivity_not_holdouts.csv).

**Limitation:** neither calculation is an independent hard holdout. Historical deduplication retains one credited origin and may suppress near-duplicates from others. The shared player selection and original survival filter also use the clip. Removing credited origins cannot reconstruct what independent generation would have produced. The handover labels these calculations **diagnostics, not holdouts**.

**Contrary interpretation worth preserving:** dependence on one good generating observation is not inherently a failure of the per-view target. That target deliberately needs good geometry only once. The result instead shows why held-out *score columns* must not be advertised as independent frame-level generalization.

**Falsifiable implication:** the later audit needs separate accounting for frames that **generated** a candidate, frames that **scored** it, and frames used only to **evaluate** it. Preserve multiple origins where possible, rather than treating a deduplicated source label as complete provenance.

## What this changes for the practical route

The original shared-court headline table remains confounded. **This extension adds a genuinely different, conditional comparison from its saved records:** sharing observations improves selection on Yellow, while a dominance certificate explains why floor-only aggregation still cannot select its closer geometry in that saved matrix. The result does not establish modern Amateur-2 line- or paint-score dominance, and the proposed modern audit remains unexecuted.

That is enough to keep the proposed **fixed-candidate scoring audit** open. It is not enough to rank pooled-line generation against shared scoring in the modern pipeline, verify camera stability, or claim an accepted CourtKeyNet-free detector.

The minimum consequential modern observation remains: **one automatically generated geometry receives discriminating support across independently established same-view observations, a fixed shared ranker improves on the same-pool independent comparator, and the already-good Amateur-3 case does not regress.** Amateur-3’s `43:22603` remains an explicit regression control, not a success newly established here.

No new annotations or expanded sweep are required by this extension. The independent direction experiments remain unchanged.

### Handover

[Markdown report](sandbox:/mnt/data/court_temporal_extension.md) · [Complete handover bundle](sandbox:/mnt/data/court_temporal_extension_handover.zip) · [Floor-score figure](sandbox:/mnt/data/court_extension/results/floor_dominance.png)

The bundle contains the complete recovered Yellow candidate record, standalone audit and recovery scripts, numerical outputs, provenance, and **15 passing regression tests**. It was unpacked and replayed in an isolated directory successfully. The tests include exact saved-score replay and unchanged selection when reference values are altered or references and corners are removed; they are software checks, not detector or camera-stability validation.
