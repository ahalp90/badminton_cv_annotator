# TrackNetV3 stride-8 shuttle hallucination guard: independent review

**NOTE: This is just an LLM WebUI assessment to a series of questions I had regarding possible oversights in our hallucination guard implementation, off-the-shelf alternatives, and whether I'd mucked up anything obvious in the RANSAC assessment of possible uncaught detector blips.<br><br>As of 02-08-2026 I have not had the time to fact-check it, nor sanity-check what's reasonable or necessary.<br><br>Committing because it might have something useful to follow-up.**

**Repository:** `ahalp90/badminton_cv_annotator`  
**Repository snapshot reviewed:** `781a589b13a2a3ec986c0562e8d5bf9f6614cef0`  
**Review date:** 2026-08-01  
**Production scope assumed:** TrackNet stride 8, InpaintNet sequence length 16, non-overlap inference  
**Repository changes made:** none

## Executive verdict

### 1. Could the current filtering approach be replaced by a standard scikit-learn outlier method, plus a tiny deterministic rule for the fixed loop?

**Not as a justified production replacement.** It could be built, but the available evidence does not support replacing the current mechanism-aware logic with a generic unsupervised detector.

A small deterministic rule should handle the known evidence-free 16-frame failure. For new tracks, the best rule is not a hard-coded centre-coordinate list. It is a source-aware rule based on an aligned InpaintNet window with no TrackNet pass-through, with a checkpoint/template fallback for legacy tracks that lack provenance.

A generic method can still be useful as a **secondary ranker**. The most credible candidates are a local motion-residual score and, if one scikit-learn baseline is wanted, Isolation Forest over one row per producer window or candidate span. They should not be the primary source-of-truth guard.

### 2. Would that be more maintainable and less dependent on guessed thresholds?

**A simpler deterministic guard would be more maintainable. A scikit-learn replacement would not automatically be so.**

The current guard contains several fixed policy choices despite its adaptive threshold: a 16-frame window, a 32-frame episode-merging gap, at least two distinct recurrence counts, a minimum of 30 episodes, a minimum 10x count gap, a 15-frame halo, global exact-coordinate matching, and a both-halves gate. [R6]

A model moves those decisions into feature scaling, the observation unit, the fitting population, `contamination`, `n_neighbors`, `nu`, `gamma`, covariance support, random seeds, and a score cutoff. The threshold has not disappeared. It has become harder to explain and version.

### 3. Is it likely to catch more genuinely invalid blips and chunks?

**It is likely to nominate more abrupt blips. It is not yet likely to reject more false positions safely.**

Local quadratic residuals are well matched to isolated coordinate jumps and some short discontinuities. The current audit already shows that such a method nominates many frames not caught by recurrence. It also nominates about 31–33% of all coordinate-valid frames, so nomination is not evidence of visual falsity. [R7]

Generic density or support estimators are poorly matched to frequent and smooth artefacts. Scikit-learn's own outlier-detection guide states that its available outlier estimators assume anomalies are in low-density regions and therefore cannot treat a dense anomaly cluster as anomalous. [E4] A common InpaintNet attractor can become exactly such a dense cluster.

For long or common Inpaint chunks, source support and aligned-window features are more promising than raw-frame anomaly scores. For non-Inpaint base-TrackNet blips, local motion residuals are the strongest current lead.

### 4. Has the investigation made any serious technical or statistical mistakes?

**The latest audit report is cautious and does not commit the worst interpretation errors. The current guard and some historical claims do contain serious technical problems.**

The most important are:

1. The live recurrence detector can fail open on a perfectly regular, highly repeated artefact.
2. Grade 3 combines two different claims and is rejected wholesale by default.
3. The live two-half check is not independent split-half discovery, although historical prose describes that stronger procedure.
4. The tracked repository does not contain the measurement record cited to justify rejecting every non-zero grade.
5. The normal production paths do not currently supply either recurrence grades or sidecar evidence to `run_video`.

The audit's RANSAC overlap, sidecar span coverage, event unions, and clustering do not establish recall or precision. The report mostly says so. Their remaining problem is that the statistical corrections needed to quantify incremental evidence were not run.

### 5. Has it missed any obvious or high-value leads?

**Yes. The highest-value missed lead is the producer's own aligned support pattern.**

The repository already has enough information to derive, for each 16-frame InpaintNet block:

- how many frames were supplied by InpaintNet;
- whether the block had any non-zero TrackNet pass-through;
- the longest selected run;
- distance to the nearest non-Inpaint, non-zero anchor on either side;
- whether the block is an interior block of a long selected span;
- whether any discontinuity is locked to the 16-frame producer boundary.

These features address the established failure mechanism without treating every Inpaint frame as wrong. They also give a better unit for any optional anomaly model.

A second high-value lead is to separate the two components of grade 3. A third is the small counterfactual that compares early coordinate rejection with the current late event mask. A fourth, conditional lead is retaining minimal TrackNet heatmap-shape evidence for visually confirmed non-Inpaint blips.

### Overall decision

Do **not** switch the complete path to a generic scikit-learn outlier detector now.

First validate a smaller source-aware stride-8 guard, split grade 3 into its actual components, and compare early versus late handling. Use one anomaly scorer only as a ranking baseline. Promote it to an automatic rejection rule only if the small visual challenge set and existing downstream labels show incremental value without clear harm.

---

## Claim ledger

| Claim | Ledger status | Finding and limit |
|---|---|---|
| The fixed 16-position loop comes from evidence-free InpaintNet input under the production checkpoint. | **Established fact** | Closed work. The upstream issue and repository investigation establish the mechanism. This review did not treat it as a new lead. [E1] [R4] |
| The training mask and inference mask regimes differ materially. | **Established fact** | Upstream training uses independent binomial masks intersected with visible ground truth; inference can present long contiguous or full-window gaps. Attribution belongs to `ahalp90` in upstream issue #22. [E1] [E2] [E3] |
| The sidecar tells whether InpaintNet supplied a saved frame. | **Established fact** | It records the raw applied-mask switch before final near-origin zeroing. It is provenance, not correctness. [R10] [R11] |
| An aligned 16-frame sidecar-selected block has no TrackNet pass-through in that block. | **Established fact** | Non-overlap Inpaint windows advance by 16. The final coordinates use Inpaint output where the mask is one and TrackNet coordinates where it is zero. [R12] [R13] |
| The live guard performs independent split-half discovery. | **Prior opinion that needed checking — rejected** | The historical script did independent discovery on each half. The live guard discovers on the full track and only checks presence in both halves. A midpoint-crossing window can count for both. [R6] [R22] |
| The guard threshold is data-derived and therefore not threshold-dependent. | **Prior opinion that needed checking — rejected** | The ratio gap is data-derived, but acceptance still requires two distinct counts, at least 30 episodes, and at least a 10x margin. [R6] [R21] |
| Grade 3 is safe to reject as event evidence. | **Open question** | Grade 3 combines a local halo with global exact coordinate reuse. It accounts for roughly 14–19% of valid frames and 26–28% of all rejected valid frames on the three fixtures. Its two components are not reported separately. [R6] [R7] |
| Rejecting grades `{1,2,3}` scored best on every fixture. | **Prior opinion that needs checking** | Commit `3f7621b` supports the shipped default with headline aggregate and pilot landing counts, but points to an untracked per-arm record. Fixture-wide superiority and the full comparison remain unauditable. [R15] |
| RANSAC candidates are missed hallucinations. | **Rejected interpretation** | They are departures from a local quadratic model. The candidate rate is about 31–33% of valid frames and includes real contacts, cuts, edges, and other non-quadratic motion. [R7] [R8] |
| RANSAC and guard overlap validates either detector. | **Measured result with limits** | Conditional overlap is 67–80%, but only 12.8–13.9 percentage points above the guard's high base marking rate. Temporal clustering prevents an independent-frame significance claim. [R7] |
| High sidecar span coverage means high hallucination recall. | **Rejected interpretation** | A span is counted as covered after one hit. This is length-biased and sidecar status is not error truth. [R9] |
| Exact-sequence clustering has found stable new families. | **Measured result with limits** | Uncaught exact sequences are mostly singletons, the selected sample is time-biased, absolute location and shape are mixed, and silhouette scores are low. It is a plotting aid, not detector evidence. [R7] [R19] |
| A standard outlier detector will remove threshold guesswork. | **Plausible-sounding prior opinion — rejected** | All candidate methods require a feature population and operating threshold. Several additionally require strong distribution or low-density assumptions. [E4] [E5] [E6] [E7] [E8] |
| A local motion score can add useful candidate-ranking power. | **Plausible inference** | It is well matched to abrupt blips. The current audit demonstrates volume, not correctness. Visual challenge labels are still needed to test yield. [R7] [R8] |
| Early removal and late event rejection are equivalent. | **Open question** | They are not equivalent in code. The event mask does not rewrite the track before smoothing and segmentation. The calibration path also supplies a pre-existing exclusion mask, further limiting early effects. [R16] [R17] |

---

## Keep the three tasks separate

### Task A: detect that InpaintNet supplied a position

For fresh stride-8 tracks, this is already solved by the sidecar. It is exact source provenance. [R10]

The recurrence guard is still useful for legacy tracks without sidecars. It should not be treated as a better provenance detector for new tracks.

### Task B: decide whether an Inpaint-supplied position is safe for a downstream use

This is not solved by provenance alone.

A partial, two-sided Inpaint fill may be useful for smoothing. The same coordinate may be too weak to terminate a rally or support a landing estimate. A fully unsupported Inpaint window under the production checkpoint is a different evidence class again.

This task needs source support, window context, and consumer policy. It does not need a claim that every selected frame is visibly wrong.

### Task C: detect visually false positions when provenance is absent, or when base TrackNet produced the error

This is where generic trajectory checks have their best role.

A local residual can detect abrupt disagreement with nearby motion. It cannot prove that the disagreement is false. A real contact, net interaction, held shuttle, landing, camera cut, or re-entry can generate the same geometry.

The three tasks should not be collapsed into one boolean anomaly label.

---

## Scikit-learn replacement assessment

## The observation unit matters more than the estimator name

### Raw-frame observations

A raw-frame feature row might contain:

- image-normalised `x` and `y`;
- first, second, and third coordinate differences;
- speed, acceleration, and jerk scaled by seconds and image diagonal;
- local quadratic residual;
- sidecar provenance;
- frame offset within the aligned 16-frame Inpaint window;
- distance to the nearest non-Inpaint, non-zero coordinate;
- scene or court-valid status.

This unit has three problems.

First, neighbouring rows are strongly dependent. A 20-frame chunk contributes 20 correlated training observations and receives 20 votes.

Second, frequent artefacts dominate the fitted distribution. The fixed loop contributes a dense, repeated cloud rather than a rare point.

Third, rare legitimate badminton events are exactly the frames most likely to have extreme derivatives.

Raw-frame models are therefore better as score generators than automatic frame rejectors.

### Window or span observations

A better unit is one row per aligned 16-frame Inpaint block, or one row per maximal candidate span for non-Inpaint errors.

Useful features include:

- selected-frame count and fraction;
- count of non-zero, non-Inpaint pass-through positions;
- longest selected run;
- whether support exists on both sides;
- distance to the nearest support on each side;
- interior versus edge position within a longer selected span;
- endpoint jump from surrounding TrackNet positions;
- median, maximum, and upper-quantile local motion residual;
- path length, displacement, curvature, and stationary fraction;
- absolute location features kept separately from translation-normalised shape features;
- 16-frame phase and boundary discontinuity indicators;
- fixed-loop/template score.

This unit reduces temporal weighting and matches the producer's actual computation. It also makes a generic ranker more interpretable.

### Frequent artefacts remain a core blind spot

Scikit-learn's current guide explicitly says that its outlier estimators assume anomalies lie in low-density regions and cannot form a dense cluster. [E4]

That is not a minor caveat here. A deterministic generator artefact can be common, internally consistent, smooth, and phase-locked. It can be more statistically normal than a real smash or net deflection.

The deterministic rule must therefore sit before, not after, a generic outlier model.

## Method-by-method assessment

| Method | What it can detect on raw-frame features | What it can detect on window/span features | Likely blind spots in this repository | Maintenance burden | Best role |
|---|---|---|---|---|---|
| **Isolation Forest** | Rare combinations of location, derivatives, residual, and provenance. It may rank isolated base-TrackNet blips well. | Mixed nonlinear combinations such as low support plus a large boundary jump. This is the best general scikit-learn baseline of the four pure outlier estimators. | A frequent loop or repeated smooth chunk can become normal. Long real events can be isolated. `contamination='auto'` supplies an algorithmic threshold, not a task-calibrated one. [E5] | **Medium.** Feature schema, fit population, random state, contamination or score cutoff, and version must be pinned. | **Secondary span ranker.** Use scores or top-k, not automatic labels initially. |
| **Local Outlier Factor** | Frames with much lower local density than nearby feature-space neighbours. It can expose a rare blip inside one motion regime. | Unusual spans relative to other spans with similar support or location. | A dense hallucination family is explicitly treated as normal. Mixed regimes and varying density make `n_neighbors` unstable. Standard outlier mode is transductive; novelty mode must only score unseen data. [E4] [E6] | **High.** Scaling, metric, neighbours, contamination, and outlier-versus-novelty semantics are all load-bearing. | **Exploratory diagnostic only.** It is not a good production default here. |
| **One-Class SVM** | Can learn a flexible boundary around scaled kinematic features. | Can model a nonlinear support region for span features. | It is sensitive to contamination, may absorb common artefacts, and can reject rare legitimate regimes. Scikit-learn states that `nu` needs fine-tuning for outlier detection. [E4] [E7] | **High.** Scaling, kernel, `gamma`, `nu`, fit population, runtime, and cutoff all matter. | **No immediate role.** It adds complexity without a matching source of clean inlier data. |
| **Elliptic Envelope** | Flags large robust Mahalanobis distances from one elliptical cloud. | Could work only on a deliberately narrow, near-Gaussian feature subset. | Shuttle motion is multimodal, bounded by image geometry, phase-dependent, and mixed across play states. The estimator assumes Gaussian inliers and learns an ellipse. [E4] [E8] | **Medium to high.** Scaling, feature selection, covariance support, and contamination are brittle. | **Reject for this problem.** It is useful only as a deliberately weak sanity baseline. |
| **RANSAC-style motion residuals** | Directly targets an isolated departure from local smooth motion. It is the best match for abrupt blips. | Span summaries can expose sustained endpoint disagreement or many residual peaks. | Smooth false chunks can fit the model. Real contacts, cuts, net events, held shuttles, and ground interactions can fail it. Missing zeros make some windows ineligible. | **Low to medium.** Assumptions and thresholds remain explicit: window duration, residual scale, inlier requirement, trials, and vote rule. | **Candidate generator and feature.** It is not a correctness label. |

## Method-specific conclusions

### Isolation Forest

This is the only pure scikit-learn outlier method worth a small controlled trial.

Its useful role is to rank **spans**, not to replace the deterministic loop rule. The model should receive one observation per span or aligned block. It should be fit on other videos and score a held-out video, or be treated as a within-video descriptive ranker with no transfer claim.

Using a contamination value to emit a binary mask would be unjustified. The true invalid fraction is unknown, and the current audit's candidate and guard rates are already very high.

### Local Outlier Factor

LOF is especially vulnerable to the common-artefact problem. Sixteen loop positions repeated thousands of times create high local density.

It also has an operational trap. With normal outlier detection, the training data is labelled through `fit_predict`. With `novelty=True`, prediction and scoring are only valid for unseen data and differ from training-set LOF scores. [E6]

That distinction is avoidable complexity for a guard whose strongest inputs are deterministic.

### One-Class SVM

There is no clean inlier-only training set. The observed tracks contain both producer artefacts and rare valid events.

The RBF boundary would therefore encode whichever mixture is present in the fitting videos. `nu` and `gamma` would become hidden policy settings. The method is more likely to create a tuning project than a maintainable guard.

### Elliptic Envelope

The inlier distribution is not one Gaussian ellipse. Absolute position, motion phase, rally state, and visibility create several modes.

Even a window-level representation would need aggressive feature restriction before the assumption becomes plausible. At that point, a transparent robust distance or rule would be easier to maintain.

### RANSAC-style residuals

RANSAC is not a drop-in answer to InpaintNet. It is a local trajectory consistency test.

The present audit is useful because it demonstrates a candidate source that is different from exact recurrence. It should be retained as a score or span proposal. Its thresholds need normalising by time and image scale before cross-domain interpretation.

## Maintenance comparison

### Current recurrence guard

Strengths:

- deterministic;
- no fitted model artefact;
- strong explanation when an exact moving motif repeats;
- works on legacy tracks.

Weaknesses:

- exact-byte equality;
- track-level adaptive threshold with hard acceptance gates;
- failure on short, trimmed, or perfectly regular cases;
- grade 3 mixes unrelated evidence;
- a global both-halves requirement;
- no producer provenance;
- normal production callers do not currently supply it.

### Generic fitted guard

New maintenance obligations:

- define and version the observation unit;
- define missing-value treatment around `(0,0)`;
- scale space and time consistently;
- select a training population;
- prevent long spans from dominating;
- pin estimator version and random state;
- define a score threshold without coordinate truth;
- detect distribution drift across videos and domains;
- explain why a rejected rare event is not valid badminton.

### Simpler source-aware guard

A source-aware stride-8 rule can be smaller than both.

For example:

1. verify sidecar status, stride, non-overlap mode, and the expected Inpaint checkpoint contract;
2. reconstruct the selected mask;
3. inspect aligned 16-frame blocks;
4. mark only the strongest no-support condition as coordinate-unusable;
5. keep a small tolerant loop/template fallback for legacy sidecarless tracks;
6. pass weaker source-risk information to consumers rather than deleting it.

This is the maintainability win. It does not come from importing an outlier estimator.

---

## Audit of the current investigation

## Genuine fumbles, ordered by severity

### 1. The recurrence detector can fail open on the canonical kind of regularity it is meant to catch

The live guard merges exact-pattern starts into one episode whenever consecutive starts are at most 32 frames apart. [R6]

The non-overlap fixed loop has a 16-frame period. Consecutive loop starts in a long fabricated run are therefore merged into one episode. A long bad chunk can count as one event rather than many.

The threshold layer then refuses to act unless there are at least two **distinct** episode-count values. The synthetic test explicitly verifies that one overwhelmingly repeated pattern plus otherwise unique footage returns all-zero grades. [R21]

A perfectly regular loop can make this worse. If all phase-shifted motifs have the same episode count and no ordinary pattern repeats twice, there is only one distinct candidate count and the guard declines to flag anything.

The minimum of 30 separated episodes adds a duration floor. Because a new episode requires a start gap greater than 32 frames, the theoretical minimum track length for 30 episodes of one 16-frame motif is 973 frames. That is about 39 seconds at 25 fps or 32 seconds at 30 fps. Real missing spans make the requirement larger.

This is not merely an unknown threshold choice. It is a structural false-negative mode for short clips, trimmed extracts, one-off long chunks, and unusually clean recurrence.

### 2. Grade 3 conflates two evidentiary claims, then the default rejects both

The live grade-3 mask is the union of:

- a local 15-frame halo around accepted recurrence cores; and
- any frame anywhere in the video whose exact integer coordinate equals any coordinate used by an accepted attractor. [R6]

The first is temporal proximity. The second is global coordinate collision.

A genuine TrackNet detection can revisit one of the 16 central loop pixels. The live code has no provenance signal with which to distinguish that case.

Grade 3 is not marginal in the current fixtures:

| Fixture | Grade-3 frames as share of coordinate-valid frames | Grade 3 as share of all rejected valid frames |
|---|---:|---:|
| `sset_01` | 14.16% | 26.72% |
| `sset_15` | 18.52% | 27.97% |
| `sset_21` | 14.34% | 25.95% |

No fixture has an accepted grade-2 flat attractor. The current all-nonzero policy is therefore effectively a grade-1-plus-grade-3 policy on this workset. [R7]

The audit does not expose `halo_only` and `global_exact_hit_only`. That prevents a direct safety assessment of the broadest live grade.

### 3. Historical “split-half validation” and the live check are different procedures

The historical `rule_recurrence_v3.py` discovers attractors independently in each half and compares the two key sets. [R22]

The live guard discovers attractors once on the full track. It then asks whether each accepted pattern has at least one occurrence overlapping each half. A window crossing the midpoint counts for both. [R6]

The live operation is a persistence gate. It is not held-out discovery, replication, or an independent validation split.

It also does not require balanced evidence. Twenty-nine episodes in one half and one in the other pass.

Finally, a failed presence check raises `ValueError` for the entire guard instead of dropping the unstable pattern. That is a brittle runtime policy for unseen domains.

### 4. The default grade policy is only partly auditable from the tracked repository

`BaseAnnotatorConfig` rejects `{1,2,3}` and cites commit `3f7621b` from
2026-07-22. Its message reports correct landing calls rising from 59 to 72 of
287, with the pilot rising from 22 to 31 of 113, at the cost of two winner
calls. It reports 46 correct landing calls when only proven-fabricated frames
were rejected. [R15]

The commit points to `records/commit12_default_pick.md` for the full comparison
and per-rally tables. That file is not tracked at the reviewed commit.

The tracked evidence supports the headline policy selection. It does not let a
source-backed review check:

- which metrics selected the arm;
- whether the run used a pre-existing exclusion mask;
- whether grade 3's two components were separable;
- whether the same fixtures were used for policy selection and reporting;
- whether the result concerned late event safety, early trajectory safety, or both.

The live default should therefore be treated as a prior measured choice, not established evidence of grade-3 correctness.

### 5. “Current filtering” is not a normal production input

The calibration path constructs recurrence grades and passes them into `run_video`. A replay-mask CLI also invokes the guard. [R16]

The normal production scraper and stroke-classifier callers do not construct `inpaint_codes`, and no normal consumer reads the sidecar. [R11] [R16]

This matters because the current repository does not have one deployed filter whose replacement can be judged. It has:

- a producer sidecar that is not consumed normally;
- a recurrence guard used in calibration and replay tooling;
- an event-mask seam in `run_video`;
- ordinary call sites that supply neither source.

Any statement that a candidate “beats the current production guard” would presently be ill-defined.

### 6. Sequence-family plots are not aligned to the producer unit

The plotting script scans every possible 16-frame start. In the Inpaint views it requires selected frames, but it does not require the start to be aligned to the non-overlap InpaintNet lattice. [R19]

A selected 16-frame slice can therefore:

- be a phase shift of one actual InpaintNet output;
- cross the boundary between two different InpaintNet calls;
- combine the tail and head of two producer windows.

That is acceptable for recurrence visualisation. It is weak evidence about distinct InpaintNet output families.

For uncaught sequences, every exact sequence is effectively a singleton. The top-256 tie-break then selects the earliest qualifying windows, producing a time-biased rather than representative sample. [R19]

## Weak or overstated interpretations

### RANSAC overlap is association, not validation

The measured counts give this derived summary:

| Fixture | Guard base rate among valid frames | RANSAC candidate rate | Guard rate within RANSAC candidates | Increment over guard base | Lift | Phi coefficient |
|---|---:|---:|---:|---:|---:|---:|
| `sset_01` | 52.99% | 30.98% | 66.93% | +13.94 pp | 1.26x | 0.19 |
| `sset_15` | 66.24% | 32.61% | 79.98% | +13.74 pp | 1.21x | 0.20 |
| `sset_21` | 55.27% | 31.41% | 68.07% | +12.80 pp | 1.23x | 0.17 |

These are modest positive associations. They are not independent confirmations.

Both masks are temporally clustered and both are functions of the same coordinates. A block-preserving or circular-shift null is needed before interpreting the excess overlap. An independent-frame permutation would give overconfident results.

The audit report and orientation brief recognise this limit. The weakness is omission of the null, not a fabricated recall claim. [R2] [R7]

### The RANSAC candidate definition is broad and non-uniform

The audit uses:

- a 16-frame quadratic model;
- starts every four frames;
- a 3-pixel residual threshold;
- 32 deterministic sample triples;
- at least eight inliers;
- a frame vote threshold of half its eligible windows. [R8]

A window is skipped if it contains any exact `(0,0)` coordinate. This is missing-not-at-random for a problem centred on tracking gaps. Severe gaps and their boundaries receive fewer or zero eligible votes.

Frames near video edges or skipped windows can also have only one eligible window. One outlier vote then suffices, whereas an interior frame normally needs multiple votes.

The 3-pixel threshold is not normalised by image scale. The 16-frame duration is also different in seconds at 25 and 30 fps.

These choices are acceptable for an exploratory candidate generator. They are too arbitrary for a production rejection rule.

### Span “any hit” coverage is length-biased

The sidecar follow-up counts a span as covered if at least one valid frame in the span is marked. [R9]

Long spans have more chances to receive a hit, even under an unrelated clustered mask. The proper comparison is by span-length stratum or against length-matched temporally shifted spans.

The report notes the caveat. The reported span percentages should not carry architectural weight until corrected.

### Event unions add little independent evidence

Impulse events and rally-ending events are downstream products of the same track and imperfect contact logic. They are not independent sensors of coordinate correctness.

A union can be useful for building a review set. It cannot establish recall merely by becoming larger.

Some coverage constructions are also close to tautological when the union contains the sidecar mask whose coverage is then discussed. The follow-up avoids the worst version by separating unions, but the remaining event context is still derivative evidence. [R7] [R9]

### Clustering results do not support a new detector

The low silhouettes, singleton-heavy exact sequences, complete-linkage choice, time-biased top-256 sample, and absolute-position-sensitive distance all limit interpretation. [R7] [R19]

The selected threshold is the best threshold on the same sample. There is no stability or held-out check.

The report treats the plots as descriptive. That is the correct level of confidence.

### Sidecar overlap is not visual truth

The guard marks 82.6–88.6% of sidecar-selected valid frames. This mostly says that the recurrence guard and producer fill regions occupy much of the same track. [R7]

It does not show that the remaining selected frames are false. It also does not show that marked non-selected frames are false.

## Acceptable exploratory choices

### Exact recurrence for the known moving motif

A moving 16-coordinate sequence recurring at many unrelated times is strong generator-signature evidence. Exact recurrence is transparent and appropriate for the closed fixed-loop mechanism.

The mistake is not using recurrence. It is allowing the track-level threshold and grade-3 expansion to carry more policy than the core proof supports.

### Deterministic RANSAC sampling

Using fixed triples makes the audit reproducible. Thirty-two trials are a reasonable exploratory compromise for a 16-frame window and an eight-inlier minimum.

The method is useful for finding review candidates. The report does not call them ground truth.

### Location and sequence plots

The plots are useful for finding obvious families and selecting challenge cases. The report records the weak silhouettes and singleton issue rather than hiding them.

### Issue #31 as a bounded challenge-set exercise

Issue #31 separates Inpaint and non-Inpaint origin, blips and chunks, mathematical and visual labels, and asks for exact frame ranges. [R1]

That is the right missing evidence for a small challenge set. It should not be reported as population precision or recall.

---

## Missed leads

| Lead | Likely value | Why it matters | Smallest useful test |
|---|---|---|---|
| **Aligned Inpaint support features** | **Very high** | They directly measure the condition under which InpaintNet is extrapolating with little or no TrackNet evidence. They use exact provenance without assuming error truth. | Reconstruct the sidecar mask. Produce one row per aligned 16-frame block with selected count, non-zero pass-through count, longest selected run, and distance to surrounding support. Stratify current guard, RANSAC candidates, and issue-31 cases by these fields. |
| **Fully selected aligned-window rule** | **Very high** | Under the reviewed non-overlap producer, a full selected block means no TrackNet coordinate passes through that Inpaint call. It catches the established evidence-free case without a video-wide recurrence threshold. | Build an analysis mask for complete, non-padded 16-frame blocks with all frames selected and final non-zero positions. Compare it with grade 1 and the known loop cases. Do not label all partial fills invalid. |
| **Split grade 3 into `halo_only` and `global_exact_hit_only`** | **High** | Current policy cannot be interpreted while two different mechanisms share one grade. The global coordinate match is the riskier component. | Reproduce the two internal masks in the audit only. Run the existing downstream comparison for grade 1, grade 1 plus halo, and grade 1 plus both components. |
| **Producer-phase and boundary diagnostics** | **High** | Other Inpaint errors may vary in coordinates but remain locked to the 16-frame lattice or to mask-pattern boundaries. Generic outlier methods ignore this structure. | Compare jump, acceleration, and residual distributions at producer boundaries versus interior offsets, stratified by selected-count/support class. Use circular phase shifts as a null. |
| **Early versus late handling counterfactual** | **High** | The current event mask does not remove false coordinates before smoothing and segmentation. It therefore cannot test whether false tracks create contacts or rally boundaries. | For only the strongest candidate mask, compare: no guard; external late event mask; copied track with selected coordinates set to `(0,0)` before `run_video`. Use existing captures and ground-truth event metrics. |
| **Minimal base-TrackNet heatmap morphology** | **Medium, conditional** | Sidecar-negative visually false positions are base TrackNet errors. The current producer thresholds the heatmap and retains only the largest binary contour, discarding confidence and ambiguity evidence. [R12] [R14] | Only if issue #31 confirms material non-Inpaint false cases, rerun those short ranges and retain peak probability, second-peak ratio, contour area, and entropy or mass. Do not export whole-video heatmaps first. |
| **Separate path shape from absolute location** | **Medium** | Current clustering mixes a central location with a trajectory shape. That can manufacture location clusters and hide repeated shapes elsewhere. | Centre each window and normalise scale for a shape distance, while keeping absolute location as separate features. Replot only the selected challenge windows. |

### The mask mismatch is not a missed lead

The training–inference mask mismatch is established work and was identified by `ahalp90` upstream. [E1]

The missed opportunity is to use the resulting **support pattern** more directly in the consumer guard. It is not to rediscover the mismatch.

---

## Minimal validation set

The following is the smallest set that can choose a viable stride-8 guard without a production refactor.

## Experiment A: resolve the current guard's policy ambiguity

### Purpose

Determine whether grade 3 adds downstream value and which part of it does so.

### Existing seams

- `BaseAnnotatorConfig.rejected_grades` already supports no guard, grade 1 only, and all non-zero grades. [R15]
- `run_video` already records rejection diagnostics when it receives source codes. [R16]

### Minimal additional analysis seam

Expose or reproduce, outside production, the two grade-3 components:

- halo only;
- global exact-attractor-coordinate hit only.

### Arms

1. no recurrence event mask;
2. grade 1 only;
3. grade 1 plus halo;
4. grade 1 plus halo plus global exact hits.

Grade 2 can be omitted from this fixture comparison because its measured count is zero in all three stride-8 tracks. Keep its policy open for future domains.

### Decision evidence

Use existing contact/rally/landing outputs and changed-case diagnostics. This tests downstream utility, not coordinate precision.

If global exact hits add no consistent value, stop rejecting them. If halo helps only one late consumer, do not promote it to a general coordinate rule.

## Experiment B: choose early versus late handling for the strongest source-aware mask

### Purpose

Determine where a high-confidence no-support condition should enter the system.

### Candidate mask

Use only complete aligned 16-frame blocks with the strongest no-support condition. Do not use all sidecar-selected frames.

### Arms

1. no candidate mask;
2. pass the mask as `shuttle_hallucination_mask` for late event handling;
3. set those coordinates to `(0,0)` in a copied track before `run_video`.

### Existing seams

- `run_video(shuttle_hallucination_mask=...)` supplies the late event arm. [R16]
- A copied track supplies the early arm without changing the producer or production path.
- `RunCapture` and rejection diagnostics expose changed segmentation, contacts, final contacts, and landings.

### Decision evidence

Use existing labelled downstream outcomes and inspect only changed cases. The question is not whether every source-selected frame is visually wrong. It is whether this strongest evidence class should be removed before or after trajectory consumers.

## Experiment C: test whether a learned ranker adds anything

### Purpose

Answer the scikit-learn question with one representative model, not five production prototypes.

### Candidates

- existing local quadratic residual summaries;
- one Isolation Forest over aligned-window or span features.

Do not spend validation effort on One-Class SVM or Elliptic Envelope unless this baseline succeeds. LOF can be a one-off diagnostic if local-density behaviour remains a specific question.

### Evaluation

Use the completed issue-31 challenge labels plus the already visually confirmed cases.

Report:

- top-k yield by producer source and failure shape;
- which confirmed false cases each ranker moves upward;
- which clearly valid edge cases it ranks highly;
- leave-one-video-out ranking where sample size permits;
- no population precision or recall claim.

Use one span/window as one observation. Do not weight a 20-frame chunk twenty times.

### Statistical correction bundled into this experiment

For the existing RANSAC/guard and sidecar-span summaries, add:

- a circular-shift or block-preserving null for overlap;
- span-length-stratified or length-matched coverage;
- phase-shift nulls for producer-boundary effects.

These are small calculations on existing arrays. They prevent another heuristic from being selected on base-rate artefacts.

## What not to build yet

- no full sklearn production pipeline;
- no per-frame coordinate annotation;
- no stride-1 guard;
- no whole-video heatmap archive;
- no large per-consumer policy matrix;
- no retraining of InpaintNet as a prerequisite for the consumer decision.

---

## Likely final architecture — provisional

This is a likely shape, not a selected design.

### 1. Preserve provenance and support as typed evidence

For fresh tracks, consume the sidecar and reconstruct the raw selected mask.

Derive aligned-window support fields. Preserve the distinction between:

- base TrackNet coordinate;
- partial Inpaint fill with support;
- fully unsupported Inpaint block;
- legacy source unknown;
- generic motion anomaly.

Do not collapse those classes immediately into one boolean.

### 2. Apply a narrow high-confidence coordinate rejection before smoothing

For the production checkpoint and stride-8 non-overlap contract, reject only the established no-support/fixed-loop class after Experiment B confirms early handling.

For legacy tracks without sidecars, keep a small fallback:

- aligned tolerant template match for the known loop; or
- a much narrower exact moving-recurrence proof without grade-3 global expansion.

The fallback should not require 30 separated episodes or both-half presence to recognise one known bad window.

### 3. Use source risk at downstream consumers

Partial Inpaint fills can remain available to smoothing while being treated conservatively by final-contact and landing rules.

The policy can remain a small number of evidence grades. It need not become a large model.

Grade 3 in its current form should not survive unchanged. Halo and global coordinate reuse should be separate signals.

### 4. Keep a generic anomaly score outside the proof path

A local motion-residual score, optionally supplemented by an Isolation Forest span rank, should initially serve:

- issue-31 candidate ranking;
- diagnostics;
- monitoring of base TrackNet blips;
- discovery of new failure families.

Only a narrowly validated score region should become automatic event rejection. Smooth, common artefacts and rare valid contacts remain fundamental blind spots.

### 5. Add producer confidence only if non-Inpaint errors justify it

If the challenge set shows that base TrackNet is a material source of harmful false positions, retain a few scalar heatmap-shape fields on targeted reruns.

Do not make this a prerequisite for the source-aware Inpaint guard.

---

## Evidence index

## Review inputs

- **Orientation brief:** uploaded `01_orientation_brief.md`. Used as a map of established facts, measured anchors, and prior concerns. It was not treated as authoritative where source inspection was possible.
- **Progressive evidence map:** uploaded `02_evidence_map.md`. Tier 1 was read first. Tier 2 was opened only to test implementation, integration, and statistical claims.

## Repository snapshot

All repository links below are pinned to commit `781a589b13a2a3ec986c0562e8d5bf9f6614cef0` unless noted.

### Tier 1 and principal repository sources

- **[R1] Issue #31 — visual challenge-set task**  
  `https://github.com/ahalp90/badminton_cv_annotator/issues/31`

- **[R2] Workset index**  
  `docs/scraper_pipeline/inpaint_hallucination_fix/README.md`  
  `https://github.com/ahalp90/badminton_cv_annotator/blob/781a589b13a2a3ec986c0562e8d5bf9f6614cef0/docs/scraper_pipeline/inpaint_hallucination_fix/README.md`

- **[R3] Closed mechanism investigation**  
  `docs/tracknet/evidence/inpaint_fabrications_20260722/inpaint_fabrications_investigation.md`  
  `https://github.com/ahalp90/badminton_cv_annotator/blob/781a589b13a2a3ec986c0562e8d5bf9f6614cef0/docs/tracknet/evidence/inpaint_fabrications_20260722/inpaint_fabrications_investigation.md`

- **[R4] Historical detector decision sheet**  
  `docs/tracknet/evidence/inpaint_fabrications_20260722/detector_options.md`  
  `https://github.com/ahalp90/badminton_cv_annotator/blob/781a589b13a2a3ec986c0562e8d5bf9f6614cef0/docs/tracknet/evidence/inpaint_fabrications_20260722/detector_options.md`

- **[R6] Live recurrence guard**  
  `src/annotator/inpaint_guard.py`  
  `https://github.com/ahalp90/badminton_cv_annotator/blob/781a589b13a2a3ec986c0562e8d5bf9f6614cef0/src/annotator/inpaint_guard.py`

- **[R7] Current audit report**  
  `docs/scraper_pipeline/inpaint_hallucination_fix/ongoing_shuttle_hallucination_issues_20260731-094523.md`  
  `https://github.com/ahalp90/badminton_cv_annotator/blob/781a589b13a2a3ec986c0562e8d5bf9f6614cef0/docs/scraper_pipeline/inpaint_hallucination_fix/ongoing_shuttle_hallucination_issues_20260731-094523.md`

### Tier 2 implementation and audit sources

- **[R8] RANSAC audit implementation**  
  `docs/scraper_pipeline/inpaint_hallucination_fix/analysis/audit_tracks.py`  
  `https://github.com/ahalp90/badminton_cv_annotator/blob/781a589b13a2a3ec986c0562e8d5bf9f6614cef0/docs/scraper_pipeline/inpaint_hallucination_fix/analysis/audit_tracks.py`

- **[R9] Sidecar coverage follow-up and implementation**  
  `docs/scraper_pipeline/inpaint_hallucination_fix/inpaint_provenance_coverage_followup_20260731-192654.md`  
  `docs/scraper_pipeline/inpaint_hallucination_fix/analysis/measure_inpaint_coverage.py`

- **[R10] Sidecar producer contract**  
  `docs/tracknet/inpaint_sidecar.md`  
  `https://github.com/ahalp90/badminton_cv_annotator/blob/781a589b13a2a3ec986c0562e8d5bf9f6614cef0/docs/tracknet/inpaint_sidecar.md`

- **[R11] Sidecar consumer state**  
  `docs/tracknet/inpaint_sidecar_consumption.md`  
  `https://github.com/ahalp90/badminton_cv_annotator/blob/781a589b13a2a3ec986c0562e8d5bf9f6614cef0/docs/tracknet/inpaint_sidecar_consumption.md`

- **[R12] TrackNetV3 inference path**  
  `src/bst_x/TrackNetV3/predict.py`  
  `https://github.com/ahalp90/badminton_cv_annotator/blob/781a589b13a2a3ec986c0562e8d5bf9f6614cef0/src/bst_x/TrackNetV3/predict.py`

- **[R13] Inference dataset windowing and coordinate inputs**  
  `src/bst_x/TrackNetV3/dataset.py`  
  `https://github.com/ahalp90/badminton_cv_annotator/blob/781a589b13a2a3ec986c0562e8d5bf9f6614cef0/src/bst_x/TrackNetV3/dataset.py`

- **[R14] Mask generation, heatmap coordinate extraction, Inpaint model, and sidecar writer**  
  `src/bst_x/TrackNetV3/inference_utils.py`  
  `src/bst_x/TrackNetV3/model.py`  
  `src/bst_x/TrackNetV3/write_inpaint_metadata.py`

- **[R15] Current grade policy**  
  `src/annotator/config.py`  
  `https://github.com/ahalp90/badminton_cv_annotator/blob/781a589b13a2a3ec986c0562e8d5bf9f6614cef0/src/annotator/config.py`

- **[R16] Runtime integration and source trace**  
  `src/annotator/run_video.py`  
  `docs/scraper_pipeline/inpaint_hallucination_fix/inpaint_hallucination_trace.md`

- **[R17] Calibration caller**  
  `src/annotator/calibration/gt_scoring.py`

- **[R18] Current stride comparison**  
  `experiments/annotator/runs/20260730-041328/measurement_verification.md`  
  `https://github.com/ahalp90/badminton_cv_annotator/blob/781a589b13a2a3ec986c0562e8d5bf9f6614cef0/experiments/annotator/runs/20260730-041328/measurement_verification.md`

- **[R19] Sequence and location plotting**  
  `docs/scraper_pipeline/inpaint_hallucination_fix/analysis/plot_recurrence_grids.py`

- **[R20] Measurement history**  
  `docs/scraper_pipeline/annotator_measurement_history.md`

- **[R21] Synthetic guard tests**  
  `tests/test_inpaint_guard.py`

- **[R22] Historical independent split-half script**  
  `docs/tracknet/evidence/inpaint_fabrications_20260722/stride1_retrack/rule_recurrence_v3.py`

### External primary sources

- **[E1] Upstream TrackNetV3 issue #22**  
  `https://github.com/qaz812345/TrackNetV3/issues/22`

- **[E2] Upstream TrackNetV3 README, InpaintNet training command**  
  `https://github.com/qaz812345/TrackNetV3/blob/77c123ad4dd449b7d275f16cc43f316ba5b54042/README.md`

- **[E3] Upstream TrackNetV3 training code**  
  `https://github.com/qaz812345/TrackNetV3/blob/77c123ad4dd449b7d275f16cc43f316ba5b54042/train.py`

- **[E4] Scikit-learn novelty and outlier detection overview, version 1.9**  
  `https://scikit-learn.org/stable/modules/outlier_detection.html`

- **[E5] Scikit-learn IsolationForest, version 1.9**  
  `https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html`

- **[E6] Scikit-learn LocalOutlierFactor, version 1.9**  
  `https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html`

- **[E7] Scikit-learn OneClassSVM, version 1.9**  
  `https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html`

- **[E8] Scikit-learn EllipticEnvelope, version 1.9**  
  `https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html`

- **[E9] Scikit-learn RANSACRegressor, version 1.9**  
  `https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html`

---

## Uncertainty statement

Without frame-level coordinate ground truth, this review cannot determine population precision, recall, or the exact false-negative rate of any position guard.

Without the issue-31 visual labels, it also cannot determine which RANSAC candidates, partial Inpaint fills, or sidecar-negative blips are visibly false. Geometry can identify disagreement with a motion model. It cannot decide every contact, landing, net event, held shuttle, camera cut, or occlusion case.

Existing contact, rally, landing, and winner labels can measure **downstream utility**. They cannot certify coordinate correctness. A guard can improve a downstream metric while rejecting some visually valid coordinates, or preserve visually wrong coordinates that never affect an event.

The three fixtures are professional broadcast videos. They do not establish behaviour on amateur, low-resolution, unusual-camera, or different-checkpoint footage.

Two exact pieces of untracked material would resolve specific remaining claims:

1. **`records/commit12_default_pick.md` and its per-arm outputs.**  
   This is needed to audit why `{1,2,3}` became the default, which downstream metrics moved, and whether the comparison tested late masking only.

2. **The exact visually confirmed uncaught frame ranges or completed issue-31 CSVs/overlay labels.**  
   These are needed to answer whether Isolation Forest or motion residuals rank more genuinely false blips and chunks, and whether the errors are Inpaint-supplied or base-TrackNet-supplied.

Raw TrackNet heatmaps or confidence exports are not required for the first guard decision. They become useful only if the completed challenge set shows that sidecar-negative base-TrackNet errors are a material residual source.

The strongest conclusion that can be made now is architectural: a generic outlier estimator is not a sound replacement for mechanism and provenance. A narrow source-aware rule plus a separately validated motion-anomaly ranker is the more defensible stride-8 direction.
