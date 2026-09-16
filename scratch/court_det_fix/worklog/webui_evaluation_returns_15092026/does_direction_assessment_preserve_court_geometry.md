# Does our direction objective preserve accurate court geometry?

**Evidence inspected.** All reads were pinned to `ahalp90/badminton_cv_annotator@7299ff3`, not the current branch head. I inspected `evaluation/README.md`, the direction/SVD/error-metric sections of `evaluation/method_excerpts.md`, `followup_status.md`, and the evidence-root `README.md`. The guide identifies `b90518c` as the scientific checkpoint. I fully decoded `gx0_control_measurements.json.gz` and verified its Git blob hash. From `evaluation/direction_records.json.gz`, I decoded the header, GX0 settings, and merged-line rows 0–34; **I did not obtain the complete decoded banks or SVD pair records, or a complete decoded `svd_fixed_measurements.json.gz`.** Consequently, the SVD figures below are published-summary evidence, not independently revalidated pair-fit results.

## Strongest conclusion

**The direction objective is not geometry-preserving, and coverage-first allocation does not guarantee preservation of the useful alternatives.** An executed synthetic construction below makes R select a lower-residual representative while worsening the best fixed-direction court fit. Separately, the saved GX0 control uses a candidate marked `capped`: R cannot select that candidate because it lies outside every allocated bucket.

The practical implication is to **keep the fixed experiments, but audit them as bounded tests of anchoring and within-bucket choice—not as a complete test of direction recovery**. A residual reduction, a better control fit, a better generated population, and a usable ranked winner are four different results.

## 1. The objectives measure different forms of line concurrence—not finished-court accuracy

### Mathematical deduction: what the angles actually measure

Let a merged line have unit normal \(n\), offset \(c\), unit tangent \(t\), and an anchor \(a\) on the line. For homogeneous VP \(v=(v_{xy},w)\), the tested ray is

$$
r=v_{xy}-wa.
$$

Because \(n^\top a+c=0\), the source angle is

$$
\theta_a(v)=
\operatorname{atan2}\!\left(
|n^\top v_{xy}+cw|,
|t^\top(v_{xy}-wa)|
\right),
$$

except that a sufficiently short ray receives \(90^\circ\). For a finite, dehomogenised VP \(p\),

$$
\tan\theta_a=\frac{|n^\top p+c|}{|t^\top(p-a)|}.
$$

Thus the angle is **normal incidence error divided by a tangential lever arm**. Changing the anchor changes that lever arm, not the observed line orientation. This follows directly from `angular_residuals` and the midpoint construction in the guide.

That distinction produces two important effects.

**An anchor effect without observational ambiguity.** For exact lines

$$
\ell_1=(1,0,0),\qquad \ell_2=(0,1,0),
$$

the exact common VP is \(v=(0,0,1)\). Both original feet are the origin. The implemented angles are therefore **\(90^\circ,90^\circ\)**, despite exact concurrence. Anchors \((0,1)\) and \((1,0)\), respectively, give **\(0^\circ,0^\circ\)**. I executed this check. The undefined-ray convention is defensible as an angular convention, but its conversion into “unsupported line” is not a valid rejection of concurrence.

More generally, on \(y=0\), compare finite VPs \(p=(100,1)\) and \(q=(1,0.1)\):

| Physical anchor |     Angle to \(p\) |     Angle to \(q\) |
| --------------- | -----------------: | -----------------: |
| \((0,0)\)       | \(0.572939^\circ\) | \(5.710593^\circ\) |
| \((99,0)\)      |       \(45^\circ\) | \(0.058465^\circ\) |

The preferred VP reverses without changing either VP or the observed line. Moving coordinates while carrying the **same physical anchor** does not cause this reversal. Recomputing the foot relative to a different origin does: it selects a different physical measurement location. The detector fixes the origin at the image centre, so this is not an uncontrolled representation bug; it is a particular anchoring choice whose relevance to the observations must be tested.

**Genuine ambiguity from short, noisy observations.** Midpoint anchoring cannot remove uncertainty in the measured orientation. Under independent endpoint normal errors of standard deviation \(\sigma\), a short segment of length \(L\) has first-order angular uncertainty approximately \(\sqrt{2}\sigma/L\). A small coherent orientation bias across fragments can therefore look extremely consistent while extrapolating incorrectly across the court. That is different from the arbitrary-anchor effect.

The finite/infinite distinction also matters. At infinity, \(w=0\), so the anchor disappears completely: B and M give the same angle for that candidate and line. Its eventual selection can nevertheless change because M changes competing finite candidates and hence allocation. Near a finite VP–anchor coincidence, the angle becomes highly sensitive and ultimately undefined; midpoint anchoring moves this weakness rather than eliminating it.

Line scale/sign and VP sign cancel in the angular formula. Strict invariance to arbitrary VP scale is interrupted by the absolute \(10^{-12}\) ray cutoff, although the estimator’s unit-normalised bank fixes that scale operationally. The centred, isotropic coordinate transform should be preserved exactly—not replaced by anisotropic image scaling.

### Mathematical deduction: the SVD objective has a different denominator

After unit-2D-normal line normalisation, `svd_direction` solves

$$
\min_{\|v\|_2=1}J(v),\qquad
J(v)=\frac1m\sum_i(\ell_i^\top v)^2.
$$

For a finite point represented as \(v=(p,1)/\sqrt{1+\|p\|^2}\),

$$
J(p)=
\frac{\frac1m\sum_i(n_i^\top p+c_i)^2}{1+\|p\|^2}.
$$

This removes arbitrary input-line scaling, but **does not make the objective a court-space distance or an intrinsic projective distance**. Translating the coordinate origin preserves the finite point-to-line distances in the numerator but changes the denominator. At infinity, offsets disappear and SVD measures orientation agreement. A small algebraic residual or a large `normalised_nullspace_gap` describes the supplied line system in its chosen normalisation; neither is a certificate of projected-court accuracy.

### Executed construction: all these scores improve while court placement worsens

Consider a synthetic frontoparallel court with

$$
H_0=
\begin{bmatrix}
30&0&300\\
0&30&50\\
0&0&1
\end{bmatrix},
$$

mapping court metres to pixels. Give both observed line families a coherent \(1^\circ\) orientation bias. On a 20-pixel fragment, rotating its line about its midpoint requires only

$$
10\tan 1^\circ=0.174551\text{ pixels}
$$

of normal displacement at each endpoint.

For each family, moving from the true direction to the biased observed direction changes:

* foot and midpoint angles from \(1^\circ\) to \(0^\circ\);
* the corresponding capped squared-angle score from \(1\) to \(0\);
* unit-normal algebraic RMS from \(0.0174524\) to numerical zero, which SVD attains.

With both biased directions fixed, the exact coordinate-least-squares court fit is

$$
H_1\approx
\begin{bmatrix}
29.9908624&-0.5234925&303.5352691\\
0.5234925&29.9908624&48.4645699\\
0&0&1
\end{bmatrix}.
$$

Every corner is **3.854306 pixels** from its true position, versus zero for \(H_0\). Both homographies are similarities, and both direction pairs are orthogonal: this example does not depend on an invalid shear or a nearly collapsed pair.

This is an **executed synthetic score/geometry counterexample**, not a result on any recorded frame. Its assumption is a coherent measurement bias; it establishes a non-guarantee, not that this bias explains GX0, GX5, or Amateur-2.

### Can label-free quality still be useful?

**Yes, conditionally—but not as an unconditional court-accuracy score.** In the finite-origin gauge used by `control_fit`,

$$
H=[\,\alpha u\;\;\beta v\;\;t\,],\qquad t_w=1.
$$

Even exact directions leave four scale/position parameters unresolved. For a projected point \(p\) with homogeneous denominator \(z\), directional perturbations contribute

$$
\delta p=
\frac1z
\begin{bmatrix}I_2&-p\end{bmatrix}
\left(\alpha x\,\delta u+\beta y\,\delta v\right),
$$

before accounting for refitting the other parameters. Unknown scale, position and perspective conditioning prevent a VP-only angular distance from being converted into a universal court-error bound.

A useful label-free measure could instead assess **conditional consistency and uncertainty**, assuming straight coplanar lines, defensible family membership or an explicit contamination model, a fragment-noise model, adequate spatial spread, and—when predicting court error—a bounded plausible court region and nondegenerate pair/projection geometry. Residual alone does not supply those assumptions.

**Contrary evidence and falsification.** Correctly assigned, noiseless, sufficiently informative lines do determine the correct VP; SVD is useful there, and midpoint anchoring can fix the exact-foot failure above. The published SVD improvements also oppose any claim that fitting is generally harmful. The mathematical counterexamples are falsifiable by replaying their stated inputs and formulas; a successful real-frame experiment would not refute a non-guarantee. Conversely, the hypothesis that this particular midpoint rule materially improves these development cases can be contradicted by correctly replayed M results with no useful geometric benefit.

## 2. Bucketing can lose the solution before precision is considered—and retained coverage can differ from allocated coverage

### Source observation: the greedy rule is not merely “find groups, then fit them”

For coverage selection—confirmed in the readable GX0 settings—`retain_pencils` orders eligible candidates by:

$$
\text{newly covered lines},\quad
\text{total count},\quad
\sum_i\max(0,1-\theta_i/1.5),\quad
\text{ascending candidate ID}.
$$

Eligibility requires at least two lines. The winner updates coverage and suppresses currently eligible candidates with mask IoU **strictly greater than 0.8**. Allocation continues for at most 16 leaders; there is no early stop merely because novelty becomes zero. The proposed R/MR rule changes only the representative inside those already allocated, disjoint buckets.

Three consequences follow.

First, **`capped` candidates are outside the allocated bucket union**. They were eligible but never suppressed or retained. A within-bucket rule cannot rescue them.

Second, mask overlap is not geometric equivalence. It neither bounds VP separation nor bounds the effect on a court paired with another direction.

Third, the representative’s own mask can differ from the leader’s. With six leader lines, retaining five still gives IoU \(5/6>0.8\). The sixth line may be spatially important, and every residual from \(1.5^\circ\) to \(90^\circ\) contributes the same capped penalty. The allocator nevertheless continues as though the **leader’s** coverage were retained. Subsequent SVD uses the representative’s own mask, not the leader mask. Thus “fixed support” is fixed within each fit, not necessarily common across B, R, M and MR.

### Saved empirical finding: R cannot select the documented GX0 candidate-ID pair

In the fully decoded GX0 control record, `selected_direction_candidates` contains:

| Candidate ID | Saved support count | Saved status |
| ------------ | ------------------: | ------------ |
| 1183         |                  16 | `redundant`  |
| 122          |                   8 | `capped`     |

I checked candidate 122 against the readable direction-record prefix: intersecting GX0 `direction_lines[1]` and `[19]`, after the specified normalisation, reproduces its saved normalised point exactly,

$$
(-0.3960237828,\;-0.0279754996,\;0.9178139980).
$$

With 106 lines, that pair’s pre-degeneracy construction ID is indeed 122. This check uses the published merged-line rows—not inferred fragment assignments.

The GX0 record’s `best_bank_fit` uses IDs 1183 and 122 and reports **0.509456 working pixels**, compared with **6.798367** for `best_retained_pair_fit`. Those are saved, label-guided local control fits, not rerun optima.

Separately, the observed-bank control’s generated line/paint winners, IDs 1864 and 5144, have the saved essentially-ideal judgements. I recomputed their maximum distances to approved candidate89 from the stored native corners: **3.120854 and 4.050048 display pixels**, matching the published measurements. Those generated results—not the 0.509456 diagnostic alone—establish the demonstrated matcher capacity.

Therefore **R cannot select candidate 122 itself under the original allocation**. It might recover another useful direction or pair; I have not inspected the complete bank sufficiently to rule that out. M can change allocation and may make previously capped candidates available.

### Executed construction: a bucket can contain materially different geometry

Use these three normalised-coordinate lines:

$$
L=
\begin{bmatrix}
0&1&0\\
-0.005&1&0.005\\
0.005&1&0.005
\end{bmatrix}.
$$

Construct the complete pair-intersection-plus-infinity bank exactly as in `estimate`. Its first bucket includes:

| Candidate | Homogeneous point, up to scale/sign | Three foot angles                   | Fixed-mask score |
| --------- | ----------------------------------- | ----------------------------------- | ---------------: |
| ID 0      | \((1,0,1)\)                         | \(0,0,0.572939^\circ\)              |         0.109420 |
| ID 1      | \((-1,0,1)\)                        | \(0,0.572939^\circ,0\)              |         0.109420 |
| ID 3      | \((1,0,0)\)                         | \(0,0.286477^\circ,0.286477^\circ\) |         0.054713 |

All have the same three-line mask, hence IoU 1. The exact greedy ordering chooses ID 0 as the first leader; R chooses ID 3. **The cap is inactive**, so this failure does not depend on truncating an outlier.

For a downstream comparison of this one direction, hold the other direction available and fixed at \((0,1,0)\). Let the synthetic reference court be

$$
H_{\rm true}=
\begin{bmatrix}
0.05&0&0\\
0&0.015&-0.005\\
0.05&0&1
\end{bmatrix}.
$$

The leader permits an exact fit. The representative replaces finite convergence with infinity and forces an axis-aligned affine court. Its globally optimal **coordinate-least-squares** fit is

$$
H_R\approx
\begin{bmatrix}
0.03831418&0&0\\
0&0.01324713&-0.00441571\\
0&0&1
\end{bmatrix}.
$$

Using the guide’s 960×540 normalised-to-working transform, its maximum corner error is **25.227929 working pixels**. R roughly halves its direction score while making the best conditional court geometry worse.

This is a constructed bucket and conditional court-fit comparison, **not a full detector run or a claim that these synthetic homographies pass the detector’s camera/player gates**. Its role is to refute interchangeability based solely on mask overlap and residual minimisation.

The synthetic checks were executed. Their complete inputs and calculations are available as [standalone NumPy code](sandbox:/mnt/data/synthetic_geometry_checks.py) and [recorded output](sandbox:/mnt/data/synthetic_geometry_checks_output.json).

**Contrary evidence and falsification.** The redundant GX0 candidate is potentially recoverable inside a bucket; R could also succeed through an alternative pair. An essentially ideal R-generated GX0 court would contradict the stronger hypothesis that the capped candidate is an unavoidable bottleneck—not the narrower fact that R cannot select ID 122. A baseline replay showing ID 122 allocated would expose a provenance/replay inconsistency requiring resolution. Replaying the synthetic bank must reproduce its leader, bucket and representative; otherwise that construction is refuted.

## 3. Pair geometry and the diagnostic–generation–ranking gap remain essential to interpreting the matrix

### Mathematical deduction: independent direction quality omits joint conditioning

For \(H=[\alpha u,\beta v,t]\),

$$
\det H=\alpha\beta (u\times v)^\top t.
$$

Neither independent residual score controls this joint quantity, the pair’s horizon \(u\times v\), or the projection denominators across the court. Consequently, the two individually preferred directions need not form the preferred pair. Per-group SVD conditioning does not establish pair conditioning.

A simple image-space construction shows the leverage: take two unit-normal lines whose tangents differ by \(0.5^\circ\). A 0.1-pixel normal-offset error on one line can move their intersection by

$$
\frac{0.1}{\sin0.5^\circ}=11.459301\text{ pixels},
$$

versus 0.1 pixels for perpendicular lines. This is genuine image-geometry conditioning, not homogeneous sign ambiguity. It does **not** imply that badminton directions should be perpendicular in the image.

### Published empirical finding: SVD helps diagnostics, but its cause and downstream effect are not settled

The status file reports these best saved local fits, in 960×540 working pixels:

| Case                 | Original | Fixed-group SVD | Control-selected-group SVD |
| -------------------- | -------: | --------------: | -------------------------: |
| GX0                  |    6.798 |           3.961 |                      0.952 |
| GX5                  |   23.800 |          22.502 |                      2.540 |
| Amateur-2 frame28019 |    2.972 |           1.626 |                      1.441 |

This supports partial diagnostic benefit and membership sensitivity, with little change on GX5. **I could not independently verify the full compressed SVD records or identify individual saved pairs that improved or regressed.** Better minima do not imply pairwise dominance, and the winning pair can change between variants. Control-selected groups are label-guided alternatives, not known-correct memberships or a performance ceiling.

`control_fit` optimises eight coordinate residuals with local nonlinear least squares and a 200-evaluation budget. It does not directly optimise maximum corner distance or certify a global optimum. Its fixed corner correspondence also differs from `corner_errors`, which permits the existing 180-degree relabelling for generated courts. Working-pixel diagnostic errors must not be mixed with 1280×720 display errors.

### The fixed matrix and its interpretation

The comparison is exactly:

| Arm | Anchor                                                             | Representative                                      |
| --- | ------------------------------------------------------------------ | --------------------------------------------------- |
| B   | Original foot                                                      | Original greedy coverage leader                     |
| M   | Projected midpoint of longest actual contributing clipped fragment | Original greedy coverage leader                     |
| R   | Original foot                                                      | Minimum capped squared angle in the leader’s bucket |
| MR  | Same projected midpoint                                            | Minimum capped squared angle in the leader’s bucket |

Equal fragment lengths use canonical observation index. Representatives minimise `mean(min(angle,1.5)**2)` on the fixed leader mask, with candidate-ID tie-breaking. Leaders alone allocate coverage; there is no cross-bucket backfill or support reselection.

All four arms and their separate fixed-support SVD variants receive ordered-pair control diagnostics on all nine views. **Only M and R enter the unchanged matcher:** 18 new case-arms, followed by 36 camera-first/all-camera rescoring case-arms. MR and SVD remain diagnostic-only. Exact merge provenance and baseline replay precede interpretation; the matrix has no published result yet.

| Comparison or outcome                                                        | What it supports                                                                                                                             | What it contradicts or leaves unresolved                                                                                                                                                                                                  |
| ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **M improves over B**                                                        | Anchor-driven selection matters on that evidence.                                                                                            | M changes masks, support scores, leaders and buckets. Improvement does not isolate “measurement at the visible location is intrinsically more accurate.”                                                                                  |
| **M regresses, or gives no useful improvement**                              | Retaining the original measure for this specified variant is justified.                                                                      | Regression opposes the proposed benefit on those cases. No change with unchanged selected inputs is an inactive intervention, not validation of the original objective.                                                                   |
| **R improves over B**                                                        | Within allocated buckets, the leader was not always the best direction for the downstream diagnostic or matcher.                             | It does not test candidates outside those buckets or prove the residual is generally geometry-aligned.                                                                                                                                    |
| **R lowers its score but geometry regresses; or R does not help**            | Regression directly exposes proxy misalignment and/or harmful pair choice.                                                                   | No benefit can also result from already-discarded alternatives. It cannot by itself refute the broader direction-selection explanation.                                                                                                   |
| **MR differs from M and R**                                                  | The two specified changes interact. A combination-only gain supports a conditional mechanism.                                                | Anchor-dependent buckets differ, so this is not a comparison on identical groups. Diagnostic-only success says nothing yet about generated or ranked courts. A combination regression opposes treating the changes as independently safe. |
| **Adding SVD improves, regresses, or leaves an arm unchanged**               | Improvement shows fixed-support refitting can help that diagnostic. Regression despite lower algebraic error exposes objective misalignment. | No change leaves membership, prior near-optimal fitting and pair limitations unresolved. Across arms, different masks mean the comparison is not solely a refitter comparison.                                                            |
| **Diagnostics improve but the matcher does not produce better courts**       | Conditional direction capacity improved.                                                                                                     | Actual recovery remains unestablished: the matcher, evidence assignment, budgets or gates may prevent use of that capacity. It is not automatically a ranking failure.                                                                    |
| **The generated population contains a usable court, but the winner is poor** | Generation has demonstrated capacity in that population.                                                                                     | Ranking/retention remains a bottleneck. Conversely, a better winner without better control-fit error is evidence against using that error as the sole definition of progress.                                                             |

A “no-change” assessment should therefore distinguish unchanged candidate identities/masks from merely unchanged best error. Likewise, a better best fit should be checked against regressions elsewhere in the nine-view development set, rather than interpreted as blanket preservation.

**Contrary evidence and falsification.** The already demonstrated GX0 control and reported SVD improvements show that useful direction changes can propagate into better fits; the mathematics does not predict universal failure. A usable M/R-generated court followed by successful unchanged ranking would support the proposed causal path on that case. A usable court already present but consistently ranked away would contradict direction loss as the explanation for that particular emission failure. Neither outcome establishes universal accuracy.

### One conditional additional check—not another search

If the specified results still leave **“bad within-bucket choice” versus “useful alternative excluded before choice”** unresolved, the one additional discriminating check is a **survival trace of the already documented control-bank pair for the unresolved case**.

For each member, record eligibility, allocated bucket or exclusion reason, its residual rank on the leader mask, and the chosen representative. This is a bounded trace of known candidates, not a new bank search or matcher experiment.

The minimum inputs are the candidate construction/IDs, merged lines, arm-specific anchors, ordered leaders and masks, and representative IDs. Original-anchor inputs are already published by construction in `direction_records.json.gz`; midpoint replay additionally needs the actual contributing-fragment provenance and midpoints. **The inspected guide already specifies pair-fit diagnostics and provenance replay, but does not establish that Claude returns this precomputed survival trace.** Its inputs may suffice to derive it; I cannot claim the trace itself is already returned. The old compatibility lists must not substitute for membership.

---

**Audit decision:** the pending matrix is useful without redesign. Its strongest possible conclusion is case-specific evidence about where precision is lost and whether a particular change survives generation and ranking. All nine views remain development data from five videos. Neither these counterexamples nor the saved results establish a universal usability threshold, a general kNN accuracy gain, or an end-to-end SVD speedup. Small corner errors remain diagnostics—not a definition of an accurate, usable court.

**Yes. At the same pinned revision, `7299ff3`, I found that GX0’s original suppression buckets cannot recover the saved bank-control precision—even with an oracle choosing their representatives.** This is an analytical constraint on the fixed-direction court diagnostic, not a prediction of visual usability. It materially changes how a partly successful or unsuccessful R arm should be interpreted.

### What I could inspect this time

I fully decoded **`gx0_control_measurements.json.gz` and `svd_fixed_measurements.json.gz`**, verified their complete compressed bytes against the repository’s Git blob hashes, and checked their saved homographies numerically. This overcomes the earlier limitation on the SVD summary.

I recovered only a prefix of **`evaluation/direction_records.json.gz`**, but that prefix contains **all 106 GX0 merged-line rows and its settings**. From those rows and the exact published construction, I reconstructed its complete **5,671-candidate bank**. All 16 original selected IDs, homogeneous working points, and support masks match the independently decoded SVD summary **exactly**. Both GX0 control candidates’ points, counts, and statuses also replay exactly. The complete three-case bank export and full individual SVD pair-fit records remain unread; I am not claiming to have inspected them.

The calculations below consist of original-selector replay, evaluation of existing saved fits, and analytical bounds. **No new court fit, matcher search, fragment-membership inference, or midpoint experiment was performed.**

## 1. GX0 has a quantified allocation bottleneck—not merely a missing candidate ID

**New mathematical deduction, evaluated on committed data.**

The reconstructed original allocation contains 16 leaders and 526 suppressed candidates: **542 candidates in the union of all allocated buckets**. Every unrefitted R representative must come from this union under the fixed specification. The other eligible candidates cannot be rescued by its representative rule.

I derived a lower bound that evaluates this entire union without fitting any new courts.

### A bound in image pixels, rather than homogeneous angular distance

Let \(a,b\) be the target endpoints of a court edge, and \(P=(P_x,P_y,w)\) a homogeneous vanishing point. Define

$$
\ell=[a_x,a_y,1]\times[b_x,b_y,1],\qquad
r_a=P_{xy}-wa,\qquad r_b=P_{xy}-wb.
$$

Then

$$
\boxed{
E(a,b;P)=
\frac{|\ell^\top P|}
{\max\!\left(\|r_a+r_b\|,\ \|r_a-r_b\|\right)}
}
$$

is the **minimum possible maximum endpoint displacement onto any image line through \(P\)**.

For a finite VP \(p\), this follows by minimizing

$$
\max\bigl(|n^\top(a-p)|,\ |n^\top(b-p)|\bigr)
$$

over unit line normals \(n\). A minimizing normal is perpendicular to either \((a-p)+(b-p)\) or \((a-p)-(b-p)\); taking the better value gives the formula. At infinity, orientation is fixed and the optimal line offset balances the two endpoint distances, yielding the same expression.

This handles finite and infinite VPs, including a VP at an endpoint. It is invariant to homogeneous scale/sign and behaves correctly under translations, rotations, and uniform image scaling.

For target corners \(c_0,c_1,c_2,c_3\) in the repository’s order, set

$$
L_x(P)=\max\{E(c_0,c_1;P),E(c_3,c_2;P)\},
$$

$$
L_y(P)=\max\{E(c_0,c_3;P),E(c_1,c_2;P)\}.
$$

Any homography preserving x-direction \(U\) and y-direction \(V\) must have maximum corner error at least

$$
\boxed{\max\{L_x(U),L_y(V)\}.}
$$

This is a necessary condition: it relaxes the coupling between edges, court dimensions, and camera/player constraints. Consequently, it is a **lower bound, not an attained fit or certified optimum**. It explicitly requires the saved comparator corners; it is **not a label-free selector objective**.

### Applied to the complete GX0 bucket union

Against approved candidate89, in **960 × 540 working pixels**, the executed calculation gives:

| Allowed directions                                                    | Analytical lower bound on maximum corner error |
| --------------------------------------------------------------------- | ---------------------------------------------: |
| Original 16 leaders                                                   |                                       6.022569 |
| **Every candidate in the original allocated buckets: 542 candidates** |                                   **3.503549** |
| Saved bank-control pair, IDs 1183 and 122                             |                                       0.423931 |

The existing bank-control homography achieves **0.509456** working pixels. That achieved value comes from the saved record, not a newly fitted court.

The decisive bucket-union constraint is the y-direction: its smallest \(L_y\) is **3.5035486225**, at candidate854. Even allowing arbitrary choices anywhere in the union—ignoring the one-representative-per-bucket restriction and the residual score—cannot beat that bound.

**Thus R’s unrefitted fixed-direction diagnostic cannot recover the documented 0.509-pixel bank-control precision. Changing only how representatives are scored inside those original buckets is insufficient.**

This strengthens the earlier observation that candidate122 is capped: it rules out the possibility that another candidate already inside those buckets supplies equivalent precision to that comparator.

### The allocation trace explains how this happens

The bounded survival check suggested previously is now completed for GX0:

* **Candidate122** supports eight merged rows. Its largest IoU with any leader is **\(7/9=0.777778\)**, below the strict suppression threshold. It therefore remains outside all allocated buckets. Its eight support rows are nevertheless all covered after the ninth leader.
* After **13 leaders, all 106 merged rows are covered**. The final three slots add zero coverage and are allocated through the remaining count/support tie-breaks.
* **Candidate1183** does enter a bucket: leader2152 suppresses it at IoU **\(15/18\)**. But on that leader’s fixed mask it ranks **27th of 74** by the proposed residual score: **0.555015**, versus the leader’s **0.432635**. R therefore cannot choose that exact candidate either.

These are executed selector calculations from the committed GX0 lines, with the named candidates’ saved statuses independently matched. They do not imply that a larger budget is necessary, or that the final zero-novelty leaders are geometrically worthless. They establish the narrower point: **complete observation coverage did not preserve the demonstrated direction precision**.

**Scope and falsification.** The bound assumes the exact saved target, coordinate scale, and preservation of the selected VP directions. A homography satisfying those conditions, using only the 542 candidates and achieving error below **3.503** working pixels, would contradict this result. A different baseline replay would challenge its numerical premises. Conversely, an R-generated court judged usable would not contradict it: usability is not defined by distance to candidate89, and the bound is not an acceptance threshold.

## 2. The committed GX0 fits contain a real score/geometry reversal on both directions

**New numerical evaluation of existing saved fits—not a synthetic example or new optimization.**

I compared these two committed homographies:

| Existing saved fit       | Direction IDs | Maximum corner error to candidate89 |
| ------------------------ | ------------- | ----------------------------------: |
| `best_retained_pair_fit` | 737, 4104     |                            6.798367 |
| `best_bank_fit`          | 1183, 122     |                            0.509456 |

Both errors reproduce from their saved homographies using the source’s float32 court-corner convention. The closer fit remains a diagnostic fit, not a newly generated or newly visually judged court.

Now evaluate both direction pairs on the **same original mask for each axis**, using exactly

$$
Q(v;S)=\operatorname{mean}_{i\in S}\min(\theta_i(v),1.5^\circ)^2.
$$

| Fixed original support mask | Original direction score | Bank-control direction score | Preferred by score |
| --------------------------- | -----------------------: | ---------------------------: | ------------------ |
| x: candidate737’s 17 rows   |             **0.476785** |                     0.506322 | Original           |
| y: candidate4104’s 8 rows   |             **0.675287** |                     0.742771 | Original           |

**The geometrically closer saved pair is worse under the capped-angle objective on both original masks.** This is an actual committed-case counterexample to treating improvement in that objective as guaranteed improvement in control-court geometry. The angle and score calculations follow the published source and fixed experiment definition.

### The reversal can be localized to particular merged rows

This is more informative than observing a poor correlation.

For the **y-direction**, the original and control candidates share seven supported rows. On those shared rows, their summed capped losses are:

$$
5.402298\quad\text{original},\qquad
3.692167\quad\text{control}.
$$

The control direction fits the shared support better. But original-only **merged row51** contributes approximately zero to the original candidate and **2.25** to the control candidate. That one row overturns the shared-support advantage.

For the **x-direction**, the 15 shared rows likewise favor the control direction:

$$
5.948954\quad\text{original},\qquad
4.107478\quad\text{control}.
$$

Original-only rows **31 and 61** add **2.156385** versus **4.5**, again reversing the ordering.

Rows31 and51 are also intersection-defining rows for original candidates737 and4104, respectively. These are **zero-based merged-line indices**, not raw-fragment IDs.

Evaluating on the control candidates’ own masks reverses both overall preferences again. That is evidence of **support-dependent ranking**, not evidence that those control-selected masks are correct memberships.

### What this narrows—and what it does not

The objective can prefer the worse court not because it fits the shared evidence better, but because the fixed original membership includes additional rows on which the alternative pays the capped penalty. This supplies a concrete mechanism for the concern that fixing a leader’s mask can preserve a preference for its own direction.

It does **not** establish why those particular rows disagree. Incorrect structures, orientation errors, anchoring, and correspondence remain possible explanations. Without actual contributing fragments and midpoints, I cannot attribute the disagreement to the original foot anchor or predict that M will repair it.

**Contrary evidence and falsification.** Another representative or pair could still make R useful; this is not a general performance estimate. A successful R result through another pair would limit the practical importance of this example, not erase it. The numerical claim is falsifiable by recomputing the published angle on these exact saved masks and finding a different ordering.

## 3. We can now bound how much a better court-fit solver could explain

**New analytical bounds on the saved direction sets, alongside independently checked saved outcomes.**

The fully decoded SVD summary provides all 16 original and refitted points for each difficult case. Therefore, even without the complete individual pair-fit records, the same bound can be minimized over **every ordered distinct pair in each saved set**.

All nine saved best-fit errors—three cases across original, fixed-support SVD, and control-selected SVD—reproject exactly from their saved homographies. Their direction columns also match their named saved groups up to homogeneous scale/sign.

The relevant original and fixed-support results are:

| Case                 | Fixed direction set  | New analytical lower bound | Existing saved best-fit error |
| -------------------- | -------------------- | -------------------------: | ----------------------------: |
| GX0                  | Original 16          |               **6.022569** |                      6.798367 |
| GX0                  | Fixed-support SVD 16 |               **3.268611** |                      3.961121 |
| GX5                  | Original 16          |               **9.728733** |                     23.800169 |
| GX5                  | Fixed-support SVD 16 |              **10.198761** |                     22.502448 |
| Amateur-2 frame28019 | Original 16          |               **2.590076** |                      2.972162 |
| Amateur-2 frame28019 | Fixed-support SVD 16 |               **1.217057** |                      1.626036 |

These are working-pixel distances to each case’s exact saved `control_source`, not interchangeable errors against one universal reference.

### The useful distinction is between unavoidable direction error and unresolved fitting error

For **original GX0**, even an ideal solver selecting any pair from the same 16 directions could improve the saved maximum error by **at most 0.775798 pixels**. For **original Amateur-2**, the corresponding maximum possible improvement is **0.382086 pixels**.

Thus their remaining discrepancy cannot primarily be dismissed as “perhaps the local control fitter simply missed a much better solution with these same directions.” The directions themselves impose most of the observed error.

For **GX5**, the bound is much looser. Its fixed-support SVD set cannot achieve error below **10.198761**, so a better solver alone cannot reproduce the saved control-selected SVD fit at **2.540327**. However, the gap between the lower bound and the saved **22.502448** is large. **Substantial fitting or pair-coupling uncertainty remains; the bound does not establish that 22.502 is close to optimal.** The control-selected groups remain label-guided alternatives, not known-correct memberships or a performance ceiling.

There is also a useful caution: on GX5, SVD improves the saved achieved error while the separate-edge lower bound gets worse, from 9.729 to 10.199. That does **not** prove the true optimum worsened. It shows why a loose necessary condition and an achieved joint fit must not be treated as interchangeable quality scores.

**Scope and falsification.** These bounds concern the supplied saved point sets. They do not constrain a different set produced by midpoint selection, representative selection followed by SVD, or support changes. Any homography preserving a pair from the relevant fixed set and beating its stated lower bound would invalidate the calculation. A better fit lying between the bound and the current saved error would be entirely consistent with it.

## What this changes in the later audit

The fixed experiment matrix need not change. The new results instead supply sharper interpretation constraints:

| Outcome in the specified experiments                                  | Interpretation now supported                                                                                                                                                                                            |
| --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Unrefitted R improves GX0, but remains above 3.503 working pixels** | Within-bucket precision selection may help, while the quantified allocation limitation remains. Improvement does not validate the score as a complete recovery objective.                                               |
| **Unrefitted R shows no change or regresses**                         | This cannot refute the broader direction-loss explanation: the allocated candidate union is already insufficient for the demonstrated bank-control precision. Score choice may still matter within that restricted set. |
| **Unrefitted M or MR gets below 3.503 on the same GX0 diagnostic**    | Its y-direction must come from outside the original bucket union. That would specifically demonstrate removal of the original allocation constraint—not merely better choice inside the old buckets.                    |
| **An SVD variant gets below the original bucket-union bound**         | This is allowed: SVD creates new VP locations. Each resulting point set needs its own bound; the original union’s bound does not apply.                                                                                 |
| **M/R improves diagnostic fits, generated courts, or ranked winners** | Those remain separate stages. Only actual generated and ranked results establish downstream recovery; MR and SVD remain diagnostic-only under the fixed specification.                                                  |

The matrix’s anchoring/provenance rules and generation scope remain exactly those in the evidence guide.

The one bounded survival check proposed in the earlier assessment has now been executed for GX0. No additional parameter search is needed to establish these findings. Actual fragment provenance remains necessary for interpreting the pending midpoint intervention.

### Reproducible audit package

The **[audit package](sandbox:/mnt/data/badminton_direction_audit_7299ff3.zip)** contains the recovered evidence, original-selector replay, row-level score decomposition, analytical bound implementation, and executed verification results. It runs locally with Python and NumPy:

```sh
python verify_audit.py
```

I also executed that command from a clean extraction of the package; its verification output matched exactly. The bound was independently checked on **80 synthetic configurations**, including finite/infinite VPs, scale/sign changes, coordinate similarities, and dense normal-angle comparisons. Those synthetic checks validate the formula, not detector performance.

The **[audit notes](sandbox:/mnt/data/direction_audit/README.md)** and **[verification results](sandbox:/mnt/data/direction_audit/verification_results.json)** are available separately.

**The contribution is a narrower diagnosis, not a new detector success claim:** GX0 loses demonstrable precision both at allocation and through support-dependent residual preference; unchanged-direction solver quality cannot explain most of the GX0 and Amateur-2 diagnostic gaps; GX5 still has a substantial unresolved fitting gap. None of this establishes a usability threshold, generalization beyond the development views, or an end-to-end SVD improvement.
