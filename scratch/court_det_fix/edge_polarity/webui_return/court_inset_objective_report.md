# Does the fixed-stripe court objective favour an inset boundary?

**Pinned source and results:** `ahalp90/badminton_cv_annotator@13f02fdbf8954ead15dbc7241a696a314473f9c5`  
**Conclusion:** with exact physical edge locations, correct marking/position identities, a correct 40 mm paint model and enough independent line constraints, the objective does **not** prefer an inset court. The true outside-boundary homography attains zero loss, which no inset can beat. Correct polarity is nevertheless insufficient in the present pipeline: it fixes only `position 1 ↔ 2`; it does not fix a true edge frozen as `position 0`, a displaced line-detector response, an initially biased sample subset, or a wrong effective stripe width.

The strongest local issue in SS03-34 is not an optimizer or weighting bug. Two retained fragments are assigned to the stripe **centre** despite large, fully sampled one-sided brightness contrasts. A minimal centre-to-edge counterfactual should be run before a wider fitting experiment. Its predicted effects differ by axis, so it can both confirm the left-inset mechanism and reject it as an explanation of the requested extra upward shift.

## 1. Objective reconstructed from the code

Court coordinates are metres, with `x` left-to-right and `y` far-to-near. The outside corners are `(0,0)`, `(6.10,0)`, `(6.10,13.40)`, `(0,13.40)`; see [`src/courtkeynet/court_corners.py`](https://github.com/ahalp90/badminton_cv_annotator/blob/13f02fdbf8954ead15dbc7241a696a314473f9c5/src/courtkeynet/court_corners.py).

[`paint_geometry.py::positioned_segments`](https://github.com/ahalp90/badminton_cv_annotator/blob/13f02fdbf8954ead15dbc7241a696a314473f9c5/experiments/annotator/independent_court/paint_geometry.py) uses offsets

\[
(0,-0.02,+0.02)\ \mathrm{m}
\]

from each 40 mm stripe centre. Therefore the model really does contain the outside boundaries: left doubles `position 1` is `x=0`, right doubles `position 2` is `x=6.10`, far baseline `position 1` is `y=0`, and near baseline `position 2` is `y=13.40`.

[`fixed_stripe_refit.py::prepare`](https://github.com/ahalp90/badminton_cv_annotator/blob/13f02fdbf8954ead15dbc7241a696a314473f9c5/experiments/annotator/independent_court/fixed_stripe_refit.py) freezes, from the parent homography:

- fragments whose assignment strength is at least `PRESENT_SUPPORT = 0.55`;
- each fragment's marking and centre/edge position;
- the nearest finite interval for each sample;
- only samples within `SUPPORT_DISTANCE_PX = 5` of that projected interval;
- sample weights and fragment/sample IDs.

For retained sample `j`, pixel point `p_j`, assigned finite projected segment `S_j(H)`, and the raw weight `u_j`, [`fixed_stripe_refit.py::residual` and `::refine`](https://github.com/ahalp90/badminton_cv_annotator/blob/13f02fdbf8954ead15dbc7241a696a314473f9c5/experiments/annotator/independent_court/fixed_stripe_refit.py) minimize exactly

\[
L(H)=\sum_j \frac{u_j}{\sum_k u_k}\,d\!\left(p_j,S_j(H)\right)^2
\]

in working-pixel squared units. The image normalization cancels the `image_scale` factor. Distance is to the closest point on the **finite** segment, so an endpoint can also contribute an along-line residual.

The weighting is internally consistent. [`stripe_observations.py::fragment_weights`](https://github.com/ahalp90/badminton_cv_annotator/blob/13f02fdbf8954ead15dbc7241a696a314473f9c5/experiments/annotator/independent_court/stripe_observations.py) apportions an observed group's union length over its fragments. `prepare` then gives each selected sample `fragment_weight / selected_sample_count`; after squaring the square-root weights in `refine`, the total objective mass of a fragment is its original fragment weight. There is no hidden area, corner-offset, or “small court” term.

## 2. When the outer boundary is recovered

Let `H*` be the true planar homography. If every retained point lies exactly on its correctly named and correctly positioned physical segment, the 40 mm model is correct, and the finite interval contains the corresponding physical point, then every residual at `H*` is zero. Since `L(H) ≥ 0`, `H*` is a global minimizer. Positive weights cannot make an inset solution better than zero.

That statement is about attaining the minimum, not uniqueness. The data determine the outer boundary only when the frozen line incidences provide eight independent local homography constraints and do not admit another zero-loss projective mapping. The refitter checks the fitted finite-difference Jacobian for rank eight, but this is a local numerical check, not a global uniqueness proof. The saved SS03-34 baseline and polarity fits both report rank eight with condition numbers about `40.9`, so the observed movement is not explained by an obvious local rank deficiency.

Thus, under the ideal model the objective is sound. An inset fit means at least one ideal premise is false or the observations do not uniquely determine the court; it does not by itself show an optimizer defect.

## 3. What can still favour an inset

### Centre assignments: yes, even at zero loss

[`stripe_observations.py::resolve_fragments`](https://github.com/ahalp90/badminton_cv_annotator/blob/13f02fdbf8954ead15dbc7241a696a314473f9c5/experiments/annotator/independent_court/stripe_observations.py) chooses the maximum reverse support over all marking/position pairs, including `position 0`. `RESOLVABLE_WIDTH_PX = 4` is used only for the paired-edge diagnostic; it does not prevent a centre assignment when the projected stripe is narrower than four pixels.

A one-dimensional specialization of the actual distance objective gives a counterexample. Let the true image map be `u = s x + b`, the left-singles physical inner edge be `t=0.50 m`, but freeze it as its centre `m=0.48 m`; keep the right outside boundary `W=6.10 m` correct. The exact zero-loss affine fit has

\[
\hat b-b=s\,(t-m)\frac{W}{W-m}.
\]

For `s=60 px/m`, this is a **1.302491 px inset with zero loss**. Correcting `m` to `t` recovers `b` exactly. This is not an optimizer bug: the wrong frozen interpretation defines a different, internally consistent data set. The accompanying script executes and asserts this result.

### Frozen selection: not intrinsically, but it can lock in biased data

With exact correctly interpreted points, freezing membership cannot turn the zero-loss true homography into a worse solution. It can, however, omit accurate outer-edge samples or retain displaced/incorrect samples because membership is chosen under the parent fit and a 5 px gate. The refit cannot recover evidence that was excluded, and it does not revisit a wrong nearest interval. Therefore selection is a possible observation bias, not an independent geometric prior for an inset.

### Weights: not with consistent data; yes in a compromise

Any positive weights preserve a zero-loss true solution. When measurements, positions or stripe width are inconsistent, weights decide which residuals dominate and can therefore favour an inset compromise. The code's weighting itself does not create a directional bias; the relevant question is whether high-weight fragments are systematically inward-displaced.

### Stripe width and measured fragment position: yes if the effective offset is wrong

A correct inner-edge observation is sufficient to recover the outside boundary because the model knows the 40 mm offset. But polarity only says which side is bright. It does not say that the fitted line coordinate is exactly the physical paint transition. A detector response displaced by blur, thresholding, crossings or line fitting acts like `t-m = δ` in the formula above. Likewise, an incorrect effective stripe width produces approximately the projected width error, with possible extrapolation amplification. Symmetric blur of an isolated ideal step need not shift its midpoint, so this mechanism requires actual profile measurements rather than assumption.

## 4. SS03-34 evidence and the discriminating local test

The saved [`scene34_smoke.json.gz`](https://github.com/ahalp90/badminton_cv_annotator/blob/13f02fdbf8954ead15dbc7241a696a314473f9c5/scratch/court_det_fix/edge_polarity/scene34_smoke.json.gz) contains two retained `position 0` fragments that [`run_probe.py::relabel`](https://github.com/ahalp90/badminton_cv_annotator/blob/13f02fdbf8954ead15dbc7241a696a314473f9c5/scratch/court_det_fix/edge_polarity/run_probe.py) deliberately leaves unchanged:

| Fragment | Marking | Position | contrast at ±1 px | contrast at ±2 px | valid pairs |
|---:|---|---:|---:|---:|---:|
| 111 | left singles | 0 | −69.999 | −90.258 | 14 / 14 |
| 179 | near short service | 0 | +80.877 | +32.955 | 14 / 14 |

These are much more edge-like than centre-like, though contrast alone is not proof. Under the saved polarity homography, the projected full 40 mm width is only `2.128–3.609 px` along the left-singles line and `0.906–0.981 px` along the near-short line—both below the 4 px resolvability threshold. A centre winner from pure geometry is therefore unsurprising.

The saved fitted objectives are almost tied: baseline `0.5272225 px²` versus polarity `0.5288148 px²`, weighted RMS `0.7261` versus `0.7272 px`. The corrected physical interpretation is not strongly disfavoured by the fitting residual; the measurements simply support both interpretations nearly equally. The separate whole-court paint score is not this objective.

### Recommended single local test

On **SS03-34 only**, start from the existing polarity constraints and evaluate the two centre fragments one at a time, then together:

1. For a `position 0` fragment with strong contrast, call the existing `expected_bright_side` logic for candidate positions `1` and `2` at that fragment's midpoint and choose the candidate whose expected sign matches the observed sign. Do not use reference labels.
2. Change only that position. Keep marking identity, selected points, interval IDs, sample IDs and weights fixed.
3. Run `111 only`, `179 only`, and `both`; record the upper-left movement, objective change, each fragment's selected sample count/total weight, and its signed normal residual before and after.

The existing signs make `111: 0→2` and `179: 0→1` the expected decisions; the test should compute rather than assume them.

A two-line projective proxy using the saved polarity corners predicts:

| Counterfactual | predicted additional upper-left movement |
|---|---:|
| fragment 111 only | about **(−1.154, +0.002) px** |
| fragment 179 only | about **(−0.287, +0.693) px** |
| both | about **(−1.440, +0.695) px** |

These are undiluted geometric effects with the opposite outer boundary held fixed, not predictions of the full weighted 34-fragment optimizer. Their signs are the important discriminator. Fragment 111 can plausibly explain another roughly one working pixel of the **left** inset. Fragment 179, under the likely `0→1` identity, predicts a **downward**, not upward, correction of the far-left corner. Therefore centre ambiguity is a plausible explanation for much of the remaining x error but is not a good explanation of the requested additional upward shift.

Reject centre assignment as the dominant left-inset mechanism if the `111 only` refit moves the upper-left less than roughly `0.3 px` left (one quarter of the proxy), moves it right, or has negligible objective leverage. If `179 only` moves the corner materially upward, the simple separable proxy is rejected and the coupled constraints/weights must be inspected. If the predicted signs hold, the next measurement should be the signed normal location of the retained horizontal fragments relative to their actual paint transitions; another label sweep would not address the remaining y discrepancy.

## 5. What is established, and what still requires real fragments

Established by code and the synthetic calculation:

- the objective has no intrinsic inset or small-court preference;
- correct exact edge assignments make the outside-boundary homography a zero-loss solution;
- positive weights cannot defeat that zero-loss solution;
- a true edge frozen as a centre can instead make an inset homography attain zero loss;
- the existing polarity probe cannot correct that case;
- SS03-34 contains two strong, fully sampled centre-assigned candidates in sub-resolvable projected stripes.

Still requiring the frozen real observations:

- the exact candidate edge chosen for fragment 111 from its own normal orientation;
- the selected sample counts and objective weights of fragments 111 and 179;
- their signed point-to-projected-edge residuals and actual photometric transition locations;
- whether other high-weight fragments are displaced inward by detector localization or an effective stripe-width error.

If the centre counterfactual does not have the predicted x effect, the assumption most in need of checking is **“the retained fragment coordinate is the physical paint boundary”**, followed by the parent-dependent membership/interval selection—not the least-squares solver.

## Checks actually run

- Read `.github/AGENTS.md` and the requested source files at the pinned commit.
- Read the additional pinned sources `detector.py`, `junction_observations.py`, `src/courtkeynet/court_corners.py`, and `scratch/court_det_fix/w5_holistic/verifier.py` to resolve projection, court constants, support threshold, weights and assignment creation.
- Decoded the pinned SS03-34 smoke result (Git blob `6243bab9f86bc45e38fc083c21ee55a3407d7339`) and checked its fragment rows, objectives, corners, Jacobian rank and condition.
- Recomputed projected stripe-width ranges from the saved polarity corners.
- Ran the accompanying NumPy script and its assertions; it reproduces the zero-loss 1.302491 px inset and the SS03-34 proxy movements above.
- Inspected `test_probe.py`: its ten saved tests cover polarity orientation, endpoint reversal, unchanged sample arrays and masked evidence, but not `position 0 → edge` resolution or subpixel boundary localization.

No full detector, image review, repository clone, parameter sweep, or production edit was run.
