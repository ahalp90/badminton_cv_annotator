# High-impact follow-up: an inset parent can trap a true edge at `centre`

**Pinned revision:** `ahalp90/badminton_cv_annotator@13f02fdbf8954ead15dbc7241a696a314473f9c5`

## Finding

The least-squares objective remains direction-neutral, but the **assignment → freeze → refit pipeline has a precise inset capture band**. A physically correct edge can be geometrically assigned to `position 0` (`centre`) when the parent court is moderately displaced. That centre assignment then receives very strong geometric support, passes the 5 px sample gate, and is invisible to the polarity probe because [`run_probe.py::relabel`](https://github.com/ahalp90/badminton_cv_annotator/blob/13f02fdbf8954ead15dbc7241a696a314473f9c5/scratch/court_det_fix/edge_polarity/run_probe.py) swaps only positions 1 and 2.

This gives a concrete way for the initial inset to reproduce itself without an optimiser bug, bad weights, or an incorrect paint-side label.

## 1. Local derivation

Consider one interior sample of a stripe. Let:

- `q > 0` be the modelled projected distance, in working pixels, from stripe centre to either edge;
- `δ` be the parent projection error along the positive stripe normal;
- the observed fragment be the true positive edge.

Relative to the true stripe centre, the observed point is at `q`. The parent's three projected targets are

\[
\delta,\qquad \delta-q,\qquad \delta+q
\]

for centre, negative edge and positive edge. Their distances from the observation are therefore

\[
d_0=|q-\delta|,\qquad d_1=|2q-\delta|,\qquad d_2=|\delta|.
\]

Because [`stripe_observations.py::resolve_fragments`](https://github.com/ahalp90/badminton_cv_annotator/blob/13f02fdbf8954ead15dbc7241a696a314473f9c5/experiments/annotator/independent_court/stripe_observations.py) maximises a Gaussian that is monotone in distance, centre wins in the straight, interior-fragment specialization when

\[
\frac q2 < \delta < \frac{3q}{2}.
\]

At exact equality the centre and one edge tie; `np.argmax` lists centre first, although floating-point details can perturb an exact tie.

Inside that capture band,

\[
d_0\leq \frac q2.
\]

Thus the wrong centre interpretation looks *more*, not less, confident as the parent moves over the true edge. It is also comfortably retained by [`fixed_stripe_refit.py::prepare`](https://github.com/ahalp90/badminton_cv_annotator/blob/13f02fdbf8954ead15dbc7241a696a314473f9c5/experiments/annotator/independent_court/fixed_stripe_refit.py): its point-to-centre residual is far below `SUPPORT_DISTANCE_PX = 5`.

With a modelled half-width `q_m`, an effective observed edge offset `q_a` and parent shift `δ`, the general centre condition is

\[
|q_a-\delta|\leq \frac{q_m}{2}.
\]

This separates two mechanisms cleanly:

- with correct localization, `q_a=q_m`, the initial parent creates the capture band above;
- even with `δ=0`, a stripe-width or detector-localization mismatch can select centre when `q_a\leq q_m/2`.

Weights play no role in the discrete choice. Once the interpretation is frozen, weights only determine how strongly this wrong target moves the compromise.

## 2. SS03-34 numbers

Using the saved polarity-fit homography, the projected centre-to-edge distances are:

| Marking | `q` range | Parent-shift capture band at each location | Largest captured centre residual | Minimum ideal Gaussian reverse support |
|---|---:|---:|---:|---:|
| left singles | 1.064–1.805 px | `q/2` to `3q/2` = 0.532–2.707 px over the line | 0.902 px | 0.903 |
| near short service | 0.453–0.490 px | 0.226–0.735 px over the line | 0.245 px | 0.993 |

The reverse-support lower bounds use the code's `DISTANCE_SIGMA_PX = 2`; both are well above `PRESENT_SUPPORT = 0.55`. These are local straight-line bounds: finite endpoints and actual sample locations still need the exact frozen constraints, but neither the strength gate nor the 5 px selection gate protects against this mechanism.

### The polarity-compatible edges can now be identified

The earlier report treated the exact edge choice as still open. The saved marking identities, canonical endpoint convention and homography make the sign robust enough to resolve it without reference labels:

- [`assignment.py::prepare_observations`](https://github.com/ahalp90/badminton_cv_annotator/blob/13f02fdbf8954ead15dbc7241a696a314473f9c5/experiments/annotator/independent_court/assignment.py) canonically orders fragment endpoints. Along the full left-singles and near-short markings, the projected positive court axis has positive dot product with that canonical fragment normal. The sign remains positive with at least **62.45°** margin after the code's 5° orientation tolerance.
- Fragment **111** has contrast `−69.999`, so [`run_probe.py::expected_bright_side`](https://github.com/ahalp90/badminton_cv_annotator/blob/13f02fdbf8954ead15dbc7241a696a314473f9c5/scratch/court_det_fix/edge_polarity/run_probe.py) selects **position 2, the positive/inner left-singles edge**.
- Fragment **179** has contrast `+80.877`, so it selects **position 1, the negative/far near-short edge**.

Fragment 111 is therefore the centre mistake with the correct sign to sustain a **left inset**. Fragment 179 has the opposite diagnostic role in image `y`: the existing two-line proxy predicts that correcting it moves the upper-left corner down, not toward the requested additional upward shift.

## 3. Why the mixed polarity result is now unsurprising

The analytic example has a modelled centre-to-edge distance of 1.2 px:

| Parent inward shift | Geometric assignment | After polarity | Fitted left-boundary inset |
|---:|---|---|---:|
| 0.5 px | positive edge | positive edge | 0 px |
| 1.0 px | centre | centre | **1.302491 px**, with zero loss |
| 2.0 px | negative edge | positive edge | 0 px |

A large enough error can choose the *opposite* edge, which polarity repairs. A moderate error chooses centre, which polarity deliberately leaves alone. That predicts partial correction rather than monotonic success and is consistent with the saved probe's mixed movements.

The executable calculation is [`court_inset_assignment_capture.py`](../../archive/20260927_code/edge_polarity/webui_return/court_inset_assignment_capture.py).

## 4. Single next local test

Run only **SS03-34 fragment 111: position `0 → 2`** on top of the existing polarity constraints.

The supplied [`scene34_fragment111_counterfactual.py`](../../archive/20260927_code/edge_polarity/webui_return/scene34_fragment111_counterfactual.py) reconstructs the exact saved subset in the original evidence checkout, chooses position 2 from brightness and geometry without reference labels, preserves points, marking, interval IDs, sample IDs and weights, and fits only the polarity arm versus this one-fragment change.

Predictions:

- the upper-left corner must move **left** if fragment 111 materially sustains the inset;
- the two-line projective proxy is **(−1.154, +0.002) working px**;
- it should not supply the remaining requested upward movement.

A converged exact refit with non-leftward movement rejects this mechanism outright. A left movement smaller than about **0.1 working px** would reject it as a material explanation of the roughly 1.9 px residual left discrepancy, even if the physical edge identity is still wrong. A movement near −1 px would confirm that centre capture explains a substantial part of the remaining left inset and would move the next measurement to raw boundary localization rather than solver changes.

## 5. Checks run and remaining gap

Checks run here:

- derived the three-position Voronoi/capture condition directly from the code's reverse-support objective;
- verified the orientation sign over the full two relevant markings, including the 5° compatibility allowance;
- recomputed the SS03-34 projected half-width ranges from the saved polarity corners;
- calculated the support and 5 px retention bounds;
- ran and byte-compiled the accompanying analytic script;
- syntax-checked the exact one-fragment local test.

The exact case-record path used by `run_probe.py` is not committed at the pinned revision, and `results.json.gz` omits the frozen points, intervals and weights. Consequently the exact fragment-111 counterfactual cannot be executed faithfully in this web session. The local test fails explicitly when that record is absent rather than silently reconstructing a different subset.
