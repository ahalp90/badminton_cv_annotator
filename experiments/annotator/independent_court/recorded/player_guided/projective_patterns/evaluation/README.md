# Evidence for independent geometric assessment

This supplement makes the direction and ranking questions assessable from committed
records. It exports existing data and exact method excerpts. It contains no new
experiment results, new annotations or additional images. The scientific basis is
checkpoint `b90518c` on `fix/court-det`.

Read the relevant question below, then open its named evidence. The pending experiment
matrix is fixed before results arrive. Independent assessments should inform the later
audit; they do not change those experiments during execution.

## Direction accuracy and selection

**Question:** does agreement with observed lines measure the precision needed for an
accurate projected court, and does the proposed selector preserve useful alternatives?

[Direction records](direction_records.json.gz) contain three difficult cases: GX frame0,
GX frame5 and Amateur-2 frame28019. Each case includes the saved estimator settings,
merged line coefficients, full bank IDs/counts/statuses, 16 selected points and support
masks, and the complete previous fixed-support SVD diagnostic. That diagnostic retains
every attempted pair fit, including convergence status and failures. Source-record MD5s
identify the exports; fields were copied without recomputing geometry.

The estimator's `direction_lines` are working-pixel homogeneous lines. Row-vector lines
transform to normalised coordinates with `lines @ normalised_to_working`. Normalised
homogeneous points transform to working pixels with `points @ normalised_to_working.T`.
The transform centres the image and scales both axes by its diagonal. Point scale/sign
is arbitrary. Bank candidates are pairwise line intersections followed by explicit
infinity points; the exact construction appears in [method excerpts](method_excerpts.md).
Candidate IDs refer to that construction before degenerate candidates are removed.
Support arrays align with the surviving `candidate_ids`, not the raw ID as an index.
Selected masks index the saved merged-line rows; they are not fragment memberships.
Court coordinates are metres: x runs from left to right (0 to 6.10), y from far to
near baseline (0 to 13.40). Corner slots are far-left, far-right, near-right, near-left.
The excerpts include the exact court corners, twelve painted intervals and their
marking mapping. Homogeneous projection uses `H @ [x,y,1]`, divided by its third coordinate.

**Actual fragment-to-merged-line membership and fragment midpoint coordinates are not
included.** Their exact replay is pending. The old `compatible_raw_ids` field records
post-fit compatibility, not merge assignments; it is deliberately excluded from the
numeric export. The data support analysis of the original angle test, bank selection
and fixed-support SVD. They cannot establish the real-frame effect of midpoint anchors.

The SVD records use the exact control named by each case's `control_source`. GX0 uses
approved generated candidate89. These controls are used in diagnostic fitting, not
in automatic selection. `control_selected_svd` is label-guided and does not establish
known correct membership or a performance ceiling. `control_fit` minimises coordinate
residuals with a bounded local solver; its reported maximum corner error is neither
the optimisation objective nor a certified optimum. See [the SVD result](../followup_status.md#candidate-experiment-svd-direction-refinement).

### Fixed, unrun comparison

The baseline angle is measured at the normalised line's closest point to the image
centre. The proposed alternative uses the longest clipped fragment that actually
contributed to the merged line. Equal lengths are resolved by canonical observation
index. Project that fragment's working midpoint `m` onto merged line `(n,c)`:
`a = m - n*(dot(n,m)+c)/dot(n,n)`, then transform `a` to normalised coordinates.
For normalised candidate `v`, use ray `v.xy - v.w*a` against tangent `(line.b,-line.a)`.
Keep the original absolute cross/dot angle and 90-degree undefined-ray convention.
This changes the anchor, not the observed line's orientation.

| Arm | Angle anchor | Representative |
| --- | --- | --- |
| B | Original foot | Original greedy coverage leader |
| M | Projected fragment midpoint | Original greedy coverage leader |
| R | Original foot | Minimum residual within the leader's suppression bucket |
| MR | Projected fragment midpoint | Minimum residual within that bucket |

For either anchor, preserve the original greedy allocation of at most 16 leaders.
A bucket contains its leader and the currently eligible candidates it suppresses at
support-mask IoU > 0.8. Buckets are disjoint. Only leaders update coverage and subsequent
allocation. R/MR choose the candidate minimising `mean(min(angle,1.5)**2)` on the fixed
leader mask, breaking ties by original candidate ID. The chosen candidate's own mask
supplies its subsequent SVD fit. No cross-bucket backfill or support reselection occurs.
The angle limit is 1.5 degrees and eligible candidates support at least two merged lines.
These anchor and score choices are hypotheses, not validated objectives.

All four arms and their separate fixed-support SVD variants receive ordered-pair
control-fit diagnostics on all nine development views. Only M and R reach the unchanged
matcher: 18 new case-arms, with 36 subsequent camera-first/all-camera rescoring case-arms.
They proceed regardless of early error improvements. MR and SVD remain diagnostic-only.
Existing budgets, camera checks, player checks and rankings stay fixed. The experiment
must first verify exact merge provenance and baseline replay. No result of this matrix
is available in this supplement.

## Ranking and visibility

**Question:** what observable evidence could distinguish a correct partial court from
a strong match to background structures, without discarding the existing good fits?

[Ranking records](ranking_records.json.gz) export both automatic winners from each of
the nine all-camera pools, plus both visually approved GX0 observed-bank control winners.
Equal line/paint winners share one entry. Stable identity is `(case_id, population,
candidate_id)`; IDs from different populations are not interchangeable. Each population
preserves native dimensions, working size and source-record MD5. Corners are native pixels;
homographies map court metres to working pixels. Entries retain the full saved stripe,
paint-profile and gate evidence. These are selected examples, not the full candidate
population or a new set of independently labelled negatives.

Read [automatic results](../automatic_axes_results.md),
[exact visual judgements](../automatic_axes_visual_judgements.md), and the corresponding
[gallery](../automatic_axes_visual_check.html). The separate
[GX0 control](../gx0_control_visual_check.html) has two essentially ideal winners,
with [measurements](../gx0_control_measurements.json.gz) preserving their approval.
The galleries already share [images](../images/); do not duplicate them.

Two decisive false paint winners are Amateur-2 frame28019 `184:4123` and ShuttleSet03
scene19 `165:6702`, both in `automatic_all_camera`. The former passes profiles for all
11 markings. The latter scores 1.0 from five available markings, with six unavailable.
Both fail the original floor gate. However, approved GX0 candidate89 and both ideal
GX0 bank-control winners also fail that gate. Counts of visible markings and the old
floor rejection therefore each face a concrete contrary example.

The exact profile and winner-selection functions are in [method excerpts](method_excerpts.md).
Visibility here requires a projected interval to clip to a positive span of at least
12 working pixels inside the image under `_visible_samples`. A shorter intersecting
interval is unavailable too. This does not establish that paint is unoccluded or
distinguishable. Twelve intervals
map to eleven markings because the centre marking has two intervals. Missing profiles
are excluded from the current score's mean. Line/paint winners both require a finite
camera error <= 0.1 and an available profile. Their scores are descriptive, not calibrated
probabilities. Floor outcomes are recorded but do not determine these diagnostic winners.
The supplement does not include the full floor-scoring implementation or all rejected
candidates. Claims about their mechanisms need additional evidence.

## Interpretation limits

These are varied development samples from five videos, with no designated holdouts.
Use the existing judgements; do not invent a pixel or paint-edge acceptance threshold.
Maximum corner distance can miss internal marking errors and disagree with visual quality.
The ordered-pair diagnostic uses fixed corner correspondence; the generated-court metric
allows the existing 180-degree relabelling. Both have exact excerpts. Display errors use
1280 × 720, while direction-fit diagnostics use 960 × 540; do not mix these scales.

Good results with control-selected directions establish matching capacity, not automatic
recovery. The automatic six-view usable union requires a human choice between rankings.
Apparent paint-line bowing has no verified cause. kNN and a larger early fitting pool
remain conditional ideas; neither is part of the fixed experiment matrix.
