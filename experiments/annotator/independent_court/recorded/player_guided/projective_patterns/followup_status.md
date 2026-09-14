# Progress through the court-detection follow-ups

The work follows the projective-pattern branch of the proposal-construction
plan. Candidate-retention and paint-ranking experiments were brought forward
when measured failures justified them. This is progress within the conditional
plan, rather than completion of five sequential implementation stages.

## Next experiments, in order

First check how line agreement is measured, then use that evidence to improve
which directions survive. These are untested hypotheses within direction
retention/refinement. The SVD result below motivates this order but does not
establish the cause of the remaining errors.

1. **Check agreement where the line is visible.** The current angle is measured
   from the line's closest point to the image centre, which can lie far from the
   observed fragment. Compare it with a test anchored at the fragment's midpoint.
   First confirm which fragments belong to each merged line; compare the tests
   on the same evidence and candidate directions. Check short fragments and
   vanishing points close to a fragment. The hypothesis is that the measurement
   location distorts group membership; a midpoint test may also have weaknesses.
2. **Separate group coverage from direction precision.** Use additional line
   coverage to decide which groups deserve space, then geometric agreement to
   choose a precise direction within each group. Preserve competing alternatives.
   The hypothesis is that a direction explaining more lines can displace a more
   accurate one. Test this after checking the agreement measure, so a new ranking
   does not inherit the same measurement problem. A selection-only change remains
   an alternative to refitting; SVD is available as a cheap fitting step.
3. **Check court recovery before changing court ranking or reducing the budget.** Start
   with saved diagnostics on GX0, GX5 and Amateur-2, changing one rule at a time.
   The two specified individual changes then go through the unchanged matcher
   on all nine varied views. Keep the search budget fixed and use approved courts
   only to evaluate results. Court ranking and acceptance remain separate checks afterwards.

A bounded execution specification now fixes the comparisons before any new results:
replay actual fragment membership, compare the original anchor with the projected
midpoint of the longest contributing fragment, then compare greedy leaders with
representatives chosen by agreement inside each leader's suppression group.
Both individual changes reach the unchanged matcher on all nine views regardless
of early diagnostic errors. Their combination and fixed-support SVD variants stay
in the cheap diagnostic comparison. This avoids choosing trials from control-fit
results. The specification is prepared; these experiments have not run.

If the midpoint comparison gives no useful improvement, retain the existing measure
for later work. Interpretation follows an independent evidence audit and any needed
visual inspection. No new annotation or production change is part of this series.

| Planned stage | Status after these experiments | What remains |
|---|---|---|
| 1. Proposal construction | Direction pruning, synthetic spacing tests, supplied-direction matching and automatic-direction matching are complete as experiments. Useful geometry is recoverable, but the automatic selector loses precise directions. | Redesign direction retention or refinement. General graph search remains optional and untested. |
| 2. Candidate retention and evidence | Camera-before-cap and removal of the global cap were compared on all nine views. Original gate outcomes were preserved. Global retention alone does not explain GX5. | Keep direction loss separate from per-pair loss. A partial-view evidence rule has not been validated. |
| 3. Ranking | Existing paint profiles were compared with line support on identical pools. They repair Amateur-2 ranking in the supplied-direction test but worsen GX0; larger automatic pools expose false winners. | Test ranking against those wrong-court/background examples. A general appearance rule is unresolved. |
| 4. Multiple frames | No new controlled temporal-generation or score-aggregation comparison. The existing ShuttleSet composites are input data, not this experiment. | Activate only after stable-view and complementary-frame evidence is established. |
| 5. Acceptance and replacement | Original gates remain in place. No replacement acceptance rule or production integration. | First obtain reliable automatic generation and selection, then measure coverage and wrong emissions on positive and negative cases. |

## Candidate experiment: SVD direction refinement

The saved-record feasibility check is complete. Singular value decomposition
(SVD) improves the best diagnostic fit when applied to the existing support groups,
with only a small improvement on GX5. It is cheap, but it does not recover the
precision previously demonstrated by selected observed-bank directions.

Each case keeps 16 directions and evaluates all 240 ordered pairs before and after
refitting. Line-group membership stays fixed. SVD uses centred/isotropic coordinates
and unit 2D line normals. Directions use image-derived groups only; the court-fit
diagnostic uses the existing controls, including approved candidate89 for GX0.

| Case | Original directions | Fixed-group SVD | SVD with control-selected groups |
| --- | ---: | ---: | ---: |
| GX frame0 | 6.798 | 3.961 | 0.952 |
| GX frame5 | 23.800 | 22.502 | 2.540 |
| Amateur-2 frame28019 | 2.972 | 1.626 | 1.441 |

Values are maximum corner distances to each case's control, in 960 × 540 pixels.
They are local diagnostic fits, not generated courts or certified optima. The
last column uses separate label-guided groups; those groups are not known correct
memberships or a performance ceiling. Their better fits show membership sensitivity.

All 1,446 pair fits returned finite results; 1,445 converged, including every best
fit. No directions collapsed to duplicates. The 16 SVD refits took 1.1–1.9 ms per
case; all three numerical diagnostics took 5.8 seconds in total. These single-run
timings do not establish an end-to-end speedup. Five synthetic checks, scoped
lint/types, input-provenance checks and the GX0 baseline replay passed (exit 0).
[Measurements](svd_fixed_measurements.json.gz) preserve the exact comparators,
winning fits, conditioning diagnostics and prior bank-fit results.

The sequence above tests line agreement before changing group selection. Keep
fitting and support re-selection separate, and preserve the existing usable cases.
No matcher run, new visual judgement or acceptance change followed this diagnostic.

The cheap SVD fits also suggest a later option: fit a larger pool of proposed
line groups, then pass a better shortlist to the expensive court matcher. This
could widen the early search while keeping the matcher budget fixed. It requires
a reliable way to choose groups and judge direction quality first. Sending more
directions straight to the matcher increases pair counts quickly: 16 directions
give 240 ordered pairs; 32 give 992. The timings do not establish the cost of
building a larger pool or selecting from it. Activate this option only after the
agreement and selection checks; no larger-pool experiment or overall speedup has
been demonstrated.

The SVD result narrows the possible role of k-nearest-neighbour (kNN) lookup.
Fitting the 16 directions is already cheap; choosing their supporting lines and
retaining precise directions remain unresolved. If a later selection method needs
to examine nearby direction hypotheses, kNN could speed up finding those neighbours.
It would not decide which neighbour gives the correct court. First define useful
proximity for court geometry: treating nearby directions as interchangeable could
again discard differences that matter to the fit.

The order is therefore: check line agreement, improve group/direction selection,
measure its runtime, then consider kNN if neighbour lookup is expensive. No separate
kNN experiment is warranted by the current evidence. Neither an accuracy gain nor
a substantial compute saving from kNN has been demonstrated. A smaller search
budget remains untested; first show that useful solutions survive automatic selection.

## How much closer is the detector?

The main advance is locating the bottleneck. The matcher produces usable courts
in all eight supplied-direction views that received visual feedback. The ninth
view's new candidates remain unjudged. Those results test matching capacity;
they do not establish automatic direction recovery.

The automatic gallery has now been inspected on all nine views. Six have a usable
winner under at least one ranking, chosen after inspection. GX0, GX5 and
Amateur-2 frame28019 have poor displayed winners and reference-selected diagnostics.
The [panel judgements](automatic_axes_visual_judgements.md) show why low corner
error and either ranking alone do not establish usability. This feedback does
not judge the earlier supplied-direction scene16 candidates.

Three difficult automatic cases now have a more precise diagnosis. Their full
image-derived direction banks contain useful directions that selection discards.
Feeding those observed directions through the unchanged matcher produces close
courts. GX5's selected line/paint court is 10.58 pixels from the manual reference;
Amateur-2 frame28019's paint winner is 7.27 pixels away. These are maximum corner
errors at 1280 × 720, from label-guided controls, not automatic success rates.

A subsequent GX0 control supplies discarded observed directions to the same
matcher. The user judged its line winner essentially perfect and its paint winner
perfect. Both are essentially ideal. This resolves the GX0 control visual check
and strengthens the direction-precision diagnosis. The directions were selected
using the approved court, so this is not an automatic improvement.

The next change can therefore target direction precision rather than replacing
the spacing matcher or repeating broad score sweeps. Ranking and acceptance
remain separate unresolved tasks. There is no defensible percentage-complete
estimate or established end-to-end improvement rate.

No new annotations are needed to expose the current failures. Two additional
hard holdout videos remain a final-readiness option, after development succeeds.
See the [automatic results](automatic_axes_results.md) for the decisive evidence.
