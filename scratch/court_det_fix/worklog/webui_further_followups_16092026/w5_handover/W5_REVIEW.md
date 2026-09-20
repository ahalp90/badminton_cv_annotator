# W5 — why the next detector should judge the whole court

The court-detection branch has done enough useful decomposition that the next step no longer looks like “find one better line heuristic”. The recurring failure is more specific:

> a useful court can be lost before ranking, and a bad court can still win after it survives because individual line evidence is easier to satisfy than a coherent court identity.

W5 tests a pivot around that distinction.

Keep the direction/axis work as a bounded way to propose several genuinely different courts. Then look at each proposal as one complete projection of the physical badminton layout: finite paint, exclusive fragment ownership, both line directions, centre-line terminations, players and the existing camera diagnostic. Start by measuring those clues rather than hard-gating them. Try one existing whole-court refit, but keep the parent too. Only bring temporal evidence back after the single-frame evidence can tell a useful court from a stable alias.

That is the architecture in one paragraph. The rest of this note explains why the branch evidence points there.

## What was reviewed

Branch: `fix/court-det`, reviewed at `82d3b871bfb784f91b283761148eaa0d6501c732`.

The original WebUI review read the relevant experimental code and the small C2/L1/L2/L3 result records. The included `numerical_checks.py` independently recalculates selected saved numbers from a compact real-data extract. Native image pixels were not re-reviewed during that first handover; existing visual rulings were taken from the branch's saved records.

For the refreshed pack I also checked the branch head and the implementation details behind the actionability questions: the L2 winner predicate, the stricter marking-diagnosis eligibility, physical paint geometry, stripe sample arrays, junction semantics and the fixed-stripe refitter. The branch head is still the same commit the original pack reviewed.

Exact source paths and what each source establishes are in [SOURCE_LEDGER.md](SOURCE_LEDGER.md).

## The branch arc in plain language

The commit sequence is easier to understand as four connected strands rather than as a long experiment log.

### 1. Independent 2D courts: can real paint rescue the geometry?

The early work built CourtKeyNet-free court proposals from image lines, then tested player evidence, the net/camera model, explicit marking identities, stripe centres/edges, finite junctions and local whole-homography refits.

That work established several things we still need:

- a complete court hypothesis can be scored without neural corners;
- raw line support is not enough — wrong courts can explain genuine paint;
- finite marking identities and the unpainted centre gap matter;
- local refitting can visibly improve some boundaries but can also make a wrong identity fit itself more tightly;
- players are useful evidence about which court is the target, but are not a substitute for the far boundary;
- the camera/net model is a useful engineering check, not a universal proof of correctness.

This is where the reusable `stripe_observations.py`, `junction_observations.py`, `paint_geometry.py` and `fixed_stripe_refit.py` pieces came from.

### 2. Projective directions: can we keep the right explanation alive?

The next strand separated line detection from line grouping/direction fitting, court generation and final ranking. It explored SVD/projective direction fits, agreement measures and alternative line groups.

The main lesson was not “one direction fitter wins”. It was that candidate identity and candidate geometry are different. A direction assignment can look strong while its chosen representative is geometrically worse, and a global cap can discard a useful court even when the relevant local direction pair contained it.

That is why W5 does not throw away the 1D work. It uses the saved G0 and G1 populations as complementary proposal sources and keeps every retained origin through the new judge.

### 3. Line identity and paint: can we score fragments without teaching to one camera?

The line-identity work made fragment ownership explicit, replayed the candidate-selection pipeline and tested paint-filtered observations. This improved some hard cases dramatically, but not uniformly.

The failure mode became clearer: a scalar line score can reward a court that explains a lot of easy evidence while being wrong in the other direction or at a finite termination. Coarse image paint can also reinforce a false court when the court happens to run through real bright lines.

W5 therefore combines fragment ownership with photometric paint **at the same named marking samples**, then balances the two court directions instead of summing everything into one view-dependent quantity.

### 4. Temporal evidence: agreement is only useful after identity is trustworthy

The temporal pilot found stable aggregate line evidence, but the selected temporal winner could still be geometrically very poor while other candidates had stronger paint and reverse support on every frame.

That is a warning, not an argument against time. Repeated frames can make a wrong court very stable. Temporal pooling should therefore operate on named-marking evidence after the single-view judge works; it should not be used to vote several weak detectors into confidence.

## The core architectural choices

### One physical court model

Use one homography from the metric badminton layout to the 960×540 working image. Corners describe the **outside court boundary**. Internal markings are finite; the centre line is split around the unpainted net gap. Preserve the existing 180° relabelling convention for comparisons.

The physical paint convention in `paint_geometry.CENTRE_SEGMENTS_M` matters. It puts stripe centres at their actual positions relative to the outside boundaries. The older nominal template used by historical scorers is slightly different. W5 keeps the old convention only inside the faithful legacy arm; the new evidence pass and refit receive the physical centres explicitly.

### Aspect ratio is not a court detector by itself

A 6.10×13.40 rectangle does not become identifiable merely because we know its metric aspect ratio. Under a free projective homography, many convex image quadrilaterals can be projections of that rectangle.

The useful metric constraints are the **shared internal layout**: where the singles lines, service lines, baselines and finite centre segments must sit under the same transform. The pinhole camera model adds another conditional constraint, but it should start as a diagnostic rather than being treated as ground truth. It can become a gate later only if the development evidence supports that use.

### Several proposal sources, one final judge

G0 and G1 represent different ways the direction pipeline retained candidates. The evidence says they are complementary, so W5 concatenates them instead of declaring one the new truth.

If a later case still has no useful court, the next search source should be the existing bounded 2D image-line/template proposer, passed through the same judge. That is preferable to building a voting ensemble in which each search method keeps its own acceptance logic.

### A refit is allowed to help; it is not allowed to erase its parent

`fixed_stripe_refit.py` already moves all eight homography parameters together while keeping fragment/interval/paint-position identities fixed. That is a useful local optimiser for a plausible seed.

It is not a global identity solver. If a wrong court has been assigned convincing fragments, the optimiser can make that wrong interpretation fit better. W5 therefore retains parent and child, recomputes all evidence on the child, and ranks them together.

## What the saved numbers actually say

The numbers below are not one homogeneous benchmark. They answer different subquestions. The populations and reference provenance matter.

### C2 — changing observations can improve one local pair while worsening the best available court

Against GX0's visually approved supplied-direction control:

| What was searched | Original observations | Paint-filtered observations |
| --- | ---: | ---: |
| Nearest product inside direction pair 143 | 34.327 px | 12.577 px |
| Nearest candidate over all saved per-pair shortlists, before the global cap | 7.662 px | 9.036 px |

The local pair improves by 21.749 px while the best candidate over the much larger saved pool gets 1.374 px worse. Those are different populations, so there is no contradiction.

The practical lesson is: **do not replace one proposal population with another merely because a local proxy improved.** Preserve useful diversity and judge the actual courts that survive.

C2 also records the Am3 case where the higher-scoring representative for the same axis assignment was much farther from the control. That is exactly the difference between “right line identity” and “right line position”.

### L1 — diversified admission helps one targeted pair and hurts another

Across six targeted direction pairs, the nearest retained shortlist improved for one, worsened for one and was unchanged for four.

The two informative changes were:

| Case / pair | Before | After |
| --- | ---: | ---: |
| Am2-28019 pair 15 | 81.263 px | 7.529 px |
| Am3 B-direction pair 43 | 4.271 px | 23.248 px |

This is not a detector accuracy rate. It says the fixed-budget admission rule can rescue a missing geometric family, but should not be treated as a universal replacement.

### L2 — better paint scoring can rescue a view, but useful courts still lose

On the fixed G0+G1 union, changing only the fragment scorer produced:

| View | S0 line winner error | S1 line winner error | Nearest available union candidate |
| --- | ---: | ---: | ---: |
| GX0 | 10.10 px | 16.19 px | 7.66 px |
| Am2-150 | 140.90 px | 12.44 px | 6.55 px |
| Am2-28019 | 9.00 px | 11.32 px | 7.55 px |
| Am3 | 11.84 px | 10.11 px | 4.27 px |

Am2-150 is the important success: paint-aware fragment scoring found a far better court. The other rows show why that success should not be turned into “paint filtering solves the detector”.

The saved visual controls are even more useful than the summary errors:

| Candidate | Saved ruling | Why W5 keeps it |
| --- | --- | --- |
| Am2-150 `30:33` | approved / very good | positive control: a stronger whole-court rule should not throw this away |
| SS03-19 `1:60` | usable | positive control against the false paint winner in the same view |
| Am2-28019 `184:4123` | rejected | false paint winner near the net; tests court identity |
| SS03-19 `165:6702` | rejected | hallucinated/unrelated paint winner; tests whether real bright lines can fool the score |

The old pixel profile explains what bright structure was accepted. It does not establish that those pixels belong to the proposed court.

### L3 — stable aggregate line score can still prefer a terrible court

For three saved GX candidates over seven frames, the recalculated medians are:

| Candidate | Median line score | Median reverse support | Median paint score | Median reference corner error |
| --- | ---: | ---: | ---: | ---: |
| `106:93818` | 0.3028 | 0.0374 | 0.2500 | 799.36 px |
| `22:4579` | 0.2539 | 0.0754 | 0.7273 | 14.07 px |
| `22:4580` | 0.2548 | 0.0786 | 0.6364 | 9.56 px |

The latter two beat `106:93818` on both reverse support and paint in every one of the seven frames, yet `106:93818` was the saved median-line winner over the full panel.

That is the clearest reason not to make “more temporal agreement” the next project. The contrary evidence already exists; the next judge needs to use it coherently.

## The W5 evidence pass in ordinary language

For each named marking, W5 asks two questions at the same finite sample positions:

1. **Is a line fragment assigned to this marking actually close to this sample?**
2. **What does the image intensity profile look like where the physical stripe should be?**

The first answer is continuous fragment support. For the second, W5 keeps the raw ridge contrast instead of immediately reducing it to a pass/fail vote. It also reproduces the old `contrast >= 10` result as a familiar reference readout, because that is useful for understanding the previous experiments. The old value is not promoted into a new truth merely because it already exists.

The centre line's two finite halves are treated as one marking. Missing/occluded photometric samples are unknown rather than negative.

W5 then summarises named markings separately in the two court directions. It keeps a geometry-only balanced readout and a historical paint-backed balanced readout, each using the weaker of the two directional means. Requiring both projective directions to contribute is a structural court prior; requiring an arbitrary number of markings or samples is not. The exact counts are therefore reported as evidence, not used as first-pass gates.

Centre-line junctions add a different kind of clue. A physically wrong court can put a convincing continuation through a place where the centre line should stop. W5 records the fragment support, image contrast, visibility and masking provenance for those continuations. It does **not** begin by declaring `8 samples`, `40% paint` or the historical `0.55` support cutoff to be a universal veto. If the five-view controls show a clear cross-scene separation, the steering model can promote an appropriate global rule with a small sensitivity check.

There is one practical occlusion wrinkle worth keeping in view. The frozen records have reliable same-image boxes for GX0 and the three amateur W5 cases. GX5's selected boxes come from frame 6 rather than its frame-5 anchor, and the four broadcast images are cached median thumbnails while their boxes come from a source frame. Those five views can still be measured photometrically, but their evidence must record that it is not person-mask-aware on the actual image being scored.

## Players and the camera/net cue

The existing full-court diagnosis used a strict player/camera eligibility rule: valid geometry, at least one player supported throughout the sampled window, two-player support in at least half the samples, and camera error at or below the existing 0.1 limit.

That historical predicate is useful evidence about what the branch has already tried. W5
preserves the raw player fractions, camera error and the historical pass/fail result. It
does **not** use that predicate to filter the initial whole-court candidate comparisons. A
visibly correct court should not disappear before we have learned whether the player or
camera clue is reliable across these views.

The camera error comes from a simplified pinhole-camera model. It may become a useful discriminator, but it is conditional on those camera assumptions. The same is true of player coverage: it is a useful ground-truth-free clue when detections are good, not a certificate that the court geometry is correct.

## What W5 is allowed to fit

The owner has no need for a held-out claim here and no capacity to make new labels. The sensible mode is adaptive development engineering.

W5 may use the existing references and visual rulings **between label-free runs** to fit a crystallised global geometric rule. What it must not do is feed references into the automatic evidence calculation or make per-view exceptions.

There is no fixed quota of two revisions. The steering model should follow the evidence until one of three things is clear: a coherent global reduction is working, a cue does not separate the relevant courts well enough to be useful, or the remaining failure is no longer a ranking problem. When a threshold matters, inspect the raw distribution and do a small rank-stability/sensitivity check in the region the data actually occupy. When a change is made, record the failure it was meant to address and rerun the same global rule across the pilot.

That is enough discipline. There is no benefit in pretending the initial threshold was sacred, manufacturing a giant parameter grid, or creating a held-out ritual that the project does not need. The result should be described as development/fit evidence on the available corpus. If it works, run the current global rule over the rest of the existing labelled/approved data and keep using that corpus for regression and further fitting. Generalisation to unseen scenes is a different question for another day.

## The nine-view panel

The current panel is deliberately concentrated on known difficult cases:

| Frozen source | Available in the frozen source pack | W5 views |
| --- | ---: | ---: |
| GX extension | 7 | 2 |
| Amateur marking/refit pack | 20 | 3 |
| Broadcast extension | 20 | 4 |
| ShuttleSet22 | not represented | 0 |

Candidate count inside one view is not extra scene diversity. This panel can tell us whether the new judge fixes known development failures; it cannot estimate a population-level success rate.

## The decision tree after W5

The most important output is not a single score. It is the answer to **where the remaining failure lives**.

### A useful candidate was present but lost

That is a ranking problem. Compare named-marking evidence between the winner and useful court. Change the failing cue globally. Do not widen the search first.

### A useful candidate existed before the global cap but disappeared from G0/G1

That is an admission/retention problem. The next task is to preserve geometrically different families under a fixed budget, not to keep more near-duplicates.

### No useful candidate existed even in the pre-global material

That is a proposal problem. Add the existing bounded 2D image-line/template proposer as another source, then pass it through the same evidence pass and current global judge. Only move to fresh line extraction if the cached fragments visibly lack the required paint.

### The parent was good and the refit damaged it

Keep the parent. Investigate fixed assignments, endpoints and conditioning. Do not make every court inherit the refit.

### The single-view judge now works

Only then test short same-camera sequences. Pool evidence per named marking and fit/score a shared mode. Do not average incompatible corners and do not let a wrong majority become confidence.

## What success would mean

The practical milestone is not “lower objective”, “more candidates”, “more stable score”, “better average corner error” or “we found one lucky threshold”. It is:

> more of the difficult existing views receive a visibly usable rank-1 complete court without losing courts that were already usable, and the agent can explain why the rule works in terms of court evidence rather than per-view exceptions.

There is deliberately no required rescue count. The known positive controls should remain sensible, the known false paint winners should become distinguishable from useful alternatives by one or more coherent evidence channels, and the ranking should have a plausible operating region rather than depend on a knife-edge number.

If the useful and false courts remain essentially indistinguishable even after the raw evidence is examined, that is a valuable result: the acceptance problem is still open and another strand — admission, proposal generation, refit or temporal evidence — should take over instead of tuning the same score indefinitely.

## What W5 deliberately does not force next

Two useful later questions do not reliably reveal themselves as W5 failure modes. A rule can look coherent on the fitted views yet behave badly on unused scenes; and a good single-frame court can still become dangerous when a stale calibration survives a cut/zoom or when automatic geometry collides with annotator state.

Those are preserved, without making them part of the make-or-break pilot, in [DEFERRED_BRANCHES.md](DEFERRED_BRANCHES.md):

- **Check the rule on unused scenes:** use a small varied batch, add no new labels, and
  freeze the current global rule for the batch
- **Guard court reuse and test annotator integration:** stop stale homographies crossing
  camera changes, then exercise automatic proposals beside the annotator before automatic
  replacement

The frontier model decides whether either branch is worth activating after W5. Luna does not infer that decision from success alone.

The full operational sequence and next-branch rules are in [W5_STEERING_PROMPT.md](W5_STEERING_PROMPT.md). The implementation contract is in [W5_EXECUTOR_PROMPT.md](W5_EXECUTOR_PROMPT.md).
