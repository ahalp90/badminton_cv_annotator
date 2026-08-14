# RANSAC production decision

- **Issue:** [#95](https://github.com/ahalp90/badminton_cv_annotator/issues/95)
- **Decision date:** 2026-08-14
- **Fixtures:** `sset_01`, `sset_15`, and `sset_21`
- **Current baseline:** recurrence guard version 4 with a three-frame halo

## Decision

Keep the local quadratic RANSAC lens out of production rejection. Use its
candidates only to rank spans for review.

The evidence does not measure candidate precision. The positive-only visual
challenge set cannot supply that measurement. A separate check also finds
many candidates on independently labelled real contacts. No tested subset has
both useful coverage and a defensible failure policy.

This is a no-go decision for the current evidence. It does not claim that
RANSAC has no diagnostic value.

## Evidence boundary

Issue #31 provides 18 deliberately high-risk spans. Curtis labelled every span
as a hallucination with high confidence. The sample contains no real-shuttle
controls, so its 18 of 18 yield cannot estimate population precision or recall.

The review covers stride-8 professional broadcast fixtures. It does not cover
amateur footage, other resolutions, other frame rates, or a different
TrackNet/InpaintNet contract. The visual audit also lacks a historical video
hash for the original 288p sources.

The RANSAC mask remains a review lead. It uses a 16-frame quadratic, a
four-frame step, 32 deterministic triples, at least eight inliers, a 3-pixel
residual, and a half-window vote. These analysis settings were not calibrated
against labelled valid and invalid coordinates.

## Current guard refresh

The tracked RANSAC audit stored guard codes from the older 15-frame halo. This
decision recomputes recurrence codes from the pinned raw tracks with current
`grade_track`. The refreshed guard reports detector version 4 and a three-frame
halo for every fixture.

`Current guard-clean lead` means:

```text
RANSAC candidate AND current recurrence guard code == 0
```

| Fixture | Valid coordinates | RANSAC candidates | Current guard-clean leads | Labelled contacts | Exact contact conflicts | Final labelled contacts | Final-contact conflicts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sset_01` | 138,764 | 42,993 | 17,542 | 1,641 | 786 | 113 | 43 |
| `sset_15` | 117,150 | 38,205 | 11,589 | 824 | 515 | 104 | 65 |
| `sset_21` | 82,945 | 26,053 | 10,349 | 663 | 355 | 75 | 39 |
| **Total** | **338,859** | **107,251** | **39,480** | **3,128** | **1,656** | **292** | **147** |

The current lead marks 52.9% of labelled contacts. It marks the final labelled
contact in 50.3% of rallies. A three-frame neighbourhood around the current
lead intersects 2,616 of 3,128 labelled contacts and 246 of 292 final labelled
contacts.

These contact labels are independent of RANSAC and the recurrence guard. They
come from `training/data/shuttleset/annotations/shots_master.csv`. They do not
prove that every nearby coordinate is visually correct. They do prove that
automatic rejection would frequently operate at known real motion changes.

## Defined variants and bounded sensitivity checks

The first decision pass scored the current guard-clean lead, residual-severity
cuts, and one impulse veto. It did not score every finite mask that the tracked
evidence can reconstruct. The expanded pass below corrects that gap.

It covers the raw RANSAC mask, current-guard relation, sidecar relation,
complete producer blocks, raw-impulse veto, vote fraction, eligible-window
count, contiguous candidate runs, and maximum residual. These are comparisons
on one existing RANSAC output, not independent fitted detectors. A positive
span counts as hit when any selected frame falls inside its half-open range.

| RANSAC candidate rule | Selected frames | Issue #31 positive spans hit | Exact contact conflicts | Final-contact conflicts |
| --- | ---: | ---: | ---: | ---: |
| Raw RANSAC candidate | 107,251 | 18 of 18 | 1,742 | 160 |
| Current guard-clean | 39,480 | 18 of 18 | 1,656 | 147 |
| Guard-clean and sidecar-selected | 20,197 | 9 of 18 | 760 | 65 |
| Guard-clean and sidecar-negative | 19,283 | 12 of 18 | 896 | 82 |
| Guard-clean and sidecar run at least 15 frames | 7,536 | 3 of 18 | 59 | 8 |
| Guard-clean and vote fraction at least 0.75 | 26,113 | 17 of 18 | 959 | 83 |
| Guard-clean and unanimous eligible-window votes | 23,210 | 17 of 18 | 712 | 63 |
| Guard-clean and at least 2 eligible windows | 29,731 | 15 of 18 | 1,567 | 140 |
| Guard-clean and at least 3 eligible windows | 19,618 | 11 of 18 | 1,372 | 121 |
| Guard-clean and 4 eligible windows | 13,445 | 6 of 18 | 1,096 | 97 |
| Guard-clean and candidate run at least 2 frames | 34,215 | 18 of 18 | 1,509 | 128 |
| Guard-clean and candidate run at least 4 frames | 25,674 | 16 of 18 | 1,045 | 86 |
| Guard-clean and candidate run at least 8 frames | 8,454 | 11 of 18 | 180 | 12 |
| Guard-clean and maximum residual at least 50 px | 18,002 | 18 of 18 | 359 | 26 |
| Guard-clean and maximum residual at least 100 px | 11,309 | 17 of 18 | 58 | 3 |
| Guard-clean and maximum residual at least 200 px | 4,716 | 13 of 18 | 1 | 0 |
| Guard-clean and maximum residual at least 250 px | 2,533 | 12 of 18 | 0 | 0 |
| Guard-clean and maximum residual at least 400 px | 262 | 6 of 18 | 0 | 0 |
| Guard-clean outside a three-frame raw-impulse radius | 11,660 | 7 of 18 | 239 | 22 |

The source-aware proposals are separate from the guard-clean cuts. The
non-overlap producer tiles 16-frame windows from frame zero.

| Source-aware rule or intersection | Selected frames | Issue #31 positive spans hit | Exact contact conflicts | Final-contact conflicts |
| --- | ---: | ---: | ---: | ---: |
| Sidecar-selected run at least 15 frames | 175,742 | 3 of 18 | 238 | 37 |
| Fully selected, coordinate-valid aligned block | 140,608 | 0 of 18 | 91 | 17 |
| Raw RANSAC and fully selected aligned block | 54,830 | 0 of 18 | 45 | 9 |
| Current guard-clean RANSAC and fully selected aligned block | 0 | 0 of 18 | 0 | 0 |

The complete-block result means that the current recurrence guard catches all
raw RANSAC candidates in this source class on these fixtures. It does not
establish the full-block rule's precision. It hits none of the 18 issue #31
spans because those spans were selected from current guard-clean material.

The 250-pixel subset still selects a frame within three frames of a labelled
contact. The 400-pixel subset avoids that narrow contact check. It leaves 262
proposed rejections and catches only six selected spans. Twenty-eight selected
frames fall inside those labelled-positive spans. The other 234 frames lack
visual labels.

The 400-pixel value was found after inspecting the same fixtures. It is tied
to the audit resolution and has no held-out validation. It cannot establish
precision for net interactions, landings, held shuttles, cuts, occlusions,
re-entry, or ordinary non-contact flight.

Protecting a three-frame neighbourhood around derived raw impulses also fails.
That rule retains 11,660 frames and only 7 of 18 challenge spans. It still
selects 239 exact labelled contacts and 22 final labelled contacts. The raw
impulse detector is derived from the same shuttle track, so it is not a safe
independent veto.

Stationary motion is not a safe subset either. Issue #31 contains three fixed
false positions, but it has no real stationary shuttle controls. A shuttle on
the ground or held by a player can create the same motion class.

The audit did not sweep alternate RANSAC window lengths, steps, trial counts,
inlier minima, or base residual radii. Those settings do not define separate
documented guards. Sweeping them on the same positive-only fixtures would add
post-hoc choices without producing a precision denominator.

## Recurrence variants and context unions

The parked follow-up also proposed splitting the current recurrence grade 3.
That is current-guard policy, not a RANSAC production rule. The expanded pass
reproduced its finite masks for completeness.

| Recurrence rejection policy | Selected frames | Issue #31 positive spans hit | Exact contact conflicts | Final-contact conflicts |
| --- | ---: | ---: | ---: | ---: |
| Grade-1 recurrence core only | 143,727 | 0 of 18 | 95 | 17 |
| Core plus three-frame halo | 157,377 | 0 of 18 | 137 | 19 |
| Core plus global exact-coordinate hits | 146,193 | 0 of 18 | 112 | 21 |
| Current all-non-zero policy | 159,835 | 0 of 18 | 154 | 23 |

PR #93 did not run these four policy arms. Its fixed replay compared no guard,
the all-non-zero policy with a 15-frame halo, and the same policy with the
current three-frame halo. The table above is a track and label cross-check,
not a substitute for that missing E2E decomposition.

The historical audit also stored several unions. They were designed as review
context, not rejection masks. Union 1 includes sidecar frames and raw impulses.
Union 2 includes raw impulses and an inductive rally-ending proxy. Treating
either union as a guard would reject evidence sources that were added to show
real event context.

| Context-only view refreshed with current guard | Selected frames | Exact contact conflicts | Final-contact conflicts |
| --- | ---: | ---: | ---: |
| Guard-clean RANSAC or sidecar | 225,214 | 2,074 | 207 |
| Union 1: guard-clean RANSAC, sidecar, or impulse | 231,610 | 2,182 | 214 |
| Union 2: guard-clean RANSAC, impulse, or TP ender | 55,201 | 1,816 | 164 |
| Current guard or Union 2 | 206,232 | 1,943 | 183 |

All four context views touch all 18 positive spans. That is expected because
each contains the guard-clean RANSAC source used to select those spans. It is
not independent evidence and does not make any union a production candidate.

Several parked ideas do not define a production mask. The ledger below records
them as unscored studies or experiments instead of quietly treating them as
failed guards.

## Proposal ledger

The parked follow-up contains broader analysis leads as well as candidate
guards. The tables above score the finite frame masks selected for this
decision from tracked inputs and stated cutoffs. They do not claim to execute
every proposed feature study. This ledger keeps that distinction explicit.

| Parked lead | Issue-95 status | Reason |
| --- | --- | --- |
| Aligned Inpaint support fields | Partly scored | The strongest all-selected aligned block is scored. Selected count, non-Inpaint pass-through count, nearest-support distance, and boundary position have no proposed rejection cutoff. |
| Fully selected aligned-window rule | Scored | The source mask and its raw and guard-clean RANSAC intersections are in the source-aware table. |
| Split recurrence grade 3 | Scored at track and label level | Core, halo, and global-hit policies are separated. The four-arm E2E experiment remains unrun. |
| Producer-phase and boundary diagnostics | Unscored analysis | Jump, acceleration, and residual comparisons by producer phase do not define a rejection mask. Phase-shift nulls were not run. |
| Early versus late handling | Unscored experiment | It requires a selected source-aware mask and fixed-clip counterfactual. No candidate passed the precision gate. |
| Isolation Forest span ranker | Unscored model | No fitted model, held-out score, or production cutoff exists. |
| LOF, One-Class SVM, and Elliptic Envelope | Not production candidates | The parked review rejects or defers them behind a successful Isolation Forest baseline. None has a fitted model or cutoff. |
| Base-TrackNet heatmap morphology | Unscored input study | The required heatmaps were not saved for these tracks, and no rule or cutoff is defined. |
| Separate path shape from absolute location | Unscored analysis | Centred and scale-normalised features were proposed for plotting, not as a rejection rule. |
| Circular-shift, block-preserving, span-length, and phase nulls | Unscored statistical checks | These can test association and plot interpretation. They cannot supply visual precision or turn a context union into a safe guard. |

The unscored items are missing experiments or undefined policies. They are not
quietly counted as failed guards. None can support production rejection from
the existing positive-only labels.

## Current consumer evidence

PR #93 already compared the fixed clips `9WVwZSzixh0` and `P3OcTzwmqeY` on
318,750 total frames. This issue reviewed that existing replay. It did not run
a new RANSAC arm.

| Variant | Filled frames | Rejected frames | Filled and rejected | Rallies | Raw contacts | Filtered contacts | Landing entries | Non-null landings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unguarded PR #83 | 140,215 | 0 | 0 | 218 | 8,086 | 3,727 | 197 | 128 |
| Current recurrence guard | 140,215 | 54,867 | 53,568 | 218 | 8,086 | 3,727 | 197 | 122 |
| Delta | 0 | +54,867 | +53,568 | 0 | 0 | 0 | 0 | -6 |

The current guard failed open on one clip because its derived margin was below
the accepted minimum. It rejected zero frames there. The other clip accounts
for all 54,867 rejected frames.

This replay shows that event masking can change landing availability without
changing contact or rally counts. It does not validate RANSAC. A fixed-clip
before-and-after replay is required only after a candidate rule passes the
precision gate.

## Failure policy

The safe policy now is to keep RANSAC outside the production event mask. A
failed local fit, an ineligible window, or an unsupported input must therefore
leave production evidence unchanged.

A future production proposal must define fail-open behaviour for gaps,
unsupported resolution or frame rate, video edges, and insufficient eligible
windows. It must also define how known contacts and scene transitions are
protected. Fail-open mechanics alone cannot fix false rejection from an
over-broad geometric rule.

## Missing evidence and next gate

A new rule needs all of the following before implementation:

1. A blind probability sample of production candidates that can estimate
   precision. A stratified sample must retain sampling weights. Use a separate
   balanced stress set for contacts, net events, landings, stationary or held
   shuttles, cuts, occlusions, re-entry, and ordinary flight. Do not use the
   balanced set alone to estimate deployment precision.
2. One observation per span or aligned producer window. Long spans must not
   receive extra statistical weight from their frame count.
3. A predeclared, resolution- and time-normalised rule. Threshold selection
   and evaluation must use different videos.
4. A stated precision target and false-rejection cost for each downstream
   consumer.
5. A fail-open contract and diagnostics for every unavailable state.
6. A correctness-only implementation commit with focused tests.
7. A before-and-after replay on the same fixed E2E clips. It must report
   filled and rejected frames, rallies, raw and filtered contacts, landing
   entries, and non-null landings.

Until those gates pass, RANSAC candidates remain review leads.

## Double confirmation

The contact result was calculated twice with separate loaders. The first pass
used pandas tables. The second pass used the standard-library CSV reader and
asserted every fixture count.

Both passes reloaded the compressed pinned tracks and RANSAC masks. Both
recomputed current guard codes through `grade_track` rather than using the
older stored guard arrays. The second pass also confirmed that labelled contact
frames are unique within each fixture.

The expanded comparison is reproduced by
`analysis/audit_production_variants.py`. It asserts that the frame-audit vote
fields reproduce each stored RANSAC candidate mask. It also reconstructs the
recurrence core, halo, and global exact-hit components and asserts that halo or
global hits outside the core reproduce public grade 3. The script prints JSON
with candidate rules, recurrence policies, source-aware proposals, and
context-only unions kept in separate groups.

A separate visual spot-check sampled three exact conflicts across each fixture
at deterministic positions in the ordered conflict list:

- `sset_01`: frames 11,934, 66,137, and 135,479;
- `sset_15`: frames 23,661, 75,932, and 130,802; and
- `sset_21`: frames 12,771, 53,528, and 93,142.

Five-frame windows from the aligned issue #31 videos show full-court active
play and player motion consistent with the labelled contact context in all
nine samples. This check supports the CSV alignment and event context. It does
not prove that each predicted coordinate is the visible shuttle.

The older stored guard gives 1,599 exact contact conflicts. The refreshed
three-frame guard gives 1,656. The increase is expected because PR #93 narrowed
the recurrence halo and leaves more RANSAC candidates guard-clean.

The verification left the audit arrays, labels, and `shots_master.csv`
unchanged.

## Scope outcome

No production source, configuration, detector threshold, or test was changed.
No RANSAC E2E arm was run because no rule passed the implementation gate.
Issue #75 performance work was not used or modified.
