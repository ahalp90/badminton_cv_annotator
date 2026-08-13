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

## Simple production subsets

The following checks use the current guard-clean lead. A challenge span counts
as hit when at least one selected frame falls inside its half-open range.

| Additional rule | Selected frames | Challenge spans hit | Exact contact conflicts | Final-contact conflicts |
| --- | ---: | ---: | ---: | ---: |
| None | 39,480 | 18 of 18 | 1,656 | 147 |
| Maximum residual at least 50 px | 18,002 | 18 of 18 | 359 | 26 |
| Maximum residual at least 100 px | 11,309 | 17 of 18 | 58 | 3 |
| Maximum residual at least 200 px | 4,716 | 13 of 18 | 1 | 0 |
| Maximum residual at least 250 px | 2,533 | 12 of 18 | 0 | 0 |
| Maximum residual at least 400 px | 262 | 6 of 18 | 0 | 0 |

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
