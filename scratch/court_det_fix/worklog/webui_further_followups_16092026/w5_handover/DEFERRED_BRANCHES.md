# Deferred experiments — useful later, not part of the W5 pilot

This file preserves two follow-ups that are easy to forget precisely because a successful W5 run may not force them back into view.

They are **not extra W5 gates**. Luna should not start either branch merely because the holistic detector looks promising. The frontier model decides whether and when one is worth activating, as a separate episode, after reviewing the current committed evidence.

The point is to preserve useful engineering questions without turning them into ceremony.

## How these experiments fit the operating model

The division of labour stays the same:

- **Luna executes.** It runs the already-chosen experiment, records compact evidence and does not broaden the question on its own.
- **The frontier model evaluates and steers by episode.** It decides whether a deferred branch has become relevant, interprets the result and chooses the next step.
- **The human is not a routine checkpoint.** Escalate only a consequential ambiguity about project intent or an image judgement that would materially change the decision.

Do not pre-build these experiments during W5. Do not make W5 wait for them. Preserve the hook and activate it when it answers a question the project actually has.

---

# Check the rule on unused scenes

## Why keep this experiment

The nine W5 views are development/stress cases. They can tell us whether the holistic design is worth pursuing and help fit a global rule, but they cannot tell us whether that rule has merely become comfortable on the scenes that shaped it.

That problem may never reveal itself inside W5. A rule can look coherent on the fitted corpus and still behave badly on an unused camera or a neighbouring-court scene. This branch is a cheap reality check before the project starts treating the rule as broadly dependable.

It is **not** a scientific holdout programme and does not require new ground-truth labels.

## When the frontier model should activate it

Run this check when all of the following are true:

- W5 has produced a single global rule worth carrying forward;
- the remaining W5 failures are understood well enough that another pass over the same nine views is unlikely to teach much;
- the project is about to rely on the detector beyond the scenes that shaped that rule, or wants a sanity check before further integration work.

Do not run it just to make the work look more rigorous.

If there is no useful pool of unused scenes, say so and keep the limitation explicit. Do not create a new labelling project merely to satisfy this branch.

## What Luna should run

Freeze the current global detector rule for the batch. Select a **small, varied set of unused views before running the detector on them**.

The earlier handover suggested roughly **12–20 views from at least three videos** because that is usually enough to expose obvious scene dependence without creating a new evaluation project. Treat that as a practical scale, not a quota. Use fewer if that is what the available footage supports; use somewhat more if selection is essentially free.

Where the existing footage allows it, include a mix such as:

- ordinary but previously unused court views;
- partly visible courts;
- neighbouring or multiple courts;
- non-court / false-proposal opportunities;
- camera-change or unusual framing views.

No new labels are required. The useful evidence is the detector's selected court or abstention, the source-frame overlay and any compact candidate evidence needed to understand a failure.

Run the current global rule **unchanged across the whole batch**. Keep the original nine W5 views as regression cases so an apparent fresh-scene repair cannot quietly damage the fitted baseline.

## What the frontier model should decide

Review the batch as an engineering reality check:

- Are rank-1 courts visibly coherent on the new scenes?
- Are there new obvious false courts or confident proposals on non-courts?
- Does the rule abstain sensibly when its evidence is weak?
- Do failures repeat one mechanism, or are they unrelated edge cases?
- Did the nine original regression views remain sensible?

Do not reduce this to a success percentage. The useful question is whether the rule still behaves like the same understandable global mechanism outside the scenes that shaped it.

If one **dominant, coherent** failure appears, the frontier model may authorise one targeted
global correction and have Luna rerun the **same whole batch plus the nine regressions**.
Do not turn the check into an endless succession of newly selected showcase frames.

Possible outcomes are simple:

- **good enough for the next engineering step** — the rule behaves coherently on the unused scenes;
- **one global repair is justified** — a repeated failure has a clear mechanism worth fixing once;
- **scene dependence is still too strong** — return to the relevant proposal/admission/ranking mechanism rather than claiming broad robustness.

This check supports an engineering decision. It is not a universal reliability certificate.

---

# Guard court reuse and test annotator integration

## Why keep this experiment

A good single-frame detector does not automatically make continuous reuse safe.

The failure mode here is operational rather than statistical: a stale court can look perfectly confident after a cut, zoom, shake or other camera change. Likewise, an apparently sensible detector can cause trouble when coordinate scaling, orientation, manual precedence or downstream court-dependent logic meets real annotator state.

Those problems may not announce themselves during W5 at all. They often appear only after the detector has been wired into the longer-lived system, which is why this branch is worth preserving explicitly.

## When the frontier model should activate it

Run this experiment when the current single-view detector is useful enough that the
project is seriously considering either:

- reusing a detected court across frames / a scene; or
- feeding automatic court proposals into the annotator workflow.

It does not need to wait for a formal unseen-scene claim. It does need a single-view baseline that the frontier model is willing to treat as an engineering candidate rather than an unresolved research prototype.

## D2a — Guard reuse across camera changes

Luna should add the smallest guard needed to prevent stale calibration from silently surviving a changed view.

Exercise real or readily available examples of:

- cuts;
- zooms or meaningful reframing;
- camera shake / displacement;
- temporary occlusion followed by recovery.

This is not a request for an exhaustive synthetic transition suite. Use enough examples to establish the mechanism.

The required behaviour is straightforward:

> an unverified or materially changed view stops borrowing the old court and asks for a new proposal; it does not inherit a confident-looking stale homography.

Where reuse is allowed, keep enough evidence to show why the scene was considered unchanged. Where reuse is rejected, show the reset/reproposal path and the first resulting overlay.

The frontier model should judge whether the guard catches the meaningful changes without turning ordinary same-camera footage into constant unnecessary re-detection.

If temporal reuse only reduces repeated work on clearly static views, that is a valid limited benefit. Do not relabel caching efficiency as detector accuracy.

## D2b — Shadow the annotator before automatic replacement

Run the automatic court proposal **beside** the annotator first. Do not automatically replace a manual or previously approved court.

Exercise the integration points that can silently corrupt otherwise good geometry:

- working/native coordinate scaling;
- the existing 180-degree relabelling convention;
- finite physical paint extents;
- missing observations and abstention;
- view/camera resets;
- manual / approved-court precedence;
- absence of accidental CourtKeyNet model or weight loading;
- court-dependent player / annotation outputs that consume the geometry.

The useful outputs are not a giant integration test report. Record the cases where automatic geometry would have changed behaviour, how often a proposal needs correction, and any downstream regression that actually appears.

The frontier model then chooses among:

- keep it in **shadow** while a specific integration problem is fixed;
- allow **guarded reuse / caching** but not automatic court replacement;
- promote the detector into the normal automatic path while preserving manual/approved precedence;
- stop integration because the operational failure cost is not justified by the detector gain.

## What not to add

Do not use this branch as an excuse for:

- checksum or byte-exact verification machinery;
- a large generic smoke-test matrix unrelated to observed integration risks;
- automatic replacement of approved geometry just to collect more data;
- per-video exceptions to keep a stale reuse rule alive.

The goal is simply to stop a promising detector being undermined by predictable state/reuse mistakes.

---

# The activation hook

At the end of W5, the frontier model should make one short call:

- **check unused scenes now** — this check would materially reduce the next engineering risk;
- **test reuse and annotator integration now** — the project is ready for that experiment;
- **later** — preserve the branch, but it does not answer the current question;
- **not relevant** — the project direction has changed enough that this branch no longer applies.

That call is a scope decision. Luna should never infer it from a generic instruction such
as “carry forward W5”.
