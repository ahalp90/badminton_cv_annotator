# What to commit so the WebUI check-ins are actually useful

The WebUI reviewer only sees Git. It does not need the whole run directory, but it does need enough committed evidence to tell **missing candidate**, **misranking** and **refit damage** apart without trusting Luna's prose.

This is the useful minimum.

## Commit the implementation that produced the result

Commit the W5 driver/verifier/rendering code and any small shared-helper change that materially affects the experiment. Normal focused tests for changed logic are useful. Do not add checksum machinery, byte-equality tests or a second replay harness just for the reviewer.

The reviewer should be able to see which proposal stage is loaded, how A/B/C are constructed, how evidence is reduced and how ties/refits are handled.

If the deferred-branch hooks are being used, commit `DEFERRED_BRANCHES.md` with the active handover material as well. The WebUI reviewer cannot act on a branch definition that exists only in Luna's local working directory.

## Commit one compact packet for each reviewable run

Under the run directory, commit the small files that explain the run:

```text
result.md
manifest.json
per_view.csv
rankings.json
fit_attempts.csv          # or an equivalent compact table
```

If `rankings.json` does not contain the cue breakdown needed to compare the selected court with plausible alternatives, also commit one compact candidate-review file, for example:

```text
review_candidates.json
```

That file does **not** need every sample tensor. For each view it should include, at minimum:

- A/B/C selected candidate IDs and parent/child provenance;
- the top few candidates under the important saved rankings;
- the best useful/reference-near candidate identified only after label-free ranking is locked;
- the known positive/negative diagnostic controls when applicable;
- hard-validity status;
- two-direction geometry and paint readouts;
- exclusive reverse support;
- per-marking/directional summaries needed to explain a ranking reversal;
- raw or compactly summarised paint-contrast values around any threshold that became decision-critical;
- junction continuation evidence where it matters;
- player fractions, camera error and the old historical predicate outcomes;
- for children, enough refit information to tell improvement from damage.

Keep this reviewer file focused on candidates a human/model might reasonably compare. The full sample arrays can stay on the execution host.

## Commit the images the reviewer must judge

The independent review is weak without the actual overlays.

Commit ordinary image files for:

- the A/B/C selected courts for each view in the current checkpoint;
- any top alternative needed to understand a bad selection;
- the four known W5 diagnostic controls during the pilot;
- parent and child side by side when refit behaviour is part of the decision.

Prediction-only overlays should be easy to inspect. A separate reference/diagnostic overlay is useful where one already exists, but do not bake reference information into the image used to understand what the automatic method selected.

A small `gallery/index.md` that links the relevant images by view/candidate is more useful to a Git-only reviewer than a large generated website. Reuse identical images when A/B/C choose the same geometry.

Do not commit hundreds of near-duplicate candidates. On a failed view, selected candidate plus the few alternatives that establish the failure mode are enough.

## Preserve the comparison across steering revisions

When a global rule changes, keep the previous compact run packet in Git rather than overwriting it. The WebUI reviewer should be able to compare the before/after rule and see what failure the revision was meant to address.

The new `manifest.json` or `result.md` should say, in plain language:

- what global rule changed;
- what observed failure motivated it;
- whether the expensive geometry was rerun or the saved evidence was only re-ranked.

That short change record is enough. There is no need for a formal experiment registry.

## For the nine-view and follow-up checkpoints

Make sure the committed packet lets the reviewer locate the failure stage. If a selected court is wrong, include enough evidence to answer:

- Was a useful candidate in G0/G1?
- If not, was one visible in a committed pre-global witness?
- If yes, why did it lose the rank?
- If a child was involved, what did the parent look like?

If the answer depends on material that remains only on the execution host, commit the smallest relevant witness: usually one candidate summary, one pre-global row or one overlay. Do not export the whole private run tree pre-emptively.


## If D1 — fresh-scene reality check — is activated

Only add these materials if the frontier model actually activates D1. The WebUI reviewer needs to see that the current W5 rule was exercised on unused scenes without quietly being retuned around them.

Commit:

- the small list/manifest of selected views and their source video/scene identity, with selection fixed before detector outcomes are inspected;
- the exact carried-forward W5 global rule in the run manifest;
- one compact per-view result table covering selection/abstention and the main cue summaries for failures;
- source-frame rank-1 or abstention overlays for the fresh views;
- a few serious alternatives only where they are needed to explain a failure;
- the nine original W5 regression outcomes under the same rule.

The original handover suggested roughly 12–20 views from at least three unused videos, including partial courts, neighbouring courts and non-court/camera-change opportunities. That is a useful scale, not a Git completeness requirement. Commit what was actually available and selected; do not manufacture extra frames to hit a number.

No new labels are required for WebUI. If the reviewer can make the needed visual call from the source-frame overlay, that is enough. If a particular view genuinely needs an existing reference to resolve ambiguity, commit the smallest existing reference artefact that answers it.

If the frontier model authorises one global repair after D1, preserve the original D1 packet and commit the rerun on the **same batch plus the nine regressions**. Do not replace the first result or select a friendlier second batch.

## If D2 — guarded reuse / shadow integration — is activated

Only add these materials if the frontier model actually activates D2. The useful Git evidence is about state transitions and integration behaviour, not a giant test suite.

For reuse guards, commit a compact event table and representative frame/overlay sequences showing the available meaningful cases such as cuts, zoom/reframe changes, camera displacement or temporary occlusion. For each event, make it possible to see:

- whether the prior court was reused or invalidated;
- the reason/state signal used by the guard;
- whether a fresh proposal was requested after invalidation;
- the first post-reset court overlay.

A handful of informative before/after frames is better than a committed video dump.

For shadow integration, commit the small evidence needed to inspect actual interactions with the annotator:

- automatic proposal versus the manual/approved court where both exist;
- whether manual/approved precedence was preserved;
- any real correction the automatic proposal needed;
- any observed court-dependent player/annotation regression;
- compact evidence for coordinate scaling, orientation relabelling, finite extents, missing-observation handling and view resets when one of those materially affected behaviour;
- evidence that the D2 path does not accidentally require CourtKeyNet weights if that risk is relevant to the changed integration code.

Do not create or commit an exhaustive environment matrix, checksum manifest or generic smoke suite for WebUI. The reviewer needs to understand the **actual reuse/integration risks encountered** and whether the guard/precedence rules handled them.

## Things not worth committing for these check-ins

Unless they become necessary to explain a specific bug or decision, leave these on the execution host:

- full per-sample response tensors;
- giant candidate dumps already represented by a compact review table;
- temporary caches;
- duplicate rendered galleries;
- environment inventories unrelated to the result;
- checksums and SHA-sum manifests;
- byte-exact replay artefacts;
- exhaustive threshold grids made only to demonstrate thoroughness.

The standard is simple: **commit enough that an independent reviewer can see what the algorithm selected, what serious alternatives existed, why they ranked differently, and where a failure entered the pipeline.**
