# Check-in 3 — after the full nine-view run

## Standing instructions for every check-in

Use the repository itself as the source of truth. Start from the relevant W5 commits on `fix/court-det`, inspect the current W5 code and the committed run packet, and look at the actual court overlays before accepting the author's interpretation of them.

The baseline before W5 is commit `82d3b871bfb784f91b283761148eaa0d6501c732`. Use later Git history to understand what changed. Do not ask the operator to supply SHA sums or prove byte-for-byte identity. Commit identity, source paths, population stage and the actual code/results are enough.

Treat this as adaptive development engineering on the available corpus. There is no held-out claim to protect and no virtue in keeping a bad threshold merely because it was written down earlier. Historical thresholds are priors and diagnostic probes unless the committed evidence gives a good reason to promote one into the global rule.

Keep these questions separate:

1. Was a useful court proposed at all?
2. If proposed, did it survive the bounded admission/retention path?
3. If it survived, did the global judge rank it sensibly?
4. If refitting was attempted, did it improve the right court or damage it?

Do not answer a ranking failure by reflexively widening search, or a proposal failure by endlessly tuning the ranker.

Judge visible court quality from the source-frame overlays, not from one scalar score. A lower refit objective, higher line score, more stable temporal score or lower extrapolated corner error is supporting evidence only. Prefer visible paint alignment and coherent complete-court identity.

Do not invent a scientific-validation programme. No significance tests, no preregistration language, no arbitrary pass counts and no demand for new held-out labels. Also do not invent verification work: no checksums, byte-exact file comparisons, redundant smoke-test ladders or exhaustive parameter grids unless a concrete failure gives a reason for one.

When a threshold or weight appears to matter, inspect the committed values around the decision and ask whether there is a plausible operating region. A small sensitivity or rank-stability check is useful; a giant sweep for completeness is not.

For each check-in, finish with:

- **What the evidence currently says** — the shortest accurate synthesis you can give.
- **Where the main remaining failure lives** — `ranking`, `admission/retention`, `proposal`, `refit`, `temporal-not-ready`, or `unclear`.
- **The next useful action** — one concrete global experiment or implementation step, not a menu of vaguely possible work.
- **Why that action is worth doing now** — tie it to a failure visible in the committed evidence.
- **What result would change your mind** — one short falsifier so the next agent knows when not to keep pushing the same idea.

If the committed evidence is insufficient to make one of those calls, say exactly what is missing and whether it is worth committing. Do not request bulky tensors or private run directories when a compact candidate summary or one extra overlay would answer the question.

---

---

Run this once a single current global configuration has been committed across all nine W5 views.

## Prompt

Independently evaluate the committed nine-view W5 holistic court-detector run on `fix/court-det`.

Do not reduce this to a win count. Inspect the nine selected overlays and the committed diagnostic alternatives, then classify the remaining failures by where they actually occur in the pipeline.

For each view whose rank-1 court is not clearly usable, determine whether the dominant issue is:

- useful candidate absent even before global retention → `proposal`;
- useful candidate existed before the cap but disappeared from G0/G1 → `admission/retention`;
- useful candidate survived but lost → `ranking`;
- useful parent existed but refit damaged or failed to improve it → `refit`;
- evidence is genuinely too ambiguous to locate the failure → `unclear`.

Keep photometric sensitivity, player clues, camera clues and junction clues as explanations **inside** a ranking diagnosis unless they truly change which pipeline stage failed.

Compare the nine-view result with the original legacy A comparator and with the known positive/negative controls. Look for regressions as carefully as rescues. A useful global judge should improve difficult cases without quietly throwing away courts that were already serviceable.

Then decide which one branch should receive the next engineering effort:

- **R — ranking:** useful candidates are present but the global evidence reduction still chooses badly;
- **A — admission/retention:** useful geometric families exist before the cap but are not preserved;
- **P — proposal:** necessary geometry is absent even in pre-global material;
- **F — refit:** correct parents are present and ranking is adequate, but local geometric correction is now the limiting opportunity;
- **T — temporal:** only if the single-view judge already distinguishes the known aliases and the remaining issue is genuinely same-camera evidence aggregation.

Choose one. Do not recommend parallel work on all five.

If you choose R, name the specific evidence failure that should be changed next rather than saying "improve scoring". If you choose A or P, explain why more score tuning would be wasted effort. If you choose T, point to the committed evidence showing that temporal pooling is no longer being asked to stabilise a wrong single-frame majority.

State clearly what this nine-view development run establishes and what it does not. Do not turn it into an unseen-scene performance claim.

Use the standing output format above.

---
