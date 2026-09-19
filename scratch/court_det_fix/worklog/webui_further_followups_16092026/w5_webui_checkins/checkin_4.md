# Check-in 4 — after the chosen follow-up branch, before closing W5

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

Run this after Luna has pursued the single branch selected from the nine-view diagnosis and committed the resulting comparison. This is the closing independent review of the make-or-break experiment, not a demand for one more iteration by default.

## Prompt

Review the complete committed W5 experiment sequence on `fix/court-det`, including the original five-view census, the fitted global rule, the nine-view run and the one follow-up branch that was chosen from the failure diagnosis.

Your job is to decide what the project has actually learned and whether another W5 iteration is still high-value.

Start from the final overlays and failure decomposition, then use Git history to trace the few material decisions that led there. Do not reward the project for producing more commits, more metrics or more elaborate machinery. Ask whether the final behaviour supports the underlying holistic-detector idea:

> Can a bounded set of genuinely different court hypotheses be kept alive and then judged using complete-court evidence that works tolerably across the existing camera regimes, without relying on per-view exceptions?

Assess separately:

- candidate-generation/admission adequacy;
- quality of the single-view global judge;
- whether physical paint and finite marking identity added useful information;
- whether junction, player and camera evidence earned a role in the rule or should remain diagnostics;
- whether the fixed-identity refit is worth keeping;
- whether temporal evidence is now justified, still premature, or unnecessary;
- whether the current method has a plausible operating region or only works at a fragile fitted point.

It is acceptable for the answer to be "the holistic direction is promising but one specific stage is still the blocker", or "this evidence channel did not separate the courts and should be dropped". Do not force a success narrative and do not force another experiment merely to be thorough.

Recommend one of these outcomes:

- **carry forward** — the current holistic rule is useful enough to become the engineering baseline and be exercised across the rest of the existing development corpus;
- **one targeted continuation** — name the single unresolved mechanism that still has a realistic chance of changing the result;
- **stop this line** — the committed evidence says further tuning of this architecture is unlikely to pay for itself, and say which alternative branch deserves attention instead.

This is still a development conclusion, not a held-out performance claim.

Before closing, read the committed `DEFERRED_BRANCHES.md`. Do not activate a deferred branch merely for completeness. Make one explicit call:

- **D1 now** — an unused-scene reality check would materially reduce the next engineering risk;
- **D2 now** — the project is ready to test guarded reuse or shadow integration;
- **later** — preserve the branch, but it does not answer the current question;
- **not relevant** — the project direction has changed enough that the branch no longer applies.

If you choose D1 or D2, explain the concrete engineering risk it addresses. Do not treat either as a missing W5 gate.

Use the standing output format above, and add a final short section called **What should be preserved from W5** listing only the ideas, code paths or evidence artefacts that are worth carrying into the next phase.
