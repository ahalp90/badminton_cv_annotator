# Check-in 2 — after the pilot has produced a candidate global rule

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

Run this after Luna/frontier steering has made one or more evidence-driven global revisions and committed a new five-view pilot result. This is the independent challenge before spending effort on the full nine-view pass.

## Prompt

Review the current W5 five-view pilot on `fix/court-det` as an independent second opinion. Compare the latest committed W5 rule/run with the earlier five-view census rather than evaluating the latest result in isolation.

Use Git history to identify the material rule changes. Do not care about cosmetic refactors. For each meaningful scoring/reduction change, reconstruct in ordinary language:

- what observed failure it was intended to fix;
- what global rule actually changed;
- whether that failure improved across the pilot;
- whether a known-good court or another camera regime regressed;
- whether the improvement depends on a knife-edge threshold or survives a small nearby sensitivity check, if one was committed.

Inspect the overlays yourself before reading any final visual verdict as authoritative. Treat earlier visual rulings as useful annotations, not a substitute for looking.

The main question is whether the current rule has become **coherent enough to deserve the nine-view pass**. Coherent does not mean perfect and it does not require one fixed threshold to have been chosen in advance. It means you can explain why the rule tends to prefer complete plausible courts over the known aliases, and the explanation is not a collection of per-view exceptions.

Explicitly check:

- whether both court directions contribute meaningfully rather than one direction carrying the whole rank;
- whether paint evidence is adding court identity information rather than merely rewarding bright lines;
- whether junction/termination evidence is useful where observable and stays neutral where it is not;
- whether player/camera clues help consistently enough to rank, or belong only in diagnostics for now;
- whether refitting adds real value beyond a good parent, and whether any inherited sample-admission cutoff is the actual bottleneck;
- whether a remaining failure is still a ranking problem or has clearly moved upstream into admission/proposal.

If the current global rule is ready, recommend the full nine-view run and state what you most want that run to resolve. If it is not ready, recommend exactly one further five-view experiment or reduction change and explain why another nine-view run would currently be low-value.

Do not demand a formal freeze, a held-out set, or extra verification artefacts.

Use the standing output format above.

---
