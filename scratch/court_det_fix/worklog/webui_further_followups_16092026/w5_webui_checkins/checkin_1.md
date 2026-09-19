# Check-in 1 — after the five-view evidence census

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

Run this once Luna has committed the first five-view A/B/C W5 pilot, before a new scoring rule has been heavily fitted around it.

## Prompt

Independently review the first W5 holistic court-detector pilot on `fix/court-det`.

Use only what is committed. Inspect the W5 implementation, the five-view run packet, the candidate/ranking summaries and the actual prediction overlays. Read the branch history far enough back to understand the relevant C2/L1/L2/L3 priors, but do not re-litigate every historical experiment.

The five pilot views should be:

- `gxBQ_window_00_frame_0`
- `am2_window_00_frame_150`
- `am2_window_01_frame_28019`
- `am3_window_00_frame_0`
- `shuttleset_03_scene_0019`

The known controls worth checking are:

- Am2-150 `30:33` — previously judged very good
- SS03-19 `1:60` — previously judged usable
- Am2-28019 `184:4123` — previously rejected false paint winner
- SS03-19 `165:6702` — previously rejected hallucinated/unrelated paint winner

Do not treat those controls as automatic ranking labels. Use them to understand whether the new cue set distinguishes complete-court identity from merely finding real bright line fragments.

First, establish whether the committed W5 run is measuring what it claims to measure. Only flag implementation issues that could change the experimental meaning: wrong candidate population/stage, references leaking into automatic ranking, coordinate/provenance mistakes, stale child scores, or a nominal-vs-physical paint mix-up. Ignore harmless formatting and tiny floating-point replay differences.

Then assess the evidence itself.

For each bad or questionable selected court, determine whether a useful alternative was present in the bounded G0+G1 union. If one was present, compare the winner and useful alternative using the committed per-marking/directional evidence, raw or summarised paint contrast, exclusive fragment support, junction continuation evidence, player clue and camera clue. If the useful court was absent, inspect any committed pre-global witness before calling it a proposal failure.

Pay special attention to whether a cue appears genuinely cross-view or is simply helping one camera while hurting another. Historical cutoffs such as the paint contrast, player predicate, camera limit and junction support thresholds should be treated as reference probes. Do not recommend promoting one just because it happens to rescue a single case.

Also inspect B versus C. If the parent is good and the child is worse, say so plainly. A lower least-squares objective is not a defence of the child.

The goal of this check-in is to answer:

> Does the first holistic evidence census contain a promising global ranking signal, and if so what is the single most informative next reduction/diagnostic to try?

Do not ask for the remaining four views merely to accumulate more rows. Recommend the nine-view extension only if the current evidence is already coherent enough that extra scenes are now the cheapest way to test whether it generalises across the existing development corpus.

Use the standing output format above.

---
