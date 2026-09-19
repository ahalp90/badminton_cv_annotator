# W5 WebUI check-ins — independent reviewer series

These prompts are for a WebUI model that can inspect the committed `fix/court-det` branch and its Git history but has no access to Luna's local execution host or hidden working notes.

The WebUI reviewer is deliberately independent. It is not there to reproduce Luna's work, bless whatever the latest `result.md` says, or enforce a frozen experimental protocol. Its job is to look at the committed implementation and evidence, form its own view of what the experiment is showing, and identify the most useful next decision.

Use the four W5 check-ins at the decision points below. Do not run them after every small implementation commit. Two **optional deferred check-ins** exist for later engineering episodes; run them only if the frontier steering decision explicitly activates the corresponding branch in `DEFERRED_BRANCHES.md`.

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

# Check-in 1 — after the five-view evidence census

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

# Check-in 2 — after the pilot has produced a candidate global rule

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

# Check-in 3 — after the full nine-view run

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

# Check-in 4 — after the chosen follow-up branch, before closing W5

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

Before closing, read the committed `DEFERRED_BRANCHES.md`. Do not activate a deferred branch merely for completeness. Make one explicit call: **D1 now**, **D2 now**, **later**, or **not relevant**. If you choose D1 or D2, tie it to a concrete engineering risk rather than treating it as a missing W5 gate.

Use the standing output format above, and add a final short section called **What should be preserved from W5** listing only the ideas, code paths or evidence artefacts that are worth carrying into the next phase.


---

# Optional deferred check-ins

These are separate post-W5 episodes, not check-ins 5 and 6 that must happen automatically.

- **D1 — fresh-scene engineering reality check:** use `deferred_checkin_D1_fresh_scene.md` only after the frontier model activates D1 and Luna commits the unused-scene batch.
- **D2 — guarded reuse and shadow integration:** use `deferred_checkin_D2_shadow_integration.md` only after the frontier model activates D2 and Luna commits the reuse/shadow evidence.

The extra Git evidence those episodes need is described in `commit_for_webui.md`.
