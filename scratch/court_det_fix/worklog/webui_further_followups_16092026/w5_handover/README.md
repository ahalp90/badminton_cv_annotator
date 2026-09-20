# W5 handover — holistic court detector pilot

## Status: closed

W5 closed on 2026-09-20. The reviewed legacy, original-only and
original-plus-adjusted results are in the
[final stage-5 packet](../../../w5_holistic/runs/w5_stage5_20260920/). The branch decision
and next-session direction are in the
[steering record](../../../w5_holistic/steering_record.md).

The collision repair did not change a scientific conclusion. The final method, which
scores original and locally adjusted candidates together, selects a usable court on eight
of nine stress views. On GX5, the current search never generates a usable foreground-court
candidate. The next experiment is therefore to add the existing independent 2D
line/template search as one bounded source. Keep the camera plausibility filter,
visible-span weighting and the rest of the final scoring rule fixed. Then rerun the same
nine regression views. The unused-scene check and guarded annotator integration remain
deferred.

The rest of this pack records the completed W5 contract. It is background for the new
candidate-generation experiment, not an instruction to rerun W5 unchanged.

This pack drove the next step after the line-only court-detector experiments on `fix/court-det`.
The branch head reviewed here is `82d3b871bfb784f91b283761148eaa0d6501c732`.

The central question is simple:

> How do we keep enough genuinely different, plausible courts alive without exploding the search, then judge those courts with evidence that does not only work for one camera view?

The proposed answer is to keep the existing direction-based work as a **candidate generator**, then look at each complete court through several complementary clues: finite physical paint, exclusive fragment ownership, both court directions, centre-line terminations, players and the existing camera diagnostic. A local whole-court refit is tested as an optional extra, not assumed to help. Temporal pooling comes later, after the single-view evidence can distinguish a useful court from a stable alias.

The important change in this revision is how W5 treats thresholds. The first pass is an **evidence census**, not an acceptance test. Historical cutoffs from earlier experiments are retained as useful reference probes, but they do not get to eliminate a plausible court merely because they were convenient in one earlier camera regime. Hard rejection is reserved for genuinely invalid geometry, numerical failure or broken provenance. The frontier model sees the raw evidence, the historical readouts and the visual controls, then decides how the global rule should reduce them.

## Files in the completed W5 pack

- **[W5_STEERING_PROMPT.md](W5_STEERING_PROMPT.md)** — give this to the frontier model first. Its job is to supervise the pilot, review the evidence, make the non-local decisions and decide what Luna Max should do next.
- **[W5_EXECUTOR_PROMPT.md](W5_EXECUTOR_PROMPT.md)** — the implementation contract for Luna Max. It tells Luna what to measure, what may be rejected outright, what must remain visible for steering, and how to return enough evidence for cheap re-ranking without rerunning expensive geometry.
- **[W5_REVIEW.md](W5_REVIEW.md)** — the human-readable technical brief: what the branch learned, why this pivot follows from it, and what W5 can and cannot establish.
- **[DEFERRED_BRANCHES.md](DEFERRED_BRANCHES.md)** — two later engineering branches that are worth preserving because W5 may not naturally remind us to run them: a fresh-scene reality check and guarded reuse/shadow integration. They are optional and frontier-activated, not W5 gates.
- **[SOURCE_LEDGER.md](SOURCE_LEDGER.md)** — where the claims and frozen inputs come from. It records source paths and the producer/import distinctions that actually matter.

## Original W5 launch sequence (complete)

This sequence is retained for provenance. Do not repeat it before adding the new court
candidate source.

1. Give the frontier model `W5_STEERING_PROMPT.md`, `W5_REVIEW.md` and `SOURCE_LEDGER.md`.
2. The frontier model gives Luna Max `W5_EXECUTOR_PROMPT.md` and supervises the five-view evidence pilot.
3. Luna implements and runs. It measures the full cue set and produces provisional readouts, but it does **not** invent new scoring semantics or silently turn a historical threshold into a new gate.
4. The frontier model reviews the candidate-survival diagnosis, cue distributions, rank stability and compact gallery. It can then keep the provisional reduction, revise one coherent piece of the global rule, or switch branch according to the steering decision tree.
5. Recompute from saved evidence where possible. Do not rerun expensive proposal/refit work merely because the frontier model wants to try a different global reduction of already-saved cues.
6. Ask the human only for genuinely ambiguous visual calls or a change of project intent. Routine implementation choices and ordinary visual comparisons should not consume human review time.
7. When W5 closes, the frontier model checks [DEFERRED_BRANCHES.md](DEFERRED_BRANCHES.md) and explicitly decides whether either later branch is useful now. Luna does not activate either branch on its own.

## The operating assumption

W5 is a **development/fit exercise**. The whole available corpus may inform a crystallised global geometric rule. Existing references and visual rulings may be used between label-free runs to understand failures and choose the next global rule. They must never be fed into the automatic scorer or used for per-view special cases.

This is deliberately adaptive. If the evidence shows that an inherited cutoff is discarding useful signal, move it, soften it or stop using it. If a cue only works in one camera regime, do not rescue it with another arbitrary threshold: inspect the raw distribution and either normalise it sensibly, use it conditionally for a principled reason, or demote it to a diagnostic.

The discipline is practical:

- keep hard validity checks separate from empirical clues;
- record what changed and which failure it was meant to fix;
- keep the rule global across views;
- make small sensitivity/rank-stability checks when a threshold matters;
- avoid changing several unrelated ideas at once when the result would become uninterpretable.

There is no held-out claim in this pack. The nine-view panel is a deliberately difficult regression/stress set, not a representative sample of badminton footage. If the method works, run the then-current global rule over the rest of the existing labelled/approved corpus and keep reporting it as development evidence. New ground-truth labels are not required for this pilot.

## Supporting numerical checks

`numerical_checks.py` reproduces selected arithmetic from C2/L1/L2/L3 using the small real-data extract in `inputs/review_inputs.json`. It does not run the detector, inspect pixels or certify a court.

From this directory:

```bash
python numerical_checks.py
```

The checked outputs are already in `results/`. They are supporting evidence for the technical brief, not a launch gate.
