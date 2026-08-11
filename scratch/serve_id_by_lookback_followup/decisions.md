# Decisions and open questions

## Deviations and open questions

### Q1. What fixed rule connects an earlier accepted contact to a later incoming path?

Recommendation: reuse PR #82's path eligibility and body-height units. Add the smallest symmetric outgoing measurement and one declared time-and-space continuity rule. Inspect current helpers before choosing numbers. Do not tune any value against GT.

Current flow:

```text
accepted Ai
    -> closest usable pre-contact run
    -> robust fitted distance decrease >= 0.05 BH means incoming
```

Required flow:

```text
accepted Aj < accepted Ai
    -> usable outgoing run after Aj
    -> usable incoming run before Ai
    -> fixed time and bbox-relative spatial continuity check
    -> connected: Aj originated the path into Ai
```

Option A, recommended: use existing path windows and eligibility wherever they already define the required boundary. Add only the missing symmetric and continuity calculation. This minimises new judgement, but the code sweep must first confirm what the existing helpers expose.

Option B: introduce new time or spatial cut-offs. This may express the intended physical connection more directly, but it adds uncalibrated numbers and creates a threshold-selection risk.

If nothing is done, the backwards trace cannot reproducibly distinguish an accepted origin from an unrelated earlier impulse.

### Q2. When does a usable backwards trace support an implied serve?

Recommendation: infer an unshown or undetected serve only when the incoming path remains usable to its observable boundary and no earlier accepted contact connects to it. A trace that ends because evidence becomes unusable returns `not enough shuttle trajectory to tell`.

This ruling is conceptually settled. The code sweep must map observable boundary reasons available in PR #82's masks and scene data before the runbook names exact fields.

## Resolved decisions

- Run the same search over all 239 one-to-one rallies; GT is scoring-only
- Keep the existing rally span as a coarse envelope
- Search accepted impulses only
- Move forward when usable post-contact evidence says an accepted impulse lacks outgoing motion
- Stop the main rule when there is not enough shuttle trajectory to tell
- Allow continue-past-unknown only as a labelled sensitivity check
- Treat outgoing motion plus measured absence of incoming motion as a visible serve
- Trace incoming motion backwards before inferring a missing serve
- Reinspect a connected earlier accepted origin using the same before/after rules
- Do not infer an exact frame for an unshown serve
- Keep the production serve-start condition out of the rule and diagnostics
- Keep raw and rejected impulses out of scope
- Keep all new work under `scratch/serve_id_by_lookback_followup/`
- Run focused tests and checks only; do not run the full repository test suite
- Commit messages are pre-authorised when they use plain language, a clear one-line summary, and at most two supporting paragraphs totalling 100 words
- Public repository material may be shared with Codex external delegates and agy
- Final conclusions require bounded AGY Claude Opus and Gemini 3.1 audits
