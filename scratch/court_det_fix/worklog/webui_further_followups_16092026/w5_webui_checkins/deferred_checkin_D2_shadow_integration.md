# Optional WebUI check-in D2 — guarded reuse and shadow integration

Run this only if the frontier steering episode explicitly activated **D2** from `DEFERRED_BRANCHES.md` and Luna has committed the reuse/shadow evidence.

The question is operational: can a useful single-frame detector be reused and integrated without stale calibration or state-handling mistakes? Do not re-open W5 scoring unless the committed evidence shows that the underlying single-frame court itself is wrong.

## Prompt

Independently review the committed D2 guarded-reuse and shadow-integration episode on `fix/court-det`.

Use Git as the source of truth. Inspect the actual transition examples and overlays rather than trusting a summary that says the guard fired.

### Reuse guard

Check representative committed examples of the camera/state changes that were actually available, such as cuts, zooms/reframing, camera displacement or temporary occlusion.

The important invariant is:

> an unverified or materially changed view must stop borrowing the previous court and request a new proposal; it must not carry a confident-looking stale homography across the change.

For each meaningful example, inspect whether:

- the old court was reused, invalidated or reset for an understandable reason;
- a reset leads to a fresh proposal rather than silent loss of geometry;
- ordinary unchanged footage is not needlessly forced into constant re-detection.

Do not demand an exhaustive synthetic transition matrix. Judge the mechanism from the available real/representative cases.

### Shadow integration

Check that automatic proposals ran **beside** the annotator rather than replacing manual/approved courts by default.

Look specifically for committed evidence around integration points that can make good geometry behave badly:

- working/native coordinate scaling;
- 180-degree relabelling;
- finite paint extents;
- missing observations / abstention;
- view resets;
- manual or approved-court precedence;
- accidental CourtKeyNet model/weight loading;
- court-dependent player/annotation outputs that changed because of the new geometry.

Focus on actual observed regressions or correction burden. Do not reward a large generic test inventory.

Finish with:

- **Reuse verdict** — `safe enough for guarded reuse`, `shadow only`, or `stale-state risk unresolved`.
- **Integration verdict** — `ready for normal automatic path with precedence preserved`, `keep shadowing while one issue is fixed`, or `integration cost outweighs current gain`.
- **Main operational failure, if any** — one mechanism.
- **Next action** — one concrete engineering step, or `none`.
- **Boundary check** — state explicitly whether any recommendation would change manual/approved precedence or automatic replacement policy.
