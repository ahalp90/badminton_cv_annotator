# Optional WebUI check-in D1 — fresh-scene engineering reality check

Run this only if the frontier steering episode explicitly activated **D1** from `DEFERRED_BRANCHES.md` and Luna has committed the resulting unused-scene batch.

This is not another W5 tuning checkpoint and not a request for new ground truth. The rule under review should have been frozen before the batch was selected and run.

## Prompt

Independently review the committed D1 fresh-scene reality check on `fix/court-det`.

Use Git as the source of truth. Inspect:

- the exact detector rule carried forward from W5;
- how the unused views were selected;
- the source-frame rank-1 / abstention overlays;
- compact candidate evidence for genuine failures;
- the nine original W5 regression results under the same rule.

Do not turn this into a statistical holdout evaluation. The purpose is narrower: determine whether the W5 rule still behaves like the same understandable global mechanism on scenes that did not shape it.

First check that the batch was chosen **before** its detector outcomes were known and that Luna did not tune settings per view. The practical scale may be around 12–20 views from several unused videos, but do not penalise a smaller useful batch merely for missing a quota.

Look for the kinds of scene dependence that W5 itself could not reveal:

- confident selection of neighbouring or unrelated courts;
- failure on partial visibility that the fitted views did not expose;
- false proposals on non-court / camera-change imagery;
- a cue that reverses behaviour in a new camera regime;
- excessive abstention caused by a rule that became too comfortable on the fitted panel.

Do not reduce the result to a success percentage. Classify failures by mechanism and ask whether one dominant global problem is visible.

If one coherent repair is strongly supported, it is reasonable to recommend **one** targeted global revision followed by a rerun of the **same D1 batch plus the nine W5 regressions**. Do not recommend a succession of newly selected showcase frames.

Finish with:

- **What D1 added beyond W5** — one short synthesis.
- **Does the current rule travel coherently to these unused scenes?** — `yes enough for the next engineering step`, `one global repair justified`, or `scene dependence still too strong`.
- **Dominant failure mechanism** — if there is one.
- **Next action** — one concrete action, or `none` if this check has done its job.
- **What this does not establish** — keep the scope honest without inventing a validation programme.
