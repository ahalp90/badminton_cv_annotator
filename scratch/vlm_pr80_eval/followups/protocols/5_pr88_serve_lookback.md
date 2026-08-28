# Follow-up 5: reconcile the clean serve result with PR 88

## Bottom line

**Status: complete.** PR 88 recomputes exactly, but the retained evidence does
not justify a VLM hybrid. No hybrid experiment was run. See
[`../5_pr88_serve_lookback.md`](../5_pr88_serve_lookback.md).

The frozen plan was to inspect PR 88 only after Follow-ups 3 and 4 were
complete, then test at most one simple serve-lookback hybrid justified by its
retained evidence.

This is a separate follow-up. It may change the recommended future route, but
it does not rewrite the clean experiments that came before it.

## Preserve the boundary

Before reading PR 88, freeze:

- the Follow-up 3 precision-first rule and held-out result;
- the Follow-up 4 evidence comparison;
- any paired model confirmation triggered by Follow-up 4;
- the chosen clean configuration and its end-to-end result.

Do not use PR 88 to retune those results after seeing its mechanisms or scores.

## Audit PR 88 first

Record what the serve-lookback work actually tested:

- its intended failure case;
- the automatic evidence available at inference time;
- the labelled data and scoring rule;
- the implementation paths;
- the retained manifests, outputs and row-level evidence;
- its limits and any claims that cannot be reproduced from the repository.

Do not infer a mechanism or benefit that the pull request does not establish.

## Choose at most one hybrid

Compare PR 88 with the frozen clean route. Select one addition only when the
retained evidence gives a concrete reason that it could address a remaining
error.

The hybrid must:

- use automatic evidence available in the real pipeline;
- make one small, explainable change;
- keep the frozen clean configuration unchanged as the baseline;
- avoid a new parameter or prompt sweep;
- avoid human labels during inference.

If PR 88 contains no justified addition, record that finding and do not invent
an experiment to fill the slot.

## Score and decide

Use the same relevant scoring fields as the frozen clean result. Keep local VLM
scores separate from end-to-end rally results.

Report:

- what changed in the hybrid;
- which cases changed;
- whether server, serve state or contact timing improved;
- whether any new unsupported exact claims appeared;
- whether complete rally annotations improved;
- whether the evidence supports keeping the hybrid.

Freeze this report as the end of the planned VLM follow-up series.
