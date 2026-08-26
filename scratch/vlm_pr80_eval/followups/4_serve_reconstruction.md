# Follow-up 4: larger serve reconstruction experiment

## Bottom line

Use the **chosen VLM only**.

The 32-case model-selection gate has already shown whether the basic serve task has promise. Now test which automatic support helps, then widen the useful result.

Everything used during inference must be automatic. Human labels are scoring-only.

## Keep the same rally-start window

Use the automatic boundary method frozen in Follow-up 2 unless the 32-case evidence shows a clear problem with it.

Do not hand-tune individual rallies.

## Test a small number of evidence levels

Use the same clips and chosen model.

Compare:

1. video + selected PySceneDetect cut;
2. add compact automatic facts from shuttle/player/court evidence;
3. also add current pipeline guesses, clearly labelled as fallible.

Do not send raw keypoint arrays or long coordinate tables.

Keep added evidence only when it improves measured results.

## Score on the reviewed rally starts

Report separately:

- server identity;
- visible / off-frame / broadcast-omitted / unclear;
- contact timing when visible;
- false frame guesses when contact was not visible.

Do not collapse these into one score.

## Widen if useful

If a configuration is genuinely promising, run **server attribution** across all three labelled fixtures:

- `sset_01`;
- `sset_15`;
- `sset_21`.

Server identity can be scored across the full rally population without pretending every ShuttleSet first-stroke frame is perfect physical-contact truth.

If exact timing remains promising, extend the human rally-start truth before making a population claim about timing.

Then test end-to-end value:

1. replace server attribution only;
2. add VLM serve time only when contact is visible;
3. use both.

Rerun normal rally logic. Better complete rally annotation is the result that matters.

## PR 88 comes later

Do not use PR 88's detailed mechanisms to design the clean serve experiment.

After this result is frozen, read PR 88 and test at most one simple hybrid justified by evidence it contains.

Do not reopen the clean experiment's tuning after reading it.
