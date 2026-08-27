# Follow-up 4: larger serve reconstruction experiment

## Bottom line

**Status: complete.** Neither enhanced InternVideo3 arm passed the predeclared
gate. Qwen and the wider fixture experiment were not run. See
[`../4_serve_reconstruction.md`](../4_serve_reconstruction.md).

The frozen plan was to start with the chosen VLM, **InternVideo3**, only. Qwen
would be added only if the predeclared model-reassessment gate passed.

The 32-case model-selection gate has already shown whether the basic serve task has promise. Now test which automatic support helps, then widen the useful result.

Everything used during inference must be automatic. Human labels are scoring-only.

## Keep the same rally-start window

Use the automatic boundary method frozen in Follow-up 2 unless the 32-case evidence shows a clear problem with it.

Do not hand-tune individual rallies.

## Test a small number of evidence levels

Use the same clips and chosen model.

Compare:

1. the frozen Follow-up 2 result for video + selected PySceneDetect cut;
2. add compact, plain-language observations from shuttle/player/court evidence;
3. also add the current pipeline's proposed server and contact time, clearly
   labelled as fallible conclusions.

Do not rerun the first arm. It is the clean Intern baseline already retained
for these exact clips.

Do not send raw keypoint arrays or long coordinate tables.

Describe what the automatic analysis observed in the supplied clip. Do not
expose unexplained scores or mask names. Keep the facts short, local and
explicitly fallible. See
[`compact_automatic_evidence.md`](compact_automatic_evidence.md).

Do not add an internal live or replay label.

Keep added evidence only when it improves measured results.

## Recheck the model choice only if the interface changes

Follow-up 2 remains the completed comparison of both models with video and the
selected cut. A better evidence format may change which model is preferable for
future use, but it does not change that historical result.

Run one Qwen confirmation on the best enhanced Intern arm only when that arm is
parse-complete and reaches at least one of these predeclared improvements over
clean Intern:

- server identity improves by at least four cases, from 23 to 27 or more;
- serve state improves by at least four cases, from 19 to 23 or more, and
  unsupported exact-frame claims fall from 13 to 9 or fewer;
- visible contact timing within project tolerance improves by at least four
  cases, from 1 to 5 or more.

The enhanced arm must also keep server and serve-state correctness within two
cases of the clean result. Unsupported exact-frame claims must not increase.

If this gate passes, freeze the evidence format before running Qwen. Compare
the two models once on that identical format. Record any changed operational
model choice as a new result rather than revising Follow-up 2.

If both enhanced arms pass, choose one before the Qwen run. Prefer, in order:
more correct servers; more correct serve states; fewer unsupported exact-frame
claims; then more visible contacts within project tolerance. Use the simpler
observations-only arm if the scored results are otherwise tied.

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
