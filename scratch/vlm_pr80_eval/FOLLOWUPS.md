# Follow-up plan

## Bottom line

Do the remaining work in this order:

1. **Audit and freeze the completed-investigation docs.**
2. **Fill the missing Qwen full-scene benchmark.** This gives a fair scene comparison, but only a provisional model preference.
3. **Run the 32-case rally-start gate on both models, then choose one VLM.** This is the final model-selection step.
4. **With that one model, test whether today's system can isolate a tiny zero-error dataset subset.**
5. **With that one model, run the larger serve-reconstruction experiment.**
6. **Only after the clean serve result is frozen, reconcile with PR 88.** Test at most one simple hybrid.

Stop a branch when its stated gate fails. Do not keep adding variants because compute is available.

Use [`WRITEUP_PRINCIPLES.md`](WRITEUP_PRINCIPLES.md) for every report.

## Record boundary

The completed investigation is frozen in the root write-ups and
`experiments/results/`. Do not revise those findings in response to later
experiments. Change them only when the user explicitly requests a historical
correction.

Keep follow-up results under [`followups/results/`](followups/results/). After
each follow-up experiment, update its report and stop before starting the next
step unless this plan explicitly says otherwise.

## Contents

- [0. Audit the completed record](#0-audit-the-completed-record)
- [1. Fill the missing Qwen scene benchmark](#1-fill-the-missing-qwen-scene-benchmark)
- [2. Choose one VLM on the 32 rally starts](#2-choose-one-vlm-on-the-32-rally-starts)
- [3. Test a precision-first dataset route](#3-test-a-precision-first-dataset-route)
- [4. Run the larger serve experiment](#4-run-the-larger-serve-experiment)
- [5. Reconcile with PR 88](#5-reconcile-with-pr-88)

## 0. Audit the completed record

Before new inference, audit the current documentation against the retained prompts, manifests, outputs, scoring code and human truth.

Freeze the corrected docs before continuing.

Detailed prompt: [`followups/0_audit_docs.md`](followups/0_audit_docs.md)

## 1. Fill the missing Qwen scene benchmark

Run Qwen on the **same 463 short scene clips** already run with Intern. Use the same 120-frame input, prompt, truth and scoring.

Compare the models on standard-view live play, unusual-view live play, non-live footage and replay. Inspect representative mistakes.

This can produce a **provisional preference**, but do not permanently choose the model yet. The next important task—serve reconstruction—is different.

Detailed spec: [`followups/1_scene_comparison.md`](followups/1_scene_comparison.md)

## 2. Choose one VLM on the 32 rally starts

Run the same clean rally-start serve task on **both models** using the existing 32 reviewed cases.

Use the same automatically built clips and the same prompt. Compare:

- who served;
- visible / off-frame / broadcast-omitted / unclear;
- contact timing when visible;
- abstentions and dangerous errors.

Combine this with the completed contact, tracker and scene evidence. Then make a qualitative choice of **one model** for all later work.

Detailed spec: [`followups/2_final_model_gate.md`](followups/2_final_model_gate.md)

## 3. Test a precision-first dataset route

With the chosen model, ask:

> Can today's automatic signals keep a non-empty set of rallies with zero observed annotation errors on held-out labelled data, even if almost everything is discarded?

Detailed spec: [`followups/3_precision_first_dataset.md`](followups/3_precision_first_dataset.md)

## 4. Run the larger serve experiment

With the chosen model, test which automatic support helps serve reconstruction, then widen server attribution across all three labelled fixtures if the 32-case gate was promising.

Detailed spec: [`followups/4_serve_reconstruction.md`](followups/4_serve_reconstruction.md)

## 5. Reconcile with PR 88

Do this only after the clean serve experiment is frozen.

Read PR 88, identify any useful automatic evidence the new route did not use, and test **at most one simple hybrid**.

Do not reopen the clean experiment's tuning after reading PR 88.
