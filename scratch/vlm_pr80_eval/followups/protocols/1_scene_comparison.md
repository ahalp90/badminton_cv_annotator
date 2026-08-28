# Follow-up 1: fill the missing Qwen scene benchmark

## Bottom line

Run Qwen on the exact same full short-scene benchmark already run with Intern.

This fills the biggest missing apples-to-apples comparison. It does **not** make the final model choice by itself.

**Status: complete.** The result is recorded in
[`../1_scene_comparison.md`](../1_scene_comparison.md). Intern is the
provisional preference. Follow-up 2 still makes the final model choice.

## Run

Run Qwen's existing 120-frame local scene prompt on the same 463 clips.

Keep fixed:

- clip selection;
- 120 consecutive frames;
- prompt;
- human truth;
- scoring.

Use the frozen Intern result for comparison.

On the 347 meaningful cases, report separately:

- standard-view live: kept / total;
- unusual-view live: kept / total;
- targets containing non-live footage: flagged for further checking / total;
- pure replay: flagged for further checking / total.

Do not merge the two live groups into one headline number.

## Compare

Inspect representative mistakes from both models.

Use the completed contact and tracker results as supporting context, but keep every claim task-specific.

Answer:

- Where is Qwen clearly better?
- Where is Intern clearly better?
- Which errors look dangerous for later annotator work?
- Is there a provisional model preference?

Do not turn this into a general model personality claim.

## Decision

Freeze the Qwen scene result.

Record a **provisional preference only**.

The final model choice happens in Follow-up 2, after both models are tested on rally-start serve reconstruction.
