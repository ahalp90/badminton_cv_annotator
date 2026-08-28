# Follow-up 2: choose one VLM on rally starts

## Bottom line

Use the existing 32 reviewed rally starts as the final model-selection gate.

Run **both models on the same automatically built clips with the same clean prompt**. Then choose one VLM for all later work.

**Status: complete.** InternVideo3 is the clean-interface choice and the model
to use first in the remaining follow-ups. Neither model was dependable for
serve-state classification or contact timing. See
[`../2_final_model_gate.md`](../2_final_model_gate.md).

## Build the clips

Use only automatic pipeline evidence.

For each rally start:

1. look near the first few accepted contact guesses;
2. prefer a nearby PySceneDetect cut followed by sustained court-view evidence;
3. build the same short dense source-rate clip for both models;
4. if no useful cut exists, fall back to the earliest accepted contact and record that fallback.

Do not use human rally-start labels to select or build clips.

## Ask both models

Use the same prompt:

```json
{
  "server": "top | bottom | unclear",
  "serve_state": "visible | off_frame | broadcast_omitted | unclear",
  "contact_frame": "integer | null"
}
```

`contact_frame` must be null unless physical contact is visible.

Start with video + selected cut only. Do not add raw keypoints or PR 88 mechanisms.

If the models are genuinely too close to choose, one compact automatic-evidence arm may be run on both before the decision. Do not start a large prompt sweep.
That arm is a contingency, not a step to run before the clean comparison. Follow
the wording and evidence limits in
[`compact_automatic_evidence.md`](compact_automatic_evidence.md).

## Score

The 32 reviewed cases contain:

- 19 visible serve contacts;
- 8 off-frame contacts;
- 4 broadcast-omitted serves;
- 1 unclear case.

Score separately:

- server identity;
- serve-state classification;
- contact timing when visible;
- false exact-frame claims when contact was not visible;
- abstention rate.

Inspect the mistakes, not just the totals.

## Choose one model

Use this result together with the frozen evidence from:

- the full scene comparison;
- contact timing;
- player attribution;
- shuttle-track checking.

Make a qualitative decision as well as a numeric one:

> Which model's demonstrated strengths and failure modes best fit the remaining automatic annotator work?

Choose **one model** and freeze the choice before Follow-up 3.
