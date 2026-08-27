# Possible next work

The main thing to try next is the new contact model. The question is simple: does it add enough missing contacts to give us more complete rallies?

There is also one small scene pilot worth running: when there is a camera cut, does the same rally continue on the other side?

The shuttle-track VLM does not need another standalone experiment. It rejected 16 of 18 known tracker hallucinations in the existing audit. Bring it back only if tracker errors are still showing up once the contact model is in the pipeline.

None of this needs new human scene labels.

## 1. Test the new contact model

Start once the binary contact model has a fixed checkpoint and a score for every proposed contact.

### Does it add missing contacts?

The current candidate pool tops out at 56 complete rallies out of 292 on `sset_01`, `sset_15`, and `sset_21`, using a ±10-frame match.

Run this again with the new contacts added:

1. reproduce the current 56/292 result;
2. add the new model's proposed contacts;
3. compare the new ceiling at ±5, ±10, and ±15 frames.

Use 30 fps as the reference and scale the frame tolerance to each source video's frame rate.

Use ±10 for the stop/go decision. If the expanded pool does not beat 56/292, stop. A later filter cannot recover contacts that were never proposed.

### How good is the contact model on its own?

If the ceiling improves, sweep a small set of contact-score thresholds on the development videos.

For each threshold, run the retained contacts through the normal pipeline:

- player attribution;
- contact ordering;
- landing;
- point outcome.

Pick the threshold on the development videos, freeze it, then run it unchanged on the remaining ShuttleSet videos.

Call that the **contact-only baseline**.

Count a rally as **complete** only if all of these are correct:

- contact count;
- contact times;
- player order;
- server;
- point outcome.

Report:

- **precision:** complete retained rallies / all retained rallies;
- **coverage:** complete retained rallies / all annotated rallies.

Any VLM-assisted version only counts as better if it either improves precision without losing complete rallies, or keeps precision the same and improves coverage.

Retaining nothing does not count as an improvement. Neither does gaining precision by losing complete rallies. A tie on both precision and coverage is not an improvement.

## 2. Run one small cross-scene continuity pilot

Ask one question at each camera cut:

> Looking at the video before and after this cut, is it still the same rally?

Use three answers:

- `same_rally`
- `different_rally`
- `unclear`

Build the cases from the existing ShuttleSet rally annotations and saved PySceneDetect cuts. Score `same_rally` only when annotated contacts on both sides of the cut belong to the same rally. Score `different_rally` only when the annotations establish that they belong to different rallies. Leave any ambiguous cut unscored.

Show the model video on both sides of the cut. Keep rally boundaries and contact times out of the model-facing data.

Start with 24 cases:

- 12 `same_rally`;
- 12 `different_rally`.

Spread them across at least three videos if possible.

Use a couple of separate smoke-test cases first to sort out the clip layout and parser. Then freeze the 24 case IDs.

Pass the pilot only if InternVideo3 gets at least 7/12 right in both classes. Count invalid and `unclear` replies as wrong.

If it passes, run the same prompt over all ShuttleSet cuts that can be scored automatically.

If it fails, stop there.

If it passes, continuity can be tried later as another input to the contact experiment. It still has to beat the contact-only baseline on complete rallies.

## 3. Only use the tracker VLM if the errors point to it

The tracker experiment is already done. InternVideo3 rejected 16 of 18 known hallucinations when the claimed shuttle path was shown in a slow, enlarged clip.

That is useful, but it is only a warning signal. The comparison paths were not independently confirmed as correct tracks.

Once the contact-only baseline exists, look at the wrong development records.

If tracker hallucinations are not showing up, leave the tracker VLM out.

If they are showing up, reuse the existing marked view and [tracker prompt](prompts.md#4-tracker-validity) on those kinds of cases.

Do not rerun the original 18-case audit unless the model, prompt, or marked rendering changes.

Judge the tracker signal end to end: does it give us more complete rallies without removing records the contact-only pipeline already got right?

## Things not worth another pass

The previous experiments are enough to stop on these:

- **More live/replay prompt variants or broad context.** Short, long, joined, and direct earlier-action inputs all failed to give us a safe scene rule.
- **More serve context, denser frames, or answer-shaped hints.** Longer inputs did not improve server identification, denser frames changed no answers, and supplied contact hints were mostly copied.
- **More aggressive filtering of the current Issue 103 records.** Only 1 of 311 records was already complete. Filtering cannot create missing contacts or repair rally boundaries.
- **Scene labels or exclusion masks as hard vetoes.** They improved some routing numbers but also removed useful live material.
- **The clean-before-marked tracker prompt.** It rejected 11 of 18 comparison paths and mostly made the model more conservative.

## Implementation notes

### Contact-model input

Use one JSONL row per proposed contact:

```json
{
  "video_id": "sset_01",
  "frame": 12345,
  "contact_score": 0.91
}
```

Keep each `(video_id, frame)` pair unique. Keep `contact_score` between 0 and 1.

Save the model checkpoint, command, code revision, and input provenance alongside the file.

The two planned tools under `experiments/` are:

- `prepare_contact_model_cleanup.py` — validate the scores and join them to the frozen candidate and rally records;
- `score_contact_model_cleanup.py` — reproduce the current ceiling, sweep contact thresholds, and run the selected rule through whole rallies.

Keep model inputs and scoring truth separate:

```text
cases/inference/manifest.json
cases/scoring/truth.json
cases/scoring/candidates.jsonl
cases/scoring/provenance.json
```

A three-video preparation run could look like this:

```bash
PYTHONPATH=scratch/vlm_pr80_eval:src \
~/.venvs/badminton-cicd/bin/python -m experiments.prepare_contact_model_cleanup \
  --candidates-jsonl CONTACT_MODEL.jsonl \
  --artifacts-root ANNOTATOR_ARTIFACTS \
  --repo-root REPO \
  --video sset_01 \
  --video sset_15 \
  --video sset_21 \
  --out RUN_ROOT/contact_cleanup/cases
```

First reproduce the normal pipeline output and the 56/292 ceiling.

Then compare contact-score thresholds using the same whole-rally boundary as `evaluate_rally_cleanup.py`.

Once we pick a threshold on development data, save it as JSON and freeze it before evaluation. Include:

- schema version;
- contact threshold;
- development input hashes;
- creation command;
- code revision.

Load that frozen rule for evaluation. Do not sweep thresholds again on evaluation data.

### Cross-scene pilot

Reuse the existing pieces:

- `rally_opening_window_join.py` for joining automatic spans to saved cuts;
- `build_multiscale_trials.py` for cut intervals and source-frame maps;
- the existing pinned InternVideo3 launchers.

Add one small adapter that:

1. selects balanced continuity cases;
2. renders the two sides of the cut;
3. writes the three-way truth separately.

Do not build a full new ShuttleSet experiment stack unless the 24-case pilot passes.

For each model-facing case, save:

- case ID;
- video ID;
- cut frame;
- source-frame map;
- clip hash;
- prompt revision.

In the separate scoring row, save:

- answer;
- supporting ShuttleSet rally IDs;
- reason for any unscored cut.

### If tracker integration becomes relevant

`build_track_trials._write_track_clip()` already renders the slow, enlarged, marked view.

Reuse that renderer, not the old case selector.

Choose tracker cases from real contact-only development errors.

Do not bring back the speculative `tracker_risk` formula as the default routing rule.

Keep `yes`, `no`, and `unclear` separate if tracker output is combined with other signals.

### Checks

Before trusting the result, check that:

- the old candidate pool still gives the 56/292 ceiling;
- the unmodified natural replay still matches the frozen annotator artefact;
- development and evaluation videos do not overlap;
- human truth never appears in a model-facing manifest;
- source-frame maps and PySceneDetect cut IDs survive rendering;
- VLM outputs match the expected cases, model, prompt, and clip hashes;
- invalid and `unclear` answers are still visible in the scores.

Report these per video and overall:

- complete-rally precision;
- coverage;
- retained records;
- complete records;
- baseline-correct rallies lost after adding any VLM signal.

## Earlier evidence

- [Scene comparison](followups/1_scene_comparison.md)
- [Precision-first filtering](followups/3_precision_first_dataset.md)
- [Automatic serve support](followups/4_serve_reconstruction.md)
- [Longer rally-opening context](followups/6_rally_opening_context.md)
- [Long-context experiments](experiments/long_context_experiments.md)
- [First VLM cleanup experiments](FIRST_EXPERIMENTS.md)
