# Later experiment: contact-model and VLM cleanup

## Bottom line

Run this experiment only after the planned binary contact model has produced a
frozen score for every candidate. The VLM should inspect tracker-risk cases
only. It should supply one signal to a final rule, rather than delete contacts
by itself.

This is a frozen experiment specification, not a command that can run today.
The contact model's output format does not exist yet. Once it does, add one
standalone adapter described below. Do not change `src/` merely to guess that
future format.

The experiment succeeds only if it improves complete rally records over the
same contact-model baseline. Candidate-level precision is diagnostic. It is
not the acceptance measure.

## Data use

- Develop the rule on `sset_01`, `sset_15`, and `sset_21`.
- Freeze the rule before checking the remaining ShuttleSet and SSet22 material.
- Use the existing ShuttleSet contact and rally annotations.
- Use the existing human scene timelines.
- Use the 18 existing human-confirmed tracker hallucinations as a separate
  negative check. They contain no positive tracker truth.
- Do not add new human labels.

## Input contract

Write one JSONL row per raw candidate. Each row needs:

```json
{
  "video_id": "sset_01",
  "frame": 12345,
  "track_start_frame": 12340,
  "track_end_frame_exclusive": 12349,
  "contact_score": 0.91,
  "existing_filter_kept": true,
  "tracker_risk": true
}
```

`contact_score` is the frozen binary contact-model score. The track interval is
the exact claim shown in the marked clip. `tracker_risk` must be derived from
existing tracker history, masks, and scene evidence. Record the exact rule and
all thresholds beside the file. Do not choose thresholds from the wider
generalisation run.

## One adapter needed after the model exists

Add `prepare_contact_model_cleanup.py` and
`score_contact_model_cleanup.py` under this directory. They must:

1. validate unique `(video_id, frame)` candidate keys and scores from 0 to 1;
2. join each row to the frozen raw-candidate and rally records;
3. render routed rows through the existing `_write_track_clip()` marked view;
4. write the normal truth-blind manifest and separate scoring sidecar;
5. replay contact-only and combined keep rules through the same natural
   annotator boundary used by `evaluate_rally_cleanup.py`;
6. write every threshold, input hash, VLM reply, keep reason, and final metric.

`build_track_trials.py` cannot perform this join. It deliberately accepts only
the old human-negative audit and structural controls. Reuse its renderer, not
its case-selection function.

Use this routing rule for the first comparison:

```text
tracker_risk =
    inpaint_code != NO_FLAG
    or track_visible is not true
    or wrist_near is not true
    or proximity_ok is not true
```

An unmeasured value therefore routes to the VLM. A contact bypasses the VLM
only when this rule is false and its contact score passes the chosen baseline
threshold.

The preparation command must have this shape:

```bash
PYTHONPATH=scratch/vlm_pr80_eval:src \
~/.venvs/badminton-cicd/bin/python -m experiments.prepare_contact_model_cleanup \
  --candidates-jsonl CONTACT_MODEL.jsonl \
  --artifacts-root ISSUE103_ARTIFACTS \
  --repo-root REPO \
  --scene-labels-dir SCENE_LABELS \
  --video sset_01 \
  --video sset_15 \
  --video sset_21 \
  --out TRIAL_ROOT/cases
```

It must write:

```text
cases/inference/manifest.json
cases/scoring/truth.json
cases/scoring/candidates.jsonl
cases/scoring/provenance.json
```

After the normal InternVideo3 launcher has written a fresh attempt directory,
the scoring command must have this shape:

```bash
PYTHONPATH=scratch/vlm_pr80_eval:src \
~/.venvs/badminton-cicd/bin/python -m experiments.score_contact_model_cleanup \
  --cases TRIAL_ROOT/cases \
  --attempts TRIAL_ROOT/attempts/intern01 \
  --artifacts-root ISSUE103_ARTIFACTS \
  --threshold-step 0.05 \
  --rule-out TRIAL_ROOT/scores/frozen-rule.json \
  --out TRIAL_ROOT/scores/development.json
```

The score file must contain the contact-only threshold sweep, the combined-rule
sweep, the natural replay identity check, and every metric listed below. Once a
rule is frozen, repeat preparation and scoring on each available wider video
with the frozen rule file:

```bash
PYTHONPATH=scratch/vlm_pr80_eval:src \
~/.venvs/badminton-cicd/bin/python -m experiments.score_contact_model_cleanup \
  --cases GENERALISATION_ROOT/cases \
  --attempts GENERALISATION_ROOT/attempts/intern01 \
  --artifacts-root ISSUE103_ARTIFACTS \
  --frozen-rule TRIAL_ROOT/scores/frozen-rule.json \
  --out GENERALISATION_ROOT/scores/result.json
```

The parser must make `--threshold-step` and `--rule-out` development-only. It
must reject either when `--frozen-rule` is present. The rule file must be JSON
with a schema version, contact threshold, VLM handling rule, tracker-risk rule
version, development input hashes, and creation command. The generalisation
scorer must copy that whole object into its output before applying it.

## Experiment sequence

### 1. Check the new candidate ceiling

Add the new model's proposed contacts to the existing raw candidate pool. Count
how many ShuttleSet rallies have every annotated contact represented within
±5, ±10, and ±15 base-30 frames.

The current raw-candidate ceiling is 56 complete rallies at ±10. If the new
model does not raise this meaningfully, stop. A cleanup stage cannot repair the
missing evidence.

### 2. Establish the contact-only baseline

On the three development videos, try contact-score thresholds from 0 to 1 in
steps of 0.05. Run every retained set through the normal attribution,
alternation, landing, and point stages. Keep the simplest threshold that gives
the best complete-rally precision, then the best rally coverage when precision
ties.

This baseline is what the VLM must beat.

### 3. Route only tracker-risk cases to InternVideo3

Render the marked, slow, enlarged tracker clip from `build_track_trials.py`.
Use the existing direct tracker question. Record `yes`, `no`, or `unclear` as a
feature. Do not use the clean-before-marked request: it rejected 11/18
orientation controls.

High-confidence contacts that are not tracker-risk cases bypass the VLM.

### 4. Choose one simple final rule

Try only monotonic rules that become more willing to keep a contact when the
contact score or other positive evidence rises. Suitable forms are:

- a higher contact threshold when InternVideo3 says `no`;
- keep any very-high-score contact, otherwise require InternVideo3 not to say
  `no` for tracker-risk cases;
- the same rule with existing scene or exclusion masks as a veto.

Treat `unclear` as its own value. Do not silently turn it into `yes` or `no`.
Rank rules by complete-rally precision first and rally coverage second. Freeze
one rule before the wider run.

### 5. Check wider material

Run the frozen contact-only baseline and frozen combined rule over the
available remaining ShuttleSet and SSet22 material. Report both wherever
existing annotations support the score. Do not add manual labels.

## Required report

For each baseline and final rule, report:

- rallies with the exact contact count;
- rallies with the correct alternating player sequence;
- rallies with the correct point outcome;
- structurally usable rallies at ±5, ±10, and ±15 base-30 frames;
- total retained rally coverage;
- VLM calls, invalid replies, and `unclear` replies;
- the 18 tracker-hallucination decisions, kept separate from the rally score.

Precision comes first. A smaller set of correctly counted rallies is useful.
A larger set containing wrong contact counts is not.

## Stop rules

Omit the VLM if any of these is true:

- the new contact model does not improve the complete-candidate ceiling;
- the combined rule does not beat the contact-only baseline on complete-rally
  precision;
- the combined rule gains precision only by discarding most otherwise usable
  rallies;
- the apparent gain does not survive the wider material.

If the combined rule passes, the next step is a full fixture run using the same
frozen scripts and thresholds. No production `src/` change is needed until that
run supports integration.
