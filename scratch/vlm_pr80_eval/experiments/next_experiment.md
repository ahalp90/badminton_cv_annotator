# Multiscale experiment: protocol and record

## Status

The experiment is complete. Broad context did not improve the close check in
any tested form.

- The paired 90- and 120-second broad pilot completed on 12 suspected spans.
  Ninety seconds tied or won. Neither model found a safe bypass.
- The three-arm close pilot completed on 19 identical 120-frame clips.
  InternVideo3 short-only won: 8/11 unsafe targets caught and 7/8 routine-live
  targets retained. Deterministic and broad VLM facts hurt.
- The joined input of 80 sparse broad frames plus the same 120 close frames
  caught only 2/11 unsafe targets. It was worse than short-only.
- Simple Qwen and Intern combinations gave no gain.
- The Step 4 wide run completed all 463 eligible target segments. On the 347
  material targets, Intern short-only kept 92.0% of routine live but caught
  only 44.7% of unsafe targets. Twenty of its 26 unsafe misses were pure replay.
- A stronger replay warning regressed on the fixed pilot.
- A direct 120-frame earlier-action plus 120-frame target comparison returned
  `different_action` for all 46 pairs. It changed none of the 24 available
  material unsafe parent mistakes.
- Step 5 is deferred. The current scene rule did not earn a wider unlabelled
  run. The next useful input change is the planned contact detector.

The rest of this document keeps the original test rules and decision path. It
is a reproducible record, not a list of untried recommendations.

## Bottom line

The intended two-scale design was tested as separate calls, parsed facts, one
joined input, and one direct action pair. None beat the plain close view. The
wide close-view result then showed too much replay blindness for an automatic
keep rule.

Keep this document as the reproducible protocol and decision record. Do not
repeat the same broad-context forms. The scene part was tested before the
planned contact model. That model is still needed for the final contact-cleanup
comparison, not for scene routing.

## The question

Can broad, cut-aware context tell a short VLM check what kind of broadcast
sequence it belongs to, without making that short check less accurate?

No. Parsed broad facts, broad frames in the same input, stronger replay wording,
and direct earlier-action pairs all failed to improve unsafe-target recall.

The intended flow is:

```text
suspected rally span + PySceneDetect cuts + existing signals
                         |
                         v
             broad broadcast reading
                         |
              small, parsed fact record
                         |
                         v
          short scene, boundary, or event check
                         |
                         v
             simple automatic keep rule
                         |
                         v
              complete-rally evaluation
```

Use separate model calls first. A parsed JSON record is easier to inspect than
conversation memory. A same-session follow-up remains an optional later trial
if the separate calls show that broad context is useful.

## Data and evidence

Develop on all three existing scene-labelled fixtures: `sset_01`, `sset_15`,
and `sset_21`. Use the wider ShuttleSet and SSet22 material later to check
generalisation wherever existing annotations support an automatic score.

Each case should start from existing automatic evidence:

- suspected rally spans from
  `annotator_result.json.gz["result"]["spans"]`;
- persisted PySceneDetect intervals from
  `stages/court/<video>/court_evidence.json.gz["raw_cuts"]`, produced by
  `annotator.court_evidence.build_raw_cut_intervals()`;
- court presence, exclusion masks, tracker visibility, contact candidates, and
  other existing named signals;
- the three existing human scene timelines, used only to select and score
  cases;
- ShuttleSet rally and contact annotations, used only for final scoring.

Do not show human labels to the VLM. Do not create new labels.

## What stays fixed

- Precision of complete rally records comes first. Coverage comes second.
- Score exact contact count, alternating player sequence, point outcome, and
  usable timing at ±5, ±10, and ±15 base-30 frames.
- Scene accuracy and candidate accuracy are diagnostic. They cannot replace
  complete-rally scoring.
- Keep model revisions fixed while comparing prompts or clip layouts.
- Change one important factor per trial.
- Keep every prompt, input manifest, frame map, model reply, parser result, and
  score.
- Do not use VLM answers to choose the evaluation cases or manufacture truth.
- Keep experimental code under `scratch/vlm_pr80_eval/experiments/`. Avoid
  changes to `src/`; stop and explain any small source change that becomes
  genuinely necessary.
- Remote runs may take as long as the evidence requires. Keep them bounded by
  one recorded question and stop only when the result can change the next step.

## Step 1: build a fixed cut-aware case set

Add a standalone builder for multiscale cases. It should reuse the existing
fixture loader and clip renderer where sensible.

Load the persisted cut intervals. Do not rerun PySceneDetect merely to rebuild
an input that the pipeline has already recorded.

For every selected suspected span, record:

- its original start and end frames;
- the two detected cuts before it, cuts inside it, and the two cuts after it;
- paired 90- and 120-second broad context windows;
- the short target windows described below;
- all automatic priors as separate fields;
- the source and derived-file hashes.

The first pilot should contain 12 development cases:

- four clear live spans;
- four replay or cutaway spans;
- four spans that cross a scene change or have mixed scene labels.

Balance across `sset_01`, `sset_15`, and `sset_21`. Freeze these case IDs before
running a model. Human scene labels may balance the sample, but must remain in
a separate scoring file.

Keep the new code small and separate:

- `build_multiscale_trials.py` builds broad clips, manifests, frame maps, and
  the separate truth sidecar;
- `build_detail_from_context.py` accepts a complete broad-pass result and
  builds the paired short-check arms;
- `score_multiscale_trials.py` scores both passes and replays final choices
  through complete rallies;
- `multiscale_prompts.py` holds the two prompt contracts.

Add focused tests for truth separation, source-frame order, required cut and
span frames, strict segment IDs, paired-arm identity, and natural pipeline
replay before any VLM decision is applied.

### Command and output contract

The broad-case builder should have this shape:

```bash
PYTHONPATH=scratch/vlm_pr80_eval:src \
python -m experiments.build_multiscale_trials \
  --artifacts-root ANNOTATOR_ARTIFACTS \
  --repo-root REPO \
  --scene-labels-dir SCENE_LABELS \
  --video sset_01 \
  --video sset_15 \
  --video sset_21 \
  --pilot-cases 12 \
  --context-seconds 90 \
  --context-seconds 120 \
  --max-frames 96 \
  --out RUN_ROOT/context/cases
```

It should write:

```text
context/cases/inference/manifest.json
context/cases/inference/clips/
context/cases/scoring/truth.json
context/cases/scoring/provenance.json
```

The manifest must contain the source-frame map, cut and span IDs, automatic
priors, clip hashes, and prompt inputs. Truth, human notes, and scene fractions
belong only under `scoring/`.

Run each model through the existing remote launcher into a new attempt
directory. Then build the three paired detail arms:

```bash
PYTHONPATH=scratch/vlm_pr80_eval:src \
python -m experiments.build_detail_from_context \
  --context-cases RUN_ROOT/context/cases \
  --context-attempts RUN_ROOT/context/attempts/SELECTED_MODEL \
  --context-seconds 90 \
  --backend SELECTED_MODEL \
  --source-video sset_01=/path/to/sset_01.avi \
  --source-video sset_15=/path/to/sset_15.avi \
  --source-video sset_21=/path/to/sset_21.avi \
  --out RUN_ROOT/detail
```

This should write one inference manifest for `short_only`, `deterministic`, and
`broad_facts`, plus one shared truth and provenance directory. All three arms
must use the same case IDs and clip hashes.

Score the broad and detail calls separately. Both scorers verify the saved
manifests, prompts, clip hashes, and attempt completeness:

```bash
PYTHONPATH=scratch/vlm_pr80_eval:src \
python -m experiments.score_multiscale_trials \
  --manifest RUN_ROOT/context/cases/inference/manifest.json \
  --truth RUN_ROOT/context/cases/scoring/truth.json \
  --attempts RUN_ROOT/context/attempts/SELECTED_MODEL \
  --backend SELECTED_MODEL \
  --out RUN_ROOT/scores/broad.json

PYTHONPATH=scratch/vlm_pr80_eval:src \
python -m experiments.score_detail_trials \
  --arm short_only=RUN_ROOT/detail/inference/short_only/manifest.json \
  --arm deterministic=RUN_ROOT/detail/inference/deterministic/manifest.json \
  --arm broad_facts=RUN_ROOT/detail/inference/broad_facts/manifest.json \
  --truth RUN_ROOT/context/cases/scoring/truth.json \
  --attempts RUN_ROOT/detail/attempts/SELECTED_MODEL \
  --backend SELECTED_MODEL \
  --out RUN_ROOT/scores/detail.json
```

The detail score includes all three paired comparisons. It fails if an arm is
missing a case or uses a different clip. Complete-rally replay remains a later
stage once a final automatic keep rule and contact-detector output exist.

## Step 2: make the broad view easy to read

The broad pass should see broadcast order, not a blurred uniform sample. Build
a storyboard-style video from the window:

- sample the whole window sparsely;
- add extra frames immediately before and after every PySceneDetect cut;
- label each cut-bounded segment with a simple ID such as `S03`;
- mark the proposed rally span as `TARGET`;
- show source time on each frame;
- use the same 96-frame starting budget for both source spans. Confirm that
  budget against both adapters before inference.

Always preserve the span boundaries and cut-adjacent frames. Use any remaining
frame budget for sparse context. For the first pilot, use only cases where both
durations fit one call. Record every excluded case and the reason. Do not give
one duration more calls than the other. Add paired splitting later only if fit
failures make the unsplit sample misleading. Store the exact source-frame map
in the truth-blind manifest.

Ask for one short record per segment, not one label per frame. The answer should
name the likely content (`live`, `replay`, `cutaway`, `other`, or `unclear`),
whether it repeats an earlier segment, and which segment needs closer review.
The prompt should state the detected cuts as fallible boundaries.

Use a strict JSON shape with a list of segment IDs, content labels, optional
repeat links, and `needs_close_check`. Reject unknown or duplicate IDs. Treat a
missing, invalid, or incomplete answer as `unclear`; it must not silently delete
a span.

Freeze the target reduction before inference. Only source-global segments that
overlap `TARGET` contribute to the target result. Expand each segment label
over its target frames and compare those frames with the human timeline.
`live-non-standard` truth counts as live for this main score. An invalid,
missing, or `unclear` label is wrong. Replay elsewhere in the broad window does
not count as finding replay in `TARGET`.

The automatic route is also target-only. Route to a close check when any target
segment is non-live, `unclear`, invalid, or marked `needs_close_check`. Bypass
only when every target segment is valid live footage and none requests a close
check. This route asks for more evidence; it does not delete a span.

Run both Qwen3-VL and InternVideo3 on the same 12 cases. Compare them with a
heuristic-only reading made from the existing signals. Do not combine the two
model answers yet.

Move on when one model returns valid structured answers for at least 11 of 12
cases and beats deterministic-prior routing on target-frame error. Report all
12 cases, including invalid and unclear replies. If 90 and 120 seconds are
close, widen both paired arms across all eligible spans before choosing. Prefer
90 seconds on a true tie because it costs less.

These are pilot gates, not claims of final quality.

## Step 3: test whether broad facts help a close check

Render one short clip for each doubtful segment or boundary. Start with four to
six seconds around a scene boundary. Use dense consecutive frames and mark the
target interval. Contact and tracker questions should continue to use their
existing two-second views.

Compare three paired arms on identical pixels:

1. the short clip alone;
2. the short clip plus deterministic facts such as cut positions and span
   membership;
3. the short clip plus those facts and the parsed broad-pass record.

The third arm must receive a small factual record, not the broad model's prose.
Label it as fallible context. Ask one narrow question, such as whether the
target is replay footage or whether the marked tracker path follows a shuttle.

Parse the answer conservatively. Accept the exact JSON form, the same JSON
inside one `json` Markdown fence, or one known label inside braces. These are
easy to recover without guessing. Reject prose, extra fields, unknown labels,
and mixed answers.

The broad pass is useful only if arm 3 beats arm 2. If arm 2 wins, retain the
cut and span priors but drop the broad VLM call. If arm 1 wins, keep the local
check independent.

## Step 4: widen the development run

The pilot selected one design for the wider run: InternVideo3, 120 consecutive
frames, and the short-only prompt. The other two prompt arms and the combined
visual input lost on the same pilot cases, so repeating them would spend GPU
time without testing a live question.

The wide builder selected all 311 eligible suspected spans in `sset_01`,
`sset_15`, and `sset_21`. No case failed the paired storyboard frame budget.
Splitting the 90-second targets at stored camera cuts produced 463 close clips.
The detail build used `--without-broad-attempts`, and the runner used
`--only-arm short_only`.

The wide run returned 463 usable replies. Its material score was 91.4%
safe-live precision, 92.0% routine-live recall, and 44.7% unsafe recall. The
strict 84.0% unsafe recall was dominated by target regions smaller than the
accepted boundary margin, so it is not the headline result.

For the keep/reject score, treat human `live` and `live-non-standard` as live.
Report them separately as a diagnostic. The human timelines label replay, but
do not say which earlier segment it repeats. Score a claimed repeat link only
as supporting evidence, not as ground truth. PySceneDetect cuts are supplied
priors, so this experiment does not pretend to measure their accuracy.

Complete-rally replay can follow once the contact detector has produced a frozen
candidate set and the final automatic keep rule exists. Its denominator is
every retained predicted span. A span is correct only when it maps one-to-one
to one truth rally and has the required contact count, alternating actors,
point result, and usable timing. Also report how many baseline-correct usable
rallies remain.

## Step 5: wider footage was not earned

The scene design did not pass the Step 4 gate, so it was not run across wider
unlabelled ShuttleSet or SSet22 footage. Such a run would add compute without a
reliable score or a safe rule.

If a later contact detector creates a materially different candidate set,
freeze these items before checking wider footage:

- the model used for each pass;
- both prompts;
- broad and short sampling layouts;
- routing and keep rules;
- every threshold;
- the exact code and input hashes.

Use only existing annotations; do not add scene labels. Report what can and
cannot be scored. If the complete-rally gain does not survive, leave the VLM
out of the cleanup path.

## Bounded self-directed iteration

The investigation may choose its next short trial from the table below. Each
cycle must state one hypothesis, change one factor, compare with its parent run,
and keep the change only when the recorded metric improves. Use the same frozen
pilot cases until a design earns a wider run.

Check each change on `sset_01`, `sset_15`, and `sset_21` separately. Keep it
when the target error count improves across the fixtures without hiding a clear
regression in one inside an aggregate gain.

| Observed failure | Next trial allowed |
|---|---|
| Broad view misses brief scene changes | Put more frames around the supplied cuts |
| Broad output collapses or becomes invalid | Use fewer segments per call or a smaller JSON shape |
| Replay is mistaken for live play | Extend context to include the earlier action and return to standard view |
| Live play is broadly rejected | Make the broad pass route uncertainty instead of vetoing the span |
| Broad facts make the short answer worse | Remove VLM-written facts and retain deterministic priors only |
| Local tracker errors remain | Improve temporal density or enlargement before changing wording |
| The models disagree | Select by development evidence; do not require agreement |
| Candidate scores improve but complete rallies do not | Stop that line of VLM work |

Stop a line after two sensible changes fail to beat its recorded parent. A new
line needs a new, evidence-led hypothesis. This keeps the search adaptive
without turning it into prompt guessing.

The investigation can continue without a user check-in while a trial stays
inside the campaign contract. Run length alone is not a reason to pause.
Pause for a production source change, a new manual label, public Git action,
or another material change of scope.

## Remote runs without repeated polling

Run each GPU job in `tmux` with an immutable run directory. The job should
write its log and `status.json`, then finish with a `DONE` or `FAILED` marker.

For unattended work, give one cheap Luna agent a single blocking
`tmux wait-for` connection. That agent reports only completion, failure, or timeout.
The main reasoning session should not repeatedly query the remote machine. If a
blocking wait is unavailable, estimate the runtime and make one delayed marker
check. Any later check should use a longer, irregular delay.

Always release the model and GPU process when a run ends, including after a
failure.

## Records to keep while running

Keep a compaction-safe campaign worklog outside the tracked experiment folder.
Its latest checkpoint must state the active run, last verified result, and next
action. Store heavy clips and raw replies under the ignored
`local_scratch/campaigns/` tree.

After each useful gate, copy a small public-safe result into this folder. The
final report should explain what changed, why it changed, and why the next
choice followed from the evidence.

## Later contact cleanup

Once the binary contact model has frozen candidate scores, continue with
[`contact_model_followup.md`](contact_model_followup.md). That experiment adds
contact score, tracker risk, and complete-rally selection to the best surviving
multiscale scene design.
