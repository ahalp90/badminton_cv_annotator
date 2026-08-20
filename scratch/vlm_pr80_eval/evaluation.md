# Why the PR 80 VLM results failed

## Bottom line

PR 80 does not show that InternVideo3 or Qwen3-VL cannot do the cleanup job.
It shows that both models failed a different and much harder job: transcribing a
broadcast into a complete scene timeline, with eight decisions for every sampled
frame and a rigid output code.

The strongest evidence is in the raw replies. InternVideo3 copied the prompt's
only example code until it hit the output limit. Qwen changed two characters in
that example and copied the other six, even where they contradicted the video.
This strongly suggests that the request and its example outweighed the pictures.
A no-example counterfactual was not run in PR 80, so the raw replies do not prove
which prompt feature caused the collapse by itself.

This was a test-design failure before it was a model failure. The follow-up
experiments gave both models short clips and one plain decision. Those fairer
tests removed the catastrophic repetition, but they did not produce a safe
general cleanup stage.

InternVideo3 is the only promising narrow component. With enlarged tracker
evidence, it rejected 16 of the 18 existing human-confirmed tracker
hallucinations. It accepted all 18 structural orientation controls. The
orientation controls are near human-labelled ShuttleSet contacts, but nobody
has labelled their tracker paths as real. Two obvious hallucinations still
passed. This result is useful evidence, not enough evidence for automatic
deletion.

The clean-pixel counterfactual rejected all 18 human-confirmed hallucinations,
but it also rejected 11 of 18 orientation controls. That request became
conservative rather than demonstrably accurate.

The most sensible next full experiment is after the planned small binary
contact model produces a better candidate set. Use InternVideo3's marked,
enlarged tracker decision as one fallible input to a simple final rule. Then
replay complete rallies through the normal attribution and scoring stages. Do
not put either VLM into production as a general keep/reject judge from the
current evidence.

## What the fair follow-up tests found

| Job | Best useful result | Ruling |
|---|---|---|
| Filtered contact timing | Qwen: 88.6% precision and 97.5% recall at ±15 frames on 60 balanced structural cases | Diagnostic only; structural timing is not visual truth |
| Complete-rally cleanup | Qwen removed 14/84 contacts; exact-count rallies stayed 4/12 | Reject the binary Qwen filter |
| Contact actor | Both models were near chance | Keep normal alternating attribution |
| Tracker validity | Intern: 11/12 development and 5/6 safety hallucinations rejected; all 18 proxy controls accepted | Promising advisory signal, unsafe deletion rule |
| Pure live versus replay/cutaway | Qwen kept 5/6 negatives; Intern missed every replay under the direct prompt | Reject both general scene filters |
| Explicit sequence prompt | Intern kept only 2/6 live controls; Qwen was unchanged | Reject; conservatism replaced replay blindness |

`Base-30` means frames normalised to a 30 FPS video. An orientation control is
a candidate near a ShuttleSet contact, used to check that the request is not
simply rejecting everything. It is not a human-confirmed real tracker path.
The marked request draws the tracker claim with a cyan ring. The clean request
shows the same enlarged pixels once without that ring, then once with it.

The current annotator also has a hard ceiling that prompting cannot remove.
Only 56 rallies can be made complete at ±10 frames by an oracle choosing from
every existing raw candidate. This count asks whether every ShuttleSet contact
in a rally has some raw candidate inside the margin. A cleanup model can remove
false contacts. It cannot invent the many missing contacts needed for a
high-coverage usable dataset.

## What PR 80 actually tested

The final request asked each model to assign five scene classes and seven other
attributes to every sampled frame. The reply had to be JSON containing one
eight-character code per frame.

InternVideo3 received a 20-minute clip in one call:

- 1,200 sampled frames at 1 FPS;
- 101,349 input tokens;
- 1,200 required output codes;
- 9,216 output tokens available.

It produced 1,316 copies of `LBRFRS9B`, then stopped in the middle of another
code at the token limit. The parser recovered the first 1,200 complete codes and
treated them as the result. Every source frame was therefore scored as `live`.

Qwen3-VL received one 10-second clip with 50 frames at 5 FPS. It returned 50
copies of `OBRFRS9G`, so every frame was scored as `other`. The clip contained
one human-labelled change from `live-non-standard` to `cutaway`.

These were not comparable runs. They used different durations and covered very
different amounts of labelled material. The Qwen run was one difficult boundary
example, not a general five-class test.

## Why the results were catastrophic

### 1. The test asked for the wrong product

The useful product is an automatic decision about an auto-annotator claim: keep
it, reject it, or mark it uncertain for a later rule. PR 80 instead asked the VLM
to reconstruct the whole broadcast timeline from scratch.

That discarded the auto-annotator's main advantage. The existing pipeline can
already find most events. The VLM only needs to check a small number of proposed
events and remove clear mistakes.

### 2. The only example taught a constant answer

The prompt's only worked example contained two `live` codes. Its first code was
`LBRFRS9B`.

- InternVideo3 copied that code exactly for its entire reply.
- Qwen returned `OBRFRS9G`. Its middle six fields still copied the example.

Those copied Qwen fields said `full_court`, `usable_standard`, and confidence
`0.9`, although the clip showed close views and Qwen's own scene field said
`other`. This is direct evidence that the code template anchored both replies.

### 3. The output was long, repetitive, and fragile

InternVideo3 had to keep the position of 1,200 nearly identical strings aligned
with 1,200 video frames. That is a serial bookkeeping task as much as a visual
task. The reply reached the output limit and collapsed into repetition.

The parser did not create the all-`live` result. That repetition is present in
the raw reply. The recovery rule did make a truncated reply look like a complete
prediction by accepting its first 1,200 codes.

### 4. Each code bundled too many decisions

Every code combined scene, phase, playback, view, continuity, data use,
confidence, and visible reason. Several fields overlap. Some combinations make
little sense, but the schema allowed them. Only the scene field mattered to the
headline accuracy score.

A cleanup decision does not need most of these fields. Asking for them creates
more ways to copy a template, contradict another field, or drift between frames.

### 5. The scene rules conflicted

An earlier InternVideo3 reply is useful evidence. On a short clip, it described
player close-ups and profile graphics that were genuinely visible. It called the
whole clip `cutaway`, although the player was actively warming up and the human
timeline called it `live-non-standard`.

The prompt said a player close-up was a cutaway. It also said real warm-up from
an unusual view was live non-standard. It did not say which cue should win when
both were present. The model followed the framing cue instead of the activity
cue. This looks like grounded viewing followed by a bad rule choice, rather than
an inability to see the clip.

The Qwen clip had the same problem. Both shots contained people and large
broadcast overlays. The prompt separately treated player close-ups as cutaways
and graphics as `other`, without a clear priority rule.

### 6. The VLM was denied the evidence that makes cleanup easy

Issue 38 asked what the existing pipeline could provide to make the VLM more
precise and efficient. The final prompt supplied the video, clip metadata, a
large sampled source-frame grid, and detected hard cuts.

It did not state the proposed event, its time, the source detector, the proposed
shuttle path, court evidence, Inpaint evidence, or nearby replay and scene
signals. The VLM therefore had to discover the question before answering it.

### 7. One sampling plan was used for two different visual jobs

Broadcast-scene cleanup needs several seconds of context around cuts. Shuttle
verification needs denser frames around one event so the model can see motion.
The Intern run sampled at 1 FPS, which is too sparse for shuttle events and can
shift short scene changes by about a second. The Qwen run sampled at 5 FPS, but
tested only one scene boundary.

These jobs should use separate evidence windows even if they share one final
keep/reject decision.

### 8. The score did not measure cleanup value

The headline score was frame-level macro F1 over five scene classes. Qwen's clip
contained only two of those classes, so even a perfect answer on the clip could
score at most 0.4 macro F1 under that calculation.

The metric that matters is the auto-annotator result after cleanup:

- how many false annotations were removed;
- how many real annotations were wrongly removed;
- final precision and recall, reported per video as well as overall.

The current 18-example shuttle-hallucination audit contains only known false
events. A model can appear perfect on it by rejecting everything. A fair test
must add matched real-event controls.

## What can be said about each model

### InternVideo3

The long call is unusable evidence about scene accuracy because its generation
collapsed. The earlier short reply shows that the model noticed real visual
details, but the prompt's overlapping rules led it to the wrong class. This is
consistent with the later bounded tests.

InternVideo3 can answer a narrow tracker-identity question when the evidence is
large enough. Its marked zoom request is the strongest candidate from this
investigation. It still confidently accepted two human-confirmed
hallucinations across 18 rows. Its replay judgement was poor, even with dense
target frames and surrounding broadcast context.

### Qwen3-VL

The short call was a genuine failure on one example. It is too small and too
ambiguous to support a general verdict. The whole-shard memory failure only
shows that a 262,144-token configuration does not fit on one L40. Short cleanup
calls fit and are the relevant use case.

The larger follow-up now supports a firmer ruling. Qwen was good at locating
structurally plausible contact times, but that did not improve complete rally
records. It accepted too many tracker hallucinations and nearly every replay
control. Qwen should not be the cleanup model for the present pipeline.

## Other limits in the benchmark

- Hard cuts were included in the prompt but were not scored as evidence in
  their own right.
- Boundary scoring counted changes in confidence and other auxiliary fields as
  boundaries, even when the scene class stayed the same.
- The human labels and benchmark input came from different encodes of the same
  public video. Their basic metadata agree, but the benchmark did not prove
  frame-for-frame alignment.
- No VLM result was connected to the production exclusion mask. PR 80 remained
  an isolated benchmark.

These issues matter to a reliable benchmark. They do not explain the constant
raw replies as strongly as the task, example, and output design do.

## Short model checks after removing the code task

The bounded follow-up used the same model revisions and PR 80 adapters on the
same two short visual situations. It removed the frame-code list and asked one
plain question per clip.

Qwen returned valid short JSON in both cases:

- It labelled the first clip `active-play` and said the player was serving the
  shuttle repeatedly.
- It described the known cut as a change from a walking player to a seated
  official. It labelled both shots `non-play-close-up`, which matches the
  pilot's activity-based definitions.

After the model was loaded, generation took about 8.0 seconds for the first clip
and 1.6 seconds for the second. The full 50-frame grids were consumed.

This does not establish cleanup accuracy. It does establish that Qwen could read
the same visible content and follow a small output contract. That result makes
the PR 80 request design a much stronger explanation than basic visual failure
for Qwen's all-`other` result.

InternVideo3 also returned complete short JSON. It accurately described the
hard cut from a player walking away to the umpire at her desk. On the warm-up
clip, it called the player stationary and selected `non-play-close-up`, while
Qwen noticed repeated serves and selected `active-play`.

Intern's reply was still varied, grounded in the scene, and correctly formatted.
The frame-code collapse was therefore not inevitable. Its warm-up mistake also
means the prompt was not the only problem: Intern showed a real weakness on this
sampled action. The balanced candidate experiment must measure whether that
weakness makes it unsafe as an automatic verifier.

## Recommendation

Stop prompt work on the present candidate population. The experiments have
answered the immediate question. Better requests remove the catastrophic PR 80
collapse, but neither model can safely turn the current outputs into usable
rallies.

The next experiment should start from the planned binary contact model. Freeze
its output on `sset_01` and `sset_15`, then run this automatic sequence:

1. bypass contacts that the contact model and existing rules both rate highly;
2. send only tracker-risk cases to InternVideo3's marked enlarged request;
3. treat the VLM reply as one input, not as the final decision;
4. apply a simple final rule using contact score, tracker history, masks, scene
   evidence, and the VLM answer;
5. replay every retained contact set through normal attribution, alternation,
   landing, and point logic;
6. freeze the rule before checking `sset_21` and any structural holdout.

The first gate is complete-rally precision. Report exact contact count, every
alternating attribution, point outcome, and contact timing at ±5, ±10, and ±15
base-30 frames. Report rally coverage second. A result that deletes ambiguous
rallies can be useful; a result that keeps wrongly counted rallies cannot.

If the new contact model does not materially raise the oracle ceiling, omit the
VLM. The bottleneck would still be missing candidate evidence rather than
visual cleanup.
