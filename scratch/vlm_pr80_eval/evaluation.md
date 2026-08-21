# Why the PR 80 VLM results failed

## Bottom line

PR 80 does **not** show that InternVideo3 or Qwen3-VL cannot help with annotator cleanup.

It shows that both models failed the benchmark they were given. That benchmark was much harder than the intended cleanup job. It asked the models to build a broadcast timeline from scratch and return a rigid eight-character code for every sampled frame.

The raw replies are the strongest evidence.

InternVideo3 copied the prompt's first example code until it hit the output limit. Qwen changed two characters in the example and copied the other six, even where those copied fields did not fit the visible video.

That makes task and output design a stronger explanation than simple visual failure.

There is one important caveat: PR 80 did not run the same task without the worked example. We therefore cannot prove that the example alone caused the collapse.

The follow-up removed the long code list and asked one small question at a time. Both models then returned short, varied, grounded answers. This shows that the PR 80 collapse was not inevitable.

The later tests also found real model limits. Neither model became reliable enough to make the final automatic cleanup decision. Intern's enlarged, marked shuttle-track check is the strongest narrow result.

## Contents

- [What cleanup was supposed to be](#what-cleanup-was-supposed-to-be)
- [What PR 80 actually did](#what-pr-80-actually-did)
- [The short counterfactual changed the diagnosis](#the-short-counterfactual-changed-the-diagnosis)
- [Why the benchmark was badly matched to the job](#why-the-benchmark-was-badly-matched-to-the-job)
- [What the fairer follow-up found](#what-the-fairer-follow-up-found)
- [The current proposal pool also limits cleanup](#the-current-proposal-pool-also-limits-cleanup)
- [What can be said about each model](#what-can-be-said-about-each-model)
- [Other benchmark limits](#other-benchmark-limits)
- [Recommendation from the completed trials](#recommendation-from-the-completed-trials)

## What cleanup was supposed to be

The annotator already had useful structure:

- proposed contacts;
- a per-frame shuttle track;
- suspected rally spans;
- scene and replay hints;
- masks and camera cuts;
- downstream rally logic.

A useful VLM role was therefore small:

1. take one existing proposal;
2. show the visual evidence that matters for it;
3. ask one question;
4. return evidence that a later rule could use.

Examples:

- Does this claimed shuttle path really follow a shuttle?
- Is this short segment current rally play, replay, or cutaway?
- Is there a real contact inside this small timing window?

PR 80 instead asked the VLM to reconstruct the broadcast timeline itself.

## What PR 80 actually did

### InternVideo3

Intern received one 20-minute clip.

- Sampling: 1 FPS
- Frames: 1,200
- Input tokens: 101,349
- Required output codes: 1,200
- Available output tokens: 9,216

Each output item was an eight-character code.

The prompt's only worked example began with `LBRFRS9B`. This was an example output code, not a hash.

Intern produced 1,316 copies of `LBRFRS9B`. It then stopped partway through another code at the output limit.

The parser took the first 1,200 complete codes, so every sampled frame was scored as `live`.

The parser did not create the repeated answer. The repetition is in the raw model reply. The parser only made a truncated reply look complete by accepting its first 1,200 codes.

### Qwen3-VL

Qwen received one 10-second clip containing 50 frames at 5 FPS.

It returned 50 copies of `OBRFRS9G`, so every frame was scored as `other`.

The clip contained one human-labelled transition from `live-non-standard` to `cutaway`.

The Intern and Qwen runs were not comparable. They used different clip lengths and covered very different amounts of labelled material. The Qwen run was one difficult boundary example, not a general five-class benchmark.

## The short counterfactual changed the diagnosis

The first follow-up used the same model adapters and the same two short visual situations. It removed the long frame-code output.

Qwen returned valid short JSON in both cases.

- On the warm-up clip, it chose `active-play` and described repeated serving action.
- On the known-cut clip, it described the change from a walking player to a seated official. It labelled both sides `non-play-close-up`, which matched the activity-based definitions used in that test.

Intern also returned valid short JSON.

- It described the hard cut correctly.
- It missed the warm-up action and called the player stationary.

So the clean conclusion is:

- the catastrophic repetition was mainly a benchmark-design failure;
- Intern also had a real weakness on the sampled warm-up action;
- valid short output does not mean either model is accurate enough for automatic cleanup.

The useful contrast is **focused task versus overloaded task**, not simply short video versus long video. The later tests changed duration, context, prompt structure, and output requirements together, so this investigation does not isolate temporal horizon as the cause.

## Why the benchmark was badly matched to the job

### 1. It asked for the wrong product

The useful product was a check on an existing annotator claim.

PR 80 asked for a fresh broadcast transcription.

That threw away the pipeline's main advantage: it had already narrowed the search space.

### 2. The only worked example strongly anchored the replies

The prompt's only example contained two `live` codes. The first was `LBRFRS9B`.

Intern copied it exactly for the whole usable reply.

Qwen returned `OBRFRS9G`. Six of its eight fields still matched the example. Those copied fields included `full_court`, `usable_standard`, and confidence `0.9`, even though the clip showed close views and Qwen's own scene field said `other`.

This is direct evidence that the output template influenced both replies.

It is not proof that the example was the only cause. The benchmark had several other problems too.

### 3. The output was too long and fragile

Intern had to keep 1,200 nearly identical strings aligned with 1,200 input frames.

That is partly a visual task. It is also a long bookkeeping task.

The reply hit the output limit, and repetition was an easy failure mode because every answer had the same shape.

### 4. Each code bundled eight decisions

Each frame code combined:

- scene;
- phase;
- playback speed;
- camera view;
- continuity;
- data use;
- confidence;
- visible reason.

Several fields overlapped. Some combinations could contradict each other. Only the scene field mattered to the headline accuracy score.

A cleanup check did not need most of these fields.

### 5. The scene rules could conflict

An earlier short Intern reply shows this clearly.

The clip showed a player warming up from a close camera angle with profile graphics. Intern described those visible details correctly, but labelled the whole clip `cutaway`.

The prompt said a player close-up could be cutaway. It also said real warm-up from an unusual view could be `live-non-standard`.

It did not say which rule should win when both were true.

That looks like a grounded visual description followed by a poor rule choice. It is different from failing to see the clip.

Qwen's PR 80 clip had a similar ambiguity: people and large broadcast overlays were both visible, while the prompt separately associated close-ups with cutaway and graphics with `other`.

### 6. The request omitted useful proposal-level evidence

Issue 38 asked how existing pipeline evidence could make the VLM more precise and efficient.

PR 80 included the video, metadata, sampled frame numbers, and detected hard cuts.

It did not give the VLM a specific annotator proposal to check. It also omitted many proposal-level signals that could have made the question smaller, such as:

- proposed contact time;
- source detector;
- proposed shuttle path;
- court evidence;
- Inpaint evidence;
- nearby replay and scene signals.

The VLM had to discover the question before answering it.

### 7. One sampling plan was being used for different visual jobs

Scene checking and shuttle checking need different evidence.

A scene decision needs enough time around camera changes to understand broadcast order.

A shuttle decision needs dense frames around a small event so the motion is visible.

Intern's 1 FPS input was too sparse for shuttle events and could move short scene changes by about a second.

Qwen used 5 FPS, but only on one boundary example.

The follow-up tests therefore split the jobs and designed separate inputs for each.

### 8. The headline metric did not measure cleanup value

PR 80 reported frame-level macro F1 over five scene classes.

Qwen's clip contained only two of those classes. Under that calculation, even a perfect answer on the clip could score at most 0.4 macro F1.

The important metric is what happens to the annotator after the VLM decision.

For contact cleanup, that means asking whether complete rallies become correct.

For scene cleanup, that means asking how much non-live footage is removed without throwing away real current-rally footage.

## What the fairer follow-up found

| Job | Model coverage | Best useful result | What it means |
|---|---|---|---|
| Contact-window validation | Both | Qwen: 63.6% precision and 96.6% recall at the prompt-aligned ±10 score; 88.6% and 97.5% at the looser ±15 score | Strong candidate-window recall on an artificial 60-case mix, but not direct timing localisation or a demonstrated pipeline gain |
| Complete-rally contact test | Qwen only | Qwen removed 14 of 84 proposals; exact-count rallies stayed 4/12 | Reject the Qwen keep/reject filter as tested |
| Contact actor | Both | Both performed poorly | Keep the existing alternating-player rule |
| Shuttle-track development | Both | Intern was much stronger on the matched development cases | Intern was the model worth carrying into tracker follow-ups |
| Held-out tracker and marker-bias checks | Intern only | 16/18 known hallucinations rejected across development + held-out; clean-first variant became over-conservative | Promising advisory evidence, not a final rule |
| 19-segment local scene pilot | Both | Intern flagged 8/11 targets containing non-live footage for checking and kept 7/8 live; Qwen flagged 6/11 and kept 8/8 | Small pilot favoured Intern for a precision-first cleanup role, but neither was adequate |
| Full-fixture scene check | Intern only | 270/290 standard-view live kept, 6/10 unusual-view live kept, 21/47 targets containing non-live footage flagged for checking | Intern was weak on unusual live views and replay; Qwen remains unmeasured on the full set |
| 90/120-second context | Both | No safe shortcut | Long context did not help in this form |
| Joined context and replay refinements | Intern only | None beat Intern's local 120-frame view | These later negative results apply to Intern, not both models |

`Base-30` means frame distances after normalising source timing to 30 FPS.

## The current proposal pool also limits cleanup

At ±10 base-30 frames, an oracle choosing perfectly from every existing raw contact candidate can make only **56 of 292 ground-truth rallies** complete across the three labelled fixtures.

That matters because the contact experiments in this investigation were mostly **proposal checks**. A model that only keeps or rejects existing proposals cannot recover a contact that is absent from the candidate set.

Do not over-extend that conclusion. These experiments did not properly test a different task where a VLM searches the video for a missing contact, returns its time, and creates a new proposal. The 56-rally ceiling therefore describes the tested proposal pool, not every possible VLM design.

## What can be said about each model

### InternVideo3

The 20-minute PR 80 run is not useful evidence about scene accuracy because generation collapsed.

The bounded follow-up shows that Intern can answer a narrow shuttle-track identity question when the relevant pixels are large enough to inspect. The marked, enlarged tracker test is the strongest result from this investigation.

It is still not safe as the final tracker judge: two of 18 known hallucinations passed, and the comparison paths were not confirmed real tracks.

Intern was also unreliable on the full-fixture scene check. It kept 270/290 standard-view live segments, but only 6/10 unusual-view live segments. It also missed more than half of the meaningful non-live segments. Replay was the main non-live failure. These full-fixture numbers are Intern-only.

### Qwen3-VL

The short PR 80 run was a genuine failure on one difficult example, but it was too small and ambiguous for a general verdict.

The bounded follow-up showed that Qwen could read the same visible content and follow a small output contract.

The larger contact trial produced a strong candidate-window score, especially at the looser ±15 margin, but that did not improve the 12 selected full-rally evaluations. On the matched tracker development set, Qwen was weaker than Intern. On the 19-segment local scene pilot, Qwen preserved all 8 live controls but flagged fewer targets containing non-live footage for further checking than Intern.

Qwen was **not** run on the 463/347 full-fixture scene set, and the matched 19-segment pilot contained no unusual-view live cases. The completed evidence therefore does not justify choosing Qwen for the cleanup role, but it also does not provide a full-fixture head-to-head comparison.

## Other benchmark limits

- Hard cuts were included in PR 80 but were not scored as evidence in their own right.
- PR 80 boundary scoring counted changes in auxiliary fields as boundaries even when the scene class did not change.
- The human labels and benchmark input came from different encodes of the same public video. Their basic metadata agreed, but PR 80 did not prove frame-for-frame alignment.
- No PR 80 VLM result was connected to the production exclusion mask. It remained an isolated benchmark.

These issues matter to benchmark quality. They do not explain the repeated raw replies as directly as the task, example, and output design do.

## Recommendation from the completed trials

Do not repeat small variations of the tested 90/120-second context design on either model. The later combined-view, stronger-replay-wording, and direct replay-pair failures were Intern-only; do not generalise those specific negative results to Qwen without testing it.

Keep two possible VLM signals for later evaluation:

- Intern's marked, enlarged shuttle-track check;
- at most, Intern's plain local scene label as one input among others.

Neither should decide what survives by itself.

At the time this investigation closed, its next planned step was a new contact detector. That recommendation is now **superseded historical context**. The current ordered plan is in [`FOLLOWUPS.md`](FOLLOWUPS.md).

The evaluation principle still stands: first require correctness inside retained rallies — exact contact count, player order, server, point outcome, and contact timing at ±5, ±10, and ±15 base-30 frames. Rally coverage comes second.
