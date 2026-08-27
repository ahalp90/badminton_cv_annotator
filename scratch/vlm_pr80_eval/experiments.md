# Experiments run

## What this file is for

This is the audit trail for the completed VLM investigation.

It answers: **what did we actually try, what did the model see, what was it asked, and what did the result change?**

Retries caused only by an invalid reply are grouped with the experiment they belong to. Prepared directories with no model attempt are not counted as experiments.

The model revisions were fixed throughout:

- InternVideo3: `yanziang/InternVideo3-8B-Instruct`
- Qwen3-VL: `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8`

Exact revisions and runtime versions are in [`experiments/results/summary.json`](experiments/results/summary.json).

## Contents

- [Experiment map](#experiment-map)
- [1. PR 80: the wrong task](#1-pr-80-the-wrong-task)
- [2. Short questions: could the models see and answer?](#2-short-questions-could-the-models-see-and-answer)
- [3–5. Contact-window validation looked promising, then failed the rally test](#35-contact-window-validation-looked-promising-then-failed-the-rally-test)
- [6–7. Tracker development used both models; the final checks were Intern-only](#67-tracker-development-used-both-models-the-final-checks-were-intern-only)
- [8. Early broadcast sequencing was unreliable](#8-early-broadcast-sequencing-was-unreliable)
- [9. Existing scene signals helped with routing, not final decisions](#9-existing-scene-signals-helped-with-routing-not-final-decisions)
- [10. Long cut-aware context did not produce a useful shortcut](#10-long-cut-aware-context-did-not-produce-a-useful-shortcut)
- [11. The simple local view was best](#11-the-simple-local-view-was-best)
- [12–13. Showing long and local views together made Intern worse](#1213-showing-long-and-local-views-together-made-intern-worse)
- [14. Intern-only full-fixture scene test exposed two weaknesses](#14-intern-only-full-fixture-scene-test-exposed-two-weaknesses)
- [15. Stronger replay wording made the same clips worse](#15-stronger-replay-wording-made-the-same-clips-worse)
- [16. Showing the earlier action directly did not help](#16-showing-the-earlier-action-directly-did-not-help)
- [Exploratory runs and retries](#exploratory-runs-and-retries)
- [Evidence](#evidence)

## Experiment map

| Trial | Models | What was tested | Main result |
|---|---|---|---|
| 1. PR 80 whole timeline | Both, on different inputs | Build a scene timeline from long or boundary video and return one eight-field code per sampled frame. | Both replies collapsed into repeated versions of the worked example. The benchmark was a poor match for cleanup. |
| 2. Short scene checks | Both | Ask one plain activity question on two 10-second clips. | Both returned grounded short JSON. Qwen saw the warm-up action; Intern missed it. |
| 3. Contact-window validation and actor | Both | Check 60 proposed contacts in two-second clips; also name the top/bottom actor. | Qwen matched reference timing well. Both were poor at actor attribution. |
| 4. Require model agreement | Both replies, combined | Keep a contact only if both models said yes. | Precision sometimes rose, but recall fell sharply. |
| 5. Full-rally contact test | Qwen only | Apply Qwen's keep/reject answers to 84 proposals from 12 selected rallies across two fixture videos. | Four exact-count rallies before, four after. No end-to-end gain. |
| 6. Shuttle-track check | Both in development; Intern held-out | Ask whether a cyan-marked tracker path follows a real shuttle. | Intern was much stronger. Its held-out check became the strongest narrow result. |
| 7. Marker-bias check | Intern only | Show the enlarged tracker interval first without the marker, then with it. | Intern rejected all known hallucinations but also rejected many comparison paths. |
| 8. Early broadcast sequence | Both | Judge a dense four-second target inside a 20-second window, then retry with explicit replay wording. | Direct wording missed replay; stronger wording sent too many live clips for further checking. |
| 9. Existing scene signals | No VLM | Test whether existing masks and tracker visibility can route human-labelled scene windows. | Helpful signal, not strong enough as the final filter. |
| 10. 90/120-second scene context | Both | Use 96 sparse frames to decide which targets need a local check. | No safe shortcut emerged. |
| 11. Local 120-frame scene check | Both | Classify 19 short segments from 120 consecutive frames, with and without extra facts. | Intern had the better cleanup tradeoff on this small pilot; neither was adequate. |
| 12. Long + local frames | Intern only | Add 80 sparse context frames before the same 120 local frames. | Targets containing non-live footage flagged for checking fell from 8/11 to 2/11. |
| 13. Two-model scene rules | Both replies, combined | Test agreement and union rules from trial 11. | None beat Intern alone on that pilot. |
| 14. Full-fixture scene test | Intern only | Run Intern's 120-frame local scene check on all 463 eligible segments. | On 347 meaningful segments: 270/290 standard-view live kept, 6/10 unusual-view live kept, 21/47 targets containing non-live footage flagged for checking. |
| 15. Stronger replay wording | Intern only | Change only the replay wording on the fixed local clips. | It flagged fewer targets containing non-live footage for checking. The larger run was cancelled. |
| 16. Direct replay comparison | Intern only | Compare an earlier 120-frame action with the 120-frame target. | Intern returned `different_action` on all 46 pairs. |

The exact prompts are catalogued in [`prompts.md`](prompts.md).

## 1. PR 80: the wrong task

PR 80 asked the models to build a broadcast scene timeline from scratch.

Intern saw one 20-minute clip sampled at 1 FPS. It had to return 1,200 eight-character frame codes. It returned 1,316 copies of `LBRFRS9B` and hit the output limit. The parser took the first 1,200 complete codes, so every sampled frame was scored as `live`.

`LBRFRS9B` was the first example output code in the prompt, not a hash.

Qwen saw one 10-second boundary clip with 50 frames. It returned 50 copies of `OBRFRS9G`, so every frame was scored as `other`.

The two runs were not comparable. More importantly, neither matched the intended cleanup job. The annotator had already proposed useful events, but the prompt did not ask the VLM to inspect those proposals.

## 2. Short questions: could the models see and answer?

The next check reused two short visual situations:

- visible warm-up action;
- a known hard camera cut.

The long frame-code output was removed.

Qwen described both clips sensibly. Intern described the cut correctly but called the warm-up clip non-play. Both returned valid short JSON.

This separated two problems:

- PR 80's catastrophic repetition came mainly from the task and output design;
- Intern still had a real visual weakness on the sampled warm-up action.

## 3–5. Contact-window validation looked promising, then failed the rally test

The trial did **not** ask either model to locate a contact time. It started from 60 existing proposed contacts and asked whether each marked window contained a real contact according to ShuttleSet timing.

The 60 cases were deliberately stratified rather than sampled from the natural pipeline population:

- 20 proposals were within ±5 normalised frames of ShuttleSet timing;
- 20 were between ±5 and ±15 frames away;
- 20 were more than ±15 frames away.

This artificial mix is useful for a controlled comparison. Its precision is not a population precision estimate.

Each input contained 50 frames over two seconds. A gold border marked the accepted timing window. A cyan ring showed the tracker claim. The prompt said both could be wrong.

The prompt's gold window was about ±10 base-30 frames. At that prompt-aligned threshold, Qwen reached **63.6% precision and 96.6% recall**. Requiring both models to agree raised precision to 72.7% but reduced recall to 55.2%.

At the looser ±15 threshold:

- Qwen: 88.6% precision, 97.5% recall;
- Intern: 83.3% precision, 50.0% recall;
- requiring both to agree: 86.4% precision, 47.5% recall.

These are candidate-window validation scores against ShuttleSet timing, not direct contact-time localisation. Some serve labels are inferred across a camera cut, so the score also does not prove that the physical impact was visible.

The contact prompt also asked which player made the contact: the top or bottom player. Both models were poor at this answer. The retained evidence does not establish why.

The decisive contact test used Qwen's keep/reject decisions on 84 existing proposals from 12 selected rallies across two fixture videos.

It removed 14 contacts.

The result did not improve:

- exact contact counts: 4/12 before, 4/12 after;
- rallies passing all experiment checks at ±10 frames: 1/12 before, 1/12 after.

Passing all checks required the correct contact count, every contact matched in time, correct player attribution, correct server, and correct point outcome.

Four replies were invalid.

So the good local timing score did not become better rally data.

## 6–7. Tracker development used both models; the final checks were Intern-only

The development tracker audit contained 12 known hallucinations and 12 comparison clips near ShuttleSet contacts. Both models were tested on the development set.

The comparison clips are not confirmed correct tracker paths. They are mainly a check against a prompt that rejects everything.

Intern improved as the claimed path became easier to inspect:

| View | Known hallucinations rejected | Comparison paths accepted |
|---|---:|---:|
| Plain marked target | 9/12 | 9/12 |
| Slow marked target | 9/12 | 12/12 |
| Slow, enlarged marked target | 11/12 | 12/12 |
| Enlarged marked target on held-out video | 5/6 | 6/6 |

Intern's held-out follow-up added six hallucinations and six comparison paths. Across Intern's development and held-out cases, the best marked view rejected 16 of 18 known hallucinations and accepted all 18 comparison paths.

Qwen rejected only 5 of 12 development hallucinations with the enlarged view, so that branch was stopped.

The marker-bias check was also Intern-only. It showed the enlarged target first without the cyan ring and then with it. Intern rejected all 18 known hallucinations, but accepted only 7 of 18 comparison paths.

That test made the model more conservative. It did not establish a safer final rule.

## 8. Early broadcast sequencing was unreliable

The first scene-sequence test used 12 controls. Each input had a four-second dense target inside a 20-second source window.

The direct prompt asked whether the target was a coherent live rally.

Qwen called 11 of 12 targets live. Intern produced three invalid replies and called the other nine live. Most replay and cutaway content passed.

A second prompt explicitly described a common broadcast pattern: standard live view, closer replay, then return to the standard view.

Qwen was essentially unchanged.

Intern found more replay, but it kept only 2 of 6 live controls. The wording replaced replay blindness with a broad tendency to send live footage for further checking.

## 9. Existing scene signals helped with routing, not final decisions

A non-VLM trial used 197 human-labelled scene windows from `sset_01` and `sset_15`.

The best simple rule asked whether the existing shuttle tracker claimed the shuttle was visible on at least 80% of the automatic rally span. The calculation was the mean of the tracker's per-frame visibility flag. It measured how persistent the track was, not whether the tracked object was really a shuttle.

At the 80% threshold, it kept 163 of 176 live windows and flagged 9 of 21 replay-or-cutaway windows for further checking. This threshold was later reused unchanged in the three-fixture Intern scene run.

That made it a useful cheap prior, but not a safe final decision rule. A persistent hallucinated shuttle track can pass it.

## 10. Long cut-aware context did not produce a useful shortcut

The later multiscale trial started from 12 suspected rally spans across `sset_01`, `sset_15`, and `sset_21`.

Each span had two long storyboards:

- 90 seconds;
- 120 seconds.

Each storyboard contained 96 selected frames. Frames near stored camera cuts and the target span were always kept.

Ninety seconds tied or beat 120 seconds, so later tests used 90 seconds.

On the 12-span routing pilot:

- Qwen correctly sent 5 of 9 non-live targets for a local check and left 2 of 3 live targets alone;
- Intern sent all 9 non-live targets for a local check, but also sent all 3 live targets.

Intern found no safe shortcut. It simply sent everything onward.

The long-view reading was then reduced to text facts and given to the local model. Those facts made the result worse or left it unchanged.

## 11. The simple local view was best

The 12 suspected spans were split at stored camera cuts. This produced 19 short segments.

Each input contained 120 consecutive source frames, usually about four seconds.

Three arms used the same pixels:

1. video plus the scene definitions;
2. the same video plus automatic cut, span, contact, court, mask, and tracker facts;
3. the same video plus those facts and the parsed long-view VLM reading.

Intern's video-only arm was best:

- 8 of 11 targets containing non-live footage flagged for further checking;
- 7 of 8 live segments kept.

Qwen flagged 6 of 11 targets containing non-live footage for further checking and kept all 8 live segments.

Adding facts reduced non-live detection. Agreement or union rules using both models also failed to beat Intern alone.

The prompt lesson is simple: in this test, the relevant motion plus one plain visual question worked better than adding more prior text.

## 12–13. Showing long and local views together made Intern worse

The joined input placed 80 sparse frames from the 90-second context before the same 120-frame local view.

Intern used all 200 frames and returned a valid reply for all 19 segments.

It flagged only 2 of 11 targets containing non-live footage for further checking, although it kept all 8 live segments.

Five targets that the local view had flagged for further checking changed back to live.

No two-model rule recovered the loss.

This closed three tested ways of adding long context:

- a separate long VLM pass;
- text facts from that pass;
- long and local frames shown together.

## 14. Intern-only full-fixture scene test exposed two weaknesses

Intern's 120-frame local design was then run on all 463 eligible segments from the three scene-labelled fixtures. Qwen was not run on this full set.

Every call used all 120 frames and returned a usable reply.

The first all-463 score said the model flagged 84.0% of targets containing non-live footage for further checking. That number was misleading because many were tiny boundary disagreements. The material evaluation excluded **116 short targets** in total: 115 targets containing non-live footage were shorter than the accepted ±10-frame margin, including 113 one-frame cases, and one additional one-frame unusual-view live target was also excluded.

That leaves 347 meaningful targets:

- 290 standard-view live;
- 10 unusual-view live (`live-non-standard` in the human truth);
- 47 targets containing at least some replay, cutaway, or other footage.

Intern:

- kept 270 of 290 standard-view live segments — 93.1%;
- kept 6 of 10 unusual-view live segments — 60.0%;
- flagged 21 of 47 targets containing non-live footage for further checking — 44.7%;
- accepted 26 such targets as live.

Twenty of those 26 misses were pure replay. The original score file merged both live classes under `live`; the separate unusual-view diagnostic promised in the plan was not produced. The recovered subgroup counts are now retained in `experiments/results/scene_live_view_split.json`. Qwen was not run on this full-fixture set, and its matched 19-target local pilot contained no unusual-view live targets. Its PR 80 boundary clip did contain `live-non-standard` frames, but that collapsed run is not useful matched evidence.

Combining Intern with that tracker-persistence rule changed the result:

- live segments kept: 276/300 → 270/300;
- targets containing non-live footage flagged for further checking: 21/47 → 27/47;
- precision among kept segments: 91.4% → 93.1%.

In other words, it flagged 6 extra targets containing non-live footage and 6 extra live targets for further checking. It still accepted 20 of 47 targets containing non-live footage as live.

## 15. Stronger replay wording made the same clips worse

The next trial changed only the wording on the fixed local-view pilot.

Intern was told to treat a smooth closer view after standard play as replay unless the pixels clearly showed a new rally.

On the 16 cases large enough to matter:

- default wording kept 7/8 live and flagged 5/8 targets containing non-live footage for checking;
- stronger replay wording kept 8/8 live but flagged only 3/8 targets containing non-live footage for checking.

The precision of the model's live calls fell from 70.0% to 61.5%.

The planned larger run was cancelled.

## 16. Showing the earlier action directly did not help

The final bounded scene trial targeted the main failure: missed replay.

For each target, automatic span order chose the nearest eligible earlier span within 90 seconds. Human truth was not used to choose the pair.

The model saw:

- 120 frames from the earlier span as `REFERENCE`;
- 120 frames from the target as `TARGET`.

The requested sample contained the 26 meaningful targets containing non-live footage that the full-fixture parent run had wrongly called live, plus 22 nearby true-live controls.

Two such targets had no eligible earlier reference, leaving 46 pairs: 24 targets containing non-live footage and 22 live controls.

Intern returned valid output for every pair. It called all 46 `different_action`.

That changed none of the 24 available parent mistakes.

Qwen was not run on a shortened pair. It had already missed all three pure replays in the local-view pilot and was weaker on the long-context task. The full 240-frame Intern pair used 18,545 input tokens, while the public Qwen runner was fixed at 16,384. Shortening the pair would have changed the visual evidence rather than reproducing the same test.

This tested replay-comparison branch is closed. Reopen it only with a genuinely different visual representation and a small balanced gate.

## Exploratory runs and retries

Earlier pilots changed clip layout, serve rules, actor wording, JSON fields, and tracker presentation while sample sizes were still small.

They informed the final experiments above but do not carry separate conclusions.

A retry caused only by an invalid reply is not counted as a new experiment.

## Evidence

- [`README.md`](README.md) — short overall explanation
- [`RESULTS_FIRST_EXPERIMENTS.md`](RESULTS_FIRST_EXPERIMENTS.md) — retained measurements and limits
- [`experiments/results/summary.json`](experiments/results/summary.json) — machine-readable counts
- [`evaluation.md`](evaluation.md) — detailed diagnosis of PR 80
- [`prompts.md`](prompts.md) — prompt history and exact builders
- [`sources.md`](sources.md) — GitHub record and input provenance
