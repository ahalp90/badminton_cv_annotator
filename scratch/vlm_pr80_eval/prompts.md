# Prompt catalogue

This document makes the prompt history readable without requiring a search
through attempt JSON. It shows the meaningful prompt changes. Retries that used
the same words are omitted.

The reusable prompt builder is
[`experiments/prompts.py`](experiments/prompts.py). It reproduces the final
contact, tracker, and broadcast-sequence prompts. Older prompts are recorded
here because they no longer have runnable public builders.

## Prompt index

| Prompt | Used for | Public source |
|---|---|---|
| PR 80 whole timeline | Long Intern run and short Qwen boundary run | [Exact immutable code](https://github.com/ahalp90/badminton_cv_annotator/blob/96e0e289a951d63fbaaa62f26c399a4beb61ae79/src/annotator/vlm_scene_benchmark/prompts.py#L21-L70) |
| Short scene checks | Warm-up and known camera cut | Historical full text below |
| Contact timing and actor | Balanced 60-case trial and rally replay | [`_EVENT_PROMPT`](experiments/prompts.py) |
| Pipeline-prior contact | Same clips with annotator observations appended | [`_format_event_priors()`](experiments/prompts.py) |
| Tracker validity | Plain, slow, enlarged, and clean-then-marked views | [`_TRACK_PROMPT`](experiments/prompts.py) |
| Direct broadcast | First 12-case scene trial | Historical full text below |
| Sequence broadcast | Second 12-case scene trial | [`_BROADCAST_PROMPT`](experiments/prompts.py) |

## 1. PR 80 whole timeline

The prompt began:

> You are labelling one complete badminton broadcast shard for later human
> review.

It supplied video metadata, every sampled source-frame number, and detected hard
cuts. It defined five scene labels: `live`, `live-non-standard`, `replay`,
`cutaway`, and `other`.

For every sampled frame, the model had to return one eight-character code. The
characters represented scene, phase, playback speed, camera view, continuity,
data use, confidence, and visible reason. The only worked example was:

```json
{"frames":["LBRFRS9B","LLRFRS9R"]}
```

The prompt then required exactly one code per frame, in order, with no grouping,
omission, repetition, frame numbers, or explanation. The exact template is in
the immutable source linked above.

This example is central to the failure. Intern repeated `LBRFRS9B`; Qwen
repeated `OBRFRS9G`, retaining six of the example's eight fields.

## 2. Short scene checks

These checks removed the frame-code list and asked one plain question per
10-second clip. The blocks below are copied verbatim from the pre-cleanup pilot
script, whose exact version remains in the local archive.

<details>
<summary>Warm-up prompt</summary>

```text
You are checking one ten-second badminton broadcast clip.

Choose the single description that best fits the visible activity:
- active-play: a player is hitting, serving, or warming up with normal badminton actions;
- non-play-close-up: a player or official is shown, but no badminton action is taking place;
- replay: earlier play is being shown again or in slow motion;
- graphic-or-transition: the broadcast graphic or transition is the main content.

Large name, result, or score overlays do not turn visible player activity into a graphic scene. When camera framing and activity disagree, base the answer on what the person is doing.

Return a bare JSON object with exactly three keys: "label", "visible_evidence", and "uncertainty". Use one short sentence for visible_evidence and uncertainty. Do not use a Markdown fence.
```

</details>

<details>
<summary>Known-cut prompt</summary>

```text
You are checking one ten-second badminton broadcast clip. There is a hard camera cut exactly five seconds after the start.

Classify the shot before the cut and the shot after the cut independently. Use exactly one label for each:
- active-play: a player is hitting, serving, or warming up with normal badminton actions;
- non-play-close-up: a player or official is shown, but no badminton action is taking place;
- replay: earlier play is being shown again or in slow motion;
- graphic-or-transition: the broadcast graphic or transition is the main content.

Large name, result, or score overlays do not turn visible people or activity into a graphic scene. When camera framing and activity disagree, base the answer on what the person is doing.

Return a bare JSON object with exactly three keys: "before_cut", "after_cut", and "visible_change". The two labels must be strings. Describe the visible change in one short sentence. Do not use a Markdown fence.
```

</details>

## 3. Contact timing and actor

The final contact prompt used a two-second clip. A gold border marked the timing
window and a cyan ring marked the tracker claim. It asked whether contact was in
the window, whether it was visible or inferred, who acted, and whether another
contact appeared elsewhere in the clip.

<details>
<summary>Final contact prompt</summary>

```text
You are checking a two-second badminton video around one automatically proposed contact.

The gold border marks the accepted timing window: eight native video frames before through eight frames after the proposed instant. This is about ±10 frames after normalising to 30 FPS. The cyan ring shows where the shuttle tracker says the shuttle is. Both markers can be wrong. Judge the underlying pixels and motion, not the marker or the proposal. A contact outside the gold-bordered window does not validate this proposal.

The labels define TOP as the far player and BOTTOM as the near player.

Answer these questions:
- Is there a real racket-shuttle contact anywhere in the gold-bordered window? Count an inferred off-screen serve only when serve preparation is visible before a cut and active rally play begins immediately after the cut. Players merely standing in standard court view are not enough.
- Is the support a visible contact, a logically inferred off-screen contact, no contact, or unclear?
- If yes, is the player on the top or bottom half of the court the actor?
- Is another clear racket-shuttle contact visible elsewhere in this short clip but outside the gold-bordered window?

Use "unclear" when the broadcast does not show enough visual or sequence evidence. Return a bare JSON object with exactly these keys: "contact_at_marker", "evidence_kind", "actor", "nearby_unmarked_contact", "visible_evidence", "uncertainty". contact_at_marker and nearby_unmarked_contact must be "yes", "no", or "unclear". evidence_kind must be "visible-contact", "inferred-contact", "no-contact", or "unclear". actor must be "top", "bottom", "no-contact", or "unclear". Give one short sentence for visible_evidence and uncertainty. Do not use a Markdown fence.
```

</details>

The `pipeline-priors` arm appended fallible observations: court presence,
tracker visibility, wrist and player proximity, suppression, exclusion masks,
and the time to neighbouring candidates. It ended with:

> Resolve any conflict in favour of the visible video.

The same observations were useful as future routing inputs. Pasting them into
the prompt did not give a clear overall improvement.

Early development prompts used a narrower actor question and, later, a
full-court panel beside a magnified panel. Those versions helped settle the
TOP/BOTTOM wording. They did not produce a useful actor result and were replaced
by the final prompt above.

## 4. Tracker validity

All tracker prompts asked one question: does the cyan ring follow a visible real
shuttle consistently? The valid object labels were `real-shuttle`,
`text-or-logo`, `player-or-racket`, `empty-or-unrelated`, and `unclear`.

The block below is a normalised reconstruction of the later slow and enlarged
prompts. `{view instruction}` marks the one paragraph that changed. The current
builder is authoritative for the enlarged marked and clean-then-marked forms.
The earliest plain trial used the same decision and output rules, but described
one normal-size gold-bordered interval instead of a slow repeat.

<details>
<summary>Normalised tracker template</summary>

```text
You are checking whether a badminton shuttle tracker follows the real shuttle in a two-second video.

The cyan ring is the tracker's claim. The clip first gives unbordered full-view context, then repeats the short target interval slowly with a gold border.{view instruction} Judge that repeated gold-bordered interval. The tracker can lock onto court text, a logo, a racket, a player, or empty background. Judge the underlying pixels and motion, not the ring. This is not a contact question.

Answer yes only when the cyan ring follows a visible real shuttle consistently through the gold-bordered interval. Answer no when it follows a different object, empty space, or a guessed path with no visible shuttle. Use unclear when compression, occlusion, or size makes the object genuinely impossible to identify.

Return a bare JSON object with exactly these keys: "tracked_object", "visible_evidence", "uncertainty". tracked_object names what the cyan ring follows and must be "real-shuttle", "text-or-logo", "player-or-racket", "empty-or-unrelated", or "unclear". Give one short sentence for visible_evidence and uncertainty. Do not use a Markdown fence.
```

</details>

The experiment changed the view instruction, not the decision rule.

### Plain marked target

The first version showed the marked interval at normal speed and size. The next
version added full-view context and repeated the target slowly.

### Enlarged marked target

The best version inserted:

> The gold-bordered replay is a fixed enlarged view around the claimed track;
> it does not change the underlying frames.

### Clean then marked

The counterfactual inserted:

> The gold-bordered target appears twice in a fixed enlarged view: first
> without the cyan marker, then with it. Use the clean replay to identify the
> actual pixels and the marked replay only to locate the tracker claim. Do not
> infer a shuttle from the marker.

The archived attempt records identify the four historical prompt strings:

| Variant | Campaign record | Prompt SHA-256 |
|---|---|---|
| Plain marked | `cycle_c_v16` | `79d32793243a3ca6c0d37b66750715703146914d47920b2aa1ec714533184aae` |
| Slow marked | `cycle_c_v17` | `1bd86d5b5ee31ea8a0a175784cb105dd4c5454e8aad058b85f5cdc7527ab399e` |
| Slow enlarged marked | `cycle_c_v18` | `6d52a0994010dd3c98523772d8cf7c43dab0e0502bd17c236289b520515616b7` |
| Clean then marked | `cycle_c_v18d2` | `26c97af4c0aa3495d6e77dc783ad6e933414c795b3ad8aa44dd11b390f913249` |

## 5. Broadcast sequence

The direct and sequence prompts used the same clips. Each clip represented a
20-second source window. Thirty consecutive gold-bordered frames densely
sampled the four-second target. The other frames sparsely showed what came
before and after.

The direct block below is historical full text copied from `cycle_d_v19c`. Its
prompt SHA-256 is
`89cbb10d72acd5cee0fcb4dd7ebdf5c2edf1bfad0c829706dc7c2adece111600`.
The current public builder contains the later sequence form.

<details>
<summary>Direct broadcast prompt</summary>

```text
You are checking a badminton sequence around an automatically proposed rally span.

The clip contains 50 ordered frames from a twenty-second source window. The 30 consecutive gold-bordered frames densely sample the four-second target. The unmarked frames sparsely show the surrounding broadcast order. Judge the gold-bordered target; use the unmarked frames only as context.

Decide whether most of the marked target is a coherent live rally sequence. A valid target can include serve preparation, an inferred off-screen serve, a camera cut, or a late final shuttle flight. A replay is not live play even when it shows real badminton action. Replays often repeat an earlier live action from a closer angle after the standard live view. Use the surrounding context to spot that sequence. Crowd or player cutaways, graphics, warm-up, and unrelated motion are also not live-rally evidence. Answer "no" when replay or cutaway footage occupies most of the marked target.

Return a bare JSON object with exactly these keys: "valid_rally_evidence", "broadcast_content", "contains_camera_cut", "visible_evidence", "uncertainty". valid_rally_evidence and contains_camera_cut must be "yes", "no", or "unclear". broadcast_content must be "live-play", "mixed", "replay", "cutaway", "other", or "unclear". Give one short sentence for visible_evidence and uncertainty. Do not use a Markdown fence.
```

</details>

The sequence version added this paragraph before the output contract:

> Before answering, compare the unmarked BEFORE context with the gold target,
> then compare the gold target with the unmarked AFTER context. Ask whether the
> target repeats or reframes action already shown, and whether the broadcast
> then returns to the standard live view or next point. A standard-view action
> followed by a closer repeat and then a return to standard view is a replay.
> Do not answer live-play merely because the target contains continuous
> badminton action.

That addition helped Intern identify replay content, but it also rejected too
many live controls. Qwen was essentially unchanged.
