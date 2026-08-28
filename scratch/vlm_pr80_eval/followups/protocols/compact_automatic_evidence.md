# Planning note: compact automatic evidence

## Bottom line

Run Follow-up 2 first with video and the selected cut only. Do not add another
experiment before that clean comparison.

The annotator does have useful automatic evidence. Its first use should be
deterministic filtering in Follow-up 3. Follow-up 4 can then test whether a
small set of plain-language observations helps the chosen model reconstruct
serves.

This note records an assessment, not an experiment. It does not change any
completed finding.

## Why the earlier facts do not settle this question

The completed deterministic-facts arm did not give the model a local account
of what changed around the target. It supplied these values for the whole
proposed rally span:

- rally duration
- total raw and filtered contact counts
- fraction of frames with a detected court
- fraction covered by each exclusion mask
- fraction with a visible shuttle track

A short replay or cutaway inside a mostly live rally can therefore inherit
mostly live-looking averages. Contact totals also discard the timing and
strength of individual events. The tested arm does not show whether concise,
target-local observations are useful.

The source of those aggregate facts is
`experiments/build_multiscale_trials.py::_automatic_priors()`. The detail
builder copied them into each inspected segment without changing their scope.

## Evidence already available

The current pipeline records several observations close to a target:

- hard camera-cut boundaries from PySceneDetect
- whether a usable court was detected
- whether the scene contained a usable court and two players
- which component of the exclusion mask fired
- whether the shuttle track was visible
- candidate contact time and motion impulse
- explicit shuttle-to-player or shuttle-to-wrist proximity
- detected serve starts and rally-span membership

These observations have different strengths. They should remain separate
rather than being collapsed into one confidence score.

The replay-mask measurement on `sset_01` illustrates the limit. The combined
mask flagged 91,521 of 142,237 scored frames with 0.981 precision and 0.972
recall. Court absence accounted for every flagged frame. Perspective shift and
shuttle-velocity drop accounted for none. This is strong evidence for finding
obvious off-court material in that video. It is not evidence that the mask can
identify every replay or distinguish replay from another cutaway.

Shuttle-track visibility provides useful corroboration, but the completed
scene controls show that it is not enough on its own. Perspective shift is
also risky here. Qwen called all 10 unusual-view live targets replay in the
completed scene comparison.

The retained `sset_01` evidence also contains 61 short replay intervals with a
human-adjudicated immediately preceding live source. Those relations could
support a later retrieval study. Exact matching source frames are not labelled,
so this is not yet an automatic relation signal.

## Use the evidence in two stages

### Follow-up 3: deterministic filtering

Use the existing measurements directly in the precision-first rule. This is
the clearest way to reduce the model's search space.

For example, court absence can identify obvious material for further checking.
A usable court, two players on court, a shuttle tracked continuously around the
candidate and explicit contact proximity can support retaining a candidate. No
single observation should be treated as proof of live play or replay.

Only explicit positive contact evidence should count. Membership in
`filtered_contacts` is insufficient because an unmeasured `wrist_near` value
can pass the current filter. Prefer facts such as `wrist_near is True` and,
where available, `proximity_ok is True`.

### Follow-up 4: model-facing observations

Test compact automatic facts only after one model has been chosen. Express
each fact as an ordinary observation about the supplied clip. Do not expose an
internal score or mask name and expect the model to infer its meaning.

Generate only the sentences supported for the current clip. For example, a
case where all five observations were measured could be described as:

```text
Automated video analysis produced the observations below. They may be wrong,
so use them only as supporting evidence.

- A hard camera cut occurs at the start of the inspected clip.
- A usable full-court view is detected through most of the clip.
- Both players are detected on court through most of the clip.
- The shuttle track is visible near the proposed contact.
- The shuttle was detected close to a player's wrist 0.08 seconds after the cut.
```

Avoid internal shorthand such as:

```text
composition: 0.96
court absent: 82%
mask cause: court absence
```

Exact scores belong in the evidence and evaluation record. If a measurement is
borderline, the model-facing sentence should preserve that uncertainty rather
than turn it into a categorical statement.

Keep the fact list short and local to the decision. Do not reintroduce a long
storyboard, raw keypoints, coordinate tables, whole-rally averages or event
trajectories.

Follow-up 4's third arm is different from these observations. It adds the
current pipeline's proposed server and contact time, clearly labelled as
fallible conclusions. It does not add an internal live or replay label.

## Evidence references

- [`build_multiscale_trials.py`](../../experiments/build_multiscale_trials.py)
  defines the whole-span automatic priors used in the completed facts arm
- [`build_detail_from_context.py`](../../experiments/build_detail_from_context.py)
  shows how those priors were copied into each inspected segment
- [`signals.md`](../../experiments/signals.md) records the available automatic
  signals, their measured strengths and the `filtered_contacts` limitation
- [`sset_01` replay and serve measurement](../../../../docs/scraper_pipeline/broadcast_nonstandard_camera_id/data/sset_01_replay_and_serve_behaviour_20260805/report.md)
  records the replay-mask component results and the 61 adjudicated source
  relations
- [Follow-up 1 scene comparison](../1_scene_comparison.md) records the
  paired Qwen and Intern scene results

## Follow-up 2 remains the clean gate

Follow-up 2 compares both models on the same rally-start clips with video and
the selected cut only. This separates model choice from evidence-format choice
and gives Follow-up 4 a clean baseline.

That clean gate chose Intern. Follow-up 4 tests the compact evidence on Intern
first. If an enhanced arm passes the material-improvement gate in
[`4_serve_reconstruction.md`](4_serve_reconstruction.md), freeze its format and
run one Qwen confirmation. That new paired result may change the operational
model choice. It does not rewrite the clean Follow-up 2 finding.

## Decision

Do not run a compact-evidence experiment before Follow-up 2.

Reuse existing evidence directly in Follow-up 3. Test its plain-language form
as the planned compact-facts arm in Follow-up 4. Treat the 61 known
replay-to-source relations as a possible later study, outside the current
follow-up sequence.
