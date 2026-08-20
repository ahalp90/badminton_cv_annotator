# Experiments run

This document answers one question: what did we actually try? It groups retries
and small tuning runs with the experiment they informed. A retry caused by an
invalid reply is not treated as a new experiment.

The model revisions were fixed throughout:

- InternVideo3: `yanziang/InternVideo3-8B-Instruct`
- Qwen3-VL: `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8`

The exact revisions and runtime versions are in
[`experiments/results/summary.json`](experiments/results/summary.json).

## Experiment map

| Trial | What the model saw | What we asked | What it showed |
|---|---|---|---|
| 1. PR 80 whole timeline | Intern: one 20-minute clip sampled at 1 FPS. Qwen: one 10-second boundary clip sampled at 5 FPS. | Eight decisions for every frame, packed into one code. [Prompt](prompts.md#1-pr-80-whole-timeline) | Both replies collapsed into a repeated version of the worked example. This was mainly a bad test of the models. |
| 2. Short scene checks | Both models saw two 10-second clips: one visible warm-up and one known camera cut. | One plain activity label, or one label on each side of the cut. [Prompts](prompts.md#2-short-scene-checks) | Both models produced short, grounded JSON. Qwen saw the warm-up action; Intern missed it. The PR 80 collapse was not inevitable. |
| 3. Contact timing and actor | 60 balanced two-second clips around proposed contacts. The gold border marked the accepted timing window. | Is there contact in the window? Is it visible or inferred? Who acted? [Prompt](prompts.md#3-contact-timing-and-actor) | Qwen gave the best structural timing result. Both models were poor at the actor answer. Pasting pipeline observations into the prompt did not give a clear overall gain. |
| 4. Model agreement | The same 60 contact cases and the two model replies. | Keep a contact only when both models said yes. | Agreement sharply reduced recall. It improved precision at ±10 frames, but was worse on both precision and recall at ±15. |
| 5. Complete-rally replay | 84 proposed contacts from 12 rallies. | Use Qwen's contact answer as a keep/reject filter, then rerun normal rally logic. | It removed 14 contacts but did not improve exact contact counts or structurally usable rallies. |
| 6. Tracker-path checks | Two-second clips with the claimed tracker path marked by a cyan ring. The view changed over three iterations: plain, slow repeat, then slow enlarged repeat. | Does the ring consistently follow a real shuttle? [Prompts](prompts.md#4-tracker-validity) | Enlarging the marked path helped Intern. It became the strongest lead, but still accepted two known hallucinations across development and safety. |
| 7. Clean tracker counterfactual | The enlarged target appeared once without the ring, then again with it. | The same tracker question. [Prompt](prompts.md#clean-then-marked) | Intern rejected all 18 hallucinations, but also rejected 11 of 18 comparison clips. Removing marker influence created a broad reject bias. |
| 8. Broadcast cleanup | Twelve targets: 6 live, 4 replay, 2 cutaway. Each had dense target frames and sparse before/after context. | Is the target a coherent live rally? First directly, then with an explicit before-target-after comparison. [Prompts](prompts.md#5-broadcast-sequence) | Qwen kept almost everything. The stronger sequence wording made Intern notice replay but also reject four live controls. Neither version was safe. |
| 9. Existing scene priors | 197 human-labelled scene windows. No VLM call. | Can existing masks and tracker visibility route cases cheaply? | `track_visible_fraction >= 0.8` kept 92.6% of live windows and rejected 9 of 21 non-live windows. Useful for routing, not strong enough as the final filter. |

## 1. PR 80: the wrong task

The Intern run required 1,200 eight-character answers from one long clip. It
returned 1,316 copies of `LBRFRS9B` and hit the output limit. The parser used
the first 1,200 codes, so every sampled frame became `live`.

The Qwen run required 50 codes from one short boundary clip. It returned 50
copies of `OBRFRS9G`, so every frame became `other`.

The runs were not comparable. More importantly, neither matched the planned
cleanup job. The auto-annotator had already proposed events, but the prompt did
not ask the model to inspect any of them.

## 2. Short questions: can the models see and answer?

The next check used the same model adapters and two short clips. The fixed frame
codes were removed.

Qwen described both clips sensibly. Intern described the hard cut correctly,
but called the warm-up clip non-play. Both returned the requested short JSON.
This isolated two facts:

- the PR 80 repetition came mainly from the task and output design;
- Intern still had a real visual weakness on the sampled warm-up action.

## 3. Contact timing: a useful-looking result that did not survive the rally test

The balanced trial had 60 proposed contacts. Each clip contained 50 frames over
two seconds. A gold border marked about ±10 base-30 frames around the proposal.
A cyan ring showed the tracker claim.

At a ±15-frame margin, Qwen's video-only answers reached 88.6% precision and
97.5% recall. Intern reached 83.3% precision and 50.0% recall. Requiring both to
agree gave 86.4% precision and 47.5% recall. At ±10 frames, agreement raised
precision from 63.6% to 72.7%, but cut recall from 96.6% to 55.2%.

This truth was only ShuttleSet timing. Some human serve labels are logical
guesses across a camera cut, so the score does not prove that the model saw a
physical contact.

The decisive test replayed Qwen's choices through 12 complete rallies. Exact
contact counts stayed at 4 of 12. Structurally usable rallies stayed at 1 of 12
at ±10 frames. Four replies were also invalid. The promising candidate score
did not become useful rally data.

## 4. Tracker validity: the one lead worth keeping

The tracker audit used 12 known hallucinations and 12 comparison clips near
ShuttleSet contacts. The comparison clips check whether a prompt rejects
everything. They are not human-confirmed real tracker paths.

Intern improved as the relevant pixels became easier to inspect:

| View | Hallucinations rejected | Comparison clips accepted |
|---|---:|---:|
| Plain marked target | 9/12 | 9/12 |
| Slow marked target | 9/12 | 12/12 |
| Slow, enlarged marked target | 11/12 | 12/12 |
| Enlarged marked target on held-out video | 5/6 | 6/6 |

Qwen rejected only 5 of 12 development hallucinations with the best enlarged
view. This closed the Qwen tracker lane.

The clean-then-marked counterfactual tested whether the cyan ring itself was
anchoring Intern. It rejected all 18 known hallucinations, but accepted only 7
of 18 comparison clips. The marked enlarged view remains the better lead.

## 5. Broadcast sequencing: still unreliable

The direct broadcast prompt gave each model 12 pure controls. Qwen labelled 11
of 12 as live. Intern produced 3 invalid replies and labelled its other 9 as
live. The direct question therefore missed most replay and cutaway content.

The second prompt explicitly described the common broadcast pattern: standard
live view, closer replay, then a return to standard view. Qwen was essentially
unchanged. Intern found the replay content but left only 2 of 6 live controls.
The wording traded replay blindness for excessive rejection.

## Exploratory runs and retries

Earlier contact pilots tested one to 24 cases while the clip layout, serve rule,
actor wording, and JSON fields were changing. Early tracker runs also tried
slightly different Qwen output fields after invalid replies. These runs guided
the final trials above. Their samples were too small or too changeable to carry
separate conclusions.

Prepared directories with no model attempts are not experiments. They are not
counted here.

## Evidence

- [`results.md`](results.md) gives the retained headline measurements.
- [`experiments/results/summary.json`](experiments/results/summary.json) holds
  the compact machine-readable counts.
- [`evaluation.md`](evaluation.md) explains the PR 80 failure in detail.
- [`sources.md`](sources.md) identifies the GitHub record and retained inputs.
