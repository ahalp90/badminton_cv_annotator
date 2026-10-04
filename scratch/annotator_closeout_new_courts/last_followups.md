# Last annotator follow-ups

**Keep the nomination veto off.** After refitting on the new courts, it loses four ShuttleSet22 rallies for a three-rally validation gain. The detector experiments rejected before the court change are in the [earlier follow-ups](../annotator_wrapup_evaluation/last_followups.md), and the old-court refit trials are in the [refit regression report](../../experiments/annotator/reports/refit_regression.md). Work still to do is in [promising_leads.md](promising_leads.md).

**Contents**  
[Nomination veto on the new courts](#nomination-veto-on-the-new-courts)  
[Earlier experiments still stand](#earlier-experiments-still-stand)

## Nomination veto on the new courts

The veto rejects a possible hit when the shuttle track is flagged unreliable and no player was selected at any of five nearby samples (−10 to +10 frames at 30 fps). On the old courts it gained nine complete rallies and lost none. It was retested on the new courts with refitted sequence and review models.

| Measure | Base | Veto |
|---|---:|---:|
| Validation complete rallies / 668 | 273 | **276** |
| ShuttleSet22 complete rallies / 3,327 | **1,744** | 1,740 |
| ShuttleSet22 timing matches | **34,200** | 34,197 |
| ShuttleSet22 gained / lost against base | — | 44 / 48 |

Simply switching the rule on, without refitting, made validation worse: 270 complete rallies. Most of the validation side-assignment gain came from one video, `sset_31`.

The review ranking briefly favoured the veto, but the advantage faded as the queue grew:

| ShuttleSet22 clips reviewed | Base: correct / wrong / unjudgeable | Veto: correct / wrong / unjudgeable |
|---|---:|---:|
| 500 | 407 / 84 / 9 | 423 / 67 / 10 |
| 1,000 | 802 / 169 / 29 | 798 / 174 / 28 |
| 2,000 | 1,356 / 462 / 182 | 1,361 / 478 / 161 |

Of the 48 ShuttleSet22 rallies the veto lost, 27 miss their first labelled hit. The rule tends to damage rally starts, where serves are already weakest.

**Decision:** keep the veto off. Its old-court gain did not carry over, and court geometry now changes which players get nominated.

Evidence: `experiments/annotator/good_court_refit/evidence/test-base-vs-veto.json.gz`, `validation-base-vs-veto.json.gz`, `validation-base-vs-base_inference_veto.json.gz`  
Report: [model_selection.md](../../experiments/annotator/reports/model_selection.md)

## Earlier experiments still stand

These were decided on the old courts and nothing here reopens them:

- **Independent edge padding:** two proposals changed, none became correct.
- **Correct chooser targets after padding:** the target mismatch is real, but the broader run lost two complete rallies.
- **One-edit repair headroom:** 58 of 119 wrong development clips repairable in hindsight; not enough for another correction model.
- **Endpoint deletion:** “after the last label” is not a safe deletion rule.
- **Repairing video 15:** excluded instead.
- **Noise-aware training:** deferred until a small verified contact set exists.

The [earlier follow-ups](../annotator_wrapup_evaluation/last_followups.md) give the counts and evidence.
