# Last contact-detector follow-ups

Keep the existing detector: the local chooser followed by fixed-membership
padding. Independent edge padding produced no complete-rally gains. Correcting
the chooser's training targets gave a small development improvement, then lost
more rallies than it repaired on the 47 broader videos. It also broke two
currently correct selected clips and repaired none.

These are scripts-only experiments. A complete rally contains the whole labelled
contact sequence, matches each contact once, and assigns the correct players.
The main timing allowance is ±10 frames on a 30 fps clock; ±5 is a secondary
check. Both allowances scale to each video’s frame rate. Automatic approval
remains off, and production wiring stays unchanged.

The chooser selects a candidate contact sequence for each proposed clip.
Fixed-membership padding extends the clip while keeping its predicted contacts.
Trusted-label scoring uses the existing cleaned population; all-source scoring
restores the excluded rallies. Unknown means the labels do not settle correctness.

## Independent edge padding

The existing boundary rule cancels both extensions when padding would admit an
outside predicted contact. The alternative keeps the available space at each
edge. It uses the same padding allowance and neighbouring-clip limits, so
contact timestamps and player evidence are unchanged.

The replay used saved final chooser outputs from the previously examined 47
ShuttleSet22 videos. They contain 3,982 proposed clips, so this is not a new
untouched test.

| Labels | Timing allowance | Existing rule | Independent edges | Repairs | Losses |
|---|---|---:|---:|---:|---:|
| Trusted: 3,422 rallies | ±10 | 1,763 | 1,763 | 0 | 0 |
| Trusted: 3,422 rallies | ±5 | 1,430 | 1,430 | 0 | 0 |
| All source: 3,965 rallies | ±10 | 1,763 | 1,763 | 0 | 0 |
| All source: 3,965 rallies | ±5 | 1,429 | 1,429 | 0 | 0 |

Only two proposed clips changed. One started three frames earlier but still had
an extra predicted contact. The other started one frame earlier and contained
the whole labelled rally, but still missed a contact. Neither became fully
correct.

The same 784 selected clips were rescored. At ±10, the trusted-label result
stayed at 616 correct, 124 wrong and 44 unknown. With all source labels, it
stayed at 615 correct, 140 wrong and 29 unknown. Selection results at ±5 also
stayed unchanged.

The proposed rule behaves as intended, but its opportunity is too small on these
saved outputs to justify replacing the existing rule. No contact model was
fitted for this check.

Evidence: [saved recount](results/last_followups/edge_padding.json.gz),
[replay script](scripts/replay_edge_padding.py).

From the repository root, use the project environment and a fresh output
filename:

```bash
PYTHONPATH="$PWD/src:$PWD" python -m \
  scratch.contact_det_closing_pass.scripts.replay_edge_padding \
  --annotations /path/to/shuttleset22/annotations \
  --output /tmp/edge_padding.json.gz
```

The replay checks that the existing trusted-label result is 1,763 before it
records the comparison.

## Chooser targets after padding

The chooser learns whether an alternative is correct before the final boundary
operation. Some alternatives already have the right contact sequence, but their
clip starts just after the labelled serve. Padding fixes that containment issue
after the chooser decides.

The census used 942,471 saved alternatives from 32 development videos in groups
A–D. Each alternative was placed into the same prediction-based reference
stream for its video. Existing padding then ran on the complete stream. Labels
scored the resulting answer; they did not determine the bounds.

| Development group | Answers changed from negative to positive | Proposals affected | Currently wrong proposals affected |
|---|---:|---:|---:|
| A | 246 | 75 | 39 |
| B | 242 | 70 | 33 |
| C | 191 | 59 | 27 |
| D | 127 | 40 | 17 |
| **Total** | **806** | **244** | **116** |

No positive answer became negative. All 59,757 excluded answers remained
excluded. Positive alternatives rose from 6,834 to 7,640, and 154 proposals
gained their first positive alternative. Changes occurred in 27 of the 32
videos.

The 806 changed alternatives are not 806 new complete rallies: several belong
to the same proposal, and some affected proposals were already correct after
padding. The 116 currently wrong proposals make this a useful target correction
to test with the existing chooser.

For example, one 25 fps clip starts at frame 55,914 while its labelled serve is
at 55,912. Its candidate matches all 20 contacts with the correct voted
players. Existing padding moves the start to 55,906 and makes that candidate
complete. Three changed examples were inspected; one succeeds at ±10 but still
fails at ±5.

Evidence: [complete target count](results/last_followups/padded_targets.json.gz),
[census script](scripts/run_padded_target_census.py).

### Controlled fit

The fit kept the same alternatives, features, opening and local models, and the
0.05 edit rule. Each development group was predicted by a chooser trained on
the other three groups. Old and corrected outputs received the same existing
padding before comparison. Cached upstream detector scores still have
cross-group dependence, so this is a development comparison rather than a
fully independent estimate.

| Timing allowance | Existing chooser | Corrected targets | Repairs | Losses |
|---|---:|---:|---:|---:|
| ±10 | 1,209 | 1,218 | 22 | 13 |
| ±5 | 958 | 965 | 14 | 7 |

These counts cover 2,691 labelled rallies and 2,850 proposals. The primary net
changes by group were A +8, B −1, C +4 and D −2. The correction improves the
total slightly, but the losses and uneven groups warrant caution. The broader
comparison does not support adopting the refit.

Evidence: [development comparison](results/last_followups/padded_fit_development.json.gz),
[fit script](scripts/run_padded_target_fit.py).

### Broader finished outputs

The final fitted chooser was replayed on the same previously examined 47
videos. The original choice reference and edit rule stayed fixed. Both choosers
then received existing padding, and source labels were read only after
prediction.

| Labels | Timing allowance | Existing chooser | Corrected targets | Repairs | Losses |
|---|---|---:|---:|---:|---:|
| Trusted: 3,422 rallies | ±10 | 1,763 | 1,761 | 21 | 23 |
| Trusted: 3,422 rallies | ±5 | 1,430 | 1,424 | 13 | 19 |
| All source: 3,965 rallies | ±10 | 1,763 | 1,761 | 21 | 23 |
| All source: 3,965 rallies | ±5 | 1,429 | 1,423 | 13 | 19 |

The chooser changed 211 of 3,982 proposals. Primary repairs occurred in 17
videos and losses in 18; seven videos had both. The saved comparison includes
each rally identity and per-video count. The aggregate trade-off is slightly
worse, despite the genuine training-target mismatch.

The same 784 clips remained selected throughout:

| Labels | Allowance | Correct before → after | Wrong before → after | Unknown | Repairs / losses |
|---|---|---:|---:|---:|---:|
| Trusted | ±10 | 616 → 614 | 124 → 126 | 44 | 0 / 2 |
| Trusted | ±5 | 549 → 547 | 191 → 193 | 44 | 0 / 2 |
| All source | ±10 | 615 → 613 | 140 → 142 | 29 | 0 / 2 |
| All source | ±5 | 549 → 547 | 207 → 209 | 28 | 0 / 2 |

At the primary allowance, trusted-label precision among judgeable selections
falls from 616/740 (83.24%) to 614/740 (82.97%). With all source labels, it
falls from 615/755 (81.46%) to 613/755 (81.19%). Crediting only verified
correct clips among all 784 selections gives 615/784 (78.44%) to 613/784
(78.19%). Unknown clips remain in the accounting.

Contact-level changes are small. Percentages below are existing → corrected.
Timing precision counts matched predictions, and recall counts recovered
labels. F1 is the harmonic mean of precision and recall. The player-aware score
also requires the correct player after sequence voting.

| Labels | Allowance | Timing precision | Timing recall | Timing F1 | Player-aware F1 |
|---|---|---:|---:|---:|---:|
| Trusted | ±10 | 81.04 → 80.97 | 88.22 → 88.25 | 84.48 → 84.45 | 81.85 → 81.89 |
| Trusted | ±5 | 79.25 → 79.20 | 86.27 → 86.32 | 82.61 → 82.61 | 80.19 → 80.26 |
| All source | ±10 | 90.10 → 90.03 | 86.85 → 86.89 | 88.45 → 88.43 | 85.58 → 85.59 |
| All source | ±5 | 88.10 → 88.04 | 84.93 → 84.97 | 86.49 → 86.48 | 83.92 → 83.94 |

Predictions rose from 41,605 to 41,652. Trusted timing matches rose from 33,716
to 33,726 out of 38,218 labels. Matched trusted serves rose from 2,781 to
2,790 out of 3,422. These small contact gains do not compensate for lost
complete rallies and correct selected clips.

The 23 primary losses have concrete sequence errors: 12 finish with too few
contacts, seven with extras, and four with mistimed replacements. Replaying the
saved sequences against the labels reproduced the reported outcomes. The
selected losses are `47/set2:33`, where two contacts become one, and
`48/set3:21`, where a required contact is removed. Repairs include
missing-contact additions, extra-contact removals and timing corrections. This
review checks labels and predicted sequences; it does not establish physical
impact times from video.

Do not adopt the refit or tune another threshold to rescue it. The training
correction was plausible and cheap enough to test once, but its finished outputs
do not improve the intended result.

Evidence: [broader comparison and changed sequences](results/last_followups/padded_fit_broader.json.gz),
[finished-output scorer](scripts/score_padded_chooser.py). The inference runner
now accepts a separate score directory so this experiment preserves the
original model scores.

## Small repairs within selected clips

The final check used 570 development clips selected by the saved ranking rule:
448 correct, 119 wrong and three unknown. Selection stayed fixed. All candidate
timestamps came from the existing option pool. Each alternative was tested one
proposal at a time in the current full-video choice map, then passed through the
same padding and player vote. This is a fixed-context opportunity count, not
model-performance evidence.

Of the 119 wrong clips, 58 have a complete small-edit alternative. Labels
identify those alternatives, so this is possible headroom with a label-guided
choice, not achieved correction performance.

| Small edit | Wrong selected proposals that could be repaired |
|---|---:|
| Delete an event before the first label | 5 |
| Delete an event after the last label | 17 |
| Insert one later contact from the existing pool | 16 |
| Replace one event with an existing candidate | 20 |
| **Total unique proposals** | **58** |

The affected proposals span all four development groups: A 20, B 11, C 22 and
D 5. The other 61 wrong clips have no complete alternative within this
small-edit count. All 448 currently correct clips also have at least one
damaging edit available. That is exposure to a bad choice, not a claim that a
correction model would necessarily break them. The three unknown clips receive
no claimed repairs.

The 20 replacements recover a label that the removed event could not match at
the primary allowance. This describes the matching result; it does not
distinguish a mistimed physical hit from a separate extra and missing hit. Saved
examples include both timing allowances, but the exhaustive opportunity count
targets ±10.

The proposed pre-serve cleanup accounts for only five possible repairs here.
Tail deletion is the larger endpoint pattern, with 17. Those tail events are
not separated by unusually long gaps: the final predicted gap ranges from
13.2 to 46.0 frames on the 30 fps clock, with a median of 26.0. Among 447 currently
correct clips with at least two predictions, the range is 7.2–68.4 and the
median is 27.6. This overlap gives no simple long-pause rule for safely dropping
the last event.

Stop before another correction fit. The count identifies specific cases but has
not established prediction evidence that separates repairs from harmful edits.
A future visual review could compare these 17 tail cases with correct rally
endings. That is narrower than repeating the [earlier broad deletion
experiment](serve_and_acceptance.md). No coverage expansion or additional model
was run in this final check.

Evidence: [selected repair count and examples](results/last_followups/selected_repairs.json.gz),
[census script](scripts/count_selected_repairs.py).

## What remains useful

Keep the existing detector for proposing rally clips and drafting contacts for
review. These follow-ups do not justify automatic exact approval. Independent
edge padding produced no gain, and the controlled chooser refit worsened the
finished output. The remaining-error count leaves a concrete tail-event review
lead without adding another model to the detector.

## Reproducing the chooser experiments

Use the project Python environment from the repository root. These commands
need the existing prepared option and feature caches. Choose a fresh output
directory and supply the source annotation directory for the broader recount.

```bash
export PYTHONPATH="$PWD/src:$PWD"
followup_run=/path/to/fresh/contact-followup
python -m scratch.contact_det_closing_pass.scripts.run_padded_target_census \
  --output-root "$followup_run/targets" --jobs 16
python -m scratch.contact_det_closing_pass.scripts.run_padded_target_fit \
  --census "$followup_run/targets" --output-root "$followup_run/fit" --jobs 4
python -m scratch.contact_det_closing_pass.scripts.run_insertion_broader \
  --variant local --models "$followup_run/fit/models.joblib" \
  --output-root "$followup_run/broader" --score-root "$followup_run/scores" --jobs 4
python -m scratch.contact_det_closing_pass.scripts.score_padded_chooser \
  --predictions "$followup_run/broader/local_broader_predictions.json.gz" \
  --annotations /path/to/shuttleset22/annotations \
  --output "$followup_run/broader/padded_comparison.json.gz"
python -m scratch.contact_det_closing_pass.scripts.count_selected_repairs \
  --output-root "$followup_run/selected_repairs" --jobs 16
```

For a one-video smoke run, the target census accepts `--limit-fixtures 1`; the
selected-repair census accepts `--limit-videos 1`. Use another fresh output
directory for the full run. The broader runner uses separate smoke filenames.

## Checks

The experiment smoke runs and full runs completed successfully (exit 0).
Both sets of four focused tests passed, and scoped Ruff checks passed (exit 0).
Serena/Pyrefly reported no diagnostics in the changed scripts. The whole-project
Pyrefly check returned exit 1 with 11 missing-import errors in unchanged tests,
helper scripts and optional video-language-model dependencies.
