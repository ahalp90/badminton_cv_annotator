# Corrected serve trajectory investigation handover

## Current state

The experiment, report and independent result checks are complete. Work is on branch `investigation/serve-start-trajectory`; no production code under `src/**` changed.

The previous EDA answered the wrong question. It used `fitted_first_all`, a rally-wide alternating-fit server label, as the player that the shuttle supposedly approached. It then flipped that scalar label. It did not use the earliest contact's own geometric attribution, prepend a contact, or rerun the alternating fit.

The corrected work has two experiments:

1. Take the earliest accepted contact in each predicted rally. Call `attribute_half` at that exact frame. Look backwards for a credible court-view shuttle path approaching that player. If the detector triggers, infer that the other player served. Otherwise the simple contact-local heuristic names the anchor player as server; an abstaining view will show the coverage trade-off.
2. Rebuild the ordered direct player guesses for every accepted contact. On a trigger, compare prepending `None` with prepending the inferred server half, then call the existing `fit_alternation` once. The first variant isolates the missing-contact parity effect. The second adds one player vote. No serve frame is invented.

The main first-return threshold uses only anchors that fall within tolerance of exactly one ShuttleSet contact, where that contact is contact 1 or 2. The existing extraction has 87 contact-1 matches and 17 contact-2 matches, but one contact-2 case is also within tolerance of contact 1. This leaves 16 clear positive examples. Later, ambiguous and unmatched anchors remain in separate all-rally server evaluations.

## Result

The main rule found 11 of 16 clear returns, made 3 false calls and missed 5. This is 78.6% precision, 68.8% recall and 0.733 F1. A stricter rule that excludes TrackNet's filled or interpolated points found 9 returns with no false calls.

The main rule fired in 16 covered rallies. Naming the other player directly was right in 13. The released alternating fit was right in 7. Prepending an unknown-player contact and refitting was right in 8. Prepending a contact by the other player and refitting was right in 9. The direct motion inference is useful; the refit is not.

Read `report.md` first. It gives the complete plain-language account and all denominators. Generated plots and compressed result tables remain ignored under `outputs/`.

The headline threshold plot uses plain units. Its x-axis is the minimum share of step-to-step shuttle movements that reduce distance to the anchor player. Precision, recall and F1 are percentages. TP, FP and FN counts appear at the chosen point. Minimum net closure is expressed in player body heights. Quadratic fit results are a separately labelled diagnostic, not proof of a real parabola.

## Read next

- `plan.md`: agreed work, checks, exclusions and commit messages.
- `findings.md`: verified source and data contracts.
- `decisions.md`: agreed and recommended choices.
- `QUESTIONS.md`: now answered; remains untracked as originally requested.
- `WEBUI_RED_TEAM_PROMPT.md`: prompt used for the WebUI review.
- `red_team_review.md`: disagreement between Gemini and Claude, with the source-grounded resolution.
- `worklog.md`: short session history and checks.

## Important source facts

- Use `min(filtered_by_rally[rally_id])`, not list position zero, as the safest earliest accepted frame.
- Raw `ContactCandidate` rows do not store impulse magnitude or direct Top/Bot attribution.
- `attribute_half` can return `None`; an exact equal-distance tie silently favours Top and must be logged.
- Earlier raw impulses may have failed the wrist gate, suppression or definitive exclusion. Record them as diagnostics unless the user asks for a new credibility rule.
- Adding a leading position leaves the phase assignments of all existing contacts unchanged. It changes first-player parity. A non-missing leading half adds one phase vote.
- The stable GT key is `(video_id, set_id, rally)`. Map it to a predicted span with `classify_all`.
- The frozen failure subset has 121 covered rallies: 99 wrong released server labels and 22 missing labels. Do not confuse this with the 136 later-stroke contact-miss subset.

## Files and Git

The new packet is under `scratch/serve_start_trajectory_exploration/20260810-210532-corrected-contact-refit/`. The root ignore rule now admits this one corrected run. Its own `.gitignore` keeps inputs, outputs, plots, case images, questions and `external_delegate/` untracked.

Approved checkpoints `464b926`, `981baae` and `87805cf` capture the setup, motion measurement and refit result. The final plain-language report and close-out documents belong in the fourth approved commit.

Generated arrays must be `.npy.xz` using XZ preset 9. Generated JSON and CSV must be `.json.gz` and `.csv.gz`.
