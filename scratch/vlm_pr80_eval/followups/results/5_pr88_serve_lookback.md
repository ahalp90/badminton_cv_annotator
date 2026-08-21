# Follow-up 5: PR 88 serve-lookback reconciliation

## Bottom line

Run no VLM hybrid from PR 88.

PR 88 contains a reproducible deterministic server rule worth testing on
unseen rallies. It raises server correctness from 163 to 170 of 239 development
rallies. It also introduces 13 errors while fixing 20, and its paired result is
not statistically clear. The pull request correctly treats this as a frozen
development candidate rather than a production result.

The retained VLM evidence does not supply a defensible way to combine the two
routes. Their overlap is small and selected. Follow-up 4 also showed that
giving Intern fallible pipeline proposals can reduce server correctness.

Keep the PR 88 rule separate. Its next test should remain an evaluation on
unseen rallies without VLM prompting or further development-set tuning.

## What PR 88 tested

The existing pipeline can accept a false early shuttle impulse as the first
contact in a rally. PR 88 searches forward through accepted contacts for the
first one followed by credible shuttle motion away from the attributed player.
It then uses the path before that contact:

- motion towards the player makes the contact look like a return, so the other
  player is proposed as server;
- other measured motion makes the contact look serve-like, so that player is
  proposed as server;
- an unusable path keeps the existing PR 82 answer.

Inference uses accepted contacts, their player attribution, shuttle tracks and
the PR 82 fallback. Path checks reject missing runs, repeated tracker positions
and large one-frame jumps. Human server and stroke labels enter only during
scoring.

The fixed development population contains 239 one-to-one rally mappings:

- 104 from `sset_01`;
- 84 from `sset_15`;
- 51 from `sset_21`.

The rule answers directly in 91 rallies and uses the PR 82 fallback in 148.
Its server score is 170/239, compared with 163/239 for PR 82. It repairs 20
PR 82 answers and damages 13. Its correct visible-start attribution rises from
125 to 132. Server and visible start are both correct in 117 rallies, compared
with 96 for PR 82.

The exact paired two-sided p-value for the server change is 0.296. The rule was
assembled after inspecting the development labels. These results therefore do
not establish an improvement on unseen data.

## Repository audit

The retained package supports its headline claims.

`python3 -m serve_id_followup.recompute --check` rebuilt the checked outputs
from eight frozen compressed records and found byte-for-byte matches. The six
package tests also passed.

The implementation matches the report:

- `preferred_decision()` uses the other player for a measured return-like
  contact;
- it uses the selected player for a measured serve-like contact;
- every unresolved case falls back to the frozen PR 82 answer;
- scoring reads the ground-truth server only after making that decision.

The row-level result contains all 239 decisions. The eight frozen source hashes
recorded in the checked metrics match during recomputation. The package does
not rerun TrackNet, pose estimation, contact generation or rally segmentation.

The curved-path proposal remains an audit claim rather than reproduced
evidence. Its exact path samples and audit helper are absent. PR 88 already
excludes it from the preferred rule.

## Why there is no hybrid experiment

Only 14 of Follow-up 2's 32 reviewed VLM cases appear in PR 88's one-to-one
population. Intern names 10 of those servers correctly. The PR 88 rule names
eight correctly. This is a retrospective comparison on cases selected for a
different audit, not a paired benchmark representative of 239 rallies.

Seven overlapping cases use PR 88's fallback branch. Intern and PR 88 agree in
six. The only disagreement favours Intern. Replacing fallback answers with
Intern would therefore change one observed case. It gives no evidence about
the other 141 fallback rallies and no predeclared selector for future cases.

Running Intern across those development fallbacks would add model inference
without fixing that evidence problem. It would evaluate a new combination on
the same labels already used to build PR 88's rule.

Prompting Intern with PR 88's server proposal is also unsupported. Follow-up 4
already tested an arm with fallible current-pipeline server and contact
proposals. That arm reduced server correctness from 23 to 18 of 32. PR 88's
proposal is different, but the retained evidence gives no basis for assuming
that Intern would use it selectively rather than follow it as a prior.

No simple hybrid therefore meets the plan's evidence bar. This is the planned
no-experiment outcome, not a failed run.

## Decision

End the VLM follow-up series here.

Keep Intern as the clean-interface model choice recorded in Follow-up 2. Keep
the negative compact-evidence result from Follow-up 4 unchanged.

Treat PR 88's preferred layered rule as a separate deterministic candidate.
Evaluate that frozen rule on unseen rallies before considering production use.
Report server side, visible-start attribution and joint correctness separately.

## Evidence

- [PR 88](https://github.com/ahalp90/badminton_cv_annotator/pull/88) contains
  the original pull-request account and commit history.
- [`../../../serve_id_by_lookback_followup/report.md`](../../../serve_id_by_lookback_followup/report.md)
  explains the retained investigation and its limits.
- [`../../../serve_id_by_lookback_followup/serve_id_followup/rules.py`](../../../serve_id_by_lookback_followup/serve_id_followup/rules.py)
  contains the frozen prediction branches.
- [`../../../serve_id_by_lookback_followup/serve_id_followup/recompute.py`](../../../serve_id_by_lookback_followup/serve_id_followup/recompute.py)
  rebuilds and checks the result.
- [`../../../serve_id_by_lookback_followup/results/development_metrics.json.gz`](../../../serve_id_by_lookback_followup/results/development_metrics.json.gz)
  contains the checked counts, diagnostics and source hashes.
- [`../../../serve_id_by_lookback_followup/results/preferred_server_rule.csv.gz`](../../../serve_id_by_lookback_followup/results/preferred_server_rule.csv.gz)
  contains all 239 scored decisions.
- [`5_pr88_serve_lookback.json.gz`](5_pr88_serve_lookback.json.gz) records this
  audit, overlap diagnostic and decision in machine-readable form.

The checked metrics file has SHA-256
`32c867162182a1d5616fa33f5a2056d625d39f537b9fe871000f4b085220d5b4`.
The row-level preferred-rule file has SHA-256
`a522b7dcb88278ae5d88abaf66e11ee98a9d2f75d6f98afef97b34746bcfc284`.
