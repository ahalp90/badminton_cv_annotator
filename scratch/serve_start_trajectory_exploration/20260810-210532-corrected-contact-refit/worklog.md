# Worklog

## Resume

- **Next action:** write the file-only Opus Batch 2 red-team brief. Verify and fix any reproduced finding before the approved Batch 2 commit.
- **Current batch:** Batch 1 was committed as `360c9b3`. Batch 2 analysis, fixed decisions and independent validator are implemented but uncommitted. The corrected analysis and validator both pass. Corrected scores have now been observed only after every rule and threshold was fixed in the planning records.
- **Verified so far:** the saved bundle contains 292 rallies, 344 predicted spans, 1,012 selected path points, 16 fixed rule rows and 270 trend-diagnostic rows. The validator independently rebuilds all saved tables and metrics without importing the analysis or feature module. Dedicated Ruff, pinned Pyrefly, the full test suite and `git diff --check` pass. Whole-project Ruff still reports the unchanged baseline of 661 unrelated findings. Serena reports no diagnostics in `validate_outputs.py`.
- **Runbook pointer:** `plan.md`, “Correction extension: evaluation accounting and readable final outputs”.

## Compaction boundary

### Overall goal

Repair the serve-start trajectory investigation so its rally mapping, contact alignment, motion evidence, server results, plots and report are numerically sound and readable in one pass.

### Must keep

- Primary 239 one-to-one rallies; 249 covered as merge sensitivity; 292 GT rallies as end-to-end segmentation-inclusive view.
- ±10 base-30fps frames is primary contact alignment; ±5 is strict and ±30 is the sanity check. Keep nearest ordinal, signed and absolute offset, and a separate multiple-match flag at every tolerance.
- For ±10-unmatched anchors, inspect later accepted contacts for the GT serve, first return and first matched rank.
- Historical rule stays fixed at 0.25-BH total movement, 0.25-BH net closure and 55% approaching steps.
- New rule uses a median-pairwise robust trend and calls incoming only at fitted decrease >= 0.05 BH. This is an engineering judgement fixed before corrected scoring.
- Residual RMS and trend-to-jitter are continuous diagnostics only. Analyse them by GT class, call correctness, path length and representative errors. Never turn them into another cutoff here.
- Apply both frozen rules unchanged to recurrence-only and recurrence-plus-producer-mask paths.
- Preserve the OUT-list, all three user audit files, per-video reporting, final WebUI fresh read and every red-team gate.
- Work remains on `investigation/serve-start-trajectory`; do not merge or touch `src/**`, frozen inputs/GT, segmentation behaviour, `experiments/**`, `.claude/**` or `.env`.

### Safe to drop after compaction

- Full text of the completed original report, historical handover and old reviewer responses; their paths and relevant corrections are recorded in the plan and decisions.
- Raw shell output from the planning inspection and the full skill manuals already applied.
- The conversational route by which 0.25 was challenged; the provenance finding and physical objection are now explicit in `decisions.md` and `plan.md`.
- Detailed prose from the current chat once the fixed populations, rules, reviews and OUT-list above survive.

### Next steps

1. Create a fresh file-only `claude-opus-4-6-thinking` Batch 2 red-team brief. Do not ask direct-host agy to run shell commands.
2. Verify every reported issue locally. Then commit Batch 2 with the approved message if clean.
3. Batch 3 rewrites the report and plots, applies both readability audits, and stops for the planned fresh WebUI cold read.

## Concerns and observations

- **Correction planning:** the current `build_feature_rows` emits one row per GT rally and reuses a predicted span's accepted-contact sequence for every `COVERED` row. The correction must derive span multiplicity before primary scoring.
- **Correction planning:** the current anchor label uses one canonical tolerance and collapses multiple in-window strokes to `ambiguous`. The correction must retain nearest ordinal and offsets at ±5, ±10 and ±30, with a separate multiple-match flag.
- **Correction planning:** the current main function selects a separate best threshold for `producer_original`. The corrected comparison must not select any rule or threshold from the producer-mask scores.
- **Correction planning:** both 0.25-body-height cut-offs first appear with the implemented analysis. The pre-implementation plan said cut-offs would be swept and gives no physical basis for 0.25, so the correction must treat them as historical same-analysis choices.
- **Correction planning:** the current validator independently rebuilds score tables, but it shares the old populations and single anchor label. Its independence must extend to mapping, tolerance, sequence-rank and fixed-threshold checks.

## Module state

- `experiment_data.py`: loads frozen tracks, player evidence, accepted contacts, rally boundaries and first/second-stroke truth. It does not yet expose the full GT stroke truth needed by the correction.
- `trajectory_features.py`: reduces an anchor/tolerance join to one category. It does not retain nearest-stroke details or a separate multiple-match flag.
- `analyse_serve_trajectory.py`: scores all `COVERED` rows, uses one anchor label and selects separate thresholds for both path masks. These are the main correction touch-points.
- `validate_outputs.py`: independently recalculates the old threshold and server tables. It must be extended to cover the corrected evaluation contract.
- `report.md` and plots: describe the original completed run. They remain unchanged until corrected outputs pass the numerical gates.

## 2026-08-10: corrected scope

The earlier EDA was challenged because it started from the rally-wide alternating-fit label rather than the earliest contact's own measured player. It also reversed the final scalar label without prepending a contact or rerunning the fit.

The corrected goal is now fixed: test incoming motion towards the directly attributed earliest accepted contact, then separately test the effect of prepending an inferred server half and running one augmented-sequence alternation fit.

Read-only source tracing established the accepted-contact flow, direct half attribution, fit semantics, GT join, release inputs and sticky reconstruction. Read-only delegated reviews checked the smallest useful motion gates and the unknown-time refit seam.

Independent Gemini and Claude reviews disagreed about the prepend. Direct inspection resolves the disagreement: adding a leading position leaves every original contact's phase assignment unchanged, but changes the first-player parity. A non-missing inferred half adds one phase vote. The plan now compares a `None` prepend with an inferred-half prepend so these effects cannot be conflated.

At this planning point, no implementation had started and no production code was going to change. The next step was user approval of the remaining choices and commit messages, followed by a WebUI and direct-host red-team review of the plan.

## Checks run by planning reviews

- 54 focused trajectory, inpaint and sticky tests: passed.
- 9 focused alternating-fit/injection tests: passed.
- 20 focused contact/serve tests: passed.
- Repository diffs from the read-only workers: none.

These checks confirm existing behaviour only. They do not validate the corrected experiment, which has not been implemented.

## 2026-08-10: user approval and WebUI review

The user approved the 30-frame maximum, raw-candidate diagnostics, both prepend variants, both no-path results and all proposed commits. They asked that the central question stay in ordinary language.

The WebUI review found no circularity. It asked for five exact rules before implementation: choose the nearest path inside the maximum window; keep the path and anchor in one tracker scene; classify GT matches by the literal count within tolerance; keep no-path abstention separate; and choose the displayed setting by first-return F1. These rules are now in `plan.md`.

Implementation may proceed autonomously.

## 2026-08-10: experiment run

The dedicated scripts linked and verified all three frozen releases, TrackNet arrays and pose arrays. The analysis rebuilt direct player attribution for every accepted contact. The unmodified alternating fits matched the frozen results before any counterfactual contact was added.

The session originally described 249 GT rallies as covered by one predicted span. The correction review found that those rows use 244 distinct predicted spans, including five spans shared by two GT rallies. Clear anchor truth under the original join was available for 87 serves and 16 first returns. The original incoming-motion rule found 11 returns, made 3 false calls and missed 5 returns.

The rule fired in 16 covered rallies. Directly naming the other player as server was right in 13. Prepending an unknown-player contact and refitting was right in 8. Prepending a contact by the other player and refitting was right in 9.

The first independent review found unclear denominators and labels in the report and plots. Those labels now say exactly what is counted. The arithmetic audit reproduced the result and confirmed that GT changes do not affect per-rally decisions. It also found that the first validator reused analysis code, so the validator was replaced with an independent recalculation.

Focused checks so far:

- trajectory tests: 24 passed;
- dedicated Ruff check: passed;
- independent output validator: passed;
- full analysis over all three videos: passed.

The full repository gates remain before close-out.

## 2026-08-10: close-out checks

The regenerated report passed a second cold read. The reviewer found the earlier denominator, method-name, abstention, axis and TrackNet-source ambiguities resolved.

Final checks:

- dedicated Ruff check: exit 0;
- independent output validator: exit 0;
- pinned whole-project Pyrefly: exit 0, no errors and 20 configured suppressions;
- whole-project pytest: exit 0, 1,456 passed and 29 skipped;
- whole-project Ruff: exit 1 on 661 existing findings across unrelated files;
- `git diff --check`: exit 0.

The dedicated directory has no Ruff findings. The repository-wide Ruff failures are outside this investigation and were left unchanged.

## 2026-08-11: correction scope and launch plan

WebUI feedback exposed substantive evaluation-accounting problems in the completed report. The report needs a corrected analysis rather than a prose-only edit.

The user approved three distinct populations: 239 one-to-one mapped rallies for the primary downstream experiments, 249 covered rallies for merge sensitivity, and 292 GT rallies for the end-to-end view including segmentation failure. They also made ±10 base-30fps frames the primary contact-alignment baseline, with ±5 strict and ±30 a sanity check.

The corrected rows will retain the nearest GT ordinal and signed and absolute offsets at every tolerance. A separate flag will show whether multiple GT strokes lie inside the tolerance. For ±10-unmatched anchors, the analysis will inspect later accepted contacts for the GT serve, first return and the rank of the first GT-matched contact.

The correction will preserve the historical rule exactly: 0.25 body heights of total movement, 0.25 body heights of net closure and 55% of steps towards the player. These values will not be retuned or described as physical facts.

A second rule was predeclared before reading corrected classification scores. It fits a median-pairwise-slope trend to shuttle-to-player distance and requires at least 0.05 apparent player body heights of fitted decrease across the observed path. It removes the absolute total-movement floor, keeps the existing gross-jump gate and does not require every step to approach. The 0.05 value is an engineering judgement, not a calibrated physical constant.

Residual RMS and trend-to-jitter remain diagnostics only. The report will compare them by GT class, call correctness, sampled path length and representative errors without adding another cutoff.

Both frozen rules will be scored on the 239-rally primary set with unique ±10 contact-1/contact-2 labels. Both will also be applied unchanged to recurrence-only and recurrence-plus-producer-mask paths. The report will compare their assumptions and decision overlap rather than choose whichever scores better.

The user supplied two additional audit records. The final report and every supporting plot must pass them, plus `write-clearly` and `de-yuck`. A fresh WebUI reader will receive the report, questions and plots without the planning records before the final branch audit.

No delegate has launched. No implementation file or generated output has changed. The next step is user approval of the runbook and its three proposed commit messages.

## 2026-08-11: launch and baseline

The user replaced the proposed trend-to-jitter classifier with a simpler fixed rule. Fit the robust distance trend and call incoming when its fitted decrease is at least 0.05 apparent player body heights. Residual RMS and trend-to-jitter remain continuous diagnostics only.

The user approved launch with this final rule. The historical rule, new rule and both inpaint masks are frozen before corrected scoring.

The original analysis ran once from the committed code and the independent validator passed. The tracked report did not change. One validated baseline capture is stored under `local_scratch/serve_start_trajectory_correction/20260811-pre-edit-baseline/`. A duplicate pre-edit run was removed from the plan as unnecessary for this deterministic local analysis.

Two approved headless Codex `gpt-5.6-luna`, effort `max`, read-only sweeps launched with the reused Serena endpoint `http://127.0.0.1:9121/mcp`:

- evaluation accounting: unified session `84558`, artefacts under `local_scratch/external_delegate/20260811-080252-trajectory-accounting-sweep/`;
- motion and reporting: unified session `98627`, artefacts under `local_scratch/external_delegate/20260811-080252-trajectory-motion-sweep/`.

Both workers have fresh named-file briefs, 25,000-token budgets and no write, commit, push or credential authority.

Both sweeps completed with exit code 0 and reported no worker-created repository diff. The accounting sweep confirmed the 292/249/244/239 mapping only when span IDs are scoped by fixture. It identified the missing full-stroke, span and accepted-sequence fields needed for independent validation. The motion sweep confirmed that common path eligibility must be separated from the historical 0.25-BH movement floor before the new rule is wired.

Primary checks reproduced 239 one-to-one rows, 249 covered rows, 244 distinct `(fixture, span_id)` keys, five merged spans and 10 merged rows. Primary rows break down as `sset_01=104`, `sset_15=84` and `sset_21=51`. Source inspection confirmed the old ±5 anchor join, unfiltered threshold population and separate per-mask threshold selection.

The user asked Serena to index this investigation. `.serena/project.yml` now adds only this directory as an additional workspace folder and replaces the broad `scratch/**` exclusion with `scratch/swarm_review/**`. Serena restarted cleanly and returned the symbol overview for `trajectory_features.py`, so no local Pyrefly profile change was needed.

The approved direct-host `claude-opus-4-6-thinking` planning red-team completed with exit code 0 and a passing Git tripwire. It found Batch 1 safe to start. It asked for explicit non-consuming GT matches and one-based full-sequence rank semantics in the later-contact diagnostic; both are now in the plan. Its suggestion to freeze a threshold from one mask was rejected because both rules already have user-fixed decisions that must not be selected from corrected scores.

## 2026-08-11: Batch 1 implementation and gates

`trajectory_features.py` now has three unwired pure measurements:

- `AnchorAlignment` keeps the nearest one-based GT stroke, anchor-minus-GT offset in base-30fps frames, absolute offset, inclusive in-window count, separate multiple flag and nearest-stroke label.
- `AcceptedSequenceSummary` checks later contacts independently after an unmatched anchor. It records serve and first-return matches, one-based full-sequence rank, nearest ordinal, first-match multiplicity and GT-ordinal reuse.
- `RobustDistanceTrend` implements the predeclared pairwise-median slope, median intercept, fitted decrease, residual RMS and zero-residual ratio convention.

The old `classify_anchor_frame` and every analysis call remain unchanged. Synthetic coverage rose from 24 to 50 tests. It covers 25/30/60fps tolerance boundaries, signed offsets, equidistant ties, multiple strokes, sequence rank and reuse, no-match cases, robust inward/outward/constant/noisy paths, a bad endpoint, zero residuals and constant rescaling.

Batch 1 gates:

- focused trajectory tests: exit 0, 52 passed;
- dedicated investigation Ruff: exit 0;
- full historical analysis: exit 0;
- independent historical validator: exit 0;
- decompressed metrics and row tables versus pre-edit capture: identical;
- report and plots versus pre-edit capture: identical;
- pinned whole-project Pyrefly: exit 0, 0 errors and 20 suppressions;
- whole-project pytest: exit 0, 1,456 passed and 29 skipped;
- whole-project Ruff: exit 1 on the same 661 unrelated findings;
- `git diff --check`: exit 0;
- Serena diagnostics for both changed Python files: none.

The first Pyrefly attempt exited 2 because the sandbox could not create a uv cache lock. The required command passed unchanged with existing-cache access outside the sandbox.

The first Gemini Pro launch produced no report because the brief asked it to run `git diff`, and direct-host read-only mode correctly denied command execution. The failed artefact remains under `local_scratch/external_delegate/20260811-083129-trajectory-batch1-redteam/run/`. A file-only retry exited 0 with a clean tripwire and judged the batch safe to commit. Its suggested single-contact and two-sample trend boundary tests were added. Its nearest-GT tie suggestion was already covered by the explicit equidistant 100/112-frame case.

## 2026-08-11: Batch 2 corrected results

Batch 1 was committed as `360c9b3` with the exact approved message. Batch 2 changes are currently limited to:

- `analyse_serve_trajectory.py`;
- `trajectory_features.py`;
- `test_trajectory_features.py`;
- `validate_outputs.py`.

The corrected analysis writes six ignored outputs:

- `rallies.csv.gz`: 292 GT-rally rows with full GT strokes, accepted frames, mapping, all three alignments, sequence recovery, both path masks, both fixed decisions and server predictions;
- `spans.csv.gz`: 344 half-open predicted spans;
- `path_points.csv.gz`: 1,012 selected-path samples for independent trend and movement reconstruction;
- `fixed_rules.csv.gz`: four fixed rule/mask arms globally and for each fixture, 16 rows;
- `trend_diagnostics.csv.gz`: both masks for 135 primary unique ±10 truth rows, 270 rows;
- `metrics.json.gz`: strict finite JSON containing global and per-fixture populations, alignments, path funnels, sequence outcomes, fixed-rule results and server scores.

The independent validator was implemented by the native `batch2_validator` worker in `validate_outputs.py` only. Primary review added checks for exact per-fixture span counts, accepted contacts inside half-open spans, `n_strokes_list`, and baseline missing/wrong/failure flags. The Batch 2 Opus red team then identified one audit-boundary gap: GT and upstream prediction fields were being trusted after the producer copied them into `rallies.csv.gz`. The validator now reloads each frozen fixture and cross-checks the saved spans, every GT field, semantic first/second truth, boundary mapping, accepted contacts, baseline prediction, raw-candidate summaries, anchor player and direct per-contact player guesses before doing its independent arithmetic. It imports neither `analyse_serve_trajectory` nor `trajectory_features`. It exits 0 with:

```text
validated 292 rallies, 344 spans, 1012 path points, 16 fixed-rule rows and all metrics
```

### Corrected accounting now observed

- All GT: 292, by fixture 113/104/75.
- Covered sensitivity: 249, by fixture 110/84/55.
- Primary one-to-one: 239, by fixture 104/84/51.
- Covered rows use 244 `(fixture, span_id)` keys: 239 one-rally spans and five two-rally spans.
- Primary unique ±10 contact-1/contact-2 truth: 135 anchors, comprising 118 serves and 17 first returns. Three additional nearest contact-1/contact-2 anchors have multiple GT strokes inside ±10 and stay outside rule scoring.
- Primary ±10-unmatched anchors: 97. Later accepted contacts recover the serve in 49. Without a later serve match, 36 recover the first return. Nine first match another later GT stroke. Three have no later GT match. The first GT-matched accepted contact has rank 2 in 56 cases; all ranks and per-fixture counts are in metrics.

### Fixed motion comparison

Global primary unique ±10 truth:

| Mask | Rule | Eligible | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Recurrence only | historical | 18 | 9 | 3 | 8 | 115 | 0.750 | 0.529 | 0.621 |
| Recurrence only | 0.05-BH trend | 19 | 9 | 4 | 8 | 114 | 0.692 | 0.529 | 0.600 |
| Recurrence + producer mask | historical | 9 | 7 | 0 | 10 | 118 | 1.000 | 0.412 | 0.583 |
| Recurrence + producer mask | 0.05-BH trend | 10 | 7 | 0 | 10 | 118 | 1.000 | 0.412 | 0.583 |

These are fixed comparisons, not selected settings. Removing the historical 0.25-BH movement floor admits one extra recurrence-mask robust call, which is an incorrect serve-as-return call in this dataset. The two producer-mask decisions agree despite one additional robust-eligible path.

Among the 19 common-eligible recurrence paths, fitted-decrease medians are similar for GT first returns (0.386 BH) and serves (0.383 BH). Correct calls have median fitted decrease 0.394 BH, residual RMS 0.089 BH and trend-to-jitter 5.75. Incorrect calls have medians 0.013 BH, 0.107 BH and -0.275. These diagnostics describe the fixed calls only and must not become another threshold. Eight common-eligible recurrence calls are incorrect: four false positives and four false negatives. Representative case plots and read-only visual checks remain Batch 3 work.

Three semantic `gt_second_frame` values differ from ordinal 2 in the full ordered GT stroke list: `sset_01` set1/r30, set2/r15 and set2/r26. `gt_first_frame` equals ordinal 1 in all 292 rows. The validator correctly keeps semantic first-return/player truth separate from the full stroke list used for ordinal alignment.

### Batch 2 gates at compaction write-out

- corrected full analysis: exit 0;
- independent corrected validator: exit 0;
- focused trajectory tests: exit 0, 55 passed;
- dedicated investigation Ruff: exit 0;
- Serena diagnostics for analysis, features and tests: none;
- `git diff --check`: exit 0;
- whole pytest: exit 0, 1,456 passed and 29 skipped;
- pinned Pyrefly: first attempt exit 2 from the sandbox uv-cache lock only; exact outside-sandbox rerun exit 0 with no errors and 20 configured suppressions;
- whole Ruff: exit 1 on the unchanged baseline of 661 unrelated findings;
- Serena diagnostics for `validate_outputs.py`: none.

### Batch 2 adversarial check-in

The file-only `claude-opus-4-6-thinking` review completed with exit 0 and a clean repository tripwire. It found no blocking defect in the 239/249/292 population construction, tolerance arithmetic, unmatched-sequence logic, fixed motion rules, inpaint control or semantic server scoring. Its one medium finding was the frozen-source boundary described above, which is now closed. A low note observed that binary precision is defined as 1.0 when a rule makes no positive calls. Every current global and per-fixture arm makes at least one positive call, so the convention does not affect these results and was left unchanged. The review artefact is:

`local_scratch/external_delegate/20260811-090357-trajectory-batch2-redteam/run/result.md`
