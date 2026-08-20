# Evidence and reproduction notes

## Decision

PR 98's RANSAC candidates are useful for review or ranking. The tested guards do not support automatic removal of shuttle positions.

The evidence comes from three pinned stride-8 fixtures. The comparison starts at PR 98 commit `3c217d4cc8ab3698e825315a218735322f938a00`. No `src/` file changed.

## Source context

- [PR 98](https://github.com/ahalp90/badminton_cv_annotator/pull/98) records the RANSAC investigation and open safety question
- [PR 93](https://github.com/ahalp90/badminton_cv_annotator/pull/93) supplies the recurrence-v4 and radius-3 impulse-veto baseline
- [PR 88](https://github.com/ahalp90/badminton_cv_annotator/pull/88) and [PR 97](https://github.com/ahalp90/badminton_cv_annotator/pull/97) explain the serve-start evidence and difficult broadcast views
- [Issue 31](https://github.com/ahalp90/badminton_cv_annotator/issues/31) contains the 18 positive-only hallucination spans

## What was held fixed

The tested baseline is the recurrence-v4-clean RANSAC mask after the existing radius-3 impulse veto. It contains 11,660 selected frames.

The full replays use fixtures generated at commit `189c5af58e45d23ae827dde516924194eb238e18`. Each retained result records the exact input hashes it checked.

All experiments use:

- `sset_01`, `sset_15` and `sset_21`
- unchanged tracks, pose, court, scene and ShuttleSet inputs
- contact margins of ±5, ±10 and ±15 frames after normalising to 30 fps
- 18 reviewed hallucination spans as a positive-only stress set
- a full replay with upstream inputs fixed when the mask result made it worthwhile

Seven of the 18 reviewed spans remain after the existing impulse veto. Every tested rule retained those seven. This checks difficult known hallucinations only. It cannot estimate precision.

The full replay keeps rally spans, raw and filtered contacts, per-rally contacts and exclusion masks identical between arms. A rule fails if it adds a rejection or changes an output that is wrong, misattributed or unscored.

## 1. Scene-aware fitting

### Reason for the test

The RANSAC fitter models 16 frames of shuttle motion. A broadcast cut breaks pixel continuity, so one quadratic fit should not cross it.

The original fitter schedules a window every four frames. It tests 32 deterministic three-frame fits, requires at least eight inliers inside a three-pixel residual, then refits the best model. A frame is selected when at least half of its eligible windows vote that it is an outlier.

The experiment changed one rule:

```text
Do not fit a window when start < scene cut < stop.
```

A cut at the first frame or exclusive end of a window does not split that window. Removing a window can add candidates because it changes both the outlier votes and the eligible-vote count.

The raw cuts come from a shot-boundary detector. They are useful boundaries, but they are not proof that every camera change was found or classified correctly.

### Result

Across 404,229 frames, 6,059 of 101,047 scheduled windows crossed a cut. Of those, 3,052 had previously contributed votes.

| Mask | Frames before → after | Exact contacts before → after | Contacts within ±10 before → after |
| --- | ---: | ---: | ---: |
| All RANSAC | 107,251 → 104,437 | 1,742 → 1,733 | 2,938 → 2,920 |
| Recurrence-v4 clean | 39,480 → 38,357 | 1,656 → 1,649 | 2,818 → 2,799 |
| Clean plus impulse veto | 11,660 → 11,330 | 239 → 242 | 1,249 → 1,243 |

The final tested mask lost 452 frames and gained 122. Exact final-contact conflicts rose from 22 to 23. The ideal ±5 measure rose from 937 to 940.

The scene-aware comparison failed the predeclared follow-up screen, so no scene-only full replay was run. The retained plot shows why the mask reduction was not a contact-safety gain:

![Scene-boundary result](results/scene_aware_change.png)

## 2. Serve-aware protection

### Reason for the test

Serve contact is easy to damage because broadcasts can show a close-up or cut around the actual contact. The test protects a short region around sustained fast shuttle runs. The pose evidence before each run must pass the existing sticky serve-setup gate.

Qualification used the exploratory wrist-distance threshold `threshold_bh=0.8`, measured in body-height units. It used trim mode, the shipped fps-scaled lookback, no close split and no stillness check. Qualification ran before the scoring stage and could not read ShuttleSet contact labels or the 18 stress spans.

The frozen qualification output repeated byte for byte. The public copy, with local paths removed, has SHA-256 `3957894d367d2090f481ca98d62e59a62637cb403819be56b1775af860ca8914`.

### Broad rule

The first arm protected ±10 base-30 frames around every qualified burst.

| Measure | Baseline → protected |
| --- | ---: |
| Rejection frames | 11,660 → 10,109 |
| Contacts with a rejection within ±10 | 1,249 → 840 |
| First contacts with a rejection within ±10 | 66 → 56 |
| Exact first-contact conflicts | 5 → 3 |

The full replay restored three correct landings. It also created one wrong `sset_21` landing. A second unscored `sset_21` landing moved by four frames and changed position. The rule failed.

### Pipeline-matching rule

With `close=None`, the pipeline uses only the first qualified burst in each active region. An active region is one continuous stretch between long rests. A long rest lasts at least `end_rest_frames` for that FPS. [`_rally_regions`](../../src/annotator/rally/spans.py) defines the boundary. A second test applied that rule to the same frozen qualification.

The rule reduced rejection frames from 11,660 to 11,563. Contacts with a rejection within ±10 fell from 1,249 to 1,221. It rescued three first contacts at ±10 and one exact first-contact conflict.

The full replay changed no final contact, landing, winner or scored rally row. It passed the safety screen on these three fixtures, but produced no useful change in later outputs. Two separate replay outputs were byte-identical with SHA-256 `d6d54036094b75550f0ddddecf03d67a842df5203002acb563284549a3132028`.

## 3. GT-qualified rally-ending counterfactual

This diagnostic test protected ±3 frames around 43 tracked rally-ending events selected with ground truth. It tests whether even a ground-truth-based signal can identify RANSAC rejections worth removing.

The rule cleared 6 of 11,660 rejection frames. It changed none of the exact, ±5, ±10 or ±15 contact-risk measures. The full replay cleared two recorded event-mask rejection rows and changed no final output.

The replay reports `diagnostic_only: true` and `deployable_evidence: false`. Two separate outputs were byte-identical with SHA-256 `2fea0be8d2afb0c4c2681964cd3f61ac27327fd447fee69d7fbe0de249e6822c`.

## Why the investigation stops here

The scene rule is cleaner but does not improve contact safety. The broad serve rule has a useful mask effect but creates a wrong output. The pipeline-matching serve rule and the ground-truth rally-ending rule do not change the final outputs.

Another hand-built veto would add complexity without a new source of evidence. The planned contact model and VLM annotations in progress can provide that independent evidence. A future rule can reuse these fixed-upstream checks and timing margins.

## Reproduce the results

Run the scene comparison and plot locally from the repository root:

```bash
~/.venvs/badminton-cicd/bin/python \
  scratch/ransac_scene_guard/scene_aware_ransac.py

~/.venvs/badminton-cicd/bin/python \
  scratch/ransac_scene_guard/plot_scene_result.py
```

Run label-blind serve qualification on a machine with the pinned PR 98 fixtures:

```bash
FIXTURES_ROOT=/path/to/pr98-source-fixtures

~/.venvs/venv-scraper/bin/python \
  scratch/ransac_scene_guard/freeze_serve_qualification.py \
  --fixtures-root "$FIXTURES_ROOT"
```

Replace `/path/to/pr98-source-fixtures` with the local path to the pinned PR 98 fixtures.

The fixture root is external data. It is not stored in this repository.

Score the pipeline-matching serve rule locally:

```bash
~/.venvs/badminton-cicd/bin/python \
  scratch/ransac_scene_guard/score_serve_protection.py \
  --burst-policy first_qualified_per_region \
  --output scratch/ransac_scene_guard/results/serve_protection_first_per_region.json.gz
```

Run its full replay on the machine that holds the fixtures:

```bash
FIXTURES_ROOT=/path/to/pr98-source-fixtures

ANNOTATOR_FIXTURES_ROOT="$FIXTURES_ROOT" \
~/.venvs/venv-scraper/bin/python \
  scratch/ransac_scene_guard/run_serve_protection_e2e.py \
  --burst-policy first_qualified_per_region \
  --output scratch/ransac_scene_guard/results/serve_protection_e2e_first_per_region.json.gz
```

Run the diagnostic rally-ending mask comparison locally:

```bash
~/.venvs/badminton-cicd/bin/python \
  scratch/ransac_scene_guard/score_rally_ender_counterfactual.py
```

Run its diagnostic full replay on the machine that holds the fixtures:

```bash
FIXTURES_ROOT=/path/to/pr98-source-fixtures

ANNOTATOR_FIXTURES_ROOT="$FIXTURES_ROOT" \
~/.venvs/venv-scraper/bin/python \
  scratch/ransac_scene_guard/run_serve_protection_e2e.py \
  --protection-source gt_qualified_rally_enders \
  --output scratch/ransac_scene_guard/results/rally_ender_e2e.json.gz
```

The scripts verify their pinned inputs and stop on a mismatch. [`results/README.md`](results/README.md) maps the retained outputs to these commands.
