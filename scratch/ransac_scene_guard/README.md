# RANSAC hallucination-guard experiments

## Bottom line

Keep [PR 98's RANSAC output](https://github.com/ahalp90/badminton_cv_annotator/pull/98) as a review or ranking signal. None of the small guards tested here improved contact safety while also producing a useful change in final rally outputs.

The broad serve rule reduced the rejection mask the most. Its full replay created one wrong landing, so it is not safe to use. The narrower rule that matches the current pipeline changed no final output.

No production source file was changed. Everything in this folder is an experiment, a focused test, or retained evidence.

## Results at a glance

| Experiment | RANSAC rejection frames | Contacts with a rejection inside ±10 | Full replay |
| --- | ---: | ---: | --- |
| Keep each RANSAC fit inside one scene | 11,660 → 11,330 | 1,249 → 1,243 | Not run because the predeclared mask screen failed |
| Protect every qualified serve-like burst | 11,660 → 10,109 | 1,249 → 840 | Three correct landings restored, but one wrong landing was added and one unscored landing shifted |
| Protect the first qualified burst between long rests | 11,660 → 11,563 | 1,249 → 1,221 | No contact, landing or winner changed |
| Protect 43 GT-qualified rally-ending events by ±3 frames | 11,660 → 11,654 | No change | Diagnostic only; it cleared two rejection rows and changed no output |

The ±10 measure uses frames normalised to 30 fps. It counts labelled contacts with at least one RANSAC rejection candidate nearby. It is a warning measure, not proof that a tracked shuttle point is right or wrong.

A full replay runs the pipeline twice with the same inputs and intermediate contact data. Only the RANSAC rejection mask changes. This isolates any changed contact, landing or winner output.

## Evaluation

Keeping each fit within one scene is a sound model constraint. A quadratic motion fit should not cross a broadcast cut. On these fixtures, however, it reduced the rejection mask by 2.8% and the ±10 contact-risk measure by only 0.5%. Exact contact conflicts rose from 239 to 242.

The broad serve rule protected every qualified burst. A qualified burst is a sustained fast shuttle run preceded by pose evidence that passes the exploratory `threshold_bh=0.8` serve gate. The rule restored four landing predictions. Three matched the labelled court half, while one `sset_21` prediction landed on the wrong half. Another unscored landing moved by four frames and changed position. That fails the safety rule.

The first-burst rule matches how `close=None` uses the serve gate in the current pipeline. It protects only the first qualified burst between pipeline-defined long rests. The fixed-upstream replay passed its safety screen on these three fixtures. It cleared only 97 rejection frames and changed no later output, so this change is not worth adding to production.

The rally-ending test used ground truth to choose 43 events. It was a diagnostic upper bound, not a production rule. It cleared six mask frames and changed no output.

## Where to go next

Avoid adding another hand-built veto on this evidence. The next useful test needs an independent contact signal, such as the planned binary contact model or the VLM annotations already in progress.

Test that signal with the same fixed-upstream replay. A useful rule may miss some rallies, but it must not create or misattribute contacts. Keep the ±5, ±10 and ±15 base-30 timing checks because ShuttleSet contact labels include timing noise and some inferred contacts.

If the RANSAC estimator is revisited, keep model windows within scene boundaries. Treat that as a cleaner model assumption, not as evidence that automatic rejection is safe.

## Read or reproduce the work

- [`EVIDENCE.md`](EVIDENCE.md) explains the methods, rationale, limits and detailed results
- [`results/README.md`](results/README.md) maps each retained result to its producing script
- [`scene_aware_ransac.py`](scene_aware_ransac.py) reruns the scene-boundary comparison
- [`freeze_serve_qualification.py`](freeze_serve_qualification.py), [`score_serve_protection.py`](score_serve_protection.py) and [`run_serve_protection_e2e.py`](run_serve_protection_e2e.py) reproduce the serve experiments
- [`score_rally_ender_counterfactual.py`](score_rally_ender_counterfactual.py) reproduces the diagnostic rally-ending mask comparison

Run the focused checks from the repository root:

```bash
~/.venvs/badminton-cicd/bin/ruff check \
  scratch/ransac_scene_guard/*.py \
  scratch/ransac_scene_guard/tests

~/.venvs/badminton-cicd/bin/pytest \
  scratch/ransac_scene_guard/tests
```

The three pinned fixtures are `sset_01`, `sset_15` and `sset_21`. The 18 reviewed hallucination spans contain known positives only. They test whether difficult examples survive; they do not measure general precision.
