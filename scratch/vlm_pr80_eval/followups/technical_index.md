# Technical index for the PR 80 follow-ups

This file is for verification and reproduction. It is deliberately separate from the readable result reports.

## Contents

- [Source boundary](#source-boundary)
- [Follow-up 1: scene comparison](#follow-up-1-scene-comparison)
- [Follow-up 2: clean rally-start gate](#follow-up-2-clean-rally-start-gate)
- [Follow-up 3: precision-first filtering](#follow-up-3-precision-first-filtering)
- [Follow-up 4: compact serve support](#follow-up-4-compact-serve-support)
- [Follow-up 5: PR 88 reconciliation](#follow-up-5-pr-88-reconciliation)
- [Follow-up 6: rally-opening context](#follow-up-6-rally-opening-context)
- [Reproduction commands](#reproduction-commands)
- [Claim boundaries to preserve](#claim-boundaries-to-preserve)

## Source boundary

The document pack was prepared against branch `vlm-pr80-followups` at commit:

`c2138bb697980b7a426264ba63f723049e174c04`

The six compact `.json.gz` decision records sit beside the larger manifests,
row-level scores, raw replies and portable inputs under
[`evidence/`](evidence/).

Human truth remained separate from inference inputs wherever the original experiment did so. The result reports do not change any prompt, score, threshold, model reply, experiment artefact, or scientific decision.

## Follow-up 1: scene comparison

**Question:** How do Qwen and Intern route the same 463 short scene targets?

- readable report: [`1_scene_comparison.md`](1_scene_comparison.md);
- machine summary: [`1_scene_comparison.json.gz`](evidence/1_scene_comparison.json.gz);
- exact input manifest: [`evidence/1_scene_manifest.json.gz`](evidence/1_scene_manifest.json.gz);
- complete Qwen row-level score: [`evidence/1_scene_qwen_score.json.gz`](evidence/1_scene_qwen_score.json.gz);
- frozen Intern comparison rows: [`evidence/1_scene_intern_reference_score.json.gz`](evidence/1_scene_intern_reference_score.json.gz);
- all Qwen raw replies and run status: [`evidence/1_scene_qwen_remote_runs.tar.gz`](evidence/1_scene_qwen_remote_runs.tar.gz);
- parent scene tools: [`../experiments/`](../experiments/).

The headline uses 347 material targets after applying the same short-boundary exclusion to both models.

## Follow-up 2: clean rally-start gate

**Question:** Which model is the stronger starting point for later experiments, and which serve fields are usable?

- readable report: [`2_final_model_gate.md`](2_final_model_gate.md);
- machine summary: [`2_final_model_gate.json.gz`](evidence/2_final_model_gate.json.gz);
- truth-free 32-case manifest: [`evidence/2_rally_start_manifest.json.gz`](evidence/2_rally_start_manifest.json.gz);
- separate scoring truth: [`evidence/2_rally_start_truth.json.gz`](evidence/2_rally_start_truth.json.gz);
- clip provenance and hashes: [`evidence/2_rally_start_provenance.json.gz`](evidence/2_rally_start_provenance.json.gz);
- paired row-level score: [`evidence/2_rally_start_score.json.gz`](evidence/2_rally_start_score.json.gz);
- all 64 raw replies and run records: [`evidence/2_rally_start_remote_runs.tar.gz`](evidence/2_rally_start_remote_runs.tar.gz);
- builder, parser, runner, and scorer: [`../experiments/rally_start_trials.py`](../experiments/rally_start_trials.py);
- remote run boundary: [`../experiments/rally_start_remote.sh`](../experiments/rally_start_remote.sh).

The clean model choice and the usability of individual fields are separate decisions.

## Follow-up 3: precision-first filtering

**Question:** Can frozen automatic signals identify a non-empty zero-observed-error record set on held-out fixtures?

- readable report: [`3_precision_first_dataset.md`](3_precision_first_dataset.md);
- machine summary: [`3_precision_first_dataset.json.gz`](evidence/3_precision_first_dataset.json.gz);
- truth-free 311-row feature table: [`evidence/3_precision_first_features.json.gz`](evidence/3_precision_first_features.json.gz);
- portable frozen annotation, court, and shuttle inputs: [`evidence/3_precision_first_inputs.tar.gz`](evidence/3_precision_first_inputs.tar.gz);
- feature builder, fixed rule ladder, held-out selector, and scorer: [`../experiments/precision_first_trials.py`](../experiments/precision_first_trials.py).

The feature table was written before the scorer opened human labels. The primary rule selection used ±5 base-30 frames; ±10 and ±15 are sensitivity checks.

## Follow-up 4: compact serve support

**Question:** Do plain automatic observations or explicit fallible proposals improve Intern's serve answers?

- readable report: [`4_serve_reconstruction.md`](4_serve_reconstruction.md);
- machine summary: [`4_serve_reconstruction.json.gz`](evidence/4_serve_reconstruction.json.gz);
- exact truth-free support sentences and hashes: [`evidence/4_serve_support.json.gz`](evidence/4_serve_support.json.gz);
- both row-level Intern scores: [`evidence/4_serve_intern_scores.json.gz`](evidence/4_serve_intern_scores.json.gz);
- all 64 raw enhanced replies and run records: [`evidence/4_serve_intern_remote_runs.tar.gz`](evidence/4_serve_intern_remote_runs.tar.gz);
- support builder and scorer: [`../experiments/rally_start_support_trials.py`](../experiments/rally_start_support_trials.py);
- remote run boundary: [`../experiments/rally_start_support_remote.sh`](../experiments/rally_start_support_remote.sh).

The observation arm's aggregate timing agreement is only interpretable alongside the row-level result that 30 of 31 parsed answers repeated the supplied inspection point.

## Follow-up 5: PR 88 reconciliation

**Question:** Does the frozen deterministic PR 88 result justify one simple VLM hybrid?

- readable report: [`5_pr88_serve_lookback.md`](5_pr88_serve_lookback.md);
- machine summary: [`5_pr88_serve_lookback.json.gz`](evidence/5_pr88_serve_lookback.json.gz);
- full deterministic report: [`../../serve_id_by_lookback_followup/report.md`](../../serve_id_by_lookback_followup/report.md);
- frozen decision branches: [`../../serve_id_by_lookback_followup/serve_id_followup/rules.py`](../../serve_id_by_lookback_followup/serve_id_followup/rules.py);
- recomputation tool: [`../../serve_id_by_lookback_followup/serve_id_followup/recompute.py`](../../serve_id_by_lookback_followup/serve_id_followup/recompute.py);
- checked development metrics: [`../../serve_id_by_lookback_followup/results/development_metrics.json.gz`](../../serve_id_by_lookback_followup/results/development_metrics.json.gz);
- all 239 scored decisions: [`../../serve_id_by_lookback_followup/results/preferred_server_rule.csv.gz`](../../serve_id_by_lookback_followup/results/preferred_server_rule.csv.gz).

No new VLM inference ran. The 14-case overlap is a selected retrospective diagnostic, not a representative hybrid benchmark.

## Follow-up 6: rally-opening context

**Question:** Does a continuous opening, a timing cue, or native frame density improve Intern's server answer?

- readable report: [`6_rally_opening_context.md`](6_rally_opening_context.md);
- machine summary: [`6_rally_opening_context.json.gz`](evidence/6_rally_opening_context.json.gz);
- truth-free 311-span join: [`evidence/6_rally_opening_window_manifest.json.gz`](evidence/6_rally_opening_window_manifest.json.gz);
- separate labelled crosswalk: [`evidence/6_rally_opening_window_truth.json.gz`](evidence/6_rally_opening_window_truth.json.gz);
- 36 frozen model inputs and prompts: [`evidence/6_rally_opening_trial_manifest.json.gz`](evidence/6_rally_opening_trial_manifest.json.gz);
- separate 12-case scoring truth: [`evidence/6_rally_opening_trial_truth.json.gz`](evidence/6_rally_opening_trial_truth.json.gz);
- totals, paired changes, and row-level score: [`evidence/6_rally_opening_score.json.gz`](evidence/6_rally_opening_score.json.gz);
- all 36 raw replies: [`evidence/6_rally_opening_intern_remote_runs.tar.gz`](evidence/6_rally_opening_intern_remote_runs.tar.gz);
- join builder: [`../experiments/rally_opening_window_join.py`](../experiments/rally_opening_window_join.py);
- trial builder and scorer: [`../experiments/rally_opening_trials.py`](../experiments/rally_opening_trials.py).

The join is a reusable infrastructure result. The 12-case model score is a small diagnostic.

## Reproduction commands

### Precision-first score

From the repository root:

```bash
run_dir=$(mktemp -d) &&
tar -xzf scratch/vlm_pr80_eval/followups/evidence/3_precision_first_inputs.tar.gz \
  -C "$run_dir" &&
gzip -cd scratch/vlm_pr80_eval/followups/evidence/3_precision_first_features.json.gz \
  > "$run_dir/features.json" &&
PYTHONPATH=src:scratch/vlm_pr80_eval \
  python -m experiments.precision_first_trials score \
  --features "$run_dir/features.json" \
  --artifacts-root "$run_dir" \
  --output "$run_dir/score.json"
```

### PR 88 checked result

From `scratch/serve_id_by_lookback_followup/`:

```bash
python3 -m serve_id_followup.recompute --check
```

### Follow-up visuals

From the repository root:

```bash
python scratch/vlm_pr80_eval/experiments/make_followup_visuals.py
```

The visual script reads the six retained compact summaries and checks their schemas before writing the figures.

## Claim boundaries to preserve

- Intern is the chosen relative model, not a validated automatic judge.
- Qwen's overall scene score is driven by ordinary live retention and does not make it the safer cleanup model.
- Follow-up 3 retained zero records because no development rule qualified; precision is undefined.
- Follow-up 4's timing gain mostly reflects copied inspection points.
- PR 88's 170/239 is reproducible development evidence, not unseen validation.
- Follow-up 6 retains useful infrastructure but no improved model route.
- A planned run that failed its gate is not missing evidence; it was deliberately not run.
