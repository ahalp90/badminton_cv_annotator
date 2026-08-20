# Retained results

These files are compact, deterministic evidence. The scripts verify pinned inputs before writing them.

| File | What it contains | Producer |
| --- | --- | --- |
| `scene_aware_ransac.json.gz` | Baseline proof, per-fixture scene comparison, contact margins and stress-span outcomes | `scene_aware_ransac.py` |
| `scene_aware_change.png` | Human-readable view of the scene comparison | `plot_scene_result.py` |
| `serve_qualification.json.gz` | Label-blind qualification of every fast burst | `freeze_serve_qualification.py` |
| `serve_protection_all_qualified.json.gz` | Mask score for protecting every qualified burst | `score_serve_protection.py` |
| `serve_protection_e2e_all_qualified.json.gz` | Failed full replay of the broad rule, including the wrong and unscored landing changes | `run_serve_protection_e2e.py` |
| `serve_protection_first_per_region.json.gz` | Mask score for the pipeline-matching serve rule | `score_serve_protection.py` |
| `serve_protection_e2e_first_per_region.json.gz` | Full replay passed but changed no output for the pipeline-matching rule | `run_serve_protection_e2e.py` |
| `rally_ender_counterfactual.json.gz` | Mask score for the ground-truth-only diagnostic test | `score_rally_ender_counterfactual.py` |
| `rally_ender_e2e.json.gz` | Diagnostic-only full replay of the rally-ending rule | `run_serve_protection_e2e.py` |

[`../EVIDENCE.md`](../EVIDENCE.md) lists the exact commands for the retained scene, pipeline-matching serve and rally-ending tests. To rerun the failed broad rule, replace `first_qualified_per_region` with `all_qualified_bursts` and use the matching `all_qualified` output filename.

Inspect a compressed result without extracting it:

```bash
gzip -dc scratch/ransac_scene_guard/results/scene_aware_ransac.json.gz | jq .
```

Repeated accepted outputs were byte-identical. Temporary repeat files and logs are not retained.
