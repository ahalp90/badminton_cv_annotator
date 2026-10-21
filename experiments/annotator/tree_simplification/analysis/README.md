# Tree simplification analysis scripts

These scripts rebuild the evidence files and figures in [`../results/`](../results/)
and check them against the committed copies. Most read tracked files only.
Four need data that is not in the repository; the table says which.

Run every script from the repository root with `PYTHONPATH=src:.` and the
project's Python. Each takes its paths as arguments, with tracked defaults.
Output files go wherever `--output` says, so a check never overwrites a
tracked file.

## The checking rule

Every checker uses one rule. Integer counts must be equal. Floats may differ
by at most 1e-12, which only allows the same arithmetic in a different order.
A checker stops at the first file that differs and prints up to 20
differences.

Some fields record how a run was made rather than a result: source records,
evaluation hashes, scrubbed paths and free-text notes. The checkers copy these
fields instead of rebuilding them. The last column lists them by key name.

## Scripts

| Script | What it rebuilds or checks | Reads | Needs data outside the repo | Copied fields |
|---|---|---|---|---|
| `plot_review_tradeoff.py` | `input-review-tradeoff.svg` | `input-preprocessing-evidence.json.gz`, key `test_review_evaluation` | no | none |
| `input_selection.py rank` | the sweep's `screen`, `shortlist` and `matched_validation`, and the recorded hash of each ranking | `streams/input-validation/` re-scored against the labels, and the configuration list in the evidence | no | none |
| `input_selection.py choose` | the two configurations chosen for the test, their objectives and the md5 of each bundle in `test-run/selection.json.gz` | the `rank` output, `test-run/`, `models/` | no | none |
| `input_selection.py rows` | `input-validation-review-rows.json.gz` and `input-test-review-rows.json.gz` | per-run review rows | yes: the local archive | none |
| `input_evidence.py` | `validation_results`, `test_results`, `test_analysis`, `test_selection`, `review_thresholds`, `test_review_evaluation` and `validation_costs` in `input-preprocessing-evidence.json.gz` | tracked streams and `scores/`, `test-run/`, both row files, `input-review-thresholds.json.gz`, `input-stage-costs.csv` | no | `source`; in `validation_costs`, the counts of run records and the split of times between the screen and the refinement |
| `validation_evidence.py selectors` | the per-seed results, means and paired differences in `five-seed-evidence.json.gz` and `selector-settings-evidence.json.gz`, and the chosen bundle's replay in `selected-production.json.gz` | `streams/selection/` re-scored against the labels | no | `source_files`, `source_validation`, `reported_totals_verified`, `result_summary_verified`, `complete`, `same_pool_and_target_contract`, `tree`, `description`, `sources` |
| `validation_evidence.py frame-rate` | `fps-normalisation-evidence.json.gz` | `streams/fps/` re-scored against the labels, and the raw seeds in `selector-settings-evidence.json.gz` | no | `baseline_source`, `interpretation`, `verification`, `sources` |
| `validation_evidence.py ranking` | the six review-ranking groups (three rankings at ±10 and ±5) in `selected-production.json.gz` | `selected-review-rows.json.gz` | no | none |
| `test_run.py materialise` | a working copy of the ten test runs in a new folder, with each bundle in place | `test-run/`, `models/` | no | none |
| `test_run.py check` | loads all ten test bundles through `input_test`'s own checks | a folder made by `materialise` | no | none |
| `flaw_rescore.py` | `test-flaw-scoring.json.gz`: the old and new annotators' test predictions scored with and without predictions inside the 524 left-out rallies | `streams/input-test/`, the old annotator's predictions in `../../tree_hp_tune/results/test_rerun/A/v32/grid-017/`, the MonoTrack benchmark manifest, `data/shuttleset22/set/` | no | none |
| `screening_evidence.py` | `screening-evidence.json.gz`: the search arms the report describes but the repo does not hold | archived run records | yes: the local archive | none |
| `plot_target_screen.py` | `target-screen.svg` | `screening-evidence.json.gz` | no | none |
| `profile_selected.py` | an inference profile like `selected-profile.json.gz` | `models/selected/` and one video's prepared arrays | yes: prepared arrays, which `prepare` rebuilds from the video fixtures | not a checker |
| `pool_coverage.py` | the option-list probe behind the ceiling in `screening-evidence.json.gz` | an option-list folder written by `pools`, and the labels | yes: option lists from a fresh run | not a checker |

"The labels" means `training/data/shuttleset/annotations/shots_master.csv` and
the tracked 32/8 split.

## Notes on particular scripts

**`flaw_rescore.py`** takes about four minutes on eight cores. Standard
scoring must give the tracked per-video test scores of both annotators, and
the MonoTrack write-up's figures for the old one. It reports fully correct
rallies under two rules. `fully_correct_rallies` is this experiment's rule,
used by every other file here. `fully_correct_rallies_edge_allowance` is the
write-up's rule, which also accepts a found rally whose edges fall up to 10
frames short of the labelled hits. The old annotator's 51.48% uses the
write-up's rule. Under this experiment's rule it gets 51.02%.

**`screening_evidence.py` and `input_selection.py rows`** take `--archive`,
the unpacked `raw-data.tar.zst` from the local campaign archive. It keeps the
run folder's layout: `remote/` holds the run records, and `fps-k5/`,
`fps-motion-v1/` and `fps-velocity-v1/` hold the frame-rate runs. Folder names
inside it are the names the runs were written with. Each script maps them to
plain labels in one table near its top.

**`test_run.py materialise --use-current-checkout`** writes this checkout's
source record into the copy. `input_test` then accepts a fresh test run at a
later commit. It needs a clean checkout.

## Example

```bash
export PYTHONPATH=src:.
python=~/.venvs/badminton-cicd/bin/python
out=$(mktemp -d)
analysis=experiments/annotator/tree_simplification/analysis
"$python" "$analysis/validation_evidence.py" selectors
"$python" "$analysis/input_selection.py" rank --output "$out/ranking.json.gz"
"$python" "$analysis/input_selection.py" choose --ranking "$out/ranking.json.gz"
"$python" "$analysis/test_run.py" materialise --output "$out/test-run"
"$python" "$analysis/test_run.py" check --output "$out/test-run"
"$python" "$analysis/plot_target_screen.py" --output "$out/target-screen.svg"
```
