# Annotator boundary-policy refit

The default annotator uses the **2 seconds before / 3 seconds after** training
policy, selected on the eight validation videos. All three trees were refitted
on the original 32 training videos with the existing hyperparameters and five
seed offsets. Hyperparameter retuning after these changes remains a TODO.

Selection maximised mean fully correct labelled-rally coverage before confidence
filtering, using the same 2/3-second evaluation rule for all three training
policies. Ties prefer 2/3, then 3/3, then 5/5. Test results did not choose the
training policy.

| Training allowance, before / after | Fully correct validation rallies, mean of five seeds | Share of 668 labelled rallies |
| --- | ---: | ---: |
| 2 / 3 seconds | 216.8 | 32.46% |
| 3 / 3 seconds | 208.6 | 31.23% |
| 5 / 5 seconds | 209.8 | 31.41% |

The allowances are maximum context, so closer valid boundaries can pass.
Discovery output retains at most two seconds before its first kept contact and
three seconds after its last, with the existing small padding and neighbour
checks. Supplied rally bounds stay fixed.

- [`selection.json.gz`](selection.json.gz) contains the selection rule and counts
  for every seed and training policy.
- [`validation-cutoffs.json.gz`](validation-cutoffs.json.gz) records the three
  confidence operating points selected from all 3,346 distinct validation scores.
  The 99% boundary target retains only 7–12 validation rallies per seed.
- [`2_3/models/`](2_3/models/) contains the selected policy's five raw-baseline
  bundles and five body-window-36 comparison bundles. The default is raw baseline,
  seed offset 0. Each bundle contains the contact, selector and confidence trees.
- [`2_3/fit-settings.json.gz`](2_3/fit-settings.json.gz) records the actual fit
  settings, training/validation videos, contact sampling orders and source revision.
  Each tree uses two CPU cores; exported models predict on the CPU.

The body-window-36 setting is a fixed comparison from the earlier validation
search. This refit does not repeat the hyperparameter or preprocessing search.

Use the [benchmark retraining and evaluation commands](../../../monotrack_benchmark/writeup_scripts/README.md)
to reproduce the three-policy comparison. The earlier bundles and experimental
results remain under the parent results directory for their historical reports.
