# Auto-annotator guide

The auto-annotator combines court geometry, shuttle tracks and player poses to
find rallies, identify hits and assign them to the near or far player. It also
ranks rallies for human review. Complete rallies remain less reliable than
individual hits, so the output needs checking before it becomes research data.

The [experiments and handover guide](../../experiments/annotator/README.md) explains
the months of development, the approaches that worked, and the remaining
problems. These pages cover using and maintaining the resulting implementation.

| Task | Guide |
|---|---|
| Run annotation and find the selected model | [Quickstart](quickstart.md) |
| Understand how the stages fit together | [How it works](how_it_works.md) |
| Read or produce saved data | [Inputs and outputs](inputs_outputs.md) |
| Understand fixed rules and learned models | [Heuristics](heuristics.md) and [model stack](tree_stack.md) |
| Fit and evaluate new models | [Retuning](retuning.md) |
| Find code or assess the effect of a change | [Code map](code_map.md) and [maintenance](maintaining.md) |

The [evaluation guide](../../experiments/annotator/evaluation.md) covers measured performance.
It distinguishes the selected model from historical results and explains which
comparisons share the same data and scoring rules.
