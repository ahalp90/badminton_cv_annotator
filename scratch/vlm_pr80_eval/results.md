# Bounded follow-up results

## Main result

Changing the task fixed the spectacular output collapse. It did not make either
model a safe general cleanup filter.

See [`experiments.md`](experiments.md) for the order of the trials and
[`prompts.md`](prompts.md) for the requests used.

| Test | Qwen3-VL | InternVideo3 | Meaning |
|---|---:|---:|---|
| Contact timing at ±15 base-30 frames | 88.6% precision, 97.5% recall | 83.3% precision, 50.0% recall | Structural timing only; not proof of visible contact |
| Marked tracker hallucinations rejected | 7/12 development | 11/12 development, 5/6 safety | InternVideo3 is the useful narrow lead |
| Marked tracker orientation controls accepted | 12/12 | 12/12 development, 6/6 safety | These are proxies near contacts, not labelled real tracks |
| Direct broadcast controls kept as live | 11/12 | 9/12 replies; 3 invalid | Both missed too much replay or cutaway content |

Qwen3-VL removed 14 of 84 contacts in the complete-rally replay. The number of
rallies with the exact contact count stayed at 4 of 12. Only 1 of 12 was
structurally usable at ±10 frames before and after filtering. Four replies were
invalid, so this run also failed its completeness gate.

InternVideo3's clean-pixel tracker counterfactual rejected all 18 known
hallucinations. It also rejected 11 of 18 orientation controls. The apparent
gain was therefore a broad reject bias.

The broadcast-sequence prompt let InternVideo3 name replay content, but only
two of six live controls survived. Qwen3-VL's result was essentially unchanged.

## Limits

- The 60 contact cases use ShuttleSet timing as structural truth. Some serves
  are off-screen or inferred across a cut.
- The tracker audit labels only hallucinations. Its positive controls are not
  human-labelled tracker paths.
- The complete-rally replay covers 12 selected rallies from two videos. It is
  a bounded decision test, not a full-fixture estimate.
- Human frame labels can be noisy. The useful timing target is about ±10
  base-30 frames, with ±5 ideal and ±15 still worth reporting.
- The raw candidate pool can make only 56 rallies complete at ±10 frames, even
  with an oracle choosing candidates. A cleanup model cannot recover contacts
  that are absent from that pool.

`experiments/results/summary.json` stores the same headline counts in a small
machine-readable form. It deliberately excludes host paths, raw clips, model
caches, and private run records.
