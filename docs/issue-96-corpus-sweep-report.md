# Issue 96 corpus sweep report

The FPS-normalised annotator sweep ran against the fixed Issue 103 replay
inputs. All 40 pinned artifact indexes were validated before the three
ground-truth fixtures were evaluated: `sset_01`, `sset_15`, and `sset_21`.

The boundary search covered 3,000 candidates. The contact search covered 108
candidates. Attempts were resumable and retained their immutable result or
failure evidence. The run completed with exit code 0.

The boundary winner improved aggregate strict F1 from `0.814685` to
`0.827094` (`+0.012409`), but failed downstream guardrails:

- getpoint: `-5` correct predictions
- landing: `-1`
- server: `-3`

The contact winner reduced aggregate raw-F1 from `0.651156` to `0.649296`
(`-0.001860`) and failed all five downstream guardrails.

Therefore no candidate is eligible for `best_config`. The shipped baseline
configuration remains selected. No production configuration change or Issue
103 replay is required.

The complete run evidence is retained on Engelbart at
`/scratch/cmarti/issue96_399d56f/evidence-v2`.
