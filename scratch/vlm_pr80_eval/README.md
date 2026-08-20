# PR 80 VLM evaluation

PR 80 does not show that Qwen3-VL or InternVideo3 cannot inspect badminton
video. It shows that the benchmark asked both models to solve the wrong task in
an awkward format. The prompt asked for a complete broadcast timeline instead
of checking one proposed annotation. Its only example also became the repeated
answer.

Shorter follow-up trials removed that catastrophic repetition. They also found
that neither model is ready to act as a general keep-or-reject filter. The one
promising narrow result was InternVideo3 checking an enlarged, marked tracker
path. That signal still made two unsafe decisions in 18 known hallucinations,
so it should be combined with the planned contact detector and the existing
rules.

Read these in order:

1. [`evaluation.md`](evaluation.md) explains why PR 80 failed and what the
   follow-up trials changed.
2. [`results.md`](results.md) gives the main measurements and their limits.
3. [`experiments/next_experiment.md`](experiments/next_experiment.md) specifies
   the next useful test after the binary contact detector is available.
4. [`experiments/README.md`](experiments/README.md) explains the retained tools
   and how to run them on a GPU machine.
5. [`sources.md`](sources.md) links the public evidence.

Nothing here changes the production annotator. The experiment code is
standalone and the launchers require machine-specific paths to be supplied as
environment variables.
