# VLM PR 80 evaluation

This directory records how vision-language models (VLMs) performed in the
cleanup investigation for pull request (PR) 80 and in six later follow-up
experiments. It is a **historical evaluation record**, not the current project
roadmap.

For the fastest orientation, open **[VISUAL_ORIENTATION.md](VISUAL_ORIENTATION.md)** first.

A later trial ran Qwen3.8 on the two strongest retained comparisons without
rewriting this historical record. See
**[QWEN3_8_COMPARISON.md](QWEN3_8_COMPARISON.md)** for the result: the model is
worth keeping as an experiment backend, but the evidence does not support
production integration.

## Contents

- [What this branch established](#what-this-branch-established)
- [Later Qwen3.8 trial](QWEN3_8_COMPARISON.md)
- [Six follow-up reports](#six-follow-up-reports)
- [Earlier investigation record](#earlier-investigation-record)
- [Technical and reproducibility material](#technical-and-reproducibility-material)
- [Historical boundary](#historical-boundary)

## What this branch established

The work did not produce a VLM that was safe to act as the final automatic cleanup authority. It did produce several useful partial signals, benchmarks, data joins, and much clearer failure tests.

The strongest follow-up findings were:

- **Server side carried useful signal.** On the same 32 reviewed rally starts, Intern identified the server correctly in 23 cases and Qwen in 14. That is a useful comparative signal; it is not a complete serve reconstruction result.
- **Serve visibility and exact contact timing were not trustworthy from the tested interface.** Both models called every reviewed case `visible` and made exact frame claims in all 13 cases where physical contact was off-frame, omitted, or unclear.
- **Confidence filtering could not rescue the predicted rally records.** At the main timing tolerance, only 1 of 311 predicted rally records was complete. The strictest tested rule kept seven records, six of which were still wrong or incomplete.
- **The most impressive prompt gain was mostly cue-following.** When the prompt named a candidate contact frame, Intern returned that same frame in 30 of 31 parsed replies.
- **More opening context did not improve the small server test.** The clean 22-second input scored 8/12; the timing hint scored 7/12; supplying every source frame changed no answer.
- **Several outputs are still useful inside a broader system.** The scene benchmark, reviewed rally-start benchmark, automatic observation features, 311-span feature table, rally-opening join, and server-side model signal can all support later ranking, routing, cross-checking, or evaluation work.

**Cross-broadcast generalisation remains unproven.** The matched scene follow-up exposed convention sensitivity: Qwen rejected all 10 meaningful unusual-view live clips, while both models accepted most pure replays as live.

## Six follow-up reports

| Follow-up | Main question | Short result |
|---|---|---|
| [1. Scene comparison](followups/1_scene_comparison.md) | Can either model safely separate live play from replay/non-live footage? | Neither was safe as final authority; the models failed in different ways. |
| [2. Reviewed rally starts](followups/2_final_model_gate.md) | Which model has the stronger server-side signal, and which serve fields are usable? | Intern had the stronger server-ID result; visibility and exact timing failed. |
| [3. Precision-first filtering](followups/3_precision_first_dataset.md) | Can automatic confidence signals isolate a tiny trustworthy rally subset? | No non-empty zero-error rule qualified. |
| [4. Automatic serve support](followups/4_serve_reconstruction.md) | Do automatic observations or proposals help Intern? | The apparent timing gain mostly came from repeating the supplied candidate frame; adding proposals reduced server-ID accuracy. |
| [5. PR 88 lookback](followups/5_pr88_serve_lookback.md) | What did the deterministic server rule justify at branch close? | Development evidence was encouraging, but unseen validation was still required. |
| [6. Longer rally opening](followups/6_rally_opening_context.md) | Does more continuous context or denser frames help server ID? | No net gain in the 12-case diagnostic; the joined 311-span rally-opening dataset remains useful for analysis. |

## Earlier investigation record

The files below describe the investigation **before** those six follow-ups. They still contain useful experiment history and should be read as the earlier evidence record, not as the latest summary for this branch.

- [FIRST_EXPERIMENTS.md](FIRST_EXPERIMENTS.md) — preserved overview of the first experiments.
- [RESULTS_FIRST_EXPERIMENTS.md](RESULTS_FIRST_EXPERIMENTS.md) — earlier aggregate results, including scene, shuttle-track and contact-window work.
- [experiments.md](experiments.md) — what the earlier experiments actually ran.
- [evaluation.md](evaluation.md) — why the original PR 80 benchmark was badly matched to the cleanup problem.
- [prompts.md](prompts.md) — retained prompt designs from the earlier investigation.
- [sources.md](sources.md) — source and evidence references.

The earlier investigation found two narrow leads. Intern's marked/enlarged
shuttle-track check was useful, and focused proposal-level questions performed
better than the original PR 80 timeline task.

## Technical and reproducibility material

- [Follow-up technical index](followups/technical_index.md) — file paths, machine-readable summaries, scripts, evidence limits and reproduction commands.
- [`experiments/`](experiments/) — earlier experiment code and retained machine-readable results.
- [`followups/`](followups/) — the six readable follow-up reports, original protocols and retained evidence records.

The charts and infographics used by the visual orientation are retained under
[`figures/`](figures/) and [`infographics/`](infographics/).

## Historical boundary

At branch close, the next evidence-supported test was an unchanged evaluation
of the PR 88 deterministic serve rule on unseen data. That statement is
**historical context only**. Work elsewhere in the project may have moved
beyond it; this directory does not try to represent developments outside the
`vlm-pr80-followups` record.
