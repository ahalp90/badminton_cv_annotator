# Decisions

## Agreed

- Anchor each rally at its earliest accepted geometry/impulse contact. The contact does not need to meet serve criteria.
- Determine the anchor player directly from contact geometry with `attribute_half`.
- Never use `fitted_first_all` as the anchor player or as a trajectory feature.
- Estimate whether the anchor is the first return by looking for incoming shuttle motion towards the anchor player.
- Require simple path structure so a shrinking distance caused by a wild hallucination does not pass by itself.
- Try both a plain direction description and a structured path comparison.
- Treat curve-fit measurements as diagnostics unless they add clear value.
- When motion identifies a return, infer the other player as server.
- Run a second experiment that prepends one missing contact and calls the existing alternating fit once on the augmented sequence.
- Use no fabricated serve frame. Temporal localisation remains a later experiment.
- Evaluate all three videos in full. Show all rallies and the frozen failure subset: 99 covered rallies with a wrong released server label plus 22 with no released label.
- Use all three videos for this EDA. Do not add a train/test split.
- Prefer existing code, NumPy, pandas, scikit-learn and Matplotlib.
- Make dedicated scripts only. Do not change `src/**`.
- Track useful scripts, tests and documents. Ignore external inputs, generated results, plots, case images and delegated-agent records.
- Follow `.github/AGENTS.md`: `.npy.xz`, `.json.gz` and `.csv.gz` for generated data.
- Use plain Australian English. Put the core account of each document within its first 800 words.
- Work only on `investigation/serve-start-trajectory`.

## Recommended, awaiting confirmation

- Use a 30 base-30-frame lookback, matching the original question, rather than the production serve rule's 25-frame lookback.
- Use unambiguous GT contact-1 versus contact-2 anchors to choose the exploratory first-return threshold. Then report server attribution across every rally without relabelling ambiguous or unmatched anchors as contact 1 or 2.
- Show a second threshold curve for final server macro-F1, but do not use it as the main threshold-selection rule.
- Record earlier rejected raw candidates as diagnostics. Do not veto a case merely because a rejected impulse exists.
- Abstain when direct anchor attribution is `None`.
- In Experiment 2, show both the parity-only and player-labelled prepend. This separates the missing-contact effect from the new player vote.
- When a direct anchor half exists but no qualifying path exists, show both a forced anchor-player attribution and an evidence-only abstention.

## Approved on 10 August 2026

- The 30-base-30-frame maximum is only a search limit. The question is whether any usable incoming path appears before the anchor.
- Choose the displayed threshold by first-return F1 on unique contact-1 and contact-2 matches. Explain this in plain language and show every count.
- Include earlier rejected impulse candidates in the analysis rather than using them as an automatic veto.
- Show both the parity-only and player-labelled prepends, with a clear explanation of the difference.
- Show both the forced anchor-player result and the evidence-only abstaining result when no usable path exists.
- The four proposed commit messages in `plan.md` are approved for this local feature branch.
