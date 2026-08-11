# WebUI red-team prompt

Please use ChatGPT WebUI Pro or Pro Extended. The repository is public and the WebUI can inspect it.

```text
Read this plan as a hostile but practical research reviewer:

scratch/serve_start_trajectory_exploration/20260810-210532-corrected-contact-refit/plan.md
scratch/serve_start_trajectory_exploration/20260810-210532-corrected-contact-refit/findings.md
scratch/serve_start_trajectory_exploration/20260810-210532-corrected-contact-refit/decisions.md

Inspect the named repository code where needed. This is a small student-project EDA, not a production redesign. Do not implement anything.

The intended question is:

Starting from the earliest accepted geometry/impulse contact in a rally and that contact's own direct Top/Bot attribution, does the shuttle approach that player beforehand strongly enough to say the contact was probably the first return? If so, infer the other player served. Then, as a separate experiment, prepend that inferred server half to the accepted-contact player sequence and rerun the existing alternating fit.

Look especially for another version of the previous circular mistake. Reject the plan if it uses `fitted_first_all`, the alternating phase, or GT server identity to choose the anchor player or calculate incoming motion. `fitted_first_all` may appear only as a baseline score. GT may be used only for evaluation and exploratory threshold choice.

Check these points:

1. Does Experiment 1 genuinely test incoming motion towards the player attributed at the earliest accepted contact?
2. Is “earliest accepted” enough to establish that no earlier accepted contact exists? Are earlier rejected raw impulses correctly treated as diagnostics rather than silently called credible contacts?
3. Are the path checks simple but sufficient to reject gaps, stationary points, gross jumps, recurrence loops and camera-cut joins?
4. Is a quadratic fit kept in its proper place as a comparison, rather than being mistaken for proof of a real parabola?
5. Is the GT first-contact versus second-contact truth join unambiguous and free from leakage? Challenge the treatment of later, unmatched and tolerance-ambiguous anchors.
6. Is choosing a threshold on the same three videos acceptable for the stated EDA, and is the chosen metric aligned with the hypothesis?
7. Is Experiment 2 a legitimate single augmented-sequence refit? It compares prepending `None`, which isolates contact-count parity, with prepending the inferred Top/Bot player, which also adds one vote. It then calls `fit_alternation` once rather than inventing a frame and rerunning geometry.
8. Does the final evaluation clearly separate: first-return detection, contact-local server attribution, the prepended alternating refit, and the old fitted baseline?
9. Do all-rally, covered-rally and known-failure denominators remain explicit? Are abstentions counted honestly?
10. Are the qualifying-path rules defined clearly enough to implement and audit? Is any part unnecessarily elaborate for a student project?

Return:

- a two-sentence statement of what each experiment actually tests;
- PASS or FAIL for Experiment 1 and Experiment 2;
- blocking faults, with exact file/function evidence;
- non-blocking risks;
- the smallest concrete corrections;
- any question that must be answered before implementation;
- a direct check of whether the parity-only and player-labelled prepends are described and interpreted correctly.

Do not propose a neural model, production refactor, new annotation campaign or broad architecture. Do not praise the plan. Concentrate on whether the experiments answer the intended question without circular reasoning.
```
