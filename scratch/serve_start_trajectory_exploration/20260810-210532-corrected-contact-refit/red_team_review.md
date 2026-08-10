# Red-team review

## Bottom line

Claude Opus judged both corrected experiments sound. Gemini Pro rejected both. The main disagreement concerned the alternating refit, and direct inspection gives a more precise answer than either verdict alone.

Adding one leading position to the contact sequence leaves the alternating phase assigned to every original contact unchanged. The longer sequence changes which player appears first. A leading `None` therefore measures the parity effect of one missing contact. A leading inferred `Top` or `Bot` measures the same parity effect and adds one vote to the matching phase.

The refit is a valid counterfactual if the report says exactly that. It would be misleading to claim that the inferred player normally overrules the later geometry. The revised plan reports both variants and the original fit margin.

## Points accepted

- Requiring both players throughout the incoming path is unnecessary. The anchor player's position is enough for the main direction test. The other player remains diagnostic.
- Exact equal distances in `attribute_half` resolve to Top. Record ties and do not present them as strong attribution.
- Quadratic fit error cannot prove that a path is genuine or parabolic. Keep it diagnostic unless a separately labelled variant clearly helps.
- There are only 16 strict second-contact anchors in the existing extraction, plus one anchor within tolerance of both contacts 1 and 2. Every precision and recall result needs raw counts and an explicit in-sample EDA warning.
- Experiment 2 should use the same binary threshold as Experiment 1. Adding a second confidence rule would not be justified unless the first result exposes a clear need.

## Points rejected

Gemini called threshold selection on the three EDA videos “truth leakage” that invalidates all-rally scoring. The threshold is in-sample and cannot support a generalisation claim, but the user explicitly chose a full-three-video EDA with plain definable parameters. Separate labelled and unmatched groups, raw counts and honest wording are enough for this scope.

Gemini also treated every earlier rejected raw impulse as strong physical evidence and demanded a veto. The source does not justify that claim. Raw candidates can fail the wrist gate, lose suppression, or fall under definitive exclusion. They will be shown as diagnostics. A new rule for promoting rejected candidates would be another experiment.

Gemini preferred the production 25-frame horizon. The original question proposed up to 30 base-30 frames and this is not the production serve rule. The horizon remains a user choice.

## Checks on reviewer conduct

Both direct-host reviews were read-only. The launcher's Git tripwire passed for both, so neither changed the repository.

## WebUI review

The user supplied a later WebUI review in `WEBUI_RED_TEAM_RESPONSE.md`. It agreed that the anchor and motion test are no longer circular, and passed both prepend versions. It asked for five small corrections before implementation:

- use the closest path inside a stated 30-frame maximum;
- keep the path and contact inside one tracker scene;
- define GT contact matching by the literal number of contacts within tolerance;
- report no usable path as an abstention as well as a forced anchor-player result;
- choose the displayed setting by first-return F1, not final server F1.

The plan now includes all five. It also replaces “independent player vote” with the more accurate “one extra vote derived from Experiment 1”.
