# Worklog

## 2026-08-10: corrected scope

The earlier EDA was challenged because it started from the rally-wide alternating-fit label rather than the earliest contact's own measured player. It also reversed the final scalar label without prepending a contact or rerunning the fit.

The corrected goal is now fixed: test incoming motion towards the directly attributed earliest accepted contact, then separately test the effect of prepending an inferred server half and running one augmented-sequence alternation fit.

Read-only source tracing established the accepted-contact flow, direct half attribution, fit semantics, GT join, release inputs and sticky reconstruction. Read-only delegated reviews checked the smallest useful motion gates and the unknown-time refit seam.

Independent Gemini and Claude reviews disagreed about the prepend. Direct inspection resolves the disagreement: adding a leading position leaves every original contact's phase assignment unchanged, but changes the first-player parity. A non-missing inferred half adds one phase vote. The plan now compares a `None` prepend with an inferred-half prepend so these effects cannot be conflated.

At this planning point, no implementation had started and no production code was going to change. The next step was user approval of the remaining choices and commit messages, followed by a WebUI and direct-host red-team review of the plan.

## Checks run by planning reviews

- 54 focused trajectory, inpaint and sticky tests: passed.
- 9 focused alternating-fit/injection tests: passed.
- 20 focused contact/serve tests: passed.
- Repository diffs from the read-only workers: none.

These checks confirm existing behaviour only. They do not validate the corrected experiment, which has not been implemented.

## 2026-08-10: user approval and WebUI review

The user approved the 30-frame maximum, raw-candidate diagnostics, both prepend variants, both no-path results and all proposed commits. They asked that the central question stay in ordinary language.

The WebUI review found no circularity. It asked for five exact rules before implementation: choose the nearest path inside the maximum window; keep the path and anchor in one tracker scene; classify GT matches by the literal count within tolerance; keep no-path abstention separate; and choose the displayed setting by first-return F1. These rules are now in `plan.md`.

Implementation may proceed autonomously.

## 2026-08-10: experiment run

The dedicated scripts linked and verified all three frozen releases, TrackNet arrays and pose arrays. The analysis rebuilt direct player attribution for every accepted contact. The unmodified alternating fits matched the frozen results before any counterfactual contact was added.

The full run covered 292 ShuttleSet rallies. One predicted span covered 249 rallies. Clear anchor truth was available for 87 serves and 16 first returns. The main incoming-motion rule found 11 returns, made 3 false calls and missed 5 returns.

The rule fired in 16 covered rallies. Directly naming the other player as server was right in 13. Prepending an unknown-player contact and refitting was right in 8. Prepending a contact by the other player and refitting was right in 9.

The first independent review found unclear denominators and labels in the report and plots. Those labels now say exactly what is counted. The arithmetic audit reproduced the result and confirmed that GT changes do not affect per-rally decisions. It also found that the first validator reused analysis code, so the validator was replaced with an independent recalculation.

Focused checks so far:

- trajectory tests: 24 passed;
- dedicated Ruff check: passed;
- independent output validator: passed;
- full analysis over all three videos: passed.

The full repository gates remain before close-out.

## 2026-08-10: close-out checks

The regenerated report passed a second cold read. The reviewer found the earlier denominator, method-name, abstention, axis and TrackNet-source ambiguities resolved.

Final checks:

- dedicated Ruff check: exit 0;
- independent output validator: exit 0;
- pinned whole-project Pyrefly: exit 0, no errors and 20 configured suppressions;
- whole-project pytest: exit 0, 1,456 passed and 29 skipped;
- whole-project Ruff: exit 1 on 661 existing findings across unrelated files;
- `git diff --check`: exit 0.

The dedicated directory has no Ruff findings. The repository-wide Ruff failures are outside this investigation and were left unchanged.
