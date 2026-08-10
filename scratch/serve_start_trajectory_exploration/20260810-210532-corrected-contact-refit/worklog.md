# Worklog

## 2026-08-10: corrected scope

The earlier EDA was challenged because it started from the rally-wide alternating-fit label rather than the earliest contact's own measured player. It also reversed the final scalar label without prepending a contact or rerunning the fit.

The corrected goal is now fixed: test incoming motion towards the directly attributed earliest accepted contact, then separately test the effect of prepending an inferred server half and running one augmented-sequence alternation fit.

Read-only source tracing established the accepted-contact flow, direct half attribution, fit semantics, GT join, release inputs and sticky reconstruction. Read-only delegated reviews checked the smallest useful motion gates and the unknown-time refit seam.

Independent Gemini and Claude reviews disagreed about the prepend. Direct inspection resolves the disagreement: adding a leading position leaves every original contact's phase assignment unchanged, but changes the first-player parity. A non-missing inferred half adds one phase vote. The plan now compares a `None` prepend with an inferred-half prepend so these effects cannot be conflated.

No implementation has started. No production code will change. The next step is user approval of the remaining choices and commit messages, followed by a WebUI and direct-host red-team review of the plan.

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
