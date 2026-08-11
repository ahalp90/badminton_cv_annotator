# Corrected serve trajectory investigation handover

## Current state

The corrected row-level analysis and first report rewrite are committed on `investigation/serve-start-trajectory`. A fourth presentation-only batch has passed its WebUI and native cold reads plus its focused gates. Only the approved fourth commit remains. No production file under `src/**` has changed.

Read `report.md` first. It now gives the four-score server result, upstream anchor evidence and next action before its optional extended summary. Per-video and sensitivity tables sit under `Supporting breakdowns (optional)`.

## Fixed result contract

- 239 one-to-one rallies are primary whenever one predicted rally must represent one GT rally.
- 249 covered rallies provide merge sensitivity under the current `COVERED` definition.
- All 292 GT rallies provide an end-to-end view that includes segmentation failures.
- ±10 base-30fps frames is the main contact baseline. ±5 is strict and ±30 is a sanity check.
- Nearest GT ordinal and signed/absolute offset survive ambiguity. A separate count records multiple GT strokes in the window.
- The historical motion rule remains unchanged at 0.25 BH total movement, 0.25 BH net closure and 55% approaching steps.
- The new rule calls incoming only when the robust fitted decrease reaches 0.05 apparent body heights. The threshold was fixed before corrected scoring and was never swept.
- Residual RMS and trend-to-jitter remain continuous diagnostics.
- Both rules use identical thresholds under recurrence-only and recurrence-plus-producer-inpaint masks.

## Headline findings

At ±10, the 239 primary anchors are nearest 119 serves, 19 first returns and 4 later strokes. Another 97 are unmatched. Five windows contain multiple GT strokes. Unique serve/return truth contains 118 serves and 17 returns.

Later accepted contacts recover the serve in 49 of the 97 unmatched sequences. Another 36 recover the first return without recovering the serve. Nine first match another GT stroke. Three never match later GT.

Usable recurrence-mask evidence exists in 24/239 primary rallies and 19/135 unique-truth anchors. The 0.05-BH rule makes 9 correct return calls and 4 false return calls. The historical rule makes the same 9 correct calls and 3 false calls. Adding the producer mask cuts usable unique-truth paths from 19 to 10, with 7 correct return calls and no false calls.

On 239 primary rallies, the released alternating fit is correct in 124. The earliest-contact player is correct in 152. The direct 0.05-BH rule is correct in 163. The producer-mask version is correct in 160. Prepending an unknown or inferred contact before the alternating fit reaches only 125 and 127.

## Checks completed

- The corrected validator reloads all frozen fixtures before checking saved GT, spans, contacts, player guesses and server inputs.
- It independently rebuilds alignment, unmatched sequences, path arithmetic, robust trends, fixed rules, server predictions and all JSON metrics.
- Batch 2 Opus review found no blocking arithmetic or population defect. Its source-boundary finding was fixed before commit `471022e`.
- All six final plots have been inspected directly against `PLOT_READABILITY_AUDIT.md`. One overlapping legend was found and fixed.
- The fresh WebUI review is preserved in `review_feedback.md`. It passed basic comprehension and identified six required plot/report clarifications. All six are implemented.
- A second readability pass found that the server plot named only 24/239 recurrence-mask paths. The final plot now also names 14/239 producer-mask paths and repeats the fallback-plus-flip method.
- The final read-only Gemini Pro audit passed with no blocking defect and a clean Git tripwire.
- Final gates: validator exit 0; 55 focused tests passed; investigation Ruff exit 0; Serena diagnostics empty; pinned Pyrefly exit 0; whole pytest exit 0 with 1,456 passed and 29 skipped; `git diff --check` exit 0. Whole-project Ruff remains exit 1 on the unchanged 661 unrelated findings.
- The progressive-disclosure WebUI audit rated the revised report suitable for the target overloaded reader. Its final suggestions compressed the Bottom line, moved anchor evidence and action to the first screen, and demoted supporting tables.
- A fresh native reader reproduced the four intended conclusions and passed the final report and all six plots with no blocking issue.
- The final analysis rerun reproduced all six decompressed CSV/JSON outputs exactly. The source-backed validator, 55 focused tests, investigation Ruff, Serena diagnostics and `git diff --check` pass.
- Whole-project Pyrefly and pytest also passed before the user clarified that whole-project gates are unnecessary for this scratch-only presentation change. Whole-project Ruff retained the unchanged 661 unrelated findings.

## Next actions

1. Make the fourth approved commit with the exact message in `plan.md`.
2. Do not merge to `main`.

## Git checkpoints

- `360c9b3`: checked alignment and accepted-sequence records.
- `471022e`: corrected populations, fixed motion rules, server results and source-backed validator.
- `9bb55a3`: corrected standalone report and six denominator-aware plots.
- Progressive-disclosure report/plot commit: approved message is recorded in `plan.md`; commit remains pending.

Generated result tables and plots remain ignored under `outputs/`. The tracked report is regenerated from those checked tables.
