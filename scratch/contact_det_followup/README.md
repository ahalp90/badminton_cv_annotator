# Contact detector follow-up

This branch found one useful repair: choose player sides across the whole rally instead of judging each contact on its own. On the frozen 47-video test, that change raised fully correct sections from **483 to 901** at the main ±5-frame tolerance. It did not break any section that was already fully correct.

The repair still leaves most sections wrong. Only **22.63%** of predicted sections were fully correct. A separate model tried to find a small, reliable group to auto-accept. It reached **40.87%** precision while keeping 16.14% of sections.

The branch answers two questions:

- **Can cheap rules recover more complete rallies?** Yes. Use the whole-rally side vote.
- **Are we close to a near-100%-precision auto-annotator, even if it rejects most rallies?** No. The tested rejector could not identify a clean subset.

## If you only have two minutes

- The whole-rally side vote is worth keeping. It repaired 418 frozen-test sections at ±5 frames and broke none.
- The global contact cut-off should stay at 0.90 for now. The best development alternative repaired 139 timing-complete sections but broke 126.
- The first-contact model is a clue, not a finished component. It repaired six sections on eight untouched videos and broke none.
- The learned delete rule broke more rallies than it repaired.
- The current keep-or-review model stayed far below the 90% precision target.
- These results do not cover new broadcast conventions. The 47 test videos were new to the model, but the results were not reported by camera layout, tournament, graphics package, or broadcast style.

## The document pack

- [Report](report.md) — the experiment story, results, meaning, and recommendation
- [Useful next work](next_steps.md) — two follow-ups that would answer the remaining product questions
- [Evidence and reproduction](evidence.md) — data splits, result types, source files, and commands

The report includes four simple plots. Matching PNG and SVG files live in `figures/`.

## What this branch changed

The experiments only changed decisions made **after** the contact model had scored frames. They reused saved contact scores, possible contact frames, pose data, shuttle tracks, side guesses, and rally sections. They did not train a new vision model or change the upstream tracker.

The scripts and committed JSON records contain the full evidence. The reader-facing documents provide the short version.
