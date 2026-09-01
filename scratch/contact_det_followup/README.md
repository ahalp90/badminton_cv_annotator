# Contact detector follow-up

This branch found one useful repair: choose player sides across the whole rally instead of judging each contact on its own. On the held-out 47-video ShuttleSet22 test, that change raised fully correct sections from **483 to 901** at the main ±5-frame tolerance. It did not break any section that was already fully correct.

The repair still leaves most ShuttleSet22 sections wrong. Only **22.63%** of predicted sections were fully correct. A separate keep-or-review model was assessed on 32 ShuttleSet development videos. It reached **40.87%** precision while accepting 16.14% of sections.

The branch answers two questions:

- **Can cheap rules recover more complete rallies?** Yes. Use the whole-rally side vote.
- **Are we close to a near-100%-precision auto-annotator, even if it rejects most rallies?** No. The tested rejector could not identify a clean subset.

## If you only have two minutes

- The whole-rally side vote is worth keeping. It repaired 418 sections on the held-out ShuttleSet22 test at ±5 frames and broke none.
- The global contact cut-off should stay at 0.90 for now. On 40 ShuttleSet development videos, the best alternative repaired 139 timing-complete sections but broke 126.
- The first-contact model is a clue, not a finished component. It repaired six sections on eight ShuttleSet validation videos and broke none.
- The learned delete rule broke more rallies than it repaired.
- The current keep-or-review model stayed far below the 90% precision target.
- These results do not cover new broadcast conventions. The 47 test videos were new to the model, but the results were not reported by camera layout, tournament, graphics package, or broadcast style.

## Which data was used

The contact detector was developed on 40 ShuttleSet videos: 32 training videos and eight validation videos. The final detector was trained on all 40 before it ran on ShuttleSet22.

The baseline and the 483-to-901 side-rule result come from 47 held-out ShuttleSet22 videos. The first-contact, deletion, and keep-or-review models use ShuttleSet development data. Only the first-contact model was later scored on the eight ShuttleSet validation videos that had not been used to develop it.

The ShuttleSet22 predictions and settings were saved before its labels were opened. Nothing was tuned against those 47 test videos.

## The document pack

- [Report](report.md) — the experiment story, results, meaning, and recommendation
- [Useful next work](next_steps.md) — two follow-ups that would answer the remaining product questions
- [Evidence and reproduction](evidence.md) — data splits, result types, source files, and commands

The report includes seven reproducible plots and two flowcharts. Matching PNG and SVG files live in `figures/`.

## What this branch changed

The experiments only changed decisions made **after** the contact model had scored frames. They reused saved contact scores, possible contact frames, pose data, shuttle tracks, side guesses, and rally sections. They did not train a new vision model or change the upstream tracker.

The scripts and committed JSON records contain the full evidence. The reader-facing documents provide the short version.
