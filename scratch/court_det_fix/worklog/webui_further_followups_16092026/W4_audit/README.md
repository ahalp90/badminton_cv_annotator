# W4 — Court detector review

**The report is [W4_FINAL_REVIEW.md](W4_FINAL_REVIEW.md).**

## Where the review leaves the project

I would keep developing the detector. Some changes help choose among courts it already proposes. Two specific trial rules made things worse and are not worth extending as replacements.

The seven-frame rank calculation is useful: it chooses an existing GX court instead of a wildly wrong one. But that court was previously judged skewed. The result is a better choice among these options, not a newly correct court.

There is no new remote job or further review requested in this pack.

## Finding what you need

| File | What it contains |
|---|---|
| [W4_FINAL_REVIEW.md](W4_FINAL_REVIEW.md) | The decision, the results that matter, and what remains unknown. |
| [HANDOVER.md](HANDOVER.md) | A short note for a colleague picking up the project. |
| [EVIDENCE_INDEX.md](EVIDENCE_INDEX.md) | The exact repository files and versions behind the findings. |
| [HOW_TO_RUN.md](HOW_TO_RUN.md) | The command to repeat the calculations, and what it will do. |
| [results/README.md](results/README.md) | The calculated results, explained before the raw files. |
| [claims.csv](claims.csv) | The report's short table of what the tests support. |
| `review_checks.py`, `inputs/`, `results/` | The code, copied data fields and numerical outputs. |

The source filenames and candidate IDs are unchanged so they still match the repository. The main report explains each test without asking you to remember the project's folder aliases.

## Repeating the arithmetic

From this folder:

```bash
python review_checks.py
```

It uses only Python's standard library and writes to this folder's `results/` directory. It does not run the detector or need a GPU. The [run notes](HOW_TO_RUN.md) also show how to use the original CSV already on your computer.

## What changed in this edition

The report, source notes, claim table and handover have been rewritten. The calculation's comments, messages and result descriptions use simpler language; its rules and numerical inputs have not changed.

I reran the calculations and checked that the numbers, selected IDs and rankings match the original pack. This edition uses the same reviewed commit, `b36402f1c994f2b000044589cdb6b001f190d0dd`. It is not a new check of the latest branch or of the images.
