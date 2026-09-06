# Archived development aids

[development_aids_2026-09-06.tar.gz](development_aids_2026-09-06.tar.gz) retains 16 small files from the completed work: about 12 KB compressed, 32 KB unpacked.

- `analysis/`: one script for comparing per-video gains, losses, costs and acceptance. It needs the saved broader results and cached chooser inputs.
- `launches/`: seven HPC command templates for training, broader comparisons, boundary fixes and the visual-model experiment.
- `audits/`: findings and checked decisions that explain useful fixes and rejected concerns. Three records were copied from the related external-audit folders.

These are historical development aids. Update machine paths and output locations before reusing a script. Current results and follow-up ideas start at [the main report](../README.md).

Inspect the contents from this directory:

```bash
tar -tzf development_aids_2026-09-06.tar.gz
```

## Which auditors helped?

| Agent | Recorded value in this closing pass |
|---|---|
| Gemini 3.8 Flash High | Concrete fixes: consistent saved-model keys, preserved player guesses on unmatched starts, and separate accepted/all counters. Its learning audits also produced several rejected claims. |
| Opus 4.6 Thinking | Mostly a second check. It flagged a valid limit on attributing gains to features introduced together; most proposed implementation concerns required no change. |
| DeepSeek Flash | No completed audit found. The learning-audit record says the available catalogue had no DeepSeek model. |

The rest of the old `worklog/` was discarded: caches, transcripts, report drafts, duplicated summaries, obsolete prompts and superseded experimental outputs. Current code, results, models and clip-review evidence were retained.
