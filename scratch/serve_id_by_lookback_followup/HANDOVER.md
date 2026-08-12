# Re-entry handover

## Current position

The investigation is complete. The preferred development candidate combines
new shuttle-motion evidence with the existing PR #82 answer:

1. Find the earliest accepted contact followed by believable motion away from
   the player
2. Use motion before that contact to decide whether it looks serve-like or
   return-like
3. Keep the PR #82 answer when the path cannot support that decision

The result is 170/239 correct server sides, 132/239 correct visible-start
attributions and 117/239 with both correct. It repairs 20 PR #82 server errors
and introduces 13.

## Next action

Run the frozen preferred rule on unseen rallies. Do not change thresholds after
reading those labels. Report server side, visible-start attribution and their
joint result separately.

The exact rule and reporting requirements are in
[docs/next_steps.md](docs/next_steps.md).

## Fast fact finding

| Question | Source |
| --- | --- |
| What happened across the three iterations? | `report.md` |
| What are the exact counts and thresholds? | `docs/results.md` |
| How is the preferred rule recomputed? | `docs/reproducibility.md` |
| Where is the frozen evidence? | `data/README.md` |
| Where are the checked outputs? | `results/README.md` |
| What was tried and abandoned? | `report.md` and `archive/ARCHIVE_MAP.md` |
| What did the old filenames mean? | `docs/results.md`, technical footnote |

## Recompute

Run from this directory:

```bash
python3 -m serve_id_followup.recompute --check
python3 -m unittest discover -s tests -v
```

The recomputation uses only the Python standard library and the eight frozen
records under `data/`.

## Boundaries

- The 239-rally result covers only cases where one prediction maps cleanly to
  one labelled rally
- Prediction branches do not read server or stroke labels
- The rule was assembled after inspecting development labels
- The result is not an end-to-end score over all 292 ground-truth rallies
- Raw TrackNet, pose, contact and scene processing are outside this bundle
- No production code changed during this investigation
