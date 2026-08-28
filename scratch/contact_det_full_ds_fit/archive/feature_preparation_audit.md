# Check before preparing features for all 40 videos

## Outcome

The feature-saving code is ready for the 40-video run.

The new code produced exactly the same 130,624 rows as the saved pilot for `sset_01`, `sset_15` and `sset_21`. This check compared every field, missing value and field type.

No ShuttleSet contact label was read while preparing these features. The 40-video run has not started yet.

## What was checked

- The accepted 32/8 video split was loaded from `shuttleset_development_split.json`.
- Video IDs, frame rates, resolutions and match details were checked against the saved ShuttleSet tables before feature work began.
- The feature code reads the existing shuttle, pose, court and annotation predictions. It has no import or file reference for the ShuttleSet contact labels.
- Each video record includes its video name, frame rate, feature ranges and row count.
- Each input file is recorded by role, filename, size and hash. A hash is a short value that changes when the file changes. Machine paths are not saved.
- A run is marked `running` before the first video starts. It is marked `complete` only after every requested video finishes.
- The three new pilot files were checked against their saved hashes before their rows were compared with the old pilot file.

## Pilot result

| Video | Feature rows | New feature-file hash |
| --- | ---: | --- |
| `sset_01` | 54,656 | `eda0e3b4a1541f59e109647ccd0258dfa698b92990a9d32f2e07aa933bade3e2` |
| `sset_15` | 38,263 | `318b91a90906cfa4a3412881c8a9578cb2590b115d1001a4139bbc7d198b1b35` |
| `sset_21` | 37,705 | `946bad7abaab4119662982dc84923aec77fe506d0ac672f234872c6be000f864` |
| Total | 130,624 | — |

The saved pilot file hash was `4a5efbd6582701a708270a3b273be2d2572bc3753085ec449b7db815dffec722`.

The check used source commit `6368d507`. The saved video-split file hash was `2f977647136508257107c26a0ab0695571cd008e525fdc32eaf48fa422a46b33`.

## Code checks

- 21 small tests passed.
- 37 tests for the reused pilot code passed.
- All 1,893 project tests passed; 29 were skipped.
- Ruff passed for this experiment directory.
- The whole-project Pyrefly check reported no errors.
- A fresh read-only reviewer found three important problems. The new-file hash check, stopped-run record and saved-metadata check were fixed. The reviewer confirmed all three fixes.

## Next step

Prepare features for all 40 listed videos using the original per-frame motion values. Keep the model comparison and ShuttleSet contact labels out of this run.
