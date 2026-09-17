# About the input files

These are small copies of the data fields W4 used. They came from the real repository results, not from made-up examples. They are not the original downloaded files.

| File | What is in it |
|---|---|
| `c2_compact.json` | The GX reference, four sets of proposed corners, and the Am3 assignment IDs and scores needed for the duplicate check. |
| `l3_scores_compact.json` | The 30 candidate IDs, their seven line scores and seven paint scores, and their recorded eligibility status. |
| `l3_recorded_reference_errors.json` | The saved per-frame corner errors used after the score-based choices have been made. |
| `build_compact_inputs.py` | The earlier script that wrote the L3 score JSON from the copied values. The JSON is already included, so this does not need to run. |

The three JSON input files are unchanged from the original W4 pack. Some field names use the repository's terminology so the existing calculation can read them.

For the L3 scores, the original review copied the needed fields from all 210 rows of `score_matrix.csv` at commit `b36402f1c994f2b000044589cdb6b001f190d0dd`. Some paint scores in the builder are written as fractions; these reproduce the recorded decimal values.

Keeping the reference errors in a separate file makes it clear that the rank calculation does not use them to choose a court. The program reads that file only after it has written the winner.

The [source notes](../EVIDENCE_INDEX.md) identify the repository paths. The [run instructions](../HOW_TO_RUN.md) show the optional mode that reads your existing original CSV and checks it against these copies. That mode has not been exercised on an original file in this session.
