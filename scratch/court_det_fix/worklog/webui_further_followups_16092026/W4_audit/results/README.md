# The calculated results

## C2: the corrected explanation holds

Filtering makes the closest GX court from pair 143 better, from **34.326758 to 12.577456 px**. Across all saved per-pair shortlists before the overall limit, the closest court gets worse, from **7.662057 to 9.035950 px**.

For the Am3 duplicate, row **29665** has the higher score and is the one the matcher keeps for that assignment. It scores **0.7844496413**, compared with **0.6579144844** for row 29686. Its advantage is **0.1265351569**.

The script checks those GX distances and Am3 scores. It does not recreate all the courts or reconstruct the Am3 corner errors.

## L3: combining ranks picks an existing, better option

The calculation ranks the same 30 entries by line score and paint score on each of seven frames. It adds each pair of ranks and averages the seven totals. Lower is better.

| Court, all from frame 0 | Total over seven frames | Average |
|---|---:|---:|
| **22:4579** | **119** | **17.0000** |
| 22:4598 | 133 | 19.0000 |
| 22:4588 | 134 | 19.1429 |

For the winner, the line ranks are `1, 2, 15, 1, 12, 14, 16`. The paint ranks are `3, 8, 8, 16, 13, 4, 6`.

Its median saved corner error is **14.068941 px**. The original shared median-line rule picks `106:93818`, at **799.356692 px**. Choosing separately by line score on each frame gives a median of **501.748229 px**.

These errors are measured against L3's frame annotations. They are not the distances to C2's approved GX reference.

**The lower error does not make this a newly correct court.** `22:4579` is the existing frame-0 line-score winner, which an earlier image review judged skewed. This calculation shows a better choice among the saved entries, not a new image judgement.

## Finding the exact numbers

| File | What it contains |
|---|---|
| [c2_check.json](c2_check.json) | The four corner calculations, their changes, and the Am3 score comparison. |
| [l3_ranks.csv](l3_ranks.csv) | All 30 entries in order, with their ranks and totals. |
| [l3_rank_sum.json](l3_rank_sum.json) | All scores, ranks, selected IDs and later comparisons with the saved errors. |
| [selection_lock.json](selection_lock.json) | The chosen result written before the reference errors are read. It contains no reference-error comparison. |

## Some names used in the raw files

`court_id` includes the frame that first produced the court and its original candidate ID. For example, `gxBQ_window_00_frame_0::22:4579` is court `22:4579` from frame 0.

`line_vector` and `paint_vector` are simply the seven saved scores. They follow frame order **0, 5, 689, 5111, 5766, 77876, 86088**.

`post_lock_reference_comparison` means the error comparison made after the score-based winner was written. `line_nondominated_ids` lists courts for which no other entry has an equal or better line score on every frame and a strictly better score on at least one. There are eight. That count does not show how many courts are correct.

The calculations were rerun for this rewrite. All numerical values and rankings match the original pack. Descriptive text was simplified. No detector or image scorer was run.
