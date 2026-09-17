# Repeating the calculations

The calculations are already done. These instructions are here so someone can check a result or use the original CSV without reconstructing the review.

## Using the data in this pack

The script needs Python 3.10 or later. It uses only the standard library; there are no packages to install.

From the extracted folder:

```bash
python review_checks.py
```

It reads the small files in `inputs/` and replaces the calculated JSON and CSV files in `results/`. It does not change the input data. To leave the supplied results alone, another output folder works too:

```bash
python review_checks.py --output /tmp/w4-check
```

## What the script does

For **C2**, it measures the four GX examples against their saved reference corners, allowing the usual 180-degree change of corner labels. It also checks the scores of the two Am3 rows that describe the same line assignment. It does not recreate all the court proposals or reconstruct Am3's reported corner errors.

For **L3**, it uses all 30 entries on all seven frames. It ranks the line and paint scores separately on each frame, adds those ranks, and averages across frames. The lowest average wins. Ties use the full candidate ID in ascending text order.

The script writes the chosen result before reading the reference errors. That file is called `selection_lock.json`. Its purpose is simply to keep the score-based choice separate from the later comparison with annotations. No reference error is used to choose a winner.

After that, the script reads the saved corner errors and calculates their medians. It does not measure corners or score images again.

## Expected results

| Check | Result |
|---|---|
| GX pair-143 error, before → after filtering | 34.326758 → 12.577456 px |
| GX error across the saved shortlists, before → after filtering | 7.662057 → 9.035950 px |
| Higher-scoring Am3 duplicate | 29665 |
| Winner when line and paint ranks are combined | Frame-0 `22:4579` |
| Winner's average rank total | 17 |
| Winner's median saved corner error | 14.068941 px |

The last row is not a fresh visual approval. The earlier image review judged that court skewed.

## Using the original CSV on your computer

The pack contains copied fields from the original L3 CSV, rather than the original file itself. The script can also read the original file already in your checkout:

```bash
python review_checks.py \
  --l3-csv /path/to/badminton_cv_annotator/scratch/court_det_fix/next_steps_20260916/L3_temporal/score_matrix.csv \
  --output /tmp/w4-from-original
```

The `/path/to/` part would be the location of your checkout. Only L3 switches to the original CSV; C2 still uses the included corner examples.

This mode checks that the candidate IDs, frames, scores and eligibility match the copied fields. It stops rather than quietly skipping a missing or ineligible row. A difference is a reason to compare the files or check the revision, not to tune the calculation until it produces the expected winner.

The original-CSV mode was not run in W4 or this rewrite, because the complete original file was not available here. The default calculation was run again for this rewrite, and its numbers and rankings match the original pack.

## Reading the output

[results/README.md](results/README.md) explains the output files. The code is [review_checks.py](review_checks.py). The input notes are in [inputs/README.md](inputs/README.md).

No network, images, GPU or remote host is needed. Nothing here reruns the detector, so nothing here can establish that a court looks right in an image.
