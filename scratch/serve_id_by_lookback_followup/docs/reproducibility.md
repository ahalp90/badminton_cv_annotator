# Reproducibility and provenance

## Supported recomputation

The clean recomputation starts from eight frozen compressed records under
`data/`. It rebuilds the preferred rule, the rank-1 fallback sensitivity and
the checked development metrics.

Run from the investigation directory:

```bash
python3 -m serve_id_followup.recompute
python3 -m serve_id_followup.recompute --check
python3 -m unittest discover -s tests -v
```

The code uses only the Python standard library. Gzip timestamps are fixed, so
the same inputs produce byte-identical outputs.

`--check` writes fresh results to a temporary directory and compares them with:

- `results/development_metrics.json.gz`
- `results/preferred_server_rule.csv.gz`
- `results/rank1_fallback_sensitivity.csv.gz`

## Frozen sources

| File | Source role |
| --- | --- |
| `strict_outgoing_search_results.csv.gz` | First per-rally outgoing-contact search |
| `strict_outgoing_search_summary.json.gz` | Strict settings and counts |
| `relaxed_contact_evidence.csv.gz` | Later per-contact path measurements |
| `relaxed_search_results.csv.gz` | Later outgoing and predecessor results |
| `relaxed_trajectory_summary.json.gz` | Later settings and counts |
| `high_shot_correction_results.csv.gz` | PR #82 baseline and high-shot decisions |
| `high_shot_correction_summary.json.gz` | High-shot and rejected-rule summaries |
| `serve_setup_sensitivity_summary.json.gz` | Wrist-proximity sensitivities |

`results/development_metrics.json.gz` records the byte size and SHA-256 digest
of every source file.

## Ground-truth boundary

The 239-rally population comes from an annotated rally-to-span crosswalk. This
is a population-selection dependency.

Within that fixed population, the prediction branches use accepted contacts,
player sides and shuttle-path fields. They choose a server before ground truth
is read for scoring.

The rule itself was assembled after development results were inspected. It is
therefore development-set model selection even though the runtime branches do
not read labels.

## What was checked

The clean recomputation verifies:

- PR #82 baseline counts from the frozen correction table
- The 91 direct selected-contact decisions
- The preferred 170/239 server result
- The 171/239 rank-1 fallback sensitivity
- Visible-start and joint scores
- Fix and damage counts relative to PR #82
- The three-frame minimum-path sensitivity
- The historical 160-rally direct-coverage diagnostic
- Source sizes and hashes

The original strict, less brittle, high-shot and pose counts are also carried
from their frozen summaries.

## What this bundle does not rerun

The supported recomputation does not run raw-video TrackNet extraction, pose
inference, accepted-contact generation or scene segmentation.

The original experiment code is preserved in
`archive/original_investigation/original_experiment_code.zip`. Its imports
reflect the historical top-level layout. Use the branch commits named in the
archived worklog when an exact rerun of that old layout is required.

The sibling `scratch/serve_start_trajectory_exploration/` remains the source of
the accepted-contact and trajectory conventions used by those experiments.

## Source packets

Two compressed packets are preserved under `archive/source_packets/`:

- `serve_id_followup_packet_v3_slim.zip` contains the third-iteration narrative,
  figures and supporting calculations
- `serve_id_followup_handover.zip` contains the later synthesis draft

Their SHA-256 digests are recorded in `archive/ARCHIVE_MAP.md`.

The third-iteration packet improved readability and produced the valid 170 and
171 re-scores. It also mixed generations of some redundant derived tables. The
live package omits those tables and produces all checked outputs through one
code path.

## Curved-path evidence boundary

The packet contains a demonstration made from points digitised from an
eight-error figure. It does not contain exact per-frame paths for all 19 usable
labelled cases.

A later audit reports exact-path timing fixes and server damage. The path inputs
and audit helper are absent from the supplied material. The live report retains
the figures as an attributed audit claim and does not mark them as reproduced.

## Figures

The four PNG charts are generated from the checked compressed records:

```bash
MPLCONFIGDIR=/tmp/badminton-matplotlib \
  ~/.venvs/badminton-cicd/bin/python scripts/render_report_figures.py
```

The decision flow is rendered from Mermaid source:

```bash
/home/ariel/.venvs/skill-utils/bin/mermaidx \
  -i figures/preferred_server_rule.mmd \
  -o figures/preferred_server_rule.svg
```
