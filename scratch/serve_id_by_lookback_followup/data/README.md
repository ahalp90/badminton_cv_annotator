# Frozen development evidence

These eight compressed records are the checked inputs to
`python3 -m serve_id_followup.recompute`.

| File | Contents |
| --- | --- |
| `strict_outgoing_search_results.csv.gz` | First outgoing-contact search, one row per rally |
| `strict_outgoing_search_summary.json.gz` | Strict path settings and result counts |
| `relaxed_contact_evidence.csv.gz` | Less brittle path measurements for 3,200 accepted contacts |
| `relaxed_search_results.csv.gz` | Outgoing-contact and predecessor searches for 239 rallies |
| `relaxed_trajectory_summary.json.gz` | Path settings, coverage and scores |
| `high_shot_correction_results.csv.gz` | PR #82 baseline and narrow high-shot decisions |
| `high_shot_correction_summary.json.gz` | High-shot and rejected-rule counts |
| `serve_setup_sensitivity_summary.json.gz` | Rejected wrist-proximity sensitivities |

The byte size and SHA-256 digest of each file are stored in
`../results/development_metrics.json.gz`.

These records are small and irreplaceable within this handover. They do not
contain raw video trajectories or rerun the upstream vision pipeline.
