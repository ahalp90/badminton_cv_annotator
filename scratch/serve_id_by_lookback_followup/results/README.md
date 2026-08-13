# Checked development outputs

Run `python3 -m serve_id_followup.recompute --check` from the parent directory
to rebuild these files in a temporary directory and compare their bytes.

| File | Contents |
| --- | --- |
| `development_metrics.json.gz` | Headline scores, branch counts, diagnostics and source checksums |
| `preferred_server_rule.csv.gz` | Per-rally decisions for the frozen 170/239 rule |
| `rank1_fallback_sensitivity.csv.gz` | Per-rally decisions for the 171/239 sensitivity |

The sensitivity is retained for comparison. The preferred rule is the candidate
to test first on unseen rallies.
