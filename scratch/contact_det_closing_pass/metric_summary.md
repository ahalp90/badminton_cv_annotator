# Contact detector: reproducible numbers

The final detector recovers complete rallies from saved vision outputs. These tables compare trusted labels with every original source label.

**Trusted GT:** 3,422 rallies and 38,218 contacts. **All GT:** 3,965 rallies and 43,159 contacts, including flagged rows. Missing labels receive no credit; restoring flawed labels does not make them reliable.

## Whole-rally recovery

Each cell is the number of fully correct rallies at **±10 / ±5** frames on a 30 fps clock.

| Detector | Trusted GT | All GT |
|---|---:|---:|
| Original detector | 995 / 901 | 993 / 900 |
| First-contact repair | 1,105 / 1,001 | 1,103 / 1,000 |
| Whole-sequence model | 1,435 / 1,224 | 1,433 / 1,223 |
| + one later contact | 1,597 / 1,327 | 1,596 / 1,326 |
| + local insertion score | 1,622 / 1,350 | 1,621 / 1,349 |
| Boundary fix only | 1,732 / 1,404 | 1,732 / 1,403 |
| Final detector | 1,763 / 1,430 | 1,763 / 1,429 |
| Wider early shortlist | 1,767 / 1,425 | 1,767 / 1,424 |

## Final contact and serve performance

Each cell is **precision / recall / F1**. All predictions stay in the contact and start denominators.

| Task | Timing window | Trusted GT | All GT |
|---|---|---:|---:|
| All contacts | ±10 | 81.0 / 88.2 / 84.5% | 90.1 / 86.9 / 88.4% |
| All contacts | ±5 | 79.3 / 86.3 / 82.6% | 88.1 / 84.9 / 86.5% |
| Contacts + correct player | ±10 | 78.5 / 85.5 / 81.8% | 87.2 / 84.0 / 85.6% |
| Contacts + correct player | ±5 | 76.9 / 83.7 / 80.2% | 85.5 / 82.4 / 83.9% |
| Start is the serve | ±10 | 70.4 / 76.7 / 73.4% | 72.1 / 67.7 / 69.8% |
| Start is the serve | ±5 | 60.0 / 65.3 / 62.5% | 61.0 / 57.3 / 59.1% |
| Start + correct server | ±10 | 68.1 / 74.1 / 71.0% | 68.5 / 64.4 / 66.4% |
| Start + correct server | ±5 | 58.2 / 63.3 / 60.6% | 58.5 / 54.9 / 56.6% |

Recall by labelled contact type (timing / timing + correct player):

| Contact type | Timing window | Trusted GT | All GT |
|---|---|---:|---:|
| Serve | ±10 | 81.3% / 77.4% | 72.0% / 67.3% |
| Serve | ±5 | 69.3% / 66.1% | 61.0% / 57.5% |
| Non-serve | ±10 | 88.9% / 86.3% | 88.4% / 85.7% |
| Non-serve | ±5 | 87.9% / 85.5% | 87.4% / 84.9% |

## The 784 selected proposals

These counts keep unknown cases visible. A whole rally can still contain contact mistakes.

| Labels | Timing window | Fully correct | Wrong | Unknown | Contains one whole rally |
|---|---|---:|---:|---:|---:|
| Trusted GT | ±10 | 616 | 124 | 44 | 728 |
| Trusted GT | ±5 | 549 | 191 | 44 | 728 |
| All GT | ±10 | 615 | 140 | 29 | 739 |
| All GT | ±5 | 549 | 207 | 28 | 739 |

## Rebuild

Run from the repository root with the original ShuttleSet22 annotation folder:

```bash
PYTHONPATH="$PWD/src:$PWD" ~/.venvs/badminton-cicd/bin/python \
  -m scratch.contact_det_closing_pass.scripts.summarise_metrics \
  --annotations /path/to/ShuttleSet22
```

The script checks the trusted-label results against the saved experiments before writing this table and `results/metric_summary.json.gz`. It does not train models or rerun vision.
