# Contact, serve and automatic-use numbers at a glance

The final detector finds contacts; the ranking model selects 784 rally clips for review.

**Trusted GT:** 3,422 rallies and 38,218 contacts. **All GT:** all 3,965 source rallies and 43,159 contacts.

The difference is 543 rallies excluded during label cleaning. Both scores use the same predictions.

Results below use **±10 frames on a 30 fps clock**. The tighter ±5 check is at the end.

## Individual contacts

| Task | Labels | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Timing only | Trusted GT | 81.0% | 88.2% | 84.5% |
| Timing only | All GT | 90.1% | 86.9% | 88.4% |
| Timing + correct player | Trusted GT | 78.5% | 85.5% | 81.8% |
| Timing + correct player | All GT | 87.2% | 84.0% | 85.6% |

## Serve and non-serve recall

| Contact type | Labels | Timing recall | Timing + correct-player recall |
|---|---|---:|---:|
| Non-serve | Trusted GT | 88.9% | 86.3% |
| Non-serve | All GT | 88.4% | 85.7% |
| Serve | Trusted GT | 81.3% | 77.4% |
| Serve | All GT | 72.0% | 67.3% |

## Does the proposed rally start at the serve?

| Task | Labels | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Start is serve | Trusted GT | 70.4% | 76.7% | 73.4% |
| Start is serve | All GT | 72.1% | 67.7% | 69.8% |
| Start + correct server | Trusted GT | 68.1% | 74.1% | 71.0% |
| Start + correct server | All GT | 68.5% | 64.4% | 66.4% |

## Automatic use

Fully correct means every contact and player is right. A whole-rally clip can still contain contact errors.

Trusted-GT precision uses the 740 judgeable clips. All-GT precision uses all 784; unknown clips receive no credit.

| Selection task | Labels | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Fully correct rally | Trusted GT | 83.2% | 18.0% | 29.6% |
| Fully correct rally | All GT | 78.4% | 15.5% | 25.9% |
| Contains one whole rally | Trusted GT | 98.4% | 21.3% | 35.0% |
| Contains one whole rally | All GT | 94.3% | 18.6% | 31.1% |

### Selected clip counts

| Labels | Fully correct | Wrong | Unknown | Contains one whole rally |
|---|---:|---:|---:|---:|
| Trusted GT | 616 | 124 | 44 | 728 |
| All GT | 615 | 140 | 29 | 739 |

## Whole-rally recovery before selection

Each entry counts fully correct rallies. The same detector predictions are checked at both timing windows.

| Detector | Trusted ±10 | All GT ±10 | Trusted ±5 | All GT ±5 |
|---|---:|---:|---:|---:|
| Original detector | 995 | 993 | 901 | 900 |
| First-contact repair | 1,105 | 1,103 | 1,001 | 1,000 |
| Whole-sequence model | 1,435 | 1,433 | 1,224 | 1,223 |
| + one later contact | 1,597 | 1,596 | 1,327 | 1,326 |
| + local insertion score | 1,622 | 1,621 | 1,350 | 1,349 |
| Boundary fix only | 1,732 | 1,732 | 1,404 | 1,403 |
| Final detector | 1,763 | 1,763 | 1,430 | 1,429 |
| Wider early shortlist | 1,767 | 1,767 | 1,425 | 1,424 |

<details>
<summary>Tighter timing check: ±5</summary>

### Individual contacts

| Task | Labels | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Timing only | Trusted GT | 79.3% | 86.3% | 82.6% |
| Timing only | All GT | 88.1% | 84.9% | 86.5% |
| Timing + correct player | Trusted GT | 76.9% | 83.7% | 80.2% |
| Timing + correct player | All GT | 85.5% | 82.4% | 83.9% |

### Serve and non-serve recall

| Contact type | Labels | Timing recall | Timing + correct-player recall |
|---|---|---:|---:|
| Non-serve | Trusted GT | 87.9% | 85.5% |
| Non-serve | All GT | 87.4% | 84.9% |
| Serve | Trusted GT | 69.3% | 66.1% |
| Serve | All GT | 61.0% | 57.5% |

### Rally starts

| Task | Labels | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Start is serve | Trusted GT | 60.0% | 65.3% | 62.5% |
| Start is serve | All GT | 61.0% | 57.3% | 59.1% |
| Start + correct server | Trusted GT | 58.2% | 63.3% | 60.6% |
| Start + correct server | All GT | 58.5% | 54.9% | 56.6% |

### Selected clips

| Labels | Fully correct | Wrong | Unknown | Contains one whole rally |
|---|---:|---:|---:|---:|
| Trusted GT | 549 | 191 | 44 | 728 |
| All GT | 549 | 207 | 28 | 739 |

</details>

## Reproduce these results

Run from the repository root with the original ShuttleSet22 annotation folder:

```bash
PYTHONPATH="$PWD/src:$PWD" ~/.venvs/badminton-cicd/bin/python \
  -m scratch.contact_det_closing_pass.scripts.summarise_metrics \
  --annotations /path/to/ShuttleSet22
```

This rebuilds the saved counts, this reference and the comparison charts from saved predictions. It checks the trusted-GT counts against the original experiments. No training or vision inference is needed.

Saved counts: `results/metric_summary.json.gz`. Clip review: [individual notes](results/selected_clip_review.csv).
