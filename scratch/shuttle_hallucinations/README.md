# Saved shuttle guard flags and inpainting near GT contacts

Saved guard codes flag a frame in about 35% of clean GT rally spans in both datasets.
The median number of sidecar-selected inpaint frames within ±5 frames of a GT contact is zero.
Among contact windows with any inpainting, the median is two frames in both datasets.

Measured on 4 October 2026. The inputs cover 40 ShuttleSet videos and
47 separate ShuttleSet22 videos. The overlap videos are omitted to avoid counting the same sources twice.

| Dataset | Rallies with any guard code 1–3 | Rallies with proof code 1 | Contact-window inpaint median | Median if any inpainting |
|---|---:|---:|---:|---:|
| ShuttleSet | 34.84% (757/2,173) | 15.60% (339/2,173) | 0 | 2 |
| ShuttleSet22 | 35.86% (1,227/3,422) | 19.81% (678/3,422) | 0 | 2 |
| Combined | 35.46% (1,984/5,595) | 18.18% (1,017/5,595) | 0 | 2 |

## What was counted

A rally span starts at its first labelled contact and ends just after its last labelled contact.
Both contact frames are included. The stroke CSVs do not timestamp the final landing.
These proportions therefore describe the labelled contact span, rather than the entire flight to rally end.

A guard event means at least one saved frame code is non-zero within that span.
Code 1 identifies a repeated varying-position attractor and is called fabricated proof.
Code 2 identifies a suspect flat attractor. Code 3 marks degraded neighbours or matching attractor positions.
The production policy rejects all three codes. No code-2 frames occur in these saved arrays.
The code-1 column preserves the stronger evidence threshold.

An inpaint-associated frame means a frame inside a producer-sidecar `inpaint_selected` span.
Each contact contributes one inclusive window from contact−5 through contact+5: 11 frames.
The main median includes windows with zero selected frames.
The conditional median uses only windows with at least one selected frame.
The sidecar records model provenance; it does not establish a visually incorrect shuttle position.
All selected frames inside the measured contact windows have visible, non-zero saved coordinates.

## Contact-window detail

| Dataset | Clean contacts | Windows with any inpainting | Inpaint mean | Windows with any guard flag | Flagged-frame median, if any |
|---|---:|---:|---:|---:|---:|
| ShuttleSet | 22,627 | 47.29% (10,700/22,627) | 1.59 | 5.86% (1,325/22,627) | 5 |
| ShuttleSet22 | 38,218 | 49.42% (18,886/38,218) | 1.66 | 6.13% (2,342/38,218) | 7 |

If “inpaint-associated” instead means a non-zero guard code, its median is also zero in both datasets.
Among windows containing a guard code, the median flagged-frame counts are five and seven, respectively.
For final labelled contacts alone, the inpaint median is one frame in both datasets.
The corresponding conditional medians are two and three frames.

## Population and boundary sensitivity

The primary population uses the existing ShuttleSet22 scorer’s whole-rally cleaning rule for both datasets.
A rally is excluded if any row marks a flaw or a contact frame is invalid.
It is also excluded if contact frames do not increase in stroke order.

| Dataset | Source rallies | Clean rallies | Flawed rallies with increasing frames | Rallies with non-increasing frames |
|---|---:|---:|---:|---:|
| ShuttleSet | 3,359 | 2,173 | 1,009 | 177 |
| ShuttleSet22 | 3,965 | 3,422 | 339 | 204 |

No rallies were excluded for out-of-range or missing frame numbers.
The exclusion columns are disjoint. A flawed rally with non-increasing frames appears in the last column.
The sensitivity rows below retain the same valid, increasing-frame requirement.
Padding is a measurement choice; it does not supply a missing landing label.

| Measurement choice | ShuttleSet: any guard flag | ShuttleSet22: any guard flag |
|---|---:|---:|
| Primary: clean first-to-last contact | 34.84% (757/2,173) | 35.86% (1,227/3,422) |
| Retain flawed rallies | 39.35% (1,252/3,182) | 37.86% (1,424/3,761) |
| Pad both ends by 5 frames | 52.14% (1,133/2,173) | 52.57% (1,799/3,422) |
| Extend after the last contact by 15 frames | 37.41% (813/2,173) | 38.14% (1,305/3,422) |
| Extend after the last contact by 30 frames | 44.87% (975/2,173) | 46.32% (1,585/3,422) |

All saved guards use detector version 4 with a three-frame halo.
ShuttleSet22 video 22 has an unavailable guard because its recurrence margin is below the accepted threshold.
Its all-zero codes remain in the primary saved-code denominator (40 clean rallies).
Excluding that video gives 36.28% (1,227/3,382) for any guard flag and 20.05% (678/3,382) for code 1.

ShuttleSet was tracked through 512×288 proxies (sidecar gap threshold 14.4 pixels).
ShuttleSet22 was inpainted from saved source-resolution tracks (threshold 54 pixels).
The results describe saved outputs from those two extraction setups using the same code meanings.

## Earlier RANSAC answer

The [2026-08-14 RANSAC production decision](../../docs/scraper_pipeline/inpaint_hallucination_fix/ransac_production_decision_20260814.md)
already gives a useful related check on `sset_01`, `sset_15` and `sset_21`.
It reports that guard-clean RANSAC candidates coincide exactly with 1,656 of 3,128 labelled contacts (52.9%).
They coincide with the final labelled contact in 147 of 292 rallies (50.3%).
Within a three-frame neighbourhood, the overlaps are 2,616 contacts and 246 final contacts.
Those findings supported keeping RANSAC out of production rejection.

That investigation measures RANSAC candidate conflicts with real contacts.
It does not provide this whole-corpus saved-guard prevalence or the ±5-frame inpaint median.
The [earlier provenance evaluation](../../docs/scraper_pipeline/inpaint_hallucination_fix/inpaint_provenance_coverage_followup_20260731-192654.md)
also reports guard coverage of sidecar-selected frames on those three videos.
It explicitly distinguishes producer provenance from hallucination recall.
The existing conclusions were reused; no RANSAC detector was rerun.

## Related result

The [annotator closeout's shuttle guard addendum](../annotator_closeout_new_courts/shuttle_guard_addendum.md)
reads the same ShuttleSet22 guard codes at the annotator's predicted hits.
The guard flags almost half of the false hits it adds after a rally ends, and almost no real final hits.

## Evidence

- [summary.json.gz](summary.json.gz): aggregate values, populations and guard-availability sensitivity
- [videos.csv.gz](videos.csv.gz): per-video frames, codes and denominator counts
- [rallies.csv.gz](rallies.csv.gz): per-rally counts, cleaning status and boundary sensitivity
- [contacts.csv.gz](contacts.csv.gz): every accepted-frame contact window, including flagged-flaw rallies
- [annotation_rule_check.json.gz](annotation_rule_check.json.gz): direct check of the scorer's cleaning rule

Every input passed its track, guard-code, sidecar and frame-count consistency checks.
A direct check of every annotation CSV confirms that the original scorer selects the same rally and contact totals.
The measurement scripts are not included because they read machine-specific input paths.
