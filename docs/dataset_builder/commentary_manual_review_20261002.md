# Manual commentary review — 2 October 2026

The pipeline pairs commentary with the rally it most plausibly describes. Each
candidate commentary–rally link is one proposed pairing.

Curtis Martin reviewed 134 commentary passages from 85 ShuttleSet and ShuttleSet22 videos. They cover 135 candidate commentary–rally links from the 3,500 links in the frozen dataset-v1 export.

The review found 116 passages that described the candidate rally, 13 that contained general discussion rather than commentary on that rally, and five that were unclear. One unclear passage had two candidate rallies, giving six unclear links.

These counts describe the reviewed set. The sampling method and its limits are
explained below.

## Results

| Association label | Passages | Candidate links |
| --- | ---: | ---: |
| `matches` | 116 | 116 |
| `general` | 13 | 13 |
| `unclear` | 5 | 6 |
| `other` | 0 | 0 |
| **Total** | **134** | **135** |

Speech timing was reviewed separately: 128 passages were marked `matches`, four `off` and two `unclear`. Some timing matches also have delay notes, which remain in the saved review data.

`general` means that the passage did not describe the specific candidate rally. It can still contain useful match or player discussion. The unclear cases included poor source rally intervals and audio that was too difficult to judge.

Passage IDs identify the dataset, video and commentary chunk: `sset_14_c19`,
for example, is ShuttleSet video 14, chunk 19.

Four passages were flagged for timing problems: `sset_14_c19`, `ss22_28_c42`, `ss22_29_c24` and `sset_19_c51`.

The review also caught several text problems:

- `ss22_28_c42` and `ss22_42_c84` contain player-name errors already present in the raw exported text;
- in `sset_34_c39`, the raw text and review note say “train”, while the cleaned text changes it to “match”;
- other notes record quiet audio or approximate paraphrasing.

Transcript fidelity was not scored systematically, and the review did not alter the frozen text or timestamps.

## Sample construction

Selection used a fixed seed and favoured coverage across videos, replay-masked cases, commentary inside a rally and commentary after a rally. Passages contained at least five words and review clips lasted no more than 105 seconds. The review proceeded in batches; later batches excluded candidate rallies
whose annotated intervals lasted only one frame.

This sampling was designed to expose different failure modes. It does not provide a population accuracy estimate or measure valid links that the pairing process missed.

## Evidence files

The review files are under [`data/commentary_review_20261002/`](data/commentary_review_20261002/) beside this document.

| File | Contents |
| --- | --- |
| `reviewed-passages.csv.gz` | 134 passage judgements with raw and cleaned text, source references, speech times, labels and notes |
| `reviewed-links.csv.gz` | 135 candidate-link judgements with frozen export keys and source rally boundaries |
| `all-links-review-status.csv.gz` | All 3,500 candidate links with review status attached; 3,365 remain unreviewed |
| `review-responses.json.gz` | Original saved review responses |
| `review-manifest.json.gz` | Selected passages, provenance and review context bounds |
| `review-summary.json.gz` | Counts and review scope |

The link tables join to the frozen export on:

```text
(run_id, source_dataset, video_id, chunk_id, rally_origin, rally_id)
```

Identifiers remain strings. A passage can have more than one candidate row. `unclear` means the reviewer inspected the link but could not judge it; `unreviewed` means it was not part of this review.

Source times are seconds in the recording. `source_url` identifies the public recording where available. Paths on the review machine, `Carmack`, record the local copies used during
review and are retained as provenance.

The review files do not include the video clips. The compressed labels and notes are sufficient to identify every reviewed passage and candidate link.
