# Manual commentary review, 2 October 2026

Curtis Martin reviewed 134 commentary passages from 85 videos across
ShuttleSet and ShuttleSet22. These represent 135 candidate links in the
frozen dataset-v1 export, which contains 3,500 links. The full dataset remains
the delivery; this review supplies quality evidence for selected examples.

Ari requested a human check of the existing video–commentary pairs before
submission. Curtis performed this labelled review. The frozen main archive
already includes timestamped transcripts, 6,254 cleaned passages and the
3,500 candidate rally links. The 2,756 passages without a rally link retain
their video identity and timestamps. They remain part of the delivery.

## Method and results

The reviewer watched the candidate rally and listened to the audio. Each
candidate was labelled `matches`, `other`, `general` or `unclear`. Speech
timing was labelled separately as `matches`, `off` or `unclear`. Original
notes and review timestamps are preserved without reclassification.

| Association label | Passages | Candidate links |
| --- | ---: | ---: |
| matches | 116 | 116 |
| general | 13 | 13 |
| unclear | 5 | 6 |
| other | 0 | 0 |
| Total | 134 | 135 |

One unclear passage has two candidates, explaining the different totals.
Passage timing labels are 128 matching, four off and two unclear. Some matching
timing labels also have delay notes; the notes qualify those judgements.

Selection used a fixed seed and favoured different videos and replay-masked,
inside-rally and post-rally cases. Passages had at least five words and
clips lasted at most 105 seconds. Later batches excluded one-frame candidate
intervals. The sample is deliberately selected, not random. It cannot
establish whole-dataset accuracy, confidence bounds or missed-link coverage.

## Findings

General discussion appeared in every main selection category. This label
means the passage was not judged to describe that specific rally; it is not
automatically an unusable video–text example. Broader match or player context
may still be useful. The 116 matching passages are 86.6% of the reviewed
sample, not an overall video–text accuracy score. Unclear cases included
poor source rally intervals and inaudible speech.

Four passages were flagged for timing problems: `sset_14_c19`,
`ss22_28_c42`, `ss22_29_c24` and `sset_19_c51`. Their original notes retain
the observed times. These do not establish a shared offset across videos.

Player-name errors were flagged in `ss22_28_c42` and `ss22_42_c84`. The
raw exported text already contains the errors, and cleaning retains them.
For `sset_34_c39`, raw text and the reviewer agree on “train,” while cleaned
text changes it to “match.” Other notes identify quiet audio or approximate
paraphrasing. Text fidelity was not systematically labelled, so it has no
measured accuracy rate. No source text or timestamp was corrected here.

## Evidence files and joining

The files are under [data/commentary_review_20261002](data/commentary_review_20261002/):

- `reviewed-passages.csv.gz`: 134 passage judgements with raw and cleaned
  text, source references, speech times, labels and notes.
- `reviewed-links.csv.gz`: 135 candidate judgements with exact frozen
  export keys and source rally boundaries.
- `all-links-review-status.csv.gz`: all 3,500 frozen link rows with review
  status and labels attached. The 3,365 unreviewed rows have empty labels.
- `review-responses.json.gz`: original saved responses.
- `review-manifest.json.gz`: selected passages, source provenance and
  review context bounds, without local clip filenames.
- `review-summary.json.gz`: counts and scope.

Join the link tables to the frozen export using `(run_id, source_dataset,
video_id, chunk_id, rally_origin, rally_id)`. A passage can have multiple
candidate rows. Keep identifiers as strings. Unreviewed is distinct from
unclear: an unclear link was inspected but could not be judged.

Source times are seconds in the recording. `source_url` identifies the
public recording where available. Carmack paths preserve the exact local
copies used during review; they are provenance, not portable download links.
No clips are required in the supplement. Developers with access to the
recordings can inspect the times or recreate context clips.

## Delivery status

The review supplements the frozen dataset without replacing its files or
changing its schema. Association remains provisional for the population.
General and unclear links are not positive rally descriptions. A matching
association does not verify its timestamp or transcript. Consult the timing
label and notes before selecting examples for downstream work.

The large local clip package was used for review. The delivery supplement
contains the compressed evidence and this document, without video clips.
