# Annotator regression cases

These small fixtures hold results from the original research functions. The
tests build the same synthetic inputs and compare the maintained implementation
with these fixed results. They need no research scripts or extracted dataset.

- `contact_features.npz` and `contact_intervals.json.gz`: all feature columns and
  eligible/search intervals at 25, 30 and 60 FPS. Inputs include invisible shuttle
  frames, missing poses, infinite wrist gaps, scene boundaries and exclusion-mask
  edges. Captured from `freeze_tree_contact_features._fixture_rows`.
- `contact_sampling.npz` and `contact_sampling_counts.json.gz`: selected candidate
  indices, labels and per-video counts for three 800-frame videos at 25, 30 and
  60 FPS, in that order. Contacts are at frames 100 and 400; every seventh frame
  is excluded from the candidate region. Captured from
  `score_contact_baseline.choose_training_rows` with seed 20260824. These protect
  the sampling radii, negative budget and seeded row order.
- `sequence_targets.npz`: serve-edit, complete-sequence and insertion labels for
  two synthetic rallies, with known or missing human sides, at all three FPS
  values. Captured from `targets.assign_targets`, `whole_rally_options.whole_targets`
  and `local_insertion.insertion_targets`. These protect target ordering and the
  distinction between wrong and unjudgeable options.
- `contact_matching.json.gz`: matching results from `matching.match_contacts` for
  equal-distance ties, duplicate frames and unsorted input.

Synthetic inputs live beside their assertions in the corresponding test files.
Feature comparisons preserve column order and NaNs. Identity and interval
storage widths may differ between the original and maintained code; their values
must still match.

`confidence.npz` stores the original confidence features and insertion-model inputs
for synthetic sequences at 25, 30 and 60 FPS. It covers boundary refresh, missing
values and an empty sequence.
