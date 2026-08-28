# Validation rally-start inputs

## Result

Keep the saved input. It reproduces the fixed validation candidate list and
adds the existing player-side answer at every kept or candidate frame. It did
not read human labels or change a candidate.

The file covers all eight validation videos. It contains 615 section lists,
1,845 total entries and 1,230 earlier candidates. These are the same list and
entry totals fixed before this save.

## Player-side answers

The existing rule returns a side for 629 of the 1,230 earlier candidates. It
has no answer for the other 601. A candidate with no answer stays in the saved
input but cannot be selected by the next model.

The rule has no answer for 11 of the 5,326 kept contacts. Two of the 615 fixed
contacts also have no answer. The later rally check will continue to count
these as unanswered, rather than treating them as correct.

## Checks

The save finished twice from the same checked inputs. The two compressed files
are equal byte for byte and have the same SHA-256 hash recorded in
`validation_rally_start_input_summary.json`.

The saved file contains no machine paths or server details. Every recorded
`labels_read` value is false. Candidate identities do not repeat, player-side
values are `Top`, `Bot` or an explicit no-answer, and every list keeps the
fixed three-entry size.

The raw file stays outside Git under `raw/validation_rally_start_inputs/`.
