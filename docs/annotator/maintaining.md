# Maintainer guide

Preserve the decisions and data contracts at each stage when changing the
annotator. [`annotate_video()`](../../src/annotator/annotate_video.py) is the main reading
entry point; the [code map](code_map.md) identifies the owner of each decision.
The [algorithm guide](how_it_works.md) explains why the stages exist.

## Invariants that matter

All arrays use the source-video timeline. Rally bounds include the start
frame and exclude the end. Frame-rate-dependent settings scale before annotation. Player evidence must
see the original shuttle track before replay masking.

Candidate and proposal order affects equal-score choices. Feature columns,
NaNs and dtypes form the model inputs. Preserve their meaning when reusing a
fitted model; a changed feature definition requires a new fit.

Contact feature construction applies the search priors and returns only candidate
rows. Training and inference consume those rows directly. Rows can skip frames;
feature offsets use the video timeline, and interval IDs refer to the original
search intervals even when an interval has no candidates.

Learned impulse features and replay-mask signals use the default thresholds for
the video's frame rate. They ignore the saved `overrides_base30` adjustments used
by segmentation. Learned impulses count invisible shuttle frames as zero when
smoothing; heuristic segmentation skips them by default. Preserve both choices
when sharing calculations between annotation paths.

A rally owns an explicit contact sequence. Full-stream contacts outside every
initial rally survive selection. Boundary padding preserves contact membership;
filtering the full stream by final bounds is not an equivalent reconstruction.

Player-half alternation happens after selection. Choose the order matching the
most individual player guesses. Break ties with the lower summed wrist distance
in player-height units, using only contacts with evidence for both court halves.
Equal or unavailable sums follow the first known guess. Keep all players unknown
when no guess is known. Training targets judge this same final player order. Review features use the raw
individual side guesses to measure agreement. They use final boundaries only
for the boundary summary.

Contact sampling order and fit-row order are separate. Preserve both, together
with seeds, target policies and the configured total weight per judged rally
when claiming that training behaviour is unchanged.

## Does a change require a new fit?

| Change | Fit and evaluation needed? |
| --- | --- |
| Rearrange code without changing model inputs or decisions | Keep the existing bundle; test the changed boundary |
| Change contact search regions or physical features | Refit contact and downstream models |
| Change initial contact selection or side evidence | Refit downstream rally/review models and compare end-to-end results |
| Change proposal rules, feature order or proposal settings | Refit rally selection and review |
| Change review features only | Refit review; check that annotations remain unchanged |
| Change winner, landing or hit-height rules | Evaluate those outcomes; contact-model fitting alone cannot validate them |
| Change upstream court, pose or shuttle extraction materially | Re-evaluate the complete path and assess refitting |

Heuristic-only annotation remains supported. Test it separately when changing
shared preparation or outcome code; learned-model agreement does not establish
heuristic agreement.

## Saved formats

`AnnotatorResult` fields are serialised by the dataset builder. Changing a field
name or meaning affects stored results, resume checks and downstream assembly.
The result retains finished rallies and grouped outcomes; derive counts and server
views from the rallies at export. Unknown contact probabilities are saved as null,
while infinities remain invalid. Both exclusion masks are persisted beside the result.
Replay masking uses court
presence and shuttle speed; court composition remains a separate preparation tool.
Frame counts use `frames_at_fps()` with half-up rounding, and per-frame speeds use
`speed_at_fps()` with inverse FPS scaling.

Newly fitted model bundles use `annotator-three-models/5`. Saved preprocessing has
one numeric override mapping, `overrides_base30`. `scale_config_to_fps()` in
`config.py` converts it into one `Thresholds` value for the video's frame rate.
The loader checks feature order and proposal settings. Player attribution uses
the current scene's geometry. Runtime
versions are recorded for reproduction, without requiring an exact version match.
Joblib records Python class paths, so moving saved classes can require rebuilding
their artefacts. Use the corresponding historical checkout for older bundles and
settings.

Training and inference share `prepare_annotation_inputs()` in
[prepare_annotation_inputs.py](../../src/annotator/inference/prepare_annotation_inputs.py). Rebuild prepared inputs in a
fresh directory when their feature layout changes. The prepared-input loader
checks the current candidate-row columns before training or scoring.

## Check a change at its boundary

Test the decision that changed and its immediate consumers. Contact changes
usually need numerical cases for candidate membership, time offsets and selected
frames. Rally changes need cases for proposals, ownership, boundaries and sides.
Use direct comparisons where they diagnose a specific calculation.

Exercise dataset-builder persistence and MonoTrack annotation when their shared
interfaces change. Check the affected result fields and masks.

Judge intentional model changes by whole-rally correctness on validation data;
use contact and player F1 to diagnose the difference. Repeat paired seeds when
fit variation could change the decision. Use the fixed test set for confirmation.

Use the repository's relevant lint, type and test checks. Broaden testing when
shared interfaces or pipeline behaviour change. The current tests cover contact
features/sampling, rally proposals and selection, model persistence, heuristic
rules, outcome handling and the two live integrations.

## Debug a wrong contact from the earliest difference

1. Check the source track, player evidence and scene geometry at that frame.
2. Check exclusion and shuttle-quality masks; they have different effects.
3. Check whether the frame entered a contact-search region.
4. Inspect its features, probability and nearby competing candidates.
5. Check whether the desired rally sequence was proposed.
6. Compare selector scores, then inspect boundary and side finishing.
7. Inspect outcome rules only after confirming the final contacts.

Pass `keep_working_data=True` to retain intermediate predictions. Exclusion
masks are always returned. The validation
overlay and review tools provide visual checks without changing the inference
path. Keep experimental settings and their saved results together, following
the [refit guide](retuning.md).
