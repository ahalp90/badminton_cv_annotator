# Maintainer guide

Preserve the decisions and data contracts at each stage when changing the
annotator. [`run_video()`](../../src/annotator/run_video.py) is the main reading
entry point; the [code map](code_map.md) identifies the owner of each decision.
The [algorithm guide](how_it_works.md) explains why the stages exist.

## Invariants that matter

All arrays use the source-video timeline. Rally bounds are half-open and
frame-rate-dependent settings scale before annotation. Player evidence must
see the original shuttle track before replay masking.

Candidate and proposal order affects equal-score choices. Feature columns,
NaNs and dtypes form the model inputs. Moving code must preserve these values,
not just the number of features or selected contact count.

A rally owns an explicit contact sequence. Full-stream contacts outside every
initial rally survive selection. Boundary padding preserves contact membership;
filtering the full stream by final bounds is not an equivalent reconstruction.

Player-half alternation happens after selection. Review features use the raw
individual side guesses to measure agreement. They use final boundaries only
for the boundary summary.

Contact sampling order and fit-row order are separate. Preserve both, together
with seeds, target policies and the weight scale from rule-based repairs, when claiming
that training behaviour is unchanged.

## Does a change require a new fit?

| Change | Fit and evaluation needed? |
| --- | --- |
| Restructure code with exactly equal inputs and outputs | Existing bundles should remain valid; prove equality |
| Change contact search regions or physical features | Refit contact and downstream models |
| Change initial contact selection or side evidence | Refit downstream rally/review models and compare end-to-end results |
| Change proposal rules, feature order or proposal settings | Refit rally selection and review |
| Change review features only | Refit review; check that annotations remain unchanged |
| Change winner, landing or hit-height rules | Evaluate those outcomes; contact-model fitting alone cannot validate them |
| Change upstream court, pose or shuttle extraction materially | Re-evaluate the complete path and assess refitting |

Heuristic-only annotation remains supported. Test it separately when changing
shared preparation or outcome code; learned-model agreement does not establish
heuristic agreement.

## Saved compatibility

`AnnotatorResult` fields are serialised by the dataset builder. Changing a field
name or meaning affects stored results, resume checks and downstream assembly.
Both exclusion masks are persisted beside the result.

Current model bundles use `annotator-rally-layers/2`. Saved preprocessing has
one numeric override mapping, `overrides_base30`. `resolve()` in `config.py`
converts it into one `Thresholds` value for the video's frame rate.
The loader checks runtime versions, feature order, proposal settings, side geometry and review mode.
Joblib also records Python class paths: moving `AnnotatorModels`,
`RallyLayerModels` or saved evidence classes needs an explicit migration.

The eleven active tree-simplification bundles and five 40-video contact refits
were converted to this schema without changing fitted trees or effective
preprocessing. Their current replay and checker records use the same representation.
Recorded numerical results and source provenance remain unchanged.

Old eight-model bundles are unsupported. Historical research results remain
readable, but old proposal caches must be regenerated after the proposal
representation change. Keep preparation provenance when reusing saved features;
a current source revision alone does not describe how an older cache was made.

## Check a change at its boundary

For contact changes, compare searched regions, feature rows, candidate scores
and selected frames. For rally changes, compare ordered proposals, feature
matrices, selected rows, final contacts/bounds/sides and review scores. Direct
array comparison with matching NaNs makes the source of a difference visible.

For integration changes, exercise both dataset-builder persistence and
MonoTrack supplied-boundary/discovery annotation. Compare every public result
field and both masks when shared preparation or outcomes change.

A GPU retrain is useful after training changes. Compare the same seeded runs
with fixed data, settings and row order. A small spread across seeds does not
by itself establish that a refactor preserved the algorithm.

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

Use `RunCapture` to retain intermediate predictions and masks. The validation
overlay and review tools provide visual checks without changing the inference
path. Keep experimental settings and their saved results together, following
the [refit guide](retuning.md).
