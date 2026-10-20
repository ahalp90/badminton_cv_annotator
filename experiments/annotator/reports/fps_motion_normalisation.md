# Motion normalisation did not improve validation performance

Keep the simplified annotator's selected raw-feature stack. Converting learned
motion features to equivalent 30 frames per second (fps) did not improve mean
rally recovery or player-attributed contact F1 across five seeds. These results
do not establish whether normalisation helps on real 60 fps or amateur footage.

## What changed

Three feature variants were compared: existing raw features, full derivative
normalisation and velocity-only normalisation. Both normalised variants multiply
shuttle velocity/speed and both players' ankle speeds by `q = fps / 30`, at five
temporal feature offsets. Full normalisation also multiplies second-difference impulse
by `q²` and recomputes its ratio with the fixed denominator floor expressed in
base-30 units. Velocity-only leaves impulse and its ratio unchanged.

Geometry, frame windows, candidate-region flags and vision evidence stay fixed.
Preparation at exactly 30 fps is unchanged. Both contact and rally models use
the transformed rows, so this is a whole-stack feature change. Changed contact
scores can also change downstream proposals.

The tree settings stay fixed: contact model with 31 leaves and 180 rounds;
one rally selector with 31 leaves and 400 rounds; optional review ranker with
seven leaves and 100 rounds. Learning rates are 0.06, 0.05 and 0.05 respectively;
all use XGBoost with minimum child weight 5. Each training video's contact fit
excludes its eight-video group, using the other 24 videos. The final contact
model uses all 32 training videos. Candidate budgets and training-target policy
match the [selected raw stack](tree_simplification.md).

The five model-seed offsets are 0, 100, 200, 300 and 400, paired across variants.
Existing raw five-seed runs provide the baseline. Each comparison retains its
raw seed's training span weight, splits and sampling order. Parallel
workers write per-video files; training input order is reconstructed explicitly.
This rules out worker completion order as the source of row-order variation in
these paths. GPU execution determinism itself was not tested.

## Population and results

The comparison uses 32 training videos and the same eight development-validation
videos, containing 668 labelled rallies and 5,696 labelled contacts. Training
has 16 videos at 25 fps and 16 at 30 fps. Validation has only one 25 fps video,
`sset_18`, and seven at 30 fps. No new test evaluation was performed.

Correct-rally recovery counts wholly correct predicted rallies against the
labels. Player-attributed contact F1 requires both matched timing and the correct
player side. The ±10 and ±5 matching tolerances are expressed in equivalent
30 fps frames. The table reports five-seed means on the same videos; these are
not five independent validation datasets.

| Features | Correct rallies / 668, ±10 | Correct rallies / 668, ±5 | Attributed F1, ±10 | Attributed F1, ±5 |
|---|---:|---:|---:|---:|
| Raw | 296.0 | 243.6 | 85.53% | 83.72% |
| Full normalisation | 293.0 | 242.2 | 85.09% | 83.29% |
| Velocity-only | 292.8 | 241.6 | 85.21% | 83.42% |

Raw has the highest mean on all four measures. Paired changes below subtract
the raw result at the same seed. Parentheses show the observed minimum and
maximum; F1 changes are percentage points.

| Variant | ±10 rallies | ±10 F1 | ±5 rallies | ±5 F1 |
|---|---:|---:|---:|---:|
| Full | −3.0 (−5 to +3) | −0.44 (−1.32 to +0.19) | −1.4 (−8 to +4) | −0.43 (−1.41 to −0.06) |
| Velocity-only | −3.2 (−23 to +14) | −0.32 (−1.06 to +0.27) | −2.0 (−21 to +14) | −0.30 (−1.22 to +0.42) |

Full normalisation loses ±5 F1 in every seed. Velocity-only has mixed paired
results and larger recovery variation. These descriptive differences do not
establish statistical significance or equivalence. They provide no measured
benefit that warrants replacing the current raw stack.

## Limits and next decision

This tests two specific feature transformations with the selected tree settings.
It does not show that raw features are better outside this validation set or
that normalisation is inherently unhelpful. Rounded smoothing windows, missing
observations and tracking noise remain frame-rate-dependent. Synthetic trajectory
tests check the derivative units, not real 60 fps tracking or amateur-video
generalisation. One 25 fps validation video cannot establish a reliable subgroup
advantage.

The production model therefore stays unchanged: two annotation models plus the
optional review ranker. No normalised production bundle is exported. The next
session will run a broader staged input/preprocessing search, starting from the
selected raw stack. It will select validation winners for attributed F1 and
fully correct rallies, then evaluate the frozen configurations on ShuttleSet22
test at five seeds. Filtering thresholds will be selected on validation and
assessed unchanged on test. This experiment did not launch that search.

## Evidence and reproduction

The [evidence record](../tree_simplification/results/fps-normalisation-evidence.json.gz)
contains per-seed counts, paired summaries and per-video changes. The
[experiment commands](../tree_simplification/README.md#fps-motion-unit-experiment)
describe the separate preparation and fitting paths. Implementation is recorded
in feature-branch commit `a0d8ddf7`; remote runs preserve their frozen source
checksums. Existing preparation and model receipts prevent incompatible reuse.

All ten experimental runs completed model fitting and replay. Each rebuilt all
eight validation pools and streams exactly. The eight repeat runs exited 0 at
every stage. Both initial runs needed auxiliary diagnostic-launch repairs;
separate retries passed, with original failure receipts preserved and no model
refits. The diagnostic failures do not invalidate completed fitting or replay.

Physical-trajectory and input-boundary tests pass. Full pytest passed 3,154 tests
with 35 skips before the velocity-only flag; 28 scoped tests passed after it.
Whole-project Pyrefly and changed-file Ruff passed. Whole-repository Ruff reports
789 findings outside the changed files. Production feature extraction is unchanged.
