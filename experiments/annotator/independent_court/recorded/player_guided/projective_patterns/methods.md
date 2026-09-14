# Methods and validation

These experiments separate proposal generation, evidence measurement, ranking
and acceptance. They preserve failed alternatives so that a poor final court
can be traced to a specific stage. All settings are shared across videos.
The reports and galleries are a result snapshot; they do not install a new
production detector or provide a standalone replay package.

## Population and measurements

Direction pruning covers 47 cached views: seven GX frames, 20 other amateur
frames and 20 ShuttleSet scenes from videos 03 and 21. Detailed scoring and
subsequent experiments use nine views from five videos. Two unverified
ShuttleSet references are excluded from scoring. Broadcast inputs are existing
median composites rather than raw single frames. All samples are development
data; none is a designated holdout.

Reported reference error is the maximum Euclidean distance across four court
corners after scaling to 1280 × 720. The spacing and automatic-direction
comparisons allow a 180-degree corner relabelling. Distances to supplied control
courts use 960 × 540 working coordinates and are labelled separately. There is
no new numerical success threshold. Visual findings retain their panel identity.

## Direction pruning

The estimator intersects observed lines and includes points at infinity. It
keeps up to 16 vanishing-point hypotheses with 1.5-degree line agreement.
Count ranking and additional-observation coverage ranking run on the same
47 views. Existing merged families retain their 32-line caps and 150 court
templates. The rectangle budget is 16,384, compared with the old random budget
of 4,096. Known seed retention is checked separately from downstream scoring.
Both known seeds appear within the first 4,096 coverage selections, but an
equal-budget scoring comparison has not been performed.

## Marking diagnosis

Finite-fragment matching and two fixed-assignment refits examine 216 saved
courts, including explicitly reference-selected diagnostics. Each refit freezes
fragment identities and fits one perspective transform. The automatic follow-up
scans 16,639,800 generated courts and scores the 3,106,429 that pass the original
geometry/player conditions. It keeps 128 starts per view for detailed matching
and refitting. Starts and fixed-position refits are ranked separately. Floor
outcomes remain recorded, with no new acceptance decision.

## Spacing matcher with supplied directions

Two homogeneous vanishing points define a rectified coordinate system. The
matcher assigns observed offsets to the existing court-coordinate patterns,
retaining competing scale, offset and direction-role choices. Directions come
from manual references or previously inspected courts; line positions come from
image fragments. This is a label-guided capacity test.

Four generation arms across nine views compare the initial matcher, earlier
player pruning, a larger axis shortlist, and finite marking support before
court retention. The final arm keeps 512 matches per axis and 256 courts per
view. Nine fixed-pool comparisons then rank these courts by complete-line
support or existing bright-stripe profiles, with line support breaking paint
score ties. The camera check and profile availability define winner eligibility.

## Automatic directions and loss diagnosis

Automatic generation reuses the frozen coverage selector's 16 directions.
It considers all 240 ordered distinct pairs and applies the existing camera
bound before matching. The bound searches the archived 200-point focal grid
from 0.4 to 4 image widths. Its rejection limit is 0.1, with a 1e-6 numerical
guard. The guard is not a universal roundoff guarantee. Native-coordinate
calibration is used; all nine evaluated inputs resize uniformly.

The matcher keeps up to 256 courts per pair using finite marking support and
a two-working-pixel diversity rule. The original arm keeps 256 globally before
camera filtering. The camera-first arm filters before global selection. The
all-camera-eligible arm removes global truncation and global diversity while
preserving per-pair limits. These are nine generation runs plus 18 rescoring
runs. The last arm measures 19,286 candidates with common complete-line and
paint evidence. Original floor outcomes are recorded but do not select these
diagnostic winners. No court is emitted from these comparisons.

Separate diagnostics measure losses after generation. Two further controls
reconstruct the full observed direction banks and use supplied geometry to
select four candidates per axis. Local fits evaluate their sixteen cross-products;
these are not certified global optima. The best measured pair then supplies
directions to the unchanged matcher. Both controls are labelled non-automatic.
The measured automatic pools are never replaced by these controls.

## Validation and limitations

Both pruning population arms and all nine scoring runs completed. Seven
synthetic tests, scoped lint/types and the whole-project type check passed.
Direct and instrumented scoring outputs agree. Known seed corners reconstruct
within the existing 1e-5 native-pixel tolerance. Recording corrections did not
change selected courts.

The marking runs reproduce frozen population counts. Of 2,304 saved refit
attempts, 1,702 converged and 602 failed existing solver/projection checks.
Three synthetic tests, scoped lint/types and browser controls passed.

All 36 supplied-direction generation runs and nine appearance comparisons
completed. Twelve synthetic cases and scoped lint/types passed. The shared
matcher extraction matches four synthetic complete-result comparisons exactly,
excluding elapsed time. Real GX5 replay differs only in six finite-support
scores, by at most 2.65e-12; its field-specific absolute tolerance is 1e-10.

The automatic stage passes 22 relevant synthetic tests, scoped lint/types,
all nine original global-selection replays, and exact final-score replays for
all 18 rescoring cases. Technical reviews checked coordinate transforms,
selection accounting and shared evaluation. Both overlay controls were checked
with JavaScript disabled. Successful checks returned exit 0.

These checks validate the recorded comparisons, not general detection quality.
Per-pair truncation remains a possible loss source. Existing floor gates reject
some useful courts and both extreme false paint winners. Direction selection,
ranking and acceptance therefore require separate evaluation before replacement.
