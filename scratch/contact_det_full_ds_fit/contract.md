# Full-dataset contact experiment contract

## Goal

Choose a simple HGB or RF contact detector on the full original ShuttleSet extraction, then evaluate the frozen setup once on non-overlapping ShuttleSet22 videos.

## Success criteria

- A portable manifest defines the eligible videos, development split and data provenance without machine-specific paths.
- Every model choice uses 32 training videos and the same eight validation videos.
- The selected setup is refitted on all 40 eligible ShuttleSet videos only after its settings are frozen.
- ShuttleSet22 is used once as the test set after overlapping videos are removed.
- Contact timing, player side and strict fully-correct rally results are reported separately.
- The report shows correctness against retained-rally yield rather than presenting contact F1 as the product score.
- Commands, random seeds, schemas and compact result artefacts are sufficient to reproduce the run.

## In scope

- The 40 completed ShuttleSet video extracts. Videos 9, 10, 12 and 27 remain excluded by the dataset record.
- A fixed 32-video fit split and eight-video validation split with whole videos kept together.
- HGB and RF baselines using the pilot's region-v2 feature design.
- Fresh selection of motion convention, model settings, score threshold, duplicate-removal distance, class weighting and negative sampling.
- A focused cleanup or bounded rescue test only when the validation errors show a clear job for it.
- The existing direct Top/Bottom rule and strict rally contract.
- The three manually reviewed fixtures as scene-cut contract checks.
- Portable scripts, tests, manifests, run records, compact results and reports under this directory.

## OUT-list

- `scratch/contact_det/` stays unchanged. It remains the three-video pilot record.
- Production code under `src/` stays unchanged unless a required reusable boundary cannot live in the experiment package. Stop and confirm before crossing this boundary.
- Upstream TrackNet, pose, court and annotation stages are reused. They are not refitted or regenerated in this pass.
- BST-X neural contact-detector work stays out of this pass.
- A new player-side model stays out until the first-stage contact result is understood.
- Pilot-fitted trees, thresholds, class weights and duplicate distances are evidence only. None are carried forward as selected settings.
- Machine paths, hostnames, access helpers, credentials and remote operating details stay out of tracked files and commit messages.
- Large reproducible feature arrays and source artefacts stay untracked.
- No push, merge, release or commit to `main` is authorised.

## Invariants

- Feature freezing remains label-blind. Ground-truth timing labels load only after the frozen feature manifest has been verified.
- Train, validation and test video identities are disjoint.
- All centres from one physical video stay in one split.
- Validation labels may select settings. Test labels may only score the frozen setup.
- Source-frame identities, frame rates and half-open span contracts remain explicit.
- Failed or partial long runs retain enough evidence to diagnose the failure without being mistaken for complete results.

## Risks and controls

- Repeated players and tournaments can make validation optimistic. The proposed split prioritises unseen players, while ShuttleSet22 supplies the later dataset shift.
- The fitted score scale may move when the selected model is refitted from 32 to 40 videos. The validation threshold remains frozen and the shift is reported.
- Full-corpus feature freezing is much larger than the pilot. Manifests and per-video progress make partial failures visible.
- The current scripts assume three fixtures throughout. Tests must pin the new roster and split contract before remote runs.
- ShuttleSet22 overlap identities and prepared stage inputs are not yet recorded in this directory. Test work pauses until both are explicit.

## Authority and disclosure

- Local edits and local commits on `contact-det-feasibility` are authorised.
- Commit messages should be short, natural notes about the useful change.
- Compute jobs and experiment-only checkout changes are authorised within the supplied dataset extraction.
- Tracked records use portable dataset identifiers and relative paths only.
- External reviewers receive only named source files, diffs and non-sensitive experiment records.

