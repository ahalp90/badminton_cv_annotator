# Final contact model fit

## Bottom line

The final contact model is fitted and ready for the ShuttleSet22 test. It uses
the unchanged reference HGB model, a score cut-off of 0.9 and a six-frame
distance for merging nearby predictions at 30 frames per second.

The model trained on all 40 development videos. Its saved file was loaded
again and gave exactly the same probabilities for the first and last candidate
row from every video.

## Final setting

Five HGB fits produced held-out scores for all 40 development videos. Each fit
trained on 32 videos and scored the other eight. The combined file contains
1,477,290 candidate scores.

The fixed comparison checked the same 57 score cut-off and nearby-contact
settings used earlier. The best setting stayed at:

- score cut-off: 0.9
- nearby-contact distance: six frames at 30 frames per second

At five frames, the held-out result across all 40 videos is:

- 33,267 labelled contacts
- 31,824 predicted contacts
- 28,801 matched contacts
- 0.9050 precision
- 0.8658 recall
- 0.8849 F1
- 1,659 of 3,359 first contacts found, or 49.4%
- 27,142 of 29,908 later contacts found, or 90.8%

The eight former validation videos reproduced their earlier candidate
identities and probabilities exactly. Group A and the final combined files
also matched their repeats byte for byte.

## Fit on all 40 videos

The final HGB fit selected 1,313,803 training rows. Of those, 94,530 are
positive rows. The model uses 85 input fields and the same settings as the
chosen baseline:

- balanced class weights
- 31 leaves
- learning rate 0.06
- 180 iterations
- at least 40 samples per leaf
- L2 value 1.0
- at most 24 negative rows per positive row

The saved model has SHA-256
`ef7b66042ce2ed594572424ddd2c13f23092afcc8b259bccc8758af8cc11a8dc`.
The saved fit result has SHA-256
`5428bb69be41aea034fe56f5b812594404d1ac458392681f853b74a26600b4ed`.
Both files stay outside Git.

The fitting environment used Python 3.11.13, NumPy 2.2.6, scikit-learn 1.6.1
and joblib 1.5.3. The ShuttleSet22 run should use the same environment because
saved scikit-learn models are not promised to work across library versions.

## Checks

An independent result check found no blocker. It checked all 40 video counts,
the two source commits, every saved input hash, the final model settings and
the 80 saved reload rows. It also loaded the model and reproduced all 80
probabilities exactly.

The fitting code passed 145 experiment tests. The whole project passed 1,893
tests with 29 skipped. Ruff passed for the experiment directory. The pinned
type check reported no errors and 21 suppressed messages.

## Next step

Write and review the exact ShuttleSet22 video list and scoring rules. Then run
the finished model once on the non-overlapping ShuttleSet22 videos. Do not use
those labels to change the model, score cut-off, nearby-contact distance or
feature calculations.
