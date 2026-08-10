# Incoming shuttle before the first accepted contact

## Answer

This experiment asks: did the shuttle travel towards the player at the first accepted contact, and if so, did adding one earlier shot by the other player make the server attribution correct?

ShuttleSet gives clear first-or-second-contact truth for 103 anchors: 87 were serves and 16 were first returns. Only 19 of those anchors had a path that passed the fixed quality checks. At the selected cut-off, the rule called 14 anchors returns. 11 calls were right, 3 were false calls, and 5 returns were missed. This is 78.6% precision, 68.8% recall and 0.733 F1.

Across all covered rallies, the rule fired 16 times: the 14 clear cases above and 2 anchors that did not match a unique first or second ShuttleSet contact. Directly naming the other player as server was right in 13/16. The released alternating fit was right in 7/16. Adding one missing contact with an unknown player was right in 8/16. Adding the contact and assigning it to the other player was right in 9/16.

The extra player label changed the fitted final player in 3/16 triggered rallies. Usually the change comes from correcting the contact count, not from the added player vote overpowering the later contacts.

The stricter path check excludes TrackNet points that the data producer filled or interpolated. It found 9/16 returns with 0 false calls. That is 100.0% precision, 56.2% recall and 0.720 F1, compared with 0.733 F1 for the main rule.

## What was measured

The anchor is the earliest accepted geometry/impulse contact in each predicted rally. Its player comes directly from `attribute_half` at that frame. The old fitted server label is never used to select the player or measure the incoming path.

The script searches at most 30 base-30 frames before the anchor. It uses the closest continuous path in the same court scene. A path needs at least 5 frames, at least 0.25 body heights of total movement, and no one-frame jump more than 4.0 times its typical movement. The displayed rule also requires the path to finish 0.25 body heights closer and at least 55% of its movements to reduce distance to the contact player.

The displayed setting was chosen by first-return F1 on these same three videos. It is exploratory, not a held-out estimate.

No usable path was available in 225/249 covered rallies. The forced motion rule names the anchor player when the incoming rule does not fire. The evidence-only version abstains when no usable path exists; with a usable path below the cut-off, it names the anchor player. It answered 24/249 covered rallies and was right in 19/24 of those answers.

The two prepend rows change the alternating fit only when incoming motion is found. `Prepend unknown player` adds one place at the start but no player vote. `Prepend other player` adds the same place and supplies the inferred server as one vote. Otherwise both rows keep the ordinary alternating fit over the measured contacts.

## Server attribution

### All 292 ShuttleSet rallies

| Method | Correct | Predictions made | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| old alternating fit | 128/292 | 227/292 | 43.8% | 0.494 |
| use the anchor player | 154/292 | 249/292 | 52.7% | 0.559 |
| use the other player when motion is incoming | 164/292 | 249/292 | 56.2% | 0.592 |
| motion answer only; abstain without a usable path | 19/292 | 24/292 | 6.5% | 0.121 |
| prepend a contact with unknown player | 129/292 | 227/292 | 44.2% | 0.498 |
| prepend a contact by the other player | 130/292 | 226/292 | 44.5% | 0.502 |

### 249 rallies covered by one predicted span

| Method | Correct | Predictions made | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| old alternating fit | 128/249 | 227/249 | 51.4% | 0.538 |
| use the anchor player | 154/249 | 249/249 | 61.8% | 0.607 |
| use the other player when motion is incoming | 164/249 | 249/249 | 65.9% | 0.644 |
| motion answer only; abstain without a usable path | 19/249 | 24/249 | 7.6% | 0.139 |
| prepend a contact with unknown player | 129/249 | 227/249 | 51.8% | 0.542 |
| prepend a contact by the other player | 130/249 | 226/249 | 52.2% | 0.547 |

### 121 rallies where the released fit was wrong or missing

| Method | Correct | Predictions made | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| old alternating fit | 0/121 | 99/121 | 0.0% | 0.000 |
| use the anchor player | 61/121 | 121/121 | 50.4% | 0.486 |
| use the other player when motion is incoming | 68/121 | 121/121 | 56.2% | 0.533 |
| motion answer only; abstain without a usable path | 11/121 | 14/121 | 9.1% | 0.160 |
| prepend a contact with unknown player | 8/121 | 99/121 | 6.6% | 0.072 |
| prepend a contact by the other player | 9/121 | 100/121 | 7.4% | 0.080 |

`Correct` includes the full table denominator. An abstention therefore counts as incorrect. `Predictions made` shows how often each method supplied either Top or Bottom, which keeps the low-coverage evidence-only result from looking like a complete server rule.

The simplest complete rule is “use the player at the first accepted contact, unless incoming motion says the other player served”. Its row is labelled `use the other player when motion is incoming`.

## First-return result by video

| Video | Anchor was serve | Anchor was first return | Usable paths | Returns found | False calls | Returns missed |
|---|---:|---:|---:|---:|---:|---:|
| sset_01 | 30 | 9 | 8 | 5 | 1 | 4 |
| sset_15 | 36 | 4 | 5 | 3 | 1 | 1 |
| sset_21 | 21 | 3 | 6 | 3 | 1 | 0 |

## Checks and useful subsets

- Clear anchor truth: 87 first contacts and 16 second contacts, 103 total.
- Clear anchors with a path passing the fixed quality checks: 19/103.
- Covered rallies with an earlier rejected raw impulse: 122.
- Incoming-motion triggers with an earlier rejected raw impulse: 3.
- If every earlier rejected impulse were used as a veto, the result would become 10 returns found, 1 false calls and 6 returns missed. This is 90.9% precision and 62.5% recall. The veto is reported only as a comparison.
- Paths that begin exactly when the court scene begins: 16.
- Exact equal-distance anchor ties that would favour Top: 0.
- Case plots written: 11; all clear false positives plus a small sample of true positives and misses.

## Plots

- `outputs/plots/first_return_threshold.png`: precision, recall and F1 as the required share of movements towards the contact player changes.
- `outputs/plots/incoming_motion_measurements.png`: percentage of movements towards the contact player and net closing distance.
- `outputs/plots/server_accuracy.png`: correct server counts before and after adding a shot.
- `outputs/plots/tracknet_source_comparison.png`: return counts with and without TrackNet's filled or interpolated points.
- `outputs/plots/cases/`: at most twelve labelled shuttle paths.

## Limits

Only 16 clear first-return anchors are available, so the threshold can move with a few rallies. Excluding repeated-position warnings does not guarantee that a TrackNet point is real. The stricter source comparison remains in the compressed tables. The experiment infers the player and order of a missing shot; it does not recover the serve frame or prove that the serve itself was visible.
